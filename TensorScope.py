"""TensorScope — inspect the exact intermediate tensors of a real local LLM forward pass.

Run the GUI with:  python TensorScope.py
Run non-GUI checks with: python TensorScope.py --self-test

TensorScope loads a local Hugging Face model in-process and captures its real
intermediate tensors with PyTorch forward hooks plus a registered attention
implementation.  There is no separate server and no build step.

The design invariant: no tensor is ever inferred, simulated, or reconstructed.
Every displayed value is a tensor the model actually computed.  Two mechanisms
are needed, because one is not sufficient:

  * Module forward hooks supply embedding, normalized_input, q/k/v,
    attention_output and layer_output.
  * `CaptureAttention` supplies attention_scores, attention_weights and the
    post-RoPE q/k/v.  These are unreachable by hooks: the pre-softmax score
    tensor is a local variable inside the attention function, and transformers
    5.x removed `output_attentions` from the model forward signature.
    `CaptureAttention` mirrors upstream's `eager_attention_forward` line for
    line and *proves* it by re-running upstream's own function on its first
    call and requiring bitwise-equal outputs.  `RunCapture.validate()` rejects
    any capture that did not pass that check.
"""

from __future__ import annotations

import datetime as dt
import html
import io
import json
import math
import os
import sqlite3
import sys
import traceback
import zlib
from contextlib import closing
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from tensorscope_content import GenerationDecision

# Keep matplotlib's cache beside the application when a user profile is locked down.
APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".matplotlib"))

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QLineF, QPointF, QRectF, QSize, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QStackedWidget, QTextEdit, QVBoxLayout,
    QWidget,
)


# In a packaged (frozen) bundle write to the OS app-data directory so the DB
# survives app updates.  In source runs keep it beside the script so existing
# captures are found immediately.
if getattr(sys, "frozen", False):
    try:
        from platformdirs import user_data_dir as _user_data_dir
        DB_PATH = Path(_user_data_dir("TensorScope", "TensorScope")) / "tensorscope_runs.sqlite3"
    except ImportError:
        DB_PATH = APP_DIR / "tensorscope_runs.sqlite3"
else:
    DB_PATH = APP_DIR / "tensorscope_runs.sqlite3"

# Bumped to 4 for per-token decision telemetry (the generation_steps table).  Nothing
# reads this number to decide behaviour: the real discriminator for a pre-telemetry run
# is simply that it has no generation_steps rows, which load() handles directly.
SCHEMA_VERSION = 4
CAPTURE_SOURCE = "pytorch-forward-hooks"

# A dense instruct model.  Dense matters: a Mixture-of-Experts model routes through
# a router and many experts between attention_output and layer_output, and none of
# that appears in the tensor set below -- the recap would silently hide the most
# interesting part of the computation.  See CLAUDE.md "Model choice".
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"

REQUIRED_LAYER_TENSORS = {
    "normalized_input", "q", "k", "v",
    # Post-RoPE (and post QK-norm where the architecture has it), KV heads expanded.
    # These are the tensors that actually produce the scores, so a reader can verify
    # scores == q_attended @ k_attended^T * scaling + mask.  The plain q/k/v above are
    # the raw projection outputs, before rotary embedding.
    "q_attended", "k_attended", "v_attended",
    "attention_scores", "attention_weights", "attention_output", "layer_output",
}

# Attention score tensors are heads x N x N, so capture cost grows with the square
# of the prompt length.  TensorScope's scope is basic prompts; refuse rather than
# quietly produce a multi-gigabyte run.
MAX_CAPTURE_TOKENS = 256

# How many of the competing next-token scores each generation step records.  One K for
# every candidate list, so the first token -- whose whole vocabulary vector is also kept --
# is not stored twice at two different lengths.  Internally configurable.
CANDIDATE_TOP_K = 10

# ── Design token tables ──────────────────────────────────────────────────────────
#
# | token          | dark        | light       | role                           |
# |----------------|-------------|-------------|--------------------------------|
# | bg             | #111114     | #f5f5f7     | window / page background       |
# | sidebar_bg     | #0d0d10     | #eeeef2     | sidebar panel background       |
# | card_bg        | #1c1c20     | #ffffff     | card / input surface           |
# | border         | #2e2e33     | #d8d8de     | 1-px borders and dividers      |
# | text_primary   | #f0f0f2     | #111114     | main readable text             |
# | text_secondary | #9f9fa8     | #52525c     | secondary / label text         |
# | text_muted     | #636370     | #8e8e9a     | captions, hints, status        |
# | accent         | #3b82f6     | #3b82f6     | interactive blue (both modes)  |
# | nav_active_bg  | #202028     | #dddde4     | active nav-item highlight      |
# | success        | #16a34a     | #15803d     | verified / ready banner        |

_TOKENS_DARK: dict = {
    "bg":            "#111114",
    "sidebar_bg":    "#0d0d10",
    "card_bg":       "#1c1c20",
    "border":        "#2e2e33",
    "text_primary":  "#f0f0f2",
    "text_secondary":"#9f9fa8",
    "text_muted":    "#636370",
    "accent":        "#3b82f6",
    "nav_active_bg": "#202028",
    "success":       "#16a34a",
}
_TOKENS_LIGHT: dict = {
    "bg":            "#f5f5f7",
    "sidebar_bg":    "#eeeef2",
    "card_bg":       "#ffffff",
    "border":        "#d8d8de",
    "text_primary":  "#111114",
    "text_secondary":"#52525c",
    "text_muted":    "#8e8e9a",
    "accent":        "#3b82f6",
    "nav_active_bg": "#dddde4",
    "success":       "#15803d",
}
# Module-level dict, replaced in __main__ after theme detection.
TOKENS: dict = _TOKENS_DARK


def _make_qss(T: dict) -> str:
    return f"""
QMainWindow, QDialog, QWidget {{
    background: {T['bg']}; color: {T['text_primary']}; font-size: 13px;
}}
QScrollArea {{ border: none; background: transparent; }}
QFrame {{ border: none; }}
QLabel {{ background: transparent; color: {T['text_primary']}; }}

/* sidebar panel */
#sidebar {{
    background: {T['sidebar_bg']};
    border-right: 1px solid {T['border']};
}}
#logoPlaceholder {{ background: {T['text_muted']}; border-radius: 6px; }}

/* sidebar nav items */
QPushButton#sidebarNav {{
    text-align: left; border: none; border-radius: 8px;
    padding: 9px 12px; background: transparent;
    color: {T['text_secondary']}; font-size: 13px;
}}
QPushButton#sidebarNav:hover {{ background: {T['nav_active_bg']}; color: {T['text_primary']}; }}
QPushButton#sidebarNav:checked {{
    background: {T['nav_active_bg']}; color: {T['text_primary']}; font-weight: 600;
}}

/* screen headers */
#screenTitle {{ font-size: 20px; font-weight: 700; color: {T['text_primary']}; }}
#screenSubtitle {{ font-size: 12px; color: {T['text_muted']}; margin-top: 2px; }}

/* cards */
#card {{
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 10px;
}}
#rowDivider {{
    background: {T['border']}; max-height: 1px; min-height: 1px; border: none;
}}
#cardTitle {{ font-size: 13px; font-weight: 600; color: {T['text_primary']}; }}
#rowLabel  {{ font-weight: 600; color: {T['text_primary']}; }}
#rowValue  {{ color: {T['text_secondary']}; font-size: 12px; }}
#statusLabel {{ font-size: 12px; color: {T['text_muted']}; padding-top: 2px; }}

/* Footer status bar: a thin rule separates it from the content it describes, so the
   database path reads as chrome rather than as floating body text. */
#statusBar {{
    background: transparent; border-top: 1px solid {T['border']};
    color: {T['text_muted']}; font-size: 11px; padding: 8px 32px;
}}

/* empty state -- deliberately unboxed: no #card wrapper, no background, no border,
   so it sits directly on the page like the rest of the screen. */
#emptyTitle {{ font-size: 14px; font-weight: 600; color: {T['text_secondary']}; }}
#emptyBody  {{ font-size: 12px; color: {T['text_muted']}; }}
#finePrint  {{ font-size: 11px; color: {T['text_muted']}; }}

/* standard buttons */
QPushButton {{
    background: {T['card_bg']}; color: {T['text_primary']};
    border: 1px solid {T['border']}; border-radius: 6px; padding: 6px 14px;
}}
QPushButton:hover  {{ border-color: {T['text_secondary']}; }}
QPushButton:pressed {{ background: {T['nav_active_bg']}; }}
QPushButton:disabled {{ color: {T['text_muted']}; border-color: {T['border']}; }}

/* primary button -- the one action a screen wants the reader to take */
QPushButton#primary {{
    background: {T['accent']}; border: 1px solid {T['accent']};
    color: #ffffff; font-weight: 600; padding: 7px 16px;
}}
QPushButton#primary:hover  {{ background: #2f6fe0; border-color: #2f6fe0; }}
QPushButton#primary:pressed {{ background: #2861cc; border-color: #2861cc; }}

/* icon-only button (header actions such as Refresh) */
QPushButton#iconButton {{
    background: transparent; border: none; border-radius: 6px; padding: 0px;
}}
QPushButton#iconButton:hover   {{ background: {T['nav_active_bg']}; }}
QPushButton#iconButton:pressed {{ background: {T['border']}; }}

/* ghost button */
QPushButton#ghost {{
    background: transparent; border: 1px solid {T['border']};
    border-radius: 6px; padding: 4px 10px;
    color: {T['text_secondary']}; font-size: 12px;
}}
QPushButton#ghost:hover {{ border-color: {T['text_secondary']}; color: {T['text_primary']}; }}

/* navItem buttons (ComputationRecap left panel) */
QPushButton#navItem {{
    text-align: left; border: none; border-radius: 6px;
    padding: 6px 10px; background: transparent; color: {T['text_secondary']};
}}
QPushButton#navItem:hover   {{ background: {T['nav_active_bg']}; color: {T['text_primary']}; }}
QPushButton#navItem:checked {{
    background: {T['nav_active_bg']}; color: {T['text_primary']}; font-weight: 600;
}}

/* inputs */
QLineEdit, QTextEdit, QComboBox {{
    background: {T['card_bg']}; color: {T['text_primary']};
    border: 1px solid {T['border']}; border-radius: 6px; padding: 5px 8px;
    selection-background-color: {T['accent']}44;
}}
QLineEdit:focus, QTextEdit:focus {{ border-color: {T['accent']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}

/* list widget */
QListWidget {{
    background: {T['card_bg']}; border: 1px solid {T['border']};
    border-radius: 8px; outline: none;
}}
QListWidget::item {{ padding: 7px 10px; border-radius: 4px; color: {T['text_primary']}; }}
QListWidget::item:hover    {{ background: {T['nav_active_bg']}; }}
QListWidget::item:selected {{ background: {T['nav_active_bg']}; color: {T['text_primary']}; }}

/* splitter */
QSplitter::handle {{ background: {T['border']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}

/* scrollbar */
QScrollBar:vertical {{
    background: transparent; width: 8px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {T['border']}; border-radius: 4px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {T['text_muted']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

/* story cards and StyledPanel frames */
#storyCard {{
    background: {T['card_bg']}; border: 1px solid {T['border']};
    border-radius: 8px; margin-top: 4px;
}}
QFrame[frameShape="6"] {{
    background: {T['card_bg']}; border: 1px solid {T['border']}; border-radius: 8px;
}}
"""


def _make_mplstyle(T: dict) -> dict:
    return {
        "figure.facecolor": T["bg"],
        "axes.facecolor":   T["card_bg"],
        "axes.edgecolor":   T["border"],
        "axes.labelcolor":  T["text_secondary"],
        "xtick.color":      T["text_muted"],
        "ytick.color":      T["text_muted"],
        "text.color":       T["text_primary"],
        "axes.titlesize":   11,
        "axes.grid":        False,
    }


APP_QSS  = _make_qss(TOKENS)
MPLSTYLE = _make_mplstyle(TOKENS)


# ── Icons ────────────────────────────────────────────────────────────────────────
#
# Painted with QPainter rather than shipped as asset files or borrowed from a text
# font.  The sidebar previously mixed four unrelated characters -- a play triangle,
# a hamburger, a gear, a middle dot -- so its stroke weight, optical size and
# vertical alignment were whatever font happened to supply each glyph.  Every icon
# here is authored on the same 24x24 grid and stroked with the same pen, so the set
# stays consistent at any pixel size, needs no bundled files, and follows the theme
# tokens.  Keep new icons single-stroke outlines: no fills, no two-tone shapes.
_ICON_GRID = 24.0
_ICON_STROKE = 1.9          # in grid units: ~1.4 px at an 18 px icon


def _draw_icon(painter: QPainter, kind: str, colour: QColor) -> None:
    """Stroke one glyph onto the shared 24x24 grid."""
    if kind == "play":                      # Live Capture
        path = QPainterPath(QPointF(9.0, 5.5))
        path.lineTo(19.0, 12.0); path.lineTo(9.0, 18.5); path.closeSubpath()
        painter.drawPath(path)
    elif kind == "layers":                  # Saved Runs
        top = QPainterPath(QPointF(12.0, 3.0))
        top.lineTo(21.0, 8.0); top.lineTo(12.0, 13.0); top.lineTo(3.0, 8.0)
        top.closeSubpath()
        painter.drawPath(top)
        painter.drawPolyline(QPolygonF([QPointF(3.0, 12.6), QPointF(12.0, 17.6), QPointF(21.0, 12.6)]))
    elif kind == "sliders":                 # Settings
        painter.drawLine(QLineF(3.5, 8.5, 20.5, 8.5))
        painter.drawLine(QLineF(9.0, 5.5, 9.0, 11.5))
        painter.drawLine(QLineF(3.5, 15.5, 20.5, 15.5))
        painter.drawLine(QLineF(15.0, 12.5, 15.0, 18.5))
    elif kind == "info":                    # About
        painter.drawEllipse(QRectF(3.6, 3.6, 16.8, 16.8))
        painter.drawLine(QLineF(12.0, 11.0, 12.0, 16.6))
        painter.drawLine(QLineF(12.0, 7.7, 12.0, 7.9))   # round cap -> dot
    elif kind == "tray":                    # empty state anchor
        painter.drawRoundedRect(QRectF(3.0, 5.0, 18.0, 14.0), 2.6, 2.6)
        painter.drawPolyline(QPolygonF([
            QPointF(3.0, 13.0), QPointF(8.0, 13.0), QPointF(9.6, 15.6),
            QPointF(14.4, 15.6), QPointF(16.0, 13.0), QPointF(21.0, 13.0),
        ]))
    elif kind == "refresh":                 # header action
        centre, radius, start = QPointF(12.0, 12.0), 7.6, 65.0
        box = QRectF(centre.x() - radius, centre.y() - radius, radius * 2, radius * 2)
        arc = QPainterPath()
        arc.arcMoveTo(box, start)
        arc.arcTo(box, start, -295.0)
        painter.drawPath(arc)
        # Arrow head on the open end, aimed along the arc's tangent there.
        angle = math.radians(start)
        tip_dir = QPointF(-math.sin(angle), -math.cos(angle))          # increasing angle
        outward = QPointF(math.cos(angle), -math.sin(angle))
        at = QPointF(centre.x() + radius * math.cos(angle), centre.y() - radius * math.sin(angle))
        painter.setBrush(colour)
        painter.drawPolygon(QPolygonF([
            at + tip_dir * 2.9,
            at - tip_dir * 1.1 + outward * 2.0,
            at - tip_dir * 1.1 - outward * 2.0,
        ]))
        painter.setBrush(Qt.NoBrush)
    else:
        raise KeyError(f"unknown icon: {kind!r}")


def icon_pixmap(kind: str, size: int = 18, colour: str | None = None,
                stroke: float = _ICON_STROKE) -> QPixmap:
    """Render one icon at `size` logical pixels, sharp on high-DPI screens."""
    app = QApplication.instance()
    ratio = float(app.devicePixelRatio()) if app is not None else 1.0
    pixels = max(1, int(round(size * ratio)))
    pixmap = QPixmap(pixels, pixels)
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.scale(pixels / _ICON_GRID, pixels / _ICON_GRID)
    tint = QColor(colour or TOKENS["text_secondary"])
    pen = QPen(tint)
    pen.setWidthF(stroke)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    _draw_icon(painter, kind, tint)
    painter.end()
    return pixmap


def state_icon(kind: str, size: int = 18) -> QIcon:
    """Icon whose tint tracks the button states the sidebar QSS already styles."""
    icon = QIcon()
    icon.addPixmap(icon_pixmap(kind, size, TOKENS["text_secondary"]), QIcon.Normal, QIcon.Off)
    icon.addPixmap(icon_pixmap(kind, size, TOKENS["text_primary"]), QIcon.Normal, QIcon.On)
    icon.addPixmap(icon_pixmap(kind, size, TOKENS["text_primary"]), QIcon.Active, QIcon.Off)
    icon.addPixmap(icon_pixmap(kind, size, TOKENS["text_primary"]), QIcon.Active, QIcon.On)
    return icon


class CaptureProtocolError(RuntimeError):
    """Raised when a capture cannot prove it contains real, complete tensors."""


def array_to_blob(array: np.ndarray) -> bytes:
    """Serialize an exact captured array without converting its numeric type."""
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return zlib.compress(stream.getvalue(), level=6)


def blob_to_array(blob: bytes) -> np.ndarray:
    """Restore an array previously written by array_to_blob."""
    return np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False)


def tensor_to_numpy(tensor) -> np.ndarray:
    """Move a captured device tensor to host, preserving every bit of its value.

    numpy has no bfloat16 dtype, so bf16 captures are *widened* to float32.  Every
    bf16 value is exactly representable in float32, so this is lossless and
    reversible.  It is not the downcast this project forbids: nothing is narrowed
    and no value changes.
    """
    import torch

    host = tensor.detach().to("cpu")
    if host.dtype == torch.bfloat16:
        host = host.to(torch.float32)
    return host.numpy()


def numeric_sample(array: np.ndarray, size: int = 5) -> np.ndarray:
    """Return the leading 5×5 (or smaller) real values for compact text display."""
    matrix = np.asarray(array)
    while matrix.ndim > 2:
        matrix = matrix[0]
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if matrix.ndim == 0:
        matrix = matrix.reshape(1, 1)
    return matrix[:size, :size]


def display_matrix(array: np.ndarray, maximum: int = 96) -> np.ndarray:
    """Stride-sample an existing tensor only for display; it never changes persisted data."""
    matrix = np.asarray(array)
    while matrix.ndim > 2:
        matrix = matrix[0]
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    row_step = max(1, int(np.ceil(matrix.shape[0] / maximum)))
    col_step = max(1, int(np.ceil(matrix.shape[1] / maximum)))
    return matrix[::row_step, ::col_step]


def sample_text(array: np.ndarray) -> str:
    return np.array2string(numeric_sample(array), precision=5, suppress_small=False, max_line_width=95)


def top_logit_indices(array: np.ndarray, count: int) -> list[int]:
    """Indices of the `count` largest scores, descending, ties broken by ascending index.

    The tie rule is not cosmetic.  Greedy decoding selects with argmax, which returns the
    *first* maximum, so any ranking shown beside "the model selected this" must break ties
    the same way or entry 0 can disagree with the token the model actually emitted.
    `np.argsort(x)[::-1]` reverses ties and gets this wrong.

    argpartition keeps this O(V) rather than sorting a 151k-entry vocabulary, which the
    previous per-render full argsort did.
    """
    flat = np.asarray(array).reshape(-1)
    count = max(0, min(int(count), flat.size))
    if count == 0:
        return []
    if count < flat.size:
        window = np.argpartition(-flat, count - 1)[:count]
    else:
        window = np.arange(flat.size)
    # Stable sort of the small window: equal scores keep their ascending-index order.
    return [int(index) for index in window[np.argsort(-flat[window], kind="stable")]]


@dataclass
class LayerCapture:
    """Exact tensors received for one model transformer layer."""
    index: int
    tensors: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RunCapture:
    """An in-memory representation of a validated capture."""
    prompt: str
    response: str = ""
    token_ids: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    prompt_tokens: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    layers: dict[int, LayerCapture] = field(default_factory=dict)
    generated_embedding: np.ndarray | None = None
    generated_layers: dict[int, LayerCapture] = field(default_factory=dict)
    # The exact score vector the model used to pick the first generated token: the row of
    # the prefill logits belonging to the final prompt position.  Optional because runs
    # saved before schema 3 do not have one; the recap degrades rather than refusing them.
    logits: np.ndarray | None = None
    # One record per generated token: the real scores that chose it, read out of the
    # forward pass that produced them.  Empty for runs saved before schema 4, which is
    # why every invariant below is guarded on the list being non-empty.
    generation_trace: list[GenerationDecision] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    completed_at: str | None = None

    def validate(self) -> None:
        """Reject incomplete captures so the UI never presents invented computations."""
        if not self.response:
            raise CaptureProtocolError("Capture ended without a generated response.")
        if self.embedding is None:
            raise CaptureProtocolError("Capture did not include the real embedding output.")
        if self.generated_embedding is None:
            raise CaptureProtocolError("Capture did not include the first generated-token forward pass embedding.")
        if not self.prompt_token_ids or len(self.prompt_token_ids) != len(self.prompt_tokens):
            raise CaptureProtocolError("Capture did not include consistent prompt tokenization.")
        if not self.token_ids or len(self.token_ids) != len(self.tokens):
            raise CaptureProtocolError("Capture token IDs and token text are missing or inconsistent.")
        if not self.layers:
            raise CaptureProtocolError("Capture did not include any transformer layers.")
        for index, layer in self.layers.items():
            missing = REQUIRED_LAYER_TENSORS - set(layer.tensors)
            if missing:
                raise CaptureProtocolError(f"Layer {index} is missing: {', '.join(sorted(missing))}.")
        for index, layer in self.generated_layers.items():
            missing = REQUIRED_LAYER_TENSORS - set(layer.tensors)
            if missing:
                raise CaptureProtocolError(f"Generated-token layer {index} is missing: {', '.join(sorted(missing))}.")
        if set(self.layers) != set(self.generated_layers):
            raise CaptureProtocolError("Prefill and generated-token captures do not cover the same layers.")
        if self.metadata.get("capture_source") != CAPTURE_SOURCE:
            raise CaptureProtocolError(f"Capture did not attest capture_source={CAPTURE_SOURCE}.")
        if self.metadata.get("faithful_to_upstream_eager") is not True:
            raise CaptureProtocolError(
                "Capture did not verify its attention implementation against upstream "
                "eager_attention_forward; the values cannot be trusted as exact."
            )
        if self.metadata.get("logits_match_stock_eager") is not True:
            raise CaptureProtocolError(
                "Capture did not prove its logits match stock eager attention, so the "
                "capture machinery may have changed what the model computed."
            )
        # Runs from before schema 3 carry no logits, so their absence is not an error.  When
        # they are present they must be the vector that actually chose the shown token,
        # otherwise the recap would attribute a real score list to the wrong word.
        if self.logits is not None:
            if self.logits.ndim != 1:
                raise CaptureProtocolError(
                    f"Final logits must be the one score per vocabulary entry for a single "
                    f"position; got shape {self.logits.shape}."
                )
            chosen = int(np.asarray(self.logits).argmax())
            if chosen != self.token_ids[0]:
                raise CaptureProtocolError(
                    f"Final logits peak at token {chosen} but the capture reports the model "
                    f"generated token {self.token_ids[0]}; the two cannot both be real."
                )
        # Per-token decisions arrived in schema 4.  An empty trace means a run saved before
        # that and must still validate, so every check here is inside this guard.  A trace
        # that *is* present has to describe the tokens the capture reports, or the Next
        # Token Explorer would attribute real scores to the wrong step.
        if self.generation_trace:
            if len(self.generation_trace) != len(self.token_ids):
                raise CaptureProtocolError(
                    f"Capture recorded {len(self.generation_trace)} generation decisions for "
                    f"{len(self.token_ids)} generated tokens; the trace does not cover the run."
                )
            for position, decision in enumerate(self.generation_trace):
                if decision.step != position:
                    raise CaptureProtocolError(
                        f"Generation decision {position} reports step {decision.step}; the "
                        f"trace is out of order."
                    )
                if decision.selected_token_id != self.token_ids[position]:
                    raise CaptureProtocolError(
                        f"Step {position} recorded token {decision.selected_token_id} but the "
                        f"capture emitted {self.token_ids[position]}."
                    )
                scores = {int(entry["id"]): float(entry["logit"])
                          for entry in decision.top_candidates}
                if decision.selected_token_id not in scores:
                    raise CaptureProtocolError(
                        f"Step {position} does not list the token it selected "
                        f"({decision.selected_token_id}) among its candidates."
                    )
                # Stated as "nothing scored higher" rather than "the winner is first" so an
                # exact tie -- which argmax resolves by index -- is not a spurious rejection.
                better = [identifier for identifier, score in scores.items()
                          if score > decision.selected_logit]
                if better:
                    raise CaptureProtocolError(
                        f"Step {position} selected token {decision.selected_token_id} while "
                        f"token {better[0]} scored higher; greedy decoding cannot do that."
                    )
            first = self.generation_trace[0]
            if self.logits is not None:
                stored = float(np.asarray(self.logits)[self.token_ids[0]])
                # isfinite first: a NaN-producing model should get a NaN diagnosis rather
                # than an equality failure that reads like a bookkeeping bug.
                if not (math.isfinite(stored) and math.isfinite(first.selected_logit)):
                    raise CaptureProtocolError(
                        "The first generated token's score is not a finite number."
                    )
                if first.selected_logit != stored:
                    raise CaptureProtocolError(
                        f"Step 0 recorded score {first.selected_logit} for the selected token "
                        f"but the stored logits vector says {stored}."
                    )


class CaptureAttention:
    """An attention implementation that mirrors upstream's eager path and captures
    the pre-softmax scores in flight.

    This is not a reconstruction: `scores` is the exact tensor handed to softmax by
    the code that actually ran.  Faithfulness is verified on the first call by
    re-running upstream's own `eager_attention_forward` with identical inputs and
    requiring bitwise-equal outputs.  If a transformers upgrade changes the eager
    path, `faithful` goes False and validate() refuses the capture.
    """

    def __init__(self, module_path: str) -> None:
        source = import_module(module_path)
        for symbol in ("repeat_kv", "eager_attention_forward"):
            if not hasattr(source, symbol):
                raise CaptureProtocolError(
                    f"{module_path} has no {symbol}; this architecture cannot be captured."
                )
        self.repeat_kv = source.repeat_kv
        self.reference = source.eager_attention_forward
        self.sink: dict | None = None
        self.faithful: bool | None = None
        self.calls = 0
        self.verified_calls = 0

    def __call__(self, module, query, key, value, attention_mask, scaling,
                 dropout=0.0, **kwargs):
        import torch

        # Mirrors upstream eager_attention_forward line for line.
        key_states = self.repeat_kv(key, module.num_key_value_groups)
        value_states = self.repeat_kv(value, module.num_key_value_groups)

        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
        attn_output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()

        self.calls += 1
        # Verify against upstream's own implementation on every call that is actually
        # being captured -- i.e. every layer of both captured phases -- not just the
        # first one. Checking a single call once passed while later layers diverged.
        if self.sink is not None or self.faithful is None:
            reference_output, reference_weights = self.reference(
                module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
            matches = bool(torch.equal(attn_output, reference_output)
                           and torch.equal(weights, reference_weights))
            self.faithful = matches if self.faithful is None else (self.faithful and matches)
            self.verified_calls += 1

        if self.sink is not None:
            self.sink["attention_scores"] = scores.detach()
            self.sink["attention_weights"] = weights.detach()
            self.sink["q_attended"] = query.detach()
            self.sink["k_attended"] = key_states.detach()
            self.sink["v_attended"] = value_states.detach()
        return attn_output, weights


class ModelCapture:
    """Loads a local Hugging Face model and captures its exact intermediate tensors.

    Capture scope is fixed: prompt prefill plus the first generated token.  Any
    further tokens are generated with capture switched off, purely to show the
    reader a complete answer.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.attention: CaptureAttention | None = None
        self._scratch: dict[tuple[int, str], Any] = {}
        self._embedding_scratch: dict[str, Any] = {}
        self._per_layer: dict[int, dict] = {}
        self._capturing = False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def describe(self) -> str:
        if not self.loaded:
            return "no model loaded"
        config = self.model.config
        return (f"{self.model_id} — {len(self._layers())} layers, "
                f"hidden {config.hidden_size}, {config.num_attention_heads} heads, "
                f"on {self.device}")

    def _layers(self):
        return self.model.model.layers

    def load(self, progress: Callable[[str], None] | None = None) -> None:
        """Import torch, fetch the model, and install the capture machinery."""
        def announce(message: str) -> None:
            if progress:
                progress(message)

        announce("Importing torch...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        announce(f"Loading {self.model_id} on {self.device} (first run downloads weights)...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        weight_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        # attn_implementation="eager" is mandatory: fused/flash attention kernels never
        # materialize the full attention matrix, so it could not be captured.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=weight_dtype, attn_implementation="eager")
        except TypeError:  # older transformers spell this torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=weight_dtype, attn_implementation="eager")
        self.model = model.to(self.device).eval()

        announce("Installing capture hooks...")
        self._install()
        announce(f"Ready — {self.describe()}")

    @staticmethod
    def _registry(module_path: str, name: str):
        try:
            return getattr(import_module(module_path), name)
        except (ImportError, AttributeError) as exc:
            raise CaptureProtocolError(
                f"Could not find {name} in this transformers version; "
                "pre-softmax attention scores cannot be captured."
            ) from exc

    def _register_implementation(self, name: str) -> None:
        """Register the capture implementation as a first-class attention backend.

        Two subtleties, both of which silently corrupt the run if you get them wrong:

        1. Registering the attention function alone is NOT enough. transformers builds
           the *causal mask* through a separate registry keyed by the same
           implementation name, so the "eager" mask builder must be registered under
           this name too.
        2. It must be registered with `.register()`, not `registry[name] = ...`.
           `__setitem__` writes to a per-instance `_local_mapping`, but
           `masking_utils._preprocess_mask_arguments` tests membership against the
           class-wide `_global_mapping` and treats a miss as "custom backend that needs
           no mask" -- returning `None`. A `None` mask makes the eager path skip masking
           entirely, so attention becomes bidirectional: every tensor still looks real
           and every value is genuinely computed, but the model is reading the future.
        """
        attentions = self._registry("transformers.modeling_utils", "ALL_ATTENTION_FUNCTIONS")
        attentions.register(name, self.attention)

        masks = self._registry("transformers.masking_utils", "ALL_MASK_ATTENTION_FUNCTIONS")
        if "eager" not in masks:
            raise CaptureProtocolError(
                "transformers has no 'eager' mask builder to borrow; cannot capture safely.")
        masks.register(name, masks["eager"])

        # Belt and braces: if a future version drops `_global_mapping`, the mask would
        # go quietly missing again. Fail loudly instead.
        global_mapping = getattr(type(masks), "_global_mapping", None)
        if global_mapping is None or name not in global_mapping:
            raise CaptureProtocolError(
                f"{name} did not reach the global mask registry; refusing to capture, "
                "because the causal mask would silently be dropped.")

    def _install(self) -> None:
        layers = self._layers()
        module_path = type(layers[0].self_attn).__module__
        self.attention = CaptureAttention(module_path)
        self._register_implementation("tensorscope_capture")

        self.model.config._attn_implementation = "tensorscope_capture"
        for layer in layers:
            layer.self_attn.config._attn_implementation = "tensorscope_capture"

        def first_tensor(value):
            return value[0] if isinstance(value, tuple) else value

        def record(layer_index: int, name: str):
            def hook(_module, _inputs, output):
                if self._capturing:
                    self._scratch[(layer_index, name)] = first_tensor(output).detach()
            return hook

        def embed_hook(_module, _inputs, output):
            if self._capturing:
                self._embedding_scratch["embedding"] = first_tensor(output).detach()

        self.model.model.embed_tokens.register_forward_hook(embed_hook)

        for index, layer in enumerate(layers):
            self._per_layer[index] = {}
            layer.input_layernorm.register_forward_hook(record(index, "normalized_input"))
            layer.self_attn.q_proj.register_forward_hook(record(index, "q"))
            layer.self_attn.k_proj.register_forward_hook(record(index, "k"))
            layer.self_attn.v_proj.register_forward_hook(record(index, "v"))
            layer.self_attn.o_proj.register_forward_hook(record(index, "attention_output"))
            layer.register_forward_hook(record(index, "layer_output"))

            def bind(layer_index):
                def pre_hook(_module, _args, _kwargs):
                    # Point the attention implementation at this layer's slot.
                    self.attention.sink = self._per_layer[layer_index] if self._capturing else None
                    return None
                return pre_hook
            layer.self_attn.register_forward_pre_hook(bind(index), with_kwargs=True)

    def _drain(self) -> tuple[dict[int, LayerCapture], np.ndarray]:
        """Convert one forward pass worth of captures to host arrays and reset."""
        layers: dict[int, LayerCapture] = {}
        for index in range(len(self._layers())):
            tensors = {name: tensor_to_numpy(tensor)
                       for (layer_i, name), tensor in self._scratch.items() if layer_i == index}
            tensors.update({name: tensor_to_numpy(tensor)
                            for name, tensor in self._per_layer[index].items()})
            layers[index] = LayerCapture(index, tensors)
        if "embedding" not in self._embedding_scratch:
            raise CaptureProtocolError("The embedding hook did not fire; capture is incomplete.")
        embedding = tensor_to_numpy(self._embedding_scratch["embedding"])
        self._scratch.clear()
        self._embedding_scratch.clear()
        for slot in self._per_layer.values():
            slot.clear()
        return layers, embedding

    def _set_implementation(self, name: str) -> None:
        self.model.config._attn_implementation = name
        for layer in self._layers():
            layer.self_attn.config._attn_implementation = name

    def _verify_undisturbed(self, input_ids, captured_logits) -> float:
        """Re-run the same prefill with stock eager attention and require identical logits.

        This is the check that matters. Verifying the capture implementation in isolation
        is not enough: a capture backend can be a perfect mirror of the eager math and
        still be handed a differently-built causal mask, yielding plausible-but-wrong
        values that all look real. Comparing end-to-end logits catches that.
        """
        import torch

        was_capturing = self._capturing
        self._capturing = False
        self.attention.sink = None
        self._set_implementation("eager")
        try:
            with torch.no_grad():
                stock = self.model(input_ids=input_ids, use_cache=False).logits
        finally:
            self._set_implementation("tensorscope_capture")
            self._capturing = was_capturing

        delta = float((stock.to(torch.float32) - captured_logits.to(torch.float32)).abs().max())
        if delta != 0.0:
            raise CaptureProtocolError(
                f"Capture perturbed the computation: prefill logits differ from stock "
                f"eager attention by {delta:.3e}. The captured tensors would not be the "
                "ones this model computes normally, so the run was discarded."
            )
        return delta

    def _top_logits(self, logits: np.ndarray,
                    count: int = CANDIDATE_TOP_K) -> list[dict[str, Any]]:
        """Name the highest-scoring next tokens, for display.

        Reads the numpy array rather than the torch tensor, so the scores shown are the
        ones that get persisted.  For the first generated token that array is the whole
        vocabulary vector the run also stores, so the table there describes a value the
        reader can check; for later tokens only this Top-K survives.

        Turning an id back into text needs the tokenizer, which a saved-run viewer has no
        access to, so decoding happens here at capture time and travels with the record.
        """
        return [{"id": index, "token": self.tokenizer.decode([index]),
                 "logit": float(logits[index])}
                for index in top_logit_indices(logits, count)]

    def _templated(self, prompt: str) -> str:
        """Format the prompt the way the model expects, when it defines a template."""
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        for extra in ({"enable_thinking": False}, {}):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **extra)
            except TypeError:
                continue
        return prompt

    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        for source in (getattr(self.tokenizer, "eos_token_id", None),
                       getattr(self.model.generation_config, "eos_token_id", None)):
            if isinstance(source, int):
                ids.add(source)
            elif isinstance(source, (list, tuple)):
                ids.update(int(i) for i in source)
        return ids

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 on_text: Callable[[str], None] | None = None) -> RunCapture:
        import torch

        if not self.loaded:
            raise CaptureProtocolError("No model is loaded.")

        encoded = self.tokenizer(self._templated(prompt), return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] > MAX_CAPTURE_TOKENS:
            raise CaptureProtocolError(
                f"Prompt is {input_ids.shape[1]} tokens; TensorScope captures at most "
                f"{MAX_CAPTURE_TOKENS}. Attention tensors grow with the square of the "
                "prompt length, so longer prompts would produce enormous runs."
            )

        capture = RunCapture(prompt=prompt)
        capture.prompt_token_ids = [int(i) for i in input_ids[0]]
        capture.prompt_tokens = [str(t) for t in
                                 self.tokenizer.convert_ids_to_tokens(input_ids[0])]

        def emit(token_id: int) -> None:
            text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if not text:  # keep the response non-empty even if the model stops at once
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            capture.token_ids.append(token_id)
            capture.tokens.append(text)
            capture.response += text
            if on_text and text:
                on_text(text)

        def record(row: Any, token_id: int, *, attested: bool) -> None:
            """Store the scores that chose the token just emitted.

            `attested` says whether a verification gate covers this vector.  Only the
            prefill is re-run under stock eager attention, so only step 0 is True; later
            steps are genuine forward-pass output that nothing independently re-derived,
            and the UI must not present the two at equal confidence.
            """
            scores = row if isinstance(row, np.ndarray) else tensor_to_numpy(row)
            capture.generation_trace.append(GenerationDecision(
                step=len(capture.token_ids) - 1,
                selected_token_id=token_id,
                selected_logit=float(scores[token_id]),
                top_candidates=self._top_logits(scores, CANDIDATE_TOP_K),
                attested=attested))
            # Not retained: tensor_to_numpy returns a view that shares memory with the
            # torch tensor when the device is already CPU.

        eos_ids = self._eos_ids()
        stop_reason = "max_new_tokens"
        stop_token_id: int | None = None
        stop_token_logit: float | None = None
        with torch.no_grad():
            # --- captured: prompt prefill ---
            self._capturing = True
            prefill = self.model(input_ids=input_ids, use_cache=True)
            capture.layers, capture.embedding = self._drain()

            # Prove the capture machinery did not change what the model computes.
            logits_delta = self._verify_undisturbed(input_ids, prefill.logits)

            first_id = int(prefill.logits[:, -1, :].argmax(dim=-1)[0])

            # Keep the score vector that chose that token: the final prompt position's row.
            # _verify_undisturbed has just proved this exact tensor is bit-identical to the
            # one stock eager attention produces, so the numbers the recap shows here carry
            # that proof with them.  Only this row is stored -- the full (1, tokens, vocab)
            # prefill logits would be tens of megabytes per run, and this is the row that
            # actually decided the word.
            capture.logits = tensor_to_numpy(prefill.logits[0, -1, :])

            # --- captured: first generated token ---
            step = self.model(
                input_ids=torch.tensor([[first_id]], device=self.device),
                past_key_values=prefill.past_key_values, use_cache=True)
            capture.generated_layers, capture.generated_embedding = self._drain()
            self._capturing = False
            self.attention.sink = None

            emit(first_id)
            # The real scores behind this choice, read from the tensor the prefill already
            # produced.  No extra forward pass, no recomputation: `capture.logits` is the
            # array _verify_undisturbed just proved bit-identical to stock eager, so step 0
            # is the one decision covered by that gate.
            record(capture.logits, first_id, attested=True)

            # --- tensors no longer captured, decisions still recorded ---------------
            # Layer tensors stop here: retaining full attention matrices for every token
            # would make runs enormous.  The decision telemetry below costs one already
            # materialised logits row per token and no additional model work.
            past = step.past_key_values
            logits = step.logits[:, -1, :]
            for _ in range(max(0, max_new_tokens - 1)):
                token_id = int(logits.argmax(dim=-1)[0])
                if token_id in eos_ids:
                    # This decision ends the answer without emitting a token, so it would
                    # otherwise be discarded -- and "why did it stop there?" is the first
                    # question the response chip strip provokes.
                    stop_reason = "eos"
                    stop_token_id = token_id
                    stop_token_logit = float(tensor_to_numpy(logits[0])[token_id])
                    break
                emit(token_id)
                # `logits` here belongs to the pass that ran *before* this token was fed in,
                # i.e. the pass whose final position chose it.  Recording after emit means
                # step == len(token_ids) - 1, which cannot drift out of step with the token
                # list the way a separately maintained counter could.
                record(logits[0], token_id, attested=False)
                step = self.model(
                    input_ids=torch.tensor([[token_id]], device=self.device),
                    past_key_values=past, use_cache=True)
                past = step.past_key_values
                logits = step.logits[:, -1, :]

        capture.metadata = {
            "capture_source": CAPTURE_SOURCE,
            "faithful_to_upstream_eager": self.attention.faithful,
            "logits_match_stock_eager": True,  # _verify_undisturbed raises otherwise
            "logits_max_abs_diff_vs_stock_eager": logits_delta,
            # One computation behind step 0's candidate list, not two that could diverge.
            "final_logits_top": capture.generation_trace[0].top_candidates,
            "decoding": "greedy",
            "candidate_top_k": CANDIDATE_TOP_K,
            "stop_reason": stop_reason,
            "stop_token_id": stop_token_id,
            "stop_token_logit": stop_token_logit,
            "schema_version": SCHEMA_VERSION,
            "model": self.model_id,
            "backend": (torch.cuda.get_device_name(0) if self.device == "cuda"
                        else "cpu"),
            "torch": torch.__version__,
            "weight_dtype": str(next(self.model.parameters()).dtype),
            "attention_calls": self.attention.calls,
            "attention_calls_verified": self.attention.verified_calls,
        }
        capture.completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        capture.validate()
        return capture


class RunDatabase:
    """SQLite persistence for exact capture blobs and searchable run metadata."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        self._setup()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def _setup(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript("""
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    token_ids_json TEXT NOT NULL,
                    tokens_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tensors (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    layer_index INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    shape_json TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    data_blob BLOB NOT NULL,
                    PRIMARY KEY (run_id, layer_index, name)
                );
                -- Schema 4.  CREATE TABLE IF NOT EXISTS means opening a database written
                -- by an earlier version simply adds this table; its existing rows are
                -- untouched and its runs load with an empty trace.
                -- No ON DELETE CASCADE here, unlike the tensors table above: PRAGMA
                -- foreign_keys is per-connection and only this one sets it, so a cascade
                -- declared here would never actually fire.
                CREATE TABLE IF NOT EXISTS generation_steps (
                    run_id INTEGER NOT NULL REFERENCES runs(id),
                    step INTEGER NOT NULL,
                    selected_token_id INTEGER NOT NULL,
                    selected_logit REAL NOT NULL,
                    attested INTEGER NOT NULL,
                    -- The decoded candidate text lives in here rather than in its own TEXT
                    -- column: byte-level BPE can emit lone surrogates, which sqlite3
                    -- rejects on a TEXT bind but json.dumps encodes without complaint.
                    candidates_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, step)
                );
            """)
            db.commit()

    def save(self, capture: RunCapture) -> int:
        capture.validate()
        with closing(self.connect()) as db:
            cursor = db.execute(
                """INSERT INTO runs (created_at, completed_at, prompt, response, token_ids_json,
                   tokens_json, metadata_json, embedding_blob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (capture.started_at, capture.completed_at, capture.prompt, capture.response,
                 json.dumps(capture.token_ids), json.dumps(capture.tokens),
                 json.dumps({**capture.metadata, "prompt_token_ids": capture.prompt_token_ids,
                             "prompt_tokens": capture.prompt_tokens}), array_to_blob(capture.embedding)),
            )
            run_id = int(cursor.lastrowid)
            for index, layer in capture.layers.items():
                for name, tensor in layer.tensors.items():
                    db.execute(
                        "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, index, name, json.dumps(list(tensor.shape)), str(tensor.dtype), array_to_blob(tensor)),
                    )
            db.execute(
                "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, -1, "first_generated_token:embedding",
                 json.dumps(list(capture.generated_embedding.shape)), str(capture.generated_embedding.dtype),
                 array_to_blob(capture.generated_embedding)),
            )
            for index, layer in capture.generated_layers.items():
                for name, tensor in layer.tensors.items():
                    db.execute(
                        "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                        (run_id, index, f"first_generated_token:{name}", json.dumps(list(tensor.shape)),
                         str(tensor.dtype), array_to_blob(tensor)),
                    )
            # Not a layer tensor, so it rides at layer_index -1 like the generated-token
            # embedding above.  The tensors table needs no new column for this.
            if capture.logits is not None:
                db.execute(
                    "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, -1, "final_logits", json.dumps(list(capture.logits.shape)),
                     str(capture.logits.dtype), array_to_blob(capture.logits)),
                )
            # Inside the same transaction as the tensors, so a run can never be committed
            # with a trace that only half describes it.
            for decision in capture.generation_trace:
                db.execute(
                    "INSERT INTO generation_steps VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, decision.step, decision.selected_token_id,
                     decision.selected_logit, int(decision.attested),
                     json.dumps(decision.top_candidates)),
                )
            db.commit()
        return run_id

    def list_runs(self) -> list[sqlite3.Row]:
        with closing(self.connect()) as db:
            return db.execute("SELECT id, created_at, prompt, response FROM runs ORDER BY id DESC").fetchall()

    def load(self, run_id: int) -> RunCapture:
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Run {run_id} was not found.")
            capture = RunCapture(
                prompt=row["prompt"], response=row["response"], token_ids=json.loads(row["token_ids_json"]),
                tokens=json.loads(row["tokens_json"]), metadata=json.loads(row["metadata_json"]),
                embedding=blob_to_array(row["embedding_blob"]), started_at=row["created_at"], completed_at=row["completed_at"],
            )
            capture.prompt_token_ids = capture.metadata.pop("prompt_token_ids", [])
            capture.prompt_tokens = capture.metadata.pop("prompt_tokens", [])
            for tensor_row in db.execute("SELECT * FROM tensors WHERE run_id = ? ORDER BY layer_index", (run_id,)):
                name = tensor_row["name"]
                tensor = blob_to_array(tensor_row["data_blob"])
                if name == "first_generated_token:embedding":
                    capture.generated_embedding = tensor
                elif name == "final_logits":
                    # Must be matched before the layer branch below: it is stored at
                    # layer_index -1, and falling through would invent a layer -1 that
                    # validate() then rejects for missing every required tensor.
                    capture.logits = tensor
                elif name.startswith("first_generated_token:"):
                    index = tensor_row["layer_index"]
                    layer = capture.generated_layers.setdefault(index, LayerCapture(index))
                    layer.tensors[name.split(":", 1)[1]] = tensor
                else:
                    layer = capture.layers.setdefault(tensor_row["layer_index"], LayerCapture(tensor_row["layer_index"]))
                    layer.tensors[name] = tensor
            # Read inside this block and before validate(), which sits outside it: after
            # validate() every reopened run would be checked with an empty trace and the
            # schema-4 invariants would never run on the path that matters most.
            #
            # An explicit existence check rather than try/except, because the table is
            # genuinely absent in two supported cases: a database written before schema 4,
            # and a read-only connection whose _setup() never ran (see test_recap.py).
            has_trace = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='generation_steps'"
            ).fetchone()
            if has_trace:
                capture.generation_trace = [
                    GenerationDecision(
                        step=int(step_row["step"]),
                        selected_token_id=int(step_row["selected_token_id"]),
                        selected_logit=float(step_row["selected_logit"]),
                        top_candidates=json.loads(step_row["candidates_json"]),
                        attested=bool(step_row["attested"]),
                    )
                    for step_row in db.execute(
                        "SELECT * FROM generation_steps WHERE run_id = ? ORDER BY step",
                        (run_id,))
                ]
        capture.validate()
        return capture


class LoadWorker(QThread):
    """Import torch and load model weights off the Qt event loop."""
    progress = pyqtSignal(str)
    loaded = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, capture_model: ModelCapture) -> None:
        super().__init__()
        self.capture_model = capture_model

    def run(self) -> None:
        try:
            self.capture_model.load(self.progress.emit)
            self.loaded.emit(self.capture_model.describe())
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=2)}")


class GenerationWorker(QThread):
    """Keep blocking local inference and capture outside Qt's event loop."""
    text_received = pyqtSignal(str)
    finished_capture = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, capture_model: ModelCapture, prompt: str, max_new_tokens: int = 128) -> None:
        super().__init__()
        self.capture_model, self.prompt, self.max_new_tokens = capture_model, prompt, max_new_tokens

    def run(self) -> None:
        try:
            capture = self.capture_model.generate(
                self.prompt, self.max_new_tokens, self.text_received.emit)
            self.finished_capture.emit(capture)
        except Exception as exc:  # worker must pass readable errors to Qt, not crash its thread
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=2)}")


# The read-only recap is separated from the protected capture and persistence code.
# Re-export copy/helpers for the existing verification entry points.
from tensorscope_content import (
    TENSOR_LABELS, TENSOR_LABEL_BY_KEY, TENSOR_EXPLANATIONS,
    InternalsStage, INTERNALS_STAGES, INTERNALS_STAGE_INDEX, INTERNALS_STEPS,
    LearnStage, LEARN_STAGES, LEARN_STAGE_INDEX, learn_facts, LOGITS_UNAVAILABLE,
)
from tensorscope_views import InternalsView, LearnView, RawView
from tensorscope_ui import ComputationRecap as _ComputationRecap


class ComputationRecap(_ComputationRecap):
    """Apply the application's detected theme to the independent recap UI."""
    def __init__(self, capture, run_id=None, parent=None):
        super().__init__(capture, run_id, parent, tokens=TOKENS)


class HistoryDialog(QDialog):
    """Database browser that restores a prior recap without model inference."""
    def __init__(self, database: RunDatabase, parent: QWidget | None = None) -> None:
        super().__init__(parent); self.database = database
        self.setWindowTitle("TensorScope — Saved Runs"); self.resize(760, 480)
        layout = QVBoxLayout(self); self.runs = QListWidget(); layout.addWidget(self.runs)
        buttons = QHBoxLayout(); refresh = QPushButton("Refresh"); open_button = QPushButton("Open Recap")
        buttons.addWidget(refresh); buttons.addWidget(open_button); layout.addLayout(buttons)
        refresh.clicked.connect(self.reload); open_button.clicked.connect(self.open_selected)
        self.runs.itemDoubleClicked.connect(lambda _: self.open_selected()); self.reload()

    def reload(self) -> None:
        self.runs.clear()
        for row in self.database.list_runs():
            item_text = f"#{row['id']}  {row['created_at']}\nPrompt: {row['prompt'][:160]}\nResponse: {row['response'][:160]}"
            item = QListWidgetItem(item_text); item.setData(Qt.UserRole, row["id"]); self.runs.addItem(item)

    def open_selected(self) -> None:
        item = self.runs.currentItem()
        if not item:
            return
        try:
            recap = ComputationRecap(self.database.load(item.data(Qt.UserRole)), item.data(Qt.UserRole), self)
            recap.exec_()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open run", str(exc))


class TensorScopeMainWindow(QMainWindow):
    """Compact utility window — fixed sidebar nav + four content screens."""

    def __init__(self, start_in_browse: bool = False, db_path: Path | None = None) -> None:
        super().__init__()
        self.database = RunDatabase(db_path or DB_PATH)
        self.capture_model = ModelCapture(os.getenv("TENSORSCOPE_MODEL", DEFAULT_MODEL_ID))
        self.worker: GenerationWorker | None = None
        self.loader: LoadWorker | None = None
        self._open_recaps: list[ComputationRecap] = []

        self.setWindowTitle("TensorScope")
        self.setMinimumWidth(640)
        self.resize(760, 580)
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(170)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(12, 18, 12, 18)
        sl.setSpacing(2)

        logo_row = QHBoxLayout()
        logo = QLabel()
        logo.setFixedSize(28, 28)
        logo.setObjectName("logoPlaceholder")
        logo_row.addWidget(logo)
        logo_row.addStretch(1)
        sl.addLayout(logo_row)
        sl.addSpacing(20)

        # One painted icon family for the whole sidebar -- see _draw_icon on why these
        # are not text glyphs.
        self._nav_btns: list[QPushButton] = []
        for label, glyph, idx in [
            ("Live Capture", "play",    0),
            ("Saved Runs",   "layers",  1),
            ("Settings",     "sliders", 2),
            ("About",        "info",    3),
        ]:
            btn = QPushButton(label)
            btn.setObjectName("sidebarNav")
            btn.setIcon(state_icon(glyph, 18))
            btn.setIconSize(QSize(18, 18))
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, i=idx: self._nav_to(i))
            sl.addWidget(btn)
            self._nav_btns.append(btn)

        sl.addStretch(1)
        root.addWidget(sidebar)

        # ── Content screens ───────────────────────────────────────────────────
        self.screens = QStackedWidget()
        root.addWidget(self.screens)

        def _card() -> tuple:
            frame = QFrame(); frame.setObjectName("card")
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(22, 16, 22, 16); fl.setSpacing(10)
            return frame, fl

        def _divider() -> QFrame:
            line = QFrame(); line.setObjectName("rowDivider"); return line

        def _icon_button(glyph: str, tooltip: str) -> QPushButton:
            btn = QPushButton()
            btn.setObjectName("iconButton")
            btn.setIcon(state_icon(glyph, 16))
            btn.setIconSize(QSize(16, 16))
            btn.setFixedSize(28, 28)
            btn.setToolTip(tooltip)
            btn.setCursor(Qt.PointingHandCursor)
            return btn

        def _screen(title: str, subtitle: str, action: QWidget | None = None) -> tuple:
            """Returns (page, body layout, footer label).

            The footer label is a status bar pinned below the scroll area, so it keeps
            its separating rule at the bottom of the screen instead of scrolling with
            the content.  It starts hidden; a caller that wants it calls setVisible.
            """
            page = QWidget()
            pl = QVBoxLayout(page)
            pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(0)

            scroll = QScrollArea(); scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            inner = QWidget(); scroll.setWidget(inner)
            l = QVBoxLayout(inner)
            l.setContentsMargins(32, 28, 32, 28); l.setSpacing(14)
            l.setAlignment(Qt.AlignTop)
            hdr = QWidget(); hrow = QHBoxLayout(hdr)
            hrow.setContentsMargins(0, 0, 0, 4); hrow.setSpacing(8)
            titles = QVBoxLayout(); titles.setContentsMargins(0, 0, 0, 0); titles.setSpacing(2)
            t = QLabel(title); t.setObjectName("screenTitle"); titles.addWidget(t)
            if subtitle:
                s = QLabel(subtitle); s.setObjectName("screenSubtitle"); titles.addWidget(s)
            hrow.addLayout(titles)
            hrow.addStretch(1)
            if action is not None:
                hrow.addWidget(action, 0, Qt.AlignTop)
            l.addWidget(hdr)
            pl.addWidget(scroll, 1)

            footer = QLabel(); footer.setObjectName("statusBar")
            footer.setWordWrap(True)   # a long database path must not widen the window
            footer.setTextInteractionFlags(Qt.TextSelectableByMouse)
            footer.setVisible(False)
            pl.addWidget(footer)
            return page, l, footer

        # Screen 0: Live Capture ──────────────────────────────────────────────
        live_scr, live_l, _live_footer = _screen(
            "Live Capture",
            "Load a model in Settings, then run a prompt to capture tensors.",
        )

        ms_card, ms_cl = _card()
        ms_row = QHBoxLayout(); ms_row.setContentsMargins(0, 0, 0, 0)
        ms_lbl = QLabel("Model"); ms_lbl.setObjectName("rowLabel"); ms_row.addWidget(ms_lbl)
        ms_row.addStretch(1)
        self.live_model_status = QLabel("No model loaded")
        self.live_model_status.setObjectName("rowValue")
        self._goto_settings_btn = QPushButton("Open Settings →")
        self._goto_settings_btn.setObjectName("ghost")
        self._goto_settings_btn.clicked.connect(lambda: self._nav_to(2))
        ms_row.addWidget(self.live_model_status)
        ms_row.addSpacing(8); ms_row.addWidget(self._goto_settings_btn)
        ms_cl.addLayout(ms_row)
        live_l.addWidget(ms_card)

        pr_card, pr_cl = _card()
        pr_lbl = QLabel("Prompt"); pr_lbl.setObjectName("rowLabel"); pr_cl.addWidget(pr_lbl)
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Ask the model something…")
        self.prompt.setFixedHeight(108); pr_cl.addWidget(self.prompt)
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Capture"); self.run_button.setEnabled(False)
        run_row.addWidget(self.run_button); run_row.addStretch(1)
        pr_cl.addLayout(run_row)
        live_l.addWidget(pr_card)

        self._out_card, out_cl = _card()
        out_lbl = QLabel("Response"); out_lbl.setObjectName("rowLabel"); out_cl.addWidget(out_lbl)
        self.output = QTextEdit(); self.output.setReadOnly(True)
        self.output.setFixedHeight(130); out_cl.addWidget(self.output)
        self._out_card.setVisible(False)
        live_l.addWidget(self._out_card)

        self.status = QLabel("Load a model in Settings to begin.")
        self.status.setObjectName("statusLabel"); live_l.addWidget(self.status)
        self.screens.addWidget(live_scr)  # index 0

        # Screen 1: Saved Runs ────────────────────────────────────────────────
        _refresh_btn = _icon_button("refresh", "Reload the run list")
        runs_scr, runs_l, self.browse_status = _screen(
            "Saved Runs", "Browse captures without loading a model.", action=_refresh_btn,
        )

        self._browse_inner = QStackedWidget()

        runs_list_w = QWidget()
        rll = QVBoxLayout(runs_list_w)
        rll.setContentsMargins(0, 0, 0, 0); rll.setSpacing(8)
        self.runs_list = QListWidget(); rll.addWidget(self.runs_list)
        browse_btns_row = QHBoxLayout()
        self.open_recap_btn = QPushButton("Open Recap"); self.open_recap_btn.setEnabled(False)
        browse_btns_row.addWidget(self.open_recap_btn)
        browse_btns_row.addStretch(1); rll.addLayout(browse_btns_row)
        self._browse_inner.addWidget(runs_list_w)  # index 0

        # Empty state: no card, no border, no background -- it sits directly on the page.
        # An outline glyph anchors it so the copy is not bare floating text.
        empty_w = QWidget()
        ecl = QVBoxLayout(empty_w)
        ecl.setContentsMargins(32, 24, 32, 24); ecl.setSpacing(0)
        ecl.addStretch(1)
        empty_icon = QLabel()
        empty_icon.setPixmap(icon_pixmap("tray", 44, TOKENS["text_muted"], stroke=1.5))
        empty_icon.setAlignment(Qt.AlignCenter)
        ecl.addWidget(empty_icon)
        ecl.addSpacing(14)
        empty_title = QLabel("No saved runs yet.")
        empty_title.setObjectName("emptyTitle"); empty_title.setAlignment(Qt.AlignCenter)
        ecl.addWidget(empty_title)
        ecl.addSpacing(6)
        self._empty_label = QLabel(
            "Switch to Live Capture on a machine with a GPU, run a prompt,\n"
            "then come back here to browse it."
        )
        self._empty_label.setObjectName("emptyBody")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setWordWrap(True)
        ecl.addWidget(self._empty_label)
        ecl.addSpacing(20)
        choose_row = QHBoxLayout(); choose_row.setContentsMargins(0, 0, 0, 0)
        self.choose_db_btn = QPushButton("Choose Database…")
        self.choose_db_btn.setObjectName("primary")
        self.choose_db_btn.setCursor(Qt.PointingHandCursor)
        choose_row.addStretch(1); choose_row.addWidget(self.choose_db_btn); choose_row.addStretch(1)
        ecl.addLayout(choose_row)
        ecl.addSpacing(10)
        # Fine print, not the primary path: TensorScope ships as a community download,
        # so the file picker above is what most readers will use.
        cli_hint = QLabel("Already have one? The command-line equivalent is  --db /path/to/runs.sqlite3")
        cli_hint.setObjectName("finePrint")
        cli_hint.setAlignment(Qt.AlignCenter); cli_hint.setWordWrap(True)
        ecl.addWidget(cli_hint)
        ecl.addStretch(1)
        self._browse_inner.addWidget(empty_w)  # index 1

        runs_l.addWidget(self._browse_inner, 1)
        self.browse_status.setVisible(True)
        self.screens.addWidget(runs_scr)  # index 1

        # Screen 2: Settings ──────────────────────────────────────────────────
        settings_scr, settings_l, _settings_footer = _screen("Settings", "")

        mdl_card, mdl_cl = _card()
        mdl_title = QLabel("Model"); mdl_title.setObjectName("cardTitle"); mdl_cl.addWidget(mdl_title)
        mdl_cl.addWidget(_divider())
        mdl_row = QHBoxLayout()
        mdl_lbl = QLabel("Model ID"); mdl_lbl.setObjectName("rowLabel"); mdl_row.addWidget(mdl_lbl)
        mdl_row.addStretch(1)
        self.model_id = QLineEdit(self.capture_model.model_id)
        self.model_id.setMinimumWidth(200); self.model_id.setMaximumWidth(300)
        mdl_row.addWidget(self.model_id)
        self.load_button = QPushButton("Load"); mdl_row.addWidget(self.load_button)
        mdl_cl.addLayout(mdl_row)
        mdl_cl.addWidget(_divider())
        self.load_status = QLabel("No model loaded.")
        self.load_status.setObjectName("rowValue"); mdl_cl.addWidget(self.load_status)
        settings_l.addWidget(mdl_card)

        db_card, db_cl = _card()
        db_title = QLabel("Database"); db_title.setObjectName("cardTitle"); db_cl.addWidget(db_title)
        db_cl.addWidget(_divider())
        db_row = QHBoxLayout()
        db_lbl = QLabel("Path"); db_lbl.setObjectName("rowLabel"); db_row.addWidget(db_lbl)
        db_row.addSpacing(16)
        self.db_path_label = QLabel(str(self.database.path))
        self.db_path_label.setObjectName("rowValue"); self.db_path_label.setWordWrap(True)
        self.db_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        db_row.addWidget(self.db_path_label, 1)
        self.settings_choose_db_btn = QPushButton("Change…")
        self.settings_choose_db_btn.setObjectName("ghost")
        db_row.addWidget(self.settings_choose_db_btn, 0, Qt.AlignTop)
        db_cl.addLayout(db_row)
        settings_l.addWidget(db_card)
        self.screens.addWidget(settings_scr)  # index 2

        # Screen 3: About ─────────────────────────────────────────────────────
        about_scr, about_l, _about_footer = _screen("About", "")

        about_card, about_cl = _card()
        about_name = QLabel("TensorScope"); about_name.setObjectName("cardTitle")
        about_cl.addWidget(about_name); about_cl.addWidget(_divider())
        about_body = QLabel(
            "Inspects the exact intermediate tensors produced during a local LLM forward pass.\n\n"
            "Every displayed value — Q, K, V, attention scores and weights, layer outputs — "
            "is a tensor the running model actually produced. "
            "No inference, simulation, or reconstruction."
        )
        about_body.setWordWrap(True); about_body.setObjectName("rowValue")
        about_cl.addWidget(about_body)
        about_l.addWidget(about_card); about_l.addStretch(1)
        self.screens.addWidget(about_scr)  # index 3

        # Signals ─────────────────────────────────────────────────────────────
        self.run_button.clicked.connect(self.run_model)
        self.load_button.clicked.connect(self.load_model)
        self.open_recap_btn.clicked.connect(self._open_selected_run)
        _refresh_btn.clicked.connect(self._reload_runs)
        self.choose_db_btn.clicked.connect(self._choose_database)
        self.settings_choose_db_btn.clicked.connect(self._choose_database)
        self.runs_list.itemDoubleClicked.connect(lambda _: self._open_selected_run())
        self.runs_list.currentItemChanged.connect(
            lambda item, _: self.open_recap_btn.setEnabled(item is not None)
        )

        self._reload_runs()
        self._nav_to(1 if start_in_browse else 0)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _nav_to(self, index: int) -> None:
        self.screens.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_btns):
            btn.setChecked(i == index)

    # ── Browse helpers ────────────────────────────────────────────────────────

    def _choose_database(self) -> None:
        """Point the browser at an existing capture database via a real file picker.

        The GUI path has to work on its own: TensorScope ships as a community download
        and most readers will never open a terminal, so --db is documented as fine print
        rather than as the instruction.
        """
        start_dir = str(self.database.path.parent if self.database.path.parent.exists() else Path.home())
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose TensorScope database", start_dir,
            "TensorScope database (*.sqlite3 *.sqlite *.db);;All files (*)",
        )
        if not chosen:
            return
        path = Path(chosen)
        try:
            # Probe read-only first.  RunDatabase.__init__ runs CREATE TABLE IF NOT
            # EXISTS, which would otherwise quietly add empty tables to whatever
            # unrelated sqlite file was picked.
            with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as probe:
                tables = {r[0] for r in probe.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"runs", "tensors"} <= tables:
                raise ValueError("This file is not a TensorScope database (no runs/tensors tables).")
            database = RunDatabase(path)
            database.list_runs()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open database", f"{path}\n\n{exc}")
            return
        self.database = database
        self.db_path_label.setText(str(self.database.path))
        self._reload_runs()
        self._nav_to(1)

    def _reload_runs(self) -> None:
        self.runs_list.clear()
        rows = self.database.list_runs()
        for row in rows:
            ts = (row["created_at"] or "")[:16].replace("T", " ")
            text = f"#{row['id']}  {ts}  |  {row['prompt'][:120]}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, row["id"])
            self.runs_list.addItem(item)
        n = len(rows)
        self._browse_inner.setCurrentIndex(0 if n > 0 else 1)
        self.open_recap_btn.setEnabled(False)
        if n > 0:
            self.browse_status.setText(
                f"{n} saved run{'s' if n != 1 else ''}  ·  {self.database.path}"
            )
        else:
            self.browse_status.setText(f"Database: {self.database.path}")

    def _open_selected_run(self) -> None:
        item = self.runs_list.currentItem()
        if not item:
            return
        try:
            run_id = item.data(Qt.UserRole)
            recap = ComputationRecap(self.database.load(run_id), run_id, self)
            self._open_recaps.append(recap)
            recap.finished.connect(
                lambda: self._open_recaps.remove(recap) if recap in self._open_recaps else None
            )
            recap.show()
        except Exception as exc:
            QMessageBox.critical(self, "Cannot open run", str(exc))

    # ── Live Capture helpers ──────────────────────────────────────────────────

    def _say(self, message: str, colour: str | None = None) -> None:
        col = colour or TOKENS["text_muted"]
        self.status.setText(message)
        self.status.setStyleSheet(f"color:{col};font-size:12px;padding-top:2px;")
        self.load_status.setText(message)
        self.load_status.setStyleSheet(f"color:{col};font-size:12px;")

    def _set_live_model_label(self, text: str, loaded: bool = False) -> None:
        self.live_model_status.setText(text)
        col = TOKENS["accent"] if loaded else TOKENS["text_muted"]
        self.live_model_status.setStyleSheet(f"color:{col};font-size:12px;")
        self._goto_settings_btn.setVisible(not loaded)

    def load_model(self) -> None:
        wanted = self.model_id.text().strip()
        if not wanted:
            QMessageBox.information(self, "Model required", "Enter a Hugging Face model id.")
            return
        try:
            import torch
            if torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if total_gb < 8.0:
                    QMessageBox.warning(
                        self, "VRAM warning",
                        f"Your GPU has ~{total_gb:.1f} GB VRAM.\n"
                        "Qwen3-4B needs ~7.6 GB peak (bf16).  "
                        "Consider Qwen/Qwen2.5-0.5B-Instruct instead.",
                    )
        except ImportError:
            pass
        self.capture_model = ModelCapture(wanted)
        self.load_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self._say("Loading model…", TOKENS["accent"])
        self._set_live_model_label("Loading…")
        self.loader = LoadWorker(self.capture_model)
        self.loader.progress.connect(lambda msg: self._say(msg, TOKENS["accent"]))
        self.loader.loaded.connect(self.model_ready)
        self.loader.failed.connect(self.load_failed)
        self.loader.start()

    def model_ready(self, description: str) -> None:
        self.load_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self._say(f"Ready — {description}", TOKENS["success"])
        self._set_live_model_label(self.capture_model.model_id, loaded=True)

    def load_failed(self, message: str) -> None:
        self.load_button.setEnabled(True)
        self._say("Model could not be loaded.", "#dc2626")
        self._set_live_model_label("Load failed")
        QMessageBox.critical(self, "Model load failed", message)

    def run_model(self) -> None:
        prompt_text = self.prompt.toPlainText().strip()
        if not prompt_text:
            QMessageBox.information(self, "Prompt required", "Enter a prompt before running.")
            return
        if not self.capture_model.loaded:
            QMessageBox.information(self, "Model required", "Load a model in Settings first.")
            return
        self.output.clear()
        self._out_card.setVisible(True)
        self.run_button.setEnabled(False)
        self._say("Running the forward pass and capturing tensors…", TOKENS["accent"])
        self.worker = GenerationWorker(self.capture_model, prompt_text)
        self.worker.text_received.connect(self.output.insertPlainText)
        self.worker.finished_capture.connect(self.persist_and_show)
        self.worker.failed.connect(self.run_failed)
        self.worker.start()

    def persist_and_show(self, capture: RunCapture) -> None:
        try:
            run_id = self.database.save(capture)
            self._say(f"Saved exact capture as run #{run_id}.", TOKENS["success"])
            recap = ComputationRecap(capture, run_id, self)
            self._open_recaps.append(recap)
            recap.finished.connect(
                lambda: self._open_recaps.remove(recap) if recap in self._open_recaps else None
            )
            recap.show()
            self._reload_runs()
        except Exception as exc:
            self._say("Generation completed but the capture could not be saved.", "#dc2626")
            QMessageBox.critical(self, "Database error", str(exc))
        finally:
            self.run_button.setEnabled(True)

    def run_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self._say("Run failed: an exact capture was not available.", "#dc2626")
        QMessageBox.critical(self, "Capture failed", message)



def self_test() -> None:
    """Small checks for persistence and real-array visualization helpers (no torch needed)."""
    import tempfile
    layer_data = {name: np.arange(36, dtype=np.float32).reshape(1, 6, 6) for name in REQUIRED_LAYER_TENSORS}
    # argmax lands on index 1, matching token_ids[0]; validate() rejects any other peak.
    logits = np.array([0.5, 9.25, 1.0, -3.0], dtype=np.float32)
    capture = RunCapture(prompt="test", response="answer", token_ids=[1], tokens=["answer"],
                         prompt_token_ids=[0], prompt_tokens=["test"], embedding=np.arange(8, dtype=np.float16),
                         generated_embedding=np.arange(8, dtype=np.float16), logits=logits,
                         metadata={"capture_source": CAPTURE_SOURCE, "faithful_to_upstream_eager": True,
                                   "logits_match_stock_eager": True,
                                   "final_logits_top": [{"id": 1, "token": "answer", "logit": 9.25}]},
                         layers={0: LayerCapture(0, layer_data)}, generated_layers={0: LayerCapture(0, layer_data)})
    with tempfile.TemporaryDirectory() as temporary:
        database = RunDatabase(Path(temporary) / "test.sqlite3"); run_id = database.save(capture); restored = database.load(run_id)
        assert np.array_equal(restored.embedding, capture.embedding)
        assert np.array_equal(restored.layers[0].tensors["q"], layer_data["q"])
        assert restored.layers[0].tensors["q"].dtype == layer_data["q"].dtype
        assert numeric_sample(layer_data["q"]).shape == (5, 5)
        assert display_matrix(np.ones((300, 300))).shape[0] <= 96
        # The final scores must survive exactly, and must not be mistaken for a layer: they
        # are stored at layer_index -1, and a fall-through would invent a layer -1.
        assert np.array_equal(restored.logits, logits) and restored.logits.dtype == logits.dtype
        assert -1 not in restored.layers and set(restored.layers) == {0}
        assert restored.metadata["final_logits_top"] == capture.metadata["final_logits_top"]

    # Runs saved before schema 3 carry no score vector; they must still open.
    legacy = RunCapture(prompt="p", response="r", token_ids=[1], tokens=["r"],
                        prompt_token_ids=[0], prompt_tokens=["p"],
                        embedding=np.zeros(4, dtype=np.float32),
                        generated_embedding=np.zeros(4, dtype=np.float32),
                        metadata={"capture_source": CAPTURE_SOURCE, "faithful_to_upstream_eager": True,
                                  "logits_match_stock_eager": True},
                        layers={0: LayerCapture(0, layer_data)},
                        generated_layers={0: LayerCapture(0, layer_data)})
    legacy.validate()

    # Scores that peak somewhere other than the token the capture says was generated cannot
    # both be real, so the pair must be refused rather than displayed together.
    mismatched = RunCapture(prompt="p", response="r", token_ids=[2], tokens=["r"],
                            prompt_token_ids=[0], prompt_tokens=["p"],
                            embedding=np.zeros(4, dtype=np.float32),
                            generated_embedding=np.zeros(4, dtype=np.float32), logits=logits,
                            metadata={"capture_source": CAPTURE_SOURCE, "faithful_to_upstream_eager": True,
                                      "logits_match_stock_eager": True},
                            layers={0: LayerCapture(0, layer_data)},
                            generated_layers={0: LayerCapture(0, layer_data)})
    try:
        mismatched.validate()
    except CaptureProtocolError:
        pass
    else:
        raise AssertionError("validate() accepted logits that disagree with the generated token")

    # An unverified attention implementation must be refused outright.
    unverified = RunCapture(prompt="p", response="r", token_ids=[1], tokens=["r"],
                            prompt_token_ids=[0], prompt_tokens=["p"],
                            embedding=np.zeros(4, dtype=np.float32),
                            generated_embedding=np.zeros(4, dtype=np.float32),
                            metadata={"capture_source": CAPTURE_SOURCE, "faithful_to_upstream_eager": False},
                            layers={0: LayerCapture(0, layer_data)},
                            generated_layers={0: LayerCapture(0, layer_data)})
    try:
        unverified.validate()
    except CaptureProtocolError:
        pass
    else:
        raise AssertionError("validate() accepted a capture with unverified attention")

    # Every label the recap renders must name a tensor the capture actually requires.
    labelled = {key for key, _ in TENSOR_LABELS}
    assert REQUIRED_LAYER_TENSORS <= labelled, REQUIRED_LAYER_TENSORS - labelled

    # Exact key parity in both directions is what keeps the plain-language copy from
    # drifting: a newly required tensor cannot reach the screen unexplained, and copy for a
    # tensor that no longer exists cannot linger.
    assert set(TENSOR_EXPLANATIONS) == labelled, set(TENSOR_EXPLANATIONS) ^ labelled
    assert all(text.strip() for text in TENSOR_EXPLANATIONS.values())
    assert len(LEARN_STAGE_INDEX) == len(LEARN_STAGES), "duplicate learn stage key"
    assert len(INTERNALS_STAGE_INDEX) == len(INTERNALS_STAGES), "duplicate internals stage key"

    # Both journeys interpolate real run quantities into their copy; a stage naming a field
    # learn_facts() does not supply would otherwise raise KeyError in front of a reader.
    facts = learn_facts(capture)
    for stage in LEARN_STAGES:
        stage.question.format(**facts)
        stage.plain.format(**facts)
    for stage in INTERNALS_STAGES:
        stage.heading.format(**facts)
        stage.plain.format(**facts)

    # Every layer step needs all three rungs of its disclosure ladder, or the technical
    # journey would show an equation with no purpose above it.
    for key, copy in INTERNALS_STEPS.items():
        assert {"purpose", "concept", "equation"} <= set(copy), key
        assert all(str(copy[field]).strip() for field in ("purpose", "concept", "equation")), key

    print("TensorScope self-test passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        demo_mode = "--demo" in sys.argv
        if not demo_mode:
            try:
                import torch as _torch_probe  # noqa: F401 — probe only, not used here
            except Exception:
                demo_mode = True

        db_path: Path | None = None
        if "--db" in sys.argv:
            idx = sys.argv.index("--db")
            if idx + 1 < len(sys.argv):
                db_path = Path(sys.argv[idx + 1])

        from PyQt5.QtGui import QFont
        application = QApplication(sys.argv)

        # Detect system theme from the palette before any widgets are built.
        # Direct assignment at module level updates TOKENS so every widget
        # __init__ that references TOKENS[...] picks up the right theme.
        _lum = application.palette().window().color().lightness()
        TOKENS = _TOKENS_DARK if _lum < 128 else _TOKENS_LIGHT
        _qss = _make_qss(TOKENS)
        _mpl = _make_mplstyle(TOKENS)

        import matplotlib
        matplotlib.rcParams.update(_mpl)

        application.setStyleSheet(_qss)
        application.setFont(QFont("Segoe UI", 10))

        window = TensorScopeMainWindow(
            start_in_browse=demo_mode,
            **({"db_path": db_path} if db_path else {}),
        )
        window.show()
        sys.exit(application.exec_())
