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

# Keep matplotlib's cache beside the application when a user profile is locked down.
APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(APP_DIR / ".matplotlib"))

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QThread, Qt, pyqtSignal
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

SCHEMA_VERSION = 3
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

# How many of the final scores the recap names.  The scores themselves are all kept.
FINAL_LOGITS_TOP_K = 8

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
#emptyState  {{ color: {T['text_muted']}; font-size: 13px; padding: 32px; }}

/* standard buttons */
QPushButton {{
    background: {T['card_bg']}; color: {T['text_primary']};
    border: 1px solid {T['border']}; border-radius: 6px; padding: 6px 14px;
}}
QPushButton:hover  {{ border-color: {T['text_secondary']}; }}
QPushButton:pressed {{ background: {T['nav_active_bg']}; }}
QPushButton:disabled {{ color: {T['text_muted']}; border-color: {T['border']}; }}

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
                    count: int = FINAL_LOGITS_TOP_K) -> list[dict[str, Any]]:
        """Name the highest-scoring next tokens, for display.

        Derived from the array that gets persisted -- not from the torch tensor -- so the
        table the recap draws provably describes the stored vector.  Turning an id back
        into text needs the tokenizer, which the recap has no access to, so this runs at
        capture time and travels in metadata beside prompt_tokens.
        """
        order = np.argsort(logits)[::-1][:count]
        return [{"id": int(index), "token": self.tokenizer.decode([int(index)]),
                 "logit": float(logits[index])} for index in order]

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

        eos_ids = self._eos_ids()
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
            final_logits_top = self._top_logits(capture.logits)

            # --- captured: first generated token ---
            step = self.model(
                input_ids=torch.tensor([[first_id]], device=self.device),
                past_key_values=prefill.past_key_values, use_cache=True)
            capture.generated_layers, capture.generated_embedding = self._drain()
            self._capturing = False
            self.attention.sink = None

            emit(first_id)

            # --- uncaptured: finish the answer so the reader sees a whole response ---
            past = step.past_key_values
            logits = step.logits[:, -1, :]
            for _ in range(max(0, max_new_tokens - 1)):
                token_id = int(logits.argmax(dim=-1)[0])
                if token_id in eos_ids:
                    break
                emit(token_id)
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
            "final_logits_top": final_logits_top,
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


class AttentionCanvas(FigureCanvas):
    """Matplotlib canvas displaying actual captured attention weights."""
    def __init__(self, weights: np.ndarray, original_shape: tuple[int, ...], parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(5.8, 3.1), tight_layout=True)
        super().__init__(figure)
        self.setParent(parent)
        axes = figure.add_subplot(111)
        image = axes.imshow(display_matrix(weights), aspect="auto", cmap="viridis", interpolation="nearest")
        axes.set_title(f"Attention weights (head 0); original shape {original_shape}")
        axes.set_xlabel("key position")
        axes.set_ylabel("query position")
        figure.colorbar(image, ax=axes, fraction=0.035, pad=0.02)


# Display label for every captured tensor, in the order the model computes them.
TENSOR_LABELS = [
    ("normalized_input", "X — normalized layer input"),
    ("q", "Q = XW<sub>q</sub> — projection, before RoPE"),
    ("k", "K = XW<sub>k</sub> — projection, before RoPE"),
    ("v", "V = XW<sub>v</sub> — projection"),
    ("q_attended", "Q as attended — after RoPE, per head"),
    ("k_attended", "K as attended — after RoPE, KV heads expanded"),
    ("v_attended", "V as attended — KV heads expanded"),
    ("attention_scores", "QKᵀ / √d<sub>k</sub> + causal mask — the input to softmax"),
    ("attention_weights", "softmax(scores)"),
    ("attention_output", "W<sub>o</sub> · concat(heads) — attention block output"),
    ("layer_output", "Layer output"),
]

TENSOR_LABEL_BY_KEY = dict(TENSOR_LABELS)

# Plain-language meaning of each captured tensor, for a reader who has never seen an
# attention head.  Rendered *above* the numbers: a caption underneath a matrix cannot help
# someone who does not yet know what the matrix is.
#
# Keys must match TENSOR_LABELS exactly, in both directions -- self_test() asserts it.  That
# is what stops this copy from describing a tensor the capture no longer produces, or a new
# required tensor from reaching the screen with no explanation.
#
# Two distinctions here are ones the codebase already had to correct once, so the wording is
# load-bearing: q/k/v are the raw projections *before* RoPE while q_attended/k_attended/
# v_attended are what actually enters QK^T, and attention_output is o_proj's output, not
# "attention x V".
TENSOR_EXPLANATIONS = {
    "normalized_input": (
        "Before anything else the layer rescales its input so the numbers in each row sit in "
        "a predictable range. Stacks this deep will not train without it. These values are "
        "what the attention step below actually receives."
    ),
    "q": (
        "The layer multiplies each row by a learned matrix to form a question: <i>what am I "
        "looking for?</i> This is the raw result of that multiplication. Where the token sits "
        "in the sentence has not been mixed in yet — that happens three rows down."
    ),
    "k": (
        "A second learned matrix turns each row into a key: <i>what do I have to offer?</i> A "
        "position attends to another position when its question matches that position's key. "
        "Raw projection output again, still with no position information in it."
    ),
    "v": (
        "A third learned matrix produces the value — the content a position actually hands "
        "over when something attends to it. Questions and keys decide who listens to whom; "
        "values are what gets passed along."
    ),
    "q_attended": (
        "The same Q after rotary position encoding folds in <i>where</i> the token sits, not "
        "just what it is — this is how the model can tell &quot;dog bites man&quot; from "
        "&quot;man bites dog&quot;. <b>This</b>, not the Q above, is the tensor that goes into "
        "the score calculation."
    ),
    "k_attended": (
        "K after position encoding, and after the key/value heads have been repeated to match "
        "the number of question heads — this model shares one set of keys and values across "
        "several question heads to save memory, and the sharing is expanded back out here. "
        "<b>This</b> is the K that goes into the score calculation."
    ),
    "v_attended": (
        "V after that same head expansion. V is deliberately <i>not</i> given a position "
        "encoding: position enters the computation only through Q and K."
    ),
    "attention_scores": (
        "Every question is compared against every key, giving one raw number per pair of "
        "positions: how strongly position i wants to hear from position j. The numbers are "
        "scaled down, then every position later in the sentence is set to negative infinity, "
        "because a position is not allowed to read the future. This grid is the input to "
        "softmax."
    ),
    "attention_weights": (
        "Softmax turns each row of scores into fractions that add up to 1 — the share of its "
        "attention that each position pays to every position at or before it. The picture "
        "makes the rule visible: everything above the diagonal is blank, because nothing can "
        "look ahead."
    ),
    "attention_output": (
        "What the attention block hands back: each head's blended result, joined together and "
        "passed through one final learned matrix. This is <b>not</b> &quot;attention &times; "
        "V&quot; — that product happens inside the block, and this is what comes out the far "
        "side of the projection that follows it."
    ),
    "layer_output": (
        "The layer's finished result, after a small feed-forward network has thought about "
        "each position on its own and the layer's own input has been added back on. Same shape "
        "as what came in, which is exactly what lets the model stack this block over and over. "
        "This is the next layer's input."
    ),
}


@dataclass(frozen=True)
class StoryStage:
    """One narrated step of the walkthrough that is not a single captured tensor.

    `heading` and `plain` may use the named fields returned by story_facts(); self_test()
    formats every stage so a copy edit that names an unknown field fails there rather than
    raising KeyError in front of a reader.
    """
    key: str
    heading: str
    plain: str


STORY_STAGES = [
    StoryStage(
        "prompt", "What just happened",
        "TensorScope ran the model on your prompt and kept every intermediate value it "
        "computed on the way to its answer. What follows is that computation, in the order it "
        "happened. Nothing below is a simulation, an estimate, or a reconstruction — every "
        "number was read out of the model as it ran.",
    ),
    StoryStage(
        "tokenization", "1. Your words become numbers",
        "The model cannot read text. A tokenizer first splits the prompt into tokens — whole "
        "words, word fragments, or punctuation — and looks up each one's row number in a fixed "
        "vocabulary. From here on your prompt is nothing but this list of {token_count} "
        "integers.<br><br>"
        "Some of the tokens below are ones you never typed, such as "
        "<code>&lt;|im_start|&gt;</code>. Chat models are trained on a fixed conversation "
        "layout, so TensorScope wraps your prompt in the same markers the model saw during "
        "training. Skipping them would give the model something it had never seen.",
    ),
    StoryStage(
        "embedding", "2. Each number becomes a list of numbers",
        "Every token id is used to look up one row of a large learned table. That row is the "
        "token's embedding: {hidden_size} numbers standing for what the token means. Stacked "
        "together they form a grid — one row per token in order, {hidden_size} columns of "
        "learned features. This grid is what flows into the layers.",
    ),
    StoryStage(
        "layers", "3. The same block of arithmetic runs {layer_count} times",
        "The grid now passes through one block of arithmetic, {layer_count} times over. Each "
        "pass has two halves: <b>attention</b>, where positions look at one another and trade "
        "information, and a small feed-forward network that considers each position on its "
        "own. Every pass owns its own learned weights, so no two do quite the same thing, and "
        "each one's output is the next one's input.<br><br>"
        "One pass is shown below in full, in computation order. Use the selector to narrate "
        "any of the other {layer_count} instead.<br><br>"
        "<i>Shapes are printed as they are stored, and each preview shows the leading corner "
        "of the grid — at most 5&times;5 values. The complete tensor is kept in the run; only "
        "the preview is trimmed.</i>",
    ),
    StoryStage(
        "logits", "4. Choosing a word",
        "After the last pass, the row belonging to the final position is multiplied by one "
        "more learned matrix — this one has a column for every token in the vocabulary. The "
        "result is a score for every word the model could say next, and the highest score "
        "wins.<br><br>"
        "These are the real scores from this run, and they are the ones TensorScope's "
        "strictest check covers: the same prefill was run again using the model's stock "
        "attention, and the scores had to come out identical to the last bit.",
    ),
    StoryStage(
        "coda", "5. And then it does all of that again",
        "That is one word. To carry on, the model adds the word it just chose to the end of "
        "the prompt and runs the entire stack again — all {layer_count} passes — to choose the "
        "word after that, and repeats until it decides to stop. The answer above was built one "
        "word at a time this way.<br><br>"
        "TensorScope captures the second of those passes in full as well. Switch to full "
        "detail to see it, along with all {layer_count} layers of both passes.",
    ),
]

STORY_STAGE_INDEX = {stage.key: stage for stage in STORY_STAGES}

HEATMAP_CAPTION = (
    "The picture shows head 0 only, stride-sampled down to at most 96×96 so it can be drawn. "
    "Every head's full attention grid is stored in the run at its original size — the "
    "sampling happens for display and changes nothing that was saved."
)

LOGITS_UNAVAILABLE = (
    "This run was saved before TensorScope kept the score vector, so there are no scores to "
    "show here. The word it chose is below. Run the prompt again to capture the scores."
)


def story_facts(capture: RunCapture) -> dict[str, Any]:
    """Real quantities from this run, for interpolation into the narration.

    Every value is read off the capture itself, so the prose cannot claim a layer count or
    width that this model does not have.
    """
    return {
        "layer_count": len(capture.layers),
        "token_count": len(capture.prompt_token_ids),
        "hidden_size": int(capture.embedding.shape[-1]) if capture.embedding is not None else 0,
    }


def note_label(text: str, parent: QWidget | None = None) -> QLabel:
    """A quiet caveat line -- used where display trims what it draws."""
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setStyleSheet("color:#64748b;font-size:12px;font-style:italic;border:none;")
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


class StoryCard(QFrame):
    """One narrated step: heading, then plain-language explanation, then real numbers.

    The explanation is always added before any numbers, which is the whole point of story
    mode -- the previous recap showed matrices with no statement of what they were.
    """

    def __init__(self, heading: str, plain: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("storyCard")
        # Scoped by object name: QLabel subclasses QFrame, so an unscoped QFrame rule here
        # would draw a border around every caption inside the card too.
        self.setStyleSheet(
            "#storyCard{background:#f8fafc;border:1px solid #e2e8f0;"
            "border-radius:6px;margin-top:4px;}")
        self._layout = QVBoxLayout(self)
        title = QLabel(heading)
        title.setTextFormat(Qt.RichText)
        title.setWordWrap(True)
        title.setStyleSheet("font-size:16px;font-weight:bold;")
        self._layout.addWidget(title)
        if plain:
            body = QLabel(plain)
            body.setTextFormat(Qt.RichText)
            body.setWordWrap(True)
            body.setStyleSheet("font-size:14px;color:#334155;")
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._layout.addWidget(body)

    def add(self, widget: QWidget) -> QWidget:
        """Attach a widget (numbers, a heatmap, a caveat) below the explanation."""
        self._layout.addWidget(widget)
        return widget

    def add_numbers(self, text: str) -> QLabel:
        """Attach monospaced captured values.  Escaped: prompts and token text are not HTML.

        For aligned tables and matrix previews, whose lines are short by construction.  Prose
        belongs in add_prose -- <pre> does not wrap, so one long line would force the whole
        dialog to scroll sideways.
        """
        label = QLabel(f"<pre>{html.escape(text)}</pre>")
        label.setTextFormat(Qt.RichText)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return self.add(label)

    def add_prose(self, fields: list[tuple[str, str]]) -> QLabel:
        """Attach wrapped labelled text -- a prompt, an answer, a chosen word."""
        label = QLabel("<br>".join(f"<b>{html.escape(name)}</b> {html.escape(value)}"
                                   for name, value in fields))
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return self.add(label)


def tensor_card(key: str, tensor: np.ndarray, parent: QWidget | None = None) -> StoryCard:
    """Narrate one captured tensor: its formal label, its meaning, then its real values.

    The heading comes from TENSOR_LABELS and the prose from TENSOR_EXPLANATIONS, so story
    mode and detail mode can never disagree about what a tensor is called.
    """
    card = StoryCard(TENSOR_LABEL_BY_KEY[key], TENSOR_EXPLANATIONS[key], parent)
    card.add_numbers(f"shape {tuple(tensor.shape)}\n{sample_text(tensor)}")
    return card


def plain_section(title: str, body: str) -> QFrame:
    """The original recap's section block, unchanged, still used by full-detail mode."""
    box = QFrame(); box.setObjectName("card"); box.setFrameShape(QFrame.StyledPanel)
    layout = QVBoxLayout(box)
    label = QLabel(title); label.setStyleSheet("font-size:15px;font-weight:bold;"); layout.addWidget(label)
    text = QLabel(body); text.setTextInteractionFlags(Qt.TextSelectableByMouse); text.setWordWrap(True); layout.addWidget(text)
    return box


class LayerSection(QFrame):
    """One collapsible transformer layer.  Contents are built on first expand so a
    36-layer recap opens immediately instead of rendering 70+ heatmaps up front.

    `explain` adds the plain-language captions and moves the heatmap next to the weights it
    draws; it defaults off so full-detail mode renders exactly what it always has.
    `expanded` builds the body during construction, for the one layer story mode narrates.
    """

    def __init__(self, index: int, layer: LayerCapture, parent: QWidget | None = None,
                 explain: bool = False, expanded: bool = False) -> None:
        super().__init__(parent)
        self.layer = layer
        self.explain = explain
        self.setObjectName("card")
        self.setFrameShape(QFrame.StyledPanel)
        self._layout = QVBoxLayout(self)
        self.toggle = QPushButton(f"▶  Layer {index}")
        self.toggle.setStyleSheet("font-size:15px;font-weight:bold;text-align:left;padding:6px;")
        self.toggle.clicked.connect(self._toggle)
        self._layout.addWidget(self.toggle)
        self.body: QWidget | None = None
        if expanded:
            self._toggle()

    def _toggle(self) -> None:
        if self.body is None:
            self.body = QWidget()
            body_layout = QVBoxLayout(self.body)
            for key, label in TENSOR_LABELS:
                tensor = self.layer.tensors.get(key)
                if tensor is None:
                    continue
                if self.explain:
                    body_layout.addWidget(tensor_card(key, tensor, self.body))
                    if key == "attention_weights":
                        # Next to the weights it draws, rather than at the end of the layer.
                        body_layout.addWidget(AttentionCanvas(tensor, tensor.shape, self.body))
                        body_layout.addWidget(note_label(HEATMAP_CAPTION, self.body))
                    continue
                text = QLabel(f"<b>{label}</b> — shape {tensor.shape}<br><pre>{sample_text(tensor)}</pre>")
                text.setTextFormat(Qt.RichText)
                text.setWordWrap(True)
                text.setTextInteractionFlags(Qt.TextSelectableByMouse)
                body_layout.addWidget(text)
            if not self.explain:
                weights = self.layer.tensors["attention_weights"]
                body_layout.addWidget(AttentionCanvas(weights, weights.shape, self.body))
            self._layout.addWidget(self.body)
        visible = not self.body.isVisible()
        self.body.setVisible(visible)
        self.toggle.setText(f"{'▼' if visible else '▶'}  Layer {self.layer.index}")


class StoryView(QWidget):
    """The guided walkthrough: one stage at a time, driven by the left navigation panel.

    Scope is the prefill pass.  The closing stage explains that the generated-token pass
    re-runs the same stack, and points at full detail for it.
    """

    def __init__(self, capture: RunCapture, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.capture = capture
        self.facts = story_facts(capture)
        self.layer_section: LayerSection | None = None
        self._stage_widget: QWidget | None = None
        self._stage_key: str = ""
        self._on_stage_change: Callable[[str], None] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._stage_layout = QVBoxLayout()
        outer.addLayout(self._stage_layout)
        outer.addStretch(1)
        self.go_to("prompt")

    def go_to(self, key: str) -> None:
        """Replace the visible stage with the one identified by key."""
        if self._stage_widget is not None:
            self._stage_widget.setParent(None)
            self._stage_widget.deleteLater()
            self._stage_widget = None
            self.layer_section = None  # was a child of the old stage widget
        widget = self._build_stage(key)
        self._stage_key = key
        self._stage_widget = widget
        self._stage_layout.addWidget(widget)
        if self._on_stage_change:
            self._on_stage_change(key)

    def _build_stage(self, key: str) -> QWidget:
        if key == "prompt":
            return self._opening()
        if key == "tokenization":
            return self._tokenization()
        if key == "embedding":
            return self._embedding()
        if key == "layers":
            return self._layers_stage()
        if key == "logits":
            return self._word_choice()
        if key == "coda":
            return self._card("coda")
        return self._card(key)

    def _layers_stage(self) -> QWidget:
        """A container holding the layers narrative card, layer picker, and LayerSection."""
        container = QWidget()
        layout = QVBoxLayout(container); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._card("layers"))
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Narrate:"))
        self.picker = QComboBox()
        for index in sorted(self.capture.layers):
            self.picker.addItem(f"Layer {index}", index)
        self.picker.currentIndexChanged.connect(self._layer_changed)
        picker_row.addWidget(self.picker)
        picker_row.addStretch(1)
        layout.addLayout(picker_row)
        self.layer_host = QVBoxLayout()
        layout.addLayout(self.layer_host)
        self._show_layer(sorted(self.capture.layers)[0])
        return container

    def _card(self, key: str) -> StoryCard:
        stage = STORY_STAGE_INDEX[key]
        return StoryCard(stage.heading.format(**self.facts),
                         stage.plain.format(**self.facts), self)

    def _opening(self) -> StoryCard:
        card = self._card("prompt")
        card.add_prose([("You asked:", self.capture.prompt.strip()),
                        ("The model answered:", self.capture.response.strip())])
        return card

    def _tokenization(self) -> StoryCard:
        card = self._card("tokenization")
        rows = "\n".join(
            f"{position:>3}  {token_id:>7}  {token}" for position, (token_id, token)
            in enumerate(zip(self.capture.prompt_token_ids, self.capture.prompt_tokens)))
        card.add_numbers(f"{'pos':>3}  {'id':>7}  token\n{rows}")
        return card

    def _embedding(self) -> StoryCard:
        card = self._card("embedding")
        card.add_numbers(f"shape {tuple(self.capture.embedding.shape)}\n"
                         f"{sample_text(self.capture.embedding)}")
        return card

    def _word_choice(self) -> StoryCard:
        """The final stage.  Scores come from the persisted vector, or say so if absent."""
        card = self._card("logits")
        top = self.capture.metadata.get("final_logits_top") or []
        if self.capture.logits is not None and top:
            rows = "\n".join(
                f"{rank:>4}  {entry['logit']:>11.4f}  {entry['id']:>7}  {entry['token']!r}"
                for rank, entry in enumerate(top, start=1))
            card.add_numbers(
                f"Scores for all {self.capture.logits.shape[-1]:,} possible next tokens; "
                f"the highest {len(top)}:\n\n"
                f"{'rank':>4}  {'score':>11}  {'id':>7}  token\n{rows}")
        else:
            card.add(note_label(LOGITS_UNAVAILABLE, card))
        if self.capture.token_ids:
            card.add_prose([("The model chose:",
                             f"{self.capture.tokens[0]!r}  (token id {self.capture.token_ids[0]})")])
        return card

    def _show_layer(self, index: int) -> None:
        if self.layer_section is not None:
            self.layer_section.setParent(None)
            self.layer_section.deleteLater()
        self.layer_section = LayerSection(index, self.capture.layers[index], self,
                                          explain=True, expanded=True)
        self.layer_host.addWidget(self.layer_section)

    def _layer_changed(self, position: int) -> None:
        index = self.picker.itemData(position)
        if index is not None:
            self._show_layer(int(index))


class DetailView(QWidget):
    """Every captured tensor of both forward passes.

    This is the recap TensorScope has always shown: LayerSection with `explain` off renders
    exactly as before, and all layers stay collapsed until clicked.
    """

    def __init__(self, capture: RunCapture, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(plain_section(
            "1. Prompt tokenization",
            "Token IDs: " + str(capture.prompt_token_ids) + "\nTokens: " + repr(capture.prompt_tokens)))
        layout.addWidget(plain_section(
            "2. Prompt-prefill embedding output",
            f"Shape: {capture.embedding.shape}\nFirst values:\n{sample_text(capture.embedding)}"))
        layout.addWidget(plain_section(
            "3. First generated-token embedding output",
            f"Shape: {capture.generated_embedding.shape}\nFirst values:\n{sample_text(capture.generated_embedding)}"))
        layout.addWidget(plain_section(
            "4. Forward-pass equations",
            "Q = XWq    K = XWk    V = XWv        (projections, shown before RoPE)\n"
            "Attention = softmax(QKᵀ / √dₖ + mask)  (using the post-RoPE Q and K)\n"
            "Attention block output = Wo · concat(heads)\n"
            "    -- o_proj applied after the heads are joined.  Not 'Attention × V':\n"
            "       that product happens inside the block, before this projection.\n\n"
            "Every value below was read out of the model as it computed these steps."))

        for phase_title, phase_layers in (("Prompt prefill forward pass", capture.layers),
                                          ("First generated-token forward pass", capture.generated_layers)):
            phase_heading = QLabel(phase_title)
            phase_heading.setStyleSheet("font-size:18px;font-weight:bold;margin-top:12px;")
            layout.addWidget(phase_heading)
            hint = QLabel("Click a layer to expand its captured tensors.")
            hint.setStyleSheet("color:#475569;"); layout.addWidget(hint)
            for index in sorted(phase_layers):
                layout.addWidget(LayerSection(index, phase_layers[index], self))
        layout.addStretch(1)


class ComputationRecap(QDialog):
    """Read-only view of one persisted or just-captured real forward pass, in two modes.

    A left navigation panel drives stage-by-stage story mode; full detail shows the
    exhaustive layer list on demand.  Banner is shared by both.
    """
    def __init__(self, capture: RunCapture, run_id: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.capture = capture
        self.setWindowTitle("TensorScope — Computation Recap")
        self.resize(1100, 800)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        banner_text = (
            f"SAVED RUN #{run_id} — real captured data, loaded from disk"
            if run_id is not None else
            "CAPTURED FROM THE RUNNING MODEL — no tensors are inferred or simulated"
        )
        banner = QLabel(banner_text)
        banner.setStyleSheet(
            f"background:{TOKENS['success']};color:white;font-weight:600;"
            "padding:9px;border-radius:0px;margin:0px;"
        )
        root.addWidget(banner)

        # ── Splitter: left nav  |  right content ─────────────────────────────────
        body = QSplitter(Qt.Horizontal)
        root.addWidget(body)

        # Left panel ──────────────────────────────────────────────────────────────
        left = QWidget()
        left.setObjectName("sidebar")
        left.setMinimumWidth(160); left.setMaximumWidth(240)
        ll = QVBoxLayout(left); ll.setContentsMargins(10, 10, 10, 10); ll.setSpacing(4)

        run_lbl = QLabel(f"Run #{run_id}" if run_id else "Current run")
        run_lbl.setStyleSheet("font-size:14px;font-weight:600;")
        ll.addWidget(run_lbl)

        model_lbl = QLabel(str(capture.metadata.get("model", "unknown")))
        model_lbl.setWordWrap(True)
        model_lbl.setStyleSheet(f"font-size:11px;color:{TOKENS['text_muted']};")
        ll.addWidget(model_lbl)

        def _sep() -> QFrame:
            f = QFrame(); f.setObjectName("rowDivider"); return f

        ll.addWidget(_sep())

        self.story_button = QPushButton("Guided walkthrough")
        self.story_button.setObjectName("navItem"); self.story_button.setCheckable(True)
        self.detail_button = QPushButton(f"Full detail  ({len(capture.layers)} layers)")
        self.detail_button.setObjectName("navItem"); self.detail_button.setCheckable(True)
        ll.addWidget(self.story_button); ll.addWidget(self.detail_button)

        ll.addWidget(_sep())

        self._nav_container = QWidget()
        nl = QVBoxLayout(self._nav_container); nl.setContentsMargins(0, 0, 0, 0); nl.setSpacing(2)
        self._nav_buttons: dict[str, QPushButton] = {}
        for key, label in [("prompt", "Overview"), ("tokenization", "Tokenization"),
                            ("embedding", "Embedding"), ("layers", "Layers"),
                            ("logits", "Word Choice"), ("coda", "Coda")]:
            btn = QPushButton(label); btn.setObjectName("navItem"); btn.setCheckable(True)
            btn.clicked.connect(lambda _, k=key: self._nav_to(k))
            nl.addWidget(btn); self._nav_buttons[key] = btn
        ll.addWidget(self._nav_container)
        ll.addStretch(1)
        body.addWidget(left)

        # Right panel ─────────────────────────────────────────────────────────────
        right_scroll = QScrollArea(); right_scroll.setWidgetResizable(True)
        rc = QWidget(); rl = QVBoxLayout(rc); right_scroll.setWidget(rc)

        info = QFormLayout()
        info.addRow("Model", QLabel(str(capture.metadata.get("model", "unknown"))))
        info.addRow("Backend / GPU", QLabel(str(capture.metadata.get("backend", "unknown"))))
        info.addRow("Weight dtype", QLabel(str(capture.metadata.get("weight_dtype", "unknown"))))
        info.addRow("Capture scope", QLabel("prompt prefill + first generated token"))
        info.addRow("Captured layers", QLabel(str(len(capture.layers))))
        info.addRow("Attention verified", QLabel(
            "bitwise identical to upstream eager_attention_forward"
            if capture.metadata.get("faithful_to_upstream_eager") else "NOT VERIFIED"))
        rl.addLayout(info)

        self.story_view = StoryView(capture, rc)
        self.story_view._on_stage_change = self._on_story_stage_change
        self.stack = QStackedWidget()
        self.stack.addWidget(self.story_view)
        self.detail: DetailView | None = None
        rl.addWidget(self.stack)
        rl.addStretch(1)
        body.addWidget(right_scroll)

        body.setStretchFactor(0, 0); body.setStretchFactor(1, 1)
        body.setSizes([200, 900])

        self.story_button.clicked.connect(self._show_story)
        self.detail_button.clicked.connect(self._show_detail)
        self._show_story()

    def _nav_to(self, key: str) -> None:
        self._show_story()
        self.story_view.go_to(key)

    def _on_story_stage_change(self, key: str) -> None:
        for k, btn in self._nav_buttons.items():
            btn.setChecked(k == key)

    def _show_story(self) -> None:
        self.stack.setCurrentIndex(0)
        self.story_button.setChecked(True)
        self.detail_button.setChecked(False)
        self._nav_container.setVisible(True)

    def _show_detail(self) -> None:
        if self.detail is None:
            self.detail = DetailView(self.capture, self.stack)
            self.stack.addWidget(self.detail)
        self.stack.setCurrentWidget(self.detail)
        self.story_button.setChecked(False)
        self.detail_button.setChecked(True)
        self._nav_container.setVisible(False)


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

        self._nav_btns: list[QPushButton] = []
        for label, idx in [
            ("▶  Live Capture", 0),
            ("≡  Saved Runs",   1),
            ("⚙  Settings",     2),
            ("·  About",        3),
        ]:
            btn = QPushButton(label)
            btn.setObjectName("sidebarNav")
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

        def _screen(title: str, subtitle: str) -> tuple:
            scroll = QScrollArea(); scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            inner = QWidget(); scroll.setWidget(inner)
            l = QVBoxLayout(inner)
            l.setContentsMargins(32, 28, 32, 28); l.setSpacing(14)
            l.setAlignment(Qt.AlignTop)
            hdr = QWidget(); hl = QVBoxLayout(hdr)
            hl.setContentsMargins(0, 0, 0, 4); hl.setSpacing(2)
            t = QLabel(title); t.setObjectName("screenTitle"); hl.addWidget(t)
            if subtitle:
                s = QLabel(subtitle); s.setObjectName("screenSubtitle"); hl.addWidget(s)
            l.addWidget(hdr)
            return scroll, l

        # Screen 0: Live Capture ──────────────────────────────────────────────
        live_scr, live_l = _screen(
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
        runs_scr, runs_l = _screen("Saved Runs", "Browse captures without loading a model.")

        self._browse_inner = QStackedWidget()

        runs_list_w = QWidget()
        rll = QVBoxLayout(runs_list_w)
        rll.setContentsMargins(0, 0, 0, 0); rll.setSpacing(8)
        self.runs_list = QListWidget(); rll.addWidget(self.runs_list)
        browse_btns_row = QHBoxLayout()
        self.open_recap_btn = QPushButton("Open Recap"); self.open_recap_btn.setEnabled(False)
        _refresh_btn = QPushButton("Refresh")
        browse_btns_row.addWidget(self.open_recap_btn); browse_btns_row.addWidget(_refresh_btn)
        browse_btns_row.addStretch(1); rll.addLayout(browse_btns_row)
        self._browse_inner.addWidget(runs_list_w)  # index 0

        self._empty_label = QLabel(
            "No saved runs yet.\n\n"
            "Switch to Live Capture on a machine with a GPU,\n"
            "run a prompt, then come back here to browse it.\n\n"
            "Or pass --db /path/to/runs.sqlite3 to point at\n"
            "an existing database."
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setObjectName("emptyState")
        self._browse_inner.addWidget(self._empty_label)  # index 1

        runs_l.addWidget(self._browse_inner)
        self.browse_status = QLabel(); self.browse_status.setObjectName("statusLabel")
        runs_l.addWidget(self.browse_status)
        self.screens.addWidget(runs_scr)  # index 1

        # Screen 2: Settings ──────────────────────────────────────────────────
        settings_scr, settings_l = _screen("Settings", "")

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
        db_val = QLabel(str(self.database.path))
        db_val.setObjectName("rowValue"); db_val.setWordWrap(True)
        db_val.setTextInteractionFlags(Qt.TextSelectableByMouse)
        db_row.addWidget(db_val, 1); db_cl.addLayout(db_row)
        settings_l.addWidget(db_card)
        self.screens.addWidget(settings_scr)  # index 2

        # Screen 3: About ─────────────────────────────────────────────────────
        about_scr, about_l = _screen("About", "")

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

    # (stub to satisfy old code paths — not a separate mode any more)
    def _set_mode(self, _mode: str) -> None:
        pass

        self.content_stack = QStackedWidget()
        layout.addWidget(self.content_stack)

        # ── Live panel (index 0) ──────────────────────────────────────────────
        live_panel = QWidget()
        live_layout = QVBoxLayout(live_panel)
        live_layout.setContentsMargins(0, 4, 0, 0)

        settings = QHBoxLayout()
        settings.addWidget(QLabel("Model:"))
        self.model_id = QLineEdit(self.capture_model.model_id)
        settings.addWidget(self.model_id)
        self.load_button = QPushButton("Load model")
        settings.addWidget(self.load_button)
        live_layout.addLayout(settings)

        live_layout.addWidget(QLabel("Prompt"))
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Ask the model something...")
        self.prompt.setFixedHeight(125)
        live_layout.addWidget(self.prompt)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Model")
        self.run_button.setEnabled(False)
        run_row.addWidget(self.run_button)
        run_row.addStretch(1)
        live_layout.addLayout(run_row)

        live_layout.addWidget(QLabel("Model answer"))
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        live_layout.addWidget(self.output)

        self.status = QLabel("Load a model to begin.")
        self.status.setStyleSheet(f"color:{C_TEXT2};")
        live_layout.addWidget(self.status)

        self.content_stack.addWidget(live_panel)

        # ── Browse panel (index 1) ────────────────────────────────────────────
        browse_panel = QWidget()
        browse_layout = QVBoxLayout(browse_panel)
        browse_layout.setContentsMargins(0, 4, 0, 0)

        self._browse_inner = QStackedWidget()

        self.runs_list = QListWidget()
        self._browse_inner.addWidget(self.runs_list)  # index 0 — populated list

        self._empty_label = QLabel(
            "No saved runs yet.\n\n"
            "Switch to Live Mode on a machine with a GPU, run a prompt,\n"
            "and the capture will appear here.\n\n"
            "Or point TensorScope at an existing database:\n"
            "  python TensorScope.py --db /path/to/runs.sqlite3"
        )
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(f"color:{C_MUTED};font-size:13px;padding:32px;")
        self._browse_inner.addWidget(self._empty_label)  # index 1 — empty state

        browse_layout.addWidget(self._browse_inner)

        browse_buttons = QHBoxLayout()
        self.open_recap_btn = QPushButton("Open Recap")
        self.open_recap_btn.setEnabled(False)
        _refresh_btn = QPushButton("Refresh")
        browse_buttons.addWidget(self.open_recap_btn)
        browse_buttons.addWidget(_refresh_btn)
        browse_buttons.addStretch(1)
        browse_layout.addLayout(browse_buttons)

        self.browse_status = QLabel()
        self.browse_status.setStyleSheet(f"color:{C_MUTED};font-size:12px;")
        browse_layout.addWidget(self.browse_status)

        self.content_stack.addWidget(browse_panel)

        # Signals
        self.run_button.clicked.connect(self.run_model)
        self.load_button.clicked.connect(self.load_model)
        self._live_btn.clicked.connect(lambda: self._set_mode("live"))
        self._browse_btn.clicked.connect(lambda: self._set_mode("browse"))
        self.open_recap_btn.clicked.connect(self._open_selected_run)
        _refresh_btn.clicked.connect(self._reload_runs)
        self.runs_list.itemDoubleClicked.connect(lambda _: self._open_selected_run())
        self.runs_list.currentItemChanged.connect(
            lambda item, _: self.open_recap_btn.setEnabled(item is not None)
        )

        self._reload_runs()
        self._set_mode("browse" if start_in_browse else "live")

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        is_live = mode == "live"
        self.content_stack.setCurrentIndex(0 if is_live else 1)
        self._live_btn.setChecked(is_live)
        self._browse_btn.setChecked(not is_live)

    # ── Browse helpers ────────────────────────────────────────────────────────

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
    assert len(STORY_STAGE_INDEX) == len(STORY_STAGES), "duplicate story stage key"

    # Narration interpolates real run quantities; a stage naming a field story_facts() does
    # not supply would otherwise raise KeyError in front of a reader.
    facts = story_facts(capture)
    for stage in STORY_STAGES:
        stage.heading.format(**facts)
        stage.plain.format(**facts)

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
