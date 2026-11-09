"""Shared read-only presentation primitives for the saved-capture UI.

Nothing here loads a model, writes a run, or mutates a captured array.  Every widget
either renders copy from `tensorscope_content` or reads a numpy array that the capture
owns.

This module exists to break an import cycle rather than for tidiness: the stage and
colour constants below are read both by `ComputationRecap` in `tensorscope_ui` and by
the views in `tensorscope_views`, so leaving them in `tensorscope_ui` would make the
views import it while it is still importing them.
"""
from __future__ import annotations

import html
import math
import os
from contextlib import contextmanager

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import (
    QCoreApplication, QEvent, QPoint, QSize, Qt, QTimer, QVariantAnimation, pyqtSignal,
)
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QActionGroup, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QMenu, QPushButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from tensor_widgets import exact_scalar
from tensorscope_content import (
    EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED, EVIDENCE_KINDS, EVIDENCE_OBSERVED,
    EVIDENCE_UNATTESTED, NOTEBOOK_TEXT, ROLE_LEGEND, STRUCTURE_NOT_TIMING_NOTE,
    readable_spelling,
)


# Qt's offscreen platform has no compositor, so an animation there would only burn time
# in the test suite.  Reveals are never gated on an animation finishing, so skipping it
# changes nothing a reader or a test can observe except the tint.
OFFSCREEN = os.environ.get('QT_QPA_PLATFORM', '') == 'offscreen'


COLORS = {'q': '#60a5fa', 'k': '#f472b6', 'v': '#34d399',
          'scores': '#fbbf24', 'weights': '#22d3ee', 'output': '#fb923c',
          'norm': '#a78bfa', 'layer': '#94a3b8'}

# The same roles, one step darker, for the light palette.  The dark-palette hues sit
# between 1.6:1 and 2.6:1 against white, which is too faint for a label drawn in them;
# these clear 3:1.  K is pink rather than purple in both, because blue and purple collapse
# into one hue under deuteranopia (OKLab delta-E 1.3) and Q and K are drawn side by side
# in the comparison picture.  Blue against pink stays apart (delta-E 10.9 dark, 18.7
# light).  Every coloured element is labelled in words as well, so the hue is never the
# only copy of a meaning.
COLORS_LIGHT = {'q': '#2563eb', 'k': '#db2777', 'v': '#059669',
                'scores': '#b45309', 'weights': '#0e7490', 'output': '#c2410c',
                'norm': '#7c3aed', 'layer': '#64748b'}


def is_light(tokens: dict | None) -> bool:
    """Whether a token set describes the light palette, judged from its page colour."""
    background = (tokens or {}).get('bg')
    if not background:
        return False
    colour = QColor(background)
    return colour.isValid() and colour.lightness() >= 128


def role_colors(tokens: dict | None = None) -> dict:
    """The role hues for the palette `tokens` describes.  Same keys in both."""
    return COLORS_LIGHT if is_light(tokens) else COLORS


# The reading column.  Prose wider than this runs past 110 characters a line at the body
# size, and a diagram stretched across a 1,366-pixel window turns a 15 x 15 grid into
# bars.  Pages centre their sections in a column of this width and let the margins grow.
COLUMN_WIDTH = 940
PROSE_WIDTH = 760

# One hue per evidence tier, chosen to stay legible on both light and dark palettes
# (the tint is an alpha wash over whatever the card background happens to be).
EVIDENCE_COLORS = {
    EVIDENCE_OBSERVED:    ('#0e9f6e', 'rgba(14, 159, 110, 0.14)'),
    EVIDENCE_UNATTESTED:  ('#b45309', 'rgba(180, 83, 9, 0.14)'),
    EVIDENCE_DERIVED:     ('#2563eb', 'rgba(37, 99, 235, 0.14)'),
    EVIDENCE_CONCEPTUAL:  ('#6b7280', 'rgba(107, 114, 128, 0.14)'),
}

# Colours for lesson furniture: reveal highlights, progress pills, quiz feedback, the
# token the reader has selected.  Deliberately *not* EVIDENCE_COLORS.  Those four hues
# answer "where did this number come from", and a reader who has learned them from
# EvidenceBadge would otherwise see the same green mean "correct answer" and the same
# blue mean "this row just appeared" -- three unrelated meanings for one palette.  One
# neutral accent carries all the chrome instead, so no decoration reads as provenance.
CHROME = {
    'accent': '#7c7cf0',
    'accent_tint': 'rgba(124, 124, 240, 0.16)',
    'ahead': '#3b3b44',
    'muted': '#8888aa',
}


# ── Label helpers ────────────────────────────────────────────────────────────

def clear_layout(layout) -> None:
    """Empty a layout and take its widgets off screen *now*.

    `takeAt` only removes the layout item: the widget keeps its parent, its geometry and
    its visibility, and `deleteLater` does not change that until the event loop gets round
    to the deferred delete.  In between, the old widgets are still painted over the new
    ones, still found by `findChildren`, and still clickable -- and clicking a stale
    navigator chip calls `go_to` with a key the current page does not have, which raises
    inside a slot, which makes PyQt abort the process with no traceback at all.

    `setParent(None)` is what actually retires them, so this helper exists rather than the
    `takeAt`/`deleteLater` pair that is easy to write and subtly wrong.
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget() if item is not None else None
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()


def label(text, *, rich=False, muted=False, title=False, small=False):
    result = QLabel(text)
    result.setTextFormat(Qt.RichText if rich else Qt.PlainText)
    result.setWordWrap(True)
    result.setTextInteractionFlags(Qt.TextSelectableByMouse)
    name = 'journeyTitle' if title else 'journeyMuted' if muted else 'journeySmall' if small else 'journeyText'
    result.setObjectName(name)
    return result


class MeasureLabel(QLabel):
    """A word-wrapped label held to a reading measure.

    `QLabel.heightForWidth` ignores the label's own maximum width, so a layout that offers
    a capped label 900 pixels is told the height of 900-pixel lines, then lays the label
    out at its 760-pixel cap -- more lines than the height it reserved -- and the page
    squeezes something else to make room.  Measuring at the width the label will really
    have keeps the reservation honest.
    """

    def heightForWidth(self, width: int) -> int:            # noqa: N802 - Qt name
        return super().heightForWidth(min(int(width), self.maximumWidth()))


def measure_label(text: str, *, width: int = 0, rich: bool = False) -> QLabel:
    """`label` for prose, capped at `width` (default: the page's prose measure)."""
    result = MeasureLabel(text)
    result.setTextFormat(Qt.RichText if rich else Qt.PlainText)
    result.setWordWrap(True)
    result.setTextInteractionFlags(Qt.TextSelectableByMouse)
    result.setObjectName('journeyText')
    result.setMaximumWidth(int(width or PROSE_WIDTH))
    return result


def formula_block(text: str) -> QLabel:
    """A formula on its own lines, in the page's mathematical face.

    The copy separates the equations of one formula with runs of spaces
    ("S = s·(Q Kᵀ) + M      A = softmax(S)").  Rendered as rich text those runs collapse
    to one space and the equations run together into one false equation, so each becomes
    its own line here, as plain text.
    """
    import re
    lines = [part.strip() for part in re.split(r' {4,}', str(text)) if part.strip()]
    result = QLabel('\n'.join(lines))
    result.setTextFormat(Qt.PlainText)
    result.setWordWrap(True)
    result.setTextInteractionFlags(Qt.TextSelectableByMouse)
    result.setObjectName('formulaBlock')
    result.setAccessibleName('Formula: ' + '; '.join(lines))
    return result


def part_label(text: str) -> QLabel:
    """A small heading for one part of a section -- 'The operation', 'The symbols'."""
    result = label(text.upper(), small=True)
    result.setObjectName('partLabel')
    return result


def shape(array):
    return '[' + ' × '.join(str(d) for d in array.shape) + ']'


def shape_line(array, *, omit_batch: bool = False) -> tuple[str, str]:
    """A shape to read, and the sentence that keeps it honest.

    Returns `(written, caption)`.  Every stored activation here carries a leading batch
    axis of 1, which is noise in an equation a person is trying to follow -- `[15 × 2560]`
    reads, `[1 × 15 × 2560]` does not.  Dropping it silently would be a different shape
    from the one on disk, so the caption always says so and always names the stored shape.
    One helper, so no surface can drop the axis without also saying it did.

    The axis is only ever dropped when it really is 1: a batched capture would print in
    full, caption-free.
    """
    dims = tuple(int(d) for d in array.shape)
    full = '[' + ' × '.join(f'{d:,}' for d in dims) + ']'
    if not (omit_batch and len(dims) > 1 and dims[0] == 1):
        return full, ''
    short = '[' + ' × '.join(f'{d:,}' for d in dims[1:]) + ']'
    return short, (f'Batch axis of size 1 omitted for readability. '
                   f'The stored array is {full}.')


def chip_text(token: str, limit: int = 18) -> str:
    """A token's spelling, safe to put on a button.

    Tokens are byte-level BPE fragments: they can be a single space, a newline, or a
    long run of whitespace.  Rendered raw they collapse to an empty-looking chip or
    stretch the strip, so show the quoted form and elide it.
    """
    text = repr(token)[1:-1] or '∅'          # strip repr's quotes, keep its escapes
    if not text.strip():
        text = '␠' * min(len(text), limit)   # visible stand-in for pure whitespace
    return text if len(text) <= limit else text[:limit - 1] + '…'


# ── Rounded numbers for reading ──────────────────────────────────────────────
# The teaching view shows a value the way a person reads it -- three significant figures,
# marked with ≈ -- and keeps the exact stored digits one hover away.  These return text and
# nothing else: the array a value came from is never touched, and the variable panel and
# the full inspector go on showing `exact_scalar` digits.

def approx(value, digits: int = 3) -> str:
    """'≈0.252' for 0.251953125.  Integers and non-finite values are shown as they are."""
    if isinstance(value, (int, np.integer)):
        return f'{int(value):,}'
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    if number == 0:
        return '0'
    if abs(number) >= 1e6:
        # A masked score is a large negative additive constant (about -3.4e38 in bf16);
        # written out in full it is a 39-digit string that pushes its box off the panel.
        return f'≈{number:.{digits}g}'
    if abs(number) >= 1000:
        return f'≈{number:,.0f}'
    return f'≈{number:.{digits}g}'


def approx_share(weight) -> str:
    """An attention weight as a share of one head's mixing weight -- never of anything else.

    Only attention weights may be read as percentages: they are one head's softmax output
    for one query, so they sum to 1 by construction.  Logits are scores and must not pass
    through here.
    """
    return NOTEBOOK_TEXT['weight_share'].format(percent=f'{float(weight) * 100:.1f}%')


def token_labels(capture, generated=False, keys=False):
    """Preserve stored token spellings, and show absolute sequence positions."""
    prompt = [f'{i}: {t}' for i, t in enumerate(capture.prompt_tokens)]
    if not generated:
        return prompt
    first = f'{len(prompt)}: {capture.tokens[0]!r}'
    return prompt + [first] if keys else [first]


def axes_for(name, tensor):
    if name == 'final_logits':
        return ['vocabulary token ID']
    if tensor.ndim == 4:
        return ['batch', 'attention head', 'query token', 'key token'] if name in (
            'attention_scores', 'attention_weights') else [
                'batch', 'attention head', 'key token' if name.startswith(('k_', 'v_')) else 'query token',
                'feature within head']
    if tensor.ndim == 3:
        return ['batch', 'token', 'projection feature' if name in ('q', 'k', 'v') else 'hidden feature']
    return [f'axis {i}' for i in range(tensor.ndim)]


# ── Captured-array helpers ────────────────────────────────────────────────────
# Read-only views over stored arrays, plus the one layout swap every selection-driven
# panel uses.  These sit here rather than in a view because the notebook, the reference
# page and the raw inspector all need them and none of those may import another.

def replace_in(layout, widget) -> None:
    """Swap the single widget held by `layout`, disposing of the previous one.

    Used by the panels that update on a token or layer selection.  Rebuilding a panel
    rather than mutating labels in place keeps every builder a pure function of the
    current selection, which is what makes them safe to call from the tests.
    """
    clear_layout(layout)
    if widget is not None:
        layout.addWidget(widget)


def bullets(items) -> QWidget:
    """A bulleted list as one selectable rich label."""
    body = '<br>'.join(f'•&nbsp;&nbsp;{text}' for text in items)
    return label(body, rich=True)


def head_slice(array, head: int = 0) -> np.ndarray:
    """A 2-D [query x key] *view* of a captured attention tensor.

    Basic indexing only, so this borrows the stored buffer and can never write to it.
    """
    view = np.asarray(array)
    if view.ndim == 4:
        view = view[0, min(head, view.shape[1] - 1)]
    elif view.ndim == 3:
        view = view[min(head, view.shape[0] - 1)]
    while view.ndim > 2:
        view = view[0]
    return view[np.newaxis, :] if view.ndim == 1 else view


def embedding_row(array, index: int):
    """One token's embedding vector, or None when this capture cannot supply it.

    Returns None rather than raising: a run whose stored embedding is not
    [batch, token, feature] should degrade to a note, not take the stage down.
    """
    view = np.asarray(array)
    if view.ndim < 2:
        return None
    while view.ndim > 2:
        view = view[0]
    return view[index] if 0 <= index < view.shape[0] else None


def head_count(array) -> int:
    view = np.asarray(array)
    return int(view.shape[-3]) if view.ndim >= 3 else 1


# ── Evidence badge ────────────────────────────────────────────────────────────

class EvidenceBadge(QLabel):
    """A small marker saying where a number on screen came from.

    Four tiers, not three: real forward-pass output that no verification gate covers
    (candidate scores after the first generated token) must not look identical to a
    tensor that survives all three gates.  See EVIDENCE_KINDS.
    """

    def __init__(self, kind: str, note: str = '', parent=None):
        super().__init__(parent)
        name, description = EVIDENCE_KINDS[kind]
        colour, tint = EVIDENCE_COLORS[kind]
        self.setObjectName('evidenceBadge')
        self.setText(name + (f'  ·  {note}' if note else ''))
        self.setToolTip(description)
        self.setStyleSheet(
            f'background: {tint}; color: {colour}; border: 1px solid {colour}; '
            'border-radius: 9px; padding: 2px 9px; font-size: 11px; font-weight: 600;')
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)


class RoleLegend(QWidget):
    """What the semantic colours mean, once, for a whole page.

    Every coloured element on a page already carries its own text label, so this is
    redundancy rather than a key the reader must consult.  It exists because the hues are
    *stable* across the page -- Q is the same blue in operation 5 and in operation 7 --
    and a reader who notices that deserves to be told what the stability means.

    Reads ROLE_LEGEND for the names and COLORS for the hues, so it cannot disagree with
    the diagrams or the variable chips about what colour a role is.
    """

    def __init__(self, roles=ROLE_LEGEND, tokens=None, parent=None):
        super().__init__(parent)
        self.setObjectName('roleLegend')
        row = QHBoxLayout(self)
        row.setContentsMargins(22, 2, 22, 6)
        row.setSpacing(0)
        muted = (tokens or {}).get('text_muted', CHROME['muted'])

        hues = role_colors(tokens)
        spoken = [name for role, name in roles if role in hues]
        # One wrapping rich-text label rather than a row of swatch-and-caption pairs: a
        # pinned strip must be able to narrow to the window, and eight pairs laid out side
        # by side have a minimum width of their own that would force the page to scroll
        # sideways at laptop size.  This reflows to a second line instead.
        swatches = '&nbsp;&nbsp;&nbsp;'.join(
            f'<span style="color:{hues[role]}">&#9632;</span>&nbsp;'
            f'{html.escape(name)}'
            for role, name in roles if role in hues)
        strip = label(f'<span style="color:{muted}">colour key&nbsp;&nbsp;</span>'
                      f'<span style="color:{muted}">{swatches}</span>',
                      rich=True, small=True)
        strip.setWordWrap(True)
        row.addWidget(strip, 1)
        # It lives at the top of the variable panel now, not pinned above the page, and
        # at the panel's 330-pixel width the eight names need three lines.  A two-line
        # ceiling cut the third off mid-word; four still keeps it from becoming a
        # paragraph.
        self.setMaximumHeight(4 * self.fontMetrics().lineSpacing() + 12)

        self._spoken = spoken
        self.setAccessibleName('Colour key: ' + ', '.join(spoken))
        strip.setAccessibleName(self.accessibleName())
        self.setToolTip('These hues mean the same thing in every picture on this page. '
                        'Every coloured element is also labelled in words.')

    def describe(self) -> list[str]:
        """The text this widget puts on screen, for the tests that forbid colour-only."""
        return ['colour key'] + list(self._spoken)


def badged(badge: QWidget, text: str) -> QWidget:
    """A badge followed by a sentence that may wrap -- unlike `badge_row`, which would
    squeeze the sentence into a narrow column beside its trailing stretch."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    row.addWidget(badge, 0, Qt.AlignTop)
    words = label(text, muted=True, small=True)
    row.addWidget(words, 1)
    holder.words = words
    return holder


def badge_row(*badges: QWidget) -> QWidget:
    """Lay badges out left to right without letting them stretch."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    for widget in badges:
        row.addWidget(widget)
    row.addStretch(1)
    return holder


# ── Visual component: pipeline flow navigator ────────────────────────────────

class PipelineNavigator(QWidget):
    """Vestigial: the shell no longer uses this (see CLAUDE.md). Horizontal clickable pipeline showing the stages of the active view.

    The stage list is a constructor argument rather than a class constant: three views
    now present different journeys through the same capture.
    """

    def __init__(self, stages, on_navigate, colors: dict | None = None, parent=None):
        super().__init__(parent)
        self._on_navigate = on_navigate
        self._colors = colors or {}
        self._buttons: dict[str, QPushButton] = {}
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.set_stages(stages)

    def set_stages(self, stages) -> None:
        clear_layout(self._layout)      # not takeAt/deleteLater -- see clear_layout
        self._buttons.clear()
        for position, (key, name) in enumerate(stages):
            if position > 0:
                arrow = QLabel('→')
                arrow.setObjectName('pipelineArrow')
                arrow.setAlignment(Qt.AlignCenter)
                self._layout.addWidget(arrow)
            button = QPushButton(name)
            button.setObjectName('pipelineChip')
            button.setCheckable(True)
            button.clicked.connect(lambda _, k=key: self._on_navigate(k))
            self._layout.addWidget(button)
            self._buttons[key] = button
        self._layout.addStretch(1)

    def keys(self) -> list[str]:
        return list(self._buttons)

    def set_current(self, key: str) -> None:
        for existing, button in self._buttons.items():
            button.setChecked(existing == key)


# ── Visual component: selectable token chips ──────────────────────────────────

class TokenChipStrip(QWidget):
    """The real tokens of a capture, as chips the reader can select.

    Used for the prompt (what the model received) and for the response (which step to
    explain).  Paginated: a run can be 128 new tokens long, and a single unbounded row
    of chips is unreadable.
    """

    selected = pyqtSignal(int)
    PAGE = 32

    def __init__(self, tokens, offset: int = 0, parent=None):
        super().__init__(parent)
        self._tokens = list(tokens)
        self._offset = offset
        self._page = 0
        self._current = 0
        self._buttons: dict[int, QPushButton] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        self._strip = QWidget()
        self._grid = QHBoxLayout(self._strip)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(4)
        outer.addWidget(self._strip)

        self._pager = QWidget()
        pager = QHBoxLayout(self._pager)
        pager.setContentsMargins(0, 0, 0, 0)
        self._back = QPushButton('‹ earlier')
        self._forward = QPushButton('later ›')
        self._range = label('', muted=True, small=True)
        for widget in (self._back, self._range, self._forward):
            pager.addWidget(widget)
        pager.addStretch(1)
        self._back.clicked.connect(lambda: self._show_page(self._page - 1))
        self._forward.clicked.connect(lambda: self._show_page(self._page + 1))
        outer.addWidget(self._pager)
        self._pager.setVisible(len(self._tokens) > self.PAGE)
        self._show_page(0)

    def _pages(self) -> int:
        return max(1, -(-len(self._tokens) // self.PAGE))

    def _show_page(self, page: int) -> None:
        self._page = max(0, min(page, self._pages() - 1))
        clear_layout(self._grid)        # not takeAt/deleteLater -- see clear_layout
        self._buttons.clear()
        start = self._page * self.PAGE
        for index in range(start, min(start + self.PAGE, len(self._tokens))):
            button = QPushButton(chip_text(self._tokens[index]))
            button.setObjectName('tokenChip')
            button.setCheckable(True)
            button.setChecked(index == self._current)
            button.setToolTip(f'position {index + self._offset}  ·  {self._tokens[index]!r}')
            button.clicked.connect(lambda _, i=index: self.select(i))
            self._grid.addWidget(button)
            self._buttons[index] = button
        self._grid.addStretch(1)
        self._back.setEnabled(self._page > 0)
        self._forward.setEnabled(self._page < self._pages() - 1)
        self._range.setText(f'tokens {start + self._offset}–'
                            f'{min(start + self.PAGE, len(self._tokens)) - 1 + self._offset} '
                            f'of {len(self._tokens)}')

    def select(self, index: int) -> None:
        self._current = index
        if not (self._page * self.PAGE <= index < (self._page + 1) * self.PAGE):
            self._show_page(index // self.PAGE)
        for existing, button in self._buttons.items():
            button.setChecked(existing == index)
        self.selected.emit(index)

    def current(self) -> int:
        return self._current


# ── Visual component: matrix shape diagram ────────────────────────────────────

class ShapeDiagram(QWidget):
    """Visual representation of an operation showing actual captured shapes.

    Dimensions come from captured arrays; nothing here is inferred from model config.
    """

    def __init__(self, operands: list[tuple[str, str, tuple]], operator: str = '@',
                 colors: dict | None = None, parent=None):
        super().__init__(parent)
        self._colors = colors or {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)
        for position, (name, color_key, shp) in enumerate(operands):
            if position > 0:
                operator_label = QLabel(operator)
                operator_label.setObjectName('shapeOp')
                operator_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(operator_label)
            layout.addWidget(self._make_box(name, color_key, shp))
        layout.addStretch(1)

    def _make_box(self, name, color_key, shp):
        color = COLORS.get(color_key, '#8888aa')
        dims = ' × '.join(str(d) for d in shp)
        box = QFrame()
        box.setObjectName('shapeDiagramBox')
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(10, 6, 10, 6)
        box_layout.setSpacing(2)
        name_label = QLabel(name)
        name_label.setObjectName('shapeDiagramName')
        name_label.setStyleSheet(f'color: {color}; font-weight: 700; font-size: 14px;')
        name_label.setAlignment(Qt.AlignCenter)
        dim_label = QLabel(f'[{dims}]')
        dim_label.setObjectName('shapeDiagramDims')
        dim_label.setStyleSheet('font-size: 11px;')
        dim_label.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(name_label)
        box_layout.addWidget(dim_label)
        box.setStyleSheet(f'QFrame#shapeDiagramBox {{ border: 2px solid {color}; border-radius: 6px; '
                          f'background: transparent; min-width: 80px; max-width: 160px; }}')
        return box


# ── Visual component: conceptual flow diagram ────────────────────────────────

class FlowDiagram(QWidget):
    """A labelled box-and-arrow sketch, for explaining a shape before showing one.

    Deliberately carries no numbers: it is the level-2 rung of the disclosure ladder
    (purpose → diagram → equation → dimensions → tensor), so putting captured values
    in it would defeat the ordering.
    """

    def __init__(self, nodes, arrow: str = '→', caption: str = '', parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 6)
        outer.setSpacing(6)
        row_holder = QWidget()
        row = QHBoxLayout(row_holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for position, node in enumerate(nodes):
            text, color_key = node if isinstance(node, tuple) else (node, 'layer')
            if position > 0:
                separator = QLabel(arrow)
                separator.setObjectName('shapeOp')
                separator.setAlignment(Qt.AlignCenter)
                row.addWidget(separator)
            colour = COLORS.get(color_key, '#8888aa')
            box = QLabel(text)
            box.setObjectName('flowNode')
            box.setAlignment(Qt.AlignCenter)
            box.setWordWrap(True)
            box.setStyleSheet(
                f'border: 1px solid {colour}; border-radius: 6px; padding: 8px 12px; '
                f'color: {colour}; font-size: 12px; font-weight: 600;')
            row.addWidget(box)
        row.addStretch(1)
        outer.addWidget(row_holder)
        if caption:
            outer.addWidget(label(caption, muted=True, small=True))


# ── Visual component: mini attention heatmap ─────────────────────────────────

class MiniHeatmap(FigureCanvas):
    """Small matplotlib heatmap of an attention slice, for display only.

    `display_matrix` stride-samples to at most 48×48 and returns a *view*, so this
    class must never write through it.  It only reads.
    """

    def __init__(self, array: np.ndarray, title: str = '',
                 mpl_style: dict | None = None, parent=None):
        # Deferred: TensorScope imports the view modules, so a module-level import here
        # would run while TensorScope is still half-initialised.
        from TensorScope import display_matrix
        figure = Figure(figsize=(3.8, 2.8), tight_layout=True)
        super().__init__(figure)
        axes = figure.add_subplot(111)
        sampled = display_matrix(array, maximum=48)
        # viridis: perceptually uniform and colourblind-safe.
        image = axes.imshow(sampled, aspect='auto', cmap='viridis', interpolation='nearest')
        figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
        if title:
            axes.set_title(title, fontsize=9)
        axes.set_xlabel('Key token index', fontsize=8)
        axes.set_ylabel('Query token index', fontsize=8)
        axes.tick_params(labelsize=7)
        if mpl_style:
            figure.set_facecolor(mpl_style.get('figure.facecolor', '#111114'))
            axes.set_facecolor(mpl_style.get('axes.facecolor', '#1c1c20'))
            # Without this the default near-black text is invisible on a dark figure.
            colour = mpl_style.get('text.color')
            if colour:
                axes.title.set_color(colour)
                axes.xaxis.label.set_color(colour)
                axes.yaxis.label.set_color(colour)
                axes.tick_params(colors=colour)
                for spine in axes.spines.values():
                    spine.set_color(colour)
                bar = image.colorbar
                if bar is not None:
                    bar.ax.tick_params(colors=colour)
                    bar.outline.set_edgecolor(colour)
        self.setMinimumSize(QSize(280, 200))
        self.setMaximumSize(QSize(500, 360))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)


# ── Card ─────────────────────────────────────────────────────────────────────

class Card(QFrame):
    def __init__(self, title, plain='', parent=None):
        super().__init__(parent)
        self.setObjectName('journeyCard')
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(20, 18, 20, 18)
        self.body.setSpacing(10)
        self.add(label(title, title=True))
        if plain:
            self.add(label(plain, rich=True))

    def add(self, widget):
        self.body.addWidget(widget)
        return widget

    def add_layout(self, layout):
        self.body.addLayout(layout)

    def field(self, heading, text, rich=False):
        self.add(label(heading.upper(), muted=True))
        return self.add(label(text, rich=rich))

    def equation(self, text):
        widget = label(text, rich=True)
        widget.setObjectName('journeyEquation')
        return self.add(widget)

    def shape_diagram(self, operands, operator='@'):
        return self.add(ShapeDiagram(operands, operator))

    def section_rule(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName('sectionRule')
        return self.add(line)


# ── Disclosure ────────────────────────────────────────────────────────────────

class Disclosure(QWidget):
    """Construct expensive numerical widgets only when the reader requests them.

    Nothing inside is built until `toggle` fires, which is why the self-test has to
    open every one of these: merely constructing a view creates no tensor widgets.
    Passing `state` remembers whether the reader had this open, so a screen the lesson
    rebuilds comes back as they left it.  Only pass it for cheap bodies -- text, a small
    table: restoring an open disclosure builds its body immediately, and doing that for a
    TensorInspector or an AttentionExplorer would rebuild heavy widgets on every switch and
    throw away the laziness this class exists for.
    """
    def __init__(self, title, build, parent=None, *, state: dict | None = None, state_key: str = ''):
        super().__init__(parent)
        self.build = build
        self.content = None
        self._state = state
        self._state_key = state_key or title
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton('＋ ' + title)
        self.button.setCheckable(True)
        self.button.setAccessibleName(title)
        self.title = title          # toggle() reads it, so it must exist before any restore
        self.button.toggled.connect(self.toggle)
        self.body.addWidget(self.button, 0, Qt.AlignLeft)
        if state is not None and state.get(self._state_key):
            self.button.setChecked(True)

    def toggle(self, checked):
        if checked and self.content is None:
            self.content = self.build()
            self.body.addWidget(self.content)
        if self.content is not None:
            self.content.setVisible(checked)
        self.button.setText(('− ' if checked else '＋ ') + self.title)
        if self._state is not None:
            self._state[self._state_key] = bool(checked)


# ── Phase badge ───────────────────────────────────────────────────────────────

class PhaseBadge(QLabel):
    """Vestigial: the shell no longer uses this (see CLAUDE.md). Coloured badge showing which captured phase is being viewed.

    Known limitation: these two colour pairs are tuned for the dark palette only.
    """

    def __init__(self, generated: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName('phaseBadge')
        self._set(generated)

    def _set(self, generated: bool) -> None:
        if generated:
            self.setText('Phase 2  ·  First generated-token pass')
            self.setStyleSheet('background: #1e3a2a; color: #34d399; border: 1px solid #34d399; '
                               'border-radius: 4px; padding: 4px 10px; font-size: 12px; font-weight: 600;')
        else:
            self.setText('Phase 1  ·  Prompt prefill')
            self.setStyleSheet('background: #1e2d40; color: #60a5fa; border: 1px solid #60a5fa; '
                               'border-radius: 4px; padding: 4px 10px; font-size: 12px; font-weight: 600;')

    def update_phase(self, generated: bool) -> None:
        self._set(generated)


# ── Token table ───────────────────────────────────────────────────────────────

def token_table(capture):
    table = QTableWidget(len(capture.prompt_tokens), 3)
    table.setHorizontalHeaderLabels(['Position', 'Token ID', 'Stored tokenizer spelling'])
    for row, (token_id, text) in enumerate(zip(capture.prompt_token_ids, capture.prompt_tokens)):
        for column, value in enumerate((row, token_id, text)):
            table.setItem(row, column, QTableWidgetItem(str(value)))
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
    table.verticalHeader().hide()
    table.setMinimumHeight(220)
    table.setMaximumHeight(360)
    return table


def read_only_table(headings, rows, *, stretch: int | None = None,
                    highlight: int | None = None, highlight_color: str = '#0e9f6e',
                    tooltips=None):
    """A small non-editable table.  Used for candidate scores and provenance.

    `tooltips`, when given, is parallel to `rows`: a rounded cell can carry its exact value.
    """
    table = QTableWidget(len(rows), len(headings))
    table.setHorizontalHeaderLabels(list(headings))
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            if tooltips is not None and row_index < len(tooltips):
                tip = tooltips[row_index][column] if column < len(tooltips[row_index]) else ''
                if tip:
                    item.setToolTip(tip)
            if row_index == highlight:
                item.setForeground(QBrush(QColor(highlight_color)))
            table.setItem(row_index, column, item)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    if stretch is not None:
        table.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.Stretch)
    table.verticalHeader().hide()
    table.setMinimumHeight(min(320, 32 + 28 * max(1, len(rows))))
    table.setMaximumHeight(360)
    return table


# ── Reveal and check primitives ───────────────────────────────────────────────
# Each of these takes an optional `state` dict plus a `state_key`, and the contract is
# the same everywhere: restore from `state` synchronously while constructing, and write
# back on every change.  The owner keeps the dict.  With the notebook's sections built
# once and never replaced, a reader's reveals and answers usually survive simply by still
# being on the page; the dicts still matter for the parts that do rebuild.

class StepReveal(QWidget):
    """Reveal one calculation a deliberate step at a time.

    A step's widget is built the first time it is revealed, the same laziness `Disclosure`
    uses.  The invariant the tests rely on: the reveal, its child widget and `revealed()`
    are all final before `reveal_next()` returns, so nothing ever waits on a timer.  The
    tint animation is decoration on top of an already-finished state change.
    """

    ANIMATION_MS = 320

    def __init__(self, steps, *, state: dict | None = None, state_key: str = '',
                 caption: str = STRUCTURE_NOT_TIMING_NOTE, animate: bool = True,
                 initial: int = 0, parent=None):
        super().__init__(parent)
        self._steps = list(steps)
        self._state = state if state is not None else {}
        self._state_key = state_key or 'reveal'
        self._animate = animate and not OFFSCREEN
        self._rows: list[QWidget] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        self._holder = QWidget()
        self._holder_layout = QVBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(0, 0, 0, 0)
        self._holder_layout.setSpacing(8)
        outer.addWidget(self._holder)

        controls = QWidget()
        control_row = QHBoxLayout(controls)
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(8)
        self.next_button = QPushButton('Reveal next step')
        self.next_button.setObjectName('revealNext')
        self.all_button = QPushButton('Show every step')
        self.all_button.setObjectName('revealControl')
        self.reset_button = QPushButton('Start over')
        self.reset_button.setObjectName('revealControl')
        for button in (self.next_button, self.all_button, self.reset_button):
            control_row.addWidget(button)
        self._counter = label('', muted=True, small=True)
        control_row.addWidget(self._counter)
        control_row.addStretch(1)
        self.next_button.clicked.connect(self.reveal_next)
        self.all_button.clicked.connect(self.reveal_all)
        self.reset_button.clicked.connect(self.reset)
        outer.addWidget(controls)
        outer.addWidget(label(caption, muted=True, small=True))

        # `initial` steps are open on arrival, so the section's focal point is a
        # calculation rather than a button; a reader who chose another count keeps it.
        self.reveal_to(int(self._state.get(self._state_key, initial) or 0))

    # -- state ----------------------------------------------------------------
    @property
    def step_count(self) -> int:
        return len(self._steps)

    def revealed(self) -> int:
        return len(self._rows)

    def reveal_next(self) -> None:
        self.reveal_to(self.revealed() + 1)

    def reveal_all(self) -> None:
        self.reveal_to(self.step_count)

    def reset(self) -> None:
        self.reveal_to(0)

    def reveal_to(self, count: int) -> None:
        count = max(0, min(int(count), self.step_count))
        try:
            while len(self._rows) > count:
                row = self._rows.pop()
                self._holder_layout.removeWidget(row)
                row.setParent(None)
                row.deleteLater()
            while len(self._rows) < count:
                row = self._build_row(len(self._rows))
                self._holder_layout.addWidget(row)
                self._rows.append(row)
                self._highlight(row)
        finally:
            # A step whose build() raises leaves _rows short but consistent; the controls
            # and remembered count must describe that, not the count that was asked for.
            self._state[self._state_key] = len(self._rows)
            self.next_button.setEnabled(len(self._rows) < self.step_count)
            self.all_button.setEnabled(len(self._rows) < self.step_count)
            self.reset_button.setEnabled(bool(self._rows))
            self._counter.setText(f'{len(self._rows)} of {self.step_count} steps shown')

    # -- construction ---------------------------------------------------------
    def _build_row(self, index: int) -> QWidget:
        step = self._steps[index]
        frame = QFrame()
        frame.setObjectName('revealStep')
        body = QVBoxLayout(frame)
        body.setContentsMargins(12, 10, 12, 10)
        body.setSpacing(6)
        heading = label(f'Step {index + 1}  ·  {step["caption"]}')
        heading.setObjectName('revealStepCaption')
        body.addWidget(heading)
        if step.get('tier'):
            body.addWidget(badge_row(EvidenceBadge(step['tier'], step.get('note', ''))))
        if step.get('body'):
            body.addWidget(label(step['body'], rich=bool(step.get('rich'))))
        build = step.get('build')
        if build is not None:
            widget = build()
            if widget is not None:
                body.addWidget(widget)
        return frame

    def _highlight(self, widget: QWidget) -> None:
        """Tint the row that just appeared, so the eye lands on it.

        The animation object is stored on the widget: an unreferenced QVariantAnimation is
        collected mid-run and the tint would simply stop partway.  Nothing about the
        revealed state depends on it finishing.
        """
        if not self._animate:
            return
        colour = CHROME['accent']
        animation = QVariantAnimation(widget)
        animation.setDuration(self.ANIMATION_MS)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)

        def apply(value) -> None:
            strength = float(value or 0.0)
            widget.setStyleSheet('' if strength <= 0.02 else
                                 f'QFrame#revealStep {{ border-left: 3px solid {colour}; }}')

        animation.valueChanged.connect(apply)
        animation.finished.connect(lambda: widget.setStyleSheet(''))
        widget._reveal_animation = animation
        animation.start()


class PredictCheck(QFrame):
    """An optional predict-then-reveal question.

    Nothing in the lesson is ever disabled or gated by one of these: answering only
    reveals the explanation, and skipping reveals the same explanation.
    """

    def __init__(self, question: str, options, answer: int, explanation: str, *,
                 state: dict | None = None, state_key: str = '', parent=None):
        super().__init__(parent)
        self.setObjectName('predictCheck')
        self._state = state if state is not None else {}
        self._state_key = state_key or 'check'
        self._answer = int(answer)
        self._buttons: list[QPushButton] = []

        body = QVBoxLayout(self)
        body.setContentsMargins(14, 12, 14, 12)
        body.setSpacing(8)
        body.addWidget(label('OPTIONAL CHECK  ·  skip it and nothing changes', muted=True, small=True))
        body.addWidget(label(question))
        for index, text in enumerate(options):
            button = QPushButton(str(text))
            button.setObjectName('predictOption')
            button.setCheckable(True)
            button.clicked.connect(lambda _, i=index: self.choose(i))
            body.addWidget(button, 0, Qt.AlignLeft)
            self._buttons.append(button)
        self.skip_button = QPushButton('Show me the answer')
        self.skip_button.setObjectName('predictSkip')
        self.skip_button.clicked.connect(lambda: self.choose(-1))
        body.addWidget(self.skip_button, 0, Qt.AlignLeft)
        self._explanation = label(explanation)
        self._explanation.setObjectName('predictExplanation')
        self._explanation.setVisible(False)
        body.addWidget(self._explanation)

        remembered = self._state.get(self._state_key)
        if remembered is not None:
            self.choose(int(remembered))

    def choose(self, index: int) -> None:
        """Record a pick (or -1 for a skip) and reveal the explanation."""
        self._state[self._state_key] = int(index)
        for position, button in enumerate(self._buttons):
            button.setChecked(position == index)
            if position == self._answer:
                button.setStyleSheet(f"color: {CHROME['accent']}; font-weight: 700;")
            elif position == index:
                button.setStyleSheet(f"color: {CHROME['muted']};")
            else:
                button.setStyleSheet('')
        self._explanation.setVisible(True)
        self.skip_button.setVisible(False)

    def chosen(self) -> int | None:
        return self._state.get(self._state_key)


class AnnotatedValueRow(QWidget):
    """A few real values with a label over each and one evidence badge for the row.

    The caller passes exactly the values it means to show, which keeps the sampling decision
    visible in the screen code rather than hidden in a widget.  `formatter` decides how a
    value *reads* -- `approx` in the teaching view, `exact_scalar` in the variable panel --
    and whatever it is, every box carries the exact stored digits as its tooltip, so rounding
    for reading never stands between the reader and the real number.  `details` adds one
    short line under a value, such as its share of a weight row.
    """

    MAX_BOXES = 8
    COLUMNS = 4

    def __init__(self, values, *, labels=None, indices=None, caption: str = '',
                 kind: str = EVIDENCE_OBSERVED, note: str = '', total: int | None = None,
                 formatter=None, details=None, max_boxes: int | None = None,
                 columns: int | None = None, parent=None):
        super().__init__(parent)
        everything = list(values)
        limit = self.MAX_BOXES if max_boxes is None else int(max_boxes)
        shown = everything[:limit]
        names = list(labels) if labels is not None else [
            f'index {i}' for i in (list(indices) if indices is not None else range(len(shown)))]
        extra = list(details or [])
        render = formatter or exact_scalar
        per_row = int(columns or self.COLUMNS)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(4)
        # The badge heads the row, beside its caption, so the values themselves line up in
        # whole rows rather than wrapping around it.
        heading = QHBoxLayout()
        heading.setContentsMargins(0, 0, 0, 0)
        heading.setSpacing(8)
        heading.addWidget(EvidenceBadge(kind, note), 0, Qt.AlignVCenter)
        if caption:
            heading.addWidget(label(caption, muted=True, small=True), 1)
        else:
            heading.addStretch(1)
        outer.addLayout(heading)
        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        for position, value in enumerate(shown):
            grid.addWidget(self._box(names[position] if position < len(names) else '',
                                     render(value), exact_scalar(value),
                                     extra[position] if position < len(extra) else ''),
                           position // per_row, position % per_row)
        grid.setColumnStretch(per_row, 1)
        outer.addWidget(grid_holder)
        hidden = (total if total is not None else len(everything)) - len(shown)
        if hidden > 0:
            outer.addWidget(label(f'… {hidden:,} more in the saved tensor', muted=True,
                                  small=True))

    def _box(self, name: str, text: str, exact: str = '', detail: str = '') -> QWidget:
        box = QFrame()
        box.setObjectName('valueBox')
        inner = QVBoxLayout(box)
        inner.setContentsMargins(8, 5, 8, 5)
        inner.setSpacing(1)
        heading = label(name, muted=True, small=True)
        heading.setWordWrap(False)
        value = label(text)
        value.setObjectName('valueBoxNumber')
        value.setWordWrap(False)
        inner.addWidget(heading)
        inner.addWidget(value)
        if detail:
            inner.addWidget(label(detail, muted=True, small=True))
        if exact:
            box.setToolTip(f'Exact stored value: {exact}')
            value.setToolTip(box.toolTip())
        return box


class ResponseTextView(QWidget):
    """The generated answer as readable text, with every token a clickable span.

    Eighty-one chips is a wall, so this is one rich label whose tokens are anchors styled
    as a faint dotted underline.  If the stored spellings do not reassemble into the stored
    response, the response is shown verbatim as plain text and the tokens fall back to a
    chip strip: the text on screen is never something this widget assembled out of pieces
    that disagree with it.

    These tokens are *not* byte-level BPE spellings and must not be run through
    `readable_spelling`.  ModelCapture fills `prompt_tokens` from
    `convert_ids_to_tokens` (so 'Gcolor' with U+0120) but `tokens` from
    `tokenizer.decode`, which is already real text, and `response` is exactly their
    concatenation.  Transforming them here would both corrupt a token that legitimately
    contains one of those characters and break the faithfulness check below, which
    compares the untransformed join.
    """

    selected = pyqtSignal(int)

    def __init__(self, tokens, response: str, *, selected: int = 0, ink: str | None = None,
                 parent=None):
        super().__init__(parent)
        self._tokens = list(tokens)
        self._response = response or ''
        self._ink = ink
        self._current = max(0, min(int(selected), max(0, len(self._tokens) - 1)))
        # Outer whitespace is trimmed from a stored response but not from its tokens; that
        # alone is not a disagreement about the text.
        self._faithful = bool(self._tokens) and ''.join(self._tokens).strip() == self._response.strip()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        self._label = label('', rich=True)
        self._label.setObjectName('responseText')
        self._label.setOpenExternalLinks(False)
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        self._label.linkActivated.connect(self.activate_anchor)
        outer.addWidget(self._label)
        self._strip = None
        if not self._faithful:
            outer.addWidget(label(
                'The stored token spellings do not reassemble into the stored response text, so '
                'the response is shown exactly as it was saved and its tokens are listed '
                'separately.', muted=True, small=True))
            self._strip = TokenChipStrip(self._tokens)
            self._strip.selected.connect(self.select)
            outer.addWidget(self._strip)
        self._render()
        # TokenChipStrip always starts on index 0, so without this the only visible
        # selector disagrees with what current() reports to the owner.
        self._sync_strip()

    def anchors(self) -> list[str]:
        return [f't:{index}' for index in range(self.anchor_count())]

    def anchor_count(self) -> int:
        return len(self._tokens) if self._faithful else 0

    def activate_anchor(self, href: str) -> None:
        """Parse an anchor the label emitted, then select that token."""
        kind, _, position = str(href).partition(':')
        if kind == 't' and position.isdigit():
            self.select(int(position))

    def is_faithful(self) -> bool:
        return self._faithful

    def select(self, index: int) -> None:
        if not self._tokens:
            return
        self._current = max(0, min(int(index), len(self._tokens) - 1))
        self._render()
        self._sync_strip()
        self.selected.emit(self._current)

    def current(self) -> int:
        return self._current

    def _sync_strip(self) -> None:
        """Move the fallback chip strip without its signal echoing back through here.

        TokenChipStrip.select emits `selected`, which is wired to our own `select`; left
        unblocked one programmatic selection re-enters and emits twice, so a consumer that
        rebuilds a panel on the signal would rebuild it twice.
        """
        if self._strip is None or self._strip.current() == self._current:
            return
        blocked = self._strip.signalsBlocked()
        self._strip.blockSignals(True)
        try:
            self._strip.select(self._current)
        finally:
            self._strip.blockSignals(blocked)

    def _render(self) -> None:
        if not self._faithful:
            self._label.setText('<div style="white-space: pre-wrap;">'
                                + html.escape(self._response).replace('\n', '<br>') + '</div>')
            return
        pieces = []
        colour, tint = CHROME['accent'], CHROME['accent_tint']
        # Explicit, because Qt's rich text does not honour `color: inherit` on an anchor and
        # falls back to the link colour, which reads as greyed-out text on either palette.
        # Before the widget is polished its palette is Qt's default, black, so the caller's
        # theme colour wins whenever it is given.
        ink = self._ink or self._label.palette().windowText().color().name()
        for index, token in enumerate(self._tokens):
            # Verbatim: these are decoded text, not BPE spellings.  See the class docstring.
            text = html.escape(token).replace('\n', '<br>') or '&nbsp;'
            if '\n' in token and not token.strip():
                # An anchor around nothing but a line break has no width to click.
                text = '↵' + text
            style = (f"text-decoration: none; border-bottom: 1px dotted {CHROME['muted']}; "
                     f'color: {ink};')
            if index == self._current:
                style += f' background: {tint}; border-bottom: 1px solid {colour};'
            pieces.append(f'<a href="t:{index}" style="{style}">{text}</a>')
        self._label.setText('<div style="white-space: pre-wrap; line-height: 160%;">'
                            + ''.join(pieces) + '</div>')


class BandedTokenStrip(QWidget):
    """Prompt tokens as selectable chips, grouped into labelled bands.

    `TokenChipStrip` pages one unbounded row, which is right for 81 response tokens and
    wrong here: a prompt is short enough to show at once, and bands only read as groups if
    they are laid out as blocks.  `bands` of None or [] degrades to one unlabelled group
    rather than inventing a grouping.
    """

    selected = pyqtSignal(int)

    def __init__(self, tokens, bands=None, *, offset: int = 0, selected: int = 0,
                 columns: int = 8, parent=None):
        super().__init__(parent)
        self._tokens = list(tokens)
        self._offset = offset
        self._columns = max(1, int(columns))
        self._buttons: dict[int, QPushButton] = {}
        self._current = max(0, min(int(selected), max(0, len(self._tokens) - 1)))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        for start, end, name in (list(bands) if bands else [(0, len(self._tokens), '')]):
            if name:
                heading = label(name.upper(), muted=True, small=True)
                heading.setObjectName('bandHeading')
                outer.addWidget(heading)
            outer.addWidget(self._band(start, end))
        self._sync()

    def _band(self, start: int, end: int) -> QWidget:
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for position, index in enumerate(range(start, end)):
            token = self._tokens[index]
            button = QPushButton(chip_text(token))
            button.setObjectName('tokenChip')
            button.setCheckable(True)
            button.setToolTip(f'position {index + self._offset}  ·  stored {token!r}  ·  '
                              f'reads as {readable_spelling(token)!r}')
            button.clicked.connect(lambda _, i=index: self.select(i))
            grid.addWidget(button, position // self._columns, position % self._columns)
            self._buttons[index] = button
        grid.setColumnStretch(self._columns, 1)
        return holder

    def select(self, index: int) -> None:
        if not self._tokens:
            return
        self._current = max(0, min(int(index), len(self._tokens) - 1))
        self._sync()
        self.selected.emit(self._current)

    def current(self) -> int:
        return self._current

    def _sync(self) -> None:
        for index, button in self._buttons.items():
            button.setChecked(index == self._current)


class StageStack(QStackedWidget):
    """A stack that is only as large as the page on show.

    QStackedWidget normally reports the largest child's size hint, so caching eight screens
    would make every one of them as tall as the tallest inside the shell's scroll area.
    """

    def sizeHint(self) -> QSize:
        current = self.currentWidget()
        return current.sizeHint() if current is not None else super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        current = self.currentWidget()
        return current.minimumSizeHint() if current is not None else super().minimumSizeHint()


class StageProgress(QWidget):
    """Vestigial: the shell no longer uses this (see CLAUDE.md). Where the reader is in whatever list of stages the active view declares.

    Count-driven, so it needs no knowledge of which view is showing: the pill count comes
    from the view's own NAV and the noun from the view itself.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pills: list[QFrame] = []
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(6)
        self._pill_holder = QWidget()
        self._pill_row = QHBoxLayout(self._pill_holder)
        self._pill_row.setContentsMargins(0, 0, 0, 0)
        self._pill_row.setSpacing(3)
        self._caption = label('', muted=True, small=True)
        self._row.addWidget(self._pill_holder)
        self._row.addWidget(self._caption)

    def set_position(self, index: int, total: int, noun: str = 'Stage') -> None:
        while len(self._pills) > total:
            pill = self._pills.pop()
            self._pill_row.removeWidget(pill)
            pill.setParent(None)
            pill.deleteLater()
        while len(self._pills) < total:
            pill = QFrame()
            pill.setObjectName('progressPill')
            pill.setFixedSize(QSize(14, 4))
            self._pill_row.addWidget(pill)
            self._pills.append(pill)
        reached, ahead = CHROME['accent'], CHROME['ahead']
        for position, pill in enumerate(self._pills):
            pill.setStyleSheet(f'background: {reached if position <= index else ahead}; '
                               'border-radius: 2px;')
        self._caption.setText(f'{noun} {index + 1} of {total}' if total else '')
        self.setVisible(bool(total))


# ── The reading bar's three controls ──────────────────────────────────────────────
# Where the reader is, a compact contents menu, and (in the shell) Previous/Next.  These
# replace the pipeline chips, the sidebar list, the breadcrumb and the progress pills, which
# were four views of the same list competing for the top of the window.

class ChapterProgress(QWidget):
    """Which chapter the reader is in, and where in the page's sections.

    Not clickable: chapters are a way of reading the page, not a way of moving around it.
    The pills are one per chapter, so four of them describe thirteen sections without
    becoming a second navigator.  A page without chapters shows the caption alone.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('chapterProgress')
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._pill_holder = QWidget()
        self._pill_holder.setAttribute(Qt.WA_TranslucentBackground)
        self._pill_holder.setStyleSheet('background: transparent;')
        self._pill_row = QHBoxLayout(self._pill_holder)
        self._pill_row.setContentsMargins(0, 0, 0, 0)
        self._pill_row.setSpacing(3)
        row.addWidget(self._pill_holder)
        self._caption = label('', small=True)
        self._caption.setWordWrap(False)
        row.addWidget(self._caption)
        row.addStretch(1)
        self._chapters: list = []
        self._pills: list[QFrame] = []

    def set_chapters(self, chapters) -> None:
        """`chapters` is a list of (title, section keys), or None for a plain list."""
        self._chapters = [(str(title), tuple(keys)) for title, keys in (chapters or [])]
        clear_layout(self._pill_row)
        self._pills = []
        for title, _keys in self._chapters:
            pill = QFrame()
            pill.setObjectName('chapterPill')
            pill.setFixedSize(QSize(26, 5))
            pill.setToolTip(title)
            self._pill_row.addWidget(pill)
            self._pills.append(pill)
        self._pill_holder.setVisible(bool(self._pills))

    def chapter_of(self, key: str):
        for index, (_title, keys) in enumerate(self._chapters):
            if key in keys:
                return index
        return None

    def set_position(self, key: str, where: str) -> None:
        """Mark `key`'s chapter and write `where` -- e.g. "Operation 7 of 12"."""
        chapter = self.chapter_of(key)
        reached, ahead = CHROME['accent'], CHROME['ahead']
        for index, pill in enumerate(self._pills):
            filled = chapter is not None and index <= chapter
            pill.setStyleSheet(f'background: {reached if filled else ahead}; '
                               'border-radius: 2px;')
        if chapter is not None:
            self._caption.setText(f'{self._chapters[chapter][0]}  ·  {where}')
        else:
            self._caption.setText(where)

    def text(self) -> str:
        return self._caption.text()


class ContentsButton(QToolButton):
    """Every section of the page, in one menu, grouped under its chapter.

    The detailed navigation lives here rather than in a permanent sidebar.  set_sections
    is safe to call on every mode switch: QMenu.clear() deletes the actions the menu owns
    immediately, so no stale entry can survive to navigate to a key the live page lacks.
    """

    navigate = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('contentsButton')
        self.setText('Contents')
        self.setAccessibleName('Contents')
        self.setToolTip('Jump to any section of this page.')
        self.setPopupMode(QToolButton.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.setFocusPolicy(Qt.StrongFocus)
        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self._group = None
        self.actions_by_key: dict = {}

    def set_sections(self, nav, chapters=None) -> None:
        self._menu.clear()
        if self._group is not None:
            self._group.setParent(None)
            self._group.deleteLater()
        self._group = QActionGroup(self._menu)
        self._group.setExclusive(True)
        self.actions_by_key = {}
        names = dict(nav)
        numbers = {key: index + 1 for index, (key, _) in enumerate(nav)}
        groups = [(title, [key for key in keys if key in names])
                  for title, keys in (chapters or [])] or [('', [key for key, _ in nav])]
        for title, keys in groups:
            if title:
                self._menu.addSection(title)
            for key in keys:
                action = QAction(f'{numbers[key]:02}   {names[key]}', self._menu)
                action.setCheckable(True)
                action.triggered.connect(lambda _=False, k=key: self.navigate.emit(k))
                self._group.addAction(action)
                self._menu.addAction(action)
                self.actions_by_key[key] = action

    def set_current(self, key: str) -> None:
        action = self.actions_by_key.get(key)
        if action is not None:
            action.setChecked(True)

    def section_actions(self) -> list:
        return [action for action in self._menu.actions() if not action.isSeparator()]


class PassTimeline(QWidget):
    """One row per forward pass of the generation loop, and what each one kept.

    Rows are buttons, not a table: a reader clicks a pass to see its decision, and the
    candidate-table helpers in the tests read tables with a "Token ID" column, which this
    must never be mistaken for.  Long runs collapse to the first two passes, the selected
    one and the last, behind a toggle -- the two that kept layer tensors are always shown.
    """

    selected = pyqtSignal(int)
    ALWAYS_SHOWN = 2

    def __init__(self, rows, *, show_all_text: str = '', show_fewer_text: str = '',
                 selected: int = 0, parent=None):
        super().__init__(parent)
        self.setObjectName('passTimeline')
        self._rows = list(rows)
        self._buttons: list = []
        self._lines: list = []
        self._expanded = False
        self._current = max(0, min(int(selected), max(0, len(self._rows) - 1)))
        self._show_all = show_all_text or f'Show all {len(self._rows)} passes'
        self._show_fewer = show_fewer_text or 'Show fewer passes'

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)
        for index, row in enumerate(self._rows):
            line = QWidget()
            line.setObjectName('passRow')
            layout = QHBoxLayout(line)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            button = QPushButton(row['title'])
            button.setObjectName('passButton')
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setToolTip(row.get('detail', '') or row['title'])
            button.clicked.connect(lambda _=False, i=index: self._clicked(i))
            layout.addWidget(button, 1)
            capabilities = label(row.get('capabilities', ''), muted=True, small=True)
            capabilities.setWordWrap(False)
            layout.addWidget(capabilities, 0)
            if row.get('tier'):
                layout.addWidget(EvidenceBadge(row['tier']), 0)
            outer.addWidget(line)
            self._buttons.append(button)
            self._lines.append(line)
        self.toggle_button = QPushButton(self._show_all)
        self.toggle_button.setObjectName('revealControl')
        self.toggle_button.clicked.connect(lambda: self.set_expanded(not self._expanded))
        outer.addWidget(self.toggle_button, 0, Qt.AlignLeft)
        self._sync()

    def count(self) -> int:
        return len(self._rows)

    def row(self, index: int) -> dict:
        return self._rows[index]

    def current(self) -> int:
        return self._current

    def is_row_shown(self, index: int) -> bool:
        return not self._lines[index].isHidden()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self._sync()

    def select(self, index: int) -> None:
        """Move the selection without emitting -- the caller already knows."""
        if not self._rows:
            return
        self._current = max(0, min(int(index), len(self._rows) - 1))
        self._sync()

    def _clicked(self, index: int) -> None:
        self.select(index)
        self.selected.emit(self._current)

    def _sync(self) -> None:
        last = len(self._rows) - 1
        for index, (button, line) in enumerate(zip(self._buttons, self._lines)):
            button.setChecked(index == self._current)
            shown = (self._expanded or index < self.ALWAYS_SHOWN or index == last
                     or index == self._current)
            line.setHidden(not shown)
        collapsible = len(self._rows) > self.ALWAYS_SHOWN + 2
        self.toggle_button.setVisible(collapsible)
        self.toggle_button.setText(self._show_fewer if self._expanded else self._show_all)


# ── A long page whose NAV entries are anchors, not replacements ──────────────────
# The older views replaced the whole screen on every navigation, which is why a reader
# could never see a calculation accumulate: the step they had just understood was gone.
# These two classes are the alternative.  A page builds all of its sections once, in
# order, and navigating scrolls to one and highlights it.  Nothing is destroyed, so every
# selection, expansion and reveal survives navigation by construction rather than by
# being saved and replayed.

class Section(QFrame):
    """One anchor on a long page: a numbered header, a lede, and a lazily built body.

    The split matters for a 36-layer model.  Constructing a section is a header and a
    sentence, which costs nothing, so a page can hold every section from the start and the
    reader can scroll freely.  The body -- diagrams, value rows, tables -- is built the
    first time the section is reached, and the heavy things inside it stay behind a
    `Disclosure` even then.

    `slot(name)` hands out named sub-layouts for the parts that change when the reader
    picks a different token or layer.  Refreshing one slot with `replace_in` repaints a few
    widgets; it does not reflow the page or lose anybody's place.
    """

    def __init__(self, key: str, title: str, lede: str = '', *, number: str = '',
                 build_body=None, kicker: str = '', preface=None, parent=None):
        super().__init__(parent)
        self.key = key
        self.setObjectName('notebookSection')
        self.setProperty('current', False)
        # Focusable so the page can be walked from the keyboard, and so `go_to` has
        # somewhere to put focus that a screen reader will read out.
        self.setFocusPolicy(Qt.StrongFocus)
        self._build_body = build_body
        self._built = False
        self._animation = None

        self._body = QVBoxLayout(self)
        # Flat on the page, not a card: the left margin leaves room for the accent rule
        # that marks the reader's section, and the generous bottom margin is what
        # separates one operation from the next.
        self._body.setContentsMargins(28, 26, 24, 30)
        self._body.setSpacing(12)
        # A preface sits above everything -- the notebook's destination card -- and a
        # kicker names the chapter that begins here.  Both scroll with the section, so a
        # chapter boundary is something the reader passes, not another control.
        self.preface = preface
        if preface is not None:
            self._body.addWidget(preface)
        self.kicker = None
        if kicker:
            self.kicker = label(kicker.upper(), small=True)
            self.kicker.setObjectName('sectionKicker')
            self._body.addWidget(self.kicker)
        heading = f'{number}   {title}' if number else title
        self.header = label(heading, title=True)
        self.header.setObjectName('sectionHeader')
        self._body.addWidget(self.header)
        self.lede = None
        if lede:
            # A sentence a reader is meant to read, so it keeps a readable measure even
            # when the column is wider than the prose.
            self.lede = measure_label(lede, rich=True)
            self.lede.setObjectName('sectionLede')
            self._body.addWidget(self.lede)
            self._body.addSpacing(4)
        self._body_slot = QVBoxLayout()
        self._body_slot.setContentsMargins(0, 0, 0, 0)
        self._body_slot.setSpacing(12)
        self._body.addLayout(self._body_slot)
        self.setAccessibleName(heading)

    # -- lazy body ------------------------------------------------------------

    def is_built(self) -> bool:
        return self._built

    def ensure_built(self) -> None:
        """Build the body once.  Idempotent, so every caller can be careless."""
        if self._built:
            return
        self._built = True                      # set first: a raising builder must not retry
        if self._build_body is not None:
            widget = self._build_body(self)
            if widget is not None:
                self._body_slot.addWidget(widget)

    def add(self, widget: QWidget) -> QWidget:
        self._body_slot.addWidget(widget)
        return widget

    def anchor(self) -> QWidget:
        """What to scroll to: the first thing in the section, so it lands at the top."""
        return self.preface or self.kicker or self.header

    # -- the reader's place --------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Mark this as where the reader is.

        A dynamic property and a re-polish, so the *stylesheet* owns the colour and the
        light and dark palettes need no branching here.
        """
        if self.property('current') == bool(active):
            return
        self.setProperty('current', bool(active))
        self.style().unpolish(self)
        self.style().polish(self)

    def flash(self) -> None:
        """A brief emphasis after a jump.  Decoration on an already-finished move.

        Nothing waits for this and nothing reads its state, which is why it can be skipped
        outright under the offscreen platform.  It says "the thing you asked for is here",
        not anything about how long an operation took.
        """
        if OFFSCREEN:
            return
        def tint(value) -> None:
            self.setProperty('flash', float(value or 0.0) > 0.05)
            self.style().unpolish(self)
            self.style().polish(self)

        animation = QVariantAnimation(self)
        animation.setDuration(220)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.valueChanged.connect(tint)
        animation.finished.connect(lambda: tint(0.0))
        self._animation = animation             # an unreferenced animation is collected
        animation.start()


class AnchoredScrollPage(QWidget):
    """A scrolling page the shell's existing navigator drives by scrolling, not swapping.

    The view contract -- `NAV`, `stage_keys()`, `current_stage()`, `go_to()`,
    `stage_changed`, `context_changed` -- is all the shell's contents menu, chapter progress
    and Previous/Next buttons need, with no layout code in the shell.  What changes is what `go_to` *means*: it moves the reader to a
    section instead of replacing the screen with it.

    Ordering inside `go_to` is deliberate.  The state change and its signal happen first
    and completely; the scroll is best-effort afterwards.  So a test can assert
    `current_stage()` without knowing anything about pixels, and a section that grows when
    it is built cannot leave the page in a half-navigated state.
    """

    stage_changed = pyqtSignal(str)
    context_changed = pyqtSignal(str)
    # (mode key, section key): "explain this" / "show me this in the run" links between the
    # reference page and the worked notebook.  The shell wires it; neither page knows the
    # other exists.
    jump_requested = pyqtSignal(str, str)

    NAV: list = []
    NOUN = 'Section'
    NEXT_LABEL = 'Next →'
    PREVIOUS_LABEL = '← Previous'

    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(parent)
        capture.validate()
        self.capture = capture
        self._tokens = tokens or {}
        self._sections: dict[str, Section] = {}
        self._order: list[str] = []
        self._stage_key = ''
        self._scrolling = False
        self._context = ''

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._header_slot = QVBoxLayout()
        self._header_slot.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self._header_slot)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        # Always on, so the viewport width does not change the moment the page grows past
        # the scrollbar threshold -- which otherwise reflows every section mid-scroll.
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        inner = QWidget()
        inner.setObjectName('pageSurface')
        self._page = QVBoxLayout(inner)
        self._page.setContentsMargins(16, 12, 16, 48)
        self._page.setSpacing(0)
        self.scroll.setWidget(inner)
        # Centre the sections in a reading column.  Margins rather than a maximum width
        # on each section, so word-wrapped labels still get their height-for-width from
        # the layout and the column simply narrows with the window.
        self.scroll.viewport().installEventFilter(self)
        # A splitter from the start, so a page that wants a panel beside its scroll area
        # can add one without the scroll area having to be re-parented afterwards.
        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.scroll)
        self.split.setStretchFactor(0, 1)
        outer.addWidget(self.split, 1)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scrolled)

    def eventFilter(self, watched, event):                  # noqa: N802 - Qt name
        if event.type() == QEvent.Resize and watched is self.scroll.viewport():
            self._fit_column(event.size().width())
        return super().eventFilter(watched, event)

    def _fit_column(self, width: int) -> None:
        side = max(16, (int(width) - COLUMN_WIDTH) // 2)
        margins = self._page.contentsMargins()
        if margins.left() != side:
            self._page.setContentsMargins(side, margins.top(), side, margins.bottom())

    # ── Construction ─────────────────────────────────────────────────────────

    def add_header(self, widget: QWidget) -> QWidget:
        """A pinned strip above the scroll area -- a context ribbon, say."""
        self._header_slot.addWidget(widget)
        return widget

    def add_side_panel(self, widget: QWidget, *, width: int = 330,
                       hidden: bool = False) -> QWidget:
        """A panel beside the scrolling page, reader-resizable and closable.

        `hidden` starts it closed, so a panel with nothing to show yet does not take a
        quarter of the window; `show_side_panel` opens it when there is something to say.
        """
        self.split.addWidget(widget)
        self.split.setStretchFactor(self.split.count() - 1, 0)
        self.split.setCollapsible(self.split.count() - 1, False)
        widget.setMinimumWidth(240)
        self._side_width = width
        self._side_panel = widget
        self.split.setSizes([900, width])
        if hidden:
            widget.hide()
        return widget

    def side_panel_open(self) -> bool:
        panel = getattr(self, '_side_panel', None)
        return panel is not None and not panel.isHidden()

    def show_side_panel(self) -> None:
        panel = getattr(self, '_side_panel', None)
        if panel is None or not panel.isHidden():
            return
        with self.preserve_place():
            panel.show()
            total = max(self.split.width(), self._side_width + 400)
            self.split.setSizes([total - self._side_width, self._side_width])

    def hide_side_panel(self) -> None:
        panel = getattr(self, '_side_panel', None)
        if panel is None or panel.isHidden():
            return
        with self.preserve_place():
            panel.hide()

    @contextmanager
    def preserve_place(self):
        """Keep the reader on the same line of the same section across a width change.

        Opening or closing the side panel changes the page's width, every section reflows,
        and the scroll offset that pointed at a line of operation 7 now points somewhere
        else -- which `_track` would then report as a different section.  So the place is
        recorded as (section, offset into it), the change happens with `_scrolling` set,
        the layout is settled synchronously, and the same place is restored.
        """
        bar = self.scroll.verticalScrollBar()
        section = self._sections.get(self._stage_key)
        offset = bar.value() - section.y() if section is not None else None
        was = self._scrolling
        self._scrolling = True
        try:
            yield
        finally:
            QCoreApplication.sendPostedEvents(None, QEvent.LayoutRequest)
            self._page.activate()
            if section is not None and offset is not None:
                bar.setValue(section.y() + offset)
            self._scrolling = was

    def add_section(self, section: Section) -> Section:
        self._sections[section.key] = section
        self._order.append(section.key)
        self._page.addWidget(section)
        return section

    def finish_sections(self) -> None:
        """Called once after the last section, so the page does not stretch them."""
        self._page.addStretch(1)

    def section(self, key: str) -> Section:
        return self._sections[key]

    def sections(self) -> list:
        return [self._sections[key] for key in self._order]

    # ── The view contract ────────────────────────────────────────────────────

    @classmethod
    def stage_keys(cls) -> list[str]:
        return [key for key, _ in cls.NAV]

    def current_stage(self) -> str:
        return self._stage_key

    def context(self) -> str:
        return self._context

    def set_context(self, text: str) -> None:
        self._context = text
        self.context_changed.emit(text)

    def go_to(self, key: str) -> None:
        """Move the reader to a section.  State first, then the scroll."""
        if key not in self._sections:
            raise KeyError(key)
        # Build the target and anything between here and there: those are about to be on
        # screen anyway, and a section growing above the target is what makes a scroll
        # land in the wrong place.
        self._build_through(key)
        previous, self._stage_key = self._stage_key, key
        if previous and previous in self._sections:
            self._sections[previous].set_active(False)
        target = self._sections[key]
        target.set_active(True)
        self.stage_changed.emit(key)
        self.context_changed.emit(self.context())
        # Everything above is final.  What follows is presentation.  Sections built just
        # now have only posted their layout requests, so their offsets are stale until the
        # requests are delivered; a scroll computed before that lands short.
        QCoreApplication.sendPostedEvents(None, QEvent.LayoutRequest)
        self._page.activate()
        self._scroll_to(key)
        # And once more after the event loop has settled any geometry the build above is
        # still propagating -- the scroll area learns its new range a turn later than the
        # page does.  Only if the reader is still on this section and has not moved.
        expected = self.scroll.verticalScrollBar().value()
        QTimer.singleShot(0, lambda: self._settle_scroll(key, expected))
        target.setFocus(Qt.OtherFocusReason)
        target.flash()

    def _scroll_to(self, key: str) -> None:
        """Put a section's first line at the top of the viewport.

        ensureWidgetVisible only makes the anchor *visible*, which going forward meant the
        new section's title sat at the bottom edge under the end of the previous one.  The
        anchor is a direct child of the section and the section a direct child of the page,
        so their offsets add up without mapping through the parent chain at all.
        """
        target = self._sections[key]
        anchor = target.anchor()
        top = target.y() + (anchor.y() if anchor.parentWidget() is target else 0)
        self._scrolling = True
        try:
            self.scroll.verticalScrollBar().setValue(max(0, top - 12))
        finally:
            self._scrolling = False

    def _settle_scroll(self, key: str, expected: int) -> None:
        try:
            if (self._stage_key == key
                    and self.scroll.verticalScrollBar().value() == expected):
                self._scroll_to(key)
        except RuntimeError:                    # the page closed before the timer fired
            pass

    def reset_scroll(self) -> None:
        self.scroll.verticalScrollBar().setValue(0)

    def _build_through(self, key: str) -> None:
        if key not in self._order:
            return
        target = self._order.index(key)
        current = self._order.index(self._stage_key) if self._stage_key in self._order else 0
        low, high = sorted((current, target))
        for position in range(low, high + 1):
            self._sections[self._order[position]].ensure_built()

    def _on_scrolled(self, _value: int) -> None:
        """Keep the navigator in step with free scrolling.

        Guarded against the scroll `go_to` performs itself: without the flag this re-emits
        `stage_changed` for whatever happens to be at the top mid-animation, and a shell
        handler that scrolled in response would oscillate.  The same flag also makes this
        non-re-entrant, which matters because `ensure_built` below grows the page and so
        can change the scrollbar's range while we are still reading it.
        """
        if self._scrolling or not self._order:
            return
        page = self.scroll.widget()
        if page is None:
            return
        self._scrolling = True
        try:
            self._track(page)
        finally:
            self._scrolling = False

    def _track(self, page: QWidget) -> None:
        # The viewport's top, in the page's own coordinates.  `mapFrom`, never `mapTo`:
        # the page is a *child* of the viewport, and QWidget.mapTo walks the parent chain
        # upwards, so viewport.mapTo(page, ...) never finds the page, runs off the top of
        # the hierarchy and dereferences null.  That was a hard access violation on the
        # first free scroll of either page -- no Python traceback, just "Segmentation
        # fault" -- and `go_to` hid it, because it sets _scrolling around its own scroll,
        # so navigating never took this path and nor did any test.
        viewport_top = page.mapFrom(self.scroll.viewport(), QPoint(0, 0)).y()
        reached = self._order[0]
        for key in self._order:
            section = self._sections[key]
            if section.y() <= viewport_top + 24:
                reached = key
            else:
                break
        self._sections[reached].ensure_built()
        if reached != self._stage_key:
            previous = self._stage_key
            self._stage_key = reached
            if previous and previous in self._sections:
                self._sections[previous].set_active(False)
            self._sections[reached].set_active(True)
            self.stage_changed.emit(reached)
