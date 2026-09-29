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
import os

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize, Qt, QVariantAnimation, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from tensor_widgets import exact_scalar
from tensorscope_content import (
    EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED, EVIDENCE_KINDS, EVIDENCE_OBSERVED,
    EVIDENCE_UNATTESTED, LAYER_STEP_ORDER, LESSON_TIMING_CAVEAT, readable_spelling,
)


# Qt's offscreen platform has no compositor, so an animation there would only burn time
# in the test suite.  Reveals are never gated on an animation finishing, so skipping it
# changes nothing a reader or a test can observe except the tint.
OFFSCREEN = os.environ.get('QT_QPA_PLATFORM', '') == 'offscreen'


# The six operations inside one decoder layer, in execution order.  Shared by the
# internals sidebar and the card builders.
LAYER_STEPS = [('input', '1  Normalize'), ('qkv', '2  Project Q / K / V'),
               ('prepare', '3  Prepare for attention'), ('scores', '4  Scores → weights'),
               ('mix', '5  Mix values & project'), ('output', '6  Finish the layer')]

assert [key for key, _ in LAYER_STEPS] == LAYER_STEP_ORDER, \
    "LAYER_STEPS and the INTERNALS_STEPS copy have drifted apart"

COLORS = {'q': '#60a5fa', 'k': '#c084fc', 'v': '#34d399',
          'scores': '#fbbf24', 'weights': '#22d3ee', 'output': '#fb923c',
          'norm': '#a78bfa', 'layer': '#94a3b8'}

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

def label(text, *, rich=False, muted=False, title=False, small=False):
    result = QLabel(text)
    result.setTextFormat(Qt.RichText if rich else Qt.PlainText)
    result.setWordWrap(True)
    result.setTextInteractionFlags(Qt.TextSelectableByMouse)
    name = 'journeyTitle' if title else 'journeyMuted' if muted else 'journeySmall' if small else 'journeyText'
    result.setObjectName(name)
    return result


def shape(array):
    return '[' + ' × '.join(str(d) for d in array.shape) + ']'


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
    """Horizontal clickable pipeline showing the stages of the active view.

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
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
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
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
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
        self.button.toggled.connect(self.toggle)
        self.body.addWidget(self.button, 0, Qt.AlignLeft)
        self.title = title
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
    """Coloured badge showing which captured phase is being viewed.

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
                    highlight: int | None = None, highlight_color: str = '#0e9f6e'):
    """A small non-editable table.  Used for candidate scores and provenance."""
    table = QTableWidget(len(rows), len(headings))
    table.setHorizontalHeaderLabels(list(headings))
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            item = QTableWidgetItem(str(value))
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


# ── Layer step sidebar ────────────────────────────────────────────────────────

class LayerStepSidebar(QWidget):
    """Vertical step navigator for the operations inside one decoder layer."""

    def __init__(self, on_step, parent=None):
        super().__init__(parent)
        self._on_step = on_step
        self._buttons: dict[str, QPushButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = label('LAYER OPERATIONS', muted=True, small=True)
        header.setObjectName('stepSidebarHeader')
        layout.addWidget(header)
        for key, title in LAYER_STEPS:
            button = QPushButton(title)
            button.setObjectName('layerStepBtn')
            button.setCheckable(True)
            button.clicked.connect(lambda _, k=key: on_step(k))
            layout.addWidget(button)
            self._buttons[key] = button
        layout.addStretch(1)

    def set_current(self, key: str) -> None:
        for existing, button in self._buttons.items():
            button.setChecked(existing == key)


# ── Lesson primitives ─────────────────────────────────────────────────────────
# Each of these takes an optional `state` dict plus a `state_key`, and the contract is
# the same everywhere: restore from `state` synchronously while constructing, and write
# back on every change.  The owner keeps the dict, which is what lets the lesson cache a
# screen, rebuild it when the phase changes, and still show the reader what they had.

class StepReveal(QWidget):
    """Reveal one calculation a deliberate step at a time.

    A step's widget is built the first time it is revealed, the same laziness `Disclosure`
    uses.  The invariant the tests rely on: the reveal, its child widget and `revealed()`
    are all final before `reveal_next()` returns, so nothing ever waits on a timer.  The
    tint animation is decoration on top of an already-finished state change.
    """

    ANIMATION_MS = 320

    def __init__(self, steps, *, state: dict | None = None, state_key: str = '',
                 caption: str = LESSON_TIMING_CAVEAT, animate: bool = True, parent=None):
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

        self.reveal_to(int(self._state.get(self._state_key) or 0))

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

    Values are formatted with `exact_scalar`, so a number here reads identically to the
    same cell in `TensorInspector`.  This widget never slices an array itself: the caller
    passes exactly the values it means to show, which keeps the sampling decision visible
    in the screen code rather than hidden in a widget.
    """

    MAX_BOXES = 8
    COLUMNS = 4

    def __init__(self, values, *, labels=None, indices=None, caption: str = '',
                 kind: str = EVIDENCE_OBSERVED, note: str = '', total: int | None = None,
                 formatter=None, parent=None):
        super().__init__(parent)
        everything = list(values)
        shown = everything[:self.MAX_BOXES]
        names = list(labels) if labels is not None else [
            f'index {i}' for i in (list(indices) if indices is not None else range(len(shown)))]
        render = formatter or exact_scalar

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(4)
        if caption:
            outer.addWidget(label(caption, muted=True, small=True))
        grid_holder = QWidget()
        grid = QGridLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        grid.addWidget(EvidenceBadge(kind, note), 0, 0, Qt.AlignLeft | Qt.AlignVCenter)
        for position, value in enumerate(shown):
            cell = position + 1
            grid.addWidget(self._box(names[position] if position < len(names) else '', render(value)),
                           cell // self.COLUMNS, cell % self.COLUMNS)
        grid.setColumnStretch(self.COLUMNS, 1)
        outer.addWidget(grid_holder)
        hidden = (total if total is not None else len(everything)) - len(shown)
        if hidden > 0:
            outer.addWidget(label(f'… {hidden} more in the saved tensor', muted=True, small=True))

    def _box(self, name: str, text: str) -> QWidget:
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

    def __init__(self, tokens, response: str, *, selected: int = 0, parent=None):
        super().__init__(parent)
        self._tokens = list(tokens)
        self._response = response or ''
        self._current = max(0, min(int(selected), max(0, len(self._tokens) - 1)))
        self._faithful = bool(self._tokens) and ''.join(self._tokens) == self._response

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
        for index, token in enumerate(self._tokens):
            # Verbatim: these are decoded text, not BPE spellings.  See the class docstring.
            text = html.escape(token).replace('\n', '<br>') or '&nbsp;'
            style = (f"text-decoration: none; border-bottom: 1px dotted {CHROME['muted']}; "
                     'color: inherit;')
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
    """Where the reader is in whatever list of stages the active view declares.

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
