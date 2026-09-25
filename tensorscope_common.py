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

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from tensorscope_content import (
    EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED, EVIDENCE_KINDS, EVIDENCE_OBSERVED,
    EVIDENCE_UNATTESTED, LAYER_STEP_ORDER,
)


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
    """
    def __init__(self, title, build, parent=None):
        super().__init__(parent)
        self.build = build
        self.content = None
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton('＋ ' + title)
        self.button.setCheckable(True)
        self.button.setAccessibleName(title)
        self.button.toggled.connect(self.toggle)
        self.body.addWidget(self.button, 0, Qt.AlignLeft)
        self.title = title

    def toggle(self, checked):
        if checked and self.content is None:
            self.content = self.build()
            self.body.addWidget(self.content)
        if self.content is not None:
            self.content.setVisible(checked)
        self.button.setText(('− ' if checked else '＋ ') + self.title)


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
