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
    QAbstractItemView, QAction, QButtonGroup, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QMenu, QPushButton, QRadioButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
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
    def __init__(self, title, build, parent=None, *, key: str | None = None,
                 start_open: bool = False, on_toggle=None):
        super().__init__(parent)
        self.build = build
        self.content = None
        self.title = title
        self.key = key
        self._on_toggle = on_toggle
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton('＋ ' + title)
        self.button.setCheckable(True)
        self.button.setAccessibleName(title)
        self.button.toggled.connect(self.toggle)
        self.body.addWidget(self.button, 0, Qt.AlignLeft)
        if start_open:
            # Restoring a reveal the reader had already opened before Back/Next rebuilt
            # the scene.  Builders are pure functions of lesson state, so re-running one
            # reproduces the same widget rather than a stale copy.
            self.button.setChecked(True)

    def toggle(self, checked):
        if checked and self.content is None:
            self.content = self.build()
            self.body.addWidget(self.content)
        if self.content is not None:
            self.content.setVisible(checked)
        self.button.setText(('− ' if checked else '＋ ') + self.title)
        if self._on_toggle is not None and self.key is not None:
            self._on_toggle(self.key, bool(checked))


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
#
# Everything below serves the guided lesson in `tensorscope_views`, and all of it is
# presentation only: no widget here computes a tensor, caches one, or writes through
# a view returned by `display_matrix`.
#
# `ValueStrip` is the one that carries a rule rather than a look.  Every numeric
# preview the lesson shows goes through it, and it refuses to render without a
# `source` string naming which captured array and which subset the numbers came from.
# That makes the spec's provenance requirement structural instead of editorial.

# Recorded answers to an optional understanding check.  Non-negative values are the
# index of the option the reader chose; these two are the other outcomes.
CHECK_REVEALED = -1
CHECK_SKIPPED = -2


def preview_number(value, digits: int = 5) -> str:
    """A short, honest rendering of one stored scalar.

    Shortest-round-trip formatting at the array's own precision, so nothing gains
    digits it does not have.  Very large or very small magnitudes switch to
    scientific notation -- the masked cells in a captured score tensor hold a number
    near -3.4e38, and rendering that positionally would fill the screen.  Exact
    values always remain available through the tensor inspector.
    """
    number = float(value)
    if number != number or number in (float('inf'), float('-inf')):
        return str(number)
    magnitude = abs(number)
    if magnitude and (magnitude >= 1e5 or magnitude < 1e-4):
        return np.format_float_scientific(number, precision=3, unique=True, trim='-')
    return np.format_float_positional(number, precision=digits, unique=True,
                                      fractional=True, trim='-')


class ValueStrip(QWidget):
    """A row of real captured numbers that always states where it came from.

    `source` is mandatory and appears under the numbers: which array, which phase,
    which position, and how much of it is shown.  A preview of eight numbers out of
    2560 that does not say so is the failure mode this class exists to prevent.
    """

    def __init__(self, values, *, source: str, title: str = '', color_key: str = 'layer',
                 highlight: int | None = None, labels=None, tokens: dict | None = None,
                 parent=None):
        super().__init__(parent)
        if not source:
            raise ValueError('ValueStrip needs a source: which array and which subset')
        palette = tokens or {}
        colour = COLORS.get(color_key, palette.get('accent', '#8888aa'))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        if title:
            heading = label(title, small=True)
            heading.setStyleSheet(f'color: {colour}; font-weight: 700;')
            outer.addWidget(heading)

        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.cells: list[QLabel] = []
        flat = list(np.asarray(values).reshape(-1))
        for index, value in enumerate(flat):
            cell = QLabel(preview_number(value))
            cell.setObjectName('valueCell')
            cell.setAlignment(Qt.AlignCenter)
            cell.setTextInteractionFlags(Qt.TextSelectableByMouse)
            emphasised = index == highlight
            cell.setStyleSheet(
                f'border: 1px solid {colour}; border-radius: 4px; padding: 5px 7px; '
                f'font-family: Consolas, monospace; font-size: 12px; '
                + (f'background: {colour}33; font-weight: 700;' if emphasised else ''))
            if labels is not None and index < len(labels):
                cell.setToolTip(str(labels[index]))
            row.addWidget(cell)
            self.cells.append(cell)
        row.addStretch(1)
        outer.addWidget(holder)
        outer.addWidget(label(source, muted=True, small=True))


class MultiplyVisual(QWidget):
    """The row-times-weights lesson, with captured endpoints and a schematic middle.

    The input row and the result row are real stored numbers.  The weight set between
    them is drawn as a labelled box carrying a *conceptual* badge, because a capture
    stores activations and not parameters.  No product is ever displayed: with the
    weights absent, any individual product would have to be invented.
    """

    def __init__(self, row_values, result_values, *, row_source: str, result_source: str,
                 row_title: str, result_title: str, matrix_label: str,
                 matrix_shape: tuple, row_color: str = 'norm', result_color: str = 'q',
                 pair: tuple[int, int] = (0, 0), note: str = '',
                 tokens: dict | None = None, parent=None):
        super().__init__(parent)
        from tensorscope_content import EVIDENCE_CONCEPTUAL as _conceptual
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        outer.addWidget(ValueStrip(row_values, source=row_source, title=row_title,
                                   color_key=row_color, highlight=pair[0], tokens=tokens))

        middle = QWidget()
        middle_row = QHBoxLayout(middle)
        middle_row.setContentsMargins(0, 0, 0, 0)
        middle_row.setSpacing(10)
        times = QLabel('×')
        times.setObjectName('shapeOp')
        middle_row.addWidget(times)
        box = QFrame()
        box.setObjectName('schematicBox')
        box.setStyleSheet(
            'QFrame#schematicBox { border: 1px dashed %s; border-radius: 6px; }'
            % COLORS.get('layer', '#94a3b8'))
        inner = QVBoxLayout(box)
        inner.setContentsMargins(12, 8, 12, 8)
        inner.setSpacing(3)
        inner.addWidget(label(matrix_label, small=True))
        inner.addWidget(label('[' + ' × '.join(str(d) for d in matrix_shape) + ']', muted=True,
                              small=True))
        inner.addWidget(EvidenceBadge(_conceptual, 'parameters not saved'))
        middle_row.addWidget(box)
        arrow = QLabel('=')
        arrow.setObjectName('shapeOp')
        middle_row.addWidget(arrow)
        middle_row.addStretch(1)
        outer.addWidget(middle)

        outer.addWidget(ValueStrip(result_values, source=result_source, title=result_title,
                                   color_key=result_color, highlight=pair[1], tokens=tokens))
        if note:
            outer.addWidget(label(note, muted=True, small=True))


class ValueBars(QWidget):
    """Aligned bars for one row of stored numbers, next to their exact values.

    Plain frames rather than matplotlib: this appears on the lesson's default path,
    where building a canvas per scene is the cost the disclosure ladder exists to
    avoid.  Masked entries are shown as masked instead of as an enormous bar; the
    test for "is this cell masked" is the same `finfo.min / 2` rule the attention
    table uses, so there is only one such rule in the codebase.
    """

    def __init__(self, values, labels, *, source: str, color_key: str = 'weights',
                 highlight: int | None = None, signed: bool = False,
                 tokens: dict | None = None, parent=None):
        super().__init__(parent)
        palette = tokens or {}
        colour = COLORS.get(color_key, palette.get('accent', '#8888aa'))
        muted = palette.get('text_muted', '#636370')
        flat = np.asarray(values).reshape(-1)
        floor = np.finfo(flat.dtype).min / 2 if flat.dtype.kind == 'f' else None
        eligible = flat if floor is None else flat[flat > floor]
        span = float(np.max(np.abs(eligible))) if eligible.size else 0.0
        low = float(np.min(eligible)) if eligible.size and signed else 0.0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(3)
        for index, value in enumerate(flat):
            number = float(value)
            masked = floor is not None and number <= floor
            line = QWidget()
            row = QHBoxLayout(line)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            name = QLabel(str(labels[index]) if index < len(labels) else str(index))
            name.setMinimumWidth(130)
            name.setMaximumWidth(130)
            name.setObjectName('journeySmall')
            name.setToolTip(str(labels[index]) if index < len(labels) else str(index))
            row.addWidget(name)

            track = QWidget()
            track.setMinimumHeight(14)
            track.setMaximumHeight(14)
            fill_row = QHBoxLayout(track)
            fill_row.setContentsMargins(0, 0, 0, 0)
            fill_row.setSpacing(0)
            if masked or span == 0.0:
                fraction = 0.0
            elif signed:
                fraction = (number - low) / (span - low) if span > low else 0.0
            else:
                fraction = abs(number) / span
            filled = max(0, min(1000, int(round(fraction * 1000))))
            bar = QFrame()
            emphasised = index == highlight
            bar.setStyleSheet(
                f'background: {colour}; border-radius: 3px;'
                if emphasised else f'background: {colour}88; border-radius: 3px;')
            fill_row.addWidget(bar, filled)
            rest = QFrame()
            rest.setStyleSheet('background: transparent;')
            fill_row.addWidget(rest, 1000 - filled)
            row.addWidget(track, 1)

            readout = QLabel('masked' if masked else preview_number(number))
            readout.setMinimumWidth(110)
            readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
            readout.setStyleSheet(
                'font-family: Consolas, monospace; font-size: 12px; '
                + (f'color: {muted};' if masked else
                   f'color: {colour}; font-weight: 700;' if emphasised else ''))
            if masked:
                readout.setToolTip(f'stored value {preview_number(number)} — a masked position')
            row.addWidget(readout)
            outer.addWidget(line)
        outer.addWidget(label(source, muted=True, small=True))


class CheckpointCard(QFrame):
    """One optional understanding check.

    Answering, revealing and skipping are all offered, and none of them touches
    navigation -- Back and Next live outside this widget entirely, so a wrong answer
    cannot block the lesson.  Every option carries its own explanation, so choosing
    the wrong one teaches instead of scoring.  A previously recorded answer is
    reflected when the scene is rebuilt.
    """

    def __init__(self, spec, facts: dict, answer=None, on_answer=None,
                 tokens: dict | None = None, parent=None):
        super().__init__(parent)
        from tensorscope_content import EVIDENCE_CONCEPTUAL as _conceptual
        self.spec = spec
        self._on_answer = on_answer
        palette = tokens or {}
        accent = palette.get('accent', '#3b82f6')
        self.setObjectName('checkpointCard')
        self.setStyleSheet(
            'QFrame#checkpointCard { border: 1px solid %s; border-radius: 8px; }' % accent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12)
        outer.setSpacing(8)
        heading = label('CHECK YOUR UNDERSTANDING  ·  OPTIONAL', small=True)
        heading.setStyleSheet(f'color: {accent}; font-weight: 700; letter-spacing: 0.6px;')
        outer.addWidget(heading)
        outer.addWidget(label(spec.question.format(**facts)))

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.options: list[QRadioButton] = []
        for index, text in enumerate(spec.options):
            button = QRadioButton(text.format(**facts))
            button.setObjectName('checkOption')
            self.group.addButton(button, index)
            outer.addWidget(button)
            self.options.append(button)
        self.group.idClicked.connect(self._chose)

        controls = QWidget()
        control_row = QHBoxLayout(controls)
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(8)
        self.reveal = QPushButton('Reveal the answer')
        self.reveal.setObjectName('secondaryBtn')
        self.reveal.clicked.connect(lambda: self._record(CHECK_REVEALED))
        self.skip = QPushButton('Skip this')
        self.skip.setObjectName('secondaryBtn')
        self.skip.clicked.connect(lambda: self._record(CHECK_SKIPPED))
        control_row.addWidget(self.reveal)
        control_row.addWidget(self.skip)
        control_row.addStretch(1)
        outer.addWidget(controls)

        self.explanation = label('', rich=True)
        self.explanation.setObjectName('checkExplanation')
        self.explanation.setVisible(False)
        outer.addWidget(self.explanation)
        self.footnote = label(
            'Nothing here is graded, and Next works whether you answer or not.',
            muted=True, small=True)
        outer.addWidget(self.footnote)
        self._facts = facts
        if answer is not None:
            self._apply(answer)

    def _chose(self, index: int) -> None:
        self._record(index)

    def _record(self, value: int) -> None:
        self._apply(value)
        if self._on_answer is not None:
            self._on_answer(self.spec.key, value)

    def _apply(self, value: int) -> None:
        """Render the outcome for a recorded answer.  Pure display; changes no state."""
        spec, facts = self.spec, self._facts
        correct = spec.options[spec.correct].format(**facts)
        if value == CHECK_SKIPPED:
            text = ('Skipped. You can answer it any time — nothing in the lesson is '
                    'locked behind it.')
        elif value == CHECK_REVEALED:
            text = (f'<b>The answer is:</b> {correct}<br><br>'
                    + spec.explain[spec.correct].format(**facts))
        else:
            if 0 <= value < len(self.options):
                self.options[value].setChecked(True)
            body = spec.explain[value].format(**facts)
            if value == spec.correct:
                text = f'<b>Correct.</b><br><br>{body}'
            else:
                text = (f'<b>Not this one.</b> {body}<br><br>'
                        f'<b>The answer is:</b> {correct}')
        self.explanation.setText(text)
        self.explanation.setVisible(True)


class StepProgress(QWidget):
    """`Step 5 of 13` plus a thin segmented bar, in one fixed place."""

    def __init__(self, total: int, tokens: dict | None = None, parent=None):
        super().__init__(parent)
        palette = tokens or {}
        self._accent = palette.get('accent', '#3b82f6')
        self._idle = palette.get('border', '#2e2e33')
        self._total = max(1, total)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(5)
        self.caption = label('', muted=True, small=True)
        outer.addWidget(self.caption)
        track = QWidget()
        row = QHBoxLayout(track)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        self.segments: list[QFrame] = []
        for _ in range(self._total):
            segment = QFrame()
            segment.setMinimumHeight(4)
            segment.setMaximumHeight(4)
            row.addWidget(segment, 1)
            self.segments.append(segment)
        outer.addWidget(track)
        self.set_current(0)

    def set_current(self, index: int) -> None:
        index = max(0, min(index, self._total - 1))
        self.caption.setText(f'Step {index + 1} of {self._total}')
        for position, segment in enumerate(self.segments):
            colour = self._accent if position <= index else self._idle
            segment.setStyleSheet(f'background: {colour}; border-radius: 2px;')


class LessonRibbon(QFrame):
    """The compact strip that keeps the real prompt and the current focus on screen.

    One line, so it costs no vertical room the card needs.  The prompt is the run's
    own stored prompt text; the focus string is whatever the current scene is looking
    at (a position, a layer, a head, a query/key pair).
    """

    def __init__(self, prompt: str, tokens: dict | None = None, parent=None):
        super().__init__(parent)
        palette = tokens or {}
        self.setObjectName('lessonRibbon')
        self.setStyleSheet(
            'QFrame#lessonRibbon { background: %s; border: 1px solid %s; border-radius: 6px; }'
            % (palette.get('card_bg', '#1c1c20'), palette.get('border', '#2e2e33')))
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 7, 12, 7)
        row.setSpacing(10)
        tag = label('PROMPT', small=True)
        tag.setStyleSheet(f"color: {palette.get('text_muted', '#636370')}; font-weight: 700;")
        row.addWidget(tag)
        shown = ' '.join((prompt or '').split()) or '(no prompt text stored)'
        self.prompt_label = QLabel(shown if len(shown) <= 90 else shown[:89] + '…')
        self.prompt_label.setObjectName('ribbonPrompt')
        self.prompt_label.setToolTip(prompt or '')
        self.prompt_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.prompt_label.setStyleSheet('font-weight: 600;')
        row.addWidget(self.prompt_label)
        row.addStretch(1)
        self.focus_label = label('', muted=True, small=True)
        self.focus_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.focus_label)

    def set_focus(self, text: str) -> None:
        self.focus_label.setText(text)


class ContentsButton(QPushButton):
    """One small button opening a menu of scenes.

    Deliberately a menu rather than a panel: the lesson removed the persistent stage
    sidebar, and replacing it with another permanent list on the other side would
    reintroduce exactly what was removed.
    """

    def __init__(self, entries, on_pick, parent=None):
        super().__init__('Contents', parent)
        self.setObjectName('secondaryBtn')
        self._menu = QMenu(self)
        self._actions: dict[str, QAction] = {}
        for position, (key, name) in enumerate(entries):
            action = self._menu.addAction(f'{position + 1}.  {name}')
            action.setCheckable(True)
            action.triggered.connect(lambda _=False, k=key: on_pick(k))
            self._actions[key] = action
        self.setMenu(self._menu)

    def set_current(self, key: str) -> None:
        for existing, action in self._actions.items():
            action.setChecked(existing == key)
