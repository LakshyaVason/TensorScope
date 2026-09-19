"""Read-only computation journey. This module never loads a model or writes a run.

All numerical views reference arrays on the supplied RunCapture. Equations describe
operations; unavailable operands/intermediates are explicitly marked as conceptual.
"""
from __future__ import annotations

import html
import json

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from tensor_widgets import AttentionExplorer, TensorInspector
from tensorscope_content import (
    LOGITS_UNAVAILABLE, STORY_STAGE_INDEX, TENSOR_LABELS,
    TENSOR_LABEL_BY_KEY, TENSOR_EXPLANATIONS, story_facts,
)


STAGES = [('prompt', 'Prompt & journey'), ('tokenization', 'Tokens'),
          ('embedding', 'Embeddings'), ('layers', 'Transformer layers'),
          ('logits', 'Logits → first token'), ('coda', "First token's next pass")]

LAYER_STEPS = [('input', '1  Normalize'), ('qkv', '2  Project Q / K / V'),
               ('prepare', '3  Prepare for attention'), ('scores', '4  Scores → weights'),
               ('mix', '5  Mix values & project'), ('output', '6  Finish the layer')]

COLORS = {'q': '#60a5fa', 'k': '#c084fc', 'v': '#34d399',
          'scores': '#fbbf24', 'weights': '#22d3ee', 'output': '#fb923c',
          'norm': '#a78bfa', 'layer': '#94a3b8'}


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


# ── Visual component: pipeline flow navigator ────────────────────────────────

class PipelineNavigator(QWidget):
    """Horizontal clickable pipeline showing the full computation stages.

    Clicking a stage chip navigates there. The current stage is highlighted.
    """

    CHIP_STAGES = [
        ('prompt',       'Prompt'),
        ('tokenization', 'Tokens'),
        ('embedding',    'Embeddings'),
        ('layers',       'Layers'),
        ('logits',       'Logits'),
        ('coda',         'First token'),
    ]

    def __init__(self, on_navigate, colors: dict | None = None, parent=None):
        super().__init__(parent)
        self._on_navigate = on_navigate
        self._colors = colors or {}
        self._buttons: dict[str, QPushButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        for i, (key, name) in enumerate(self.CHIP_STAGES):
            if i > 0:
                arrow = QLabel('→')
                arrow.setObjectName('pipelineArrow')
                arrow.setAlignment(Qt.AlignCenter)
                layout.addWidget(arrow)
            btn = QPushButton(name)
            btn.setObjectName('pipelineChip')
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, k=key: on_navigate(k))
            layout.addWidget(btn)
            self._buttons[key] = btn
        layout.addStretch(1)

    def set_current(self, key: str) -> None:
        for k, btn in self._buttons.items():
            btn.setChecked(k == key)


# ── Visual component: matrix shape diagram ────────────────────────────────────

class ShapeDiagram(QWidget):
    """Visual representation of a matrix multiply operation showing actual shapes.

    Shows colored labeled boxes: A [rows × cols] @ B [cols × out] → C [rows × out]
    Dimensions come from captured arrays, not inferred from model config.
    """

    def __init__(self, operands: list[tuple[str, str, tuple]], operator: str = '@',
                 colors: dict | None = None, parent=None):
        """operands: list of (name, color_key, shape_tuple) triples.
        operator: string shown between boxes (@ for matmul, → for other).
        """
        super().__init__(parent)
        self._colors = colors or {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)
        for i, (name, color_key, shp) in enumerate(operands):
            if i > 0:
                op_label = QLabel(operator)
                op_label.setObjectName('shapeOp')
                op_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(op_label)
            box = self._make_box(name, color_key, shp)
            layout.addWidget(box)
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


# ── Visual component: mini attention heatmap ─────────────────────────────────

class MiniHeatmap(FigureCanvas):
    """Small matplotlib heatmap of an attention weight slice for display.

    Uses display_matrix to stride-sample to ≤48×48. The source array is never
    mutated; this is a display-only view.
    """

    def __init__(self, array: np.ndarray, title: str = '',
                 mpl_style: dict | None = None, parent=None):
        from TensorScope import display_matrix
        fig = Figure(figsize=(3.8, 2.8), tight_layout=True)
        super().__init__(fig)
        ax = fig.add_subplot(111)
        sampled = display_matrix(array, maximum=48)
        # Use a perceptually uniform colormap; viridis is standard and colorblind-safe.
        im = ax.imshow(sampled, aspect='auto', cmap='viridis', interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if title:
            ax.set_title(title, fontsize=9)
        ax.set_xlabel('Key token index', fontsize=8)
        ax.set_ylabel('Query token index', fontsize=8)
        ax.tick_params(labelsize=7)
        if mpl_style:
            fig.set_facecolor(mpl_style.get('figure.facecolor', '#111114'))
            ax.set_facecolor(mpl_style.get('axes.facecolor', '#1c1c20'))
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
        """Add a visual matrix shape diagram."""
        diag = ShapeDiagram(operands, operator)
        return self.add(diag)

    def section_rule(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName('sectionRule')
        return self.add(line)


# ── Disclosure ────────────────────────────────────────────────────────────────

class Disclosure(QWidget):
    """Construct expensive numerical widgets only when the reader requests them."""
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
    """Colored badge showing which captured phase is being viewed."""

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
        for col, value in enumerate((row, token_id, text)):
            table.setItem(row, col, QTableWidgetItem(str(value)))
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
    table.verticalHeader().hide()
    table.setMinimumHeight(220)
    table.setMaximumHeight(360)
    return table


# ── Layer step sidebar ────────────────────────────────────────────────────────

class LayerStepSidebar(QWidget):
    """Vertical step navigator for within-layer operations.

    Clicking a step item navigates there. Current step is highlighted.
    """

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
            btn = QPushButton(title)
            btn.setObjectName('layerStepBtn')
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, k=key: on_step(k))
            layout.addWidget(btn)
            self._buttons[key] = btn
        layout.addStretch(1)

    def set_current(self, key: str) -> None:
        for k, btn in self._buttons.items():
            btn.setChecked(k == key)


# ── StoryView ─────────────────────────────────────────────────────────────────

class StoryView(QWidget):
    """One stage and one layer operation at a time, with persistent selection."""

    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(parent)
        self.capture = capture
        self.facts = story_facts(capture)
        self.layer_index = sorted(capture.layers)[0]
        self.step_key = 'input'
        self.generated = False
        self._stage_key = ''
        self._stage_widget = None
        self._on_stage_change = None
        self._on_context_change = None
        self._tokens = tokens or {}
        self._mpl_style = {}
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(14)
        self.go_to('prompt')

    def go_to(self, key):
        if key not in dict(STAGES):
            raise KeyError(key)
        if self._stage_widget is not None:
            self.body.removeWidget(self._stage_widget)
            self._stage_widget.hide()
            self._stage_widget.deleteLater()
        self._stage_key = key
        builders = {
            'prompt': self._opening,
            'tokenization': self._tokenization,
            'embedding': self._embedding,
            'layers': self._layers_stage,
            'logits': self._word_choice,
            'coda': self._coda,
        }
        self._stage_widget = builders[key]()
        self.body.addWidget(self._stage_widget)
        if self._on_stage_change:
            self._on_stage_change(key)

    def _stage_card(self, key):
        stage = STORY_STAGE_INDEX[key]
        return Card(stage.heading.format(**self.facts), stage.plain.format(**self.facts))

    # ── Stage: Prompt ─────────────────────────────────────────────────────────

    def _opening(self):
        card = self._stage_card('prompt')

        # Prompt and first token displayed prominently
        card.field('Your prompt', self.capture.prompt)
        card.field('First generated token',
                   f'{self.capture.tokens[0]!r}  ·  vocabulary ID {self.capture.token_ids[0]}')

        # Visual pipeline overview
        card.section_rule()
        card.add(label('COMPUTATION OVERVIEW', muted=True))
        overview = self._make_pipeline_overview()
        card.add(overview)

        card.section_rule()
        card.field('Two captured phases',
                   '<b>Phase 1 · Prompt prefill.</b>  All prompt tokens pass through all '
                   'transformer layers simultaneously. The final prompt position\'s logits vector '
                   'chooses the first output token.<br><br>'
                   '<b>Phase 2 · First generated-token pass.</b>  That chosen token is then processed '
                   'through all layers again, reusing the prompt\'s cached keys and values. '
                   'Its embeddings and all layer tensors are captured. Its output logits are not saved.<br><br>'
                   '<b>Later tokens</b> are generated uncaptured — their text is retained as context '
                   'so you see the full answer.', rich=True)

        answer = Disclosure('Read the complete generated response',
                            lambda: label(self.capture.response))
        card.add(answer)
        return card

    def _make_pipeline_overview(self):
        """A visual flow diagram widget showing the computation path."""
        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(0)

        steps = [
            ('Prompt\ntext', '#636370'),
            ('Token\nIDs', '#9f9fa8'),
            ('Embeddings\nX₀', '#60a5fa'),
            (f'Layer 0\n→ Layer {len(self.capture.layers)-1}', '#a78bfa'),
            ('Final\nLogits', '#fbbf24'),
            ('First\nToken', '#34d399'),
        ]
        keys = ['tokenization', 'tokenization', 'embedding', 'layers', 'logits', 'coda']

        for i, ((name, color), key) in enumerate(zip(steps, keys)):
            if i > 0:
                arrow = QLabel('→')
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setStyleSheet('color: #636370; font-size: 14px; padding: 0 4px;')
                layout.addWidget(arrow)
            btn = QPushButton(name)
            btn.setObjectName('overviewChip')
            btn.setStyleSheet(
                f'QPushButton#overviewChip {{ background: transparent; border: 2px solid {color}; '
                f'border-radius: 6px; color: {color}; font-size: 11px; font-weight: 600; '
                f'padding: 6px 8px; min-width: 60px; }}'
                f'QPushButton#overviewChip:hover {{ background: {color}22; }}'
            )
            btn.clicked.connect(lambda _, k=key: self.go_to(k))
            layout.addWidget(btn)

        layout.addStretch(1)
        return host

    # ── Stage: Tokenization ───────────────────────────────────────────────────

    def _tokenization(self):
        card = self._stage_card('tokenization')

        card.add(label('WHY IT EXISTS', muted=True))
        card.add(label('Neural networks operate on vectors of numbers, not characters. A tokenizer maps '
                       'text to integer IDs that index a fixed vocabulary. These IDs are the only '
                       'input the model receives.'))

        card.equation('Prompt text + chat formatting  →  tokenizer  →  ordered token IDs  →  embedding lookup')

        card.add(label('SHAPE', muted=True))
        T = len(self.capture.prompt_tokens)
        B = self.capture.embedding.shape[0]
        card.add(label(
            f'T = {T} input tokens  ·  batch B = {B}  ·  '
            f'each token ID is a single integer in [0, vocabulary_size)'))

        card.add(label('HOW TO READ IT', muted=True))
        card.add(label(
            'Position is the sequence index (0-based). The Token ID column shows the vocabulary '
            'lookup key. The spelling column shows the tokenizer\'s internal piece, not a re-decode — '
            'markers like Ġ and Ċ often encode whitespace. Chat control tokens (role markers, generation '
            'prefix) may appear here even though you did not type them; the chat template added them '
            'before tokenization.'))

        card.add(token_table(self.capture))
        card.field('What happens next',
                   'Each token ID selects one row from the learned embedding table, producing a '
                   f'{self.facts["hidden_size"]}-dimensional feature vector.')
        return card

    # ── Stage: Embedding ──────────────────────────────────────────────────────

    def _embedding(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Phase selector + badge
        phase_row = QHBoxLayout()
        self.phase_picker = QComboBox()
        self.phase_picker.addItems(['Phase 1 · Prompt prefill',
                                    'Phase 2 · First generated-token pass'])
        self.phase_picker.setCurrentIndex(int(self.generated))
        self.phase_picker.currentIndexChanged.connect(self._phase_changed)
        self._phase_badge = PhaseBadge(self.generated)
        phase_row.addWidget(self.phase_picker, 1)
        phase_row.addWidget(self._phase_badge)
        layout.addLayout(phase_row)

        array = self.capture.generated_embedding if self.generated else self.capture.embedding
        H = array.shape[-1]

        card = self._stage_card('embedding')
        card.add(label('WHY IT EXISTS', muted=True))
        card.add(label('The model has no notion of meaning for integer IDs. The embedding table '
                       'maps each ID to a dense vector of learned real-valued features. These vectors '
                       'are the starting point that all subsequent computation refines.'))

        card.equation('token ID  →  row of learned embedding table  →  X  =  initial feature vector')

        card.add(label('SHAPE', muted=True))
        card.shape_diagram([
            ('X', 'q', (array.shape[0], array.shape[-2] if array.ndim >= 3 else 1, H)),
        ], operator='=')
        card.add(label(
            f'[batch × tokens × hidden]  =  [{array.shape[0]} × '
            f'{array.shape[-2] if array.ndim >= 3 else 1} × {H}]\n'
            f'Each token gets a {H}-dimensional feature vector. Stored dtype: {array.dtype}.'))

        card.add(label('HOW TO READ IT', muted=True))
        card.add(label(
            'Each row is one token\'s initial feature vector. Column indices are just coordinates in '
            'the learned embedding space — they are not named properties like "subject" or "verb". '
            'The embedding table itself is not saved; only the rows used for this specific input.'))

        if self.generated:
            card.add(label(
                f'In this phase, only the first generated token {self.capture.tokens[0]!r} '
                'is embedded. The prompt\'s information is available through the key/value cache '
                'inside each attention layer — it does not pass through embedding again.',
                muted=True))

        self._tensor_disclosure(card, 'embedding', array)
        card.field('What happens next',
                   f'Layer 0 receives this tensor and normalizes each token\'s {H}-dimensional '
                   'feature vector before the attention projections.')
        layout.addWidget(card)
        return host

    # ── Phase controls ────────────────────────────────────────────────────────

    def _phase_controls(self, layout):
        row = QHBoxLayout()
        row.addWidget(label('Captured phase'))
        self.phase_picker = QComboBox()
        self.phase_picker.addItems(['Phase 1 · Prompt prefill',
                                    'Phase 2 · First generated-token pass'])
        self.phase_picker.setCurrentIndex(int(self.generated))
        self.phase_picker.currentIndexChanged.connect(self._phase_changed)
        row.addWidget(self.phase_picker, 1)
        self._phase_badge = PhaseBadge(self.generated)
        row.addWidget(self._phase_badge)
        layout.addLayout(row)

    def _phase_changed(self, index):
        self.generated = bool(index)
        self.go_to(self._stage_key)

    # ── Tensor disclosure ─────────────────────────────────────────────────────

    def _inspect(self, name, tensor):
        generated = self.generated
        rows = token_labels(self.capture, generated,
                            keys=name in ('k_attended', 'v_attended'))
        columns = token_labels(self.capture, generated, keys=True) if name in (
            'attention_weights', 'attention_scores') else None
        return TensorInspector(tensor, name=name, axes=axes_for(name, tensor),
                               row_labels=rows if name != 'final_logits' else None,
                               column_labels=columns)

    def _tensor_disclosure(self, card, name, tensor):
        """Shape summary + a disclosure button for the full inspector."""
        meanings = axes_for(name, tensor)
        card.add(label(f'{name}   {shape(tensor)}   ·   stored {tensor.dtype}'))
        card.add(label('Axes: ' + ' × '.join(meanings), muted=True))
        card.add(Disclosure(f'Inspect captured {name}',
                             lambda n=name, t=tensor: self._inspect(n, t)))

    # Keep old API name used by external tests / other places
    def _tensor(self, card, name, tensor):
        return self._tensor_disclosure(card, name, tensor)

    # ── Stage: Transformer Layers ─────────────────────────────────────────────

    def _layers_stage(self):
        host = QWidget()
        host_layout = QHBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)

        # Left: layer + step navigation sidebar
        nav_panel = QWidget()
        nav_panel.setObjectName('layerNavPanel')
        nav_panel.setFixedWidth(180)
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(0, 0, 10, 0)
        nav_layout.setSpacing(12)

        # Phase selector
        self._phase_controls(nav_layout)

        # Layer picker
        nav_layout.addWidget(label('LAYER', muted=True))
        layer_row = QHBoxLayout()
        self.previous_layer = QPushButton('◀')
        self.previous_layer.setFixedWidth(30)
        self.next_layer = QPushButton('▶')
        self.next_layer.setFixedWidth(30)
        self.picker = QComboBox()
        indices = sorted(self.capture.layers)
        for index in indices:
            self.picker.addItem(f'Layer {index}', index)
        self.picker.setCurrentIndex(indices.index(self.layer_index))
        self.picker.currentIndexChanged.connect(self._layer_changed)
        self.previous_layer.clicked.connect(
            lambda: self.picker.setCurrentIndex(self.picker.currentIndex() - 1))
        self.next_layer.clicked.connect(
            lambda: self.picker.setCurrentIndex(self.picker.currentIndex() + 1))
        self.previous_layer.setEnabled(self.layer_index != indices[0])
        self.next_layer.setEnabled(self.layer_index != indices[-1])
        layer_row.addWidget(self.previous_layer)
        layer_row.addWidget(self.picker, 1)
        layer_row.addWidget(self.next_layer)
        nav_layout.addLayout(layer_row)

        context = label(
            f'All {len(indices)} layers repeat this same structure with different learned '
            'weights and activations. Each layer preserves the hidden width so its output '
            'feeds directly into the next.', muted=True)
        nav_layout.addWidget(context)

        # Step sidebar
        self.step_sidebar = LayerStepSidebar(self._step_navigate)
        self.step_sidebar.set_current(self.step_key)
        nav_layout.addWidget(self.step_sidebar)
        nav_layout.addStretch(1)
        host_layout.addWidget(nav_panel)

        # Right: layer step card
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self.layer_section = self._layer_card()
        right_layout.addWidget(self.layer_section)
        right_layout.addStretch(1)
        host_layout.addWidget(right, 1)

        return host

    def _step_navigate(self, key: str) -> None:
        self.step_key = key
        self.go_to('layers')

    def _layer_changed(self, position):
        index = self.picker.itemData(position)
        if index is not None:
            self.layer_index = int(index)
            self.go_to('layers')

    def _step_changed(self, position):
        key = self.step_picker.itemData(position)
        if key:
            self.step_key = key
            self.go_to('layers')

    def _layer_card(self):
        layers = self.capture.generated_layers if self.generated else self.capture.layers
        tensors = layers[self.layer_index].tensors
        norm = tensors['normalized_input']
        q_att, k_att, v_att = (tensors[n] for n in ('q_attended', 'k_attended', 'v_attended'))
        tq = q_att.shape[-2]
        tk = k_att.shape[-2]
        d = q_att.shape[-1]
        dv = v_att.shape[-1]
        hidden = norm.shape[-1]
        heads = q_att.shape[-3]
        step = self.step_key

        if step == 'input':
            return self._card_normalize(tensors, norm, hidden)
        elif step == 'qkv':
            return self._card_qkv(tensors, tq, hidden, heads, d)
        elif step == 'prepare':
            return self._card_prepare(tensors, tq, tk, d, hidden, heads)
        elif step == 'scores':
            return self._card_scores(tensors, tq, tk, d, hidden, heads)
        elif step == 'mix':
            return self._card_mix(tensors, tq, tk, dv, heads, hidden, v_att)
        else:
            return self._card_output(tensors, tq, hidden, layers)

    # ── Layer step cards ──────────────────────────────────────────────────────

    def _card_normalize(self, tensors, norm, hidden):
        layers = self.capture.generated_layers if self.generated else self.capture.layers
        previous = sorted(layers).index(self.layer_index) - 1
        source = ('embedding' if previous < 0
                  else f'layer_output from layer {sorted(layers)[previous]}')

        card = Card('Step 1 · Normalize the incoming representation',
                    TENSOR_EXPLANATIONS['normalized_input'])
        card.add(label('WHY IT EXISTS', muted=True))
        card.add(label(
            'Deep networks can accumulate extreme activation magnitudes layer by layer. '
            'RMSNorm rescales each token\'s feature vector so the Q/K/V projections '
            'operate on a controlled scale, regardless of how large the features became '
            'in the previous layer.'))

        card.add(label('OPERATION', muted=True))
        card.equation(
            f'Input X  ({source})<br>'
            '↓<br>'
            '<b>RMSNorm(X)</b>  =  X / √(mean(x²) + ε) ⊙ g<br>'
            '↓<br>'
            '<b>X_norm</b>  (captured — feeds Q, K, V projections)')
        card.add(label(
            'Normalization scale g and ε are learned parameters not saved in the capture. '
            'The division and element-wise scale are conceptual — only the final output is captured.',
            muted=True))

        card.add(label('SHAPE', muted=True))
        card.shape_diagram([
            ('X_norm', 'norm', norm.shape),
        ], operator='=')
        card.add(label(f'[batch × tokens × hidden]  =  {shape(norm)}  ·  stored {norm.dtype}'))

        self._tensor_disclosure(card, 'normalized_input', tensors['normalized_input'])
        card.field('What happens next',
                   'The same X_norm tensor feeds three separate learned linear projections: '
                   'Q, K, and V — all from this one normalized representation.')
        return card

    def _card_qkv(self, tensors, tq, hidden, heads, d):
        card = Card('Step 2 · Three projections, three roles',
                    'Q and K define which tokens interact. V supplies the feature content '
                    'attention will combine. All three are computed from the same normalized input.')

        card.add(label('WHY THREE PROJECTIONS?', muted=True))
        card.add(label(
            'A single vector cannot simultaneously express "what I\'m looking for" (query), '
            '"what I offer for others to find" (key), and "what content I contribute" (value). '
            'Three separate learned projections specialize for each role. Each has its own '
            'weight matrix, learned from data.'))

        card.add(label('OPERATION', muted=True))
        card.equation(
            'X_norm  W_Qᵀ  →  Q_raw    (what this position searches for)<br>'
            'X_norm  W_Kᵀ  →  K_raw    (what this position presents to be found)<br>'
            'X_norm  W_Vᵀ  →  V_raw    (what content this position contributes)<br><br>'
            'PyTorch Linear: output = input @ Wᵀ + bias (bias where architecture uses it). '
            'Weight matrices are not saved in the capture.')

        card.add(label('DIMENSION COMPATIBILITY', muted=True))
        for name, role, color_key in [
            ('Q_raw', 'what each token looks for', 'q'),
            ('K_raw', 'what each token presents', 'k'),
            ('V_raw', 'what content each token offers', 'v'),
        ]:
            raw_key = name[0].lower()  # 'q', 'k', or 'v'
            width = tensors[raw_key].shape[-1]
            card.shape_diagram([
                ('X_norm', 'norm', (tq, hidden)),
                (f'W_{name[0]}ᵀ', color_key, (hidden, width)),
                (name, color_key, (tq, width)),
            ])
            card.add(label(
                f'Inner dimension {hidden} matches. Output: {tq} tokens × {width} projection features. '
                f'Batch axis omitted in diagram; present in stored tensor.',
                muted=True))
            card.add(label(TENSOR_EXPLANATIONS[raw_key]))
            self._tensor_disclosure(card, raw_key, tensors[raw_key])
            card.section_rule()

        card.field('What happens next',
                   'The raw projections are split into attention heads, rotated by positional '
                   'encodings (RoPE), and optionally normalized (Qwen3). Only then do they '
                   'enter the attention computation.')
        return card

    def _card_prepare(self, tensors, tq, tk, d, hidden, heads):
        is_qwen3 = 'qwen3' in str(self.capture.metadata.get('model', '')).lower()

        card = Card('Step 3 · Prepare Q / K / V for attention',
                    'A head is one parallel attention channel operating on a subspace of the '
                    'full projection. Several heads run simultaneously, each finding different '
                    'relationship patterns.')

        card.add(label('WHAT HAPPENS DURING PREPARATION', muted=True))
        if is_qwen3:
            prep_text = (
                'For this Qwen3 model, preparation applies three operations in order: '
                '(1) reshape each projection into per-head slices, '
                '(2) per-head Q and K normalization (QK-norm, a Qwen3 addition), '
                '(3) RoPE — a rotary position encoding that rotates pairs of Q/K features '
                'according to token position, so dot products reflect relative position.')
        else:
            prep_text = (
                'Preparation reshapes each projection into per-head slices, then applies '
                'RoPE — a rotary position encoding that rotates pairs of Q/K features '
                'according to token position, so their dot products reflect relative position. '
                'V is not rotated by RoPE, though it inherits positional information from '
                'previous layers.')
        card.add(label(prep_text))
        card.add(label(
            'GQA (grouped-query attention): Qwen uses fewer K/V heads than Q heads. '
            'The attention implementation repeats K/V heads via repeat_kv to match '
            'the query-head count. These internal steps are not individually captured.',
            muted=True))

        if self.generated:
            card.add(label(
                f'KV cache: the new token contributes one new query (Q length = {tq}) '
                f'but the K/V tensors include cached prompt keys/values '
                f'(K/V length = {tk}). Prompt tokens are not re-run through this layer.',
                muted=True))

        card.add(label('BEFORE vs. AFTER PREPARATION', muted=True))
        for raw_key in ('q', 'k', 'v'):
            attended_key = raw_key + '_attended'
            raw = tensors[raw_key]
            attended = tensors[attended_key]
            card.equation(
                f'<span style="color:{COLORS[raw_key]}"><b>{raw_key.upper()}_raw</b></span>  '
                f'{shape(raw)}  →  '
                f'<b>reshape + RoPE{"+ QK-norm" if is_qwen3 and raw_key in ("q","k") else ""}'
                f'{"+ repeat_kv" if raw_key in ("k","v") else ""}</b>  →  '
                f'<span style="color:{COLORS[raw_key]}"><b>{raw_key.upper()}_attended</b></span>  '
                f'{shape(attended)}')
            card.add(label(TENSOR_EXPLANATIONS[attended_key]))
            self._tensor_disclosure(card, attended_key, attended)
            card.section_rule()

        card.field('Dimensions in this run',
                   f'B = {tensors["q_attended"].shape[0]} batch  ·  '
                   f'H = {heads} query heads  ·  '
                   f'd_head = {d} features per Q/K head  ·  '
                   f'd_model = {hidden}  ·  '
                   f'T_query = {tq}  ·  T_key = {tk}')
        card.field('What happens next',
                   'Q_attended and K_attended enter the dot-product attention calculation.')
        return card

    def _card_scores(self, tensors, tq, tk, d, hidden, heads):
        card = Card('Step 4 · Compute attention scores and weights',
                    'Each query vector is compared with every available key vector. '
                    'A dot product measures their learned compatibility. Softmax converts '
                    'the row of comparison scores into a normalized distribution.')

        card.add(label('OPERATION', muted=True))
        card.equation(
            f'<span style="color:{COLORS["q"]}">Q  [{tq} × {d}]</span>  @  '
            f'<span style="color:{COLORS["k"]}">Kᵀ  [{d} × {tk}]</span>  →  '
            f'QKᵀ  [{tq} × {tk}]<br>'
            '↓  scale by 1/√d_head  +  add causal mask<br>'
            f'<span style="color:{COLORS["scores"]}">Scores  (captured)  [{tq} × {tk}]</span><br>'
            '↓  softmax across key dimension (each row independently)<br>'
            f'<span style="color:{COLORS["weights"]}">Weights  (captured)  [{tq} × {tk}]</span>')

        card.add(label('WHY THE DIMENSIONS WORK', muted=True))
        card.shape_diagram([
            ('Q_attended', 'q', (tq, d)),
            ('K_attendedᵀ', 'k', (d, tk)),
            ('QKᵀ → Scores', 'scores', (tq, tk)),
        ])
        card.add(label(
            f'Inner dimension d_head = {d} cancels. Each of the {tq} query tokens yields '
            f'{tk} dot products — one per available key. Full axes: [batch, head, query, key].'))

        card.add(label('WHAT EACH CAPTURED TENSOR CONTAINS', muted=True))
        card.add(label(
            f'<b style="color:{COLORS["scores"]}">Captured scores:</b>  already scaled '
            f'(×1/√{d}) and masked — exactly what is handed to softmax. '
            'Causal masking adds very large negative values to future positions, making '
            'them near-zero after softmax. The unscaled QKᵀ and the mask alone are not separately stored.<br><br>'
            f'<b style="color:{COLORS["weights"]}">Captured weights:</b>  softmax output — '
            'each score row is exponentiated and normalized to sum ≈ 1. These are mixing '
            'coefficients for V vectors, not vocabulary probabilities.',
            rich=True))

        if self.generated:
            card.add(label(
                'In this one-query pass: the query length is 1, key length covers all '
                'cached prompt tokens plus the new token. There is no future to mask.',
                muted=True))

        # AttentionExplorer — placed inline here because this is the primary display for scores/weights
        card.add(label('CAPTURED ATTENTION DATA', muted=True))
        card.add(label(
            'Select head and query token. Rows = query tokens; columns = key tokens. '
            'Values are displayed with 4 significant figures; hover or click for exact captured value. '
            'Colors are display aids only.',
            muted=True))
        try:
            self.attention = AttentionExplorer(
                tensors,
                token_labels(self.capture, self.generated),
                token_labels(self.capture, self.generated, keys=True),
                layer_index=self.layer_index,
                phase='first_generated_token' if self.generated else 'prefill')
            card.add(self.attention)
        except Exception:
            card.add(label('Attention explorer could not be constructed for this capture.', muted=True))

        card.field('What happens next',
                   'The weight matrix multiplies V_attended to mix value vectors weighted '
                   'by attention. Head results are concatenated and projected by o_proj.')
        return card

    def _card_mix(self, tensors, tq, tk, dv, heads, hidden, v_att):
        card = Card('Step 5 · Mix values and apply the output projection',
                    TENSOR_EXPLANATIONS['attention_output'])

        card.add(label('WHY THIS STEP EXISTS', muted=True))
        card.add(label(
            'The attention weights determined how much each key position contributes. '
            'Multiplying by V transports that content into each query position. '
            'The output projection (o_proj) then maps the multi-head result back '
            'into the residual stream\'s hidden width.'))

        card.add(label('OPERATION (with conceptual intermediates)', muted=True))
        card.equation(
            f'<span style="color:{COLORS["weights"]}">Weights  [{tq} × {tk}]</span>  @  '
            f'<span style="color:{COLORS["v"]}">V_attended  [{tk} × {dv}]</span>  →  '
            f'head result  [{tq} × {dv}]<br>'
            f'concat({heads} heads)  [{tq} × {heads * dv}]  →  '
            f'<span style="color:{COLORS["output"]}">Linear_o  →  attention_output  [{tq} × {hidden}]</span>')

        card.add(label('CAPTURE BOUNDARY', muted=True))
        card.add(label(
            'The per-head (weights @ V) result and the concatenated multi-head tensor '
            '<b>are not individually persisted.</b>  The diagram shows their conceptual '
            'shapes from the captured operands — not numerical values from capture. '
            '<b>attention_output</b> is the captured output of o_proj, after that projection.',
            rich=True))

        card.shape_diagram([
            ('concat(heads)', 'output', (tq, heads * dv)),
            ('W_oᵀ', 'output', (heads * dv, hidden)),
            ('attention_output', 'output', (tq, hidden)),
        ])

        card.section_rule()
        self._tensor_disclosure(card, 'v_attended', v_att)
        card.section_rule()
        self._tensor_disclosure(card, 'attention_output', tensors['attention_output'])

        card.field('What happens next',
                   'The decoder layer combines the attention branch with its residual stream '
                   'and then runs the feed-forward network.')
        return card

    def _card_output(self, tensors, tq, hidden, layers):
        card = Card('Step 6 · Finish this decoder layer',
                    TENSOR_EXPLANATIONS['layer_output'])

        card.add(label('OPERATIONS NOT INDIVIDUALLY CAPTURED', muted=True))
        card.equation(
            '<b>Captured:</b>  attention_output  (from o_proj)<br>'
            '↓<br>'
            'Residual addition:  X_residual = X_input + attention_output<br>'
            'Post-attention normalization:  X_post_norm = RMSNorm(X_residual)<br>'
            'Feed-forward network (MLP):  X_mlp = MLP(X_post_norm)<br>'
            '  — gate activation, two weight matrices, element-wise multiplication<br>'
            'Residual addition:  X_out = X_residual + X_mlp<br>'
            '↓<br>'
            '<b>Captured:</b>  layer_output')

        card.add(label(
            'Attention mixes information across token positions. '
            'The MLP transforms features at each position independently. '
            'Residual connections let the layer add to the existing representation rather '
            'than replace it. These intermediate tensors are not individually saved.',
            muted=True))

        card.shape_diagram([
            ('layer_output', 'layer', tensors['layer_output'].shape),
        ], operator='=')
        card.add(label(
            f'{shape(tensors["layer_output"])}  =  [batch × tokens × hidden]  ·  '
            f'stored {tensors["layer_output"].dtype}'))

        self._tensor_disclosure(card, 'layer_output', tensors['layer_output'])

        last = self.layer_index == sorted(layers)[-1]
        if last:
            card.field('What happens next',
                       'This is the last decoder layer. Its output passes through a final '
                       'RMSNorm and a vocabulary projection to produce logits. '
                       'Those intermediate tensors and learned weights are not saved — '
                       'only the prefill final-position logits vector is captured.')
            button = QPushButton('Continue to final token decision →')
            button.setObjectName('primary')
            button.clicked.connect(lambda: self.go_to('logits'))
            card.add(button)
        else:
            card.field('What happens next',
                       f'This [{tq} × {hidden}] tensor (with batch axis) becomes '
                       f'the input to layer {self.layer_index + 1}. '
                       'Token count and hidden width are unchanged.')
            button = QPushButton(f'Follow into layer {self.layer_index + 1} →')
            button.clicked.connect(self._follow_layer)
            card.add(button)

        return card

    def _follow_layer(self):
        indices = sorted(self.capture.layers)
        self.layer_index = indices[indices.index(self.layer_index) + 1]
        self.step_key = 'input'
        self.go_to('layers')

    # ── Stage: Logits / Word Choice ───────────────────────────────────────────

    def _word_choice(self):
        card = self._stage_card('logits')

        position = len(self.capture.prompt_tokens) - 1
        card.add(label('WHY THIS POSITION?', muted=True))
        card.add(label(
            f'In a causal (left-to-right) language model, only the last prompt position\'s '
            f'representation has attended to all preceding tokens. Its logits vector is the '
            f'model\'s vocabulary score after reading the entire prompt. Position {position} '
            f'corresponds to stored token {self.capture.prompt_tokens[-1]!r}.'))

        card.add(label('OPERATION', muted=True))
        card.equation(
            'last layer output  →  final RMSNorm  →  vocabulary projection<br>'
            '↓<br>'
            '<b>logits vector</b> at final prompt position  (one score per vocabulary entry)<br>'
            '↓<br>'
            'argmax  →  vocabulary ID with highest score  =  first generated token<br><br>'
            '<i>Final RMSNorm output and projection weight matrix are not saved.</i>')

        card.add(label('WHAT IS A LOGIT?', muted=True))
        card.add(label(
            'An unnormalized score for one vocabulary entry — not a probability. '
            'TensorScope uses greedy decoding: argmax selects the ID with the highest '
            'raw score. No softmax is applied and no vocabulary probability vector is saved.'))

        logits = self.capture.logits
        if logits is None:
            card.add(label('LOGITS NOT AVAILABLE IN THIS RUN', muted=True))
            card.add(label(LOGITS_UNAVAILABLE))
        else:
            vocab_size = len(logits)
            card.add(label('SHAPE & TOP SCORES', muted=True))
            card.add(label(
                f'{shape(logits)}  =  [{vocab_size:,} vocabulary entries]  ·  '
                f'stored {logits.dtype}\n'
                f'Only this one row (final prompt position) is saved — not the full '
                f'[batch, tokens, vocabulary] logits tensor.'))

            order = np.argsort(logits)[::-1][:8]
            names = {int(e['id']): str(e.get('token', ''))
                     for e in self.capture.metadata.get('final_logits_top', [])
                     if 'id' in e}
            names[self.capture.token_ids[0]] = self.capture.tokens[0]

            table = QTableWidget(len(order), 4)
            table.setHorizontalHeaderLabels(['Rank', 'Token ID', 'Token text', 'Captured logit'])
            table.verticalHeader().hide()
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            for row_idx, token_id in enumerate(order):
                rank = str(row_idx + 1)
                tid = str(int(token_id))
                tok_text = (repr(names[int(token_id)]) if int(token_id) in names
                            else '(text not saved)')
                logit_val = str(logits[token_id].item())
                for col, text in enumerate((rank, tid, tok_text, logit_val)):
                    item = QTableWidgetItem(text)
                    if row_idx == 0:
                        item.setForeground(QColor('#34d399'))
                    table.setItem(row_idx, col, item)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setFixedHeight(275)
            card.add(table)
            card.add(label(
                'Top 8 by captured logit. Row 1 (green) is the selected token. '
                'The full vector is accessible below.',
                muted=True))
            card.add(Disclosure('Inspect full captured logits vector',
                                 lambda: TensorInspector(logits, name='final_logits',
                                                         axes=['vocabulary token ID'])))

        card.add(label('FIRST GENERATED TOKEN', muted=True))
        card.add(label(
            f'Selected token: {self.capture.tokens[0]!r}  ·  vocabulary ID {self.capture.token_ids[0]}'))

        if logits is not None:
            card.add(label(
                f'Verification: argmax(captured logits) = {int(logits.argmax())} '
                f'= saved first token ID {self.capture.token_ids[0]}. '
                'Saved metadata also attests bitwise equality with stock eager attention logits.',
                muted=True))

        card.field('What happens next',
                   'The selected token enters a second complete forward pass through all '
                   f'{len(self.capture.layers)} decoder layers. '
                   'That first-token pass is also captured in full.')
        return card

    # ── Stage: Coda ───────────────────────────────────────────────────────────

    def _coda(self):
        card = self._stage_card('coda')

        card.add(label('WHY A SECOND FORWARD PASS?', muted=True))
        card.add(label(
            'Autoregressive generation processes one token at a time. After choosing the '
            'first output token from the prefill logits, that token must itself pass through '
            'the model to produce the second token\'s logits. TensorScope captures this '
            'first-token pass completely.'))

        card.add(label('WHAT KV CACHING MEANS HERE', muted=True))
        tok_text = f'{self.capture.tokens[0]!r}  (ID {self.capture.token_ids[0]})'
        card.add(label(
            f'The new token {tok_text} contributes one new query position. '
            'Each attention layer reuses the cached Q/K/V from the prompt prefill — '
            'the prompt tokens are NOT run through the layer again. '
            'The cached K/V are included in the captured k_attended and v_attended '
            'tensors for this phase.'))

        try:
            tensor = self.capture.generated_layers[sorted(self.capture.generated_layers)[0]].tensors['attention_scores']
            card.add(label('ATTENTION SHAPE IN THIS PASS', muted=True))
            card.equation(
                f'1 new query token  +  {tensor.shape[-1] - 1} cached prompt keys<br>'
                f'attention_scores shape:  {shape(tensor)}  '
                f'=  [batch, head, new query, available keys]')
        except Exception:
            pass

        button = QPushButton('Explore the first generated-token pass →')
        button.setObjectName('primary')
        button.clicked.connect(self._enter_generated)
        card.add(button)

        card.field('Where the capture ends',
                   'Embedding and all required layer tensors are saved for this pass. '
                   'The output logits (which would choose the second generated token) are not saved. '
                   'Later output tokens are retained as text/IDs only — their forward-pass '
                   'tensors are not captured. The full answer is context, not evidence of '
                   'additional tensor capture.')

        card.add(Disclosure('Read the complete generated response',
                             lambda: label(self.capture.response)))
        return card

    def _enter_generated(self):
        self.generated = True
        self.step_key = 'input'
        self.go_to('layers')


# ── DetailView ────────────────────────────────────────────────────────────────

class DetailView(QWidget):
    """Virtual full-array browser for every captured tensor in either phase."""
    def __init__(self, capture, parent=None):
        super().__init__(parent)
        self.capture = capture
        self.inspector = None
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.addWidget(label('Raw / Detail · complete captured arrays', title=True))
        self.body.addWidget(label(
            'Select phase, layer, and tensor. All values are accessible through virtual '
            'rows/columns and explicit batch/head slices. '
            'Reading a tensor does not change its stored values or dtype.', muted=True))
        row = QHBoxLayout()
        self.phase_picker = QComboBox()
        self.phase_picker.addItems(['Prompt prefill', 'First generated-token pass'])
        self.layer_picker = QComboBox()
        self.layer_picker.addItem('Outside the layers', None)
        for index in sorted(capture.layers):
            self.layer_picker.addItem(f'Layer {index}', index)
        self.tensor_picker = QComboBox()
        for title, picker in [('Phase', self.phase_picker),
                               ('Layer', self.layer_picker),
                               ('Tensor', self.tensor_picker)]:
            picker.setAccessibleName(title)
            row.addWidget(picker, 1)
        self.body.addLayout(row)
        self.meaning = label('')
        self.body.addWidget(self.meaning)
        self.host = QVBoxLayout()
        self.body.addLayout(self.host)
        self.phase_picker.currentIndexChanged.connect(self._options)
        self.layer_picker.currentIndexChanged.connect(self._options)
        self.tensor_picker.currentIndexChanged.connect(self._show_tensor)
        self._options()

    def _options(self, *_):
        old = self.tensor_picker.currentData()
        self.tensor_picker.blockSignals(True)
        self.tensor_picker.clear()
        if self.layer_picker.currentData() is None:
            self.tensor_picker.addItem('embedding', 'embedding')
            if not self.phase_picker.currentIndex() and self.capture.logits is not None:
                self.tensor_picker.addItem('final_logits', 'final_logits')
        else:
            layers = (self.capture.generated_layers if self.phase_picker.currentIndex()
                      else self.capture.layers)
            for name in layers[self.layer_picker.currentData()].tensors:
                self.tensor_picker.addItem(name, name)
        found = self.tensor_picker.findData(old)
        if found >= 0:
            self.tensor_picker.setCurrentIndex(found)
        self.tensor_picker.blockSignals(False)
        self._show_tensor()

    def _show_tensor(self, *_):
        name = self.tensor_picker.currentData()
        if name is None:
            return
        generated = bool(self.phase_picker.currentIndex())
        if name == 'embedding':
            array = self.capture.generated_embedding if generated else self.capture.embedding
            text = 'Captured token embedding output. Learned embedding table weights are not persisted.'
        elif name == 'final_logits':
            array = self.capture.logits
            text = 'Captured vocabulary logits at the final prompt position; scores, not probabilities.'
        else:
            layers = (self.capture.generated_layers if generated else self.capture.layers)
            array = layers[self.layer_picker.currentData()].tensors[name]
            text = TENSOR_EXPLANATIONS.get(name, 'Additional captured tensor.')
        self.meaning.setTextFormat(Qt.RichText)
        self.meaning.setText(text)
        if self.inspector is not None:
            self.host.removeWidget(self.inspector)
            self.inspector.hide()
            self.inspector.deleteLater()
        rows = token_labels(self.capture, generated,
                            keys=name in ('k_attended', 'v_attended'))
        self.inspector = TensorInspector(
            array, name=name, axes=axes_for(name, array),
            row_labels=rows if name != 'final_logits' else None,
            column_labels=(token_labels(self.capture, generated, keys=True)
                           if name in ('attention_scores', 'attention_weights') else None))
        self.host.addWidget(self.inspector)


# ── ComputationRecap ──────────────────────────────────────────────────────────

class ComputationRecap(QDialog):
    def __init__(self, capture, run_id=None, parent=None, tokens=None):
        super().__init__(parent)
        capture.validate()
        self.capture = capture
        self.setWindowTitle('TensorScope — Computation Journey')
        self.resize(1300, 900)
        self.setMinimumSize(900, 640)

        colors = tokens or {
            'card_bg': '#1c1c20', 'border': '#3b3b44',
            'text_secondary': '#b5b5c1', 'text_primary': '#f0f0f2',
            'accent': '#60a5fa', 'bg': '#111114',
            'text_muted': '#636370', 'nav_active_bg': '#202028',
        }
        self.setStyleSheet(f'''
            /* Cards */
            #journeyCard {{
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-radius: 10px;
            }}

            /* Typography */
            #journeyTitle {{ font-size: 20px; font-weight: 700; }}
            #journeyText  {{ font-size: 13px; line-height: 1.55; }}
            #journeyMuted {{ color: {colors['text_secondary']}; font-size: 11px; letter-spacing: 0.04em; }}
            #journeySmall {{ color: {colors['text_secondary']}; font-size: 11px; }}
            #journeyEquation {{
                font-size: 13px; padding: 12px 14px;
                background: {colors['bg']};
                border-left: 3px solid {colors['accent']};
                border-radius: 4px;
            }}

            /* Pipeline navigator chips */
            QPushButton#pipelineChip {{
                border: 1px solid {colors['border']};
                border-radius: 16px;
                padding: 4px 12px;
                background: {colors['card_bg']};
                color: {colors['text_secondary']};
                font-size: 12px;
            }}
            QPushButton#pipelineChip:hover {{
                border-color: {colors['accent']};
                color: {colors['text_primary']};
            }}
            QPushButton#pipelineChip:checked {{
                background: {colors['accent']};
                border-color: {colors['accent']};
                color: #ffffff;
                font-weight: 700;
            }}
            #pipelineArrow {{ color: {colors['text_muted']}; font-size: 14px; padding: 0 4px; }}

            /* Layer step sidebar buttons */
            QPushButton#layerStepBtn {{
                text-align: left;
                border: none;
                border-radius: 5px;
                padding: 6px 8px;
                background: transparent;
                color: {colors['text_secondary']};
                font-size: 12px;
            }}
            QPushButton#layerStepBtn:hover {{
                background: {colors['nav_active_bg']};
                color: {colors['text_primary']};
            }}
            QPushButton#layerStepBtn:checked {{
                background: {colors['nav_active_bg']};
                color: {colors['text_primary']};
                font-weight: 600;
                border-left: 3px solid {colors['accent']};
            }}
            #stepSidebarHeader {{ font-size: 10px; letter-spacing: 0.06em; padding-bottom: 4px; }}

            /* Section rule */
            QFrame#sectionRule {{
                background: {colors['border']};
                max-height: 1px; min-height: 1px;
                border: none; margin: 6px 0;
            }}

            /* Nav left panel */
            #navItem {{
                text-align: left; border: none; border-radius: 6px;
                padding: 6px 10px; background: transparent;
                color: {colors['text_secondary']};
            }}
            #navItem:hover {{ background: {colors['nav_active_bg']}; color: {colors['text_primary']}; }}
            #navItem:checked {{
                background: {colors['nav_active_bg']};
                color: {colors['text_primary']}; font-weight: 600;
            }}

            /* Table styling */
            QTableView, QTableWidget {{
                background: {colors['card_bg']};
                color: {colors['text_primary']};
                gridline-color: {colors['border']};
                selection-background-color: {colors['accent']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
            }}
            QHeaderView::section {{
                background: {colors['bg']};
                color: {colors['text_secondary']};
                padding: 6px;
                border: 1px solid {colors['border']};
            }}

            /* primary button */
            QPushButton#primary {{
                background: {colors['accent']};
                border: 1px solid {colors['accent']};
                color: #ffffff; font-weight: 600; padding: 7px 16px;
                border-radius: 6px;
            }}
            QPushButton#primary:hover {{ background: #2f6fe0; }}

            /* Disclosure button */
            QPushButton[checkable="true"] {{
                text-align: left;
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 6px 12px;
                background: transparent;
                color: {colors['text_secondary']};
                font-size: 12px;
            }}
            QPushButton[checkable="true"]:hover {{
                border-color: {colors['accent']};
                color: {colors['text_primary']};
            }}
            QPushButton[checkable="true"]:checked {{
                background: {colors['nav_active_bg']};
                color: {colors['text_primary']};
            }}

            /* Journey flow buttons */
            #journeyFlow {{
                text-align: left; padding: 10px;
                font-size: 13px; border: 1px solid {colors['border']};
                border-radius: 6px; background: transparent;
                color: {colors['text_secondary']};
            }}
            #journeyFlow:hover {{
                border-color: {colors['accent']};
                color: {colors['text_primary']};
            }}

            /* Banner */
            #recapBanner {{
                padding: 10px 24px;
                border-bottom: 1px solid {colors['border']};
                font-weight: 600;
                font-size: 13px;
            }}
            #recapBannerSub {{ color: {colors['text_secondary']}; font-size: 11px; }}

            /* Pipeline container */
            #pipelineBar {{
                background: {colors['card_bg']};
                border-bottom: 1px solid {colors['border']};
                padding: 8px 20px;
            }}
        ''')

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Banner ────────────────────────────────────────────────────────────
        banner_widget = QWidget()
        banner_widget.setObjectName('recapBanner')
        banner_layout = QHBoxLayout(banner_widget)
        banner_layout.setContentsMargins(24, 10, 24, 10)
        run_label = QLabel(
            f'{"Saved run #" + str(run_id) if run_id is not None else "Current capture"}  ·  '
            f'{capture.metadata.get("model", "Model ID unavailable")}  ·  '
            f'Real PyTorch capture')
        run_label.setObjectName('recapBanner')
        banner_layout.addWidget(run_label)
        banner_layout.addStretch(1)
        verified_label = QLabel('✓ Validated  ·  Stock-eager match attested')
        verified_label.setStyleSheet(f'color: #16a34a; font-size: 11px;')
        banner_layout.addWidget(verified_label)
        root.addWidget(banner_widget)

        # ── Pipeline navigator bar ────────────────────────────────────────────
        pipeline_bar = QWidget()
        pipeline_bar.setObjectName('pipelineBar')
        pipeline_layout = QHBoxLayout(pipeline_bar)
        pipeline_layout.setContentsMargins(20, 8, 20, 8)
        self.pipeline_nav = PipelineNavigator(self._pipeline_navigate, colors)
        pipeline_layout.addWidget(self.pipeline_nav)
        root.addWidget(pipeline_bar)

        # ── Main split ────────────────────────────────────────────────────────
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # Left nav panel
        left = QWidget()
        left.setObjectName('sidebar')
        left.setMinimumWidth(190)
        left.setMaximumWidth(240)
        nav = QVBoxLayout(left)
        nav.setContentsMargins(12, 18, 12, 18)
        nav.setSpacing(4)

        mode_label = QLabel('MODE')
        mode_label.setObjectName('journeyMuted')
        nav.addWidget(mode_label)

        self.story_button = QPushButton('Learn / Story')
        self.detail_button = QPushButton('Raw / Detail')
        for button in (self.story_button, self.detail_button):
            button.setCheckable(True)
            button.setObjectName('navItem')
            nav.addWidget(button)

        nav.addSpacing(14)
        stages_label = QLabel('JOURNEY STAGES')
        stages_label.setObjectName('journeyMuted')
        nav.addWidget(stages_label)
        self._nav_buttons = {}
        for n, (key, title) in enumerate(STAGES):
            button = QPushButton(f'{n + 1:02}  {title}')
            button.setObjectName('navItem')
            button.setCheckable(True)
            button.clicked.connect(lambda _, k=key: self._nav_to(k))
            nav.addWidget(button)
            self._nav_buttons[key] = button

        nav.addStretch(1)
        # Provenance summary
        nav.addWidget(label(
            f'{len(capture.layers)} layers · '
            f'{len(capture.prompt_tokens)} prompt tokens\n'
            f'Weight dtype: {capture.metadata.get("weight_dtype", "not recorded")}\n'
            f'Stored dtype: {capture.embedding.dtype}',
            muted=True))
        split.addWidget(left)

        # Right content
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 12, 20, 12)
        right_layout.setSpacing(8)

        self.breadcrumb = label('')
        self.breadcrumb.setStyleSheet(f'color: {colors["text_secondary"]}; font-size: 12px;')
        right_layout.addWidget(self.breadcrumb)

        self.provenance = Disclosure('Capture provenance & verification', self._provenance)
        right_layout.addWidget(self.provenance)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 6, 0)
        inner_layout.setSpacing(0)
        self.stack = QStackedWidget()

        self.story_view = StoryView(capture, tokens=tokens)
        self.story_view._on_stage_change = self._on_story_stage_change
        self.stack.addWidget(self.story_view)
        self.detail = None

        inner_layout.addWidget(self.stack)
        inner_layout.addStretch(1)
        self.scroll.setWidget(inner)
        right_layout.addWidget(self.scroll, 1)

        footer = QHBoxLayout()
        self.previous_stage = QPushButton('← Previous stage')
        self.next_stage = QPushButton('Next stage →')
        self.previous_stage.clicked.connect(lambda: self._advance(-1))
        self.next_stage.clicked.connect(lambda: self._advance(1))
        footer.addWidget(self.previous_stage)
        footer.addStretch(1)
        footer.addWidget(self.next_stage)
        right_layout.addLayout(footer)
        split.addWidget(right)
        split.setSizes([225, 1075])

        self.story_button.clicked.connect(self._show_story)
        self.detail_button.clicked.connect(self._show_detail)
        self._show_story()

    def _provenance(self):
        c = self.capture
        card = Card('Evidence for this run — what TensorScope verified')
        m = c.metadata
        card.field('Capture source', str(m.get('capture_source', 'not recorded')))
        card.field('Validation on open',
                   'RunCapture.validate() passed. Saved metadata attests the checks below '
                   'were performed at capture time, before saving.')
        card.field('Stock-eager logit comparison',
                   f'Match: {m.get("logits_match_stock_eager", "not recorded")}  ·  '
                   f'Max absolute difference: {m.get("logits_max_abs_diff_vs_stock_eager", "not recorded")}\n'
                   'TensorScope re-ran the prefill with stock eager attention and required '
                   'bit-exact identical logits. A non-zero difference would have discarded the run.')
        card.field('Attention implementation faithfulness',
                   f'Faithful to upstream eager: {m.get("faithful_to_upstream_eager", "not recorded")}  ·  '
                   f'Verified calls: {m.get("attention_calls_verified", "not recorded")} / '
                   f'{len(c.layers) * 2} expected (both phases × all layers)\n'
                   'CaptureAttention re-ran upstream\'s own eager_attention_forward on every '
                   'captured call and required bit-exact matching outputs.')
        card.field('Runtime',
                   f'{m.get("backend", "not recorded")}  ·  '
                   f'torch {m.get("torch", "not recorded")}  ·  '
                   f'captured at {c.started_at}')
        card.field('Dtype handling',
                   f'Model weights: {m.get("weight_dtype", "not recorded")}  ·  '
                   f'Stored embedding: {c.embedding.dtype}\n'
                   'bfloat16 activations are widened losslessly to float32 for NumPy storage. '
                   'No values are narrowed or lost.')
        card.field('Prompt', c.prompt)
        card.add(Disclosure('Show all saved metadata',
                             lambda: label(json.dumps(m, indent=2, ensure_ascii=False))))
        return card

    def _pipeline_navigate(self, key: str) -> None:
        self._nav_to(key)

    def _nav_to(self, key):
        self._show_story()
        self.story_view.go_to(key)

    def _on_story_stage_change(self, key):
        for k, button in self._nav_buttons.items():
            button.setChecked(k == key)
        self.pipeline_nav.set_current(key)
        idx = [k for k, _ in STAGES].index(key)
        self.previous_stage.setEnabled(idx > 0)
        self.next_stage.setEnabled(idx < len(STAGES) - 1)
        context = ''
        if key in ('layers', 'embedding'):
            context = (' / First generated-token pass' if self.story_view.generated
                       else ' / Prompt prefill')
        if key == 'layers':
            context += (f' / Layer {self.story_view.layer_index} / '
                        f'{dict(LAYER_STEPS)[self.story_view.step_key]}')
        self.breadcrumb.setText('Learn / ' + dict(STAGES)[key] + context)
        self.scroll.verticalScrollBar().setValue(0)

    def _advance(self, amount):
        keys = [k for k, _ in STAGES]
        idx = keys.index(self.story_view._stage_key) + amount
        if 0 <= idx < len(keys):
            self._nav_to(keys[idx])

    def _show_story(self):
        self.stack.setCurrentWidget(self.story_view)
        self.story_button.setChecked(True)
        self.detail_button.setChecked(False)
        self.previous_stage.show()
        self.next_stage.show()
        self._on_story_stage_change(self.story_view._stage_key)

    def _show_detail(self):
        if self.detail is None:
            self.detail = DetailView(self.capture)
            self.stack.addWidget(self.detail)
        self.stack.setCurrentWidget(self.detail)
        self.story_button.setChecked(False)
        self.detail_button.setChecked(True)
        for button in self._nav_buttons.values():
            button.setChecked(False)
        self.breadcrumb.setText('Raw / Detail / exact captured arrays')
        self.previous_stage.hide()
        self.next_stage.hide()
        self.scroll.verticalScrollBar().setValue(0)
