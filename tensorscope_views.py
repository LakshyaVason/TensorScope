"""Reader-facing journeys over one saved capture.  No model, no writes, no new numbers.

Four views, in the order a first-time reader should meet them:

* :class:`LessonView` -- the default.  Eight screens from prompt to first token, each
  walking one real calculation a deliberate step at a time.
* :class:`LearnView` -- five questions about the run.  Five stages, each answering a question a person
  actually asks, with the explanation above the numbers rather than beside them.
* :class:`InternalsView` -- the optional technical journey.  Same capture, ordered
  purpose -> diagram -> equation -> dimensions -> tensor, so the array is the last rung
  of the ladder instead of the first thing on screen.
* :class:`RawView` -- every captured array, reachable directly.

Every widget here reads arrays the capture owns.  Nothing is recomputed to stand in for
a tensor capture did not retain: where the architecture has an intermediate this run did
not keep, it is described and badged conceptual.  Values TensorScope itself calculated
for presentation -- rankings, margins, argmax positions -- carry the derived badge, so a
reader can always tell them from what the model produced.

The shell that hosts these views is `ComputationRecap` in `tensorscope_ui`; the widgets
and copy they are built from live in `tensorscope_common` and `tensorscope_content`.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget,
)

from tensor_widgets import AttentionExplorer, TensorInspector, exact_scalar
from tensorscope_common import (
    COLORS, AnnotatedValueRow, BandedTokenStrip, Card, Disclosure, EvidenceBadge,
    FlowDiagram, LAYER_STEPS, LayerStepSidebar, MiniHeatmap, PhaseBadge, PredictCheck,
    ResponseTextView, ShapeDiagram, StepReveal, TokenChipStrip, axes_for, badge_row,
    chip_text, label, read_only_table, shape, token_labels, token_table,
)
from tensorscope_content import (
    EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED, EVIDENCE_KINDS, EVIDENCE_OBSERVED,
    EVIDENCE_UNATTESTED, INTERNALS_STAGE_INDEX, INTERNALS_STAGES, INTERNALS_STEPS,
    LEARN_STAGE_INDEX, LEARN_STAGES, LESSON_CHECKS, LESSON_DISPLAY_CALC,
    LESSON_STAGE_INDEX, LESSON_STAGES, LESSON_STEPS, LOGITS_UNAVAILABLE,
    CACHED_KV_NOTE, PROMPT_SPAN_UNKNOWN, STOP_REASONS, SYMBOLIC_WEIGHTS_NOTE,
    TELEMETRY_UNAVAILABLE, TENSOR_EXPLANATIONS, TENSOR_LABEL_BY_KEY,
    WHAT_WE_CANNOT_CONCLUDE, WHAT_WE_KNOW, learn_facts, prompt_token_bands,
    readable_spelling,
)


# The (key, sidebar label) lists the shell's navigator and pipeline bar are built from.
# The shell numbers its own navigator, so the leading "1  " in `nav` is dropped here.
LESSON_NAV = [(stage.key, stage.nav.split(None, 1)[-1]) for stage in LESSON_STAGES]
LEARN_NAV = [(stage.key, stage.nav) for stage in LEARN_STAGES]
INTERNALS_NAV = [(stage.key, stage.nav) for stage in INTERNALS_STAGES]

# Which captured tensors each layer step is responsible for showing.  The step copy in
# INTERNALS_STEPS says what the step is for; this says what capture kept from it.
STEP_TENSORS = {
    'input': ['normalized_input'],
    'qkv': ['q', 'k', 'v'],
    'prepare': ['q_attended', 'k_attended', 'v_attended'],
    'scores': ['attention_scores', 'attention_weights'],
    'mix': ['attention_output'],
    'output': ['layer_output'],
}

# Level 2 of the disclosure ladder: a numberless sketch of each step.  Structure lives
# here rather than in the content module because it describes this view's layout; the
# words under it are INTERNALS_STEPS[step]['concept'].
STEP_FLOW = {
    'input': [('X  (layer input)', 'layer'), ('RMSNorm', 'norm'), ('X_norm', 'norm')],
    'qkv': [('X_norm', 'norm'), ('W_Q / W_K / W_V', 'layer'),
            ('Q_raw', 'q'), ('K_raw', 'k'), ('V_raw', 'v')],
    'prepare': [('Q_raw / K_raw', 'q'), ('split into heads', 'layer'),
                ('RoPE by position', 'layer'), ('Q / K', 'k')],
    'scores': [('Q', 'q'), ('K', 'k'), ('scale + mask', 'layer'),
               ('S', 'scores'), ('softmax', 'layer'), ('A', 'weights')],
    'mix': [('A', 'weights'), ('V', 'v'), ('A @ V  (not kept)', 'layer'),
            ('W_o', 'layer'), ('attention output', 'output')],
    'output': [('X + attention output', 'output'), ('RMSNorm', 'norm'),
               ('MLP  (interior not kept)', 'layer'), ('X_next', 'layer')],
}

LAYER_STEP_TITLES = dict(LAYER_STEPS)

# Captured arrays that are not part of the per-layer set, so have no TENSOR_LABELS row.
EXTRA_LABELS = {
    'embedding': 'Embedding rows for this pass',
    'final_logits': 'Vocabulary scores for the final prompt position',
}


# ── Small shared helpers ──────────────────────────────────────────────────────

def replace_in(layout, widget) -> None:
    """Swap the single widget held by `layout`, disposing of the previous one.

    Used by the panels that update on a token or layer selection.  Rebuilding a panel
    rather than mutating labels in place keeps every builder a pure function of the
    current selection, which is what makes them safe to call from the tests.
    """
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            # Bind first: setParent(None) can leave the item's widget() returning None.
            child.setParent(None)
            child.deleteLater()
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


# ── Journey base ──────────────────────────────────────────────────────────────

class JourneyView(QWidget):
    """One stage on screen at a time, with selections that survive stage changes.

    Subclasses supply NAV (the stage list), STAGE_INDEX (the copy), HEADING (which copy
    field is the stage's heading) and `builders()`.  The shell listens to the two signals
    rather than reaching into the view, so a view can be driven by its own controls or by
    the sidebar with no difference in behaviour.
    """

    stage_changed = pyqtSignal(str)
    context_changed = pyqtSignal(str)   # breadcrumb detail, '' when there is none

    NAV: list = []
    STAGE_INDEX: dict = {}
    HEADING = 'heading'

    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(parent)
        capture.validate()
        self.capture = capture
        self.facts = learn_facts(capture)
        self.layer_indices = sorted(capture.layers)
        self.layer_index = self.layer_indices[0]
        self.step_key = 'input'
        self.generated = False
        self._stage_key = ''
        self._stage_widget = None
        self._tokens = tokens or {}
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(14)
        self.prepare()
        self.go_to(self.stage_keys()[0])

    # ── Stage plumbing ────────────────────────────────────────────────────────

    def prepare(self) -> None:
        """Hook for state a subclass needs before its first stage is built.

        The base class creates nothing here.  It exists so a subclass can set up caches or
        per-phase selections after QWidget.__init__ has run but before `go_to` reaches a
        builder -- assigning those attributes any earlier would mean touching a QWidget
        that does not exist yet.
        """

    @classmethod
    def stage_keys(cls) -> list[str]:
        return [key for key, _ in cls.NAV]

    def current_stage(self) -> str:
        return self._stage_key

    def builders(self) -> dict:
        raise NotImplementedError

    def go_to(self, key: str) -> None:
        builders = self.builders()
        if key not in builders:
            raise KeyError(key)
        if self._stage_widget is not None:
            self.body.removeWidget(self._stage_widget)
            self._stage_widget.hide()
            self._stage_widget.deleteLater()
        self._stage_key = key
        self._stage_widget = builders[key]()
        self.body.addWidget(self._stage_widget)
        self.stage_changed.emit(key)
        self.context_changed.emit(self.context())

    def context(self) -> str:
        """Breadcrumb detail for the current stage.  Overridden where there is any."""
        return ''

    def stage_card(self, key: str) -> Card:
        stage = self.STAGE_INDEX[key]
        return Card(getattr(stage, self.HEADING).format(**self.facts),
                    stage.plain.format(**self.facts))

    # ── Captured-array access ─────────────────────────────────────────────────

    @property
    def layers(self) -> dict:
        return self.capture.generated_layers if self.generated else self.capture.layers

    @property
    def phase_name(self) -> str:
        return 'first_generated_token' if self.generated else 'prefill'

    def tensors(self, layer_index: int | None = None) -> dict:
        return self.layers[self.layer_index if layer_index is None else layer_index].tensors

    def embedding(self):
        return self.capture.generated_embedding if self.generated else self.capture.embedding

    def inspector(self, name: str, tensor) -> TensorInspector:
        """A full inspector over the stored array, with token-aware headers."""
        rows = token_labels(self.capture, self.generated,
                            keys=name in ('k_attended', 'v_attended'))
        columns = (token_labels(self.capture, self.generated, keys=True)
                   if name in ('attention_scores', 'attention_weights') else None)
        return TensorInspector(tensor, name=name, axes=axes_for(name, tensor),
                               row_labels=None if name == 'final_logits' else rows,
                               column_labels=columns)

    def tensor_disclosure(self, card: Card, name: str, tensor) -> None:
        """Shape, axes, evidence tier, and the inspector behind one click."""
        card.add(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, name)))
        title = TENSOR_LABEL_BY_KEY.get(name) or EXTRA_LABELS.get(name, name)
        card.add(label(f'{title}   {shape(tensor)}   ·   '
                       f'stored {tensor.dtype}'))
        card.add(label('Axes: ' + ' × '.join(axes_for(name, tensor)), muted=True))
        card.add(Disclosure(f'Inspect captured {name}',
                            lambda n=name, t=tensor: self.inspector(n, t)))

    def mpl_style(self) -> dict:
        """Figure colours for MiniHeatmap, taken from the host's design tokens."""
        tokens = self._tokens
        if not tokens:
            return {}
        return {'figure.facecolor': tokens.get('card_bg', '#1c1c20'),
                'axes.facecolor': tokens.get('bg', '#111114'),
                'text.color': tokens.get('text_secondary', '#9f9fa8')}

    # ── Phase controls ────────────────────────────────────────────────────────

    def phase_controls(self) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        self.phase_picker = QComboBox()
        self.phase_picker.setAccessibleName('Captured phase')
        self.phase_picker.addItems(['Phase 1 · Prompt prefill',
                                   'Phase 2 · First generated-token pass'])
        self.phase_picker.setCurrentIndex(int(self.generated))
        self.phase_picker.currentIndexChanged.connect(self._phase_changed)
        row.addWidget(self.phase_picker, 1)
        row.addWidget(PhaseBadge(self.generated))
        return host

    def _phase_changed(self, index: int) -> None:
        self.generated = bool(index)
        self.go_to(self._stage_key)


# ── LessonView ────────────────────────────────────────────────────────────────

class LessonView(JourneyView):
    """The default journey: eight screens, one real calculation per screen.

    Everything on a screen is read from the capture.  Where a value is arithmetic
    TensorScope performed for the display (a product, a sum, a position of a maximum) it
    is badged derived; where the architecture has a step this capture did not keep, the
    step is drawn as symbols and badged conceptual.  Selections and reveal progress live
    in dicts this view owns, so a screen that is rebuilt comes back as the reader left it.
    """

    NAV = LESSON_NAV
    STAGE_INDEX = LESSON_STAGE_INDEX
    HEADING = 'question'
    NOUN = 'Screen'
    HEAD = 0
    PREVIEW = 8

    def prepare(self) -> None:
        last = max(0, len(self.capture.prompt_tokens) - 1)
        self._reveals: dict = {}
        self._checks: dict = {}
        self._open: dict = {}
        self._refresh = None
        self.bands = prompt_token_bands(self.capture.prompt, self.capture.prompt_tokens)
        self.token_pos = last              # the deciding position, unless the reader moves it
        self.query_pos = last
        self.key_pos = 0
        self.mask_pos = max(0, last - 1)   # second to last: has a later position to mask
        self.decision = 0

    def builders(self) -> dict:
        return {'received': self._received, 'vectors': self._vectors,
                'transform': self._transform, 'compare': self._compare,
                'weights': self._weights, 'combine': self._combine,
                'select': self._select, 'continue': self._continue}

    def context(self) -> str:
        if self._stage_key in ('transform', 'compare', 'weights', 'combine'):
            return f'Layer {self.layer_index}'
        return 'Final layer' if self._stage_key == 'select' else ''

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _spelling(self, index: int) -> str:
        tokens = self.capture.prompt_tokens
        return tokens[index] if 0 <= index < len(tokens) else f'position {index}'

    def _layer_picker(self, card: Card) -> None:
        picker = QComboBox()
        picker.setAccessibleName('Layer')
        for index in self.layer_indices:
            picker.addItem(f'Layer {index}', index)
        picker.setCurrentIndex(self.layer_indices.index(self.layer_index))
        picker.currentIndexChanged.connect(self._layer_changed)
        card.add(picker)

    def _layer_changed(self, position: int) -> None:
        self.layer_index = self.layer_indices[position]
        if self._refresh is not None:
            self._refresh()
        self.context_changed.emit(self.context())

    def _slot(self, card: Card) -> QVBoxLayout:
        slot = QVBoxLayout()
        slot.setContentsMargins(0, 0, 0, 0)
        card.add_layout(slot)
        return slot

    def _panel(self) -> tuple[QWidget, QVBoxLayout]:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(10)
        return host, layout

    def _check(self, layout, name: str, **fields) -> None:
        check = LESSON_CHECKS[name]
        values = {**self.facts, **fields}
        layout.addWidget(PredictCheck(
            check['question'].format(**values),
            [option.format(**values) for option in check['options']],
            check['answer'], check['explanation'].format(**values),
            state=self._checks, state_key=name))

    def _reveal(self, layout, screen: str, builds, steps=None) -> None:
        source = steps if steps is not None else LESSON_STEPS[screen]
        prepared = []
        for step, build in zip(source, builds):
            prepared.append({**step, 'caption': step['caption'].format(**self.facts),
                             'body': step['body'].format(**self.facts), 'build': build})
        layout.addWidget(StepReveal(prepared, state=self._reveals, state_key=screen))

    def _values(self, values, *, kind=EVIDENCE_OBSERVED, note='', caption='', labels=None,
                total=None, formatter=None) -> AnnotatedValueRow:
        shown = list(values)[:self.PREVIEW]
        return AnnotatedValueRow(
            shown, labels=labels, indices=None if labels else range(len(shown)),
            caption=caption, kind=kind, note=note,
            total=total if total is not None else len(shown), formatter=formatter)

    def _tensor_block(self, name: str, tensor) -> QWidget:
        host, layout = self._panel()
        title = TENSOR_LABEL_BY_KEY.get(name) or EXTRA_LABELS.get(name, name)
        layout.addWidget(label(f'{title}   {shape(tensor)}   ·   stored {tensor.dtype}'))
        layout.addWidget(Disclosure(f'Inspect captured {name}',
                                    lambda n=name, t=tensor: self.inspector(n, t)))
        return host

    def _pair(self, *widgets) -> QWidget:
        host, layout = self._panel()
        for widget in widgets:
            layout.addWidget(widget)
        return host

    # ── 1  Received ───────────────────────────────────────────────────────────

    def _received(self) -> QWidget:
        card = self.stage_card('received')
        card.field('You typed', self.capture.prompt)
        if self.facts['template_token_count'] > 0:
            host, layout = self._panel()
            self._check(layout, 'tokens')
            card.add(host)
        card.add(label('WHAT THE MODEL RECEIVED', muted=True))
        if self.bands is None:
            card.add(label(PROMPT_SPAN_UNKNOWN, muted=True))
        strip = BandedTokenStrip(self.capture.prompt_tokens, self.bands,
                                 selected=self.token_pos)
        strip.selected.connect(self._pick_received)
        card.add(strip)
        card.add(label('Spellings are the tokenizer\'s own, so a chip that looks odd is '
                       'usually a space or newline stored as part of a piece.', muted=True))
        self._received_slot = self._slot(card)
        self._show_received(self.token_pos)
        card.add(Disclosure('Show the whole prompt as a table',
                            lambda: token_table(self.capture),
                            state=self._open, state_key='received-table'))
        return card

    def _pick_received(self, index: int) -> None:
        self.token_pos = index
        self._show_received(index)

    def _show_received(self, index: int) -> None:
        host, layout = self._panel()
        token = self._spelling(index)
        layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'stored token')))
        layout.addWidget(label(
            f'Position {index}  ·  vocabulary ID {self.capture.prompt_token_ids[index]}  ·  '
            f'stored {token!r}  ·  reads as {readable_spelling(token)!r}'))
        replace_in(self._received_slot, host)

    # ── 2  Vectors ────────────────────────────────────────────────────────────

    def _vectors(self) -> QWidget:
        card = self.stage_card('vectors')
        card.add(label('PICK A POSITION', muted=True))
        strip = BandedTokenStrip(self.capture.prompt_tokens, None, selected=self.token_pos)
        strip.selected.connect(self._pick_vector)
        card.add(strip)
        self._vector_slot = self._slot(card)
        self._show_vector(self.token_pos)
        return card

    def _pick_vector(self, index: int) -> None:
        self.token_pos = index
        self._show_vector(index)

    def _show_vector(self, index: int) -> None:
        host, layout = self._panel()
        row = embedding_row(self.capture.embedding, index)
        token_id = self.capture.prompt_token_ids[index]
        if row is None:
            layout.addWidget(label(
                'This run\'s stored embedding is not shaped [batch, token, feature], so a '
                'single position cannot be sliced out of it.', muted=True))
        else:
            layout.addWidget(label(
                f'ID {token_id} ({self._spelling(index)!r}) became {row.shape[-1]} numbers. '
                f'The same ID always selects the same row; position is added later, by '
                f'attention.'))
            layout.addWidget(self._values(
                row, note='embedding row', total=row.shape[-1],
                caption=f'First {min(self.PREVIEW, row.shape[-1])} of {row.shape[-1]} features'))
            layout.addWidget(Disclosure(
                f'Inspect all {row.shape[-1]} stored values',
                lambda r=row, i=index: TensorInspector(
                    r, name=f'embedding · prompt position {i}', axes=['model feature'])))
        replace_in(self._vector_slot, host)

    # ── 3  Transformed ────────────────────────────────────────────────────────

    def _transform(self) -> QWidget:
        card = self.stage_card('transform')
        self._layer_picker(card)
        self._transform_slot = self._slot(card)
        self._refresh = self._show_transform
        self._show_transform()
        return card

    def _show_transform(self) -> None:
        host, layout = self._panel()
        tensors = self.tensors()
        pos = self.token_pos
        layout.addWidget(label(
            f'Working on position {pos} ({self._spelling(pos)!r}) in layer '
            f'{self.layer_index}. Change the position on the previous screen.', muted=True))
        normalized = embedding_row(tensors['normalized_input'], pos)
        width = self.facts['hidden_size']
        if normalized is None:
            layout.addWidget(label('This run\'s stored normalized input is not shaped '
                                   '[batch, token, feature], so one position cannot be '
                                   'sliced out of it.', muted=True))
            replace_in(self._transform_slot, host)
            return

        def normalised():
            return self._values(normalized, note='normalized_input', total=len(normalized))

        def pairing():
            symbols = [f'w{index}' for index in range(self.PREVIEW)]
            return self._pair(
                self._values(normalized, caption='Row: captured values', total=len(normalized),
                             labels=[f'x{i}' for i in range(self.PREVIEW)]),
                self._values(symbols, kind=EVIDENCE_CONCEPTUAL, note='learned, not captured',
                             caption='Column: symbols', formatter=str,
                             labels=[f'column entry {i}' for i in range(self.PREVIEW)]),
                label(SYMBOLIC_WEIGHTS_NOTE, muted=True, small=True))

        def summing():
            equation = label(f'output feature = x0·w0 + x1·w1 + … + x{max(width - 1, 0)}·'
                             f'w{max(width - 1, 0)}', rich=True)
            equation.setObjectName('journeyEquation')
            return equation

        def purposes():
            return FlowDiagram([('normalized vector', 'norm'), ('× W_Q  →  query', 'q'),
                                ('× W_K  →  key', 'k'), ('× W_V  →  value', 'v')],
                               caption='Three learned projections read the same vector.')

        def produced():
            blocks = []
            for name in ('q', 'k', 'v'):
                row = embedding_row(tensors[name], pos)
                blocks.append(self._tensor_block(name, tensors[name]))
                if row is not None:
                    blocks.append(self._values(row, note=name, total=len(row)))
            return self._pair(*blocks)

        self._reveal(layout, 'transform', [normalised, pairing, summing, purposes, produced])
        replace_in(self._transform_slot, host)

    # ── 4  Compared ───────────────────────────────────────────────────────────

    def _compare(self) -> QWidget:
        card = self.stage_card('compare')
        self._layer_picker(card)
        card.add(label(
            f'Query: the final prompt position, {self.query_pos} '
            f'({self._spelling(self.query_pos)!r}). Pick the key it is compared with.',
            muted=True))
        strip = BandedTokenStrip(self.capture.prompt_tokens, None, selected=self.key_pos)
        strip.selected.connect(self._pick_key)
        card.add(strip)
        self._compare_slot = self._slot(card)
        self._refresh = self._show_compare
        self._show_compare()
        return card

    def _pick_key(self, index: int) -> None:
        self.key_pos = index
        self._show_compare()

    def _show_compare(self) -> None:
        host, layout = self._panel()
        tensors = self.tensors()
        queries = head_slice(tensors['q_attended'], self.HEAD)
        keys = head_slice(tensors['k_attended'], self.HEAD)
        scores = head_slice(tensors['attention_scores'], self.HEAD)
        qp = min(self.query_pos, queries.shape[0] - 1)
        kp = min(self.key_pos, keys.shape[0] - 1)
        query, key = queries[qp], keys[kp]
        product = np.asarray(query, dtype=np.float64) * np.asarray(key, dtype=np.float64)
        total = float(product.sum())
        scale = len(query) ** -0.5
        captured = scores[qp, kp]
        self._check(layout, 'dot')

        def operands():
            return self._pair(
                self._values(query, note='q_attended', total=len(query),
                             caption=f'Query, position {qp}, head {self.HEAD}'),
                self._values(key, note='k_attended', total=len(key),
                             caption=f'Key, position {kp}, head {self.HEAD}'))

        def multiplied():
            return self._pair(
                self._values(product, kind=EVIDENCE_DERIVED, note='calculated here',
                             total=len(product), caption='Query × key, entry by entry'),
                label(LESSON_DISPLAY_CALC, muted=True, small=True))

        def summed():
            return self._pair(
                self._values([total], kind=EVIDENCE_DERIVED, note='calculated here',
                             labels=[f'sum of {len(product)} products']),
                label(LESSON_DISPLAY_CALC, muted=True, small=True))

        def scaled():
            return self._pair(
                self._values([total * scale], kind=EVIDENCE_DERIVED, note='calculated here',
                             labels=['scaled sum']),
                self._values([captured], note='attention_scores', labels=['captured score']),
                label(f'Difference: {exact_scalar(abs(total * scale - float(captured)))}',
                      muted=True, small=True))

        self._reveal(layout, 'compare', [operands, multiplied, summed, scaled])
        replace_in(self._compare_slot, host)

    # ── 5  Weights ────────────────────────────────────────────────────────────

    def _weights(self) -> QWidget:
        card = self.stage_card('weights')
        self._layer_picker(card)
        card.add(label('Pick the query position whose row to follow. Earlier positions have '
                       'more of the row masked.', muted=True))
        strip = BandedTokenStrip(self.capture.prompt_tokens, None, selected=self.mask_pos)
        strip.selected.connect(self._pick_mask)
        card.add(strip)
        self._weights_slot = self._slot(card)
        self._refresh = self._show_weights
        self._show_weights()
        return card

    def _pick_mask(self, index: int) -> None:
        self.mask_pos = index
        self._show_weights()

    def _show_weights(self) -> None:
        host, layout = self._panel()
        tensors = self.tensors()
        scores = head_slice(tensors['attention_scores'], self.HEAD)
        weights = head_slice(tensors['attention_weights'], self.HEAD)
        qp = min(self.mask_pos, scores.shape[0] - 1)
        score_row, weight_row = scores[qp], weights[qp]
        masked = list(range(qp + 1, len(score_row)))
        layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'attention_weights')))
        layout.addWidget(MiniHeatmap(
            weights, f'Layer {self.layer_index}, head {self.HEAD} — captured weights',
            self.mpl_style()))
        if masked:
            self._check(layout, 'mask')

        def raw_scores():
            return self._values(score_row, note='attention_scores', total=len(score_row),
                                caption=f'Row for position {qp}: first scores')

        def masking():
            if not masked:
                return label('Nothing comes after this position, so nothing in its row is '
                             'masked. Pick an earlier position to see the mask.', muted=True)
            return self._values(score_row[masked], note='masked columns', total=len(masked),
                                labels=[f'key {j}' for j in masked[:self.PREVIEW]],
                                caption='Scores at positions after this one')

        def softmaxed():
            blocks = [self._values(weight_row, note='attention_weights', total=len(weight_row),
                                   caption='First weights of the row')]
            if masked:
                blocks.append(self._values(
                    weight_row[masked], note='masked columns', total=len(masked),
                    labels=[f'key {j}' for j in masked[:self.PREVIEW]],
                    caption='Weights at the masked positions'))
            return self._pair(*blocks)

        def adding_up():
            return self._pair(
                self._values([float(np.asarray(weight_row, dtype=np.float64).sum())],
                             kind=EVIDENCE_DERIVED, note='calculated here',
                             labels=['sum of the row']),
                label(LESSON_DISPLAY_CALC, muted=True, small=True))

        self._reveal(layout, 'weights', [raw_scores, masking, softmaxed, adding_up])
        replace_in(self._weights_slot, host)

    # ── 6  Combined ───────────────────────────────────────────────────────────

    def _combine(self) -> QWidget:
        card = self.stage_card('combine')
        self._layer_picker(card)
        self._combine_slot = self._slot(card)
        self._refresh = self._show_combine
        self._show_combine()
        return card

    def _show_combine(self) -> None:
        host, layout = self._panel()
        tensors = self.tensors()
        pos = self.token_pos
        layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'steps not kept')))
        layout.addWidget(FlowDiagram(
            [('weights', 'weights'), ('× values', 'v'), ('blend (not kept)', 'layer'),
             ('join heads', 'layer'), ('W_o', 'layer'), ('attention output', 'output'),
             ('+ residual (not kept)', 'layer'), ('MLP (interior not kept)', 'layer'),
             ('layer output', 'output')],
            caption='Steps marked "not kept" are described, not reconstructed.'))
        layout.addWidget(label(f'What was captured, for position {pos} '
                               f'({self._spelling(pos)!r}):', muted=True))
        for name in ('attention_output', 'layer_output'):
            row = embedding_row(tensors[name], pos)
            layout.addWidget(self._tensor_block(name, tensors[name]))
            if row is not None:
                layout.addWidget(self._values(row, note=name, total=len(row)))
        replace_in(self._combine_slot, host)

    # ── 7  Selected ───────────────────────────────────────────────────────────

    def _select(self) -> QWidget:
        card = self.stage_card('select')
        host, layout = self._panel()
        logits = self.capture.logits
        final = self.capture.layers[max(self.layer_indices)].tensors
        pos = len(self.capture.prompt_tokens) - 1
        flat = np.asarray(logits).reshape(-1) if logits is not None else None
        order = np.argsort(-flat)[:5] if flat is not None else []

        if flat is not None and len(order) >= 2:
            winner, runner = float(flat[order[0]]), float(flat[order[1]])
            if runner > 0:
                self._check(layout, 'logits', winner=f'{winner:.4g}',
                            runner_up=f'{runner:.4g}', ratio=f'{winner / runner:.2f}')

        def finished():
            row = embedding_row(final['layer_output'], pos)
            return (self._values(row, note='final layer output', total=len(row))
                    if row is not None else label('Not shaped [batch, token, feature].'))

        def normalization():
            return FlowDiagram([('final vector', 'output'), ('RMSNorm (not kept)', 'norm')])

        def projection():
            return FlowDiagram([('normalized vector', 'norm'),
                                ('× vocabulary matrix (not kept)', 'layer'),
                                ('one score per entry', 'scores')],
                               caption=SYMBOLIC_WEIGHTS_NOTE)

        def vector():
            return self._pair(
                self._tensor_block('final_logits', logits),
                label(f'{len(flat):,} scores, one per vocabulary entry.', muted=True))

        def candidates():
            trace = self.capture.generation_trace
            if trace:
                rows = [(rank + 1, entry['id'], repr(str(entry.get('token', ''))),
                         f'{float(entry["logit"]):.6f}')
                        for rank, entry in enumerate(trace[0].top_candidates[:5])]
            else:
                rows = [(rank + 1, int(index), '', exact_scalar(flat[index]))
                        for rank, index in enumerate(order[:5])]
            return read_only_table(['Rank', 'Token ID', 'Token', 'Captured score'], rows,
                                   stretch=2, highlight=0)

        def selection():
            best = int(np.argmax(flat))
            return self._pair(
                self._values([best], kind=EVIDENCE_DERIVED, formatter=str,
                             note='position of the maximum', labels=['token ID']),
                self._values([flat[best]], note='final_logits', labels=['its score']),
                label(f'That ID is the token this run produced: '
                      f'{self.capture.tokens[0]!r}.'))

        builds = [finished, normalization, projection, vector, candidates, selection]
        steps = list(LESSON_STEPS['select'])
        if flat is None:
            builds = builds[:3] + [lambda: label(LOGITS_UNAVAILABLE)]
            steps = steps[:3] + [{'caption': 'The score vector was not retained',
                                  'body': LOGITS_UNAVAILABLE, 'tier': EVIDENCE_CONCEPTUAL}]
        self._reveal(layout, 'select', builds, steps)
        card.add(host)
        return card

    # ── 8  Continues ──────────────────────────────────────────────────────────

    def _continue(self) -> QWidget:
        card = self.stage_card('continue')
        card.add(label('THE ANSWER — SELECT A TOKEN', muted=True))
        view = ResponseTextView(self.capture.tokens, self.capture.response,
                                selected=self.decision)
        view.selected.connect(self._pick_decision)
        card.add(view)
        self._continue_slot = self._slot(card)
        self._show_continue(view.current())
        card.section_rule()
        card.add(label(CACHED_KV_NOTE, muted=True))
        card.add(Disclosure(
            'Show the first generated pass\'s attention (head 0)',
            lambda: self._generated_attention()))
        return card

    def _generated_attention(self) -> QWidget:
        weights = self.capture.generated_layers[self.layer_index].tensors['attention_weights']
        host, layout = self._panel()
        layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'attention_weights')))
        layout.addWidget(MiniHeatmap(
            head_slice(weights, self.HEAD),
            f'Layer {self.layer_index}, head {self.HEAD} — the new token against every '
            f'earlier position', self.mpl_style()))
        return host

    def _pick_decision(self, index: int) -> None:
        self.decision = index
        self._show_continue(index)

    def _show_continue(self, step: int) -> None:
        capture = self.capture
        host, layout = self._panel()
        if 0 <= step < len(capture.tokens):
            layout.addWidget(label(f'Step {step}  ·  selected {capture.tokens[step]!r}  ·  '
                                   f'vocabulary ID {capture.token_ids[step]}'))
            trace = capture.generation_trace
            if step >= len(trace):
                layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'not recorded')))
                layout.addWidget(label(TELEMETRY_UNAVAILABLE))
            else:
                decision = trace[step]
                tier = EVIDENCE_OBSERVED if (step == 0 or decision.attested) \
                    else EVIDENCE_UNATTESTED
                layout.addWidget(badge_row(
                    EvidenceBadge(tier, 'scores from the forward pass'),
                    EvidenceBadge(EVIDENCE_DERIVED, 'ranking')))
                winner = next((i for i, entry in enumerate(decision.top_candidates)
                               if int(entry['id']) == decision.selected_token_id), None)
                layout.addWidget(read_only_table(
                    ['Rank', 'Token ID', 'Token', 'Captured score'],
                    [(rank + 1, entry['id'], repr(str(entry.get('token', ''))),
                      f'{float(entry["logit"]):.6f}')
                     for rank, entry in enumerate(decision.top_candidates)],
                    stretch=2, highlight=winner))
        replace_in(self._continue_slot, host)


# ── LearnView ─────────────────────────────────────────────────────────────────

class LearnView(JourneyView):
    """The default journey: five questions, answered with this run's own numbers."""

    NAV = LEARN_NAV
    STAGE_INDEX = LEARN_STAGE_INDEX
    HEADING = 'question'

    def builders(self) -> dict:
        return {'overview': self._overview, 'tokens': self._tokens_stage,
                'context': self._context_stage, 'generation': self._generation,
                'limits': self._limits}

    def context(self) -> str:
        return f'Layer {self.layer_index}' if self._stage_key == 'context' else ''

    # ── Stage 1: prompt and response ──────────────────────────────────────────

    def _overview(self) -> QWidget:
        card = self.stage_card('overview')
        card.field('What you asked', self.capture.prompt)
        card.field('What the model answered', self.capture.response)

        card.section_rule()
        card.add(label('THE PATH EVERY TOKEN TAKES', muted=True))
        card.add(FlowDiagram(
            [('your text', 'layer'), ('token IDs', 'norm'), ('feature vectors', 'q'),
             (f'{self.facts["layer_count"]} layers', 'weights'),
             ('vocabulary scores', 'scores'), ('one token', 'output')],
            caption='The whole run is this path, repeated once per token of the answer. '
                    'The screens that follow walk it with this run\'s own values.'))

        card.add(label('WHAT THIS RUN DID', muted=True))
        stop = self.capture.metadata.get('stop_reason', 'unknown')
        card.add(read_only_table(
            ['Quantity', 'This run'],
            [('Prompt tokens the model received', self.facts['prompt_token_count']),
             ('Tokens it generated', self.facts['generated_count']),
             ('Decoder layers each pass went through', self.facts['layer_count']),
             ('Features per token vector', self.facts['hidden_size']),
             ('Decoding rule', self.capture.metadata.get('decoding', 'not recorded')),
             ('Why it stopped', STOP_REASONS.get(stop, STOP_REASONS['unknown']))],
            stretch=1))
        card.add(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'counts of stored values')))

        card.section_rule()
        card.field('How much of it was recorded at tensor level',
                   '<b>Two full passes.</b> The prompt prefill, where every prompt token '
                   'goes through every layer at once, and the pass for the first generated '
                   'token, which reuses the prompt\'s cached keys and values. Every '
                   'required tensor of both is stored.<br><br>'
                   '<b>The rest of the answer</b> was generated with capture off, so you '
                   'can read a complete response. For those tokens TensorScope kept the '
                   'scores that chose each one, but not their internal tensors.', rich=True)
        return card

    # ── Stage 2: tokens ───────────────────────────────────────────────────────

    def _tokens_stage(self) -> QWidget:
        card = self.stage_card('tokens')
        card.add(label('THE PROMPT AS THE MODEL RECEIVED IT', muted=True))
        strip = TokenChipStrip(self.capture.prompt_tokens)
        strip.selected.connect(self._show_token)
        card.add(strip)
        card.add(label('Spellings are quoted, so a chip that looks empty is a real space '
                       'or newline token. Select one to see what was stored for it.',
                       muted=True))
        self._token_slot = QVBoxLayout()
        self._token_slot.setContentsMargins(0, 0, 0, 0)
        card.add_layout(self._token_slot)
        self._show_token(strip.current())

        card.section_rule()
        card.add(Disclosure('Show the whole prompt as a table',
                            lambda: token_table(self.capture)))
        card.field('Why the IDs are all it gets',
                   'The tokenizer is not part of the network. By the time the model runs '
                   'the text is gone and only these integers remain, which is why a model '
                   'can be confidently wrong about spelling or letter counts.')
        return card

    def _show_token(self, index: int) -> None:
        """Rebuild the per-token panel for the selected prompt position."""
        token = self.capture.prompt_tokens[index]
        token_id = self.capture.prompt_token_ids[index]
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(label(f'Position {index}  ·  vocabulary ID {token_id}  ·  '
                               f'stored spelling {token!r}'))
        if index == len(self.capture.prompt_tokens) - 1:
            layout.addWidget(label(
                'This is the final prompt position. Because the model only reads '
                'leftwards, it is the only position that has seen the whole prompt, so '
                'its vector is the one that chooses the first token of the answer.',
                muted=True))

        row = embedding_row(self.capture.embedding, index)
        if row is None:
            layout.addWidget(label(
                'This run\'s stored embedding is not shaped [batch, token, feature], so a '
                'single position cannot be sliced out of it.', muted=True))
        else:
            layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'embedding row')))
            layout.addWidget(label(
                f'The model replaced this token with {row.shape[-1]} numbers: the row the '
                f'embedding table produced for ID {token_id} in this run, stored '
                f'{row.dtype}. The coordinates are learned, not named, so no column means '
                f'"plural" or "verb".'))
            layout.addWidget(Disclosure(
                f'Inspect the {row.shape[-1]} stored values for this token',
                lambda r=row, i=index: TensorInspector(
                    r, name=f'embedding · prompt position {i}', axes=['model feature'])))
        replace_in(self._token_slot, host)

    # ── Stage 3: building context ─────────────────────────────────────────────

    def _context_stage(self) -> QWidget:
        card = self.stage_card('context')
        card.add(label('WHAT ATTENTION DOES TO ONE VECTOR', muted=True))
        card.add(FlowDiagram(
            [('a position\'s vector', 'q'), ('score every earlier position', 'scores'),
             ('blend their values', 'weights'), ('updated vector', 'output')],
            caption='Repeated once per layer with different learned weights each time. '
                    'This is the only step in the whole model that moves information '
                    'between positions.'))

        heads = head_count(self.tensors()['attention_weights'])
        card.add(label('ONE REAL EXAMPLE FROM THIS RUN', muted=True))
        picker = QComboBox()
        picker.setAccessibleName('Layer')
        for index in self.layer_indices:
            picker.addItem(f'Layer {index}', index)
        picker.setCurrentIndex(self.layer_indices.index(self.layer_index))
        picker.currentIndexChanged.connect(self._context_layer_changed)
        card.add(picker)
        card.add(label(
            f'All {self.facts["layer_count"]} layers repeat the same structure with their '
            f'own learned weights, and each runs {heads} heads in parallel. Below is '
            f'head 0 of the selected layer.'))

        self._context_slot = QVBoxLayout()
        self._context_slot.setContentsMargins(0, 0, 0, 0)
        card.add_layout(self._context_slot)
        self._show_context()
        return card

    def _context_layer_changed(self, position: int) -> None:
        self.layer_index = self.layer_indices[position]
        self._show_context()
        self.context_changed.emit(self.context())

    def _show_context(self) -> None:
        tensors = self.tensors()
        weights = tensors['attention_weights']
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'attention_weights')))
        layout.addWidget(MiniHeatmap(
            head_slice(weights, 0),
            f'Layer {self.layer_index}, head 0 — captured weights', self.mpl_style()))
        layout.addWidget(label(
            'Each row is one position deciding how much to take from each other position; '
            'brighter means more weight. The upper-right triangle is dark because a '
            'position cannot read what comes after it.', muted=True))

        row = head_slice(weights, 0)[-1]
        best = int(np.argmax(row))
        spellings = list(self.capture.prompt_tokens) + list(self.capture.tokens[:1])
        target = repr(spellings[best]) if best < len(spellings) else f'key index {best}'
        layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_DERIVED, 'argmax of one row')))
        layout.addWidget(label(
            f'In that head the last query position\'s largest weight went to position '
            f'{best} ({target}). TensorScope found that by taking the argmax of a stored '
            f'row: the model never computed a "most attended" position, and one head\'s '
            f'weights are a mechanism, not the model\'s reasoning.'))
        layout.addWidget(Disclosure(
            'Explore every head and position of this layer',
            lambda t=tensors, i=self.layer_index: AttentionExplorer(
                t, token_labels(self.capture, self.generated),
                token_labels(self.capture, self.generated, keys=True),
                layer_index=i, phase=self.phase_name)))
        replace_in(self._context_slot, host)

    # ── Stage 4: choosing each token ──────────────────────────────────────────

    def _generation(self) -> QWidget:
        card = self.stage_card('generation')
        card.add(label('THE ANSWER, ONE TOKEN PER CHIP', muted=True))
        strip = TokenChipStrip(self.capture.tokens,
                               offset=len(self.capture.prompt_tokens))
        strip.selected.connect(self._show_decision)
        card.add(strip)
        card.add(label(
            f'Decoding was {self.capture.metadata.get("decoding", "not recorded")}: the '
            f'highest-scoring vocabulary entry is taken with no sampling, so nothing here '
            f'is random. Select a token to see the scores that chose it.', muted=True))
        self._decision_slot = QVBoxLayout()
        self._decision_slot.setContentsMargins(0, 0, 0, 0)
        card.add_layout(self._decision_slot)
        self._show_decision(strip.current())
        return card

    def _show_decision(self, step: int) -> None:
        """Rebuild the panel for one generated token's recorded decision."""
        capture = self.capture
        if not 0 <= step < len(capture.tokens):
            return
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)
        chosen = capture.tokens[step]
        layout.addWidget(label(
            f'Step {step}  ·  selected {chosen!r}  ·  vocabulary ID '
            f'{capture.token_ids[step]}'))

        trace = capture.generation_trace
        if step >= len(trace):
            layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'not recorded')))
            layout.addWidget(label(TELEMETRY_UNAVAILABLE))
            replace_in(self._decision_slot, host)
            return

        decision = trace[step]
        tier = EVIDENCE_OBSERVED if decision.attested else EVIDENCE_UNATTESTED
        layout.addWidget(badge_row(
            EvidenceBadge(tier, 'scores from the forward pass'),
            EvidenceBadge(EVIDENCE_DERIVED, 'ranking and margin')))
        layout.addWidget(label(
            'These scores are covered by the stock-eager comparison: the prompt pass was '
            're-run with unmodified attention and produced identical logits.'
            if decision.attested else
            'Only the prompt pass is re-run under stock eager attention, so these scores '
            'are real output of the real forward pass but carry no independent check.',
            muted=True))

        rows = [(rank + 1, entry['id'], repr(str(entry.get('token', ''))),
                 f'{float(entry["logit"]):.6f}')
                for rank, entry in enumerate(decision.top_candidates)]
        winner = next((index for index, entry in enumerate(decision.top_candidates)
                       if int(entry['id']) == decision.selected_token_id), None)
        layout.addWidget(read_only_table(
            ['Rank', 'Token ID', 'Token', 'Captured score'], rows,
            stretch=2, highlight=winner))

        margin = decision.margin
        runner_up = decision.runner_up
        if margin is not None and runner_up is not None:
            layout.addWidget(label(
                f'The winner scored {margin:.6f} above {str(runner_up.get("token", ""))!r}. '
                f'That difference is a subtraction TensorScope performed for this screen; '
                f'the model only produced the scores themselves. A logit is not a '
                f'probability and no softmax was applied to pick this token.'))

        if step == 0:
            layout.addWidget(self._first_token_scores())
        replace_in(self._decision_slot, host)

    def _first_token_scores(self) -> QWidget:
        """The stored prefill score vector, which is unique to step 0."""
        logits = self.capture.logits
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        if logits is None:
            layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'not retained')))
            layout.addWidget(label(LOGITS_UNAVAILABLE))
            return host
        layout.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'final_logits')))
        layout.addWidget(label(
            f'For this first token the whole score vector was kept, not just the leaders: '
            f'{shape(logits)} — one score per vocabulary entry, stored {logits.dtype}. '
            f'argmax of the stored vector is {int(np.asarray(logits).argmax())}, which is '
            f'the token ID the capture reports; a run where those disagreed is refused on '
            f'open rather than shown.'))
        layout.addWidget(Disclosure(
            'Inspect the full captured score vector',
            lambda: self.inspector('final_logits', logits)))
        return host

    # ── Stage 5: what can and cannot be said ──────────────────────────────────

    def _limits(self) -> QWidget:
        card = self.stage_card('limits')
        card.add(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'this run')))
        card.add(label('WHAT THIS CAPTURE ESTABLISHES', muted=True))
        card.add(bullets(WHAT_WE_KNOW))

        card.section_rule()
        card.add(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'outside a forward pass')))
        card.add(label('WHAT IT CANNOT SETTLE', muted=True))
        card.add(bullets(WHAT_WE_CANNOT_CONCLUDE))

        card.section_rule()
        card.add(label('HOW EVERY NUMBER ON SCREEN IS MARKED', muted=True))
        for kind, (name, description) in EVIDENCE_KINDS.items():
            card.add(badge_row(EvidenceBadge(kind)))
            card.add(label(description, muted=True))
        card.add(label(
            'The distinction is the point of the tool: a value the model produced and a '
            'value TensorScope calculated to present it must never look alike.'))
        return card


# ── InternalsView ─────────────────────────────────────────────────────────────

class InternalsView(JourneyView):
    """The optional technical journey: purpose, diagram, equation, shapes, then tensor."""

    NAV = INTERNALS_NAV
    STAGE_INDEX = INTERNALS_STAGE_INDEX
    HEADING = 'heading'

    def builders(self) -> dict:
        return {'embedding': self._embedding_stage, 'layers': self._layers_stage,
                'head': self._head_stage, 'decode': self._decode_stage}

    def context(self) -> str:
        phase = 'first generated token' if self.generated else 'prefill'
        if self._stage_key == 'layers':
            return (f'{phase}  ·  Layer {self.layer_index}  ·  '
                    f'{LAYER_STEP_TITLES[self.step_key]}')
        return phase if self._stage_key in ('embedding', 'decode') else ''

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embedding_stage(self) -> QWidget:
        host = QWidget()
        outer = QVBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        outer.addWidget(self.phase_controls())

        array = self.embedding()
        card = self.stage_card('embedding')
        card.add(label('WHAT IT IS FOR', muted=True))
        card.add(label(
            'Integer IDs carry no usable structure — ID 5001 is not "larger" than ID 5000 '
            'in any meaningful sense. The embedding table replaces each ID with a learned '
            'vector, and every later operation works on those vectors.'))
        card.equation('token ID  →  row of the learned embedding table  →  X<br><br>'
                      '<i>The table itself is not saved, only the rows this pass used.</i>')
        card.add(label('DIMENSIONS IN THIS RUN', muted=True))
        tokens_axis = array.shape[-2] if array.ndim >= 3 else 1
        card.shape_diagram([('X', 'q', (array.shape[0], tokens_axis, array.shape[-1]))],
                           operator='=')
        if self.generated:
            card.add(label(
                f'Only the first generated token {self.capture.tokens[0]!r} is embedded in '
                f'this phase. The prompt is present through each layer\'s cached keys and '
                f'values; it does not pass through embedding again.', muted=True))
        self.tensor_disclosure(card, 'embedding', array)
        card.field('What happens next',
                   f'Layer {self.layer_indices[0]} receives this tensor and normalizes each '
                   f'token\'s vector before the attention projections.')
        outer.addWidget(card)
        return host

    # ── Layer walk ────────────────────────────────────────────────────────────

    def _layers_stage(self) -> QWidget:
        host = QWidget()
        columns = QHBoxLayout(host)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(0)

        panel = QWidget()
        panel.setFixedWidth(190)
        side = QVBoxLayout(panel)
        side.setContentsMargins(0, 0, 12, 0)
        side.setSpacing(12)
        side.addWidget(self.phase_controls())

        side.addWidget(label('LAYER', muted=True))
        row = QHBoxLayout()
        previous = QPushButton('◀')
        following = QPushButton('▶')
        self.layer_picker = QComboBox()
        self.layer_picker.setAccessibleName('Layer')
        for index in self.layer_indices:
            self.layer_picker.addItem(f'Layer {index}', index)
        self.layer_picker.setCurrentIndex(self.layer_indices.index(self.layer_index))
        self.layer_picker.currentIndexChanged.connect(self._layer_changed)
        for button, delta in ((previous, -1), (following, 1)):
            button.setFixedWidth(30)
            button.clicked.connect(
                lambda _, d=delta: self.layer_picker.setCurrentIndex(
                    self.layer_picker.currentIndex() + d))
        previous.setEnabled(self.layer_index != self.layer_indices[0])
        following.setEnabled(self.layer_index != self.layer_indices[-1])
        row.addWidget(previous)
        row.addWidget(self.layer_picker, 1)
        row.addWidget(following)
        side.addLayout(row)
        side.addWidget(label(
            f'All {len(self.layer_indices)} layers share this structure with their own '
            f'learned weights, and each keeps the hidden width so its output is the next '
            f'layer\'s input.', muted=True))

        self.step_sidebar = LayerStepSidebar(self._step_changed)
        self.step_sidebar.set_current(self.step_key)
        side.addWidget(self.step_sidebar)
        side.addStretch(1)
        columns.addWidget(panel)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self._step_card())
        right_layout.addStretch(1)
        columns.addWidget(right, 1)
        return host

    def _layer_changed(self, position: int) -> None:
        self.layer_index = self.layer_indices[position]
        self.go_to('layers')

    def _step_changed(self, key: str) -> None:
        self.step_key = key
        self.go_to('layers')

    def _step_shapes(self, tensors: dict):
        """Operands for the dimensions diagram, taken from the captured arrays only."""
        norm = tensors['normalized_input']
        q_att, k_att, v_att = (tensors[name] for name in
                               ('q_attended', 'k_attended', 'v_attended'))
        hidden = norm.shape[-1]
        queries, keys = q_att.shape[-2], k_att.shape[-2]
        head_dim, value_dim = q_att.shape[-1], v_att.shape[-1]
        step = self.step_key
        if step == 'input':
            return [('X', 'layer', (queries, hidden)),
                    ('X_norm', 'norm', (queries, hidden))], '→'
        if step == 'qkv':
            return [('Q_raw', 'q', (queries, tensors['q'].shape[-1])),
                    ('K_raw', 'k', (keys if not self.generated else tensors['k'].shape[-2],
                                    tensors['k'].shape[-1])),
                    ('V_raw', 'v', (tensors['v'].shape[-2], tensors['v'].shape[-1]))], '·'
        if step == 'prepare':
            return [('Q', 'q', q_att.shape[-3:]), ('K', 'k', k_att.shape[-3:]),
                    ('V', 'v', v_att.shape[-3:])], '·'
        if step == 'scores':
            return [('Q', 'q', (queries, head_dim)), ('Kᵀ', 'k', (head_dim, keys)),
                    ('S → A', 'scores', (queries, keys))], '@'
        if step == 'mix':
            return [('A', 'weights', (queries, keys)), ('V', 'v', (keys, value_dim)),
                    ('block output', 'output',
                     (queries, tensors['attention_output'].shape[-1]))], '@'
        return [('attention output', 'output', (queries, hidden)),
                ('X_next', 'layer', (queries, tensors['layer_output'].shape[-1]))], '→'

    def _step_card(self) -> QWidget:
        step = self.step_key
        copy = INTERNALS_STEPS[step]
        tensors = self.tensors()
        card = Card(f'Layer {self.layer_index} · {LAYER_STEP_TITLES[step]}',
                    copy['purpose'])

        card.add(label('WHAT MOVES', muted=True, small=True))
        card.add(FlowDiagram(STEP_FLOW[step], caption=copy['concept']))

        card.add(label('THE OPERATION', muted=True, small=True))
        card.equation(copy['equation'])

        card.add(label('DIMENSIONS IN THIS RUN', muted=True, small=True))
        operands, operator = self._step_shapes(tensors)
        card.shape_diagram(operands, operator)
        card.add(label(
            'Every number above is read off the captured arrays for this layer and phase, '
            'not from the config.', muted=True, small=True))

        card.section_rule()
        for name in STEP_TENSORS[step]:
            tensor = tensors.get(name)
            if tensor is None:
                continue
            self.tensor_disclosure(card, name, tensor)

        if step == 'scores':
            card.add(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL,
                                             'the mask and the 1/√d scale')))
            card.add(label(
                'The causal mask and the 1/√d scale are applied inside the attention call, '
                'so the captured scores already include them. TensorScope never captured '
                'them as separate tensors, and the masked positions are simply the '
                '−inf entries you can read in the scores above.', muted=True))
            card.add(Disclosure('Open the attention explorer for this layer',
                                lambda: self._explorer(self.layer_index)))
        if step == 'mix':
            card.add(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'A @ V')))
            card.add(label(
                'The per-head A @ V result and the concatenation that follows it live '
                'inside the attention call and are never module outputs, so they were not '
                'captured. What is captured is o_proj\'s output above — the whole block\'s '
                'contribution, after the projection.', muted=True))
        if step == 'output':
            card.add(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL,
                                             'residual sums and MLP interior')))
            card.add(label(
                'Between the attention output and the layer output the model adds the '
                'residual, normalises again, and runs the MLP. Those intermediates are not '
                'in the captured set: you are seeing the two ends of that stretch, not '
                'its middle.', muted=True))
        return card

    def _explorer(self, layer_index: int) -> QWidget:
        return AttentionExplorer(
            self.tensors(layer_index),
            token_labels(self.capture, generated=self.generated),
            token_labels(self.capture, generated=self.generated, keys=True),
            layer_index=layer_index, phase=self.phase_name)

    # ── Vocabulary scores and the next pass ───────────────────────────────────

    def _head_stage(self) -> QWidget:
        card = self.stage_card('head')
        prompt = self.capture.prompt_tokens
        last = len(prompt) - 1
        card.field('The position that decides',
                   f'prefill position {last} of {len(prompt)}, holding '
                   f'"{chip_text(prompt[last], 28)}"')
        card.add(FlowDiagram([('final position', 'layer'), ('final RMSNorm', 'norm'),
                              ('vocabulary projection', 'q'),
                              ('one score per entry', 'scores')],
                             caption='the same two operations run after every pass, on the '
                                     'last position only'))
        card.add(label('THE OPERATION', muted=True, small=True))
        card.equation('logits  =  RMSNorm(X_final[last])  W_Uᵀ<br><br>'
                      '<i>Neither the normalized vector nor the projection matrix W_U is '
                      'saved. The score vector it produced is.</i>')
        card.section_rule()

        logits = getattr(self.capture, 'logits', None)
        if logits is None:
            card.add(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'not in this run')))
            card.add(label(LOGITS_UNAVAILABLE, muted=True))
            return card

        card.add(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'final_logits')))
        card.field('Captured vector', f'{shape(logits)} · {logits.dtype}')
        chosen = int(np.asarray(logits).argmax())
        first = self.capture.tokens[0] if self.capture.tokens else '—'
        card.add(badge_row(EvidenceBadge(EVIDENCE_DERIVED, 'argmax of the vector')))
        card.add(label(
            f'The largest of the {self.facts["vocab_size"]} scores is at vocabulary id '
            f'{chosen}, which is the token the run actually emitted first '
            f'("{chip_text(first, 28)}"). TensorScope checks this agreement before it '
            f'will save a run, so a score list can never be shown beside the wrong word.',
            muted=True))
        self.tensor_disclosure(card, 'final_logits', logits)
        return card

    def _decode_stage(self) -> QWidget:
        card = self.stage_card('decode')
        card.add(FlowDiagram([('prompt', 'layer'), ('pass 1 · captured', 'q'),
                              ('first token', 'output'),
                              ('pass 2 · captured', 'k'), ('second token', 'output'),
                              ('later passes · not captured', 'norm')],
                             caption='each pass appends one token and then re-enters the '
                                     'model as input'))
        prefill = sorted(self.capture.layers)
        generated = sorted(getattr(self.capture, 'generated_layers', {}) or {})
        rows = [['prefill', f'{len(self.capture.prompt_tokens)} positions',
                 f'{len(prefill)} layers'],
                ['first generated token', '1 position', f'{len(generated)} layers']]
        remaining = max(len(self.capture.tokens) - 1, 0)
        rows.append(['remaining generation', f'{remaining} positions', 'not captured'])
        card.add(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'capture coverage')))
        card.add(read_only_table(['Phase', 'Extent', 'Tensors'], rows, stretch=0))
        card.add(label(
            'Both captured phases carry the full required tensor set for every layer; that '
            'is checked before a run can be saved. The remaining tokens were generated with '
            'capture switched off so the answer is complete to read, which is why switching '
            'phase above offers two options and not one per token.', muted=True))
        return card


# ── RawView ───────────────────────────────────────────────────────────────────

class RawView(QWidget):
    """Every captured array, reachable in two clicks.

    This is the escape hatch from both journeys: no framing, no sampling, no ordering
    imposed on the reader.  It has no stages, so `NAV` is empty and the shell hides the
    stage navigator for it; it still reports its position through `context_changed` so
    the breadcrumb keeps working.
    """

    stage_changed = pyqtSignal(str)
    context_changed = pyqtSignal(str)

    NAV: list = []

    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(parent)
        capture.validate()
        self.capture = capture
        self._tokens = tokens or {}
        self._inspector = None
        self._context = ''

        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        body.addWidget(label('Raw tensors · every captured array', title=True))
        body.addWidget(label(
            'Pick a phase, a layer and a tensor. The inspector pages through the complete '
            'array — virtual rows and columns, explicit batch and head slices — at the '
            'dtype it was stored in. Reading never rewrites it.', muted=True))
        body.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'as captured')))

        row = QHBoxLayout()
        self.phase_picker = QComboBox()
        self.phase_picker.addItems(['Phase 1 · Prompt prefill',
                                    'Phase 2 · First generated-token pass'])
        self.layer_picker = QComboBox()
        self.layer_picker.addItem('Outside the layers', None)
        for index in sorted(capture.layers):
            self.layer_picker.addItem(f'Layer {index}', index)
        self.tensor_picker = QComboBox()
        for name, picker in (('Phase', self.phase_picker), ('Layer', self.layer_picker),
                             ('Tensor', self.tensor_picker)):
            picker.setAccessibleName(name)
            row.addWidget(picker, 1)
        body.addLayout(row)

        self.meaning = label('', muted=True)
        self.meaning.setTextFormat(Qt.RichText)
        body.addWidget(self.meaning)
        self._host = QVBoxLayout()
        body.addLayout(self._host, 1)

        self.phase_picker.currentIndexChanged.connect(self._refresh_tensors)
        self.layer_picker.currentIndexChanged.connect(self._refresh_tensors)
        self.tensor_picker.currentIndexChanged.connect(self._show_tensor)
        self._refresh_tensors()

    # The shell talks to every view through these three.
    @classmethod
    def stage_keys(cls) -> list[str]:
        return []

    def current_stage(self) -> str:
        return ''

    def go_to(self, key: str) -> None:       # nothing to navigate to
        return

    def announce(self) -> None:
        """Re-emit the current position, for a shell that connected after construction."""
        self.context_changed.emit(self._context)

    @property
    def generated(self) -> bool:
        return bool(self.phase_picker.currentIndex())

    def _phase_layers(self) -> dict:
        return (self.capture.generated_layers if self.generated
                else self.capture.layers)

    def _refresh_tensors(self, *_) -> None:
        previous = self.tensor_picker.currentData()
        self.tensor_picker.blockSignals(True)
        self.tensor_picker.clear()
        layer_index = self.layer_picker.currentData()
        if layer_index is None:
            self.tensor_picker.addItem('embedding', 'embedding')
            if not self.generated and getattr(self.capture, 'logits', None) is not None:
                self.tensor_picker.addItem('final_logits', 'final_logits')
        else:
            layer = self._phase_layers().get(layer_index)
            for name in (layer.tensors if layer is not None else {}):
                self.tensor_picker.addItem(name, name)
        found = self.tensor_picker.findData(previous)
        if found >= 0:
            self.tensor_picker.setCurrentIndex(found)
        self.tensor_picker.blockSignals(False)
        self._show_tensor()

    def _array(self, name: str):
        """The stored array itself — never a copy, never a sample."""
        if name == 'embedding':
            return (self.capture.generated_embedding if self.generated
                    else self.capture.embedding), (
                'Captured embedding output for this pass. The learned lookup table itself '
                'is not persisted, only the rows this pass produced.')
        if name == 'final_logits':
            return self.capture.logits, (
                'Captured vocabulary scores at the final prompt position — the vector that '
                'chose the first generated token. Scores, not probabilities.')
        layer = self._phase_layers().get(self.layer_picker.currentData())
        if layer is None:
            return None, ''
        return layer.tensors.get(name), TENSOR_EXPLANATIONS.get(
            name, 'Additional captured tensor.')

    def _show_tensor(self, *_) -> None:
        name = self.tensor_picker.currentData()
        if name is None:
            return
        array, text = self._array(name)
        if array is None:
            return
        self.meaning.setText(text)
        if self._inspector is not None:
            self._host.removeWidget(self._inspector)
            self._inspector.hide()
            self._inspector.deleteLater()
        rows = token_labels(self.capture, self.generated,
                            keys=name in ('k_attended', 'v_attended'))
        self._inspector = TensorInspector(
            array, name=name, axes=axes_for(name, array),
            row_labels=None if name == 'final_logits' else rows,
            column_labels=(token_labels(self.capture, self.generated, keys=True)
                           if name in ('attention_scores', 'attention_weights') else None))
        self._host.addWidget(self._inspector)
        layer_index = self.layer_picker.currentData()
        where = 'outside the layers' if layer_index is None else f'Layer {layer_index}'
        phase = 'first generated token' if self.generated else 'prefill'
        self._context = f'{phase}  ·  {where}  ·  {name}'
        self.context_changed.emit(self._context)
