"""The body of each operation in the worked notebook.

Every operation is taught in the same order, so a reader learns the shape of a step once
and then only has to read what is new in it:

1. **Purpose** -- the section's one-sentence lede, set by the page.
2. **Formula** -- an `EquationRow` whose symbols are clickable variables.
3. **In / does / out** -- what representation entered, what happened, what came out.
4. **Visual action** -- a diagram with the followed position highlighted.
5. **Numerical example** -- a few labelled, rounded values chosen because they take part in
   the calculation shown, with the exact stored digits on every tooltip.
6. **Note** -- the one misunderstanding worth heading off here, if there is one.
7. **Next** -- how the result is used by the following operation.

Detail lives on the reference page (the "Explain this operation" link) or behind a
`Disclosure`, so no paragraph is said twice.

What a builder may and may not do:

* It reads captured arrays through the registry, which hands back the stored object.
  Slicing is basic indexing only, so nothing here can write through to a capture.
* Where the architecture has an intermediate this capture did not keep, the builder names
  the variable and lets the panel explain it.  There is no path from a builder to a
  fabricated tensor, because `Resolved.array` is simply `None` for those.
* Arithmetic on captured values is limited to choosing and summarising them for display --
  which entries are largest, a head-mean of one weight row, a row total, the margin between
  two scores -- plus operation 8's single worked scalar of A @ V.  All of it is badged
  derived.  Nothing here computes a quantity the architecture names and the capture
  discarded: no RMS of X, no difference of two captured tensors, no H or F.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from tensor_widgets import AttentionExplorer, exact_scalar
from tensorscope_common import (
    AnnotatedValueRow, BandedTokenStrip, Disclosure, EvidenceBadge, EvidenceLegend,
    PassTimeline, PredictCheck, ResponseTextView, StepReveal, approx, approx_share,
    badge_row, badged, bullets, chip_text, described_line, evidence_frame, evidence_mark,
    head_slice, label, measure_label, observed_line, read_only_table, replace_in,
    set_evidence, shape_line, token_labels, token_table, PROSE_WIDTH,
)
from tensorscope_content import (
    ATTENTION_SINK_THRESHOLD, CONTINUATION, OPERATION_INDEX, CACHED_KV_NOTE,
    DISPLAY_CALC_NOTE, EOS_IS_A_SCORED_ENTRY, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED,
    EVIDENCE_KINDS, EVIDENCE_OBSERVED, EVIDENCE_UNATTESTED, LOGITS_UNAVAILABLE,
    NOTEBOOK_CHECKS, NOTEBOOK_TEXT, PROMPT_SPAN_UNKNOWN, TEMPLATE_BAND,
    TELEMETRY_UNAVAILABLE, WHAT_WE_CANNOT_CONCLUDE, prompt_token_bands, readable_spelling,
    what_we_know,
)
from tensorscope_diagrams import (
    AttentionGrid, CausalMaskTriangle, DecisionChain, EmbeddingLookupRow, GatedFFNPath,
    HeadSplit, KVShare, LayerStack, MatMulShapes, MatrixGrid, PositionScope, QKCompare,
    ResidualBypass, SoftmaxBars, VectorStrip, VocabularyProjection,
)
from tensorscope_variables import EquationRow, VariableChip


# How many entries a numerical example shows.  Small on purpose: these are chosen because
# they take part in the illustrated calculation, and the whole array is one click away.
TRACKED_FEATURES = 4
BAR_KEYS = 12
MIX_KEYS = 6


# ── Contextual controls ─────────────────────────────────────────────────────────
# The head, layer and phase pickers used to sit in a ribbon above the whole page, all of
# them, all the time.  Now each appears in the operation that explains it.  Several copies
# of one picker feed the same registry, so every copy syncs from it with its own signals
# blocked -- a copy that echoed would re-enter `set_context` and emit twice -- and each sync
# is a bound method, so Qt disconnects it when the widget is destroyed.

class HeadPicker(QWidget):
    """'Head [n] of N', bound to the registry."""

    def __init__(self, registry, count: int, parent=None):
        super().__init__(parent)
        self._registry = registry
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(label('Head', small=True))
        self.combo = QComboBox()
        self.combo.setAccessibleName('Attention head')
        self.combo.setToolTip('Which attention head the pictures and numbers describe.')
        for index in range(max(1, int(count))):
            self.combo.addItem(str(index), index)
        self.combo.currentIndexChanged.connect(self._picked)
        row.addWidget(self.combo)
        row.addWidget(label(f'of {max(1, int(count))}', small=True))
        row.addStretch(1)
        registry.context_changed.connect(self.sync)
        self.sync()

    def _picked(self, position: int) -> None:
        self._registry.set_context(head=self.combo.itemData(position))

    def sync(self) -> None:
        value = min(self._registry.head, self.combo.count() - 1)
        if self.combo.currentIndex() != value:
            blocked = self.combo.signalsBlocked()
            self.combo.blockSignals(True)
            try:
                self.combo.setCurrentIndex(value)
            finally:
                self.combo.blockSignals(blocked)

    def setFocus(self, *args):                              # noqa: N802 - Qt name
        self.combo.setFocus(*args)

    def accessibleName(self) -> str:                        # noqa: N802 - Qt name
        return self.combo.accessibleName()


class LayerPicker(QWidget):
    """'◀ Layer [n] ▶', bound to the registry."""

    def __init__(self, registry, parent=None):
        super().__init__(parent)
        self._registry = registry
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.back = QPushButton('◀')
        self.back.setAccessibleName('Previous layer')
        self.back.setObjectName('revealControl')
        self.combo = QComboBox()
        self.combo.setAccessibleName('Layer')
        self.combo.setToolTip('Which layer operations 4 to 11 describe.')
        for index in registry.layer_indices:
            self.combo.addItem(f'Layer {index}', index)
        self.forward = QPushButton('▶')
        self.forward.setAccessibleName('Next layer')
        self.forward.setObjectName('revealControl')
        self.back.clicked.connect(lambda: self._step(-1))
        self.forward.clicked.connect(lambda: self._step(1))
        self.combo.currentIndexChanged.connect(self._picked)
        for widget in (self.back, self.combo, self.forward):
            row.addWidget(widget)
        row.addStretch(1)
        registry.context_changed.connect(self.sync)
        self.sync()

    def _picked(self, position: int) -> None:
        self._registry.set_context(layer_index=self.combo.itemData(position))

    def _step(self, amount: int) -> None:
        indices = self._registry.layer_indices
        if not indices:
            return
        position = indices.index(self._registry.layer_index) + amount
        if 0 <= position < len(indices):
            self._registry.set_context(layer_index=indices[position])

    def sync(self) -> None:
        indices = self._registry.layer_indices
        value = (indices.index(self._registry.layer_index)
                 if self._registry.layer_index in indices else 0)
        if self.combo.currentIndex() != value:
            blocked = self.combo.signalsBlocked()
            self.combo.blockSignals(True)
            try:
                self.combo.setCurrentIndex(value)
            finally:
                self.combo.blockSignals(blocked)
        self.back.setEnabled(bool(indices) and self._registry.layer_index != indices[0])
        self.forward.setEnabled(bool(indices) and self._registry.layer_index != indices[-1])

    def accessibleName(self) -> str:                        # noqa: N802 - Qt name
        return self.combo.accessibleName()


class PhaseToggle(QPushButton):
    """Switch operations 4-11 between the two passes that kept layer tensors."""

    def __init__(self, registry, parent=None):
        super().__init__(parent)
        self._registry = registry
        self.setObjectName('revealControl')
        self.setCheckable(True)
        self.setAccessibleName('Captured phase')
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.toggled.connect(lambda on: self._registry.set_context(generated=bool(on)))
        registry.context_changed.connect(self.sync)
        self.sync()

    def sync(self) -> None:
        on = bool(self._registry.generated)
        if self.isChecked() != on:
            blocked = self.signalsBlocked()
            self.blockSignals(True)
            try:
                self.setChecked(on)
            finally:
                self.blockSignals(blocked)
        self.setText(NOTEBOOK_TEXT['phase_toggle_off' if on else 'phase_toggle_on'])


class NotebookContext:
    """Everything a builder needs, and the small factories they all share."""

    def __init__(self, capture, facts, registry, *, tokens=None, state=None):
        self.capture = capture
        self.facts = facts
        self.registry = registry
        self.tokens = tokens or {}
        self.state = state if state is not None else {}
        self.bands = prompt_token_bands(capture.prompt, capture.prompt_tokens)
        # The contextual controls each operation built, by kind, so the page can report
        # them (and the tests can check they appear only where they explain something).
        self.pickers: dict[str, list] = {'head': [], 'layer': [], 'phase': []}

    # -- copy -----------------------------------------------------------------

    def text(self, template: str, **extra) -> str:
        return template.format(**{**self.facts, **extra})

    def say(self, which: str, /, **extra) -> str:
        """A NOTEBOOK_TEXT caption, formatted against this run."""
        return NOTEBOOK_TEXT[which].format(**{**self.facts, **extra})

    # -- the parts every operation is made of ---------------------------------

    def equation(self, terms) -> QWidget:
        return EquationRow(self.registry, terms)

    def chip(self, key: str) -> VariableChip:
        return VariableChip(self.registry, key)

    def io_row(self, operation) -> QWidget:
        """IN (operands) · DOES (the operation) · OUT (results), in one line.

        Registry keys become compact chips that open the variable panel; anything else --
        "your text", "token IDs" -- is written as plain text, because there is no array
        behind it to open.
        """
        host = QWidget()
        host.setObjectName('ioRow')
        row = QHBoxLayout(host)
        row.setContentsMargins(14, 8, 14, 8)
        row.setSpacing(10)

        def terms(keys) -> None:
            for key in keys:
                if self.registry.known(key):
                    chip = VariableChip(self.registry, key, compact=True)
                    chip.setObjectName('equationTerm')
                    row.addWidget(chip)
                else:
                    plain = label(self.text(key), small=True)
                    plain.setObjectName('ioPlain')
                    plain.setWordWrap(False)
                    row.addWidget(plain)

        for heading, part in (('io_in', operation.inputs), ('io_does', None),
                              ('io_out', operation.outputs)):
            tag = label(NOTEBOOK_TEXT[heading], muted=True)
            tag.setObjectName('ioTag')
            tag.setWordWrap(False)
            row.addWidget(tag)
            if part is None:
                action = label(self.text(operation.action), small=True)
                action.setObjectName('ioAction')
                action.setWordWrap(False)
                row.addWidget(action)
                row.addWidget(label('→', muted=True))
            else:
                terms(part)
                if heading == 'io_in':
                    row.addWidget(label('→', muted=True))
        row.addStretch(1)
        host.setAccessibleName(
            f'In: {", ".join(operation.inputs)}. Does: {self.text(operation.action)}. '
            f'Out: {", ".join(operation.outputs)}.')
        return host

    def note(self, operation) -> QWidget | None:
        """The one misunderstanding worth heading off at this step, as a callout."""
        if not operation.note:
            return None
        callout = measure_label(self.text(operation.note), width=PROSE_WIDTH + 40)
        callout.setObjectName('teachingNote')
        return callout

    def takeaway(self, operation) -> QWidget | None:
        """How the result is used next -- the last line of every operation."""
        if not operation.takeaway:
            return None
        line = label(f'{NOTEBOOK_TEXT["takeaway_prefix"]}   '
                     f'{self.text(operation.takeaway)}', small=True)
        line.setObjectName('takeawayLine')
        return line

    def variables(self, keys) -> QWidget:
        """The operation's variable list: one clickable row each, shapes and provenance."""
        host = QWidget()
        rows = QVBoxLayout(host)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(3)
        for key in keys:
            if self.registry.known(key):
                rows.addWidget(self.chip(key), 0, Qt.AlignLeft)
        return host

    def variable_list(self, keys, *, note: str = '') -> Disclosure:
        """The full variable list, collapsed: the formula and the in/out line carry the
        operands, and this is where each one's shape and provenance are spelled out --
        with any qualification about what the capture holds for them."""
        def build() -> QWidget:
            host = self.variables(keys)
            if note:
                host.layout().addWidget(label(note, muted=True, small=True))
            return host
        return self.disclosure('Every variable in this step, with its shape and source',
                               build, remember=True)

    def check(self, name: str, **extra) -> QWidget | None:
        """An optional check, or None where this run cannot fill one in honestly."""
        copy = NOTEBOOK_CHECKS.get(name)
        if copy is None:
            return None
        values = {**self.facts, **extra}
        missing = [field for field in copy['fields'] if field not in extra]
        if missing:
            return None
        return PredictCheck(
            copy['question'].format(**values),
            [option.format(**values) for option in copy['options']],
            copy['answer'], copy['explanation'].format(**values),
            state=self.state.setdefault('checks', {}), state_key=name)

    def add_check(self, body, name: str, **extra) -> None:
        """An optional check, collapsed behind one control so it never crowds the lesson.

        Its answer is kept in the notebook's state dict, so it survives being rebuilt and
        survives navigation either way.
        """
        if self.check(name, **extra) is None:
            return
        body.addWidget(self.disclosure(
            'Check your understanding (optional)', remember=True,
            build=lambda: self.check(name, **extra)))

    def values(self, array, *, labels=None, caption='', kind=EVIDENCE_OBSERVED, note='',
               total=None, formatter=None, details=None, max_boxes=None) -> AnnotatedValueRow:
        """A few values, rounded for reading; every box carries the exact stored digits."""
        shown = list(np.asarray(array).reshape(-1))
        return AnnotatedValueRow(
            shown, labels=labels, caption=caption.upper(), kind=kind, note=note,
            total=total if total is not None else int(np.asarray(array).size),
            formatter=formatter or approx, details=details, max_boxes=max_boxes)

    def disclosure(self, title: str, build, *, remember: bool = False) -> Disclosure:
        """A collapsed body.  `remember` only for cheap bodies -- never a tensor widget."""
        return Disclosure(title, build,
                          state=self.state.setdefault('open', {}) if remember else None,
                          state_key=title)

    # -- contextual controls --------------------------------------------------

    def head_picker(self) -> QWidget:
        picker = HeadPicker(self.registry, int(self.facts['query_head_count']) or 1)
        self.pickers['head'].append(picker)
        return picker

    def layer_picker(self) -> QWidget:
        picker = LayerPicker(self.registry)
        self.pickers['layer'].append(picker)
        return picker

    def phase_toggle(self) -> QWidget:
        toggle = PhaseToggle(self.registry)
        self.pickers['phase'].append(toggle)
        return toggle

    # -- captured arrays ------------------------------------------------------

    def array(self, key: str):
        return self.registry.resolve(key).array

    def row_index(self, key: str, *, position=None) -> int:
        """Which row of a captured array the current position actually reaches.

        The axes differ in length between the phases -- one query row against many cached
        key rows -- so a caption that repeated the selection would sometimes name a row
        that was not the one read.
        """
        array = self.array(key)
        index = self.registry.position if position is None else position
        if array is None:
            return max(0, index)
        view = np.asarray(array)
        if view.ndim < 2:
            return 0
        return min(max(0, index), int(view.shape[-2]) - 1)

    def row(self, key: str, *, position=None, head=None):
        """One row of a captured array for the current selection, or None.

        Basic indexing throughout: the result borrows the stored buffer.
        """
        array = self.array(key)
        if array is None:
            return None
        view = np.asarray(array)
        index = self.registry.position if position is None else position
        slot = self.registry.head if head is None else head
        if view.ndim == 3:
            return view[0, min(max(0, index), view.shape[-2] - 1)]
        if view.ndim == 4:
            return view[0, min(max(0, slot), view.shape[1] - 1),
                        min(max(0, index), view.shape[-2] - 1)]
        if view.ndim == 1:
            return view
        return None

    def tracked_features(self, count: int = TRACKED_FEATURES) -> list[int]:
        """The feature indices followed through operations 2, 4, 9 and 10.

        The largest-magnitude coordinates of the followed row of this layer's input, in
        index order.  A display choice -- a ranking of captured values -- made once per
        selection so the same coordinates can be watched changing from step to step.
        """
        values = self.row('layer_input')
        if values is None or not np.asarray(values).size:
            return []
        magnitude = np.abs(np.asarray(values, dtype=np.float64))
        chosen = np.argsort(-magnitude, kind='stable')[:count]
        return sorted(int(index) for index in chosen)

    def feature_values(self, key: str, features: list[int]):
        row = self.row(key)
        if row is None:
            return None
        view = np.asarray(row)
        return [view[index] for index in features if index < view.shape[-1]]

    def spelling(self, index: int) -> str:
        tokens = self.capture.prompt_tokens
        return tokens[index] if 0 <= index < len(tokens) else ''

    def key_spellings(self) -> list[str]:
        """Readable spelling of every key position in the current phase, for labels."""
        spellings = [chip_text(readable_spelling(token), 7)
                     for token in self.capture.prompt_tokens]
        if self.registry.generated and self.capture.tokens:
            spellings.append(chip_text(self.capture.tokens[0], 7))
        return spellings

    def on_selection(self, callback) -> None:
        """Re-run a slot's builder whenever phase, layer, head or position changes."""
        self.registry.context_changed.connect(callback)
        callback()

    def slot(self, body, build) -> QVBoxLayout:
        """A region of `body` that rebuilds itself whenever the selection changes.

        It attaches at the point in the builder's own layout where it is called, and that
        is the whole reason it takes the layout rather than the section: a slot appended to
        the section lands *before* the body the builder returns, so the reader would meet
        an operation's numbers above its own equation.

        Rebuilding a slot rather than mutating labels in place keeps every builder a pure
        function of the selection, which is what makes them safe to drive from a test.
        """
        holder = self.plain_slot(body)
        self.on_selection(lambda: replace_in(holder, build()))
        return holder

    def plain_slot(self, body) -> QVBoxLayout:
        """A region of `body` the caller refreshes itself, on a signal of its own."""
        holder = QVBoxLayout()
        holder.setContentsMargins(0, 0, 0, 0)
        holder.setSpacing(8)
        body.addLayout(holder)
        return holder

    def finish(self, body, operation) -> None:
        """The closing lines every operation shares: its note, then what comes next."""
        for widget in (self.note(operation), self.takeaway(operation)):
            if widget is not None:
                body.addWidget(widget)


def _panel(spacing: int = 8) -> tuple[QWidget, QVBoxLayout]:
    host = QWidget()
    body = QVBoxLayout(host)
    body.setContentsMargins(0, 2, 0, 0)
    body.setSpacing(spacing)
    return host, body


def _side_by_side(*widgets: QWidget) -> QWidget:
    host = QWidget()
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(12)
    for widget in widgets:
        row.addWidget(widget, 1)
    return host


def _operation(key: str):
    return OPERATION_INDEX.get(key) or CONTINUATION


def _threshold_text() -> str:
    return f'{ATTENTION_SINK_THRESHOLD:.0%}'


def sink_callout(ctx: NotebookContext, row, *, averaged: bool = False) -> QWidget | None:
    """The attention-sink note, or None when the weight row on screen does not trip it.

    Reads exactly one number: the row's weight on position 0, which is a captured value
    (one head's row) or a head-mean of captured values (`averaged`, badged derived).  The
    rule that decides whether the note appears is stated inside the note itself.
    """
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values[0]):
        return None
    share = float(values[0])
    if share < ATTENTION_SINK_THRESHOLD:
        return None
    phrase = (ctx.say('mean_weight_share', percent=f'{share * 100:.1f}%') if averaged
              else approx_share(share))
    host, lines = _panel(4)
    if averaged:
        lines.addWidget(badge_row(EvidenceBadge(EVIDENCE_DERIVED, 'head-mean')))
    note = measure_label(ctx.say(
        'attention_sink', spelling=repr(readable_spelling(ctx.spelling(0))), share=phrase),
        width=PROSE_WIDTH + 40)
    note.setObjectName('sinkNote')
    note.setToolTip(f'Exact weight on position 0: '
                    f'{repr(share) if averaged else exact_scalar(row[0])}')
    lines.addWidget(note)
    lines.addWidget(_sink_rule(ctx))
    host.setObjectName('sinkCallout')
    return host if averaged else set_evidence(host, EVIDENCE_OBSERVED)


def _sink_rule(ctx: NotebookContext) -> QWidget:
    """When the note appears, as a smaller secondary line beneath it."""
    rule = label(ctx.say('attention_sink_rule', threshold=_threshold_text()),
                 muted=True, small=True)
    rule.setObjectName('sinkRule')
    return rule


def _tracked(ctx: NotebookContext, lines: QVBoxLayout, rows) -> None:
    """The followed features of one or more captured rows, under one derived badge.

    `rows` is a list of (registry key, symbol).  The values are observed; which features
    are shown is TensorScope's choice, and the badge on the caption says so once.
    """
    features = ctx.tracked_features()
    if not features:
        return
    position = ctx.row_index('layer_input')
    line = badged(EvidenceBadge(EVIDENCE_DERIVED, 'features picked for display'),
                  ctx.say('tracked_features_caption', count=len(features), position=position))
    line.words.setToolTip(ctx.say('tracked_features_note'))
    lines.addWidget(line)
    names = [f'feature {index}' for index in features]
    for key, symbol in rows:
        values = ctx.feature_values(key, features)
        if values is None:
            continue
        lines.addWidget(ctx.values(values, labels=names,
                                   caption=f'{symbol}, position {ctx.row_index(key)}',
                                   note=key, total=len(values)))


# ── 1  The model's input ─────────────────────────────────────────────────────

def build_input(ctx: NotebookContext, section) -> QWidget:
    capture = ctx.capture
    operation = _operation('input')
    host, body = _panel()
    body.addWidget(ctx.io_row(operation))

    strip = BandedTokenStrip(capture.prompt_tokens, ctx.bands,
                             selected=ctx.registry.position)
    strip.selected.connect(lambda index: ctx.registry.set_context(position=index))
    body.addWidget(label(ctx.say('focus_hint'), muted=True, small=True))
    body.addWidget(strip)

    def follow() -> None:
        # The strip mirrors the registry without echoing: a selection made in the reading
        # bar's Following control moves the highlighted chip here too.
        try:
            if strip.current() != ctx.registry.position:
                blocked = strip.signalsBlocked()
                strip.blockSignals(True)
                try:
                    strip.select(ctx.registry.position)
                finally:
                    strip.blockSignals(blocked)
        except RuntimeError:                    # the page is being torn down
            pass

    ctx.registry.context_changed.connect(follow)
    if ctx.bands is None:
        # No alignment, so no claim about which tokens were typed: nothing is faded and
        # no count is given.
        body.addWidget(label(PROMPT_SPAN_UNKNOWN, muted=True, small=True))
    elif int(ctx.facts.get('template_token_count') or 0):
        why = measure_label(ctx.say('template_tokens_why'))
        why.setObjectName('templateTokensWhy')
        body.addWidget(why)
    body.addWidget(label(ctx.say('control_markers'), muted=True, small=True))

    def selected() -> QWidget:
        index = ctx.registry.position
        inner, lines = _panel(4)
        if not (0 <= index < len(capture.prompt_tokens)):
            lines.addWidget(label('No position selected.', muted=True))
            return inner
        stored = capture.prompt_tokens[index]
        band = next((name for start, end, name in (ctx.bands or [])
                     if start <= index < end), '')
        lines.addWidget(observed_line(
            f'Position {index}   ·   token ID {capture.prompt_token_ids[index]}   ·   '
            f'reads as {readable_spelling(stored)!r}   ·   stored {stored!r}'))
        if band == TEMPLATE_BAND:
            lines.addWidget(label(ctx.say('template_position'), muted=True, small=True))
        return inner

    ctx.slot(body, selected)
    ctx.finish(body, operation)
    body.addWidget(ctx.disclosure('Show every position as a table',
                                  lambda: token_table(capture), remember=True))
    return host


# ── 2  Embedding lookup ─────────────────────────────────────────────────────

def build_embed(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('embed')
    host, body = _panel()
    body.addWidget(ctx.equation(('E', '[', 'token ID', ']', '→', 'embedding')))
    body.addWidget(ctx.io_row(operation))

    def row() -> QWidget:
        index = ctx.registry.position
        inner, lines = _panel(6)
        identifier = (ctx.capture.prompt_token_ids[index]
                      if 0 <= index < len(ctx.capture.prompt_token_ids) else None)
        lines.addWidget(EmbeddingLookupRow(
            int(ctx.facts.get('vocab_size_raw') or 0), int(ctx.facts['hidden_size']),
            token_id=identifier, token_text=ctx.spelling(index), tokens=ctx.tokens,
            caption='The ID picks a row. The row is the vector, as wide as the model, not '
                    'as wide as the vocabulary.'))
        if ctx.row('embedding') is None:
            lines.addWidget(label('This capture’s embedding is not shaped '
                                  '[batch, token, feature], so a single row cannot be '
                                  'picked out of it.', muted=True))
            return inner
        _tracked(ctx, lines, [('embedding', 'E[id]')])
        return inner

    ctx.slot(body, row)
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables))
    ctx.add_check(body, 'embed')
    return host


# ── 3  The input tensor ─────────────────────────────────────────────────────

def build_shape(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('shape')
    host, body = _panel()
    body.addWidget(ctx.io_row(operation))

    def shape_block() -> QWidget:
        array = ctx.array('layer_input')
        inner, lines = _panel(6)
        if array is None:
            lines.addWidget(label('This phase has no layer input to describe.', muted=True))
            return inner
        written, caption = shape_line(array, omit_batch=False)
        formula = label(f'shape(X) = {written}', rich=False)
        formula.setObjectName('journeyEquation')
        lines.addWidget(formula)
        view = np.asarray(array)
        lines.addWidget(MatrixGrid(
            int(view.shape[-2]), int(view.shape[-1]), row_axis='token position',
            col_axis='model feature', role='layer', symbol='X',
            highlight_row=ctx.row_index('layer_input'), tokens=ctx.tokens,
            caption=ctx.say('row_highlight')))
        if caption:
            lines.addWidget(label(caption, muted=True, small=True))
        return inner

    ctx.slot(body, shape_block)
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables))
    ctx.add_check(body, 'shape')
    return host


# ── 4  Input normalization ──────────────────────────────────────────────────

def build_norm(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('norm')
    host, body = _panel()
    body.addWidget(ctx.equation(('normalized_input', '=', 'RMSNorm', '(', 'layer_input',
                                 ')')))
    body.addWidget(ctx.io_row(operation))

    def comparison() -> QWidget:
        inner, lines = _panel(6)
        before, after = ctx.array('layer_input'), ctx.array('normalized_input')
        if before is None or after is None:
            lines.addWidget(label('This phase did not capture both sides of the '
                                  'normalization.', muted=True))
            return inner
        first, second = np.asarray(before), np.asarray(after)
        row = ctx.row_index('layer_input')
        lines.addWidget(_side_by_side(
            MatrixGrid(int(first.shape[-2]), int(first.shape[-1]),
                       row_axis='position', col_axis='model feature', role='layer',
                       symbol='X', highlight_row=row, tokens=ctx.tokens),
            MatrixGrid(int(second.shape[-2]), int(second.shape[-1]),
                       row_axis='position', col_axis='model feature', role='norm',
                       symbol='X_norm', highlight_row=row, tokens=ctx.tokens)))
        _tracked(ctx, lines, [('layer_input', 'X'), ('normalized_input', 'X_norm')])
        return inner

    ctx.slot(body, comparison)
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables))
    return host


# ── 5  Raw projections ──────────────────────────────────────────────────────

def _head_slice_of(ctx: NotebookContext, key: str, count: int = TRACKED_FEATURES):
    """The first `count` features of the selected head's slice of a raw projection row.

    Raw Q is [positions × heads·head_dim]; head h owns features h·d … h·d + d − 1.  Showing
    the start of that slice here and the start of the prepared head in operation 6 puts the
    same features before and after the rotation side by side.
    """
    row = ctx.row(key)
    head_dim = int(ctx.facts['head_dim'] or 0)
    if row is None or not head_dim:
        return None, []
    head = min(ctx.registry.head, max(0, np.asarray(row).shape[-1] // head_dim - 1))
    start = head * head_dim
    indices = list(range(start, min(start + count, np.asarray(row).shape[-1])))
    return [np.asarray(row)[index] for index in indices], indices


def build_project(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('project')
    host, body = _panel()
    body.addWidget(ctx.equation(('q', '=', 'normalized_input', '@', 'W_Q')))
    body.addWidget(ctx.io_row(operation))
    body.addWidget(label(ctx.say('projection_widths'), muted=True, small=True))

    def shapes() -> QWidget:
        inner, lines = _panel(6)
        normalized, projected = ctx.array('normalized_input'), ctx.array('q')
        if normalized is None or projected is None:
            lines.addWidget(label('This phase did not capture the projection.', muted=True))
            return inner
        left, out = np.asarray(normalized), np.asarray(projected)
        lines.addWidget(MatMulShapes(
            (int(left.shape[-2]), int(left.shape[-1])),
            (int(out.shape[-1]), int(left.shape[-1])),
            (int(out.shape[-2]), int(out.shape[-1])),
            left_symbol='X_norm', right_symbol='W_Qᵀ', out_symbol='Q_raw',
            tokens=ctx.tokens,
            caption='The highlighted edge is the dimension both operands share; it is why '
                    'the multiplication is defined. The weight is written transposed '
                    'because that is how a linear layer stores it.'))
        # Width is what changes here: one row in, one wider row out.
        lines.addWidget(_side_by_side(
            VectorStrip(int(left.shape[-1]), symbol='X_norm', role='norm',
                        tokens=ctx.tokens),
            VectorStrip(int(out.shape[-1]), symbol='Q_raw', role='q', tokens=ctx.tokens)))
        values, indices = _head_slice_of(ctx, 'q')
        if values:
            lines.addWidget(ctx.values(
                values, labels=[f'feature {index}' for index in indices],
                caption=ctx.say('head_slice_caption', count=len(values),
                                head=ctx.registry.head, position=ctx.row_index('q'))
                + ' of Q_raw', note='q', total=int(out.shape[-1])))
        return inner

    ctx.slot(body, shapes)
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables))
    return host


# ── 6  Preparing for attention ──────────────────────────────────────────────

def build_prepare(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('prepare')
    facts = ctx.facts
    host, body = _panel()
    body.addWidget(ctx.equation(('q', '→', 'q_attended')))
    body.addWidget(ctx.io_row(operation))
    body.addWidget(ctx.head_picker())

    def split() -> QWidget:
        inner, lines = _panel(6)
        # The reader's position stays highlighted: the point of this picture is that the
        # positions survive both stages, which is only visible if *their* row survives.
        lines.addWidget(HeadSplit(
            int(facts['hidden_size']), int(facts['q_projection_width']),
            int(facts['query_head_count']), int(facts['head_dim']),
            int(facts['position_count']), highlight_head=ctx.registry.head,
            highlight_row=ctx.row_index('q_attended'), tokens=ctx.tokens,
            caption='The projection computed new numbers; the reshape only regrouped '
                    'them. The highlighted row is the position you are following.'))
        if int(facts['kv_head_count']):
            lines.addWidget(KVShare(
                int(facts['kv_head_count']), int(facts['query_head_count']),
                highlight_query_head=ctx.registry.head, tokens=ctx.tokens,
                caption=ctx.say('kv_sharing')))
        before, indices = _head_slice_of(ctx, 'q')
        after = ctx.row('q_attended')
        if before and after is not None:
            names = [f'feature {index - indices[0]} of head {ctx.registry.head}'
                     for index in indices]
            lines.addWidget(ctx.values(
                before, labels=names, note='q', total=len(ctx.row('q')),
                caption=f'Q_raw, head {ctx.registry.head}, position {ctx.row_index("q")}'
                        ' — before'))
            lines.addWidget(ctx.values(
                np.asarray(after)[:len(before)], labels=names, note='q_attended',
                total=len(after),
                caption=f'Q, head {ctx.registry.head}, position '
                        f'{ctx.row_index("q_attended")} — after the rotation'))
        return inner

    ctx.slot(body, split)
    if facts.get('head_width_decoupled'):
        body.addWidget(label(ctx.say('head_width_decoupled'), muted=True, small=True))
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables, note=ctx.say('rotation_unrecorded')))
    return host


# ── 7  Scores and weights ───────────────────────────────────────────────────

def _masked_columns(score_row) -> list[int]:
    """Which columns of a captured score row were masked.

    Not assumed to be negative infinity, and not assumed to be the upper triangle either:
    the mask is an additive constant whose magnitude depends on the dtype, so the test is
    simply "far below the rest of this row".  Columns the row does not reach are not
    guessed about at all.
    """
    row = np.asarray(score_row, dtype=np.float64).reshape(-1)
    finite = row[np.isfinite(row)]
    if finite.size == 0:
        return []
    live = finite[finite > np.finfo(np.float32).min / 4]
    if live.size == 0:
        return []
    floor = float(live.min()) - 1000.0
    return [index for index, value in enumerate(row)
            if not np.isfinite(value) or value < floor]


def _heaviest(weight_row, allowed, count: int) -> list[int]:
    """The `count` allowed keys with the most weight, back in position order."""
    weights = np.asarray(weight_row, dtype=np.float64)
    ranked = sorted(allowed, key=lambda index: (-weights[index], index))
    return sorted(ranked[:count])


def build_scores(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('scores')
    host, body = _panel()
    body.addWidget(ctx.equation(('attention_scores', '→ softmax →',
                                 'attention_weights')))
    body.addWidget(ctx.io_row(operation))
    body.addWidget(ctx.head_picker())
    # `ctx.slot` hands its layout back after running the builder once, so a click that
    # only changes this operation's own key column has something to refresh.
    region: dict = {}

    def panel() -> QWidget:
        inner, lines = _panel(8)
        scores, weights = ctx.array('attention_scores'), ctx.array('attention_weights')
        if scores is None or weights is None:
            lines.addWidget(label('This phase did not capture the attention matrices.',
                                  muted=True))
            return inner
        score_view, weight_view = np.asarray(scores), np.asarray(weights)
        queries, keys = int(score_view.shape[-2]), int(score_view.shape[-1])
        head = min(ctx.registry.head, int(score_view.shape[1]) - 1)
        query = min(ctx.registry.position, queries - 1)
        cached = max(keys - queries, 0)
        spellings = ctx.key_spellings()

        key_choice = min(max(0, ctx.state.setdefault('key_position', 0)), keys - 1)
        mask = CausalMaskTriangle(
            queries, keys, selected_query=query, selected_key=key_choice, cached=cached,
            clickable=True, tokens=ctx.tokens, caption=ctx.say('mask_caption'))

        def pick(row: int, column: int) -> None:
            # The key choice is this operation's own, so it is remembered in the notebook's
            # state rather than in the registry: no other operation has a key axis.
            ctx.state['key_position'] = int(column)
            if queries > 1:
                ctx.registry.set_context(position=row)
            # One query row -- the generated pass -- has nothing to select: the followed
            # prompt position must not silently become position 0 for every other step.
            if region.get('holder') is not None and (queries == 1
                                                     or row == ctx.registry.position):
                replace_in(region['holder'], panel())

        mask.cellClicked.connect(pick)
        lines.addWidget(mask)

        # The two operands of the selected cell, with the shared head width bracketed
        # across both -- which is the thing a written dot product hides.
        lines.addWidget(QKCompare(
            int(ctx.facts['head_dim']), query_pos=query, key_pos=key_choice,
            query_label=(ctx.capture.tokens[0] if ctx.registry.generated and ctx.capture.tokens
                         else ctx.spelling(query)),
            key_label=spellings[key_choice] if key_choice < len(spellings) else '',
            tokens=ctx.tokens, caption=ctx.say('compare_caption')))
        if cached:
            lines.addWidget(label(ctx.say('cached_keys'), muted=True, small=True))

        score_row = score_view[0, head, query]
        weight_row = weight_view[0, head, query]
        masked = _masked_columns(score_row)
        allowed = [index for index in range(len(score_row)) if index not in masked]
        shown_keys = (allowed if len(allowed) <= BAR_KEYS
                      else _heaviest(weight_row, allowed, BAR_KEYS))
        lines.addWidget(SoftmaxBars(
            score_row, weight_row, labels=[str(index) for index in range(len(score_row))],
            sublabels=spellings, indices=shown_keys, masked_indices=masked,
            selected=key_choice, tokens=ctx.tokens,
            caption=(ctx.say('bars_caption_all', head=head, query=query)
                     if len(shown_keys) == len(allowed) else
                     ctx.say('bars_caption', head=head, query=query,
                             count=len(shown_keys)))))

        # The heaviest weight, read the way a person reads it, with its exact stored
        # digits on the tooltip.  The bars above carry every other shown key.
        heaviest = max(allowed, key=lambda index: (float(weight_row[index]), -index))
        weight = weight_row[heaviest]
        sink = sink_callout(ctx, weight_row)
        # When the sink note shows and position 0 is also the heaviest key, the note
        # already names it and its share, so the heaviest-key line would say it twice.  If
        # some other key is heavier still, that line stays: the note alone would let a
        # reader assume position 0 led the row.
        if sink is None or heaviest != 0:
            line = observed_line(ctx.say(
                'heaviest_key', key=heaviest,
                spelling=repr(spellings[heaviest]) if heaviest < len(spellings) else '',
                weight=approx(weight), percent=f'{float(weight) * 100:.1f}%'))
            line.words.setToolTip(f'Exact stored value: {exact_scalar(weight)}')
            lines.addWidget(line)
        if sink is not None:
            lines.addWidget(sink)
        total = float(np.asarray(weight_row, dtype=np.float64).sum())
        lines.addWidget(badged(
            EvidenceBadge(EVIDENCE_DERIVED, 'row total'),
            f'{ctx.say("weight_row_total")}: {approx(total)}, added up here for the display. '
            + ctx.say('scores_scaled_masked')))

        head_matrix = head_slice(weight_view, head)
        lines.addWidget(ctx.disclosure(
            ctx.say('head_grid_title', head=head),
            lambda matrix=head_matrix, h=head: AttentionGrid(
                matrix, selected_query=query, tokens=ctx.tokens,
                caption=f'Head {h}. Every captured weight for this head, painted rather '
                        f'than plotted.')))
        lines.addWidget(ctx.disclosure(ctx.say('explorer_title'), lambda: _explorer(ctx)))
        return inner

    region['holder'] = ctx.slot(body, panel)
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables))
    ctx.add_check(body, 'scores')
    return host


def _explorer(ctx: NotebookContext) -> QWidget:
    layers = ctx.registry.layers()
    layer = layers.get(ctx.registry.layer_index)
    if layer is None:
        return label('This layer is not in the capture.', muted=True)
    return AttentionExplorer(
        layer.tensors, token_labels(ctx.capture, ctx.registry.generated),
        token_labels(ctx.capture, ctx.registry.generated, keys=True),
        layer_index=ctx.registry.layer_index, phase=ctx.registry.phase_name)


# ── 8  Mixing the values ────────────────────────────────────────────────────

def mix_example(ctx: NotebookContext):
    """The worked example's numbers: which keys are shown, and every product.

    Returns None when this phase lacks the operands.  Otherwise a dict with the shown keys
    (the heaviest, in position order), the remaining keys, and float64 weights, values and
    products for all of them -- so the shown products plus the remainder *are* the total,
    not an approximation of it.
    """
    weights, values = ctx.array('attention_weights'), ctx.array('v_attended')
    if weights is None or values is None:
        return None
    weight_view, value_view = np.asarray(weights), np.asarray(values)
    head = min(ctx.registry.head, int(weight_view.shape[1]) - 1)
    query = min(ctx.registry.position, int(weight_view.shape[-2]) - 1)
    weight_row = np.asarray(weight_view[0, head, query], dtype=np.float64)
    value_column = np.asarray(value_view[0, min(head, value_view.shape[1] - 1), :, 0],
                              dtype=np.float64)
    span = min(len(weight_row), len(value_column))
    if span == 0:
        return None
    weight_row, value_column = weight_row[:span], value_column[:span]
    products = weight_row * value_column
    shown = _heaviest(weight_row, list(range(span)), MIX_KEYS)
    rest = [index for index in range(span) if index not in shown]
    return {'head': head, 'query': query, 'span': span, 'shown': shown, 'rest': rest,
            'weights': weight_row, 'values': value_column, 'products': products,
            'total': float(products.sum())}


def build_mix(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('mix')
    host, body = _panel()
    body.addWidget(ctx.equation(('Z_head', '=', 'attention_weights', '@', 'v_attended')))
    body.addWidget(ctx.io_row(operation))
    body.addWidget(ctx.head_picker())

    def worked() -> QWidget:
        inner, lines = _panel(6)
        example = mix_example(ctx)
        if example is None:
            lines.addWidget(label('This phase did not capture the weights and values.',
                                  muted=True))
            return inner
        head, query = example['head'], example['query']
        shown, rest = example['shown'], example['rest']
        spellings = ctx.key_spellings()
        names = [f'{index}  {spellings[index] if index < len(spellings) else ""}'
                 for index in shown]
        remainder = ctx.say('mix_remainder', count=len(rest)) if rest else ''
        weights = [example['weights'][index] for index in shown]
        products = [example['products'][index] for index in shown]
        rest_weight = float(example['weights'][rest].sum()) if rest else 0.0
        rest_product = float(example['products'][rest].sum()) if rest else 0.0
        # The operands are observed and say so on their boxes; only the arithmetic done
        # here for the display keeps a badge.
        lines.addWidget(badge_row(
            EvidenceBadge(EVIDENCE_DERIVED, 'the products and their total')))
        sink = sink_callout(ctx, example['weights'])
        if sink is not None:
            lines.addWidget(sink)
        # One step at a time, at the reader's pace.  This is the one operation whose
        # result the model discarded, so it is the one worth walking rather than stating.
        lines.addWidget(StepReveal([
            {'caption': ctx.say('mix_step_weights'),
             'body': ctx.say('mix_step_weights_body'),
             'tier': EVIDENCE_OBSERVED, 'note': 'attention_weights',
             'build': lambda: ctx.values(
                 weights + ([rest_weight] if rest else []),
                 labels=names + ([remainder] if rest else []),
                 details=[approx_share(w) for w in weights]
                 + ([approx_share(rest_weight)] if rest else []),
                 caption=ctx.say('mix_weights_caption', count=len(shown), head=head,
                                 query=query),
                 note='attention_weights', total=len(shown) + (1 if rest else 0))},
            {'caption': ctx.say('mix_step_values'),
             'body': ctx.say('mix_step_values_body'),
             'tier': EVIDENCE_OBSERVED, 'note': 'v_attended',
             'build': lambda: ctx.values(
                 [example['values'][index] for index in shown], labels=names,
                 caption=ctx.say('mix_values_caption', head=head), note='v_attended',
                 total=len(shown))},
            {'caption': ctx.say('mix_step_products'),
             'body': ctx.say('mix_step_products_body'),
             'tier': EVIDENCE_DERIVED, 'note': 'calculated for this display',
             'build': lambda: ctx.values(
                 products + ([rest_product] if rest else []),
                 labels=names + ([remainder] if rest else []), kind=EVIDENCE_DERIVED,
                 caption=ctx.say('mix_products_caption'),
                 note='calculated for this display', total=len(shown) + (1 if rest else 0))},
            {'caption': ctx.say('mix_step_total'),
             'body': ctx.say('mix_step_total_body'),
             'tier': EVIDENCE_DERIVED, 'note': 'calculated for this display',
             'build': lambda: ctx.values(
                 [example['total']], labels=['feature 0 of Z_h'], kind=EVIDENCE_DERIVED,
                 caption=ctx.say('mix_total_caption', head=head),
                 note='calculated for this display', total=1)},
        ], state=ctx.state.setdefault('reveals', {}), state_key='mix', initial=1))
        lines.addWidget(label(DISPLAY_CALC_NOTE, muted=True, small=True))
        return inner

    ctx.slot(body, worked)
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables))
    return host


# ── 9  Joining the heads ────────────────────────────────────────────────────

def build_join(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('join')
    host, body = _panel()
    body.addWidget(ctx.equation(('attention_output', '=', 'Z_joined', '@', 'W_O')))
    body.addWidget(ctx.io_row(operation))

    def shapes() -> QWidget:
        inner, lines = _panel(6)
        output = ctx.array('attention_output')
        if output is None:
            lines.addWidget(label('This phase did not capture the block output.',
                                  muted=True))
            return inner
        view = np.asarray(output)
        joined = int(ctx.facts['q_projection_width'])
        lines.addWidget(MatMulShapes(
            (int(view.shape[-2]), joined), (int(view.shape[-1]), joined),
            (int(view.shape[-2]), int(view.shape[-1])),
            left_symbol='Z_joined', right_symbol='W_Oᵀ', out_symbol='O',
            left_role='v', out_role='output', tokens=ctx.tokens,
            caption=ctx.say('join_shape_caption')))
        lines.addWidget(label(ctx.say('join_side_by_side'), muted=True, small=True))
        _tracked(ctx, lines, [('layer_input', 'X'), ('attention_output', 'O')])
        return inner

    ctx.slot(body, shapes)
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables))
    return host


# ── 10  Residual and feed-forward ───────────────────────────────────────────

def build_block(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('block')
    host, body = _panel()
    body.addWidget(ctx.equation(('layer_output', '=', 'H', '+', 'F')))
    body.addWidget(ctx.io_row(operation))
    body.addWidget(ResidualBypass(
        [('attention block', 'O', 'H'), ('feed-forward network', 'F', 'X_next')],
        trunk_symbols=('X', 'H'), tokens=ctx.tokens, caption=ctx.say('residual_caption')))
    body.addWidget(PositionScope(tokens=ctx.tokens, caption=ctx.say('scope_caption')))

    def output() -> QWidget:
        inner, lines = _panel(4)
        if ctx.row('layer_output') is None:
            lines.addWidget(label('This phase did not capture the layer output.',
                                  muted=True))
            return inner
        _tracked(ctx, lines, [('layer_input', 'X'), ('layer_output', 'X_next')])
        lines.addWidget(label(ctx.say('layer_out_caption'), muted=True, small=True))
        return inner

    ctx.slot(body, output)
    ctx.finish(body, operation)
    body.addWidget(ctx.disclosure(
        'Inside the feed-forward network', remember=True,
        build=lambda: GatedFFNPath(int(ctx.facts['hidden_size']), None, tokens=ctx.tokens,
                                   caption=ctx.text('{ffn_interior_note}'))))
    body.addWidget(ctx.variable_list(operation.variables))
    ctx.add_check(body, 'block')
    return host


# ── 11  The remaining layers ────────────────────────────────────────────────

def layer_summary_rows(ctx: NotebookContext) -> list[dict]:
    """One row per layer, for the followed position, from captured arrays only.

    For each layer: where this position's attention weight went, averaged over heads (the
    head-mean of one captured weight row, then its argmax), and the largest-magnitude
    coordinate of that layer's captured output.  Both are rankings of stored values; the
    model reported neither.
    """
    layers = ctx.registry.layers()
    rows = []
    for index in sorted(layers):
        tensors = layers[index].tensors
        weights, output = tensors.get('attention_weights'), tensors.get('layer_output')
        entry = {'layer': index, 'key': None, 'weight': None, 'feature': None, 'value': None,
                 'sink': None, 'query': None}
        if weights is not None and np.asarray(weights).ndim == 4:
            view = np.asarray(weights)
            query = min(ctx.registry.position, view.shape[-2] - 1)
            mean = np.asarray(view[0, :, query, :], dtype=np.float64).mean(axis=0)
            entry['key'] = int(np.argmax(mean))
            entry['weight'] = float(mean[entry['key']])
            # The head-mean row's weight on position 0, for the attention-sink note.
            entry['sink'] = float(mean[0]) if mean.size else None
            entry['query'] = int(query)
        if output is not None and np.asarray(output).ndim == 3:
            view = np.asarray(output)
            row = view[0, min(ctx.registry.position, view.shape[-2] - 1)]
            entry['feature'] = int(np.argmax(np.abs(np.asarray(row, dtype=np.float64))))
            entry['value'] = row[entry['feature']]
        rows.append(entry)
    return rows


def layer_sink_callout(ctx: NotebookContext, summary) -> QWidget | None:
    """Op 11's form of the sink note: how many layers' head-mean rows trip the threshold."""
    tripped = [entry['sink'] for entry in summary
               if entry['sink'] is not None and entry['sink'] >= ATTENTION_SINK_THRESHOLD]
    if not tripped:
        return None
    query = next(entry['query'] for entry in summary if entry['query'] is not None)
    host, lines = _panel(4)
    note = measure_label(ctx.say(
        'attention_sink_layers', count=len(tripped), query=query,
        spelling=repr(readable_spelling(ctx.spelling(0))), threshold=_threshold_text(),
        low=f'{min(tripped) * 100:.1f}%', high=f'{max(tripped) * 100:.1f}%'),
        width=PROSE_WIDTH + 40)
    note.setObjectName('sinkNote')
    lines.addWidget(note)
    lines.addWidget(_sink_rule(ctx))
    host.setObjectName('sinkCallout')
    return host


def _layer_features(ctx: NotebookContext, summary) -> QWidget:
    """The largest coordinate of each layer's output, kept out of the main table.

    In the shipped runs one feature leads by magnitude in nearly every layer, so the column
    mostly showed one coordinate's size growing.  When a single feature leads in most
    layers this says so, with the count read off the same rows.
    """
    inner, lines = _panel(4)
    features = [entry['feature'] for entry in summary if entry['feature'] is not None]
    if features:
        leader = max(set(features), key=lambda feature: (features.count(feature), -feature))
        count = features.count(leader)
        key = ('layer_feature_dominance' if count * 2 > len(features)
               else 'layer_feature_note')
        lines.addWidget(badged(EvidenceBadge(EVIDENCE_DERIVED, 'ranking'),
                               ctx.say(key, feature=leader, count=count)))
    rows, tips = [], []
    for entry in summary:
        rows.append((entry['layer'],
                     '' if entry['feature'] is None else f'feature {entry["feature"]}',
                     '' if entry['value'] is None else approx(entry['value'])))
        tips.append(('', '', '' if entry['value'] is None else
                     f'Exact stored value: {exact_scalar(entry["value"])}'))
    grid = read_only_table(['Layer', 'Largest coordinate', '≈ value'], rows, stretch=1,
                           tooltips=tips)
    lines.addWidget(evidence_frame(grid, EVIDENCE_OBSERVED))
    return inner


def build_layers(ctx: NotebookContext, section) -> QWidget:
    operation = _operation('layers')
    host, body = _panel()
    body.addWidget(ctx.io_row(operation))

    def stack() -> QWidget:
        return LayerStack(len(ctx.capture.layers) or 1, ctx.registry.layer_index,
                          tokens=ctx.tokens, caption=ctx.say('layer_stack_caption'))

    ctx.slot(body, stack)
    body.addWidget(ctx.layer_picker())

    def table() -> QWidget:
        inner, lines = _panel(4)
        spellings = ctx.key_spellings()
        summary = layer_summary_rows(ctx)
        lines.addWidget(badge_row(
            EvidenceBadge(EVIDENCE_DERIVED, 'head-mean and rankings')))
        sink = layer_sink_callout(ctx, summary)
        if sink is not None:
            lines.addWidget(sink)
        lines.addWidget(label(ctx.say('layer_table_caption'), muted=True, small=True))
        rows, tips, current = [], [], None
        for position, entry in enumerate(summary):
            if entry['layer'] == ctx.registry.layer_index:
                current = position
            key = entry['key']
            rows.append((
                entry['layer'],
                '' if key is None else
                f'{key}  {spellings[key] if key < len(spellings) else ""}',
                '' if entry['weight'] is None else approx(entry['weight'])))
            tips.append(('', '',
                         '' if entry['weight'] is None else
                         f'Exact head-mean weight: {entry["weight"]!r}'))
        grid = read_only_table(['Layer', 'Most-attended position', '≈ head-mean weight'],
                               rows, stretch=1, highlight=current, tooltips=tips)
        for column, width in ((0, 56), (2, 160)):
            grid.setColumnWidth(column, width)
        grid.setMinimumHeight(min(260, 32 + 28 * max(1, len(rows))))
        grid.cellClicked.connect(
            lambda row, _column: ctx.registry.set_context(layer_index=summary[row]['layer']))
        grid.setToolTip('Click a row to point the operations above at that layer.')
        lines.addWidget(grid)
        lines.addWidget(label(ctx.say('layer_table_note'), muted=True, small=True))
        lines.addWidget(ctx.disclosure(ctx.say('layer_feature_title'),
                                       lambda rows=summary: _layer_features(ctx, rows)))
        return inner

    ctx.slot(body, table)
    ctx.finish(body, operation)
    first = min(ctx.capture.layers) if ctx.capture.layers else 0
    body.addWidget(ctx.disclosure(ctx.say('layer_shapes_title'),
                                  lambda: _layer_summary(ctx, first), remember=True))
    return host


def _layer_summary(ctx: NotebookContext, index: int) -> QWidget:
    host, body = _panel(4)
    layer = ctx.capture.layers.get(index)
    if layer is None:
        body.addWidget(label('Not in this capture.', muted=True))
        return host
    rows = []
    for name, tensor in layer.tensors.items():
        written, _ = shape_line(tensor, omit_batch=True)
        rows.append((name, written, str(tensor.dtype)))
    body.addWidget(evidence_frame(read_only_table(['Captured tensor', 'Shape', 'Stored dtype'],
                                                  sorted(rows), stretch=0), EVIDENCE_OBSERVED))
    return host


# ── 12  The token decision ──────────────────────────────────────────────────

def build_decide(ctx: NotebookContext, section) -> QWidget:
    capture = ctx.capture
    operation = _operation('decide')
    facts = ctx.facts
    final = int(facts['final_position'])
    host, body = _panel()
    # The formula and the in/out line first, as in every other operation; the hand-off to
    # the final position follows, directly above the picture that shows it.
    body.addWidget(ctx.equation(('final_logits', '=', 'X_final_norm', '@', 'W_U')))
    body.addWidget(ctx.io_row(operation))

    # The hand-off.  Everything above followed one position; the decision is made at the
    # final one, and the page says so rather than letting the followed row look as if it
    # chose.  The selection is not moved automatically: the reader is offered the jump.
    def shift() -> QWidget:
        inner, lines = _panel(6)
        followed = ctx.registry.position
        spelled = repr(readable_spelling(ctx.spelling(followed)))
        if followed == final:
            text = ctx.say('focus_is_final')
        else:
            text = ctx.say('focus_shift', focus_position=followed, focus_token_text=spelled)
        callout = label(text)
        callout.setObjectName('focusShift')
        lines.addWidget(callout)
        if followed != final:
            jump = QPushButton(ctx.say('follow_final'))
            jump.setObjectName('crossPageLink')
            jump.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            jump.clicked.connect(lambda: ctx.registry.set_context(position=final))
            lines.addWidget(jump, 0, Qt.AlignLeft)
        lines.addWidget(VocabularyProjection(
            int(facts['hidden_size']), int(facts.get('vocab_size_raw') or 0),
            positions=int(facts['position_count']), final_position=final,
            marked=int(facts.get('candidate_top_k') or 0), focus_position=followed,
            tokens=ctx.tokens, caption=ctx.say('projection_caption')))
        return inner

    ctx.slot(body, shift)
    # The chain the formula stands for: which links a capture holds (solid) and which
    # happened in the model and were not kept (dashed).
    body.addWidget(DecisionChain(
        [(f'row {final} of X_final', True), ('final RMSNorm', False), ('× W_Uᵀ', False),
         ('logits', capture.logits is not None), ('argmax', True), ('token', True)],
        tokens=ctx.tokens))

    logits = capture.logits
    if logits is None:
        body.addWidget(described_line(LOGITS_UNAVAILABLE))
        body.addWidget(label(
            f'The token this run selected is still known — it is '
            f'{capture.tokens[0]!r}, vocabulary ID {capture.token_ids[0]} — because the '
            f'emitted text was saved. What cannot be shown is the score vector behind it.',
            muted=True, small=True))
        body.addWidget(label(facts['selection_rule_sentence'], muted=True, small=True))
        ctx.finish(body, operation)
        body.addWidget(ctx.variable_list(operation.variables))
        return host

    flat = np.asarray(logits).reshape(-1)
    chosen = int(flat.argmax())
    body.addWidget(badge_row(
        EvidenceBadge(EVIDENCE_DERIVED, 'the ranking, the argmax and the margin'),
        label(ctx.say('score_count', count=f'{flat.size:,}'), muted=True, small=True)))
    body.addWidget(evidence_frame(_candidates(ctx, flat), EVIDENCE_OBSERVED))
    rule = label(ctx.say('selected_line', token_id=chosen, token=repr(capture.tokens[0])))
    rule.setObjectName('journeyEquation')
    body.addWidget(rule)
    order = np.argsort(-flat, kind='stable')[:2]
    closing = facts['selection_rule_sentence']
    if len(order) > 1:
        margin = float(flat[order[0]]) - float(flat[order[1]])
        closing = ctx.say('margin_line', margin=approx(margin).lstrip('≈')) + ' ' + closing
    body.addWidget(label(closing, muted=True, small=True))
    ctx.finish(body, operation)
    body.addWidget(ctx.variable_list(operation.variables))
    _add_decide_check(ctx, body, flat)
    return host


def _score_rows(entries, *, highlight_id=None):
    """Candidate rows, rounded for reading, with the exact score on each cell's tooltip."""
    rows, tips, winner = [], [], None
    for rank, (identifier, token, score) in enumerate(entries):
        if highlight_id is not None and int(identifier) == int(highlight_id):
            winner = rank
        rows.append((rank + 1, int(identifier), repr(str(token)) if token != '' else '',
                     approx(score)))
        tips.append(('', '', '', f'Exact stored score: {exact_scalar(score)}'))
    return rows, tips, winner


def _candidates(ctx: NotebookContext, flat) -> QWidget:
    """The best-scoring entries, preferring the run's own recorded list.

    The recorded list and a ranking computed here can disagree on ties, so where the run
    saved one it is used; the fallback is marked as TensorScope's own ranking.
    """
    trace = ctx.capture.generation_trace
    if trace:
        entries = [(entry['id'], entry.get('token', ''), float(entry['logit']))
                   for entry in trace[0].top_candidates]
    else:
        order = np.argsort(-np.asarray(flat), kind='stable')[:10]
        entries = [(int(index), '', flat[index]) for index in order]
    rows, tips, _ = _score_rows(entries)
    return read_only_table(['Rank', 'Token ID', 'Token', 'Score (logit)'], rows,
                           stretch=2, highlight=0, tooltips=tips)


def _add_decide_check(ctx: NotebookContext, body, flat) -> None:
    """The ratio question, only where this run has two real scores to put in it."""
    order = np.argsort(-np.asarray(flat))[:2]
    if len(order) < 2:
        return
    winner, runner = float(flat[order[0]]), float(flat[order[1]])
    if runner == 0:
        return
    ctx.add_check(body, 'decide', winner=f'{winner:.4g}', runner_up=f'{runner:.4g}',
                  ratio=f'{winner / runner:.2f}')


# ── The continuation, and the closing summary ───────────────────────────────

def timeline_rows(ctx: NotebookContext) -> list[dict]:
    """One row per forward pass, saying what that pass kept.  Read from the run.

    Pass 1 is the prompt prefill, which chose token 1.  Pass 2 processed token 1 and chose
    token 2 -- and it is the one later pass whose layer tensors were captured.  Every pass
    after that kept, at most, its leading candidate scores.
    """
    capture = ctx.capture
    trace = capture.generation_trace
    rows = []
    for step, token in enumerate(capture.tokens):
        if step == 0:
            title = ctx.say('timeline_first', token=repr(token))
        else:
            title = ctx.say('timeline_next', previous=step, number=step + 1, token=repr(token))
        layers = ctx.say('timeline_layers_saved' if step <= 1 else 'timeline_layers_not_saved')
        if step == 0 and capture.logits is not None:
            vector, tier = ctx.say('timeline_full_vector'), EVIDENCE_OBSERVED
        elif step < len(trace):
            vector = ctx.say('timeline_leading')
            tier = (EVIDENCE_OBSERVED if (step == 0 or trace[step].attested)
                    else EVIDENCE_UNATTESTED)
        else:
            vector, tier = ctx.say('timeline_not_recorded'), EVIDENCE_CONCEPTUAL
        rows.append({'title': title, 'capabilities': f'{layers}  ·  {vector}', 'tier': tier,
                     'step': step, 'layers': step <= 1, 'vector': vector})
    if ctx.facts['stop_token_known']:
        rows.append({'title': ctx.say('timeline_stop', previous=len(capture.tokens)),
                     'capabilities': f'{ctx.say("timeline_layers_not_saved")}  ·  '
                                     f'stopping score saved',
                     'tier': EVIDENCE_UNATTESTED, 'step': None, 'layers': False,
                     'vector': 'stopping score saved'})
    return rows


def build_continuation(ctx: NotebookContext, section) -> QWidget:
    capture = ctx.capture
    host, body = _panel()

    body.addWidget(label(ctx.say('answer_heading'), muted=True))
    view = ResponseTextView(capture.tokens, capture.response, selected=0,
                            ink=ctx.tokens.get('text_primary'))
    # linkActivated is mouse-only, so the anchors alone would put every token after the
    # first out of reach from the keyboard.
    view.setFocusPolicy(Qt.StrongFocus)
    body.addWidget(view)
    body.addWidget(_token_stepper(ctx, view))

    body.addWidget(label(ctx.say('timeline_heading'), muted=True))
    rows = timeline_rows(ctx)
    timeline = PassTimeline(rows, selected=0,
                            show_all_text=ctx.say('timeline_show_all', count=len(rows)),
                            show_fewer_text=ctx.say('timeline_show_fewer'))
    body.addWidget(timeline)

    slot = ctx.plain_slot(body)

    def show(step: int) -> None:
        timeline.select(step)
        replace_in(slot, _decision(ctx, step))

    def from_timeline(index: int) -> None:
        if index < len(capture.tokens):
            view.select(index)

    view.selected.connect(show)
    timeline.selected.connect(from_timeline)
    show(view.current())

    body.addWidget(label(ctx.say('kv_reuse'), muted=True, small=True))
    toggle = ctx.phase_toggle()
    toggle.setToolTip(ctx.say('phase_toggle_note'))
    body.addWidget(toggle, 0, Qt.AlignLeft)
    body.addWidget(_stopping(ctx))
    body.addWidget(_summary(ctx))
    body.addWidget(ctx.disclosure('What this capture does and does not establish',
                                  lambda: _limits(ctx), remember=True))
    return host


def _token_stepper(ctx: NotebookContext, view: ResponseTextView) -> QWidget:
    """Keyboard access to every emitted token, since the anchors need a mouse."""
    host = QWidget()
    row = QHBoxLayout(host)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    previous = QPushButton('← Previous token')
    following = QPushButton('Next token →')
    for button in (previous, following):
        button.setObjectName('revealControl')
        row.addWidget(button)
    caption = label('', muted=True, small=True)
    row.addWidget(caption)
    row.addStretch(1)

    def update() -> None:
        caption.setText(f'token {view.current() + 1} of {len(ctx.capture.tokens)}')
        previous.setEnabled(view.current() > 0)
        following.setEnabled(view.current() < len(ctx.capture.tokens) - 1)

    previous.clicked.connect(lambda: view.select(view.current() - 1))
    following.clicked.connect(lambda: view.select(view.current() + 1))
    view.selected.connect(lambda _: update())
    update()
    return host


def _decision(ctx: NotebookContext, step: int) -> QWidget:
    capture = ctx.capture
    host, body = _panel(4)
    if not (0 <= step < len(capture.tokens)):
        body.addWidget(label('No token selected.', muted=True))
        return host
    body.addWidget(label(
        f'Token {step + 1}   ·   selected {capture.tokens[step]!r}   ·   '
        f'vocabulary ID {capture.token_ids[step]}'))
    trace = capture.generation_trace
    if step >= len(trace):
        body.addWidget(described_line(TELEMETRY_UNAVAILABLE))
        return host
    decision = trace[step]
    tier = EVIDENCE_OBSERVED if (step == 0 or decision.attested) else EVIDENCE_UNATTESTED
    # Unattested keeps its words -- it is the tier a reader must not mistake for observed
    # -- and observed scores say so with the edge on their table below.
    marks = [mark for mark in (evidence_mark(tier, 'scores from the forward pass'),
                               EvidenceBadge(EVIDENCE_DERIVED, 'the ranking')) if mark]
    body.addWidget(badge_row(*marks))
    if tier is EVIDENCE_UNATTESTED:
        body.addWidget(label(
            'Real output of the real forward pass, but only the prompt pass was re-run '
            'under stock attention, so no verification gate independently re-derived '
            'this one.', muted=True, small=True))
    rows, tips, winner = _score_rows(
        [(entry['id'], entry.get('token', ''), float(entry['logit']))
         for entry in decision.top_candidates], highlight_id=decision.selected_token_id)
    body.addWidget(evidence_frame(
        read_only_table(['Rank', 'Token ID', 'Token', 'Score (logit)'], rows,
                        stretch=2, highlight=winner, tooltips=tips), tier))
    margin = decision.margin
    if margin is not None:
        body.addWidget(label(ctx.say('margin_line', margin=approx(margin).lstrip('≈')),
                             muted=True, small=True))
    body.addWidget(label(ctx.facts['selection_rule_sentence'], muted=True, small=True))
    return host


def _stopping(ctx: NotebookContext) -> QWidget:
    """The stopping decision, shown as its own step when the run recorded one."""
    facts = ctx.facts
    host, body = _panel(4)
    body.addWidget(label('WHY IT STOPPED', muted=True))
    body.addWidget(label(f"{facts['stop_reason_text']} {EOS_IS_A_SCORED_ENTRY}"))
    if not facts['stop_token_known']:
        body.addWidget(described_line(
            'This run does not record which entry won the final scoring step, so '
            'TensorScope does not say.'))
        return host
    # Unattested, not observed: this score came from a generation pass, and only the
    # prompt prefill is covered by the stock-eager comparison.
    body.addWidget(badge_row(EvidenceBadge(EVIDENCE_UNATTESTED, 'the stopping score')))
    body.addWidget(read_only_table(
        ['The final scoring step', 'Value'],
        [('Winning vocabulary ID', facts['stop_token_id_text']),
         ('Its spelling, where the prompt also contains that ID',
          facts['stop_token_spelling']),
         ('Its captured score', facts['stop_token_logit_text']),
         ('Emitted into the answer?', 'No')], stretch=1))
    body.addWidget(label(
        ctx.text('Its winner was the stopping marker, so it was not appended: the answer '
                 'is {generated_count} tokens, not {generated_count} plus one.'),
        muted=True, small=True))
    return host


def _summary(ctx: NotebookContext) -> QWidget:
    facts = ctx.facts
    host, body = _panel(4)
    body.addWidget(label('THIS RUN, IN FULL', muted=True))
    body.addWidget(evidence_frame(read_only_table(['Quantity', 'This run'], [
        ('Prompt token positions', facts['position_count']),
        ('Emitted response tokens', facts['generated_count']),
        ('Tokens after the first', facts['continuation_count']),
        ('Layers', facts['layer_count']),
        ('Query heads', facts['query_head_count']),
        ('Key/value heads', facts['kv_head_count']),
        ('Features per head', facts['head_dim']),
        ('Captured phases', facts['phase_count']),
    ], stretch=1), EVIDENCE_OBSERVED))
    return host


def _limits(ctx: NotebookContext) -> QWidget:
    host, body = _panel(6)
    body.addWidget(label('WHAT THIS CAPTURE ESTABLISHES', muted=True))
    body.addWidget(bullets(what_we_know(ctx.facts)))
    body.addWidget(label('WHAT IT DOES NOT', muted=True))
    body.addWidget(bullets(WHAT_WE_CANNOT_CONCLUDE))
    body.addWidget(label(ctx.say('limits_marks_heading'), muted=True))
    body.addWidget(EvidenceLegend())
    for kind, (name, description) in EVIDENCE_KINDS.items():
        body.addWidget(label(f'{name}: {description}', muted=True, small=True))
    body.addWidget(label(CACHED_KV_NOTE, muted=True, small=True))
    return host


BUILDERS = {
    'input': build_input,
    'embed': build_embed,
    'shape': build_shape,
    'norm': build_norm,
    'project': build_project,
    'prepare': build_prepare,
    'scores': build_scores,
    'mix': build_mix,
    'join': build_join,
    'block': build_block,
    'layers': build_layers,
    'decide': build_decide,
    'continuation': build_continuation,
}
