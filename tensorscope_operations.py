"""The body of each operation in the worked notebook.

One builder per operation, each a pure function of the reader's current selection, so a
selection change refreshes one slot inside one section rather than rebuilding a page.
`NotebookContext` carries everything a builder needs and nothing it does not: the capture,
this run's facts, the shared variable registry, the reader's state dicts, and small
factories for the four things every operation is made of -- an equation, a variable list,
a row of real values, and an optional check.

What a builder may and may not do:

* It reads captured arrays through the registry, which hands back the stored object.
  Slicing is basic indexing only, so nothing here can write through to a capture.
* Where the architecture has an intermediate this capture did not keep, the builder names
  the variable and lets the panel explain it.  There is no path from a builder to a
  fabricated tensor, because `Resolved.array` is simply `None` for those.
* One operation does arithmetic on captured values: operation 8 works a single scalar of
  A @ V, because that is the step whose result was discarded and a reader who has never
  seen the multiplication happen has not understood it.  It is one number, in double
  precision, badged derived, beside the note that says the model computed in bfloat16.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from tensor_widgets import AttentionExplorer, exact_scalar
from tensorscope_common import (
    AnnotatedValueRow, BandedTokenStrip, Disclosure, EvidenceBadge, PredictCheck,
    ResponseTextView, StepReveal, badge_row, bullets, chip_text, embedding_row, head_slice,
    label, read_only_table, replace_in, shape_line, token_labels, token_table,
)
from tensorscope_content import (
    BATCH_AXIS_NOTE, CACHED_KV_NOTE, DISPLAY_CALC_NOTE, EOS_IS_A_SCORED_ENTRY,
    EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED, EVIDENCE_KINDS, EVIDENCE_OBSERVED,
    EVIDENCE_UNATTESTED, LOGITS_UNAVAILABLE, NOTEBOOK_CHECKS, PROMPT_SPAN_UNKNOWN,
    SHAPE_IS_NOT_CONTENTS, TELEMETRY_UNAVAILABLE, TEMPLATE_BAND, WHAT_WE_CANNOT_CONCLUDE,
    WHAT_WE_KNOW, prompt_token_bands, readable_spelling,
)
from tensorscope_diagrams import (
    AttentionGrid, CausalMaskTriangle, EmbeddingLookupRow, GatedFFNPath, HeadSplit, KVShare,
    LayerStack, MatMulShapes, MatrixGrid, QKCompare, ResidualBypass, SoftmaxBars,
    VectorStrip, VocabularyProjection,
)
from tensorscope_variables import EquationRow, VariableChip


SAMPLE_WIDTH = 8


class NotebookContext:
    """Everything a builder needs, and the small factories they all share."""

    def __init__(self, capture, facts, registry, *, tokens=None, state=None):
        self.capture = capture
        self.facts = facts
        self.registry = registry
        self.tokens = tokens or {}
        self.state = state if state is not None else {}
        self.bands = prompt_token_bands(capture.prompt, capture.prompt_tokens)

    # -- copy -----------------------------------------------------------------

    def text(self, template: str, **extra) -> str:
        return template.format(**{**self.facts, **extra})

    # -- the four things an operation is made of ------------------------------

    def equation(self, terms) -> QWidget:
        return EquationRow(self.registry, terms)

    def chip(self, key: str) -> VariableChip:
        return VariableChip(self.registry, key)

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

    def values(self, array, *, labels=None, caption='', kind=EVIDENCE_OBSERVED, note='',
               total=None, formatter=None) -> AnnotatedValueRow:
        shown = list(np.asarray(array).reshape(-1)[:SAMPLE_WIDTH])
        return AnnotatedValueRow(
            shown, labels=labels, caption=caption.upper(), kind=kind, note=note,
            total=total if total is not None else int(np.asarray(array).size),
            formatter=formatter or exact_scalar)

    def disclosure(self, title: str, build, *, remember: bool = False) -> Disclosure:
        """A collapsed body.  `remember` only for cheap bodies -- never a tensor widget."""
        return Disclosure(title, build,
                          state=self.state.setdefault('open', {}) if remember else None,
                          state_key=title)

    def mpl_style(self) -> dict:
        return {'figure.facecolor': self.tokens.get('card_bg', '#1c1c20'),
                'axes.facecolor': self.tokens.get('bg', '#111114'),
                'text.color': self.tokens.get('text_secondary', '#9f9fa8')}

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

    def spelling(self, index: int) -> str:
        tokens = self.capture.prompt_tokens
        return tokens[index] if 0 <= index < len(tokens) else ''

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


def _feature_labels(count: int, prefix: str = 'feature') -> list[str]:
    return [f'{prefix} {index}' for index in range(count)]


# ── 1  The model's input ─────────────────────────────────────────────────────

def build_input(ctx: NotebookContext, section) -> QWidget:
    capture = ctx.capture
    host, body = _panel()

    strip = BandedTokenStrip(capture.prompt_tokens, ctx.bands,
                             selected=ctx.registry.position)
    strip.selected.connect(lambda index: ctx.registry.set_context(position=index))
    body.addWidget(strip)
    if ctx.bands is None:
        body.addWidget(label(PROMPT_SPAN_UNKNOWN, muted=True, small=True))
    body.addWidget(label(
        'Positions whose stored spelling is written <|like this|> are control markers by '
        'convention of the spelling — that is what the chat template writes, and it is '
        'all this capture records about them. TensorScope does not claim which vocabulary '
        'entries the tokenizer flags as special, because a saved run does not contain the '
        'tokenizer.', muted=True, small=True))

    def selected() -> QWidget:
        index = ctx.registry.position
        inner, lines = _panel(4)
        if not (0 <= index < len(capture.prompt_tokens)):
            lines.addWidget(label('No position selected.', muted=True))
            return inner
        stored = capture.prompt_tokens[index]
        band = next((name for start, end, name in (ctx.bands or [])
                     if start <= index < end), '')
        lines.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'saved tokenization')))
        lines.addWidget(label(
            f'Position {index}   ·   token ID {capture.prompt_token_ids[index]}   ·   '
            f'stored spelling {stored!r}'))
        lines.addWidget(label(f'Reads as {readable_spelling(stored)!r}'
                              + (f'   ·   {band}' if band else ''), muted=True))
        if band == TEMPLATE_BAND:
            lines.addWidget(label(
                'You did not type this one. It is here because the chat template wraps '
                'your text in markers that say whose turn it is.', muted=True, small=True))
        return inner

    ctx.slot(body, selected)
    body.addWidget(ctx.disclosure('Show every position as a table',
                                  lambda: token_table(capture), remember=True))
    return host


# ── 2  Embedding lookup ─────────────────────────────────────────────────────

def build_embed(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()
    body.addWidget(ctx.equation(('E', '[', 'token ID', ']', '→', 'embedding')))
    body.addWidget(ctx.variables(('embedding', 'E')))

    def row() -> QWidget:
        index = ctx.registry.position
        values = ctx.row('embedding', position=index)
        inner, lines = _panel(6)
        identifier = (ctx.capture.prompt_token_ids[index]
                      if 0 <= index < len(ctx.capture.prompt_token_ids) else None)
        lines.addWidget(EmbeddingLookupRow(
            int(ctx.facts.get('vocab_size_raw') or 0), int(ctx.facts['hidden_size']),
            token_id=identifier, token_text=ctx.spelling(index), tokens=ctx.tokens,
            caption='The ID picks a row. The row is the vector. Its width is the '
                    'model’s feature count, not the vocabulary size.'))
        if values is None:
            lines.addWidget(label('This capture’s embedding is not shaped '
                                  '[batch, token, feature], so a single row cannot be '
                                  'picked out of it.', muted=True))
            return inner
        lines.addWidget(ctx.values(
            values, labels=_feature_labels(SAMPLE_WIDTH),
            caption=f'the first {SAMPLE_WIDTH} of {len(values):,} coordinates for '
                    f'position {index} ({ctx.spelling(index)!r})',
            note='embedding', total=len(values)))
        return inner

    ctx.slot(body, row)
    check = ctx.check('embed')
    if check is not None:
        body.addWidget(check)
    return host


# ── 3  The input tensor ─────────────────────────────────────────────────────

def build_shape(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()

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
        lines.addWidget(label(SHAPE_IS_NOT_CONTENTS, muted=True, small=True))
        lines.addWidget(label(BATCH_AXIS_NOTE, muted=True, small=True))
        view = np.asarray(array)
        lines.addWidget(MatrixGrid(
            int(view.shape[-2]), int(view.shape[-1]), row_axis='token position',
            col_axis='model feature', role='layer', symbol='X',
            highlight_row=ctx.registry.position, tokens=ctx.tokens,
            caption='Positions down, features across. The highlighted row is the position '
                    'you have selected, and it stays highlighted through every tensor '
                    'below that has a position axis.'))
        if caption:
            lines.addWidget(label(caption, muted=True, small=True))
        return inner

    ctx.slot(body, shape_block)
    body.addWidget(ctx.variables(('layer_input',)))
    check = ctx.check('shape')
    if check is not None:
        body.addWidget(check)
    return host


# ── 4  Input normalization ──────────────────────────────────────────────────

def build_norm(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()
    body.addWidget(ctx.equation(('normalized_input', '=', 'RMSNorm', '(', 'layer_input',
                                 ')')))
    body.addWidget(ctx.variables(('layer_input', 'g', 'epsilon', 'rms',
                                  'normalized_input')))

    def comparison() -> QWidget:
        inner, lines = _panel(6)
        before, after = ctx.array('layer_input'), ctx.array('normalized_input')
        if before is None or after is None:
            lines.addWidget(label('This phase did not capture both sides of the '
                                  'normalization.', muted=True))
            return inner
        first, second = np.asarray(before), np.asarray(after)
        lines.addWidget(_side_by_side(
            MatrixGrid(int(first.shape[-2]), int(first.shape[-1]),
                       row_axis='token position', col_axis='model feature', role='layer',
                       symbol='X', highlight_row=ctx.registry.position, tokens=ctx.tokens),
            MatrixGrid(int(second.shape[-2]), int(second.shape[-1]),
                       row_axis='token position', col_axis='model feature', role='norm',
                       symbol='X_norm', highlight_row=ctx.registry.position,
                       tokens=ctx.tokens)))
        lines.addWidget(label(
            'Same arrangement, same row for the same position. Only the values moved, and '
            'none of them crossed between rows.', muted=True, small=True))
        index = ctx.registry.position
        for key, title in (('layer_input', 'X'), ('normalized_input', 'X_norm')):
            values = ctx.row(key, position=index)
            if values is not None:
                lines.addWidget(ctx.values(
                    values, labels=_feature_labels(SAMPLE_WIDTH),
                    caption=f'{title}, position {index}', note=key, total=len(values)))
        return inner

    ctx.slot(body, comparison)
    return host


# ── 5  Raw projections ──────────────────────────────────────────────────────

def build_project(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()
    body.addWidget(ctx.equation(('q', '=', 'normalized_input', '@', 'W_Q')))
    body.addWidget(label(
        ctx.text('The key and value projections work the same way, with their own learned '
                 'matrices: K_raw = X_norm W_Kᵀ and V_raw = X_norm W_Vᵀ. They are '
                 'narrower — {kv_projection_width} features rather than '
                 '{q_projection_width} — because several query heads share each '
                 'key/value head.'),
        muted=True, small=True))

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
            caption='The highlighted edge is the dimension both operands share. It is the '
                    'only reason this multiplication is defined, and the weight matrix is '
                    'written transposed because that is how a linear layer stores it.'))
        index = ctx.registry.position
        for key, title in (('q', 'Q_raw'), ('k', 'K_raw'), ('v', 'V_raw')):
            values = ctx.row(key, position=index)
            if values is not None:
                lines.addWidget(ctx.values(
                    values, labels=_feature_labels(SAMPLE_WIDTH),
                    caption=f'{title}, position {index}', note=key, total=len(values)))
        return inner

    ctx.slot(body, shapes)
    body.addWidget(ctx.variables(('normalized_input', 'W_Q', 'q', 'W_K', 'k', 'W_V', 'v')))
    return host


# ── 6  Preparing for attention ──────────────────────────────────────────────

def build_prepare(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()
    facts = ctx.facts

    def split() -> QWidget:
        # In a slot, so the selected token's row stays highlighted here too: the whole
        # point of this picture is that the positions survive both stages, which is only
        # visible if the reader can see *their* position surviving them.
        return HeadSplit(
            int(facts['hidden_size']), int(facts['q_projection_width']),
            int(facts['query_head_count']), int(facts['head_dim']),
            int(facts['position_count']), highlight_head=ctx.registry.head,
            highlight_row=ctx.row_index('q_attended'), tokens=ctx.tokens,
            caption='Two stages, and they are not the same kind of step. The projection '
                    'computed new numbers; the reshape only regrouped numbers that already '
                    'existed. The highlighted row is your selected position, surviving '
                    'both.')

    ctx.slot(body, split)
    if int(facts['kv_head_count']):
        body.addWidget(KVShare(
            int(facts['kv_head_count']), int(facts['query_head_count']), tokens=ctx.tokens,
            caption=ctx.text('Each of the {kv_head_count} key/value heads is copied out to '
                             'the {kv_group_size} query heads that share it, so the '
                             'captured K and V already have {query_head_count} heads.')))
    body.addWidget(label(
        'The captured Q and K are also after the rotary position transformation, and after '
        'any per-head normalization this architecture applies. A saved run holds no model '
        'configuration, so TensorScope reports the finished tensors rather than claiming '
        'which of those steps ran — the reference page says what this family of models '
        'does.', muted=True, small=True))
    body.addWidget(ctx.variables(('q', 'q_attended', 'k', 'k_attended', 'v',
                                  'v_attended')))

    def prepared() -> QWidget:
        inner, lines = _panel(4)
        head = ctx.registry.head
        for key, title in (('q_attended', 'Q'), ('k_attended', 'K'), ('v_attended', 'V')):
            values = ctx.row(key)
            if values is not None:
                # Q is indexed by query position and K/V by key position, and in a
                # generated pass those axes are different lengths, so the row actually
                # read is reported rather than the selection that asked for it.
                row = ctx.row_index(key)
                lines.addWidget(ctx.values(
                    values, labels=_feature_labels(SAMPLE_WIDTH),
                    caption=f'{title}, head {head}, position {row}',
                    note=key, total=len(values)))
        if not lines.count():
            lines.addWidget(label('This phase did not capture the prepared tensors.',
                                  muted=True))
        return inner

    ctx.slot(body, prepared)
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


def build_scores(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()
    body.addWidget(ctx.equation(('attention_scores', '→ softmax →',
                                 'attention_weights')))
    body.addWidget(ctx.variables(('q_attended', 'k_attended', 'QK_raw', 'M',
                                  'attention_scores', 'attention_weights')))
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

        key_choice = min(max(0, ctx.state.setdefault('key_position', 0)), keys - 1)
        mask = CausalMaskTriangle(
            queries, keys, selected_query=query, selected_key=key_choice, cached=cached,
            clickable=True, tokens=ctx.tokens,
            caption='Click a cell to select its query row and its key column, and the two '
                    'vectors that were compared appear below it. Blocked cells had a large '
                    'negative number added to their score before softmax; the value is '
                    'whatever the dtype allows, not necessarily negative infinity.')

        def pick(row: int, column: int) -> None:
            # The key choice is this operation's own, so it is remembered in the notebook's
            # state rather than in the registry: no other operation has a key axis.
            ctx.state['key_position'] = int(column)
            ctx.registry.set_context(position=row)
            if region.get('holder') is not None and queries == 1:
                # One query means the registry's position did not change, so nothing was
                # emitted and this slot has to refresh itself.
                replace_in(region['holder'], panel())

        mask.cellClicked.connect(pick)
        lines.addWidget(mask)

        # The two operands of the selected cell, with the shared head width bracketed
        # across both -- which is the thing a written dot product hides.
        labels_for = token_labels(ctx.capture, ctx.registry.generated, keys=True)
        lines.addWidget(QKCompare(
            int(ctx.facts['head_dim']), query_pos=query, key_pos=key_choice,
            query_label=ctx.spelling(ctx.row_index('q_attended')),
            key_label=(labels_for[key_choice].split(': ', 1)[-1]
                       if key_choice < len(labels_for) else ''),
            tokens=ctx.tokens,
            caption='The query row and the key row of the cell you selected. Their dot '
                    'product is the one number that cell holds.'))
        if cached:
            lines.addWidget(label(ctx.text(
                'This phase attends to {cached_key_count} key positions but has only one '
                'query: the keys for the prompt were computed in the previous pass and '
                'reused from the cache, so they are inside this captured tensor. That is '
                'why this matrix is one row wide and not square.'), muted=True, small=True))

        score_row = score_view[0, head, query]
        weight_row = weight_view[0, head, query]
        masked = _masked_columns(score_row)
        labels = token_labels(ctx.capture, ctx.registry.generated, keys=True)
        lines.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'attention_scores'),
                                  EvidenceBadge(EVIDENCE_OBSERVED, 'attention_weights')))
        lines.addWidget(SoftmaxBars(
            score_row, weight_row,
            labels=[name.split(':', 1)[0] for name in labels[:len(score_row)]],
            masked_indices=masked, selected=key_choice, tokens=ctx.tokens,
            caption=f'Head {head}, query position {query}. The two rows are aligned by key '
                    f'position: a score above, the weight softmax gave it below. Hatched '
                    f'columns are the masked ones.'))
        total = float(np.asarray(weight_row, dtype=np.float64).sum())
        lines.addWidget(ctx.values(
            [total], labels=['sum of this weight row'], kind=EVIDENCE_DERIVED,
            caption='added up here for the display', note='sum of captured weights',
            total=1))
        lines.addWidget(label(
            ctx.text('Each query’s weight row is normalized across the key positions, '
                     'so it adds up to 1. The scale {attention_scale_text} and the mask '
                     'are already inside the captured scores — they are exactly what was '
                     'handed to softmax.'), muted=True, small=True))

        head_matrix = head_slice(weight_view, head)
        lines.addWidget(ctx.disclosure(
            f'Show every query and key for head {head}',
            lambda matrix=head_matrix, h=head: AttentionGrid(
                matrix, selected_query=query, tokens=ctx.tokens,
                caption=f'Head {h}. Every captured weight for this head, painted rather '
                        f'than plotted.')))
        lines.addWidget(ctx.disclosure(
            'Open the full attention explorer for this layer',
            lambda: _explorer(ctx)))
        return inner

    region['holder'] = ctx.slot(body, panel)
    check = ctx.check('scores')
    if check is not None:
        body.addWidget(check)
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

def build_mix(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()
    body.addWidget(ctx.equation(('Z_head', '=', 'attention_weights', '@', 'v_attended')))
    body.addWidget(ctx.variables(('attention_weights', 'v_attended', 'Z_head')))

    def worked() -> QWidget:
        inner, lines = _panel(6)
        weights, values = ctx.array('attention_weights'), ctx.array('v_attended')
        if weights is None or values is None:
            lines.addWidget(label('This phase did not capture the weights and values.',
                                  muted=True))
            return inner
        weight_view, value_view = np.asarray(weights), np.asarray(values)
        head = min(ctx.registry.head, int(weight_view.shape[1]) - 1)
        query = min(ctx.registry.position, int(weight_view.shape[-2]) - 1)
        weight_row = weight_view[0, head, query]
        value_column = value_view[0, min(head, value_view.shape[1] - 1), :, 0]
        span = min(len(weight_row), len(value_column))
        if span == 0:
            lines.addWidget(label('Nothing to multiply in this phase.', muted=True))
            return inner
        # One scalar of Z_h, from two captured operands, in double precision.  The model
        # computed in bfloat16 and did not keep its answer, so this is shown as a display
        # calculation beside the note saying what that means -- not as the model's value.
        product = (np.asarray(weight_row[:span], dtype=np.float64)
                   * np.asarray(value_column[:span], dtype=np.float64))
        lines.addWidget(badge_row(
            EvidenceBadge(EVIDENCE_OBSERVED, 'both operands'),
            EvidenceBadge(EVIDENCE_DERIVED, 'the products and their total')))
        keys = [f'key {index}' for index in range(SAMPLE_WIDTH)]
        # One step at a time, at the reader's pace. This is the only operation whose
        # result the model discarded, so it is the one worth walking rather than stating:
        # a reader who has never watched a weighted average happen has not seen attention.
        lines.addWidget(StepReveal([
            {'caption': 'The weights this head gave',
             'body': 'One weight per key position, from the previous operation. They add '
                     'up to 1, so what follows is an average rather than a sum.',
             'tier': EVIDENCE_OBSERVED, 'note': 'attention_weights',
             'build': lambda: ctx.values(
                 weight_row, labels=keys,
                 caption=f'captured weights · head {head}, query {query}',
                 note='attention_weights', total=len(weight_row))},
            {'caption': 'The values it was choosing between',
             'body': 'Each attended position contributes a whole vector. To keep this '
                     'readable, only the first feature of each is shown — the same '
                     'arithmetic runs independently for every one of the '
                     f'{ctx.facts["head_dim"]} features.',
             'tier': EVIDENCE_OBSERVED, 'note': 'v_attended',
             'build': lambda: ctx.values(
                 value_column, labels=keys,
                 caption=f'captured values · head {head}, feature 0 of each attended '
                         f'position', note='v_attended', total=len(value_column))},
            {'caption': 'Multiply each weight by its value',
             'body': 'A position with a large weight contributes most of its value; one '
                     'with a weight near zero contributes almost nothing, which is what '
                     'the masking achieved.',
             'tier': EVIDENCE_DERIVED, 'note': 'calculated for this display',
             'build': lambda: ctx.values(
                 product, labels=keys, kind=EVIDENCE_DERIVED,
                 caption='each weight times its value',
                 note='calculated for this display', total=span)},
            {'caption': f'Add all {span} products',
             'body': 'The total is one number: feature 0 of this head’s output for '
                     'this query position. Repeat for every feature and every head and you '
                     'have what the next operation projects.',
             'tier': EVIDENCE_DERIVED, 'note': 'calculated for this display',
             'build': lambda: ctx.values(
                 [float(product.sum())], labels=['feature 0 of Z_h'],
                 kind=EVIDENCE_DERIVED,
                 caption=f'all {span} products added up', note='calculated for this '
                 'display', total=1)},
        ], state=ctx.state.setdefault('reveals', {}), state_key='mix'))
        lines.addWidget(label(DISPLAY_CALC_NOTE, muted=True, small=True))
        lines.addWidget(label(
            'The model computed the whole of Z_h this way and then discarded it. The next '
            'operation shows the tensor that did survive, which is the attention block’s '
            'output after the heads were joined and projected — a different thing from '
            'this.', muted=True, small=True))
        return inner

    ctx.slot(body, worked)
    return host


# ── 9  Joining the heads ────────────────────────────────────────────────────

def build_join(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()
    body.addWidget(ctx.equation(('attention_output', '=', 'Z_joined', '@', 'W_O')))
    body.addWidget(ctx.variables(('Z_joined', 'W_O', 'attention_output')))

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
            caption='The joined-head tensor was not kept, so its shape here is derived '
                    'from the captured widths on either side of it: the output projection '
                    'reads that many features and returns the model’s own width.'))
        index = ctx.registry.position
        values = ctx.row('attention_output', position=index)
        if values is not None:
            lines.addWidget(ctx.values(
                values, labels=_feature_labels(SAMPLE_WIDTH),
                caption=f'O, position {index} — after the output projection',
                note='attention_output', total=len(values)))
        lines.addWidget(label(
            'This captured tensor is the block’s output. It is not A @ V, and it is not '
            'the joined heads: both of those happened before the projection and neither '
            'was retained.', muted=True, small=True))
        return inner

    ctx.slot(body, shapes)
    return host


# ── 10  Residual and feed-forward ───────────────────────────────────────────

def build_block(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()
    body.addWidget(ResidualBypass(
        [('attention block', 'O', 'H'), ('feed-forward network', 'F', 'X_next')],
        trunk_symbols=('X', 'H'), tokens=ctx.tokens,
        caption='Each branch proposes a change and the input travels around it. Neither '
                'sum was captured; the layer output at the end was.'))
    body.addWidget(GatedFFNPath(
        int(ctx.facts['hidden_size']), None, tokens=ctx.tokens,
        caption=ctx.text('{ffn_interior_note}')))
    body.addWidget(label(
        'The feed-forward network processes each position on its own. Attention was the '
        'only step that moved information between positions, and it is finished for this '
        'layer.', muted=True, small=True))
    body.addWidget(ctx.variables(('layer_input', 'attention_output', 'H', 'H_norm',
                                  'W_gate', 'W_up', 'W_down', 'F', 'layer_output')))

    def output() -> QWidget:
        inner, lines = _panel(4)
        index = ctx.registry.position
        values = ctx.row('layer_output', position=index)
        if values is None:
            lines.addWidget(label('This phase did not capture the layer output.',
                                  muted=True))
            return inner
        lines.addWidget(ctx.values(
            values, labels=_feature_labels(SAMPLE_WIDTH),
            caption=f'X_next, position {index} — this layer’s output, and the next '
                    f'layer’s X', note='layer_output', total=len(values)))
        return inner

    ctx.slot(body, output)
    check = ctx.check('block')
    if check is not None:
        body.addWidget(check)
    return host


# ── 11  The remaining layers ────────────────────────────────────────────────

def build_layers(ctx: NotebookContext, section) -> QWidget:
    host, body = _panel()

    def stack() -> QWidget:
        return LayerStack(
            len(ctx.capture.layers) or 1, ctx.registry.layer_index, tokens=ctx.tokens,
            caption='Layers run one after another, and none of their learned parameters '
                    'are shared. The highlighted one is the layer the operations above are '
                    'describing.')

    ctx.slot(body, stack)
    body.addWidget(label(
        ctx.text('Operations 4 to 10 happened once per layer, {layer_count} times in all, '
                 'each with its own learned parameters. The {position_count} token '
                 'positions stayed separate rows throughout. Every layer was captured in '
                 'full — open one to point the operations above at it.'),
        muted=True, small=True))
    body.addWidget(ctx.disclosure(
        ctx.text('Show all {layer_count} layers'),
        lambda: _layer_list(ctx), remember=True))
    return host


def _layer_list(ctx: NotebookContext) -> QWidget:
    """One collapsed row per layer.  Opening a row builds that layer and nothing else.

    This is where a 36-layer model would otherwise build seventy-odd heavy widgets up
    front, so each row's body is its own `Disclosure` and the rows themselves are text.
    """
    host, body = _panel(2)
    for index in sorted(ctx.capture.layers):
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(8)
        jump = QPushButton(f'Point the operations above at layer {index}')
        jump.setObjectName('crossPageLink')
        jump.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        jump.clicked.connect(lambda _, i=index: ctx.registry.set_context(layer_index=i))
        line.addWidget(ctx.disclosure(f'Layer {index}',
                                      lambda i=index: _layer_summary(ctx, i)), 1)
        line.addWidget(jump, 0)
        body.addWidget(row)
    return host


def _layer_summary(ctx: NotebookContext, index: int) -> QWidget:
    host, body = _panel(4)
    layer = ctx.capture.layers.get(index)
    if layer is None:
        body.addWidget(label('Not in this capture.', muted=True))
        return host
    body.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, f'layer {index}')))
    rows = []
    for name, tensor in layer.tensors.items():
        written, _ = shape_line(tensor, omit_batch=True)
        rows.append((name, written, str(tensor.dtype)))
    body.addWidget(read_only_table(['Captured tensor', 'Shape', 'Stored dtype'],
                                   sorted(rows), stretch=0))
    return host


# ── 12  The token decision ──────────────────────────────────────────────────

def build_decide(ctx: NotebookContext, section) -> QWidget:
    capture = ctx.capture
    host, body = _panel()
    facts = ctx.facts
    body.addWidget(VocabularyProjection(
        int(facts['hidden_size']), int(facts.get('vocab_size_raw') or 0),
        positions=int(facts['position_count']),
        final_position=max(int(facts['position_count']) - 1, 0),
        marked=int(facts.get('candidate_top_k') or 0), tokens=ctx.tokens,
        caption='Every position was computed. Only the final one has a next token that '
                'has not been written yet, and only its score row was stored.'))
    body.addWidget(ctx.equation(('final_logits', '→ argmax →',
                                 'selected token ID')))
    body.addWidget(ctx.variables(('layer_output', 'X_final_norm', 'W_U', 'final_logits',
                                 'prefill_logits_full')))

    logits = capture.logits
    if logits is None:
        body.addWidget(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'not retained')))
        body.addWidget(label(LOGITS_UNAVAILABLE))
        body.addWidget(label(
            f'The token this run selected is still known — it is '
            f'{capture.tokens[0]!r}, vocabulary ID {capture.token_ids[0]} — because the '
            f'emitted text was saved. What cannot be shown is the score vector behind it.',
            muted=True, small=True))
        return host

    flat = np.asarray(logits).reshape(-1)
    chosen = int(flat.argmax())
    body.addWidget(badge_row(
        EvidenceBadge(EVIDENCE_OBSERVED, 'final_logits'),
        EvidenceBadge(EVIDENCE_DERIVED, 'the ranking and the argmax')))
    body.addWidget(label(
        f'The last prompt position is position {len(capture.prompt_tokens) - 1}. Its '
        f'score vector holds {flat.size:,} numbers, one per vocabulary entry.'))
    body.addWidget(_candidates(ctx, flat))
    body.addWidget(ctx.values(
        [chosen], labels=['selected_token_id'], kind=EVIDENCE_DERIVED, formatter=str,
        caption='argmax of the captured scores — the position of the largest',
        note='calculated for this display', total=1))
    body.addWidget(ctx.values(
        [flat[chosen]], labels=['its captured score'], caption='from final_logits',
        note='final_logits', total=1))
    body.addWidget(label(
        f'Decoded, vocabulary ID {chosen} is the token this run emitted: '
        f'{capture.tokens[0]!r}. The model produced the scores; the generation code '
        f'applied the recorded selection rule — {facts["decoding_rule"]} — to pick '
        f'among them.'))
    check = _decide_check(ctx, flat)
    if check is not None:
        body.addWidget(check)
    return host


def _candidates(ctx: NotebookContext, flat) -> QWidget:
    """The best-scoring entries, preferring the run's own recorded list.

    The recorded list and a ranking computed here can disagree on ties, so where the run
    saved one it is used; the fallback is marked as TensorScope's own ranking.
    """
    trace = ctx.capture.generation_trace
    if trace:
        rows = [(rank + 1, entry['id'], repr(str(entry.get('token', ''))),
                 f'{float(entry["logit"]):.6f}')
                for rank, entry in enumerate(trace[0].top_candidates)]
        return read_only_table(['Rank', 'Token ID', 'Token', 'Captured score'], rows,
                               stretch=2, highlight=0)
    order = np.argsort(-np.asarray(flat))[:10]
    rows = [(rank + 1, int(index), '', exact_scalar(flat[index]))
            for rank, index in enumerate(order)]
    return read_only_table(['Rank', 'Token ID', 'Token', 'Captured score'], rows,
                           stretch=2, highlight=0)


def _decide_check(ctx: NotebookContext, flat):
    """The ratio question, only where this run has two real scores to put in it."""
    order = np.argsort(-np.asarray(flat))[:2]
    if len(order) < 2:
        return None
    winner, runner = float(flat[order[0]]), float(flat[order[1]])
    if runner == 0:
        return None
    return ctx.check('decide', winner=f'{winner:.4g}', runner_up=f'{runner:.4g}',
                     ratio=f'{winner / runner:.2f}')


# ── The continuation, and the closing summary ───────────────────────────────

def build_continuation(ctx: NotebookContext, section) -> QWidget:
    capture = ctx.capture
    host, body = _panel()

    body.addWidget(label('THE ANSWER — SELECT ANY TOKEN', muted=True))
    view = ResponseTextView(capture.tokens, capture.response, selected=0)
    # linkActivated is mouse-only, so the anchors alone would put every token after the
    # first out of reach from the keyboard.
    view.setFocusPolicy(Qt.StrongFocus)
    body.addWidget(view)
    body.addWidget(_token_stepper(ctx, view))

    slot = ctx.plain_slot(body)

    def show(step: int) -> None:
        replace_in(slot, _decision(ctx, step))

    view.selected.connect(show)
    show(view.current())

    body.addWidget(label(
        ctx.text('Later steps reuse the same learned parameters — nothing is retrained '
                 'between tokens. ') + CACHED_KV_NOTE, muted=True, small=True))
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
        caption.setText(f'token {view.current()} of {len(ctx.capture.tokens) - 1}')
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
        f'Step {step}   ·   selected {capture.tokens[step]!r}   ·   '
        f'vocabulary ID {capture.token_ids[step]}'))
    trace = capture.generation_trace
    if step >= len(trace):
        body.addWidget(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'not recorded')))
        body.addWidget(label(TELEMETRY_UNAVAILABLE))
        return host
    decision = trace[step]
    tier = EVIDENCE_OBSERVED if (step == 0 or decision.attested) else EVIDENCE_UNATTESTED
    body.addWidget(badge_row(EvidenceBadge(tier, 'scores from the forward pass'),
                             EvidenceBadge(EVIDENCE_DERIVED, 'the ranking')))
    if tier is EVIDENCE_UNATTESTED:
        body.addWidget(label(
            'Real output of the real forward pass, but only the prompt pass was re-run '
            'under stock attention, so no verification gate independently re-derived '
            'this one.', muted=True, small=True))
    winner = next((position for position, entry in enumerate(decision.top_candidates)
                   if int(entry['id']) == decision.selected_token_id), None)
    body.addWidget(read_only_table(
        ['Rank', 'Token ID', 'Token', 'Captured score'],
        [(rank + 1, entry['id'], repr(str(entry.get('token', ''))),
          f'{float(entry["logit"]):.6f}')
         for rank, entry in enumerate(decision.top_candidates)],
        stretch=2, highlight=winner))
    body.addWidget(label(
        ctx.text('The model produced these scores; the generation code applied the '
                 'recorded selection rule — {decoding_rule} — to them.'),
        muted=True, small=True))
    return host


def _stopping(ctx: NotebookContext) -> QWidget:
    """The stopping decision, shown as its own step when the run recorded one."""
    facts = ctx.facts
    host, body = _panel(4)
    body.addWidget(label('WHY IT STOPPED', muted=True))
    body.addWidget(label(EOS_IS_A_SCORED_ENTRY))
    body.addWidget(label(facts['stop_reason_text']))
    if not facts['stop_token_known']:
        body.addWidget(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'not recorded')))
        body.addWidget(label(
            'This run does not record which entry won the final scoring step, so '
            'TensorScope does not say.', muted=True, small=True))
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
        ctx.text('That scoring step ran exactly like the {generated_count} before it. Its '
                 'winner was the stopping marker, so the generation code stopped and did '
                 'not append it — which is why the answer is {generated_count} tokens '
                 'and not {generated_count} plus one.'), muted=True, small=True))
    return host


def _summary(ctx: NotebookContext) -> QWidget:
    facts = ctx.facts
    host, body = _panel(4)
    body.addWidget(label('THIS RUN, IN FULL', muted=True))
    body.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'counts of stored values')))
    body.addWidget(read_only_table(['Quantity', 'This run'], [
        ('Prompt token positions', facts['position_count']),
        ('Emitted response tokens', facts['generated_count']),
        ('Tokens after the first', facts['continuation_count']),
        ('Layers', facts['layer_count']),
        ('Query heads', facts['query_head_count']),
        ('Key/value heads', facts['kv_head_count']),
        ('Features per head', facts['head_dim']),
        ('Captured phases', facts['phase_count']),
        ('Selection rule', facts['decoding_rule']),
        ('Stop reason', facts['stop_reason_text']),
    ], stretch=1))
    body.addWidget(label(
        'Detailed layer tensors exist for two passes only: the prompt prefill and the '
        'first generated-token pass. For every later step this run kept the candidate '
        'scores and nothing else — no layer tensors, and no full vocabulary vector.',
        muted=True, small=True))
    return host


def _limits(ctx: NotebookContext) -> QWidget:
    host, body = _panel(6)
    body.addWidget(label('WHAT THIS CAPTURE ESTABLISHES', muted=True))
    body.addWidget(bullets(WHAT_WE_KNOW))
    body.addWidget(label('WHAT IT DOES NOT', muted=True))
    body.addWidget(bullets(WHAT_WE_CANNOT_CONCLUDE))
    body.addWidget(label('HOW TO READ THE BADGES', muted=True))
    for kind, (name, description) in EVIDENCE_KINDS.items():
        body.addWidget(badge_row(EvidenceBadge(kind)))
        body.addWidget(label(description, muted=True, small=True))
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
