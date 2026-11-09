"""Page 1: Understand the calculation.

Twelve topics in the same order and the same notation as the worked notebook, each one
purpose, formula, symbols, picture, and -- only on request -- a longer explanation.  A
reader is expected to arrive here from an operation they are in the middle of, read one
topic, and go back; the "Follow this run" button on every topic is the way back, and the
notebook has the matching button pointing here.

Nothing on this page reads a captured *value*.  It reads this run's shapes and counts --
how many layers, how wide a head, how many heads share a key/value head, how many
positions the prompt became -- so the pictures and the formulas describe the model the
reader is actually looking at rather than a generic one.  That line matters: the moment a
reference page starts quoting the numbers a model computed, a reader can no longer tell
which page is telling them about the architecture and which is telling them about the run.
Captured values live on the other page, next to the badge that says where they came from.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from tensorscope_common import (
    AnchoredScrollPage, Card, Disclosure, EvidenceBadge, Section, badge_row, formula_block,
    label, measure_label, part_label,
)
from tensorscope_content import (
    EVIDENCE_CONCEPTUAL, ILLUSTRATIONS, OPERATION_INDEX, REFERENCE_TOPICS, REFERENCE_TOPIC_INDEX,
    ROW_VECTOR_NOTE, learn_facts,
)
from tensorscope_diagrams import (
    CausalMaskTriangle, EmbeddingLookupRow, GatedFFNPath, HeadSplit, KVShare, LayerStack,
    MatMulShapes, MatrixGrid, QKCompare, ResidualBypass, SoftmaxFlow, StoppingLoop,
    TrainingVsInference, VectorStrip, VocabularyProjection,
)
from tensorscope_variables import VariableRegistry


REFERENCE_NAV = [(topic.key, topic.nav) for topic in REFERENCE_TOPICS]

# The notebook operation each topic links to, so "Follow this run" lands somewhere useful.
# The first operation a topic names is the one it opens; the rest are what it also covers.
TOPIC_ENTRY = {topic.key: (topic.operations[0] if topic.operations else '')
               for topic in REFERENCE_TOPICS}


def prose(text: str) -> QWidget:
    """A paragraph of explanation, with blank lines and bold runs honoured.

    The copy module stays plain text so it can be read, diffed and formatted against
    `learn_facts()` without HTML in the way; the two conversions it needs happen here.
    """
    body = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    pieces = body.split('**')
    # Odd positions were between a pair of asterisks.
    body = ''.join(piece if index % 2 == 0 else f'<b>{piece}</b>'
                   for index, piece in enumerate(pieces))
    return label(body.replace('\n\n', '<br><br>').replace('\n', '<br>'), rich=True)


def _explains(topic) -> str:
    """'Explains operation 4' -- so the reader can line a topic up with the worked run."""
    numbers = sorted(OPERATION_INDEX[key].number for key in topic.operations
                     if key in OPERATION_INDEX)
    if not numbers:
        return ''
    if len(numbers) == 1:
        return f'Explains operation {numbers[0]}'
    return 'Explains operations ' + ', '.join(str(number) for number in numbers)


class ReferenceView(AnchoredScrollPage):
    """The illustrated reference, as one scrolling page of anchored topics."""

    NAV = REFERENCE_NAV
    NOUN = 'Topic'
    NEXT_LABEL = 'Next topic →'
    PREVIOUS_LABEL = '← Previous topic'

    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(capture, parent=parent, tokens=tokens)
        self.facts = learn_facts(capture)
        # The reference page has no phase, layer or position of its own; it borrows a
        # registry purely so its symbol glossary uses the notebook's own definitions and
        # cannot drift into a second set of names.
        self.registry = VariableRegistry(capture, self.facts, parent=self)
        for position, topic in enumerate(REFERENCE_TOPICS, start=1):
            self.add_section(Section(
                topic.key, topic.heading.format(**self.facts),
                topic.purpose.format(**self.facts), number=f'{position:02}',
                kicker=_explains(topic),
                preface=self._preamble() if position == 1 else None,
                build_body=lambda section, t=topic: self._body(t)))
        self.finish_sections()
        self.go_to(REFERENCE_TOPICS[0].key)

    def context(self) -> str:
        topic = REFERENCE_TOPIC_INDEX.get(self.current_stage())
        return topic.nav if topic else ''

    def progress_text(self, key: str) -> str:
        keys = self.stage_keys()
        return (f'Topic {keys.index(key) + 1} of {len(keys)}' if key in keys else '')

    # ── Chrome ───────────────────────────────────────────────────────────────

    def _preamble(self) -> QWidget:
        # The page's one framing sentence, at the top of the first topic rather than in a
        # band pinned above the page: it needs reading once, not on every screen.
        host = QWidget()
        host.setObjectName('referencePreamble')
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 4)
        row.addWidget(label(
            'The ideas behind each operation, in the order the model performs them. '
            'Nothing here is a captured value — open "Follow this run" for those.',
            muted=True, small=True), 1)
        return host

    # ── One topic ────────────────────────────────────────────────────────────

    def _body(self, topic) -> QWidget:
        host = QWidget()
        body = QVBoxLayout(host)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)

        if topic.formula:
            body.addWidget(part_label('The operation'))
            body.addWidget(formula_block(topic.formula.format(**self.facts)))

        if topic.key == 'notation':
            body.addWidget(label(ROW_VECTOR_NOTE, muted=True, small=True))

        if topic.symbols:
            body.addWidget(part_label('The symbols in it'))
            body.addWidget(self._glossary(topic.symbols))

        for name in topic.visual:
            widget = self._visual(name)
            if widget is not None:
                body.addWidget(widget)
        if topic.caption:
            body.addWidget(label(topic.caption.format(**self.facts), muted=True, small=True))

        if topic.illustration:
            body.addWidget(self._illustration(topic.illustration))

        if topic.deeper:
            body.addWidget(Disclosure(
                'Go deeper', lambda text=topic.deeper: prose(text.format(**self.facts))))

        entry = TOPIC_ENTRY.get(topic.key, '')
        if entry:
            button = QPushButton('Follow this in the run  →')
            button.setObjectName('crossPageLink')
            button.setToolTip('Open the worked notebook at the operation this topic '
                              'explains.')
            button.clicked.connect(
                lambda _, key=entry: self.jump_requested.emit('run', key))
            body.addWidget(button, 0, Qt.AlignLeft)
        return host

    def _glossary(self, symbols) -> QWidget:
        """Each symbol with its one-line meaning, read from the shared definitions.

        Not clickable: a symbol on this page has no phase, layer or head behind it, and a
        panel that said "[15 x 2,560]" here would be quoting a run the reader has not
        opened.  Shapes stay on the run page, where they are facts.
        """
        # One grid, so every meaning starts at the same x and the symbol column is only
        # as wide as its widest symbol.
        host = QWidget()
        rows = QGridLayout(host)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setHorizontalSpacing(18)
        rows.setVerticalSpacing(6)
        index = 0
        for key in symbols:
            if not self.registry.known(key):
                continue
            resolved = self.registry.resolve(key)
            symbol = label(resolved.symbol)
            symbol.setObjectName('glossarySymbol')
            symbol.setWordWrap(False)
            rows.addWidget(symbol, index, 0, Qt.AlignLeft | Qt.AlignTop)
            meaning = measure_label(resolved.role or resolved.title, width=640)
            meaning.setObjectName('journeySmall')
            rows.addWidget(meaning, index, 1)
            index += 1
        # The spare width goes to an empty last column, not between symbol and meaning.
        rows.setColumnStretch(0, 0)
        rows.setColumnStretch(1, 0)
        rows.setColumnStretch(2, 1)
        rows.setColumnMinimumWidth(1, 520)
        return host

    def _illustration(self, key: str) -> QWidget:
        """Invented content, in a box that cannot be mistaken for a captured result."""
        illustration = ILLUSTRATIONS[key]
        card = Card(illustration.heading)
        card.setObjectName('illustrationBox')
        card.add(badge_row(EvidenceBadge(EVIDENCE_CONCEPTUAL, 'illustration only')))
        card.add(label(illustration.disclaimer, muted=True, small=True))
        card.add(prose(illustration.body))
        return card

    # ── Pictures ─────────────────────────────────────────────────────────────

    def _visual(self, name: str):
        """Build a topic's diagram from this run's geometry.

        Every argument is a number `learn_facts` read off a captured array, so the picture
        describes the model in front of the reader.  A run whose geometry could not be read
        leaves zeros, and a diagram asked to draw nothing is simply left out.
        """
        facts = self.facts
        tokens = self._tokens
        hidden = int(facts['hidden_size'])
        heads = int(facts['query_head_count'])
        head_dim = int(facts['head_dim'])
        positions = int(facts['position_count'])
        projected = int(facts['q_projection_width'])
        vocabulary = int(facts.get('vocab_size_raw') or 0)

        if name == 'token_ids':
            return VectorStrip(positions, symbol='ids', role='layer', unit='token positions',
                               tokens=tokens,
                               caption='One cell per position, each holding one '
                                       'vocabulary ID.')
        if name == 'embedding_lookup':
            # A run with no stored logits vector never recorded the vocabulary size.  The
            # diagram draws anyway and says the row count is unknown, because the point it
            # makes is about the width of one row.
            if not hidden:
                return None
            return EmbeddingLookupRow(vocabulary, hidden, tokens=tokens,
                                      caption='A token ID is a row number. The row is the '
                                              'vector.')
        if name == 'input_tensor':
            if not (hidden and positions):
                return None
            return MatrixGrid(positions, hidden, row_axis='token position',
                              col_axis='model feature', role='layer', symbol='X',
                              tokens=tokens,
                              caption='Positions down, features across. The batch axis is '
                                      'in front of both and is always 1 here.')
        if name == 'training_vs_inference':
            return TrainingVsInference(
                tokens=tokens,
                caption='The capture recorded one pass of the lower row.')
        if name == 'norm_pair':
            if not (hidden and positions):
                return None
            return MatrixGrid(positions, hidden, row_axis='token position',
                              col_axis='model feature', role='norm', symbol='X_norm',
                              tokens=tokens,
                              caption='The same arrangement as X. Only the values differ, '
                                      'and no number crossed between rows.')
        if name == 'qkv_projection':
            if not (hidden and projected and positions):
                return None
            return MatMulShapes((positions, hidden), (projected, hidden),
                                (positions, projected), left_symbol='X_norm',
                                right_symbol='W_Qᵀ', out_symbol='Q_raw', tokens=tokens,
                                caption='The shared input width is the reason this '
                                        'multiplication is defined.')
        if name == 'head_split':
            if not (hidden and projected and heads and head_dim):
                return None
            return HeadSplit(hidden, projected, heads, head_dim, positions, tokens=tokens,
                             caption='Two different kinds of step, with a boundary between '
                                     'them: one computes, one regroups.')
        if name == 'kv_share':
            if not (heads and int(facts['kv_head_count'])):
                return None
            return KVShare(int(facts['kv_head_count']), heads, tokens=tokens,
                           caption='Each key/value head is copied out to the query heads '
                                   'that share it, before any comparison.')
        if name == 'layer_stack':
            return LayerStack(int(facts['layer_count']) or 1, 0, tokens=tokens,
                              caption='Layers are sequential, and none of their parameters '
                                      'are shared.')
        if name == 'qk_compare':
            if not head_dim:
                return None
            return QKCompare(head_dim, query_pos=0, key_pos=0, tokens=tokens,
                             caption='One dot product per query-key pair. That is the '
                                     'whole of the decision.')
        if name == 'attention_cells':
            if not positions:
                return None
            return MatrixGrid(positions, positions, row_axis='query token',
                              col_axis='key token', role='scores', symbol='S',
                              clickable=True, tokens=tokens,
                              caption='One cell per query-key pair: how relevant that '
                                      'key is to that query.')
        if name == 'softmax_flow':
            return SoftmaxFlow(tokens=tokens,
                               caption='Deliberately numberless: the shape of the '
                                       'operation, before any captured value.')
        if name == 'causal_mask':
            if not positions:
                return None
            return CausalMaskTriangle(positions, positions, tokens=tokens,
                                      caption='Added before softmax, so the blocked '
                                              'weights come out at effectively zero.')
        if name == 'residual':
            return ResidualBypass([('attention block', 'O', 'H'),
                                   ('feed-forward network', 'F', 'X_next')],
                                  trunk_symbols=('X', 'H'), tokens=tokens,
                                  caption='Each block proposes a change to the stream '
                                          'rather than replacing it.')
        if name == 'gated_ffn':
            if not hidden:
                return None
            # The interior width is genuinely not in a capture, so it is never a number.
            return GatedFFNPath(hidden, None, tokens=tokens,
                                caption='Gated: one branch decides how much of the other '
                                        'to let through.')
        if name == 'vocabulary_projection':
            if not (hidden and positions):
                return None
            return VocabularyProjection(hidden, vocabulary, positions=positions,
                                        final_position=max(positions - 1, 0),
                                        tokens=tokens,
                                        caption='Every position is computed; only the last '
                                                'one has an unwritten next token.')
        if name == 'stopping_loop':
            return StoppingLoop(
                tokens=tokens,
                caption='One loop, and one way out of it.')
        return None
