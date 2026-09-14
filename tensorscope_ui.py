"""Read-only computation journey. This module never loads a model or writes a run.

All numerical views reference arrays on the supplied RunCapture. Equations describe
operations; unavailable operands/intermediates are explicitly marked as conceptual.
"""
from __future__ import annotations

import html
import json

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QAbstractItemView, QHeaderView,
)

from tensor_widgets import AttentionExplorer, TensorInspector
from tensorscope_content import (
    LOGITS_UNAVAILABLE, STORY_STAGE_INDEX, TENSOR_LABELS,
    TENSOR_LABEL_BY_KEY, TENSOR_EXPLANATIONS, story_facts,
)


STAGES = [('prompt', 'Prompt & journey'), ('tokenization', 'Tokens'),
          ('embedding', 'Embeddings'), ('layers', 'Transformer layers'),
          ('logits', 'Logits → first token'), ('coda', 'First token’s next pass')]
LAYER_STEPS = [('input', '1  Normalize'), ('qkv', '2  Project Q / K / V'),
               ('prepare', '3  Prepare for attention'), ('scores', '4  Scores → weights'),
               ('mix', '5  Mix values & project'), ('output', '6  Finish the layer')]
COLORS = {'q': '#60a5fa', 'k': '#c084fc', 'v': '#34d399',
          'scores': '#fbbf24', 'weights': '#22d3ee', 'output': '#fb923c'}


def label(text, *, rich=False, muted=False, title=False):
    result = QLabel(text)
    result.setTextFormat(Qt.RichText if rich else Qt.PlainText)
    result.setWordWrap(True)
    result.setTextInteractionFlags(Qt.TextSelectableByMouse)
    result.setObjectName('journeyTitle' if title else 'journeyMuted' if muted else 'journeyText')
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


class Card(QFrame):
    def __init__(self, title, plain='', parent=None):
        super().__init__(parent)
        self.setObjectName('journeyCard')
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(20, 18, 20, 18)
        self.body.setSpacing(12)
        self.add(label(title, title=True))
        if plain:
            self.add(label(plain, rich=True))

    def add(self, widget):
        self.body.addWidget(widget)
        return widget

    def field(self, heading, text, rich=False):
        self.add(label(heading.upper(), muted=True))
        return self.add(label(text, rich=rich))

    def equation(self, text):
        widget = label(text, rich=True)
        widget.setObjectName('journeyEquation')
        return self.add(widget)


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


class StoryView(QWidget):
    """One stage and one layer operation at a time, with persistent selection."""
    def __init__(self, capture, parent=None):
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
        builders = {'prompt': self._opening, 'tokenization': self._tokenization,
                    'embedding': self._embedding, 'layers': self._layers_stage,
                    'logits': self._word_choice, 'coda': self._coda}
        self._stage_widget = builders[key]()
        self.body.addWidget(self._stage_widget)
        if self._on_stage_change:
            self._on_stage_change(key)

    def _stage_card(self, key):
        stage = STORY_STAGE_INDEX[key]
        return Card(stage.heading.format(**self.facts), stage.plain.format(**self.facts))

    def _opening(self):
        card = self._stage_card('prompt')
        card.field('Your prompt', self.capture.prompt)
        card.field('The first selected token', f'{self.capture.tokens[0]!r}  ·  ID {self.capture.token_ids[0]}')
        card.field('Follow the computation',
                   'Each stage explains its purpose, then its operation and dimensions. Open the captured data when you want to inspect values.')
        for key, title in STAGES[1:-1]:
            button = QPushButton('↓  ' + title)
            button.setObjectName('journeyFlow')
            button.clicked.connect(lambda _, k=key: self.go_to(k))
            card.add(button)
        card.field('Two captured phases',
                   '<b>1 · Prompt prefill.</b> All prompt tokens pass through the layers. The final prompt position’s logits choose the first output token.<br><br>'
                   '<b>2 · First generated-token pass.</b> The chosen token is processed with the cached prompt keys and values. '
                   'Its embeddings and layer tensors are saved; its output logits are not saved. Later tokens’ forward-pass tensors are not captured.', rich=True)
        answer = Disclosure('Read the complete generated response', lambda: label(self.capture.response))
        card.add(answer)
        return card

    def _tokenization(self):
        card = self._stage_card('tokenization')
        card.field('Why it exists', 'Token IDs address a fixed vocabulary. A token can be a word, part of a word, punctuation, or a control marker.')
        card.equation('Prompt + chat formatting → tokenizer → ordered token IDs')
        card.field('Shape', f'T = {len(self.capture.prompt_tokens)} input tokens; batch B = {self.capture.embedding.shape[0]}.')
        card.field('How to read it',
                   'Position is the sequence index. The strings below are the saved tokenizer spellings, not a new decode. '
                   'Markers such as Ġ and Ċ commonly encode whitespace in these tokenizers. Chat control tokens can appear even though you did not type them. '
                   'The formatted prompt text and full tokenizer vocabulary are not persisted.')
        card.add(token_table(self.capture))
        card.field('Next', 'Each token ID selects one learned embedding vector.')
        return card

    def _phase_controls(self, layout):
        row = QHBoxLayout()
        row.addWidget(label('Captured phase'))
        self.phase_picker = QComboBox()
        self.phase_picker.addItems(['1 · Prompt prefill', '2 · First generated-token pass'])
        self.phase_picker.setCurrentIndex(int(self.generated))
        self.phase_picker.currentIndexChanged.connect(self._phase_changed)
        row.addWidget(self.phase_picker, 1)
        layout.addLayout(row)

    def _phase_changed(self, index):
        self.generated = bool(index)
        self.go_to(self._stage_key)

    def _inspect(self, name, tensor):
        generated = self.generated
        rows = token_labels(self.capture, generated,
                            keys=name in ('k_attended', 'v_attended'))
        columns = token_labels(self.capture, generated, keys=True) if name in (
            'attention_weights', 'attention_scores') else None
        return TensorInspector(tensor, name=name, axes=axes_for(name, tensor),
                               row_labels=rows if name != 'final_logits' else None,
                               column_labels=columns)

    def _tensor(self, card, name, tensor):
        """Shape and meaning precede any request to instantiate a numerical view."""
        meanings = axes_for(name, tensor)
        card.add(label(f'{name}   {shape(tensor)}   ·   stored {tensor.dtype}'))
        card.add(label('Axes: ' + ' × '.join(meanings), muted=True))
        card.add(Disclosure(f'Inspect captured {name}', lambda: self._inspect(name, tensor)))

    def _embedding(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        self._phase_controls(layout)
        card = self._stage_card('embedding')
        array = self.capture.generated_embedding if self.generated else self.capture.embedding
        card.field('Why it exists', 'Attention operates on feature vectors, not vocabulary IDs. The embedding lookup supplies the initial vector for each token.')
        card.equation('token ID → row of learned embedding table → X')
        card.field('How to read it', f'Each token has {array.shape[-1]} learned features. Columns are numerical coordinates, not named meanings. '
                   'This is the captured lookup output; the learned embedding table itself is not saved.')
        if self.generated:
            card.add(label(f'Input for this pass: {self.capture.tokens[0]!r}. Only this new token is embedded; prompt K/V are reused in attention.', muted=True))
        self._tensor(card, 'embedding', array)
        card.field('Next', 'Layer 0 receives this tensor and normalizes each token’s feature vector before its attention projections.')
        layout.addWidget(card)
        return host

    def _layers_stage(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self._phase_controls(layout)
        row = QHBoxLayout()
        self.previous_layer = QPushButton('← Previous layer')
        self.next_layer = QPushButton('Next layer →')
        self.picker = QComboBox()
        indices = sorted(self.capture.layers)
        for index in indices:
            self.picker.addItem(f'Layer {index}', index)
        self.picker.setCurrentIndex(indices.index(self.layer_index))
        self.picker.currentIndexChanged.connect(self._layer_changed)
        self.previous_layer.clicked.connect(lambda: self.picker.setCurrentIndex(self.picker.currentIndex() - 1))
        self.next_layer.clicked.connect(lambda: self.picker.setCurrentIndex(self.picker.currentIndex() + 1))
        self.previous_layer.setEnabled(self.layer_index != indices[0])
        self.next_layer.setEnabled(self.layer_index != indices[-1])
        row.addWidget(self.previous_layer)
        row.addWidget(self.picker, 1)
        row.addWidget(self.next_layer)
        layout.addLayout(row)
        layout.addWidget(label(f'All {len(indices)} layers repeat this general computation with different learned parameters and activations. '
                               'A layer preserves the hidden width so its output can feed the next layer.', muted=True))
        self.step_picker = QComboBox()
        for key, title in LAYER_STEPS:
            self.step_picker.addItem(title, key)
        self.step_picker.setCurrentIndex([k for k, _ in LAYER_STEPS].index(self.step_key))
        self.step_picker.currentIndexChanged.connect(self._step_changed)
        layout.addWidget(self.step_picker)
        self.layer_section = self._layer_card()
        layout.addWidget(self.layer_section)
        controls = QHBoxLayout()
        prev = QPushButton('← Previous operation')
        nxt = QPushButton('Next operation →')
        pos = [k for k, _ in LAYER_STEPS].index(self.step_key)
        prev.setEnabled(pos > 0)
        nxt.setEnabled(pos < len(LAYER_STEPS) - 1)
        prev.clicked.connect(lambda: self.step_picker.setCurrentIndex(pos - 1))
        nxt.clicked.connect(lambda: self.step_picker.setCurrentIndex(pos + 1))
        controls.addWidget(prev)
        controls.addStretch(1)
        controls.addWidget(nxt)
        layout.addLayout(controls)
        return host

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
        q, k, v = (tensors[n] for n in ('q_attended', 'k_attended', 'v_attended'))
        # Dimensions are taken from the captured operands, never hidden_size / heads.
        tq, tk, d = q.shape[-2], k.shape[-2], q.shape[-1]
        hidden = norm.shape[-1]
        heads = q.shape[-3]
        step = self.step_key
        if step == 'input':
            card = Card('Normalize the incoming representation', TENSOR_EXPLANATIONS['normalized_input'])
            previous = sorted(layers).index(self.layer_index) - 1
            source = 'embedding' if previous < 0 else f'layer_output from layer {sorted(layers)[previous]}'
            card.field('Where it came from', f'The {source} is this layer’s input X.')
            card.field('Why the model needs it', 'Normalization controls the scale of each token’s features before the learned projections, helping keep a deep stack numerically well behaved.')
            card.equation('X → input_layernorm(X) → X_norm<br>For Qwen RMSNorm: x / √(mean(x²) + ε) ⊙ g')
            card.add(label('Equation only: normalization parameters g and ε and its internal arithmetic are not stored in the run.', muted=True))
            self._tensor(card, 'normalized_input', norm)
            card.field('Next', 'The same normalized vector feeds three different learned linear projections: Q, K, and V.')
        elif step == 'qkv':
            card = Card('Three projections, three roles', 'Q and K define which tokens interact. V supplies the feature content that attention will combine. '
                        'These are learned linear transformations of the same normalized input.')
            card.field('Operation', 'For PyTorch Linear: X_norm @ Wᵀ + bias (where present). The saved run contains each projection output; it does not contain W or bias.')
            for name, role in [('q', 'Query · what features this position seeks'),
                               ('k', 'Key · features against which queries are compared'),
                               ('v', 'Value · content available for weighted mixing')]:
                width = tensors[name].shape[-1]
                color = COLORS[name]
                card.equation(f'<span style="color:{color}"><b>{name.upper()} · {role}</b></span><br>'
                              f'[{tq} × <b>{hidden}</b>] @ [<b>{hidden}</b> × {width}] → [{tq} × {width}]')
                card.add(label(f'Inner dimensions {hidden} match. Weight dimensions are inferred from input/output shapes for the equation; weight values are unavailable. Batch axis is retained below.', muted=True))
                card.add(label(TENSOR_EXPLANATIONS[name], rich=True))
                self._tensor(card, name, tensors[name])
            card.field('Next', 'Split projection features into heads, apply architecture-specific preparation, and align Q/K/V for attention.')
        elif step == 'prepare':
            card = Card('Before preparation → tensors used by attention',
                        'A head is one parallel attention computation with its own query/key feature subspace. '
                        'Several query heads can share key/value heads (grouped-query attention, GQA).')
            is_qwen3 = 'qwen3' in str(self.capture.metadata.get('model', '')).lower()
            prep = ('For this Qwen3 model, preparation includes per-head Q/K normalization and rotary position encoding (RoPE) on Q/K. '
                    if is_qwen3 else 'Preparation can include model-specific Q/K normalization and positional transformations such as RoPE. ')
            card.field('What happens', prep + 'RoPE rotates pairs of Q/K coordinates according to token position, so dot products can depend on relative position. '
                       'K/V heads are expanded by repeat_kv for the attention computation. These internal steps are not separately persisted.')
            if self.generated:
                card.field('KV cache', 'The new token supplies one new query and new K/V. Cached prompt K/V are included before attention. '
                           f'That is why query length is {tq} but key length is {tk}. Prompt tokens are not run through the entire layer again.')
            for name in ('q', 'k', 'v'):
                attended = name + '_attended'
                card.equation(f'<span style="color:{COLORS[name]}"><b>{name.upper()}</b></span> '
                              f'{shape(tensors[name])} → {shape(tensors[attended])}')
                card.add(label(TENSOR_EXPLANATIONS[attended], rich=True))
                self._tensor(card, attended, tensors[attended])
            card.field('Dimensions in this run', f'B = {q.shape[0]} batch · H = {heads} query heads · d_head = {d} features per Q/K head. '
                       f'd_model = {hidden}. These are independent dimensions read from this run; H × d_head need not equal d_model.')
            card.field('Next', 'Use the prepared Q and K, not the raw projections, to calculate attention scores.')
        elif step == 'scores':
            card = Card('From comparisons to a distribution',
                        'Each query vector is compared with every available key vector. A dot product measures their learned compatibility. '
                        'Softmax then determines how much of each value vector the query will receive.')
            card.equation(f'<span style="color:{COLORS["q"]}">Q [{tq} × <b>{d}</b>]</span> @ '
                          f'<span style="color:{COLORS["k"]}">Kᵀ [<b>{d}</b> × {tk}]</span> → QKᵀ [{tq} × {tk}]<br>'
                          'QKᵀ → scaling + causal mask → <b>captured scores</b> → softmax over keys → <b>captured weights</b>')
            card.field('Why the dimensions work', f'For one head and batch, the two inner dimensions both equal {d}. '
                       f'Each of the {tq} queries yields {tk} dot products. Full score/weight axes are [batch, head, query token, key token].')
            card.field('Captured versus conceptual',
                       'The saved scores are already scaled and masked: the exact input handed to softmax. '
                       'The unscaled QKᵀ product, scale parameter and mask alone are not saved. The conventional scale is 1/√d_head; no scale value is reconstructed here.')
            card.field('How to read the grids',
                       'Row = query token; column = key token. Softmax exponentiates and normalizes across each row. '
                       'Captured weights sum approximately to one because of dtype rounding. They are mixing weights, not vocabulary probabilities or a causal explanation of the answer. '
                       'Future positions have suppressed scores (possibly very negative finite values or −∞) and zero weights under causal masking.')
            if self.generated:
                card.add(label('This is a one-query pass over cached prompt keys plus the new token. All displayed keys are available; do not expect a triangular future-token region.', muted=True))
            card.add(label(f'Captured scores {shape(tensors["attention_scores"])} → weights {shape(tensors["attention_weights"])}'))
            self.attention = AttentionExplorer(tensors, token_labels(self.capture, self.generated),
                                              token_labels(self.capture, self.generated, keys=True),
                                              layer_index=self.layer_index,
                                              phase='first_generated_token' if self.generated else 'prefill')
            card.add(self.attention)
            card.field('Next', 'Use each captured weight row to combine V vectors, then join head results and apply the output projection.')
        elif step == 'mix':
            dv = v.shape[-1]
            card = Card('Mix value vectors, then project the head results', TENSOR_EXPLANATIONS['attention_output'])
            card.field('Why the model needs it', 'The weights determine which positions contribute. Multiplication by V transports their feature content into each query position. '
                       'The output projection mixes head features back into the residual stream’s hidden width.')
            card.equation(f'<span style="color:{COLORS["weights"]}">weights [{tq} × <b>{tk}</b>]</span> @ '
                          f'<span style="color:{COLORS["v"]}">V [<b>{tk}</b> × {dv}]</span> → head result [{tq} × {dv}]<br>'
                          f'concat({heads} heads) [{tq} × {heads * dv}] → Linear_o → [{tq} × {hidden}]')
            card.field('Capture boundary', 'The per-head weights @ V result and the concatenated tensor before o_proj are NOT persisted. '
                       'The diagram gives their conceptual shapes from the captured operands, not numerical tensors. '
                       'attention_output is the captured output of o_proj, after that learned output projection.')
            self._tensor(card, 'v_attended', v)
            self._tensor(card, 'attention_output', tensors['attention_output'])
            card.field('Next', 'The decoder layer combines the attention branch with its residual stream and runs its remaining computation.')
        else:
            card = Card('Finish this decoder layer', TENSOR_EXPLANATIONS['layer_output'])
            card.equation('Captured attention_output<br>↓<br>'
                          'Residual addition → post-attention normalization → feed-forward / MLP → residual addition<br>'
                          '<i>Conceptual decoder computation · these intermediates are not separately captured</i><br>↓<br>'
                          'Captured layer_output')
            card.field('Why there is more computation', 'Attention exchanges information between positions. The feed-forward network transforms features at each position. '
                       'Residual paths add branch results back to the running representation. The shown order describes the supported Qwen dense decoder blocks.')
            self._tensor(card, 'layer_output', tensors['layer_output'])
            last = self.layer_index == sorted(layers)[-1]
            card.field('Next', ('After the last layer, final normalization and the vocabulary projection produce logits. '
                               'Their internal tensors and learned weights are not saved; only the prefill final-position logits vector is persisted.') if last else
                       f'This [{tq} × {hidden}] representation (plus batch axis) becomes the next layer’s input. The token count and hidden width are compatible without reshaping.')
            button = QPushButton('Go to final token decision →' if last else 'Follow into the next layer →')
            if last:
                button.clicked.connect(lambda: self.go_to('logits'))
            else:
                button.clicked.connect(self._follow_layer)
            card.add(button)
        return card

    def _follow_layer(self):
        indices = sorted(self.capture.layers)
        self.layer_index = indices[indices.index(self.layer_index) + 1]
        self.step_key = 'input'
        self.go_to('layers')

    def _word_choice(self):
        card = self._stage_card('logits')
        position = len(self.capture.prompt_tokens) - 1
        card.field('Which position makes this decision?', f'Prompt prefill · final position {position} · stored token spelling {self.capture.prompt_tokens[-1]!r}. '
                   'Its contextual representation includes the preceding prompt. This stage always describes prefill, regardless of the phase selected in the layer explorer.')
        card.equation('last layer → final normalization → vocabulary projection → logits at final prompt position → argmax → first token')
        card.field('What is a logit?', 'An unnormalized score for one vocabulary entry. It is not a probability. TensorScope uses greedy selection: argmax returns the index of the largest captured score. '
                   'No vocabulary probabilities are calculated or displayed.')
        logits = self.capture.logits
        if logits is None:
            card.field('Logits not available in this saved run', LOGITS_UNAVAILABLE)
        else:
            card.field('Shape', f'{shape(logits)} · {len(logits):,} vocabulary entries · stored {logits.dtype}. '
                       'Only this one prompt position’s logits row is saved, not the full [batch, tokens, vocabulary] tensor.')
            # Rank is derived presentation metadata; every score is read from the array,
            # even if persisted top-k metadata is missing or stale.
            order = np.argsort(logits)[::-1][:8]
            names = {int(e['id']): str(e.get('token', '')) for e in self.capture.metadata.get('final_logits_top', []) if 'id' in e}
            names[self.capture.token_ids[0]] = self.capture.tokens[0]
            table = QTableWidget(len(order), 3)
            table.setHorizontalHeaderLabels(['Token ID', 'Saved token text', 'Captured logit'])
            table.verticalHeader().hide()
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            for row, token_id in enumerate(order):
                for col, text in enumerate((str(token_id), repr(names[int(token_id)]) if int(token_id) in names else 'Text not saved', str(logits[token_id].item()))):
                    table.setItem(row, col, QTableWidgetItem(text))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setFixedHeight(275)
            card.add(table)
            card.add(label('Display subset: the eight highest captured logits, sorted for reading. The full vector remains available below. Ties are resolved by argmax’s first index.', muted=True))
            card.add(Disclosure('Inspect all captured logits', lambda: TensorInspector(logits, name='final_logits', axes=['vocabulary token ID'])))
        card.field('First generated token', f'{self.capture.tokens[0]!r}  ·  ID {self.capture.token_ids[0]}')
        if logits is not None:
            card.field('Selection check', f'argmax(captured logits) = {int(logits.argmax())} = saved first token ID. '
                       'This index comparison is a display check on the saved vector. The stored capture metadata also attests bitwise equality with stock eager logits.')
        card.field('Next', 'The selected token is fed into the next captured pass. Its own forward pass did not choose itself; the prefill logits above chose it.')
        return card

    def _coda(self):
        card = self._stage_card('coda')
        card.field('New input', f'{self.capture.tokens[0]!r}  ·  token ID {self.capture.token_ids[0]}')
        tensor = self.capture.generated_layers[sorted(self.capture.generated_layers)[0]].tensors['attention_scores']
        card.equation(f'1 new token + cached prompt K/V → {len(self.capture.layers)} layers<br>'
                      f'Captured attention shape: {shape(tensor)} = [batch, head, new query, available keys]')
        card.field('What caching means', 'Previous tokens’ keys and values are retained and reused. Each layer processes the new token’s query against those cached keys plus its own key. '
                   'This avoids computing every prompt token’s layer activations again.')
        button = QPushButton('Explore the first generated-token pass →')
        button.setObjectName('primary')
        button.clicked.connect(self._enter_generated)
        card.add(button)
        card.field('Where the capture ends', 'The new token’s embedding and all required layer tensors are saved. Logits from this second pass are not persisted. '
                   'Later output tokens are retained as text/IDs only; their forward-pass tensors are not captured. '
                   'The full answer is context, not evidence of additional tensor capture.')
        card.add(Disclosure('Read the complete generated response', lambda: label(self.capture.response)))
        return card

    def _enter_generated(self):
        self.generated = True
        self.step_key = 'input'
        self.go_to('layers')


class DetailView(QWidget):
    """Virtual full-array browser for every captured tensor in either phase."""
    def __init__(self, capture, parent=None):
        super().__init__(parent)
        self.capture = capture
        self.inspector = None
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.addWidget(label('Raw / Detail · complete captured arrays', title=True))
        self.body.addWidget(label('Select phase, layer, and tensor. All values are accessible through virtual rows/columns and explicit batch/head slices. '
                                  'Reading a tensor does not change its stored values or dtype.', muted=True))
        row = QHBoxLayout()
        self.phase_picker = QComboBox()
        self.phase_picker.addItems(['Prompt prefill', 'First generated-token pass'])
        self.layer_picker = QComboBox()
        self.layer_picker.addItem('Outside the layers', None)
        for index in sorted(capture.layers):
            self.layer_picker.addItem(f'Layer {index}', index)
        self.tensor_picker = QComboBox()
        for title, picker in [('Phase', self.phase_picker), ('Layer', self.layer_picker), ('Tensor', self.tensor_picker)]:
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
            layers = self.capture.generated_layers if self.phase_picker.currentIndex() else self.capture.layers
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
            layers = self.capture.generated_layers if generated else self.capture.layers
            array = layers[self.layer_picker.currentData()].tensors[name]
            text = TENSOR_EXPLANATIONS.get(name, 'Additional captured tensor.')
        self.meaning.setTextFormat(Qt.RichText)
        self.meaning.setText(text)
        if self.inspector is not None:
            self.host.removeWidget(self.inspector)
            self.inspector.hide()
            self.inspector.deleteLater()
        rows = token_labels(self.capture, generated, keys=name in ('k_attended', 'v_attended'))
        self.inspector = TensorInspector(array, name=name, axes=axes_for(name, array),
                                         row_labels=rows if name != 'final_logits' else None,
                                         column_labels=token_labels(self.capture, generated, keys=True) if name in ('attention_scores', 'attention_weights') else None)
        self.host.addWidget(self.inspector)


class ComputationRecap(QDialog):
    def __init__(self, capture, run_id=None, parent=None, tokens=None):
        super().__init__(parent)
        capture.validate()
        self.capture = capture
        self.setWindowTitle('TensorScope — Computation journey')
        self.resize(1240, 880)
        self.setMinimumSize(850, 620)
        colors = tokens or {'card_bg': '#1c1c20', 'border': '#3b3b44', 'text_secondary': '#b5b5c1',
                            'text_primary': '#f0f0f2', 'accent': '#60a5fa', 'bg': '#111114'}
        self.setStyleSheet(f'''
            #journeyCard {{ background:{colors['card_bg']}; border:1px solid {colors['border']}; border-radius:10px; }}
            #journeyTitle {{ font-size:21px; font-weight:600; }}
            #journeyText {{ font-size:14px; }}
            #journeyMuted {{ color:{colors['text_secondary']}; font-size:12px; }}
            #journeyEquation {{ font-size:16px; padding:12px; background:{colors['bg']}; border-radius:6px; }}
            #journeyFlow {{ text-align:left; padding:10px; font-size:14px; }}
            QTableView {{ background:{colors['card_bg']}; color:{colors['text_primary']}; gridline-color:{colors['border']}; selection-background-color:{colors['accent']}; }}
            QHeaderView::section {{ background:{colors['bg']}; color:{colors['text_secondary']}; padding:6px; border:1px solid {colors['border']}; }}
        ''')
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        banner = label(f'{"Saved run #" + str(run_id) if run_id is not None else "Current capture"}  ·  '
                       f'{capture.metadata.get("model", "Model ID unavailable")}  ·  real PyTorch capture')
        banner.setStyleSheet(f'padding:12px; font-weight:600; border-bottom:1px solid {colors["border"]};')
        root.addWidget(banner)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        left = QWidget()
        left.setObjectName('sidebar')
        left.setMinimumWidth(190)
        left.setMaximumWidth(250)
        nav = QVBoxLayout(left)
        nav.setContentsMargins(12, 18, 12, 18)
        nav.addWidget(label('COMPUTATION JOURNEY', muted=True))
        self.story_button = QPushButton('Learn / Story')
        self.detail_button = QPushButton('Raw / Detail')
        for button in (self.story_button, self.detail_button):
            button.setCheckable(True)
            button.setObjectName('navItem')
            nav.addWidget(button)
        nav.addSpacing(18)
        self._nav_buttons = {}
        for n, (key, title) in enumerate(STAGES):
            button = QPushButton(f'{n + 1:02}  {title}')
            button.setObjectName('navItem')
            button.setCheckable(True)
            button.clicked.connect(lambda _, k=key: self._nav_to(k))
            nav.addWidget(button)
            self._nav_buttons[key] = button
        nav.addStretch(1)
        nav.addWidget(label(f'{len(capture.layers)} layers · {len(capture.prompt_tokens)} prompt tokens\n'
                            f'Model dtype: {capture.metadata.get("weight_dtype", "not recorded")}\n'
                            f'Stored embedding: {capture.embedding.dtype}', muted=True))
        nav.addWidget(label('Validated saved capture\nStock-eager match attested in metadata', muted=True))
        split.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 12, 20, 12)
        self.breadcrumb = label('')
        right_layout.addWidget(self.breadcrumb)
        self.provenance = Disclosure('Capture provenance & verification', self._provenance)
        right_layout.addWidget(self.provenance)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 6, 0)
        self.stack = QStackedWidget()
        self.story_view = StoryView(capture)
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
        split.setSizes([225, 1015])
        self.story_button.clicked.connect(self._show_story)
        self.detail_button.clicked.connect(self._show_detail)
        self._show_story()

    def _provenance(self):
        c = self.capture
        card = Card('Evidence for this run')
        m = c.metadata
        card.field('Source', str(m.get('capture_source', 'not recorded')))
        card.field('Validation', 'RunCapture.validate() passed on opening. Saved metadata attests the following checks at capture time:')
        card.field('Stock eager logits', f'Match: {m.get("logits_match_stock_eager", "not recorded")} · '
                   f'max absolute difference: {m.get("logits_max_abs_diff_vs_stock_eager", "not recorded")}')
        card.field('Attention reference', f'Faithful to upstream eager: {m.get("faithful_to_upstream_eager", "not recorded")} · '
                   f'verified calls: {m.get("attention_calls_verified", "not recorded")} · {len(c.layers) * 2} layer calls across the two captured phases')
        card.field('Runtime', f'{m.get("backend", "not recorded")} · torch {m.get("torch", "not recorded")} · {c.started_at}')
        card.field('Dtype', f'Model weights: {m.get("weight_dtype", "not recorded")} · stored embedding: {c.embedding.dtype}. '
                   'bfloat16 activations are widened losslessly to float32 for NumPy storage; values are preserved.')
        card.field('Prompt', c.prompt)
        card.add(Disclosure('Read all saved metadata', lambda: label(json.dumps(m, indent=2, ensure_ascii=False))))
        return card

    def _nav_to(self, key):
        self._show_story()
        self.story_view.go_to(key)

    def _on_story_stage_change(self, key):
        for k, button in self._nav_buttons.items():
            button.setChecked(k == key)
        idx = [k for k, _ in STAGES].index(key)
        self.previous_stage.setEnabled(idx > 0)
        self.next_stage.setEnabled(idx < len(STAGES) - 1)
        context = ''
        if key in ('layers', 'embedding'):
            context = ' / ' + ('First generated-token pass' if self.story_view.generated else 'Prompt prefill')
        if key == 'layers':
            context += f' / Layer {self.story_view.layer_index} / {dict(LAYER_STEPS)[self.story_view.step_key]}'
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
