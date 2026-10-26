"""Variables as the thing the reader clicks, and one shared panel that answers.

A variable here is a symbol in an equation plus a claim about where its numbers came from.
Clicking `X_norm` in the normalization step and `X_norm` in the projection step must reach
the same definition and the same array, so there is exactly one registry per capture and
exactly one panel per page.  A chip holds a *key*, not a tensor: the registry resolves it
against whichever phase, layer, head and position the reader currently has selected, which
is what lets a selection change repaint chips instead of rebuilding sections.

The honesty rules are structural rather than editorial:

* A key whose provenance is not `captured` has **no code path to a tensor widget**.  The
  panel does not build a sample row or an inspector rung for one, so a learned weight
  matrix or a discarded intermediate cannot accidentally be shown as data.  There is
  nothing to fabricate from, and nothing that would.
* A captured key resolves to the **stored array itself**, never a copy and never a
  resample.  `resolve()` hands back the same object the capture holds, so the inspector
  reads the model's own bytes.
* Shapes written short always carry `shape_line`'s caption saying the batch axis was
  dropped.  One helper, so no surface can drop it quietly.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from tensor_widgets import TensorInspector, exact_scalar
from tensorscope_common import (
    AnnotatedValueRow, Disclosure, EvidenceBadge, axes_for, badge_row, label, shape_line,
    token_labels,
)
from tensorscope_content import (
    CONCEPTUAL_VARIABLES, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED, EVIDENCE_OBSERVED,
    EXTRA_TENSOR_EXPLANATIONS, EXTRA_TENSOR_LABELS, INTERMEDIATE_NOT_KEPT,
    LEARNED_VALUES_NOT_SAVED, PROVENANCE_ABSENT, PROVENANCE_CAPTURED, PROVENANCE_DISPLAY,
    PROVENANCE_LEARNED, PROVENANCE_TIERS, TENSOR_EXPLANATIONS, TENSOR_LABEL_BY_KEY,
    TENSOR_SYMBOLS, VARIABLE_ROLES,
)


# Arrays that live on the capture rather than inside a layer, so they are addressed without
# a layer index.  `layer_input` is neither: see `VariableRegistry._captured_array`.
CAPTURE_LEVEL = ('embedding', 'final_logits')

SAMPLE_WIDTH = 8


def inspector_for(capture, generated: bool, name: str, tensor) -> TensorInspector:
    """A full inspector over a stored array, with token-aware headers.

    Lifted out of the old `JourneyView.inspector` so the variable panel and `RawView` share
    one wiring instead of two that can disagree about which axis is the key axis.
    """
    rows = token_labels(capture, generated, keys=name in ('k_attended', 'v_attended'))
    columns = (token_labels(capture, generated, keys=True)
               if name in ('attention_scores', 'attention_weights') else None)
    return TensorInspector(tensor, name=name, axes=axes_for(name, tensor),
                           row_labels=None if name == 'final_logits' else rows,
                           column_labels=columns)


class Resolved:
    """One variable, as it stands for the reader's current selection."""

    __slots__ = ('key', 'symbol', 'title', 'role', 'provenance', 'tier', 'array',
                 'shape_display', 'shape_caption', 'axes', 'note', 'colour',
                 'captured_instead', 'explanation', 'source_note')

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields.get(name))

    @property
    def captured(self) -> bool:
        return self.provenance == PROVENANCE_CAPTURED and self.array is not None

    def __repr__(self) -> str:                      # pragma: no cover - debugging aid
        return f'<Resolved {self.key} {self.provenance} {self.shape_display}>'


class VariableRegistry(QObject):
    """Resolves a variable key against one capture and the reader's current selection.

    The selection lives here rather than on a page because every chip, every diagram and
    the panel must agree about which layer and which position they are describing.  Setting
    it emits `context_changed`, which chips use to refresh their shape text; `select` emits
    `selected`, which the panel listens to.
    """

    selected = pyqtSignal(str)
    context_changed = pyqtSignal()

    def __init__(self, capture, facts: dict, parent=None):
        super().__init__(parent)
        self.capture = capture
        self.facts = dict(facts)
        self.layer_indices = sorted(capture.layers)
        self.generated = False
        self.layer_index = self.layer_indices[0] if self.layer_indices else 0
        self.head = 0
        self.position = max(len(capture.prompt_tokens) - 1, 0)
        self._current = ''

    # ── Selection context ────────────────────────────────────────────────────

    def set_context(self, *, generated=None, layer_index=None, head=None,
                    position=None) -> None:
        changed = False
        if generated is not None and bool(generated) != self.generated:
            self.generated, changed = bool(generated), True
        if layer_index is not None and int(layer_index) != self.layer_index:
            self.layer_index, changed = int(layer_index), True
        if head is not None and int(head) != self.head:
            self.head, changed = int(head), True
        if position is not None and int(position) != self.position:
            self.position, changed = int(position), True
        if changed:
            self.context_changed.emit()

    @property
    def phase_name(self) -> str:
        return 'first_generated_token' if self.generated else 'prefill'

    def layers(self) -> dict:
        return self.capture.generated_layers if self.generated else self.capture.layers

    def keys(self) -> list[str]:
        """Every key this registry answers for, captured first."""
        return list(TENSOR_SYMBOLS) + list(CONCEPTUAL_VARIABLES)

    def known(self, key: str) -> bool:
        return key in TENSOR_SYMBOLS or key in CONCEPTUAL_VARIABLES

    # ── Captured arrays ──────────────────────────────────────────────────────

    def _captured_array(self, key: str):
        """The stored array for a captured key, or `(None, reason)` when it is absent.

        `layer_input` is the one indirection: a layer's input X is captured, but under
        another name -- the embedding output at layer 0, and the previous layer's own output
        at every later layer.  Returning that array is reading a capture, not reconstructing
        one, and the note says which array it is so the reader is never misled about where
        the numbers came from.
        """
        capture = self.capture
        if key == 'embedding':
            array = capture.generated_embedding if self.generated else capture.embedding
            return array, ''
        if key == 'final_logits':
            # Prefill only, and absent entirely from pre-schema-3 runs.
            if self.generated:
                return None, ('The vocabulary scores were captured for the prompt pass '
                              'only; this phase has none.')
            if capture.logits is None:
                return None, 'This run was saved before the score vector was retained.'
            return capture.logits, ''
        if key == 'layer_input':
            if self.layer_index == self.layer_indices[0]:
                array = capture.generated_embedding if self.generated else capture.embedding
                return array, ('This is the captured embedding output: at the first layer, '
                               'the layer input and the embedding rows are the same tensor.')
            previous = self.layer_indices[self.layer_indices.index(self.layer_index) - 1]
            layer = self.layers().get(previous)
            array = None if layer is None else layer.tensors.get('layer_output')
            return array, (f'This is the captured output of layer {previous}, which is '
                           f'exactly what layer {self.layer_index} received.')
        layer = self.layers().get(self.layer_index)
        array = None if layer is None else layer.tensors.get(key)
        return array, ('' if array is not None else
                       'This layer does not carry that tensor in this capture.')

    # ── Resolution ───────────────────────────────────────────────────────────

    def resolve(self, key: str) -> Resolved:
        if key in TENSOR_SYMBOLS:
            return self._resolve_captured(key)
        if key in CONCEPTUAL_VARIABLES:
            return self._resolve_conceptual(key)
        raise KeyError(key)

    def _resolve_captured(self, key: str) -> Resolved:
        array, reason = self._captured_array(key)
        title = TENSOR_LABEL_BY_KEY.get(key) or EXTRA_TENSOR_LABELS.get(key, key)
        explanation = (TENSOR_EXPLANATIONS.get(key)
                       or EXTRA_TENSOR_EXPLANATIONS.get(key, ''))
        if array is None:
            # A captured key with nothing behind it in this phase or this saved run.  It
            # degrades to the same described-not-shown treatment as an intermediate: the
            # panel builds no sample and no inspector, because there is nothing to read.
            return Resolved(
                key=key, symbol=TENSOR_SYMBOLS[key], title=title,
                role=VARIABLE_ROLES.get(key, ''), provenance=PROVENANCE_ABSENT,
                tier=EVIDENCE_CONCEPTUAL, array=None, shape_display='not in this capture',
                shape_caption='', axes=[], note=reason, colour='layer',
                captured_instead=(), explanation=explanation, source_note='')
        written, caption = shape_line(array, omit_batch=True)
        return Resolved(
            key=key, symbol=TENSOR_SYMBOLS[key], title=title,
            role=VARIABLE_ROLES.get(key, ''), provenance=PROVENANCE_CAPTURED,
            tier=EVIDENCE_OBSERVED, array=array, shape_display=written,
            shape_caption=caption, axes=list(axes_for(key, array)), note='',
            colour=_colour_for(key), captured_instead=(), explanation=explanation,
            source_note=reason)

    def _resolve_conceptual(self, key: str) -> Resolved:
        spec = CONCEPTUAL_VARIABLES[key]
        # Shapes are declared as symbols formatted against facts read off captured arrays.
        # A fact that is 0 or unknown leaves the placeholder's own wording in place rather
        # than printing a confident zero.
        try:
            written = spec.shape_text.format(**self.facts) if spec.shape_text else ''
        except (KeyError, IndexError):               # a placeholder this run cannot fill
            written = spec.shape_text
        note = (LEARNED_VALUES_NOT_SAVED if spec.provenance == PROVENANCE_LEARNED
                else INTERMEDIATE_NOT_KEPT)
        return Resolved(
            key=key, symbol=spec.symbol, title=spec.title, role=spec.role,
            provenance=spec.provenance, tier=PROVENANCE_TIERS[spec.provenance],
            array=None, shape_display=written or 'shape not recorded', shape_caption='',
            axes=[part.strip() for part in spec.axes_text.split('×') if part.strip()],
            note=note, colour=spec.colour, captured_instead=tuple(spec.captured_instead),
            explanation='', source_note='')

    # ── The reader's current variable ────────────────────────────────────────

    def select(self, key: str) -> None:
        if not self.known(key):
            raise KeyError(key)
        self._current = key
        self.selected.emit(key)

    def current_symbol(self) -> str:
        return self._current


def _colour_for(key: str) -> str:
    """The role colour for a captured tensor, by what it is rather than by its name."""
    if key in ('q', 'q_attended'):
        return 'q'
    if key in ('k', 'k_attended'):
        return 'k'
    if key in ('v', 'v_attended'):
        return 'v'
    if key == 'attention_scores' or key == 'final_logits':
        return 'scores'
    if key == 'attention_weights':
        return 'weights'
    if key == 'attention_output':
        return 'output'
    if key in ('normalized_input', 'embedding'):
        return 'norm'
    return 'layer'


# ── A small, labelled, honest sample ─────────────────────────────────────────

def sample_for(resolved: Resolved, *, position: int, head: int):
    """A handful of real values from a captured array, plus labels and a caption.

    Basic indexing only, so this borrows the stored buffer and can never write through it.
    The caption names the slice: a reader must never have to guess which row of a
    four-dimensional tensor eight numbers came from.  Returns `(values, labels, caption,
    total)`, or `None` when the array's shape is not one this knows how to slice -- in
    which case the panel shows no sample rather than an arbitrary corner.
    """
    array = resolved.array
    if array is None:
        return None
    view = np.asarray(array)
    key = resolved.key
    if view.ndim == 1:
        shown = view[:SAMPLE_WIDTH]
        unit = 'vocabulary ID' if key == 'final_logits' else 'index'
        return (shown, [f'{unit} {index}' for index in range(len(shown))],
                f'the first {len(shown)} of {view.shape[0]:,} entries, in ID order '
                f'(not a ranking)', int(view.shape[0]))
    if view.ndim == 3:
        row = min(max(0, position), view.shape[-2] - 1)
        shown = view[0, row, :SAMPLE_WIDTH]
        return (shown, [f'feature {index}' for index in range(len(shown))],
                f'row for token position {row}', int(view.shape[-1]))
    if view.ndim == 4:
        head_index = min(max(0, head), view.shape[1] - 1)
        row = min(max(0, position), view.shape[-2] - 1)
        shown = view[0, head_index, row, :SAMPLE_WIDTH]
        if key in ('attention_scores', 'attention_weights'):
            return (shown, [f'key {index}' for index in range(len(shown))],
                    f'head {head_index}, query position {row}', int(view.shape[-1]))
        return (shown, [f'feature {index}' for index in range(len(shown))],
                f'head {head_index}, position {row}', int(view.shape[-1]))
    return None


# ── Widgets ──────────────────────────────────────────────────────────────────

class VariableChip(QPushButton):
    """One row of an operation's variable list, and the thing the reader clicks.

    Reads `symbol`, the shape for the current selection, and the provenance as a **word**.
    The colour agrees with the diagrams, but it is never the only carrier of the meaning:
    a reader who cannot distinguish the hues still sees "Captured" or "Learned".
    """

    PROVENANCE_WORDS = {
        PROVENANCE_CAPTURED: 'Captured',
        PROVENANCE_LEARNED: 'Learned parameter · values not saved',
        PROVENANCE_ABSENT: 'Computed by the model · not kept',
        PROVENANCE_DISPLAY: 'Calculated for this display',
    }

    def __init__(self, registry: VariableRegistry, key: str, *, compact: bool = False,
                 parent=None):
        super().__init__(parent)
        self._registry = registry
        self._key = key
        # A chip inside an equation shows its symbol alone, so the formula stays readable;
        # the variable list beneath the equation carries the shape and the provenance word.
        self._compact = compact
        self.setObjectName('variableChip')
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.clicked.connect(lambda: registry.select(key))
        registry.context_changed.connect(self.refresh)
        self.refresh()

    def key(self) -> str:
        return self._key

    def refresh(self) -> None:
        resolved = self._registry.resolve(self._key)
        word = self.PROVENANCE_WORDS.get(resolved.provenance, '')
        self.setText(resolved.symbol if self._compact else
                     f'▸  {resolved.symbol}      {resolved.shape_display}      {word}')
        self.setAccessibleName(f'{resolved.symbol}, {resolved.title}, {word}')
        self.setToolTip(f'{resolved.title}\n{resolved.role}\n{word}\n'
                        f'{resolved.shape_display}')
        colour = resolved.colour
        self.setProperty('role', colour)
        self.style().unpolish(self)
        self.style().polish(self)


class EquationRow(QWidget):
    """An equation whose terms are the same chips the variable list uses.

    `terms` interleaves keys and plain strings: a key becomes a clickable chip, anything
    else is drawn as an operator.  The whole equation also goes on the row as an accessible
    description, so a screen reader gets the formula as one phrase rather than as a run of
    buttons.
    """

    def __init__(self, registry: VariableRegistry, terms, *, parent=None):
        super().__init__(parent)
        self.setObjectName('equationRow')
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(6)
        spoken = []
        self.chips: list[VariableChip] = []
        for term in terms:
            if registry.known(term):
                chip = VariableChip(registry, term, compact=True)
                chip.setObjectName('equationTerm')
                row.addWidget(chip)
                self.chips.append(chip)
                spoken.append(registry.resolve(term).symbol)
            else:
                operator = label(str(term))
                operator.setObjectName('equationOperator')
                row.addWidget(operator)
                spoken.append(str(term))
        row.addStretch(1)
        self.setAccessibleDescription(' '.join(spoken))


class VariablePanel(QWidget):
    """The one shared inspection panel.  Rebuilt on selection; the array rung stays lazy.

    The ordering is the disclosure ladder the project already uses: what it is, where it
    came from, its shape, its axes, a small labelled sample, and only then the complete
    array.  For anything not captured the last two rungs are **not constructed at all** --
    there is no array, so there is no widget that could imply one.
    """

    def __init__(self, registry: VariableRegistry, *, tokens: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName('variablePanel')
        self._registry = registry
        self._tokens = tokens or {}
        self._body = QVBoxLayout(self)
        self._body.setContentsMargins(14, 12, 14, 12)
        self._body.setSpacing(8)
        self._slot = QVBoxLayout()
        self._slot.setContentsMargins(0, 0, 0, 0)
        self._slot.setSpacing(8)
        self._body.addWidget(label('VARIABLE', muted=True))
        self._body.addLayout(self._slot)
        self._body.addStretch(1)
        registry.selected.connect(self.show_variable)
        registry.context_changed.connect(self._refresh)
        self._current = ''
        self._show_placeholder()

    def current_symbol(self) -> str:
        return self._current

    # -- rendering ------------------------------------------------------------

    def _clear(self) -> None:
        while self._slot.count():
            item = self._slot.takeAt(0)
            child = item.widget()
            if child is not None:
                child.setParent(None)
                child.deleteLater()

    def _add(self, widget: QWidget) -> QWidget:
        self._slot.addWidget(widget)
        return widget

    def _show_placeholder(self) -> None:
        self._clear()
        self._add(label('Click any variable — in an equation or in a list — to see '
                        'what it is, what shape it has in this run, where its numbers came '
                        'from, and the complete stored array.', muted=True))

    def _refresh(self) -> None:
        if self._current:
            self.show_variable(self._current)

    def show_variable(self, key: str) -> None:
        resolved = self._registry.resolve(key)
        self._current = key
        self._clear()
        self._add(label(f'{resolved.symbol}   ·   {resolved.title}', title=True))
        self._add(badge_row(EvidenceBadge(
            resolved.tier, VariableChip.PROVENANCE_WORDS.get(resolved.provenance, ''))))
        if resolved.role:
            self._add(label(resolved.role))
        self._add(label('SHAPE IN THIS RUN', muted=True))
        self._add(label(resolved.shape_display))
        if resolved.shape_caption:
            self._add(label(resolved.shape_caption, muted=True, small=True))
        if resolved.axes:
            self._add(label('AXES', muted=True))
            self._add(label('\n'.join(
                f'·  {name}' for name in resolved.axes)))
        if resolved.captured:
            self._show_captured(resolved)
        else:
            self._show_described(resolved)

    def _show_captured(self, resolved: Resolved) -> None:
        if resolved.source_note:
            self._add(label(resolved.source_note, muted=True, small=True))
        sample = sample_for(resolved, position=self._registry.position,
                            head=self._registry.head)
        if sample is not None:
            values, names, caption, total = sample
            self._add(AnnotatedValueRow(
                list(values), labels=names, caption=caption.upper(),
                kind=EVIDENCE_OBSERVED, note=resolved.key, total=total,
                formatter=exact_scalar))
        # `axes_for` and `token_labels` already branch on the key and the array's rank, so
        # the key goes straight through rather than being second-guessed here.
        self._add(Disclosure(
            'Show the complete exact array',
            lambda key=resolved.key, array=resolved.array: inspector_for(
                self._registry.capture, self._registry.generated, key, array)))
        if resolved.explanation:
            self._add(Disclosure('What this tensor is, in full',
                                 lambda text=resolved.explanation: label(text)))

    def _show_described(self, resolved: Resolved) -> None:
        """No array means no array widget: the panel says what it is and stops."""
        self._add(label(resolved.note))
        if resolved.captured_instead:
            self._add(label('WHAT THIS RUN DID CAPTURE INSTEAD', muted=True))
            holder = QWidget()
            row = QVBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for key in resolved.captured_instead:
                if self._registry.known(key):
                    row.addWidget(VariableChip(self._registry, key))
            self._add(holder)
