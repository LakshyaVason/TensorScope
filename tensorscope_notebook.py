"""Page 2: Follow this run.

One scrolling worked calculation over one capture.  Thirteen anchors -- twelve operations
and then the continuation -- all present on the page at once, so a reader can scroll back
to a variable they met four steps ago instead of navigating away from it.  The shell's
footer button is the stable "Next operation" control; it never scrolls away, because it
belongs to the shell and not to the page.

Three pieces of state, all shared and all in one place:

* `VariableRegistry` holds the phase, layer, head and selected position, and resolves any
  variable against them.  Changing a selection emits one signal, every chip refreshes its
  shape, and every section's value slot rebuilds itself.  Nothing else moves.
* `VariablePanel` is the single docked answer to "what is this?", so clicking the same
  symbol in two different operations lands in the same place.
* The reveal, check and disclosure state dicts, which exist because a section is built
  once and never replaced -- so in practice a reader's expansions survive navigation by
  simply still being there, and these dicts only matter for the parts that do rebuild.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QPushButton, QSizePolicy, QWidget,
)

from tensorscope_common import (
    AnchoredScrollPage, PhaseBadge, RoleLegend, Section, chip_text, label,
)
from tensorscope_content import (
    CONTINUATION, NOTEBOOK_OPERATIONS, OPERATION_INDEX, learn_facts,
)
from tensorscope_operations import BUILDERS, NotebookContext
from tensorscope_variables import VariablePanel, VariableRegistry


# The shell numbers its own stage buttons, so these labels carry no number of their own.
NOTEBOOK_NAV = ([(operation.key, operation.nav) for operation in NOTEBOOK_OPERATIONS]
                + [(CONTINUATION.key, CONTINUATION.nav)])

SECTIONS = NOTEBOOK_OPERATIONS + [CONTINUATION]


class NotebookView(AnchoredScrollPage):
    """The worked notebook: the context ribbon, the page, and the variable panel."""

    NAV = NOTEBOOK_NAV
    NOUN = 'Operation'
    NEXT_LABEL = 'Next operation →'
    PREVIOUS_LABEL = '← Previous operation'

    # Emitted when the reader asks for every captured array at once.  The shell opens the
    # dialog, because a modeless dialog needs an owner that outlives this page.
    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(capture, parent=parent, tokens=tokens)
        self.facts = learn_facts(capture)
        self.registry = VariableRegistry(capture, self.facts, parent=self)
        self.state: dict = {'reveals': {}, 'checks': {}, 'open': {}}
        self.context_builder = NotebookContext(
            capture, self.facts, self.registry, tokens=tokens, state=self.state)

        self.add_header(self._ribbon())
        # A second pinned strip, because the hues are stable for the whole page and a
        # legend that scrolled away would only be readable next to one operation.
        self.legend = self.add_header(RoleLegend(tokens=tokens))
        for operation in SECTIONS:
            number = f'{operation.number:02}' if operation.number else '—'
            self.add_section(Section(
                operation.key, operation.heading.format(**self.facts),
                operation.lede.format(**self.facts), number=number,
                build_body=lambda section, op=operation: self._body(op, section)))
        self.finish_sections()
        self.panel = self.add_side_panel(
            VariablePanel(self.registry, tokens=tokens))
        # The breadcrumb's detail follows the selection, not just the section.
        self.registry.context_changed.connect(
            lambda: self.context_changed.emit(self.context()))
        self.go_to(SECTIONS[0].key)

    # ── Section bodies ───────────────────────────────────────────────────────

    def _body(self, operation, section) -> QWidget:
        builder = BUILDERS.get(operation.key)
        host = builder(self.context_builder, section) if builder is not None else None
        if operation.topic and host is not None:
            # Into the body's own layout, so the link sits under the operation it belongs
            # to rather than above it.
            host.layout().addWidget(self._explain_button(operation), 0, Qt.AlignLeft)
        return host

    def _explain_button(self, operation) -> QWidget:
        button = QPushButton('Explain this operation  →')
        button.setObjectName('crossPageLink')
        button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        button.setToolTip('Open the illustrated reference at the topic behind this step.')
        button.clicked.connect(
            lambda _, key=operation.topic: self.jump_requested.emit('understand', key))
        return button

    # ── The context ribbon ───────────────────────────────────────────────────

    def _ribbon(self) -> QWidget:
        capture = self.capture
        host = QWidget()
        host.setObjectName('contextRibbon')
        row = QHBoxLayout(host)
        row.setContentsMargins(22, 8, 22, 8)
        row.setSpacing(10)

        prompt = label(f'“{chip_text(capture.prompt, 48)}”', small=True)
        prompt.setToolTip(capture.prompt)
        prompt.setWordWrap(False)
        row.addWidget(prompt)

        self.phase_picker = QComboBox()
        self.phase_picker.setAccessibleName('Captured phase')
        self.phase_picker.addItems(['Phase 1 · Prompt prefill',
                                    'Phase 2 · First generated-token pass'])
        self.phase_picker.currentIndexChanged.connect(
            lambda index: self.registry.set_context(generated=bool(index)))
        row.addWidget(self.phase_picker)
        self.phase_badge = PhaseBadge(False)
        row.addWidget(self.phase_badge)

        self.layer_back = QPushButton('◀')
        self.layer_back.setAccessibleName('Previous layer')
        self.layer_picker = QComboBox()
        self.layer_picker.setAccessibleName('Layer')
        for index in self.registry.layer_indices:
            self.layer_picker.addItem(f'Layer {index}', index)
        self.layer_forward = QPushButton('▶')
        self.layer_forward.setAccessibleName('Next layer')
        self.layer_back.clicked.connect(lambda: self._step_layer(-1))
        self.layer_forward.clicked.connect(lambda: self._step_layer(1))
        self.layer_picker.currentIndexChanged.connect(
            lambda position: self.registry.set_context(
                layer_index=self.layer_picker.itemData(position)))
        for widget in (self.layer_back, self.layer_picker, self.layer_forward):
            row.addWidget(widget)

        self.head_picker = QComboBox()
        self.head_picker.setAccessibleName('Attention head')
        for index in range(max(1, int(self.facts['query_head_count']))):
            self.head_picker.addItem(f'Head {index}', index)
        self.head_picker.currentIndexChanged.connect(
            lambda position: self.registry.set_context(
                head=self.head_picker.itemData(position)))
        row.addWidget(self.head_picker)

        self.position_label = label('', small=True)
        self.position_label.setWordWrap(False)
        row.addWidget(self.position_label)
        row.addStretch(1)

        self.raw_button = QPushButton('All captured arrays…')
        self.raw_button.setObjectName('crossPageLink')
        self.raw_button.setToolTip('Every stored array for every phase and layer, with no '
                                  'framing at all.')
        self.raw_button.clicked.connect(lambda: self.jump_requested.emit('raw', ''))
        row.addWidget(self.raw_button)

        self.registry.context_changed.connect(self._sync_ribbon)
        self._sync_ribbon()
        return host

    def _step_layer(self, amount: int) -> None:
        indices = self.registry.layer_indices
        if not indices:
            return
        position = indices.index(self.registry.layer_index) + amount
        if 0 <= position < len(indices):
            self.registry.set_context(layer_index=indices[position])

    def _sync_ribbon(self) -> None:
        """Mirror the registry into the ribbon, without the controls echoing back.

        The pickers are wired to the registry, so moving one programmatically would
        re-enter `set_context` and, on a chain of them, emit twice.
        """
        registry = self.registry
        for widget, value in ((self.phase_picker, int(registry.generated)),
                              (self.layer_picker,
                               registry.layer_indices.index(registry.layer_index)
                               if registry.layer_index in registry.layer_indices else 0),
                              (self.head_picker, min(registry.head,
                                                     self.head_picker.count() - 1))):
            if widget.currentIndex() == value:
                continue
            blocked = widget.signalsBlocked()
            widget.blockSignals(True)
            try:
                widget.setCurrentIndex(value)
            finally:
                widget.blockSignals(blocked)
        self.phase_badge.update_phase(registry.generated)
        indices = registry.layer_indices
        self.layer_back.setEnabled(bool(indices)
                                   and registry.layer_index != indices[0])
        self.layer_forward.setEnabled(bool(indices)
                                      and registry.layer_index != indices[-1])
        spelling = (self.capture.prompt_tokens[registry.position]
                    if 0 <= registry.position < len(self.capture.prompt_tokens) else '')
        self.position_label.setText(
            f'Position {registry.position}  {chip_text(spelling, 12)}' if spelling
            else f'Position {registry.position}')

    # ── The shell's breadcrumb ───────────────────────────────────────────────

    def context(self) -> str:
        registry = self.registry
        phase = 'Phase 2' if registry.generated else 'Phase 1'
        operation = OPERATION_INDEX.get(self.current_stage())
        if operation is None:                   # the continuation has no layer or head
            return f'{phase}  ·  token {registry.position}'
        return (f'{phase}  ·  Layer {registry.layer_index}  ·  '
                f'head {registry.head}  ·  position {registry.position}')

    # ── A small surface for the shell and the tests ──────────────────────────

    def select(self, **fields) -> None:
        """Move the reader's selection: phase, layer, head or token position."""
        self.registry.set_context(**fields)

    def select_position(self, index: int) -> None:
        self.registry.set_context(position=index)

    def selection(self) -> dict:
        registry = self.registry
        return {'generated': registry.generated, 'layer_index': registry.layer_index,
                'head': registry.head, 'position': registry.position}

    def head_count(self) -> int:
        return self.head_picker.count()
