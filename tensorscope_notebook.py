"""Page 2: Follow this run.

One scrolling worked calculation over one capture.  Thirteen anchors -- twelve operations
and then the continuation -- grouped into four chapters, all present on the page at once,
so a reader can scroll back to a variable they met four steps ago instead of navigating
away from it.

The page opens on its destination: the prompt, the start of the actual answer, and the
token whose calculation it is about to follow.  From there it follows one prompt position
through every operation, and hands over explicitly to the final position at the decision.

Three pieces of state, all shared and all in one place:

* `VariableRegistry` holds the phase, layer, head and followed position, and resolves any
  variable against them.  Changing a selection emits one signal, every chip refreshes its
  shape, and every section's value slot rebuilds itself.  Nothing else moves.
* `VariablePanel` is the single answer to "what is this?", so clicking the same symbol in
  two different operations lands in the same place.  It is closed until a variable is
  chosen, and closable again.
* The reveal, check and disclosure state dicts, which exist because a section is built
  once and never replaced -- so in practice a reader's expansions survive navigation by
  simply still being there, and these dicts only matter for the parts that do rebuild.

What this page offers the shell beyond the base contract: `CHAPTERS` and `chapter_of` for
the progress indicator, `progress_text` for its caption, and `toolbar_widget()` -- the
Following control -- for the reading bar.  The shell imports none of this module's content;
it only calls those.
"""
from __future__ import annotations

import html

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QFrame, QGridLayout, QMenu, QPushButton, QSizePolicy, QToolButton, QVBoxLayout,
    QWidget,
)

import numpy as np

from tensorscope_common import (
    CHROME, AnchoredScrollPage, RoleLegend, Section, approx, badge_row, badged, chip_text,
    described_line, display_spelling, evidence_frame, label, measure_label, observed_line,
    replace_in, set_evidence,
)
from tensorscope_content import (
    CHAPTER_OF, CHAPTERS, CONTINUATION, EVIDENCE_CONCEPTUAL, EVIDENCE_DERIVED,
    EVIDENCE_OBSERVED, LOGITS_UNAVAILABLE, NOTEBOOK_OPERATIONS, NOTEBOOK_TEXT, OPERATION_INDEX,
    OVERVIEW_TIERS, TEMPLATE_BAND, TYPED_BAND, WEIGHTS_NOT_PROBABILITIES, learn_facts,
    prompt_token_bands, readable_spelling,
)
from tensorscope_operations import (
    BUILDERS, NotebookContext, build_overview, depth_rows, sink_callout,
)
from tensorscope_diagrams import LayerDepthView, PipelineStrip, ScoreDotPlot
from tensorscope_variables import VariablePanel, VariableRegistry


# The shell numbers its own contents entries, so these labels carry no number of their own.
NOTEBOOK_NAV = ([(operation.key, operation.nav) for operation in NOTEBOOK_OPERATIONS]
                + [(CONTINUATION.key, CONTINUATION.nav)])

SECTIONS = NOTEBOOK_OPERATIONS + [CONTINUATION]


class FocusChip(QToolButton):
    """'Following ' color' · position 4', and a menu to follow another position.

    The followed position is the thread through the whole page, so the control for it
    lives in the reading bar where it never scrolls away.  It also offers the way back
    when the reader has wandered to another layer or to the generated-token pass.
    """

    def __init__(self, capture, registry: VariableRegistry, parent=None):
        super().__init__(parent)
        self._capture = capture
        self._registry = registry
        self.setObjectName('focusChip')
        self.setAccessibleName('Followed token position')
        self.setToolTip('The prompt position the operations follow. Choose another here or '
                        'by clicking a token in operation 1.')
        self.setPopupMode(QToolButton.InstantPopup)
        self.setFocusPolicy(Qt.StrongFocus)
        self._menu = QMenu(self)
        self._menu.aboutToShow.connect(self._populate)
        self.setMenu(self._menu)
        registry.context_changed.connect(self.sync)
        self.sync()

    def _spelling(self, index: int) -> str:
        tokens = self._capture.prompt_tokens
        return chip_text(readable_spelling(tokens[index]), 14) if 0 <= index < len(tokens) else ''

    def sync(self) -> None:
        registry = self._registry
        text = f'Following “{self._spelling(registry.position)}” at {registry.position}'
        if registry.layer_indices and registry.layer_index != registry.layer_indices[0]:
            text += f' · layer {registry.layer_index}'
        if registry.generated:
            text += ' · generated-token pass'
        self.setText(text)

    def _populate(self) -> None:
        self._menu.clear()
        registry = self._registry
        self._menu.addSection('Follow a prompt position')
        for index, _token in enumerate(self._capture.prompt_tokens):
            action = QAction(f'{index:>3}   {self._spelling(index)}', self._menu)
            action.setCheckable(True)
            action.setChecked(index == registry.position)
            action.triggered.connect(
                lambda _=False, i=index: registry.set_context(position=i))
            self._menu.addAction(action)
        extras = []
        if registry.layer_indices and registry.layer_index != registry.layer_indices[0]:
            extras.append((f'Back to layer {registry.layer_indices[0]}',
                           lambda: registry.set_context(layer_index=registry.layer_indices[0])))
        if registry.generated:
            extras.append(('Back to the prompt pass',
                           lambda: registry.set_context(generated=False)))
        if extras:
            self._menu.addSeparator()
            for text, slot in extras:
                action = QAction(text, self._menu)
                action.triggered.connect(slot)
                self._menu.addAction(action)


class NotebookView(AnchoredScrollPage):
    """The worked notebook: the page, and the variable panel beside it."""

    NAV = NOTEBOOK_NAV
    NOUN = 'Operation'
    NEXT_LABEL = 'Next →'
    PREVIOUS_LABEL = '← Previous'
    CHAPTERS = [(chapter.title, chapter.sections) for chapter in CHAPTERS]

    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(capture, parent=parent, tokens=tokens)
        self.facts = learn_facts(capture)
        self.registry = VariableRegistry(capture, self.facts, parent=self)
        self.state: dict = {'reveals': {}, 'checks': {}, 'open': {}}
        self.context_builder = NotebookContext(
            capture, self.facts, self.registry, tokens=tokens, state=self.state)

        starts = {chapter.sections[0]: chapter.title for chapter in CHAPTERS}
        for operation in SECTIONS:
            number = f'{operation.number:02}' if operation.number else '—'
            self.add_section(Section(
                operation.key, operation.heading.format(**self.facts),
                operation.lede.format(**self.facts), number=number,
                kicker=starts.get(operation.key, ''),
                preface=self._destination() if operation.key == SECTIONS[0].key else None,
                build_body=lambda section, op=operation: self._body(op, section)))
        self.finish_sections()
        # The colour key lives in the panel: it is where a reader asking "what does this
        # colour mean?" is already looking, and it no longer pins a band above the page.
        self.legend = RoleLegend(tokens=tokens)
        self.panel = self.add_side_panel(
            VariablePanel(self.registry, tokens=tokens, legend=self.legend), hidden=True)
        self.registry.selected.connect(lambda _key: self.show_side_panel())
        self.panel.close_requested.connect(self.hide_side_panel)
        self.panel.raw_requested.connect(lambda: self.jump_requested.emit('raw', ''))
        self._focus_chip = FocusChip(capture, self.registry)
        # The breadcrumb's detail follows the selection, not just the section.
        self.registry.context_changed.connect(
            lambda: self.context_changed.emit(self.context()))
        self.go_to(SECTIONS[0].key)

    # ── The destination ──────────────────────────────────────────────────────

    def _destination(self) -> QWidget:
        """Where the page is going, before any of the steps that get there.

        YOU ASKED / THE MODEL ANSWERED, then the whole run at a glance: the outcome the run
        chose (Level 0) and the map of how it got there (Level 1), built by
        `build_overview`.  The prompt is a row of chips, not a label -- the tokens you did
        not type are faded, and the deciding position is marked -- so the reader sees at a
        glance what reached the model before watching one row change through the steps.
        """
        facts = self.facts
        capture = self.capture
        final = int(facts['final_position'])
        bands = self.context_builder.bands
        typed = {name for _start, _end, name in (bands or [])}

        card = QFrame()
        card.setObjectName('destinationCard')
        body = QVBoxLayout(card)
        body.setContentsMargins(24, 18, 24, 20)
        body.setSpacing(4)
        body.addWidget(label(NOTEBOOK_TEXT['destination_asked'], muted=True))

        # The prompt as chips: the tokens you typed stand out, the template markers are
        # faded, and the deciding position -- where the run's answer was chosen -- is
        # marked.  Non-checkable: the Following control in the reading bar owns the
        # selection, and these chips are only ever read.  They are `prefaceChip`, not
        # `tokenChip`, so the op-1 chip-count test does not count them.
        columns = 10
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        for index, token in enumerate(capture.prompt_tokens):
            chip = QPushButton(chip_text(display_spelling(token)))
            chip.setObjectName('prefaceChip')
            chip.setCheckable(False)
            chip.setProperty('template', index not in typed)
            chip.setProperty('deciding', index == final)
            chip.setToolTip(f'position {index}')
            grid.addWidget(chip, index // columns, index % columns)
        grid.setColumnStretch(columns, 1)
        prompt = QWidget()
        prompt.setLayout(grid)
        body.addWidget(prompt)
        body.addSpacing(4)

        body.addWidget(label(NOTEBOOK_TEXT['destination_answered'], muted=True))
        excerpt = str(facts['response_excerpt'])
        first = capture.tokens[0] if capture.tokens else ''
        # The first token, marked where it begins the excerpt: that is the token whose
        # calculation the page follows.  Only marked when it really is the excerpt's start.
        if first and excerpt.startswith(first):
            # Qt's rich text has no border-bottom, so the old rule drew nothing but bold;
            # colour and an underline are what it can render.
            shown = (f'<span style="color: {CHROME["accent"]}; text-decoration: underline; '
                     f'font-weight: 700;">{html.escape(first, quote=False)}</span>'
                     + html.escape(excerpt[len(first):], quote=False))
        else:
            shown = html.escape(excerpt, quote=False)
        answer = label(shown, rich=True)
        answer.setObjectName('destinationAnswer')
        body.addWidget(answer)
        body.addSpacing(6)

        # Level 0 and Level 1: the outcome and the map, before the steps that reach them.
        body.addWidget(build_overview(self.context_builder, self.go_to))

        body.addSpacing(6)
        key = 'destination_follow' if facts['prompt_span_known'] else 'destination_unaligned'
        body.addWidget(label(NOTEBOOK_TEXT[key].format(**facts), small=True))
        return card

    # ── Section bodies ───────────────────────────────────────────────────────

    def _body(self, operation, section) -> QWidget:
        builder = BUILDERS.get(operation.key)
        host = builder(self.context_builder, section) if builder is not None else None
        if operation.topic and host is not None:
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

    # ── What the shell reads ─────────────────────────────────────────────────

    def chapter_of(self, key: str) -> str:
        chapter = CHAPTER_OF.get(key)
        return chapter.title if chapter is not None else ''

    def progress_text(self, key: str) -> str:
        operation = OPERATION_INDEX.get(key)
        if operation is None:
            return 'After the first token'
        return f'Operation {operation.number} of {len(NOTEBOOK_OPERATIONS)}'

    def toolbar_widget(self) -> QWidget:
        return self._focus_chip

    def pickers(self, kind: str) -> list:
        """The contextual controls of one kind that the built sections contain."""
        return [widget for widget in self.context_builder.pickers.get(kind, [])
                if _alive(widget)]

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
        return max(1, int(self.facts['query_head_count']))


def _alive(widget) -> bool:
    try:
        widget.objectName()
        return True
    except RuntimeError:
        return False
