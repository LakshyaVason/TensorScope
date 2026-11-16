"""The shell that hosts the reader-facing views of one capture.

`ComputationRecap` owns the window, the palette, the banner, the provenance evidence and
the navigation between two pages:

* **Follow this run** (`NotebookView`) — the default, and listed first.  One scrolling
  worked calculation over this capture, from the prompt to the selected token and the
  continuation, in four chapters.
* **Understand the calculation** (`ReferenceView`) — twelve illustrated topics: purpose,
  formula, symbols, picture, and an optional deeper explanation.

Two bands of chrome, and the rest of the window is the page:

* the **banner** — which run, the mode toggle, "validated" with its evidence a click
  away, and every captured array a click away;
* the **reading bar** — the prompt, where the reader is (chapter and operation), the
  page's own toolbar control (the notebook's Following control), a Contents menu, and
  Previous / Next.

That replaced a pipeline of thirteen chips, a sidebar list of the same thirteen, a
breadcrumb, progress pills, a footer and a pinned ribbon of pickers -- six bands and five
ways of navigating one list.  The pickers now appear inside the operation that explains
them.

The shell knows nothing about tensors.  Each page exposes a small contract — a `NAV` list
of `(section key, label)`, `go_to(key)`, `current_stage()`, `stage_keys()`, the
`stage_changed` / `context_changed` signals, and optionally `CHAPTERS`, `progress_text(key)`
and `toolbar_widget()` — and the shell drives its controls from those alone.  It imports
`tensorscope_common` and `tensorscope_views` and nothing else of ours.
"""
from __future__ import annotations

import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics, QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QShortcut,
    QSizePolicy, QVBoxLayout, QWidget,
)

from tensorscope_common import (
    COLORS, TEMPLATE_CHIP_INK, Card, ChapterProgress, ContentsButton, Disclosure,
    StageStack, chip_text, evidence_edge_rules, label, role_colors,
)
from tensorscope_views import NotebookView, RawTensorDialog, ReferenceView


# (key, mode label, one-line purpose, view class).  The worked run first, because it is
# the page a reader arriving at a finished capture wants and the one that opens; the
# reference second, for when a step stops making sense.  Both use the same order and the
# same notation.
#
# Every captured array is still reachable, through the banner's "All captured arrays..."
# button, which opens `RawTensorDialog`.  It is deliberately not a third row here: a mode
# in this list competes with the pages for the reader's attention.
MODES = [
    ('run', 'Follow this run',
     'One worked calculation, prompt to selected token', NotebookView),
    ('understand', 'Understand the calculation',
     'Purpose, formula and a picture for each operation', ReferenceView),
]

DEFAULT_MODE = 'run'

MODE_INDEX = {key: (name, purpose, view) for key, name, purpose, view in MODES}

DEFAULT_COLORS = {
    'card_bg': '#1c1c20', 'border': '#3b3b44',
    'text_secondary': '#b5b5c1', 'text_primary': '#f0f0f2',
    'accent': '#60a5fa', 'bg': '#111114',
    'text_muted': '#636370', 'nav_active_bg': '#202028',
}


def _role_chip_rules(tokens: dict | None = None) -> str:
    """One stylesheet rule per semantic role, read from the diagrams' own palette.

    `role_colors` (over `COLORS`) is the single source of these hues, so a chip and the
    picture beside it can never disagree about what colour Q is -- in either palette.
    """
    return '\n'.join(
        f'QPushButton#variableChip[role="{role}"] {{ border-left-color: {hue}; }}'
        for role, hue in role_colors(tokens).items())


class ComputationRecap(QDialog):
    """Read-only window over one saved or just-captured run."""

    def __init__(self, capture, run_id=None, parent=None, tokens=None):
        super().__init__(parent)
        capture.validate()
        self.capture = capture
        self._tokens = tokens
        self._views: dict[str, QWidget] = {}
        self._hosts: dict[str, QWidget] = {}
        # Modeless, so a reader can hold an array beside the operation that produced it.
        # Held on the shell because it has to outlive the page that asked for it.
        self._raw_dialog = None
        self._mode = None
        self._context = ''
        self._details_dialog = None
        self._toolbars: dict[str, QWidget] = {}
        self.setWindowTitle('TensorScope — Computation Recap')
        # Designed at laptop size: the page gets the window, the chrome gets two rows.
        self.resize(1366, 768)
        self.setMinimumSize(1024, 640)

        colors = dict(DEFAULT_COLORS)
        colors.update(tokens or {})
        self._colors = colors
        self.setStyleSheet(f'''
            /* Cards */
            #journeyCard {{
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-radius: 10px;
            }}

            /* Typography */
            #journeyTitle {{ font-size: 20px; font-weight: 700; }}
            #journeyText  {{ font-size: 14px; }}
            #journeyMuted {{ color: {colors['text_secondary']}; font-size: 12px; letter-spacing: 0.04em; }}
            #journeySmall {{ color: {colors['text_secondary']}; font-size: 12px; }}
            #journeyEquation {{
                font-size: 14px; padding: 12px 16px;
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-left: 3px solid {colors['accent']};
                border-radius: 8px;
            }}
            /* A formula set on its own lines, in a mathematical face where one exists. */
            #formulaBlock {{
                font-family: 'Cambria Math', 'Cambria', 'Latin Modern Math', 'STIX Two Math',
                             'Times New Roman', serif;
                font-size: 19px; padding: 14px 20px;
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-left: 3px solid {colors['accent']};
                border-radius: 8px;
            }}
            #partLabel {{
                color: {colors['text_muted']}; font-size: 11px; font-weight: 700;
                letter-spacing: 0.1em; padding-top: 6px;
            }}

            /* Section rule */
            QFrame#sectionRule {{
                background: {colors['border']};
                max-height: 1px; min-height: 1px;
                border: none; margin: 6px 0;
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
                padding: 7px 12px;
                background: transparent;
                color: {colors['text_secondary']};
                font-size: 13px;
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
            QWidget#recapBanner {{
                background: {colors['bg']};
                border-bottom: 1px solid {colors['border']};
            }}
            #recapBannerTitle {{ font-weight: 700; font-size: 14px; }}
            #recapBannerModel {{ color: {colors['text_secondary']}; font-size: 13px; }}
            #recapBannerSub {{ color: {colors['text_secondary']}; font-size: 11px; }}

            /* The reading bar: prompt, progress, toolbar, contents, previous/next */
            #readingBar {{
                background: {colors['card_bg']};
                border-bottom: 1px solid {colors['border']};
            }}
            #promptChip {{
                color: {colors['text_primary']}; font-size: 12px; font-weight: 600;
                padding: 3px 10px; border: 1px solid {colors['border']}; border-radius: 12px;
            }}
            QToolButton#contentsButton, QToolButton#focusChip {{
                border: 1px solid {colors['border']}; border-radius: 6px;
                padding: 5px 10px; background: transparent;
                color: {colors['text_primary']}; font-size: 12px;
            }}
            QToolButton#contentsButton:hover, QToolButton#focusChip:hover,
            QToolButton#contentsButton:focus, QToolButton#focusChip:focus {{
                border-color: {colors['accent']};
            }}
            /* The two pages as one segmented control. */
            QWidget#modeSwitch {{
                background: {colors['card_bg']};
                border: 1px solid {colors['border']}; border-radius: 8px;
            }}
            QPushButton#modeButton {{
                border: none; border-radius: 6px;
                padding: 5px 14px; background: transparent;
                color: {colors['text_secondary']}; font-size: 13px; text-align: center;
            }}
            QPushButton#modeButton:hover {{ color: {colors['text_primary']}; }}
            QPushButton#modeButton:focus {{ border: 1px solid {colors['accent']}; }}
            QPushButton#modeButton:checked {{
                background: {colors['accent']}; color: #ffffff; font-weight: 600;
            }}
            QPushButton#stepButton {{
                border: 1px solid {colors['border']}; border-radius: 6px;
                padding: 5px 12px; background: transparent;
                color: {colors['text_primary']}; font-size: 12px;
            }}
            QPushButton#stepButton:hover, QPushButton#stepButton:focus {{
                border-color: {colors['accent']};
            }}
            QPushButton#stepButton:disabled {{ color: {colors['text_muted']}; }}

            /* Teaching structure inside an operation */
            #sectionKicker {{
                color: {colors['accent']}; font-size: 12px; font-weight: 700;
                letter-spacing: 0.12em; padding-bottom: 2px;
            }}
            #sectionLede {{ font-size: 15px; color: {colors['text_secondary']}; }}
            QFrame#destinationCard {{
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-radius: 12px;
            }}
            #destinationPrompt {{ font-size: 22px; font-weight: 700; }}
            #destinationAnswer {{ font-size: 16px; }}
            QWidget#ioRow {{
                background: transparent;
                border: 1px solid {colors['border']};
                border-radius: 8px;
            }}
            #ioTag {{
                color: {colors['text_muted']};
                font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
            }}
            #ioAction {{ font-size: 13px; font-style: italic; color: {colors['text_secondary']}; }}
            #ioPlain {{ font-family: Consolas, 'Courier New', monospace; font-size: 13px; }}
            #templateTokensWhy {{ font-size: 13px; color: {colors['text_secondary']}; }}
            #sinkRule {{ font-size: 11px; color: {colors['text_secondary']}; padding-left: 4px; }}
            #teachingNote, #sinkNote {{
                padding: 10px 14px;
                border: 1px solid {colors['border']};
                border-left: 3px solid {colors['text_secondary']};
                background: {colors['card_bg']};
                border-radius: 6px; font-size: 13px; color: {colors['text_primary']};
            }}
            #takeawayLine {{
                padding: 12px 0 2px 0; font-size: 13px; font-weight: 600;
                color: {colors['text_primary']};
                border-top: 1px solid {colors['border']};
            }}
            #focusShift {{
                padding: 12px 16px; font-size: 14px;
                background: {colors['card_bg']};
                border: 1px solid {colors['accent']}; border-left: 4px solid {colors['accent']};
                border-radius: 8px;
            }}
            QPushButton#passButton {{
                text-align: left; border: 1px solid {colors['border']}; border-radius: 5px;
                padding: 4px 10px; background: transparent;
                color: {colors['text_primary']}; font-size: 12px;
            }}
            QPushButton#passButton:checked {{
                border-color: {colors['accent']}; background: {colors['nav_active_bg']};
            }}
            QToolButton#panelClose {{
                border: none; padding: 2px 6px; color: {colors['text_secondary']};
                font-size: 13px;
            }}
            QToolButton#panelClose:hover {{ color: {colors['text_primary']}; }}

            /* Token chips.  An ID selector, so it beats the generic checkable rule
               above rather than inheriting its left-aligned padding. */
            QPushButton#tokenChip {{
                text-align: center;
                border: 1px solid {colors['border']};
                border-radius: 5px;
                padding: 3px 7px;
                background: {colors['card_bg']};
                color: {colors['text_primary']};
                font-family: Consolas, 'Courier New', monospace;
                font-size: 12px;
            }}
            QPushButton#tokenChip[template="true"] {{
                color: {colors[TEMPLATE_CHIP_INK]};
                background: transparent;
                border-style: dashed;
            }}
            QPushButton#tokenChip:hover {{ border-color: {colors['accent']}; }}
            QPushButton#tokenChip:checked {{
                background: {colors['accent']};
                border-color: {colors['accent']};
                color: #ffffff;
                font-weight: 700;
            }}

            /* The prompt chips in the destination card.  Same shape as a token chip, but
               not checkable -- they are the page's starting point, not a selection the
               reader moves.  The deciding position is marked with a solid accent border. */
            QPushButton#prefaceChip {{
                text-align: center;
                border: 1px solid {colors['border']};
                border-radius: 5px;
                padding: 3px 7px;
                background: {colors['card_bg']};
                color: {colors['text_primary']};
                font-family: Consolas, 'Courier New', monospace;
                font-size: 12px;
            }}
            QPushButton#prefaceChip[template="true"] {{
                color: {colors[TEMPLATE_CHIP_INK]};
                background: transparent;
                border-style: dashed;
            }}
            QPushButton#prefaceChip[deciding="true"] {{
                border: 1px solid {colors['accent']};
                color: {colors['accent']};
                font-weight: 700;
            }}
            QPushButton#prefaceChip:hover {{ border-color: {colors['accent']}; }}

            /* Lesson: stepwise reveal, optional checks, value boxes, bands */
            QFrame#revealStep {{
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-radius: 8px;
            }}
            #revealStepCaption {{ font-size: 14px; font-weight: 700; }}
            QPushButton#revealNext {{
                background: {colors['accent']}; border: 1px solid {colors['accent']};
                color: #ffffff; font-weight: 600; padding: 6px 14px; border-radius: 6px;
            }}
            QPushButton#revealNext:disabled {{
                background: transparent; border-color: {colors['border']};
                color: {colors['text_muted']};
            }}
            QPushButton#revealControl {{
                border: 1px solid {colors['border']}; border-radius: 6px;
                padding: 6px 12px; background: transparent;
                color: {colors['text_secondary']};
            }}
            QPushButton#revealControl:hover {{ border-color: {colors['accent']}; }}
            QPushButton#revealControl:disabled {{ color: {colors['text_muted']}; }}
            QFrame#predictCheck {{
                background: {colors['bg']};
                border: 1px dashed {colors['border']};
                border-radius: 8px;
            }}
            QPushButton#predictOption, QPushButton#predictSkip {{
                text-align: left; border: 1px solid {colors['border']};
                border-radius: 6px; padding: 6px 12px; background: transparent;
                color: {colors['text_primary']};
            }}
            QPushButton#predictOption:hover, QPushButton#predictSkip:hover {{
                border-color: {colors['accent']};
            }}
            QPushButton#predictOption:checked {{ border-color: {colors['accent']}; }}
            #predictExplanation {{
                padding: 8px 10px; border-left: 3px solid {colors['accent']};
            }}
            QFrame#valueBox {{
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-radius: 6px;
            }}
            {evidence_edge_rules()}
            #valueBoxNumber {{
                font-family: Consolas, 'Courier New', monospace; font-size: 14px;
                font-weight: 600;
            }}
            #bandHeading {{ font-size: 11px; letter-spacing: 0.08em; padding-top: 6px; }}
            #responseText {{ font-size: 15px; }}

            /* Operators between shape/flow boxes */
            #shapeOp {{
                color: {colors['text_muted']};
                font-size: 16px;
                padding: 0 2px;
            }}

            /* ── The two scrolling pages ──────────────────────────────────── */

            /* One anchored section of a long page.  The reader's current section is
               marked with a dynamic property rather than an inline stylesheet, so both
               palettes are handled here and the widget code stays colour-free. */
            /* Sections sit flat on the page, like the pages of a textbook: a hairline
               between them, and an accent rule in the margin beside the one being read.
               Figures, formulas and values are the only raised surfaces, so the eye lands
               on them. */
            QFrame#notebookSection {{
                background: {colors['bg']};
                border: none;
                border-left: 3px solid transparent;
                border-bottom: 1px solid {colors['border']};
                border-radius: 0px;
            }}
            QFrame#notebookSection[current="true"] {{
                border-left: 3px solid {colors['accent']};
            }}
            QFrame#notebookSection[flash="true"] {{
                border-left: 3px solid {colors['accent']};
                background: {colors['nav_active_bg']};
            }}
            QFrame#notebookSection:focus {{
                border-left: 3px solid {colors['accent']};
            }}
            #sectionHeader {{ font-size: 24px; font-weight: 700; padding-top: 2px; }}

            /* Invented content.  A dashed border and a different surface, so a made-up
               example can never be skim-read as a captured result. */
            QFrame#illustrationBox {{
                background: {colors['bg']};
                border: 1px dashed {colors['accent']};
                border-radius: 10px;
            }}

            /* A clickable variable.  The provenance is in the text; the colour only
               agrees with the diagrams.  VariableChip.refresh sets a dynamic `role`
               property and re-polishes, and the rules below are what make that visible:
               one colour bar down the left edge, so the hue never touches the text's
               contrast and a reader who cannot tell the hues apart still reads
               "Captured" or "Learned". */
            QPushButton#variableChip {{
                text-align: left;
                border: 1px solid {colors['border']};
                border-left-width: 3px;
                border-radius: 6px;
                padding: 5px 10px;
                background: transparent;
                color: {colors['text_primary']};
                font-family: Consolas, 'Courier New', monospace;
                font-size: 12px;
            }}
            {_role_chip_rules(colors)}
            QPushButton#variableChip:hover {{
                border-color: {colors['accent']};
                background: {colors['nav_active_bg']};
            }}
            QPushButton#variableChip:focus {{ border-color: {colors['accent']}; }}

            QWidget#equationRow {{
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-left: 3px solid {colors['accent']};
                border-radius: 8px;
            }}
            QPushButton#equationTerm {{
                text-align: center;
                border: 1px solid transparent;
                border-bottom: 1px dotted {colors['text_muted']};
                border-radius: 3px;
                padding: 2px 6px;
                background: transparent;
                color: {colors['text_primary']};
                font-family: Consolas, 'Courier New', monospace;
                font-size: 17px;
                font-weight: 700;
            }}
            QPushButton#equationTerm:hover {{
                border-color: {colors['accent']};
                background: {colors['nav_active_bg']};
            }}
            QPushButton#equationTerm:focus {{ border-color: {colors['accent']}; }}
            #equationOperator {{
                color: {colors['text_secondary']};
                font-family: 'Cambria Math', 'Cambria', 'Latin Modern Math', serif;
                font-size: 19px;
                padding: 0 2px;
            }}

            /* The one shared inspection panel. */
            QWidget#variablePanel {{
                background: {colors['card_bg']};
                border-left: 1px solid {colors['border']};
            }}

            #glossarySymbol {{
                font-family: Consolas, 'Courier New', monospace;
                font-size: 13px;
                font-weight: 600;
                color: {colors['text_primary']};
            }}

            /* Reference page <-> notebook links. */
            QPushButton#crossPageLink {{
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 6px 10px;
                background: transparent;
                color: {colors['accent']};
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton#crossPageLink:hover, QPushButton#crossPageLink:focus {{
                border-color: {colors['accent']};
                background: {colors['nav_active_bg']};
            }}
            #roleLegend {{
                border-bottom: 1px solid {colors['border']};
                padding-bottom: 4px;
            }}
        ''')

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.banner = self._banner(run_id)
        root.addWidget(self.banner)
        self.reading_bar = self._reading_bar()
        root.addWidget(self.reading_bar)
        # The shell does not own a scroll area.  A page that scrolls as one long document
        # controls its own scrollbar -- it navigates by moving the reader to a section.
        # Views that do not scroll themselves get wrapped in `_host`.
        self.stack = StageStack()
        root.addWidget(self.stack, 1)

        for sequence, amount in (('Alt+Right', 1), ('Alt+Left', -1)):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(lambda step=amount: self._advance(step))

        self._show_mode(DEFAULT_MODE)

    # ── Chrome ────────────────────────────────────────────────────────────────

    def _banner(self, run_id) -> QWidget:
        host = QWidget()
        host.setObjectName('recapBanner')
        row = QHBoxLayout(host)
        row.setContentsMargins(16, 8, 16, 8)
        row.setSpacing(8)
        title = QLabel(
            f'{"Saved run #" + str(run_id) if run_id is not None else "Current capture"}  ·  '
            f'{self.capture.metadata.get("model", "Model ID unavailable")}')
        # Its own name: sharing the host's made the label draw the banner's bottom rule
        # under itself, a stray underline beneath the run title.
        title.setObjectName('recapBannerTitle')
        title.setToolTip(title.text())
        title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        title.setMinimumWidth(120)
        row.addWidget(title, 1)
        self._mode_buttons: dict[str, QPushButton] = {}
        switch = QWidget()
        switch.setObjectName('modeSwitch')
        switch_row = QHBoxLayout(switch)
        switch_row.setContentsMargins(3, 3, 3, 3)
        switch_row.setSpacing(2)
        row.addWidget(switch)
        row.addSpacing(8)
        for key, name, purpose, _ in MODES:
            button = QPushButton(name)
            button.setObjectName('modeButton')
            button.setCheckable(True)
            button.setToolTip(purpose)
            button.setAccessibleName(name)
            button.clicked.connect(lambda _, k=key: self._show_mode(k))
            # Room for the bold checked state, which is wider than the hint it was sized by.
            bold = button.font()
            bold.setBold(True)
            button.setMinimumWidth(QFontMetrics(bold).horizontalAdvance(name) + 34)
            switch_row.addWidget(button)
            self._mode_buttons[key] = button
        verified = QLabel('✓ Validated capture')
        verified.setStyleSheet(
            f'color: {self._colors.get("success", "#16a34a")}; font-size: 12px; '
            'font-weight: 600;')
        verified.setToolTip('This run passed every verification gate when it was captured, '
                            'and RunCapture.validate() passed when it was opened.')
        row.addWidget(verified)
        details = QPushButton('Details…')
        details.setObjectName('crossPageLink')
        details.setToolTip('What TensorScope verified for this run, and every saved field.')
        details.clicked.connect(self.open_details)
        row.addWidget(details)
        self.details_button = details
        raw = QPushButton('All captured arrays…')
        raw.setObjectName('crossPageLink')
        raw.setToolTip('Every stored array for every phase and layer, with no framing.')
        raw.clicked.connect(self.open_raw_tensors)
        row.addWidget(raw)
        self.raw_button = raw
        return host

    def _reading_bar(self) -> QWidget:
        host = QWidget()
        host.setObjectName('readingBar')
        row = QHBoxLayout(host)
        row.setContentsMargins(16, 6, 16, 6)
        row.setSpacing(8)
        prompt = QLabel(f'“{chip_text(self.capture.prompt, 40)}”')
        prompt.setObjectName('promptChip')
        prompt.setToolTip(self.capture.prompt)
        prompt.setTextFormat(Qt.PlainText)
        prompt.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        prompt.setMinimumWidth(60)
        row.addWidget(prompt)
        self.progress = ChapterProgress()
        self.progress.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.progress.setMinimumWidth(140)
        row.addWidget(self.progress, 1)
        # The page's own control -- the notebook's Following chip -- goes here, so the
        # thread the page follows is always on screen.  Each page's widget is added once
        # and shown only while its page is.
        self._toolbar_row = QHBoxLayout()
        self._toolbar_row.setContentsMargins(0, 0, 0, 0)
        self._toolbar_row.setSpacing(0)
        row.addLayout(self._toolbar_row)
        self.contents = ContentsButton()
        self.contents.navigate.connect(self._nav_to)
        row.addWidget(self.contents)
        self.previous_stage = QPushButton('← Previous')
        self.next_stage = QPushButton('Next →')
        for button in (self.previous_stage, self.next_stage):
            button.setObjectName('stepButton')
            button.setFocusPolicy(Qt.StrongFocus)
            button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.previous_stage.clicked.connect(lambda: self._advance(-1))
        self.next_stage.clicked.connect(lambda: self._advance(1))
        row.addWidget(self.previous_stage)
        row.addWidget(self.next_stage)
        return host

    # ── Navigation ────────────────────────────────────────────────────────────

    def _view(self, key: str) -> QWidget:
        """Build a mode's view on first visit; the others cost nothing until asked for."""
        if key not in self._views:
            view = MODE_INDEX[key][2](self.capture, tokens=self._tokens)
            view.stage_changed.connect(self._stage_changed)
            view.context_changed.connect(self._context_changed)
            # Only the scrolling pages offer this, and the shell stays out of their way.
            jump = getattr(view, 'jump_requested', None)
            if jump is not None:
                jump.connect(self._jump_to)
            make_toolbar = getattr(view, 'toolbar_widget', None)
            if make_toolbar is not None:
                toolbar = make_toolbar()
                if toolbar is not None:
                    self._toolbar_row.addWidget(toolbar)
                    self._toolbars[key] = toolbar
            self._views[key] = view
            self._hosts[key] = self._host(view)
            self.stack.addWidget(self._hosts[key])
        return self._views[key]

    def _host(self, view: QWidget) -> QWidget:
        """What goes in the stack: the view, or the view inside a scroll area.

        A page that owns its scrolling says so by offering `reset_scroll`.  Anything else
        is a plain tall widget that used to rely on the shell's scroll area, so it gets one
        of its own — nesting two scroll areas would give the reader two scrollbars for one
        page.
        """
        if hasattr(view, 'reset_scroll'):
            return view
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 6, 0)
        inner_layout.setSpacing(0)
        inner_layout.addWidget(view)
        inner_layout.addStretch(1)
        area.setWidget(inner)
        return area

    def _reset_scroll(self) -> None:
        """Put a view back at its top, for the views where that is the right gesture.

        A scrolling page is navigated by `go_to`, which moves the reader to a section; the
        shell forcing it to the top afterwards would undo exactly that.
        """
        view = self._views.get(self._mode)
        if view is None or hasattr(view, 'reset_scroll'):
            return
        host = self._hosts.get(self._mode)
        if isinstance(host, QScrollArea):
            host.verticalScrollBar().setValue(0)

    def open_details(self) -> QWidget:
        """The provenance evidence, in its own window: read once, not on every screen."""
        if self._details_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle('Evidence for this run')
            dialog.resize(720, 620)
            layout = QVBoxLayout(dialog)
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.NoFrame)
            area.setWidget(self._provenance())
            layout.addWidget(area)
            self._details_dialog = dialog
        self._details_dialog.show()
        self._details_dialog.raise_()
        return self._details_dialog

    def open_raw_tensors(self) -> QWidget:
        """Every captured array, as a secondary control rather than a competing mode."""
        if self._raw_dialog is None:
            self._raw_dialog = RawTensorDialog(self.capture, self, tokens=self._tokens)
        self._raw_dialog.show()
        self._raw_dialog.raise_()
        return self._raw_dialog

    def _jump_to(self, mode: str, stage: str) -> None:
        """Follow a cross-page link.

        Three kinds: the reference page's "follow this in the run", the notebook's
        "explain this operation", and the ribbon's request for every captured array --
        which is a window, not a mode, so it is handled before the mode lookup.
        """
        if mode == 'raw' and not stage:
            self.open_raw_tensors()
            return
        if mode not in MODE_INDEX:
            return
        self._show_mode(mode)
        if stage and stage in self.view.stage_keys():
            self._nav_to(stage)

    @property
    def view(self) -> QWidget:
        return self._views[self._mode]

    def _show_mode(self, key: str) -> None:
        view = self._view(key)
        self._mode = key
        self._context = ''
        self.stack.setCurrentWidget(self._hosts[key])
        for existing, button in self._mode_buttons.items():
            button.setChecked(existing == key)
        for existing, toolbar in self._toolbars.items():
            toolbar.setVisible(existing == key)

        stages = list(view.NAV)
        self.contents.set_sections(stages, getattr(view, 'CHAPTERS', None))
        self.progress.set_chapters(getattr(view, 'CHAPTERS', None))
        for control in (self.contents, self.progress, self.previous_stage, self.next_stage):
            control.setVisible(bool(stages))
        self.previous_stage.setText(getattr(view, 'PREVIOUS_LABEL', '← Previous'))
        self.next_stage.setText(getattr(view, 'NEXT_LABEL', 'Next →'))
        if stages:
            self._stage_changed(view.current_stage())
        else:
            announce = getattr(view, 'announce', None)
            if announce is not None:
                announce()
            self._reset_scroll()

    def _nav_to(self, key: str) -> None:
        # Guarded because this is a slot, and `go_to` raises KeyError for a key its page
        # does not own.  An exception escaping a slot makes PyQt call abort(): the process
        # dies with no traceback and an exit code indistinguishable from a native crash,
        # which is how a plain KeyError here reads to a user as "Segmentation fault".
        # A control left over from the other page should do nothing, not kill the app.
        if key in self.view.stage_keys():
            self.view.go_to(key)

    def _advance(self, amount: int) -> None:
        keys = self.view.stage_keys()
        current = self.view.current_stage()
        if not keys or current not in keys:
            return
        position = keys.index(current) + amount
        if 0 <= position < len(keys):
            self._nav_to(keys[position])

    def _stage_changed(self, key: str) -> None:
        keys = self.view.stage_keys()
        self.contents.set_current(key)
        if key in keys:
            position = keys.index(key)
            names = dict(self.view.NAV)
            self.previous_stage.setEnabled(position > 0)
            self.next_stage.setEnabled(position < len(keys) - 1)
            self.previous_stage.setToolTip(
                f'{names[keys[position - 1]]}  (Alt+Left)' if position > 0 else '')
            self.next_stage.setToolTip(
                f'{names[keys[position + 1]]}  (Alt+Right)' if position < len(keys) - 1
                else '')
            where = getattr(self.view, 'progress_text', None)
            self.progress.set_position(
                key, where(key) if where is not None else
                f'{getattr(self.view, "NOUN", "Section")} {position + 1} of {len(keys)}')
        self._reset_scroll()

    def _context_changed(self, context: str) -> None:
        # The reader's selection, kept for the progress indicator's tooltip.  The page's
        # own Following control is where it is shown.
        self._context = context
        self.progress.setToolTip(context)

    # ── Evidence ──────────────────────────────────────────────────────────────

    def _provenance(self) -> Card:
        capture = self.capture
        metadata = capture.metadata
        card = Card('Evidence for this run — what TensorScope verified')
        card.field('Capture source', str(metadata.get('capture_source', 'not recorded')))
        card.field('Validation on open',
                   'RunCapture.validate() passed. Saved metadata attests the checks below '
                   'were performed at capture time, before saving.')
        card.field('Stock-eager logit comparison',
                   f'Match: {metadata.get("logits_match_stock_eager", "not recorded")}  ·  '
                   f'Max absolute difference: '
                   f'{metadata.get("logits_max_abs_diff_vs_stock_eager", "not recorded")}\n'
                   'TensorScope re-ran the prefill with stock eager attention and required '
                   'bit-exact identical logits. A non-zero difference would have discarded '
                   'the run.')
        card.field('Attention implementation faithfulness',
                   f'Faithful to upstream eager: '
                   f'{metadata.get("faithful_to_upstream_eager", "not recorded")}  ·  '
                   f'Verified calls: '
                   f'{metadata.get("attention_calls_verified", "not recorded")} / '
                   f'{len(capture.layers) * 2} expected (both phases × all layers)\n'
                   "CaptureAttention re-ran upstream's own eager_attention_forward on every "
                   'captured call and required bit-exact matching outputs.')
        card.field('Runtime',
                   f'{metadata.get("backend", "not recorded")}  ·  '
                   f'torch {metadata.get("torch", "not recorded")}  ·  '
                   f'captured at {capture.started_at}')
        card.field('Dtype handling',
                   f'Model weights: {metadata.get("weight_dtype", "not recorded")}  ·  '
                   f'Stored embedding: {capture.embedding.dtype}\n'
                   'bfloat16 activations are widened losslessly to float32 for NumPy '
                   'storage. No values are narrowed or lost.')
        card.field('Prompt', capture.prompt)
        card.add(Disclosure(
            'Show all saved metadata',
            lambda: label(json.dumps(metadata, indent=2, ensure_ascii=False))))
        return card
