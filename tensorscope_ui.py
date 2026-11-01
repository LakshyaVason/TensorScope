"""The shell that hosts the reader-facing views of one capture.

`ComputationRecap` owns the window, the palette, the banner, the provenance evidence and
the navigation between two pages:

* **Understand the calculation** (`ReferenceView`) — twelve illustrated topics: purpose,
  formula, symbols, picture, and an optional deeper explanation.
* **Follow this run** (`NotebookView`) — the default.  One scrolling worked calculation
  over this capture, from the prompt to the selected token and the continuation.

Every captured array stays reachable through the notebook ribbon's "All captured
arrays..." button, which opens `RawView` in a modeless `RawTensorDialog`.  That is
deliberately not a third row in `MODES`: see the comment there.

The shell knows nothing about tensors.  Each page exposes the same small contract — a
`NAV` list of `(section key, label)`, a `go_to(key)`, a `current_stage()`, and the
`stage_changed` / `context_changed` signals — and the shell drives its navigator,
sidebar, breadcrumb, progress pills and footer from those alone.  The pages themselves
live in `tensorscope_notebook` and `tensorscope_reference`, re-exported through
`tensorscope_views`; the widgets they are built from live in `tensorscope_common`.
"""
from __future__ import annotations

import json

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QShortcut, QSplitter,
    QVBoxLayout, QWidget,
)

from tensorscope_common import (
    COLORS, Card, Disclosure, PipelineNavigator, StageProgress, StageStack,
    clear_layout, label,
)
from tensorscope_views import NotebookView, RawTensorDialog, ReferenceView


# (key, sidebar/mode label, one-line purpose, view class).  Two pages, in the order the
# reader is meant to think of them: the reference is page one and the worked run is page
# two, and they use the same order and the same notation as each other.
#
# Every captured array is still reachable, through the notebook ribbon's
# "All captured arrays..." button, which opens `RawTensorDialog`.  It is deliberately not
# a third row here: a mode in this list competes with the pages for the reader's
# attention, and the whole point of the redesign was to stop offering four readings of one
# run instead of one that accumulates.
MODES = [
    ('understand', 'Understand the calculation',
     'Purpose, formula and a picture for each operation', ReferenceView),
    ('run', 'Follow this run',
     'One worked calculation, prompt to selected token', NotebookView),
]

# Which page opens first.  Not MODES[0]: the reference is page one by numbering, but a
# reader arriving at a finished capture wants the run, and comes back here when a step
# stops making sense.
DEFAULT_MODE = 'run'

MODE_INDEX = {key: (name, purpose, view) for key, name, purpose, view in MODES}

DEFAULT_COLORS = {
    'card_bg': '#1c1c20', 'border': '#3b3b44',
    'text_secondary': '#b5b5c1', 'text_primary': '#f0f0f2',
    'accent': '#60a5fa', 'bg': '#111114',
    'text_muted': '#636370', 'nav_active_bg': '#202028',
}


def _role_chip_rules() -> str:
    """One stylesheet rule per semantic role, read from the diagrams' own palette.

    `COLORS` is the single source of these hues, so a chip and the picture beside it can
    never disagree about what colour Q is.
    """
    return '\n'.join(
        f'QPushButton#variableChip[role="{role}"] {{ border-left-color: {hue}; }}'
        for role, hue in COLORS.items())


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
        self.setWindowTitle('TensorScope — Computation Recap')
        self.resize(1300, 900)
        self.setMinimumSize(900, 640)

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
            QPushButton#tokenChip:hover {{ border-color: {colors['accent']}; }}
            QPushButton#tokenChip:checked {{
                background: {colors['accent']};
                border-color: {colors['accent']};
                color: #ffffff;
                font-weight: 700;
            }}

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
                background: {colors['bg']};
                border: 1px solid {colors['border']};
                border-radius: 5px;
            }}
            #valueBoxNumber {{
                font-family: Consolas, 'Courier New', monospace; font-size: 12px;
            }}
            #bandHeading {{ font-size: 10px; letter-spacing: 0.06em; padding-top: 4px; }}
            #responseText {{ font-size: 14px; }}

            /* Operators between shape/flow boxes */
            #shapeOp {{
                color: {colors['text_muted']};
                font-size: 16px;
                padding: 0 2px;
            }}

            /* Mode purpose line under each mode button */
            #modePurpose {{ color: {colors['text_muted']}; font-size: 10px; padding-left: 10px; }}

            /* ── The two scrolling pages ──────────────────────────────────── */

            /* One anchored section of a long page.  The reader's current section is
               marked with a dynamic property rather than an inline stylesheet, so both
               palettes are handled here and the widget code stays colour-free. */
            QFrame#notebookSection {{
                background: {colors['card_bg']};
                border: 1px solid {colors['border']};
                border-left: 3px solid transparent;
                border-radius: 10px;
            }}
            QFrame#notebookSection[current="true"] {{
                border-left: 3px solid {colors['accent']};
            }}
            QFrame#notebookSection[flash="true"] {{
                border-left: 3px solid {colors['accent']};
                background: {colors['nav_active_bg']};
            }}
            QFrame#notebookSection:focus {{
                border-color: {colors['accent']};
            }}
            #sectionHeader {{ font-size: 17px; font-weight: 700; }}

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
            {_role_chip_rules()}
            QPushButton#variableChip:hover {{
                border-color: {colors['accent']};
                background: {colors['nav_active_bg']};
            }}
            QPushButton#variableChip:focus {{ border-color: {colors['accent']}; }}

            QWidget#equationRow {{
                background: {colors['bg']};
                border-left: 3px solid {colors['accent']};
                border-radius: 4px;
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
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton#equationTerm:hover {{
                border-color: {colors['accent']};
                background: {colors['nav_active_bg']};
            }}
            QPushButton#equationTerm:focus {{ border-color: {colors['accent']}; }}
            #equationOperator {{
                color: {colors['text_secondary']};
                font-family: Consolas, 'Courier New', monospace;
                font-size: 14px;
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
                border: 1px solid {colors['border']};
                border-radius: 6px;
                padding: 6px 12px;
                background: transparent;
                color: {colors['accent']};
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#crossPageLink:hover {{
                border-color: {colors['accent']};
                background: {colors['nav_active_bg']};
            }}
            #referencePreamble, #contextRibbon {{
                background: {colors['bg']};
                border-bottom: 1px solid {colors['border']};
            }}
            /* The colour key sits directly under the ribbon, so it carries the bottom
               border and the ribbon above it does not draw one between the two. */
            #contextRibbon {{ border-bottom: none; }}
            #roleLegend {{
                background: {colors['bg']};
                border-bottom: 1px solid {colors['border']};
            }}
        ''')

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._banner(run_id))

        pipeline_bar = QWidget()
        pipeline_bar.setObjectName('pipelineBar')
        pipeline_layout = QHBoxLayout(pipeline_bar)
        pipeline_layout.setContentsMargins(20, 8, 20, 8)
        self.pipeline_nav = PipelineNavigator([], self._nav_to, colors)
        pipeline_layout.addWidget(self.pipeline_nav, 1)
        self.progress = StageProgress()
        pipeline_layout.addWidget(self.progress)
        root.addWidget(pipeline_bar)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        split.addWidget(self._sidebar())
        split.addWidget(self._content())
        split.setSizes([235, 1065])

        for sequence, amount in (('Alt+Right', 1), ('Alt+Left', -1)):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(lambda step=amount: self._advance(step))

        self._show_mode(DEFAULT_MODE)

    # ── Chrome ────────────────────────────────────────────────────────────────

    def _banner(self, run_id) -> QWidget:
        host = QWidget()
        host.setObjectName('recapBanner')
        row = QHBoxLayout(host)
        row.setContentsMargins(24, 10, 24, 10)
        title = QLabel(
            f'{"Saved run #" + str(run_id) if run_id is not None else "Current capture"}  ·  '
            f'{self.capture.metadata.get("model", "Model ID unavailable")}  ·  '
            f'Real PyTorch capture')
        title.setObjectName('recapBanner')
        row.addWidget(title)
        row.addStretch(1)
        verified = QLabel('✓ Validated  ·  Stock-eager match attested')
        verified.setStyleSheet('color: #16a34a; font-size: 11px;')
        row.addWidget(verified)
        return host

    def _sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName('sidebar')
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(250)
        nav = QVBoxLayout(panel)
        nav.setContentsMargins(12, 18, 12, 18)
        nav.setSpacing(4)

        nav.addWidget(label('MODE', muted=True))
        self._mode_buttons: dict[str, QPushButton] = {}
        for key, name, purpose, _ in MODES:
            button = QPushButton(name)
            button.setObjectName('navItem')
            button.setCheckable(True)
            button.clicked.connect(lambda _, k=key: self._show_mode(k))
            nav.addWidget(button)
            caption = QLabel(purpose)
            caption.setObjectName('modePurpose')
            caption.setWordWrap(True)
            nav.addWidget(caption)
            self._mode_buttons[key] = button

        nav.addSpacing(14)
        self._stage_header = label('STAGES', muted=True)
        nav.addWidget(self._stage_header)
        self._stage_box = QVBoxLayout()
        self._stage_box.setContentsMargins(0, 0, 0, 0)
        self._stage_box.setSpacing(2)
        self._stage_buttons: dict[str, QPushButton] = {}
        nav.addLayout(self._stage_box)

        nav.addStretch(1)
        nav.addWidget(label(
            f'{len(self.capture.layers)} layers · '
            f'{len(self.capture.prompt_tokens)} prompt tokens\n'
            f'Weight dtype: {self.capture.metadata.get("weight_dtype", "not recorded")}\n'
            f'Stored dtype: {self.capture.embedding.dtype}',
            muted=True))
        return panel

    def _content(self) -> QWidget:
        colors = self._colors
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(8)

        self.breadcrumb = label('')
        self.breadcrumb.setStyleSheet(
            f'color: {colors["text_secondary"]}; font-size: 12px;')
        layout.addWidget(self.breadcrumb)
        layout.addWidget(Disclosure('Capture provenance & verification', self._provenance))

        # The shell no longer owns a scroll area.  A page that scrolls as one long
        # document has to control its own scrollbar — it navigates by moving the reader to
        # a section, which is the opposite of the scroll-to-top this used to force on every
        # stage change.  Views that do not scroll themselves get wrapped in `_host`.
        self.stack = StageStack()
        layout.addWidget(self.stack, 1)

        footer = QHBoxLayout()
        self.previous_stage = QPushButton('← Previous')
        self.next_stage = QPushButton('Next →')
        self.next_stage.setObjectName('primary')
        self.previous_stage.clicked.connect(lambda: self._advance(-1))
        self.next_stage.clicked.connect(lambda: self._advance(1))
        footer.addWidget(self.previous_stage)
        footer.addStretch(1)
        footer.addWidget(self.next_stage)
        layout.addLayout(footer)
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

        stages = list(view.NAV)
        self._rebuild_stage_buttons(stages)
        self.pipeline_nav.set_stages(stages)
        self.pipeline_nav.setVisible(bool(stages))
        self._stage_header.setVisible(bool(stages))
        self.previous_stage.setVisible(bool(stages))
        self.next_stage.setVisible(bool(stages))
        self.previous_stage.setText(getattr(view, 'PREVIOUS_LABEL', '← Previous'))
        self.next_stage.setText(getattr(view, 'NEXT_LABEL', 'Next →'))
        if stages:
            self._stage_changed(view.current_stage())
        else:
            # A stageless view built before the shell connected still has a position.
            announce = getattr(view, 'announce', None)
            if announce is not None:
                announce()
            else:
                self._refresh_breadcrumb('')
            self._reset_scroll()

    def _rebuild_stage_buttons(self, stages) -> None:
        clear_layout(self._stage_box)   # not takeAt/deleteLater -- see clear_layout
        self._stage_buttons.clear()
        for position, (key, name) in enumerate(stages):
            button = QPushButton(f'{position + 1:02}  {name}')
            button.setObjectName('navItem')
            button.setCheckable(True)
            button.clicked.connect(lambda _, k=key: self._nav_to(k))
            self._stage_box.addWidget(button)
            self._stage_buttons[key] = button

    def _nav_to(self, key: str) -> None:
        # Guarded because this is a slot, and `go_to` raises KeyError for a key its page
        # does not own.  An exception escaping a slot makes PyQt call abort(): the process
        # dies with no traceback and an exit code indistinguishable from a native crash,
        # which is how a plain KeyError here reads to a user as "Segmentation fault".
        # A control left over from the other page should do nothing, not kill the app.
        if key in self.view.stage_keys():
            self.view.go_to(key)

    def _advance(self, amount: int) -> None:
        keys = list(self._stage_buttons)
        if not keys:
            return
        position = keys.index(self.view.current_stage()) + amount
        if 0 <= position < len(keys):
            self._nav_to(keys[position])

    def _stage_changed(self, key: str) -> None:
        for existing, button in self._stage_buttons.items():
            button.setChecked(existing == key)
        self.pipeline_nav.set_current(key)
        keys = list(self._stage_buttons)
        if key in keys:
            position = keys.index(key)
            self.previous_stage.setEnabled(position > 0)
            self.next_stage.setEnabled(position < len(keys) - 1)
            self.progress.set_position(position, len(keys), getattr(self.view, 'NOUN', 'Stage'))
        self._refresh_breadcrumb(key)
        self._reset_scroll()

    def _context_changed(self, context: str) -> None:
        self._context = context
        self._refresh_breadcrumb(self.view.current_stage())

    def _refresh_breadcrumb(self, stage_key: str) -> None:
        trail = [MODE_INDEX[self._mode][0]]
        stage_titles = dict(self.view.NAV)
        if stage_key in stage_titles:
            trail.append(stage_titles[stage_key])
        if self._context:
            trail.append(self._context)
        self.breadcrumb.setText('  /  '.join(trail))

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
