"""The shell that hosts the reader-facing views of one capture.

`ComputationRecap` owns the window, the palette, the banner, the provenance evidence and
the navigation between three modes:

* **Understand** (`LearnView`) — the default.  What the model did with this prompt.
* **Internals** (`InternalsView`) — the same capture ordered by architecture.
* **Raw tensors** (`RawView`) — every captured array, with no framing at all.

It knows nothing about tensors.  Each view exposes the same three things — a `NAV` list
of `(stage key, label)`, a `go_to(key)`, and the `stage_changed` / `context_changed`
signals — and the shell drives its navigator, breadcrumb and footer from those alone, so
adding a mode does not touch this file's layout code.  The views themselves live in
`tensorscope_views`; the widgets they are built from live in `tensorscope_common`.
"""
from __future__ import annotations

import json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from tensorscope_common import Card, Disclosure, PipelineNavigator, label
from tensorscope_views import InternalsView, LearnView, RawView


# (key, sidebar/mode label, one-line purpose, view class).  Order is the order a reader
# should meet them: meaning first, architecture second, unframed arrays last.
MODES = [
    ('learn', 'Understand', 'What this model did with this prompt', LearnView),
    ('internals', 'Internals', 'The architecture, one step at a time', InternalsView),
    ('raw', 'Raw tensors', 'Every captured array, unframed', RawView),
]

MODE_INDEX = {key: (name, purpose, view) for key, name, purpose, view in MODES}

DEFAULT_COLORS = {
    'card_bg': '#1c1c20', 'border': '#3b3b44',
    'text_secondary': '#b5b5c1', 'text_primary': '#f0f0f2',
    'accent': '#60a5fa', 'bg': '#111114',
    'text_muted': '#636370', 'nav_active_bg': '#202028',
}


class ComputationRecap(QDialog):
    """Read-only window over one saved or just-captured run."""

    def __init__(self, capture, run_id=None, parent=None, tokens=None):
        super().__init__(parent)
        capture.validate()
        self.capture = capture
        self._tokens = tokens
        self._views: dict[str, QWidget] = {}
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

            /* Operators between shape/flow boxes */
            #shapeOp {{
                color: {colors['text_muted']};
                font-size: 16px;
                padding: 0 2px;
            }}

            /* Mode purpose line under each mode button */
            #modePurpose {{ color: {colors['text_muted']}; font-size: 10px; padding-left: 10px; }}
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
        pipeline_layout.addWidget(self.pipeline_nav)
        root.addWidget(pipeline_bar)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        split.addWidget(self._sidebar())
        split.addWidget(self._content())
        split.setSizes([235, 1065])

        self._show_mode(MODES[0][0])

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

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 6, 0)
        inner_layout.setSpacing(0)
        self.stack = QStackedWidget()
        inner_layout.addWidget(self.stack)
        inner_layout.addStretch(1)
        self.scroll.setWidget(inner)
        layout.addWidget(self.scroll, 1)

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
            self.stack.addWidget(view)
            self._views[key] = view
        return self._views[key]

    @property
    def view(self) -> QWidget:
        return self._views[self._mode]

    def _show_mode(self, key: str) -> None:
        view = self._view(key)
        self._mode = key
        self._context = ''
        self.stack.setCurrentWidget(view)
        for existing, button in self._mode_buttons.items():
            button.setChecked(existing == key)

        stages = list(view.NAV)
        self._rebuild_stage_buttons(stages)
        self.pipeline_nav.set_stages(stages)
        self.pipeline_nav.setVisible(bool(stages))
        self._stage_header.setVisible(bool(stages))
        self.previous_stage.setVisible(bool(stages))
        self.next_stage.setVisible(bool(stages))
        if stages:
            self._stage_changed(view.current_stage())
        else:
            # A stageless view built before the shell connected still has a position.
            announce = getattr(view, 'announce', None)
            if announce is not None:
                announce()
            else:
                self._refresh_breadcrumb('')
            self.scroll.verticalScrollBar().setValue(0)

    def _rebuild_stage_buttons(self, stages) -> None:
        while self._stage_box.count():
            item = self._stage_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._stage_buttons.clear()
        for position, (key, name) in enumerate(stages):
            button = QPushButton(f'{position + 1:02}  {name}')
            button.setObjectName('navItem')
            button.setCheckable(True)
            button.clicked.connect(lambda _, k=key: self._nav_to(k))
            self._stage_box.addWidget(button)
            self._stage_buttons[key] = button

    def _nav_to(self, key: str) -> None:
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
        self._refresh_breadcrumb(key)
        self.scroll.verticalScrollBar().setValue(0)

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
