"""The unframed view of a capture, and the window that holds it.

This module is what is left of the four journeys the recap used to offer. The two reader
facing pages now live in `tensorscope_notebook` and `tensorscope_reference`, and are
re-exported from here so the shell still imports exactly one of our modules.

`RawView` itself is unchanged. It was always the escape hatch -- every stored array,
reachable in two clicks, with no framing at all -- and that is exactly the role it keeps.
What changed is how it is reached: through the "All captured arrays..." button in the
banner and in the variable panel, rather than as a mode competing with the pages.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QVBoxLayout, QWidget,
)

from tensorscope_common import EvidenceBadge, badge_row, label
from tensorscope_content import (
    EVIDENCE_OBSERVED, EXTRA_TENSOR_EXPLANATIONS, TENSOR_EXPLANATIONS,
)
from tensorscope_notebook import NotebookView
from tensorscope_reference import ReferenceView
from tensorscope_variables import inspector_for

# Re-exported so `tensorscope_ui` imports one module of ours rather than three.  The two
# pages are not used in this file; they are its public surface.
__all__ = ['NotebookView', 'RawTensorDialog', 'RawView', 'ReferenceView']


# ── RawView ───────────────────────────────────────────────────────────────────

class RawView(QWidget):
    """Every captured array, reachable in two clicks.

    This is the escape hatch from both journeys: no framing, no sampling, no ordering
    imposed on the reader.  It has no stages, so `NAV` is empty and the shell hides the
    stage navigator for it; it still reports its position through `context_changed` so
    the breadcrumb keeps working.
    """

    stage_changed = pyqtSignal(str)
    context_changed = pyqtSignal(str)

    NAV: list = []

    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(parent)
        capture.validate()
        self.capture = capture
        self._tokens = tokens or {}
        self._inspector = None
        self._context = ''

        body = QVBoxLayout(self)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)
        body.addWidget(label('Raw tensors · every captured array', title=True))
        body.addWidget(label(
            'Pick a phase, a layer and a tensor. The inspector pages through the complete '
            'array — virtual rows and columns, explicit batch and head slices — at the '
            'dtype it was stored in. Reading never rewrites it.', muted=True))
        body.addWidget(badge_row(EvidenceBadge(EVIDENCE_OBSERVED, 'as captured')))

        row = QHBoxLayout()
        self.phase_picker = QComboBox()
        self.phase_picker.addItems(['Phase 1 · Prompt prefill',
                                    'Phase 2 · First generated-token pass'])
        self.layer_picker = QComboBox()
        self.layer_picker.addItem('Outside the layers', None)
        for index in sorted(capture.layers):
            self.layer_picker.addItem(f'Layer {index}', index)
        self.tensor_picker = QComboBox()
        for name, picker in (('Phase', self.phase_picker), ('Layer', self.layer_picker),
                             ('Tensor', self.tensor_picker)):
            picker.setAccessibleName(name)
            row.addWidget(picker, 1)
        body.addLayout(row)

        self.meaning = label('', muted=True)
        self.meaning.setTextFormat(Qt.RichText)
        body.addWidget(self.meaning)
        self._host = QVBoxLayout()
        body.addLayout(self._host, 1)

        self.phase_picker.currentIndexChanged.connect(self._refresh_tensors)
        self.layer_picker.currentIndexChanged.connect(self._refresh_tensors)
        self.tensor_picker.currentIndexChanged.connect(self._show_tensor)
        self._refresh_tensors()

    # The shell talks to every view through these three.
    @classmethod
    def stage_keys(cls) -> list[str]:
        return []

    def current_stage(self) -> str:
        return ''

    def go_to(self, key: str) -> None:       # nothing to navigate to
        return

    def announce(self) -> None:
        """Re-emit the current position, for a shell that connected after construction."""
        self.context_changed.emit(self._context)

    @property
    def generated(self) -> bool:
        return bool(self.phase_picker.currentIndex())

    def _phase_layers(self) -> dict:
        return (self.capture.generated_layers if self.generated
                else self.capture.layers)

    def _refresh_tensors(self, *_) -> None:
        previous = self.tensor_picker.currentData()
        self.tensor_picker.blockSignals(True)
        self.tensor_picker.clear()
        layer_index = self.layer_picker.currentData()
        if layer_index is None:
            self.tensor_picker.addItem('embedding', 'embedding')
            if not self.generated and getattr(self.capture, 'logits', None) is not None:
                self.tensor_picker.addItem('final_logits', 'final_logits')
        else:
            layer = self._phase_layers().get(layer_index)
            for name in (layer.tensors if layer is not None else {}):
                self.tensor_picker.addItem(name, name)
        found = self.tensor_picker.findData(previous)
        if found >= 0:
            self.tensor_picker.setCurrentIndex(found)
        self.tensor_picker.blockSignals(False)
        self._show_tensor()

    def _array(self, name: str):
        """The stored array itself — never a copy, never a sample."""
        if name == 'embedding':
            array = (self.capture.generated_embedding if self.generated
                     else self.capture.embedding)
            return array, EXTRA_TENSOR_EXPLANATIONS['embedding']
        if name == 'final_logits':
            return self.capture.logits, EXTRA_TENSOR_EXPLANATIONS['final_logits']
        layer = self._phase_layers().get(self.layer_picker.currentData())
        if layer is None:
            return None, ''
        return layer.tensors.get(name), TENSOR_EXPLANATIONS.get(
            name, 'Additional captured tensor.')

    def _show_tensor(self, *_) -> None:
        name = self.tensor_picker.currentData()
        if name is None:
            return
        array, text = self._array(name)
        if array is None:
            return
        self.meaning.setText(text)
        if self._inspector is not None:
            self._host.removeWidget(self._inspector)
            self._inspector.hide()
            self._inspector.deleteLater()
        self._inspector = inspector_for(self.capture, self.generated, name, array)
        self._host.addWidget(self._inspector)
        layer_index = self.layer_picker.currentData()
        where = 'outside the layers' if layer_index is None else f'Layer {layer_index}'
        phase = 'first generated token' if self.generated else 'prefill'
        self._context = f'{phase}  ·  {where}  ·  {name}'
        self.context_changed.emit(self._context)


class RawTensorDialog(QDialog):
    """Every captured array, in a window of its own.

    `RawView` survives the redesign unchanged; what changes is how it is reached. It is no
    longer a mode competing with the two pages for the reader's attention, but a secondary
    control on the notebook's ribbon -- so nothing became unreachable and nothing has to be
    framed to be inspected. Modeless on purpose: a reader comparing an array against the
    operation that produced it needs both on screen.
    """

    def __init__(self, capture, parent=None, tokens: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle('TensorScope — every captured array')
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.resize(1100, 760)
        body = QVBoxLayout(self)
        body.setContentsMargins(16, 14, 16, 14)
        self.view = RawView(capture, tokens=tokens)
        body.addWidget(self.view)
