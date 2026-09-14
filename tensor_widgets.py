"""Read-only Qt views of captured NumPy arrays; no model or persistence dependencies.

TensorInspector exposes every element through a virtual table. AttentionExplorer is
an explicitly paged display of the same arrays, with full inspectors one click away.
Neither widget writes to, casts, normalizes, or replaces the supplied tensors.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QApplication, QComboBox, QDialog,
    QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy, QSpinBox,
    QSplitter, QTableView, QVBoxLayout, QWidget,
)


def exact_scalar(value) -> str:
    """A full round-trip scalar representation, independent of NumPy print options.

    Widening float16/32 to Python float preserves the stored value and makes its
    binary precision explicit. Extended NumPy floats must not narrow to float64.
    """
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        return np.format_float_scientific(value, unique=True, trim="k")
    if isinstance(value, np.complexfloating) and value.dtype.itemsize > 16:
        real = exact_scalar(value.real)
        imag = exact_scalar(abs(value.imag))
        return f"({real}{'-' if np.signbit(value.imag) else '+'}{imag}j)"
    return repr(value.item() if isinstance(value, np.generic) else value)


def _plain_label(text: str = "", parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setMinimumWidth(0)
    return label


def _token_label(labels: Sequence | None, index: int) -> str:
    return repr(str(labels[index])) if labels is not None and index < len(labels) else ""


class TensorTableModel(QAbstractTableModel):
    """Virtual, non-editable last-two-axis view; leading axes are explicit slices.

    A vector has one value column and indexed rows. A scalar has one cell. The
    original array remains accessible as ``array`` without a whole-tensor copy.
    ``tensor_index`` maps any table cell to its full original tensor coordinates.
    """

    def __init__(self, array: np.ndarray, axes: Sequence[str] | None = None,
                 row_labels: Sequence | None = None, column_labels: Sequence | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.array = np.asarray(array)
        self.axes = tuple(axes) if axes is not None else tuple(
            f"axis {i}" for i in range(self.array.ndim))
        if len(self.axes) != self.array.ndim:
            raise ValueError("Provide one axis label for every tensor dimension.")
        self.row_labels = row_labels
        self.column_labels = column_labels
        self.leading_indices = tuple(0 for _ in self.array.shape[:-2])

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent.isValid() or any(size == 0 for size in self.array.shape[:-2]):
            return 0
        return int(self.array.shape[-2] if self.array.ndim >= 2
                   else self.array.shape[0] if self.array.ndim else 1)

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return int(self.array.shape[-1] if self.array.ndim >= 2 else 1)

    def tensor_index(self, row: int, column: int) -> tuple[int, ...]:
        if not 0 <= row < self.rowCount() or not 0 <= column < self.columnCount():
            raise IndexError("Table cell is outside the selected tensor slice.")
        if self.array.ndim == 0:
            return ()
        if self.array.ndim == 1:
            return (row,)
        return self.leading_indices + (row, column)

    def value_at(self, row: int, column: int):
        return self.array[self.tensor_index(row, column)]

    def set_slice(self, indices: Sequence[int]) -> None:
        values = tuple(int(i) for i in indices)
        if len(values) != len(self.leading_indices):
            raise ValueError("A slice index is required for each leading dimension.")
        if any(not 0 <= i < self.array.shape[axis] for axis, i in enumerate(values)):
            raise IndexError("Tensor slice index is outside the captured shape.")
        if values != self.leading_indices:
            self.beginResetModel()
            self.leading_indices = values
            self.endResetModel()

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return exact_scalar(self.value_at(index.row(), index.column()))
        if role == Qt.ToolTipRole:
            return self.cell_description(index.row(), index.column())
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def cell_description(self, row: int, column: int) -> str:
        coordinate = self.tensor_index(row, column)
        context = ", ".join(f"{axis}={index}" for axis, index in zip(self.axes, coordinate))
        token_context = []
        if coordinate:
            row_index = coordinate[-2] if len(coordinate) > 1 else coordinate[0]
            if self.row_labels is not None:
                token_context.append(f"row token {_token_label(self.row_labels, row_index)}")
            if self.column_labels is not None and len(coordinate) > 1:
                token_context.append(f"column token {_token_label(self.column_labels, coordinate[-1])}")
        tail = " · " + " · ".join(token_context) if token_context else ""
        return f"Index {coordinate} ({context}){tail}\nCaptured value: {exact_scalar(self.array[coordinate])}"

    def flags(self, index: QModelIndex):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable if index.isValid() else Qt.NoItemFlags

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        if orientation == Qt.Horizontal:
            if self.array.ndim < 2:
                return "Captured value"
            axis, labels = self.axes[-1], self.column_labels
        else:
            axis = self.axes[-2] if self.array.ndim >= 2 else self.axes[0] if self.axes else "scalar"
            labels = self.row_labels
        token = _token_label(labels, section)
        if role == Qt.ToolTipRole:
            return f"{axis} {section}" + (f": {token}" if token else "")
        return f"{section}: {token}" if token else str(section)


class _CopyTable(QTableView):
    """Table clipboard actions always use the source value, including in a preview."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setAlternatingRowColors(False)
        self.setWordWrap(False)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(29)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        for title, shortcut, indexed in (
            ("Copy selected values", QKeySequence.Copy, False),
            ("Copy tensor indices and values", QKeySequence("Ctrl+Shift+C"), True),
        ):
            action = QAction(title, self)
            action.setShortcut(shortcut)
            action.setShortcutContext(Qt.WidgetShortcut)
            action.triggered.connect(lambda checked=False, include=indexed: self.copy_selection(include))
            self.addAction(action)

    def copy_selection(self, include_indices: bool = False) -> str:
        cells = sorted(self.selectionModel().selectedIndexes(), key=lambda i: (i.row(), i.column()))
        model = self.model()
        rows: dict[int, list[str]] = {}
        for index in cells:
            row, column = index.row(), index.column()
            value = exact_scalar(model.value_at(row, column))
            if include_indices:
                value = f"{model.tensor_index(row, column)}\t{value}"
            rows.setdefault(row, []).append(value)
        text = "\n".join("\t".join(values) for values in rows.values())
        QApplication.clipboard().setText(text)
        return text


class TensorInspector(QWidget):
    """Complete tensor access with explicit dimensions, slicing, and exact cells.

    Public controls: ``model``, ``table``, ``slice_selectors`` (axis-order QSpinBox
    list), ``row_jump``, ``column_jump``, and ``selection_label``. ``axes`` labels
    each original dimension; token labels describe the last two dimensions (or
    vector rows). No tensor values are eagerly turned into strings or Qt items.
    """

    def __init__(self, array: np.ndarray, name: str = "", axes: Sequence[str] | None = None,
                 row_labels: Sequence | None = None, column_labels: Sequence | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.model = TensorTableModel(array, axes, row_labels, column_labels, self)
        self.array = self.model.array
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = _plain_label(
            f"{name or 'Captured tensor'} · shape {self.array.shape} · dtype {self.array.dtype} · "
            f"{self.array.size:,} values", self)
        self.summary.setStyleSheet("font-weight:600;")
        layout.addWidget(self.summary)
        anatomy = " · ".join(f"axis {i}: {axis} = {size:,}" for i, (axis, size)
                             in enumerate(zip(self.model.axes, self.array.shape))) or "Scalar (no axes)"
        layout.addWidget(_plain_label(anatomy))
        layout.addWidget(_plain_label(
            "Full captured tensor. Scroll or jump to any cell; leading-axis selectors choose a slice. "
            "Cell text and clipboard values preserve full scalar precision."))
        slices = QHBoxLayout()
        self.slice_selectors: list[QSpinBox] = []
        for axis, size in enumerate(self.array.shape[:-2]):
            slices.addWidget(QLabel(f"{self.model.axes[axis]}:"))
            selector = QSpinBox()
            selector.setObjectName(f"tensorSliceAxis{axis}")
            selector.setRange(0, max(0, size - 1))
            selector.setEnabled(size > 0)
            selector.setToolTip(f"Original axis {axis}, size {size}")
            selector.valueChanged.connect(self._slice_changed)
            slices.addWidget(selector)
            self.slice_selectors.append(selector)
        slices.addStretch(1)
        if self.slice_selectors:
            layout.addLayout(slices)
        self.table = _CopyTable(self)
        self.table.setObjectName("fullTensorTable")
        self.table.setModel(self.model)
        self.table.horizontalHeader().setDefaultSectionSize(190)
        self.table.verticalHeader().setMaximumWidth(210)
        self.table.setMinimumHeight(250)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.table, 1)
        navigation = QHBoxLayout()
        self.row_jump = QSpinBox()
        self.row_jump.setRange(0, max(0, self.model.rowCount() - 1))
        self.row_jump.setPrefix("Row ")
        self.column_jump = QSpinBox()
        self.column_jump.setRange(0, max(0, self.model.columnCount() - 1))
        self.column_jump.setPrefix("Column ")
        navigation.addWidget(self.row_jump)
        navigation.addWidget(self.column_jump)
        jump = QPushButton("Go to cell")
        jump.clicked.connect(self._jump)
        jump.setEnabled(self.model.rowCount() > 0 and self.model.columnCount() > 0)
        navigation.addWidget(jump)
        navigation.addStretch(1)
        copy = QPushButton("Copy selection")
        copy.clicked.connect(lambda: self.table.copy_selection())
        navigation.addWidget(copy)
        layout.addLayout(navigation)
        self.selection_label = _plain_label("Select a cell for its full tensor index and captured value.")
        layout.addWidget(self.selection_label)
        self.table.selectionModel().currentChanged.connect(self._current_changed)

    def _slice_changed(self) -> None:
        self.model.set_slice([selector.value() for selector in self.slice_selectors])
        self.selection_label.setText("Select a cell in this slice for its full tensor index and captured value.")

    def _jump(self) -> None:
        index = self.model.index(self.row_jump.value(), self.column_jump.value())
        self.table.setCurrentIndex(index)
        self.table.scrollTo(index, QAbstractItemView.PositionAtCenter)

    def _current_changed(self, current: QModelIndex, previous=QModelIndex()) -> None:
        if current.isValid():
            self.selection_label.setText(self.model.cell_description(current.row(), current.column()))


class AttentionTableModel(TensorTableModel):
    """At most 32×32 actual cells, with a display-only color scale and compact text."""

    PAGE_SIZE = 32

    def __init__(self, array, axes, row_labels, column_labels, kind, parent=None):
        super().__init__(array, axes, row_labels, column_labels, parent)
        self.kind = kind
        self.row_offset = 0
        self.column_offset = 0
        self.selected_query = 0
        self.color_min, self.color_max = (0.0, 1.0)
        self.update_view(self.leading_indices, 0, 0, 0)

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return min(self.PAGE_SIZE, max(0, int(self.array.shape[-2]) - self.row_offset))

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return min(self.PAGE_SIZE, max(0, int(self.array.shape[-1]) - self.column_offset))

    def tensor_index(self, row, column):
        if not 0 <= row < self.rowCount() or not 0 <= column < self.columnCount():
            raise IndexError("Attention cell is outside this display page.")
        return self.leading_indices + (row + self.row_offset, column + self.column_offset)

    def _color_eligible(self, values):
        valid = np.isfinite(values)
        if self.kind == "scores" and np.issubdtype(values.dtype, np.floating):
            # Upstream additive causal masks often use finfo.min (not -inf).
            # Exclude only from color scaling, never from values or cell access.
            valid = valid & (values > np.finfo(values.dtype).min / 2)
        return valid

    def update_view(self, leading_indices, row_offset, column_offset, selected_query):
        self.beginResetModel()
        self.leading_indices = tuple(leading_indices)
        self.row_offset, self.column_offset = int(row_offset), int(column_offset)
        self.selected_query = int(selected_query)
        if self.kind == "scores":
            page = self.array[self.leading_indices + (
                slice(self.row_offset, self.row_offset + self.PAGE_SIZE),
                slice(self.column_offset, self.column_offset + self.PAGE_SIZE))]
            valid = page[self._color_eligible(page)]
            self.color_min = float(valid.min()) if valid.size else 0.0
            self.color_max = float(valid.max()) if valid.size else 0.0
        self.endResetModel()

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        value = self.value_at(index.row(), index.column())
        if role == Qt.DisplayRole:
            return format(float(value), ".4g")
        if role == Qt.BackgroundRole:
            if not self._color_eligible(np.asarray(value)).item():
                return QBrush(QColor("#343c4a"))
            span = self.color_max - self.color_min
            fraction = (float(value) - self.color_min) / span if span else 0.5
            fraction = max(0.0, min(1.0, fraction))
            low, high = ((25, 43, 77), (111, 76, 192)) if self.kind == "scores" else ((20, 42, 51), (21, 128, 110))
            return QBrush(QColor(*(round(a + fraction * (b - a)) for a, b in zip(low, high))))
        if role == Qt.ForegroundRole:
            return QBrush(QColor("#f4f7fc"))
        if role == Qt.FontRole and index.row() + self.row_offset == self.selected_query:
            font = QFont()
            font.setBold(True)
            return font
        return super().data(index, role)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        original = section + (self.column_offset if orientation == Qt.Horizontal else self.row_offset)
        text = super().headerData(original, orientation, role)
        if role == Qt.DisplayRole and orientation == Qt.Vertical and original == self.selected_query:
            return f"▶ {text}"
        return text


class AttentionExplorer(QWidget):
    """Compare captured scores and weights for one head and a token-aware page.

    ``tensors`` must contain equally shaped attention_scores and attention_weights
    of shape [B,H,Q,K], [H,Q,K], or [Q,K]. Labels correspond to Q and K respectively;
    generated-token Q=1 and cache-expanded K are intentionally independent.
    Public controls: ``head_selector``, ``batch_selector``, ``query_selector``,
    ``query_page``, ``key_page``; dictionaries ``models`` and ``tables``; and
    ``selection_label``. ``open_full_tensor(kind)`` accepts 'scores' or 'weights'.
    """

    def __init__(self, tensors: Mapping[str, np.ndarray], query_labels: Sequence | None,
                 key_labels: Sequence | None, layer_index: int = 0, phase: str = "prefill",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tensors = {kind: np.asarray(tensors[f"attention_{kind}"]) for kind in ("scores", "weights")}
        scores = self.tensors["scores"]
        if scores.shape != self.tensors["weights"].shape or scores.ndim not in (2, 3, 4):
            raise ValueError("Attention scores and weights must share [B,H,Q,K], [H,Q,K], or [Q,K] shape.")
        if any(size == 0 for size in scores.shape):
            raise ValueError("Attention exploration requires nonempty captured axes.")
        self.layer_index, self.phase = layer_index, phase
        self.query_labels, self.key_labels = query_labels, key_labels
        self.axes = (("batch", "head") if scores.ndim == 4 else ("head",) if scores.ndim == 3 else ()) + (
            "query token", "key token")
        self._syncing = False
        self._dialogs: list[QDialog] = []
        self.models: dict[str, AttentionTableModel] = {}
        self.tables: dict[str, _CopyTable] = {}
        self.color_labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.batch_selector = QSpinBox()
        self.batch_selector.setRange(0, scores.shape[0] - 1 if scores.ndim == 4 else 0)
        self.head_selector = QSpinBox()
        self.head_selector.setObjectName("attentionHead")
        self.head_selector.setRange(0, scores.shape[-3] - 1 if scores.ndim >= 3 else 0)
        self.head_selector.setPrefix("Head ")
        if scores.ndim == 4 and scores.shape[0] > 1:
            controls.addWidget(QLabel("Batch"))
            controls.addWidget(self.batch_selector)
        controls.addWidget(self.head_selector)
        controls.addWidget(QLabel("Query token"))
        self.query_selector = QComboBox()
        self.query_selector.setObjectName("attentionQuery")
        self.query_selector.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.query_selector.setMinimumContentsLength(15)
        self.query_selector.setMinimumWidth(0)
        for index in range(scores.shape[-2]):
            self.query_selector.addItem(f"{index}: {_token_label(query_labels, index)}", index)
        controls.addWidget(self.query_selector, 1)
        layout.addLayout(controls)
        pages = QHBoxLayout()
        self.query_page, self.key_page = QSpinBox(), QSpinBox()
        self.query_page.setRange(1, math.ceil(scores.shape[-2] / AttentionTableModel.PAGE_SIZE))
        self.key_page.setRange(1, math.ceil(scores.shape[-1] / AttentionTableModel.PAGE_SIZE))
        pages.addWidget(QLabel("Query page"))
        pages.addWidget(self.query_page)
        pages.addWidget(QLabel("Key page"))
        pages.addWidget(self.key_page)
        pages.addStretch(1)
        layout.addLayout(pages)
        self.page_label = _plain_label()
        layout.addWidget(self.page_label)
        layout.addWidget(_plain_label(
            "Rows = query tokens; columns = key tokens. Display labels use 4 significant digits; "
            "click or hover for the exact captured value. Colors aid reading; they are not extra model outputs."))
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)
        for kind, heading in (("scores", "Captured scores · already scaled / masked"),
                              ("weights", "Captured weights · the model's softmax output")):
            pane = QWidget()
            pane.setMinimumWidth(0)
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(0, 0, 0, 0)
            title = _plain_label(heading)
            title.setStyleSheet("font-weight:600;")
            pane_layout.addWidget(title)
            model = AttentionTableModel(self.tensors[kind], self.axes, query_labels, key_labels, kind, self)
            table = _CopyTable()
            table.setObjectName(f"attention{kind.title()}Table")
            table.setModel(model)
            table.horizontalHeader().setDefaultSectionSize(85)
            table.verticalHeader().setMaximumWidth(130)
            table.setMinimumHeight(250)
            table.setMinimumWidth(0)
            table.selectionModel().currentChanged.connect(
                lambda current, previous, selected_kind=kind: self._cell_selected(selected_kind, current))
            pane_layout.addWidget(table)
            self.color_labels[kind] = _plain_label()
            pane_layout.addWidget(self.color_labels[kind])
            open_button = QPushButton(f"Open full {kind} tensor")
            open_button.clicked.connect(lambda checked=False, selected_kind=kind: self.open_full_tensor(selected_kind))
            pane_layout.addWidget(open_button)
            self.models[kind], self.tables[kind] = model, table
            splitter.addWidget(pane)
        splitter.setSizes([500, 500])
        self.selection_label = _plain_label()
        layout.addWidget(self.selection_label)
        self.batch_selector.valueChanged.connect(self._refresh)
        self.head_selector.valueChanged.connect(self._refresh)
        self.query_selector.currentIndexChanged.connect(self._query_changed)
        self.query_page.valueChanged.connect(self._page_changed)
        self.key_page.valueChanged.connect(self._refresh)
        self._refresh()

    def _leading_indices(self):
        ndim = self.tensors["scores"].ndim
        return ((self.batch_selector.value(), self.head_selector.value()) if ndim == 4 else
                (self.head_selector.value(),) if ndim == 3 else ())

    def _query_changed(self) -> None:
        page = self.query_selector.currentIndex() // AttentionTableModel.PAGE_SIZE + 1
        self.query_page.blockSignals(True)
        self.query_page.setValue(page)
        self.query_page.blockSignals(False)
        self._refresh()

    def _page_changed(self) -> None:
        first = (self.query_page.value() - 1) * AttentionTableModel.PAGE_SIZE
        self.query_selector.blockSignals(True)
        self.query_selector.setCurrentIndex(first)
        self.query_selector.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        row_start = (self.query_page.value() - 1) * AttentionTableModel.PAGE_SIZE
        column_start = (self.key_page.value() - 1) * AttentionTableModel.PAGE_SIZE
        query = self.query_selector.currentIndex()
        for kind, model in self.models.items():
            model.update_view(self._leading_indices(), row_start, column_start, query)
            self.tables[kind].setCurrentIndex(model.index(query - row_start, 0))
            scale = (f"Color range on this page: {model.color_min:.4g} to {model.color_max:.4g}. "
                     "Gray = nonfinite or dtype-min mask-scale score; its value is preserved."
                     if kind == "scores" else "Color range: 0 to 1. Each cell is a captured attention weight.")
            self.color_labels[kind].setText(scale)
        shape = self.tensors["scores"].shape
        self.page_label.setText(
            f"Display page · {self.phase} · layer {self.layer_index} · batch {self.batch_selector.value()} · "
            f"head {self.head_selector.value()} · queries {row_start}–{min(row_start + 31, shape[-2] - 1)} "
            f"of {shape[-2]} · keys {column_start}–{min(column_start + 31, shape[-1] - 1)} of {shape[-1]}. "
            f"Full shape {shape}; dtype {self.tensors['scores'].dtype}. Maximum page 32 × 32.")
        self._syncing = False
        self._describe_cell(query, column_start)

    def _cell_selected(self, kind, current) -> None:
        if self._syncing or not current.isValid():
            return
        model = self.models[kind]
        coordinate = model.tensor_index(current.row(), current.column())
        query, key = coordinate[-2:]
        self._syncing = True
        self.query_selector.blockSignals(True)
        self.query_selector.setCurrentIndex(query)
        self.query_selector.blockSignals(False)
        for other_kind, other_model in self.models.items():
            old_query = other_model.selected_query
            other_model.selected_query = query
            if old_query != query:
                other_model.headerDataChanged.emit(Qt.Vertical, 0, other_model.rowCount() - 1)
                other_model.dataChanged.emit(other_model.index(0, 0), other_model.index(
                    other_model.rowCount() - 1, other_model.columnCount() - 1), [Qt.FontRole])
            if other_kind != kind:
                self.tables[other_kind].setCurrentIndex(other_model.index(current.row(), current.column()))
        self._syncing = False
        self._describe_cell(query, key)

    def _describe_cell(self, query, key) -> None:
        coordinate = self._leading_indices() + (query, key)
        self.selection_label.setText(
            f"{self.phase} · layer {self.layer_index} · batch {self.batch_selector.value()} · "
            f"head {self.head_selector.value()} · tensor index {coordinate}\n"
            f"Query {query}: {_token_label(self.query_labels, query)} → "
            f"Key {key}: {_token_label(self.key_labels, key)}\n"
            f"Exact captured score: {exact_scalar(self.tensors['scores'][coordinate])}\n"
            f"Exact captured weight: {exact_scalar(self.tensors['weights'][coordinate])}")

    def open_full_tensor(self, kind: str) -> QDialog:
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.setWindowTitle(f"{self.phase} · layer {self.layer_index} · attention {kind}")
        dialog.resize(1000, 700)
        layout = QVBoxLayout(dialog)
        inspector = TensorInspector(self.tensors[kind], f"Captured attention {kind}", self.axes,
                                    self.query_labels, self.key_labels, dialog)
        layout.addWidget(inspector)
        for selector, selected in zip(inspector.slice_selectors, self._leading_indices()):
            selector.setValue(selected)
        self._dialogs.append(dialog)
        dialog.destroyed.connect(lambda: self._dialogs.remove(dialog) if dialog in self._dialogs else None)
        dialog.show()
        return dialog
