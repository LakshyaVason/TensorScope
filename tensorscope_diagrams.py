"""QPainter sketches of the *structure* of each operation.  No capture, no model, no copy.

Every widget here draws a shape: how many token positions there are, how wide a vector is,
which row the reader selected, which inner dimension has to match for a multiplication to
be defined.  Three rules make that safe to reuse and safe to test.

* **A diagram never reads the capture.**  Constructors take ints, strings, and small numpy
  slices the caller already produced with basic indexing.  Nothing here imports
  `TensorScope`, so nothing here can reach `display_matrix` or a stored array.  That is
  also why this module sits below the variable registry in the import graph: a diagram
  cannot be the thing that decides what a tensor means.
* **Cells on screen are never confused with sizes in the run.**  A 2,560-feature vector is
  drawn as a dozen cells and an ellipsis, with the real count printed beside it.  Nothing
  draws N cells and then labels them N.
* **Colour never carries the only copy of a meaning.**  Every coloured element also has a
  text label, every colour resolves at paint time from the host's design tokens, and
  `describe()` returns exactly the text that was drawn -- which is what lets the tests
  assert the content of a picture they cannot read pixels from.

Transitions are decoration layered on an already-final state change.  Nothing a caller
does waits on one, and they are skipped under the offscreen platform, so the test suite
never spends time on a tint.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from tensorscope_common import CHROME, COLORS, OFFSCREEN, label


# How many cells a strip or a grid axis draws before it gives up and shows an ellipsis.
# Deliberately small: the point of a sketch is the shape, and a reader counting 2,560 boxes
# has learned nothing the printed number does not say better.
DRAWN_CELLS = 12
DRAWN_ROWS = 8

ELLIPSIS = '…'


def _number(value) -> str:
    """Thousands-separated, so 151936 never reads as a shape on screen."""
    return f'{int(value):,}'


def _count(value, singular: str, plural: str = '') -> str:
    """A count with a noun that agrees with it.  '1 queries x 16 keys' reads as a bug."""
    amount = int(value)
    noun = singular if amount == 1 else (plural or singular + 's')
    return f'{amount:,} {noun}'


class _Canvas(QWidget):
    """The painted surface of a `Diagram`.

    Separate from `Diagram` so a diagram can own a word-wrapped caption *label* beneath
    its drawing without `paintEvent` having to lay text out by hand.  `QWidget.grab()` on
    the diagram renders this child too, which is how the tests force a real paint.
    """

    def __init__(self, diagram: 'Diagram', parent=None):
        super().__init__(parent)
        self._diagram = diagram
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(diagram.BODY_HEIGHT)

    def sizeHint(self) -> QSize:
        return QSize(self._diagram.BODY_WIDTH, self._diagram.BODY_HEIGHT)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        try:
            self._diagram.draw(painter, self.rect().adjusted(1, 1, -2, -2))
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:
        if not self._diagram.handle_click(event.pos()):
            super().mousePressEvent(event)


class Diagram(QWidget):
    """Base class: theme-aware colours, a drawn body, and an optional caption below it.

    Subclasses override `draw(painter, rect)` and `describe()`.  Colours come from the
    host's design tokens first and the widget palette second, so the same widget is legible
    on the light and the dark palette with no branching in `draw`.
    """

    HIGHLIGHT_MS = 180
    BODY_HEIGHT = 120
    BODY_WIDTH = 360

    def __init__(self, *, tokens: dict | None = None, caption: str = '', parent=None):
        super().__init__(parent)
        self._tokens = dict(tokens or {})
        self._caption = caption
        self._selected_row: int | None = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(4)
        self.canvas = _Canvas(self)
        outer.addWidget(self.canvas)
        if caption:
            outer.addWidget(label(caption, muted=True, small=True))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # ── Colours ──────────────────────────────────────────────────────────────

    def _token(self, name: str, fallback: str) -> QColor:
        return QColor(self._tokens.get(name) or fallback)

    def ink(self) -> QColor:
        """Primary text and outline colour."""
        if self._tokens.get('text_primary'):
            return QColor(self._tokens['text_primary'])
        return self.palette().windowText().color()

    def muted(self) -> QColor:
        return self._token('text_muted', CHROME['muted'])

    def faint(self) -> QColor:
        return self._token('border', CHROME['ahead'])

    def surface(self) -> QColor:
        return self._token('card_bg', '#1c1c20')

    def accent(self) -> QColor:
        return QColor(CHROME['accent'])

    def role(self, key: str) -> QColor:
        return QColor(COLORS.get(key, CHROME['muted']))

    @staticmethod
    def wash(colour: QColor, alpha: int = 46) -> QColor:
        """A translucent fill of a role colour, for an area that must stay readable."""
        tinted = QColor(colour)
        tinted.setAlpha(alpha)
        return tinted

    # ── Text ─────────────────────────────────────────────────────────────────

    def font_at(self, points: int = 8, bold: bool = False) -> QFont:
        chosen = QFont(self.font())
        chosen.setPointSize(max(6, points))
        chosen.setBold(bold)
        return chosen

    def text_height(self, points: int = 8) -> int:
        """A floor, because a minimal offscreen container can report near-zero metrics."""
        return max(11, QFontMetrics(self.font_at(points)).height())

    def write(self, painter: QPainter, rect, text: str, *, colour: QColor | None = None,
              points: int = 8, bold: bool = False,
              align=Qt.AlignLeft | Qt.AlignVCenter) -> None:
        painter.save()
        painter.setFont(self.font_at(points, bold))
        painter.setPen(QPen(colour or self.ink()))
        painter.drawText(QRectF(rect), int(align), text)
        painter.restore()

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def outline(self, painter: QPainter, rect, colour: QColor, *, radius: int = 3,
                width: float = 1.0, fill: QColor | None = None, dashed: bool = False) -> None:
        pen = QPen(colour, width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.save()
        painter.setPen(pen)
        painter.setBrush(fill if fill is not None else Qt.NoBrush)
        painter.drawRoundedRect(QRectF(rect), radius, radius)
        painter.restore()

    def arrow(self, painter: QPainter, x1: float, y: float, x2: float, colour: QColor,
              *, dashed: bool = False) -> None:
        pen = QPen(colour, 1.4)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.save()
        painter.setPen(pen)
        painter.drawLine(QRectF(x1, y, 0, 0).topLeft(), QRectF(x2 - 6, y, 0, 0).topLeft())
        head = QPolygonF([QRectF(x2, y, 0, 0).topLeft(),
                          QRectF(x2 - 7, y - 4, 0, 0).topLeft(),
                          QRectF(x2 - 7, y + 4, 0, 0).topLeft()])
        painter.setPen(Qt.NoPen)
        painter.setBrush(colour)
        painter.drawPolygon(head)
        painter.restore()

    def cell_row(self, painter: QPainter, rect: QRect, count: int, colour: QColor, *,
                 drawn: int = DRAWN_CELLS, highlight: int | None = None) -> None:
        """A horizontal run of cells standing in for `count` entries.

        When `count` exceeds `drawn` an ellipsis cell takes the second-to-last slot, so the
        strip never implies it is showing everything.  The caller prints the real count.
        """
        shown = min(int(count), drawn)
        truncated = int(count) > shown
        slots = max(1, shown)
        width = rect.width() / slots
        for index in range(slots):
            box = QRect(int(rect.x() + index * width) + 1, rect.y(),
                        max(4, int(width) - 2), rect.height())
            is_gap = truncated and index == slots - 2
            if is_gap:
                self.write(painter, box, ELLIPSIS, colour=self.muted(), points=9,
                           align=Qt.AlignCenter)
                continue
            picked = highlight is not None and index == min(int(highlight), slots - 1)
            self.outline(painter, box, colour if picked else self.faint(),
                         radius=2, width=1.6 if picked else 1.0,
                         fill=self.wash(colour, 70 if picked else 22))

    # ── State ────────────────────────────────────────────────────────────────

    def set_selected_row(self, index: int | None) -> None:
        """Carry the reader's selected token position into this picture.

        Implemented on the base class so every diagram with a position axis answers the
        same call; diagrams without one simply have nothing to repaint.
        """
        self._selected_row = None if index is None else int(index)
        self.canvas.update()

    def selected_row(self) -> int | None:
        return self._selected_row

    def handle_click(self, point) -> bool:
        """Return True when the diagram consumed the click.  Default: it did not."""
        return False

    def flash(self) -> None:
        """A one-shot emphasis the reader asked for.  Pure decoration; never awaited."""
        if OFFSCREEN:
            return
        self.canvas.update()

    def describe(self) -> list[str]:
        """Every text label this diagram draws, in reading order.

        The tests assert against this because they cannot read the painted pixels, so a
        label that stops being drawn must stop being listed here too.
        """
        return [self._caption] if self._caption else []

    def draw(self, painter: QPainter, rect: QRect) -> None:   # pragma: no cover - abstract
        raise NotImplementedError


# ── Vectors and matrices ─────────────────────────────────────────────────────

class VectorStrip(Diagram):
    """One token's feature vector as a row of cells, with its real length printed."""

    BODY_HEIGHT = 64

    def __init__(self, length: int, *, symbol: str = '', role: str = 'layer',
                 unit: str = 'features', highlight: int | None = None,
                 drawn: int = DRAWN_CELLS, tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._length = int(length)
        self._symbol = symbol
        self._role = role
        self._unit = unit
        self._highlight = highlight
        self._drawn = drawn

    def _tail(self) -> str:
        return f'{_number(self._length)} {self._unit}'

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        label_width = 0
        if self._symbol:
            label_width = 64
            self.write(painter, QRect(rect.x(), rect.y(), label_width, rect.height() - line),
                       self._symbol, colour=self.role(self._role), points=10, bold=True)
        cells = QRect(rect.x() + label_width, rect.y() + 2,
                      rect.width() - label_width, max(14, rect.height() - line - 8))
        self.cell_row(painter, cells, self._length, self.role(self._role),
                      drawn=self._drawn, highlight=self._highlight)
        self.write(painter, QRect(rect.x() + label_width, rect.bottom() - line,
                                  rect.width() - label_width, line),
                   self._tail(), colour=self.muted())

    def describe(self) -> list[str]:
        return ([self._symbol] if self._symbol else []) + [self._tail()] + super().describe()


class MatrixGrid(Diagram):
    """A tensor's two interesting axes as an outlined grid with both real sizes printed.

    `highlight_row` is how the reader's selected token position stays visible across every
    tensor that has a position axis.
    """

    BODY_HEIGHT = 150
    cellClicked = pyqtSignal(int, int)

    def __init__(self, rows: int, cols: int, *, row_axis: str = 'token position',
                 col_axis: str = 'feature', role: str = 'layer', symbol: str = '',
                 highlight_row: int | None = None, highlight_col: int | None = None,
                 clickable: bool = False, drawn_rows: int = DRAWN_ROWS,
                 drawn_cols: int = DRAWN_CELLS, tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._rows, self._cols = int(rows), int(cols)
        self._row_axis, self._col_axis = row_axis, col_axis
        self._role, self._symbol = role, symbol
        self._selected_row = highlight_row
        self._highlight_col = highlight_col
        self._drawn_rows, self._drawn_cols = drawn_rows, drawn_cols
        self._grid = QRect()
        if clickable:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setCursor(Qt.PointingHandCursor)
        self._clickable = clickable

    # -- geometry -------------------------------------------------------------
    def _drawn(self) -> tuple[int, int, bool, bool]:
        shown_rows = min(self._rows, self._drawn_rows)
        shown_cols = min(self._cols, self._drawn_cols)
        return shown_rows, shown_cols, self._rows > shown_rows, self._cols > shown_cols

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        header = QRect(rect.x(), rect.y(), rect.width(), line)
        if self._symbol:
            self.write(painter, header, f'{self._symbol}   [{_number(self._rows)} × '
                                        f'{_number(self._cols)}]',
                       colour=self.role(self._role), points=9, bold=True)
        side = 74
        self._grid = QRect(rect.x() + side, rect.y() + line + 4,
                           max(40, rect.width() - side - 4),
                           max(30, rect.height() - 2 * line - 12))
        self.write(painter, QRect(rect.x(), self._grid.y(), side - 6, self._grid.height()),
                   self._row_axis, colour=self.muted(),
                   align=Qt.AlignRight | Qt.AlignVCenter)
        self.write(painter, QRect(self._grid.x(), rect.bottom() - line,
                                  self._grid.width(), line),
                   f'{self._col_axis}  →  {_number(self._cols)}', colour=self.muted())

        shown_rows, shown_cols, more_rows, more_cols = self._drawn()
        colour = self.role(self._role)
        cell_h = self._grid.height() / max(1, shown_rows)
        cell_w = self._grid.width() / max(1, shown_cols)
        for row in range(shown_rows):
            gap_row = more_rows and row == shown_rows - 2
            for col in range(shown_cols):
                gap_col = more_cols and col == shown_cols - 2
                box = QRect(int(self._grid.x() + col * cell_w) + 1,
                            int(self._grid.y() + row * cell_h) + 1,
                            max(3, int(cell_w) - 2), max(3, int(cell_h) - 2))
                if gap_row or gap_col:
                    if gap_row and col == shown_cols // 2:
                        self.write(painter, box, ELLIPSIS, colour=self.muted(),
                                   align=Qt.AlignCenter)
                    elif gap_col and row == shown_rows // 2:
                        self.write(painter, box, ELLIPSIS, colour=self.muted(),
                                   align=Qt.AlignCenter)
                    continue
                picked_row = self._selected_row is not None and row == min(
                    self._selected_row, shown_rows - 1)
                picked_col = self._highlight_col is not None and col == min(
                    self._highlight_col, shown_cols - 1)
                alpha = 96 if (picked_row and picked_col) else 58 if picked_row else \
                    48 if picked_col else 18
                self.outline(painter, box, colour if (picked_row or picked_col) else self.faint(),
                             radius=1, width=1.4 if picked_row else 1.0,
                             fill=self.wash(colour, alpha))
        if self._selected_row is not None:
            marker = QRect(rect.x(), int(self._grid.y() + min(
                self._selected_row, shown_rows - 1) * cell_h), side - 8, max(10, int(cell_h)))
            self.write(painter, marker, f'row {self._selected_row}', colour=self.accent(),
                       points=8, bold=True, align=Qt.AlignRight | Qt.AlignVCenter)

    def handle_click(self, point) -> bool:
        if not self._clickable or not self._grid.contains(point):
            return False
        shown_rows, shown_cols, _, _ = self._drawn()
        col = int((point.x() - self._grid.x()) / (self._grid.width() / max(1, shown_cols)))
        row = int((point.y() - self._grid.y()) / (self._grid.height() / max(1, shown_rows)))
        self._selected_row = max(0, min(row, shown_rows - 1))
        self._highlight_col = max(0, min(col, shown_cols - 1))
        self.canvas.update()
        self.cellClicked.emit(self._selected_row, self._highlight_col)
        return True

    def describe(self) -> list[str]:
        head = [f'{self._symbol}   [{_number(self._rows)} × {_number(self._cols)}]'] \
            if self._symbol else []
        return head + [self._row_axis, f'{self._col_axis}  →  {_number(self._cols)}'] \
            + ([f'row {self._selected_row}'] if self._selected_row is not None else []) \
            + super().describe()


class EmbeddingLookupRow(Diagram):
    """The embedding table as V x H, with one row pulled out.

    Exists to kill one specific misconception: a 2,560-wide embedding is one point in a
    2,560-dimensional space, not a 2560 x 2560 anything.  Both numbers are printed, and the
    caption says which is which.
    """

    BODY_HEIGHT = 150

    def __init__(self, vocab_size: int, hidden: int, *, token_id: int | None = None,
                 token_text: str = '', tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._vocab, self._hidden = int(vocab_size), int(hidden)
        self._token_id, self._token_text = token_id, token_text

    def _row_label(self) -> str:
        if self._token_id is None:
            return 'the row for this token'
        spelled = f'  {self._token_text!r}' if self._token_text else ''
        return f'row {_number(self._token_id)}{spelled}'

    def _pulled_label(self) -> str:
        return f'one vector: {_number(self._hidden)} numbers'

    def _rows_label(self) -> str:
        return (_count(self._vocab, 'row') if self._vocab
                else 'one row per vocabulary entry (count not recorded in this run)')

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        table = QRect(rect.x(), rect.y() + line, 118, rect.height() - 2 * line - 4)
        self.write(painter, QRect(table.x(), rect.y(), table.width(), line),
                   'embedding table', colour=self.muted())
        self.outline(painter, table, self.faint(), radius=3)
        rows = 9
        step = table.height() / rows
        picked = rows // 2
        for index in range(rows):
            strip = QRect(table.x() + 4, int(table.y() + index * step) + 1,
                          table.width() - 8, max(3, int(step) - 2))
            if index == picked:
                self.outline(painter, strip, self.accent(), radius=1, width=1.4,
                             fill=self.wash(self.accent(), 90))
            else:
                self.outline(painter, strip, self.faint(), radius=1,
                             fill=self.wash(self.muted(), 20))
        self.write(painter, QRect(table.x(), table.bottom() + 2, table.width(), line),
                   self._rows_label(), colour=self.muted())

        mid = int(table.y() + picked * step + step / 2)
        self.arrow(painter, table.right() + 6, mid, table.right() + 44, self.accent())
        pulled = QRect(table.right() + 52, mid - 14, max(60, rect.right() - table.right() - 56), 28)
        self.cell_row(painter, pulled, self._hidden, self.role('norm'))
        self.write(painter, QRect(pulled.x(), pulled.bottom() + 2, pulled.width(), line),
                   self._pulled_label(), colour=self.muted())
        self.write(painter, QRect(pulled.x(), pulled.y() - line - 2, pulled.width(), line),
                   self._row_label(), colour=self.accent(), points=8, bold=True)

    def describe(self) -> list[str]:
        return ['embedding table', self._rows_label(), self._row_label(),
                self._pulled_label()] + super().describe()


# ── Projection, heads, and the shapes of a multiplication ────────────────────

class MatMulShapes(Diagram):
    """Two operands and a result, with the inner dimension tied together.

    The tie line and the shared tint are the whole point: the reason a multiplication is
    defined at all is that one number appears on both operands, and that is the thing a
    reader most often cannot see in a written equation.
    """

    BODY_HEIGHT = 132

    def __init__(self, left: tuple, right: tuple, out: tuple, *,
                 left_symbol: str = '', right_symbol: str = '', out_symbol: str = '',
                 left_role: str = 'norm', right_role: str = 'layer', out_role: str = 'q',
                 tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._shapes = (tuple(left), tuple(right), tuple(out))
        self._symbols = (left_symbol, right_symbol, out_symbol)
        self._roles = (left_role, right_role, out_role)
        # The shared dimension: the last axis of the left operand and -- because every
        # equation here is written with a transposed weight -- the last axis of the right.
        self._inner = tuple(left)[-1] if left else 0

    def _shape_text(self, shape) -> str:
        return '[' + ' × '.join(_number(d) for d in shape) + ']'

    def _tie_text(self) -> str:
        return f'both are {_number(self._inner)} — these must match'

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        gap = 26
        width = max(54, (rect.width() - 2 * gap - 8) // 3)
        boxes = []
        for index, (shape, symbol, role) in enumerate(
                zip(self._shapes, self._symbols, self._roles)):
            box = QRect(rect.x() + index * (width + gap), rect.y() + line + 2,
                        width, max(34, rect.height() - 3 * line - 10))
            colour = self.role(role)
            self.outline(painter, box, colour, radius=4, width=1.4,
                         fill=self.wash(colour, 24))
            self.write(painter, QRect(box.x(), rect.y(), box.width(), line), symbol,
                       colour=colour, points=10, bold=True, align=Qt.AlignCenter)
            self.write(painter, QRect(box.x(), box.bottom() + 2, box.width(), line),
                       self._shape_text(shape), colour=self.muted(), align=Qt.AlignCenter)
            boxes.append(box)
            if index < 2:
                centre = box.y() + box.height() // 2
                self.write(painter, QRect(box.right(), centre - line // 2, gap, line),
                           '×' if index == 0 else '=', colour=self.muted(), points=12,
                           align=Qt.AlignCenter)

        # The inner dimension: tint the touching edge of both operands, then tie them.
        edge = self.accent()
        for box, side in ((boxes[0], 'right'), (boxes[1], 'right')):
            strip = QRect(box.right() - 5, box.y() + 2, 4, box.height() - 4)
            self.outline(painter, strip, edge, radius=1, width=1.0,
                         fill=self.wash(edge, 150))
        tie = rect.bottom() - line
        path = QPainterPath()
        path.moveTo(boxes[0].right() - 3, boxes[0].bottom() + line + 2)
        path.lineTo(boxes[0].right() - 3, tie)
        path.lineTo(boxes[1].right() - 3, tie)
        path.lineTo(boxes[1].right() - 3, boxes[1].bottom() + line + 2)
        painter.save()
        painter.setPen(QPen(edge, 1.2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.restore()
        self.write(painter, QRect(boxes[1].right() + 6, tie - line // 2,
                                  rect.right() - boxes[1].right() - 6, line),
                   self._tie_text(), colour=edge, points=8)

    def describe(self) -> list[str]:
        parts = [f'{symbol} {self._shape_text(shape)}'.strip()
                 for symbol, shape in zip(self._symbols, self._shapes)]
        return parts + [self._tie_text()] + super().describe()


class HeadSplit(Diagram):
    """Learned projection and reshaping, drawn as two stages with a labelled boundary.

    These are different kinds of operation and readers routinely merge them: 2,560 features
    become 4,096 because a learned matrix *computed* them, and 4,096 becomes 32 groups of
    128 because the same numbers were *regrouped*.  The boundary caption is drawn by this
    widget rather than passed in, so a caller cannot leave the distinction out.
    """

    BODY_HEIGHT = 168

    def __init__(self, features: int, projected: int, heads: int, head_dim: int,
                 positions: int, *, symbol: str = 'Q', role: str = 'q',
                 highlight_head: int | None = None, highlight_row: int | None = None,
                 tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._features, self._projected = int(features), int(projected)
        self._heads, self._head_dim = int(heads), int(head_dim)
        self._positions = int(positions)
        self._symbol, self._role = symbol, role
        self._highlight_head = highlight_head
        self._selected_row = highlight_row

    def _stage_labels(self) -> list[str]:
        return [
            f'{_number(self._features)} input features',
            'learned projection  × Wᵀ  (arithmetic)',
            f'{_number(self._projected)} projected features',
            'reshape only  —  no arithmetic',
            f'{_number(self._heads)} heads × {_number(self._head_dim)}',
            f'{_number(self._positions)} token positions stay separate rows',
        ]

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        rows = min(self._positions, 5)
        band = max(10, (rect.height() - 3 * line - 16) // max(1, rows))
        column = max(58, (rect.width() - 100) // 3)
        xs = [rect.x(), rect.x() + column + 44, rect.x() + 2 * column + 88]
        colour = self.role(self._role)

        for stage, x in enumerate(xs):
            for row in range(rows):
                top = rect.y() + line + 4 + row * band
                box = QRect(x, top, column, max(6, band - 3))
                picked = self._selected_row is not None and row == min(
                    self._selected_row, rows - 1)
                if stage == 2:
                    groups = min(self._heads, 4)
                    step = column / groups
                    for group in range(groups):
                        cell = QRect(int(x + group * step) + 1, top,
                                     max(4, int(step) - 3), max(6, band - 3))
                        hot = self._highlight_head is not None and group == min(
                            self._highlight_head, groups - 1)
                        self.outline(painter, cell, colour if (picked or hot) else self.faint(),
                                     radius=1, width=1.4 if hot else 1.0,
                                     fill=self.wash(colour, 96 if hot else 56 if picked else 20))
                else:
                    self.outline(painter, box, colour if picked else self.faint(), radius=1,
                                 width=1.4 if picked else 1.0,
                                 fill=self.wash(colour, 56 if picked else 20))
            width_label = (f'{_number(self._features)}' if stage == 0 else
                           f'{_number(self._projected)}' if stage == 1 else
                           f'{_number(self._heads)} × {_number(self._head_dim)}')
            self.write(painter, QRect(x, rect.y(), column, line), width_label,
                       colour=colour, points=9, bold=True, align=Qt.AlignCenter)

        middle = rect.y() + line + 4 + (rows * band) // 2
        self.arrow(painter, xs[0] + column + 6, middle, xs[1] - 6, colour)
        self.arrow(painter, xs[1] + column + 6, middle, xs[2] - 6, self.muted(), dashed=True)
        boundary = rect.y() + line + 4 + rows * band + 2
        self.write(painter, QRect(xs[0], boundary, column + 44, line),
                   'learned projection', colour=colour, points=8, bold=True,
                   align=Qt.AlignCenter)
        self.write(painter, QRect(xs[1] + 6, boundary, column + 44, line),
                   'reshape only — no arithmetic', colour=self.muted(), points=8,
                   align=Qt.AlignCenter)
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   f'{_number(self._positions)} token positions stay separate rows '
                   'through both stages', colour=self.muted())

    def describe(self) -> list[str]:
        return self._stage_labels() + super().describe()


class KVShare(Diagram):
    """Grouped-query attention: a few key/value heads serving many query heads."""

    BODY_HEIGHT = 120

    def __init__(self, kv_heads: int, query_heads: int, *, tokens=None, caption: str = '',
                 parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._kv, self._queries = max(1, int(kv_heads)), max(1, int(query_heads))

    def _group_text(self) -> str:
        return (f'{_count(self._kv, "key/value head")} → '
                f'{_count(self._queries, "query head")} '
                f'({self._queries // self._kv} query heads share each one)')

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        shown_kv = min(self._kv, 4)
        shown_q = min(self._queries, 8)
        left = QRect(rect.x(), rect.y() + line, 72, rect.height() - 2 * line - 6)
        right = QRect(rect.right() - 72, left.y(), 72, left.height())
        self.write(painter, QRect(left.x(), rect.y(), left.width(), line), 'K / V heads',
                   colour=self.role('k'), points=9, bold=True)
        self.write(painter, QRect(right.x(), rect.y(), right.width(), line), 'Q heads',
                   colour=self.role('q'), points=9, bold=True, align=Qt.AlignRight)
        kv_step = left.height() / shown_kv
        q_step = right.height() / shown_q
        for index in range(shown_kv):
            box = QRect(left.x(), int(left.y() + index * kv_step) + 1, left.width(),
                        max(6, int(kv_step) - 3))
            self.outline(painter, box, self.role('k'), radius=2,
                         fill=self.wash(self.role('k'), 40))
        for index in range(shown_q):
            box = QRect(right.x(), int(right.y() + index * q_step) + 1, right.width(),
                        max(5, int(q_step) - 3))
            self.outline(painter, box, self.role('q'), radius=2,
                         fill=self.wash(self.role('q'), 34))
        painter.save()
        painter.setPen(QPen(self.muted(), 1.0))
        per_group = max(1, shown_q // shown_kv)
        for index in range(shown_q):
            source = min(index // per_group, shown_kv - 1)
            y1 = left.y() + source * kv_step + kv_step / 2
            y2 = right.y() + index * q_step + q_step / 2
            painter.drawLine(QRectF(left.right() + 2, y1, 0, 0).topLeft(),
                             QRectF(right.x() - 2, y2, 0, 0).topLeft())
        painter.restore()
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   'copied, not recomputed', colour=self.muted(), align=Qt.AlignCenter)

    def describe(self) -> list[str]:
        return ['K / V heads', 'Q heads', self._group_text(), 'copied, not recomputed'] \
            + super().describe()


class QKCompare(Diagram):
    """One query vector against one key vector, with the shared width bracketed."""

    BODY_HEIGHT = 128

    def __init__(self, head_dim: int, *, query_pos: int = 0, key_pos: int = 0,
                 query_label: str = '', key_label: str = '', tokens=None, caption: str = '',
                 parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._head_dim = int(head_dim)
        self._query_pos, self._key_pos = int(query_pos), int(key_pos)
        self._query_label, self._key_label = query_label, key_label

    def _lines(self) -> list[str]:
        return [f'Q row for position {self._query_pos}'
                + (f'  {self._query_label!r}' if self._query_label else ''),
                f'K row for position {self._key_pos}'
                + (f'  {self._key_label!r}' if self._key_label else ''),
                f'both hold {_number(self._head_dim)} numbers — the head width',
                'one number out: how well this query matches this key']

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        strip_h = 22
        body = rect.width() - 110
        for index, (role, text) in enumerate((('q', self._lines()[0]), ('k', self._lines()[1]))):
            top = rect.y() + line + index * (strip_h + line + 6)
            self.write(painter, QRect(rect.x(), top - line, body, line), text,
                       colour=self.role(role), points=8, bold=True)
            self.cell_row(painter, QRect(rect.x(), top, body, strip_h), self._head_dim,
                          self.role(role))
        bracket_top = rect.y() + line
        bracket_bottom = rect.y() + line + 2 * (strip_h + line + 6) - line - 6
        painter.save()
        painter.setPen(QPen(self.accent(), 1.2))
        painter.drawLine(QRectF(rect.x() - 1, bracket_top, 0, 0).topLeft(),
                         QRectF(rect.x() - 1, bracket_bottom, 0, 0).topLeft())
        painter.restore()
        self.write(painter, QRect(rect.x(), bracket_bottom + 2, body, line),
                   self._lines()[2], colour=self.accent())
        out = QRect(rect.right() - 92, rect.y() + rect.height() // 2 - 16, 40, 32)
        self.arrow(painter, rect.x() + body + 4, out.y() + 16, out.x() - 4, self.role('scores'))
        self.outline(painter, out, self.role('scores'), radius=3, width=1.4,
                     fill=self.wash(self.role('scores'), 40))
        self.write(painter, QRect(out.right() + 4, out.y(), 50, 32), 'one\nscore',
                   colour=self.role('scores'), points=8)

    def describe(self) -> list[str]:
        return self._lines() + super().describe()


# ── Scores, masking and softmax ──────────────────────────────────────────────

class CausalMaskTriangle(Diagram):
    """Which key positions each query is allowed to read.

    `cached` draws the keys that came from the KV cache in their own band, so the
    generated-token pass reads as one query row against the cached prompt keys plus itself
    rather than as a square that happens to be thin.
    """

    BODY_HEIGHT = 158
    cellClicked = pyqtSignal(int, int)

    def __init__(self, queries: int, keys: int, *, selected_query: int | None = None,
                 selected_key: int | None = None, cached: int = 0, clickable: bool = False,
                 tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._queries, self._keys = max(1, int(queries)), max(1, int(keys))
        self._selected_row = selected_query
        self._selected_key = selected_key
        self._cached = max(0, int(cached))
        self._clickable = clickable
        self._grid = QRect()
        if clickable:
            self.canvas.setFocusPolicy(Qt.StrongFocus)
            self.canvas.setCursor(Qt.PointingHandCursor)

    def _shape_text(self) -> str:
        return (f'{_count(self._queries, "query", "queries")} × '
                f'{_count(self._keys, "key")}')

    def _mask_text(self) -> str:
        return ('blocked positions have a large negative number added to their score, '
                'not necessarily minus infinity')

    # The grid draws at most DRAWN_ROWS rows and DRAWN_CELLS + 4 columns, so for any real
    # run it is a sample of the matrix rather than the whole of it.  Drawing and clicking
    # therefore need the same drawn <-> real mapping, in both directions: `_selected_row`
    # and `_selected_key` are always REAL positions, because that is what the caller
    # passes in and what `cellClicked` consumers index captured tensors with.

    def _drawn_counts(self) -> tuple[int, int]:
        return min(self._queries, DRAWN_ROWS), min(self._keys, DRAWN_CELLS + 4)

    @staticmethod
    def _to_real(index: int, shown: int, total: int) -> int:
        """The real position that drawn row/column `index` stands for."""
        if shown <= 1:
            return max(0, total - 1)
        return round(index * (total - 1) / (shown - 1))

    @staticmethod
    def _to_drawn(position: int, shown: int, total: int) -> int:
        """The drawn row/column standing for real `position` -- the inverse of _to_real."""
        if shown <= 1 or total <= 1:
            return 0
        return max(0, min(shown - 1, round(position * (shown - 1) / (total - 1))))

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        self.write(painter, QRect(rect.x(), rect.y(), rect.width(), line),
                   self._shape_text(), colour=self.role('scores'), points=9, bold=True)
        shown_q, shown_k = self._drawn_counts()
        self._grid = QRect(rect.x() + 58, rect.y() + line + 4,
                           max(60, rect.width() - 62), max(28, rect.height() - 3 * line - 12))
        cell_w = self._grid.width() / shown_k
        cell_h = self._grid.height() / shown_q
        allowed, blocked = self.role('weights'), self.muted()
        # Map a drawn cell back to a real position, so the triangle edge is where the real
        # causal boundary is rather than where the drawn grid's diagonal happens to fall.
        picked_row = (None if self._selected_row is None
                      else self._to_drawn(self._selected_row, shown_q, self._queries))
        picked_col = (None if self._selected_key is None
                      else self._to_drawn(self._selected_key, shown_k, self._keys))
        for row in range(shown_q):
            real_q = self._to_real(row, shown_q, self._queries)
            for col in range(shown_k):
                real_k = self._to_real(col, shown_k, self._keys)
                box = QRect(int(self._grid.x() + col * cell_w) + 1,
                            int(self._grid.y() + row * cell_h) + 1,
                            max(3, int(cell_w) - 2), max(3, int(cell_h) - 2))
                # In the cached case every drawn query is the single new position, which may
                # read every key; otherwise a query may read keys up to its own index.
                may_read = real_k <= real_q + self._cached
                colour = allowed if may_read else blocked
                picked = picked_row is not None and row == picked_row
                cell = picked and picked_col is not None and col == picked_col
                self.outline(painter, box, colour if may_read else self.faint(), radius=1,
                             width=1.8 if cell else 1.3 if picked else 1.0,
                             fill=self.wash(colour, 110 if (may_read and cell) else
                                            70 if (may_read and picked) else
                                            40 if may_read else 0))
                if not may_read:
                    painter.save()
                    painter.setPen(QPen(self.faint(), 0.8))
                    painter.drawLine(QRectF(box.x(), box.bottom(), 0, 0).topLeft(),
                                     QRectF(box.right(), box.y(), 0, 0).topLeft())
                    painter.restore()
        if self._cached:
            band = QRect(self._grid.x(), self._grid.y() - 3,
                         int(cell_w * min(shown_k, max(1, round(
                             self._cached * shown_k / self._keys)))), 2)
            self.outline(painter, band, self.role('k'), radius=1, width=1.0,
                         fill=self.wash(self.role('k'), 170))
            self.write(painter, QRect(self._grid.x(), rect.bottom() - 2 * line,
                                      self._grid.width(), line),
                       f'{_count(self._cached, "key")} came from the cache',
                       colour=self.role('k'))
        self.write(painter, QRect(rect.x(), self._grid.y(), 52, self._grid.height()),
                   'query', colour=self.muted(), align=Qt.AlignRight | Qt.AlignVCenter)
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   'allowed / blocked', colour=self.muted())

    def handle_click(self, point) -> bool:
        if not self._clickable or not self._grid.contains(point):
            return False
        shown_q, shown_k = self._drawn_counts()
        row = int((point.y() - self._grid.y()) / (self._grid.height() / shown_q))
        col = int((point.x() - self._grid.x()) / (self._grid.width() / shown_k))
        # Convert to real positions before storing or emitting.  Emitting the drawn index
        # pointed every click at a lower position than the one under the cursor as soon as
        # the matrix was larger than the drawn grid -- which is every real run.
        self._selected_row = self._to_real(max(0, min(row, shown_q - 1)),
                                           shown_q, self._queries)
        self._selected_key = self._to_real(max(0, min(col, shown_k - 1)),
                                           shown_k, self._keys)
        self.canvas.update()
        self.cellClicked.emit(self._selected_row, self._selected_key)
        return True

    def describe(self) -> list[str]:
        extra = [f'{_count(self._cached, "key")} came from the cache'] if self._cached else []
        return [self._shape_text(), 'query', 'allowed / blocked', self._mask_text()] \
            + extra + super().describe()


class SoftmaxFlow(Diagram):
    """What softmax does, with no numbers at all.

    The reference page's level-2 rung: deliberately numberless, so a reader meets the shape
    of the operation before any captured value.
    """

    BODY_HEIGHT = 112

    STAGES = ('one score per key', 'exponentiate', 'divide by the row total',
              'weights that add up to 1')

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        heights = ((0.3, 0.9, 0.5, 0.7), (0.1, 1.0, 0.3, 0.6), (0.1, 1.0, 0.3, 0.6))
        column = max(56, (rect.width() - 3 * 22) // 4)
        for stage in range(4):
            x = rect.x() + stage * (column + 22)
            plot = QRect(x, rect.y() + 2, column, max(22, rect.height() - 2 * line - 8))
            if stage == 1:
                self.write(painter, plot, 'eˣ', colour=self.muted(), points=14,
                           align=Qt.AlignCenter)
            elif stage == 2:
                self.write(painter, plot, '÷ total', colour=self.muted(), points=11,
                           align=Qt.AlignCenter)
            else:
                bars = heights[0] if stage == 0 else heights[2]
                colour = self.role('scores') if stage == 0 else self.role('weights')
                step = plot.width() / len(bars)
                for index, fraction in enumerate(bars):
                    height = max(4, int(plot.height() * fraction))
                    bar = QRect(int(plot.x() + index * step) + 2, plot.bottom() - height,
                                max(4, int(step) - 4), height)
                    self.outline(painter, bar, colour, radius=1,
                                 fill=self.wash(colour, 110))
            self.write(painter, QRect(x, rect.bottom() - 2 * line, column, 2 * line),
                       self.STAGES[stage], colour=self.muted(), align=Qt.AlignHCenter | Qt.AlignTop)
            if stage < 3:
                self.arrow(painter, x + column + 4, plot.y() + plot.height() // 2,
                           x + column + 18, self.muted())

    def describe(self) -> list[str]:
        return list(self.STAGES) + super().describe()


class SoftmaxBars(Diagram):
    """A captured score row and its captured weight row, aligned by key position.

    The one diagram in this module that is handed real values.  It is handed exactly the
    two 1-D slices it draws -- the caller does the indexing -- and it prints the numbers
    beside the bars rather than asking the reader to measure a length.
    """

    BODY_HEIGHT = 170

    def __init__(self, scores, weights, *, labels=None, masked_indices=(),
                 selected: int | None = None, drawn: int = DRAWN_CELLS,
                 tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._scores = np.asarray(scores).reshape(-1)
        self._weights = np.asarray(weights).reshape(-1)
        self._labels = list(labels or [])
        self._masked = set(int(index) for index in masked_indices)
        self._selected_row = selected
        self._drawn = drawn

    def _shown(self) -> int:
        return int(min(self._drawn, self._scores.size, self._weights.size))

    def _rows(self) -> list[str]:
        return ['captured scores · already scaled and masked',
                'captured weights · the softmax output the model used']

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        shown = self._shown()
        if shown == 0:
            self.write(painter, rect, 'no values to draw', colour=self.muted())
            return
        column = rect.width() / shown
        finite = self._scores[np.isfinite(self._scores)]
        # Masked entries are a large negative additive value, so including them in the
        # range would flatten every real score into one pixel.  They are still drawn and
        # still labelled -- excluded from the *scale* only, exactly as the attention table
        # model does.
        visible = np.array([value for index, value in enumerate(finite[:shown])
                            if index not in self._masked]) if finite.size else finite
        low = float(visible.min()) if visible.size else 0.0
        high = float(visible.max()) if visible.size else 1.0
        span = (high - low) or 1.0
        weight_high = float(self._weights[:shown].max()) if self._weights.size else 1.0
        weight_high = weight_high or 1.0

        for band, (name, colour) in enumerate(
                zip(self._rows(), (self.role('scores'), self.role('weights')))):
            top = rect.y() + band * (rect.height() // 2)
            self.write(painter, QRect(rect.x(), top, rect.width(), line), name,
                       colour=colour, points=8, bold=True)
            plot = QRect(rect.x(), top + line + 1, rect.width(),
                         max(18, rect.height() // 2 - 2 * line - 4))
            for index in range(shown):
                value = float(self._scores[index] if band == 0 else self._weights[index])
                fraction = ((value - low) / span) if band == 0 else (value / weight_high)
                fraction = max(0.03, min(1.0, fraction))
                height = max(3, int(plot.height() * fraction))
                bar = QRect(int(rect.x() + index * column) + 2, plot.bottom() - height,
                            max(4, int(column) - 4), height)
                masked = index in self._masked
                picked = self._selected_row is not None and index == self._selected_row
                self.outline(painter, bar, colour if not masked else self.faint(), radius=1,
                             width=1.4 if picked else 1.0,
                             fill=self.wash(self.accent() if picked else colour,
                                            0 if masked else 120))
                if masked:
                    painter.save()
                    painter.setPen(QPen(self.faint(), 0.8, Qt.DashLine))
                    painter.drawLine(QRectF(bar.x(), bar.bottom(), 0, 0).topLeft(),
                                     QRectF(bar.right(), bar.y(), 0, 0).topLeft())
                    painter.restore()
                self.write(painter, QRect(bar.x() - 4, plot.bottom() + 1,
                                          max(14, int(column)), line),
                           f'{value:.3g}', colour=self.muted(), points=7,
                           align=Qt.AlignCenter)
        if self._labels:
            for index in range(shown):
                text = str(self._labels[index]) if index < len(self._labels) else ''
                self.write(painter, QRect(int(rect.x() + index * column),
                                          rect.bottom() - line, max(14, int(column)), line),
                           text[:6], colour=self.muted(), points=7, align=Qt.AlignCenter)

    def describe(self) -> list[str]:
        shown = self._shown()
        return self._rows() + [f'{shown} of {self._scores.size} key positions drawn'] \
            + super().describe()


class AttentionGrid(Diagram):
    """A small attention matrix as painted cells, in place of a matplotlib canvas.

    A 15x15 or 1x16 matrix is not worth a `FigureCanvas`, and a `MiniHeatmap` per layer is
    exactly the cost `Disclosure` laziness exists to avoid.  The caller passes a 2-D slice
    it produced with basic indexing; this never resamples and never writes.
    """

    BODY_HEIGHT = 190

    def __init__(self, matrix, *, row_labels=None, col_labels=None,
                 selected_query: int | None = None, tokens=None, caption: str = '',
                 parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        view = np.asarray(matrix)
        self._matrix = view if view.ndim == 2 else view.reshape(1, -1)
        self._row_labels = list(row_labels or [])
        self._col_labels = list(col_labels or [])
        self._selected_row = selected_query

    def _shape_text(self) -> str:
        return (f'{_count(self._matrix.shape[0], "query", "queries")} × '
                f'{_count(self._matrix.shape[1], "key")}')

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        self.write(painter, QRect(rect.x(), rect.y(), rect.width(), line),
                   self._shape_text(), colour=self.role('weights'), points=9, bold=True)
        rows, cols = self._matrix.shape
        shown_r, shown_c = min(rows, 24), min(cols, 32)
        grid = QRect(rect.x() + 40, rect.y() + line + 2, max(40, rect.width() - 44),
                     max(24, rect.height() - 2 * line - 8))
        cell_w, cell_h = grid.width() / shown_c, grid.height() / shown_r
        finite = self._matrix[np.isfinite(self._matrix)]
        low = float(finite.min()) if finite.size else 0.0
        high = float(finite.max()) if finite.size else 1.0
        span = (high - low) or 1.0
        base = self.role('weights')
        for row in range(shown_r):
            for col in range(shown_c):
                value = float(self._matrix[row, col])
                strength = 0.0 if not np.isfinite(value) else (value - low) / span
                box = QRect(int(grid.x() + col * cell_w), int(grid.y() + row * cell_h),
                            max(2, int(cell_w)), max(2, int(cell_h)))
                painter.fillRect(box, self.wash(base, int(18 + 210 * max(0.0, min(1.0, strength)))))
        self.outline(painter, grid, self.faint(), radius=2)
        if self._selected_row is not None and shown_r:
            top = int(grid.y() + min(self._selected_row, shown_r - 1) * cell_h)
            self.outline(painter, QRect(grid.x(), top, grid.width(), max(3, int(cell_h))),
                         self.accent(), radius=1, width=1.4)
        self.write(painter, QRect(rect.x(), grid.y(), 34, grid.height()), 'query',
                   colour=self.muted(), align=Qt.AlignRight | Qt.AlignVCenter)
        self.write(painter, QRect(grid.x(), rect.bottom() - line, grid.width(), line),
                   f'key position  →   darker means more weight', colour=self.muted())

    def describe(self) -> list[str]:
        return [self._shape_text(), 'query', 'key position',
                'darker means more weight'] + super().describe()


# ── Assembling the layer ─────────────────────────────────────────────────────

class ResidualBypass(Diagram):
    """A trunk with a bypass arc leaving before a branch and rejoining after it.

    Drawn as an arc *around* the branch rather than as another box in a row, because the
    thing worth seeing is that the input survives the branch untouched.
    """

    BODY_HEIGHT = 140

    def __init__(self, branches, *, trunk_symbols=('X',), tokens=None, caption: str = '',
                 parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        # branches: a sequence of (branch_label, output_symbol, result_symbol)
        self._branches = [tuple(item) for item in branches]
        self._trunk = list(trunk_symbols)

    def _lines(self) -> list[str]:
        out = []
        for index, (branch, output, result) in enumerate(self._branches):
            carried = self._trunk[index] if index < len(self._trunk) else 'X'
            out += [branch, f'{result} = {carried} + {output}',
                    f'{carried} travels around {branch} unchanged']
        return out

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        count = max(1, len(self._branches))
        column = max(90, (rect.width() - 40) // count)
        trunk_y = rect.y() + rect.height() - 2 * line - 8
        painter.save()
        painter.setPen(QPen(self.role('layer'), 1.6))
        painter.drawLine(QRectF(rect.x(), trunk_y, 0, 0).topLeft(),
                         QRectF(rect.right(), trunk_y, 0, 0).topLeft())
        painter.restore()
        for index, (branch, output, result) in enumerate(self._branches):
            x = rect.x() + 18 + index * column
            box = QRect(x, rect.y() + line + 6, column - 56, max(22, trunk_y - rect.y() - 3 * line))
            colour = self.role('output' if index == 0 else 'norm')
            self.outline(painter, box, colour, radius=4, width=1.3,
                         fill=self.wash(colour, 26))
            self.write(painter, box, branch, colour=colour, points=8, align=Qt.AlignCenter)
            # the bypass arc: leaves the trunk before the branch, rejoins at the add
            arc = QPainterPath()
            add_x = box.right() + 26
            arc.moveTo(x - 10, trunk_y)
            arc.cubicTo(x - 10, trunk_y + 22, add_x, trunk_y + 22, add_x, trunk_y)
            painter.save()
            painter.setPen(QPen(self.accent(), 1.3))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(arc)
            painter.restore()
            self.arrow(painter, box.x() + box.width() // 2, box.bottom() + 3,
                       box.x() + box.width() // 2, colour)
            painter.save()
            painter.setPen(QPen(colour, 1.2))
            painter.drawLine(QRectF(box.center().x(), box.bottom(), 0, 0).topLeft(),
                             QRectF(box.center().x(), trunk_y - 1, 0, 0).topLeft())
            painter.restore()
            plus = QRect(add_x - 9, trunk_y - 9, 18, 18)
            self.outline(painter, plus, self.accent(), radius=9, width=1.4,
                         fill=self.surface())
            self.write(painter, plus, '+', colour=self.accent(), points=10, bold=True,
                       align=Qt.AlignCenter)
            carried = self._trunk[index] if index < len(self._trunk) else 'X'
            self.write(painter, QRect(x - 14, trunk_y + 22, column, line),
                       f'{result} = {carried} + {output}', colour=self.accent(), points=8,
                       bold=True)
        self.write(painter, QRect(rect.x(), rect.y(), rect.width(), line),
                   'the residual stream', colour=self.role('layer'), points=9, bold=True)

    def describe(self) -> list[str]:
        return ['the residual stream'] + self._lines() + super().describe()


class GatedFFNPath(Diagram):
    """The gated feed-forward path: two projections, a gate, a product, a projection back.

    `intermediate=None` prints that the interior width was not recorded, because this
    capture genuinely does not contain it and a plausible-looking number would be invented.
    """

    BODY_HEIGHT = 150

    def __init__(self, hidden: int, intermediate: int | None = None, *, tokens=None,
                 caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._hidden = int(hidden)
        self._intermediate = None if intermediate is None else int(intermediate)

    def _width_text(self) -> str:
        return (f'{_number(self._intermediate)} features' if self._intermediate is not None
                else 'interior width not recorded in this capture')

    def _lines(self) -> list[str]:
        return ['gate(x)', 'SiLU', 'up(x)', 'multiply element by element', 'down(…)',
                f'in and out: {_number(self._hidden)} features', self._width_text(),
                'learned nonlinear processing of one position at a time — '
                'it does not squash values into 0…1']

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        colour = self.role('norm')
        product = self.role('output')
        box_w = max(52, (rect.width() - 150) // 3)
        top_y = rect.y() + line + 2
        mid_y = rect.y() + rect.height() // 2 - 10
        low_y = mid_y + 34

        start = QRect(rect.x(), mid_y, 34, 22)
        self.outline(painter, start, colour, radius=3, fill=self.wash(colour, 30))
        self.write(painter, start, 'x', colour=colour, points=10, bold=True,
                   align=Qt.AlignCenter)

        gate = QRect(start.right() + 20, top_y, box_w, 22)
        silu = QRect(gate.right() + 14, top_y, 44, 22)
        up = QRect(start.right() + 20, low_y, box_w, 22)
        for box, text in ((gate, 'gate(x)'), (silu, 'SiLU'), (up, 'up(x)')):
            self.outline(painter, box, colour, radius=3, width=1.2,
                         fill=self.wash(colour, 24))
            self.write(painter, box, text, colour=colour, points=8, align=Qt.AlignCenter)
        times = QRect(max(silu.right(), up.right()) + 18, mid_y, 20, 22)
        self.outline(painter, times, product, radius=10, width=1.4, fill=self.surface())
        self.write(painter, times, '⊙', colour=product, points=11, bold=True,
                   align=Qt.AlignCenter)
        down = QRect(times.right() + 18, mid_y, box_w, 22)
        self.outline(painter, down, product, radius=3, width=1.3,
                     fill=self.wash(product, 26))
        self.write(painter, down, 'down(…)', colour=product, points=8,
                   align=Qt.AlignCenter)

        self.arrow(painter, start.right() + 2, mid_y + 11, gate.x() - 2, colour)
        self.arrow(painter, gate.right() + 2, top_y + 11, silu.x() - 2, colour)
        self.arrow(painter, silu.right() + 2, top_y + 11, times.x() - 2, colour)
        self.arrow(painter, up.right() + 2, low_y + 11, times.x() - 2, colour)
        self.arrow(painter, times.right() + 2, mid_y + 11, down.x() - 2, product)
        self.write(painter, QRect(times.x() - 30, mid_y - line - 2, 80, line),
                   'multiply element by element', colour=product, points=7,
                   align=Qt.AlignCenter)
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   f'in and out: {_number(self._hidden)} features  ·  '
                   f'{self._width_text()}', colour=self.muted())

    def describe(self) -> list[str]:
        return self._lines() + super().describe()


class VocabularyProjection(Diagram):
    """Every position is computed; only the final position's score row chose a token.

    The greyed rows are the point: it is a common and reasonable assumption that the model
    scores one position, and the captured `final_logits` is one row precisely because the
    rest were never stored.
    """

    BODY_HEIGHT = 158

    def __init__(self, hidden: int, vocab: int, *, positions: int, final_position: int,
                 marked: int = 0, tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._hidden, self._vocab = int(hidden), int(vocab)
        self._positions, self._final = int(positions), int(final_position)
        self._marked = int(marked)

    def _scores_label(self) -> str:
        return (f'{_number(self._vocab)} scores, one per vocabulary entry' if self._vocab
                else 'one score per vocabulary entry (count not recorded in this run)')

    def _lines(self) -> list[str]:
        marked = [f'{_number(self._marked)} of them were recorded by name'] if self._marked else []
        return [f"{_count(self._positions, 'token position')}",
                f'only position {self._final} chose the next token',
                self._scores_label()] + marked

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        shown = min(self._positions, 7)
        column = 96
        rows = QRect(rect.x(), rect.y() + line, column, rect.height() - 2 * line - 4)
        step = rows.height() / shown
        final_row = shown - 1
        for index in range(shown):
            box = QRect(rows.x(), int(rows.y() + index * step) + 1, column,
                        max(5, int(step) - 3))
            if index == final_row:
                self.outline(painter, box, self.accent(), radius=2, width=1.4,
                             fill=self.wash(self.accent(), 90))
            else:
                self.outline(painter, box, self.faint(), radius=2,
                             fill=self.wash(self.muted(), 16))
        self.write(painter, QRect(rows.x(), rect.y(), column, line),
                   _count(self._positions, 'position'), colour=self.muted())
        mid = int(rows.y() + final_row * step + step / 2)
        self.arrow(painter, rows.right() + 4, mid, rows.right() + 40, self.accent())
        bar = QRect(rows.right() + 48, mid - 9, max(50, rect.right() - rows.right() - 52), 18)
        self.outline(painter, bar, self.role('scores'), radius=2, width=1.3,
                     fill=self.wash(self.role('scores'), 70))
        self.write(painter, QRect(bar.x(), bar.bottom() + 2, bar.width(), line),
                   f'{_number(self._vocab)} scores' if self._vocab
                   else 'score count not recorded', colour=self.muted())
        self.write(painter, QRect(bar.x(), bar.y() - line - 2, bar.width(), line),
                   f'position {self._final} only', colour=self.accent(), points=8, bold=True)
        if self._marked:
            width = max(6, bar.width() // 12)
            self.outline(painter, QRect(bar.x(), bar.y(), width, bar.height()),
                         self.accent(), radius=2, width=1.4,
                         fill=self.wash(self.accent(), 120))
            self.write(painter, QRect(bar.x(), rect.bottom() - line, bar.width(), line),
                       f'{_number(self._marked)} of them were recorded by name',
                       colour=self.accent(), points=7)

    def describe(self) -> list[str]:
        return self._lines() + super().describe()


class LayerStack(Diagram):
    """The decoder stack, with the layer on screen marked."""

    BODY_HEIGHT = 112

    def __init__(self, layer_count: int, current: int = 0, *, tokens=None,
                 caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._count, self._current = max(1, int(layer_count)), int(current)

    def _lines(self) -> list[str]:
        return [_count(self._count, 'layer'), f'layer {self._current} is on screen',
                'each layer has its own learned parameters']

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        shown = min(self._count, 18)
        step = rect.width() / shown
        body = QRect(rect.x(), rect.y() + line + 2, rect.width(),
                     max(20, rect.height() - 3 * line - 6))
        marked = min(int(round(self._current * (shown - 1) / max(1, self._count - 1))),
                     shown - 1) if shown > 1 else 0
        for index in range(shown):
            box = QRect(int(rect.x() + index * step) + 1, body.y(),
                        max(4, int(step) - 3), body.height())
            if index == marked:
                self.outline(painter, box, self.accent(), radius=2, width=1.5,
                             fill=self.wash(self.accent(), 110))
            else:
                self.outline(painter, box, self.faint(), radius=2,
                             fill=self.wash(self.role('layer'), 26))
        self.write(painter, QRect(rect.x(), rect.y(), rect.width(), line),
                   _count(self._count, 'layer'), colour=self.role('layer'), points=9,
                   bold=True)
        self.write(painter, QRect(rect.x(), body.bottom() + 1, rect.width(), line),
                   f'layer {self._current} is on screen', colour=self.accent(), points=8,
                   bold=True)
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   'each layer has its own learned parameters', colour=self.muted())

    def describe(self) -> list[str]:
        return self._lines() + super().describe()


class TrainingVsInference(Diagram):
    """Where the weights come from, and that running the model does not change them."""

    BODY_HEIGHT = 136

    STAGES = ('text', 'predict the next token', 'measure the error',
              'nudge every weight', 'frozen weights', 'this prompt', 'this answer')

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        half = max(26, (rect.height() - 3 * line - 10) // 2)
        learn, run = self.role('norm'), self.role('output')
        for band, (title, stages, colour) in enumerate((
                ('TRAINING · writes the weights', self.STAGES[:4], learn),
                ('INFERENCE · only reads them', self.STAGES[4:], run))):
            top = rect.y() + line + band * (half + line + 6)
            self.write(painter, QRect(rect.x(), top - line, rect.width(), line), title,
                       colour=colour, points=8, bold=True)
            width = max(50, (rect.width() - 24 * len(stages)) // max(1, len(stages)))
            for index, stage in enumerate(stages):
                box = QRect(rect.x() + index * (width + 24), top, width, half)
                self.outline(painter, box, colour, radius=4, width=1.2,
                             fill=self.wash(colour, 24))
                self.write(painter, box, stage, colour=colour, points=7,
                           align=Qt.AlignCenter)
                if index < len(stages) - 1:
                    self.arrow(painter, box.right() + 3, top + half // 2,
                               box.right() + 21, colour)
            if band == 0:
                # the loop arrow, back to the start: training repeats, inference does not
                painter.save()
                painter.setPen(QPen(colour, 1.1, Qt.DashLine))
                path = QPainterPath()
                path.moveTo(rect.x() + 3 * (width + 24) + width // 2, top + half)
                path.cubicTo(rect.x() + 2 * width, top + half + 16,
                             rect.x() + width // 2, top + half + 16,
                             rect.x() + width // 2, top + half)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
                painter.restore()
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   'TensorScope captured the lower row. No weight changed during it.',
                   colour=self.muted())

    def describe(self) -> list[str]:
        return ['TRAINING · writes the weights', 'INFERENCE · only reads them']             + list(self.STAGES)             + ['TensorScope captured the lower row. No weight changed during it.']             + super().describe()


class StoppingLoop(Diagram):
    """The generation loop, and the one exit from it."""

    BODY_HEIGHT = 128

    STAGES = ('score every entry', 'apply the selection rule', 'append the token')

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        colour = self.role('scores')
        stop = self.role('output')
        width = max(62, (rect.width() - 110 - 2 * 22) // 3)
        top = rect.y() + line + 4
        height = max(26, rect.height() - 3 * line - 14)
        boxes = []
        for index, stage in enumerate(self.STAGES):
            box = QRect(rect.x() + index * (width + 22), top, width, height)
            self.outline(painter, box, colour, radius=4, width=1.2,
                         fill=self.wash(colour, 24))
            self.write(painter, box, stage, colour=colour, points=7, align=Qt.AlignCenter)
            boxes.append(box)
            if index < len(self.STAGES) - 1:
                self.arrow(painter, box.right() + 3, top + height // 2,
                           box.right() + 19, colour)
        # the loop back to scoring
        painter.save()
        painter.setPen(QPen(colour, 1.1, Qt.DashLine))
        path = QPainterPath()
        path.moveTo(boxes[-1].center().x(), boxes[-1].bottom())
        path.cubicTo(boxes[-1].center().x(), boxes[-1].bottom() + 18,
                     boxes[0].center().x(), boxes[0].bottom() + 18,
                     boxes[0].center().x(), boxes[0].bottom())
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.restore()
        self.write(painter, QRect(boxes[0].x(), boxes[0].bottom() + 18, rect.width(), line),
                   'the next step reuses the same weights and the cached keys and values',
                   colour=self.muted(), points=7)
        exit_box = QRect(boxes[-1].right() + 28, top, max(72, rect.right() - boxes[-1].right() - 30),
                         height)
        self.arrow(painter, boxes[-1].right() + 6, top + height // 2, exit_box.x() - 4, stop)
        self.outline(painter, exit_box, stop, radius=4, width=1.4,
                     fill=self.wash(stop, 30))
        self.write(painter, exit_box, 'the stopping entry won', colour=stop,
                   points=7, align=Qt.AlignCenter)
        self.write(painter, QRect(exit_box.x(), rect.bottom() - line, exit_box.width(), line),
                   'not appended', colour=stop, points=7, align=Qt.AlignCenter)

    def describe(self) -> list[str]:
        return list(self.STAGES) + [
            'the next step reuses the same weights and the cached keys and values',
            'the stopping entry won', 'not appended'] + super().describe()


# Every concrete diagram, for the gallery below and for the test that paints all of them.
DIAGRAMS = (VectorStrip, MatrixGrid, EmbeddingLookupRow, MatMulShapes, HeadSplit, KVShare,
            QKCompare, CausalMaskTriangle, SoftmaxFlow, SoftmaxBars, AttentionGrid,
            ResidualBypass, GatedFFNPath, VocabularyProjection, LayerStack,
            TrainingVsInference, StoppingLoop)


def gallery(tokens: dict | None = None) -> list[Diagram]:
    """One of each diagram, built from literals shaped like the Qwen3-4B runs.

    Used by `__main__` to eyeball both themes and by the test suite to paint every widget
    without a capture.  The numbers here are arguments, not captured values.
    """
    scores = np.array([2.5, 3.1, 1.2, 4.8, 0.4, -1.1, 2.2, -3.4e38, -3.4e38], dtype=np.float32)
    weights = np.array([0.12, 0.2, 0.05, 0.55, 0.02, 0.01, 0.05, 0.0, 0.0], dtype=np.float32)
    matrix = np.triu(np.ones((15, 15), dtype=np.float32)).T * np.linspace(0.1, 1.0, 15)
    return [
        VectorStrip(2560, symbol='X', role='norm', highlight=3, tokens=tokens,
                    caption='One token position, 2,560 features.'),
        MatrixGrid(15, 2560, role='layer', symbol='X', highlight_row=2, clickable=True,
                   tokens=tokens, caption='Token positions down, features across.'),
        EmbeddingLookupRow(151936, 2560, token_id=38490, token_text='favorite',
                           tokens=tokens, caption='One row of the table, not a square.'),
        MatMulShapes((15, 2560), (4096, 2560), (15, 4096), left_symbol='X_norm',
                     right_symbol='W_Qᵀ', out_symbol='Q_raw', tokens=tokens,
                     caption='The shared 2,560 is why this is defined.'),
        HeadSplit(2560, 4096, 32, 128, 15, highlight_head=1, highlight_row=2, tokens=tokens,
                  caption='Projection computes; reshaping only regroups.'),
        KVShare(8, 32, tokens=tokens, caption='Grouped-query attention.'),
        QKCompare(128, query_pos=14, key_pos=4, query_label='ĊĊ', key_label='Ġcolor',
                  tokens=tokens, caption='One query against one key.'),
        CausalMaskTriangle(15, 15, selected_query=3, clickable=True, tokens=tokens,
                           caption='Each query may read itself and earlier positions.'),
        SoftmaxFlow(tokens=tokens, caption='Deliberately numberless.'),
        SoftmaxBars(scores, weights, labels=[str(i) for i in range(9)],
                    masked_indices=(7, 8), selected=3, tokens=tokens,
                    caption='Captured scores above, captured weights below.'),
        AttentionGrid(matrix, selected_query=3, tokens=tokens,
                      caption='Painted, so no matplotlib canvas is created.'),
        ResidualBypass([('attention block', 'O', 'H'), ('feed-forward', 'F', 'X_next')],
                       trunk_symbols=('X', 'H'), tokens=tokens,
                       caption='The input travels around each branch.'),
        GatedFFNPath(2560, None, tokens=tokens,
                     caption='Qwen3 gates its feed-forward path.'),
        VocabularyProjection(2560, 151936, positions=15, final_position=14, marked=10,
                             tokens=tokens, caption='Only the final row was stored.'),
        LayerStack(36, 17, tokens=tokens, caption='One of thirty-six.'),
        TrainingVsInference(tokens=tokens, caption='Weights are written once, then read.'),
        StoppingLoop(tokens=tokens, caption='One loop, one exit.'),
    ]


if __name__ == '__main__':   # pragma: no cover - a visual check, not a test
    import sys
    from PyQt5.QtWidgets import (
        QApplication, QGridLayout, QHBoxLayout, QScrollArea, QWidget as _W,
    )

    _DARK = {'bg': '#111114', 'card_bg': '#1c1c20', 'border': '#2e2e33',
             'text_primary': '#f0f0f2', 'text_secondary': '#9f9fa8', 'text_muted': '#636370'}
    _LIGHT = {'bg': '#f5f5f7', 'card_bg': '#ffffff', 'border': '#d8d8de',
              'text_primary': '#111114', 'text_secondary': '#52525c', 'text_muted': '#8e8e9a'}

    application = QApplication(sys.argv)
    window = _W()
    window.setWindowTitle('TensorScope diagram gallery')
    window.resize(1500, 950)
    row = QHBoxLayout(window)
    for name, theme in (('dark', _DARK), ('light', _LIGHT)):
        panel = _W()
        panel.setStyleSheet(f"background: {theme['bg']}; color: {theme['text_primary']};")
        grid = QGridLayout(panel)
        grid.setSpacing(14)
        for position, diagram in enumerate(gallery(theme)):
            holder = _W()
            holder.setStyleSheet(f"background: {theme['card_bg']}; "
                                 f"border: 1px solid {theme['border']}; border-radius: 8px;")
            inner = QVBoxLayout(holder)
            inner.setContentsMargins(12, 10, 12, 10)
            inner.addWidget(label(type(diagram).__name__, small=True))
            inner.addWidget(diagram)
            grid.addWidget(holder, position // 3, position % 3)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(panel)
        row.addWidget(area)
    window.show()
    sys.exit(application.exec_())
