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
from PyQt5.QtCore import QRect, QRectF, QSize, Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget, QToolTip

from tensorscope_common import CHROME, COLORS, OFFSCREEN, is_light, label, role_colors


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
        self.setMinimumHeight(diagram.BODY_HEIGHT + 2 * diagram.PAD)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        pad = 2 * self._diagram.PAD
        return QSize(self._diagram.BODY_WIDTH + pad, self._diagram.BODY_HEIGHT + pad)

    def drawing_rect(self) -> QRect:
        """Where the sketch is drawn: inside the figure's padding, no wider than the
        diagram's own `MAX_WIDTH`, centred.  A sketch of a shape has a natural width;
        stretched across a wide window a 15 x 15 grid stops reading as a square."""
        pad = self._diagram.PAD
        inner = self.rect().adjusted(pad, pad, -pad, -pad)
        width = min(inner.width(), self._diagram.MAX_WIDTH)
        left = inner.x() + max(0, (inner.width() - width) // 2)
        return QRect(left, inner.y(), width, inner.height()).adjusted(1, 1, -2, -2)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        try:
            if self._diagram.PAD:
                # The figure: one raised panel per picture, so a diagram reads as a
                # figure in a textbook rather than as lines floating among the prose.
                painter.save()
                painter.setPen(QPen(self._diagram.faint(), 1.0))
                painter.setBrush(self._diagram.surface())
                painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5),
                                        8, 8)
                painter.restore()
            self._diagram.draw(painter, self.drawing_rect())
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:
        if not self._diagram.handle_click(event.pos()):
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        text = self._diagram.tooltip_at(event.pos())
        QToolTip.showText(event.globalPos(), text, self) if text else QToolTip.hideText()

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()


class Diagram(QWidget):
    """Base class: theme-aware colours, a drawn body, and an optional caption below it.

    Subclasses override `draw(painter, rect)` and `describe()`.  Colours come from the
    host's design tokens first and the widget palette second, so the same widget is legible
    on the light and the dark palette with no branching in `draw`.
    """

    HIGHLIGHT_MS = 180
    BODY_HEIGHT = 120
    BODY_WIDTH = 360
    # The widest the sketch itself is drawn, and the figure padding around it.
    MAX_WIDTH = 760
    PAD = 14

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

    def set_body_height(self, height: int) -> None:
        """Give this instance a drawing height of its own -- for a sketch whose natural
        height depends on what it draws, such as a grid with one row per position."""
        self.BODY_HEIGHT = int(height)
        self.canvas.setMinimumHeight(self.BODY_HEIGHT + 2 * self.PAD)
        self.canvas.updateGeometry()

    # ── Colours ──────────────────────────────────────────────────────────────

    def _token(self, name: str, fallback: str) -> QColor:
        return QColor(self._tokens.get(name) or fallback)

    def ink(self) -> QColor:
        """Primary text and outline colour."""
        if self._tokens.get('text_primary'):
            return QColor(self._tokens['text_primary'])
        return self.palette().windowText().color()

    def muted(self) -> QColor:
        """Secondary text.  The page's secondary ink rather than its faintest grey: a
        diagram's labels are part of what it says, and the muted token sits at about 3:1
        on the figure surface."""
        return self._token('text_secondary', self._token('text_muted', CHROME['muted']).name())

    def faint(self) -> QColor:
        """Outlines of cells that are drawn but not selected: a third of the way from the
        border colour towards the secondary ink, so an empty grid still reads as a grid
        on the raised figure surface."""
        border = self._token('border', CHROME['ahead'])
        toward = self.muted()
        mix = 0.33
        return QColor(round(border.red() + (toward.red() - border.red()) * mix),
                      round(border.green() + (toward.green() - border.green()) * mix),
                      round(border.blue() + (toward.blue() - border.blue()) * mix))

    def surface(self) -> QColor:
        return self._token('card_bg', '#1c1c20')

    def accent(self) -> QColor:
        return QColor(CHROME['accent'])

    def light(self) -> bool:
        return is_light(self._tokens)

    def role(self, key: str) -> QColor:
        """A role's hue for this palette: one step darker on the light palette, where the
        dark-palette hues are too faint to carry a label."""
        return QColor(role_colors(self._tokens).get(key, CHROME['muted']))

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

    # ── Which rows to draw ───────────────────────────────────────────────────

    @staticmethod
    def row_window(total: int, budget: int, *marks) -> list:
        """The real rows a sketch draws, with `None` where a run of rows is skipped.

        Always the first and the last row, every marked row and its neighbours, then
        rows nearest the top until the budget is spent.  The old rule drew the first few
        rows and clamped a selection onto the last drawn one, so position 13 of 15 was
        painted on the row that stood for position 14 and labelled "row 13".  Here a drawn
        row *is* a real row, and its label is its own index.
        """
        total = max(0, int(total))
        budget = max(3, int(budget))
        if total <= budget:
            return list(range(total))
        keep = {0, total - 1}
        for mark in marks:
            if mark is None:
                continue
            centre = max(0, min(int(mark), total - 1))
            keep.update(index for index in (centre - 1, centre, centre + 1)
                        if 0 <= index < total)

        def slots(rows) -> list:
            out, previous = [], None
            for index in sorted(rows):
                if previous is not None and index != previous + 1:
                    out.append(None)
                out.append(index)
                previous = index
            return out

        for candidate in range(1, total - 1):
            if len(slots(keep | {candidate})) > budget:
                break
            keep.add(candidate)
        return slots(keep)

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

    def tooltip_at(self, point) -> str:
        """The exact text a hover at `point` shows, or '' for none.

        Kept separate from painting so the tests can read the value a reader sees without
        reading pixels.  Default: no hover text.
        """
        return ''

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
        shown_cols = min(self._cols, self._drawn_cols)
        return len(self.drawn_rows()), shown_cols, False, self._cols > shown_cols

    def drawn_rows(self) -> list:
        """The real row behind each drawn row, `None` for a skipped run."""
        return self.row_window(self._rows, self._drawn_rows, self._selected_row)

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        header = QRect(rect.x(), rect.y(), rect.width(), line)
        if self._symbol:
            self.write(painter, header, f'{self._symbol}   [{_number(self._rows)} × '
                                        f'{_number(self._cols)}]',
                       colour=self.role(self._role), points=9, bold=True)
        # Wide enough for the axis name and the "row N" marker, whatever the font.
        metrics = QFontMetrics(self.font_at(8, bold=True))
        side = max(74, metrics.horizontalAdvance(self._row_axis) + 12,
                   metrics.horizontalAdvance(f'row {self._rows}') + 12)
        self._grid = QRect(rect.x() + side, rect.y() + line + 4,
                           max(40, rect.width() - side - 4),
                           max(30, rect.height() - 2 * line - 12))
        # The row axis is named on the bottom line, under the row labels, so it can never
        # sit on top of the "row N" marker beside the followed row.
        self.write(painter, QRect(rect.x(), rect.bottom() - line, side - 6, line),
                   self._row_axis, colour=self.muted(),
                   align=Qt.AlignRight | Qt.AlignVCenter)
        self.write(painter, QRect(self._grid.x(), rect.bottom() - line,
                                  self._grid.width(), line),
                   f'{self._col_axis}  →  {_number(self._cols)}', colour=self.muted())

        shown_rows, shown_cols, more_rows, more_cols = self._drawn()
        window = self.drawn_rows()
        colour = self.role(self._role)
        cell_h = self._grid.height() / max(1, shown_rows)
        cell_w = self._grid.width() / max(1, shown_cols)
        for row in range(shown_rows):
            real = window[row]
            gap_row = real is None
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
                picked_row = self._selected_row is not None and real == self._selected_row
                picked_col = self._highlight_col is not None and col == min(
                    self._highlight_col, shown_cols - 1)
                alpha = 96 if (picked_row and picked_col) else 58 if picked_row else \
                    48 if picked_col else 18
                self.outline(painter, box, colour if (picked_row or picked_col) else self.faint(),
                             radius=1, width=1.4 if picked_row else 1.0,
                             fill=self.wash(colour, alpha))
        if self._selected_row is not None and self._selected_row in window:
            marker = QRect(rect.x(), int(self._grid.y() + window.index(self._selected_row)
                                         * cell_h), side - 8, max(10, int(cell_h)))
            self.write(painter, marker, f'row {self._selected_row}', colour=self.accent(),
                       points=8, bold=True, align=Qt.AlignRight | Qt.AlignVCenter)

    def handle_click(self, point) -> bool:
        if not self._clickable or not self._grid.contains(point):
            return False
        shown_rows, shown_cols, _, _ = self._drawn()
        window = self.drawn_rows()
        col = int((point.x() - self._grid.x()) / (self._grid.width() / max(1, shown_cols)))
        row = int((point.y() - self._grid.y()) / (self._grid.height() / max(1, shown_rows)))
        real = window[max(0, min(row, shown_rows - 1))]
        if real is None:                    # the skipped run: nothing real to select
            return True
        self._selected_row = real
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

    def drawn_rows(self) -> list:
        return self.row_window(self._positions, 5, self._selected_row)

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        window = self.drawn_rows()
        rows = max(1, len(window))
        band = max(8, (rect.height() - 3 * line - 16) // max(1, rows))
        column = max(58, (rect.width() - 100) // 3)
        xs = [rect.x(), rect.x() + column + 44, rect.x() + 2 * column + 88]
        colour = self.role(self._role)

        for stage, x in enumerate(xs):
            for row in range(rows):
                top = rect.y() + line + 4 + row * band
                box = QRect(x, top, column, max(6, band - 3))
                if window[row] is None:
                    self.write(painter, box, ELLIPSIS, colour=self.muted(),
                               align=Qt.AlignCenter)
                    continue
                picked = self._selected_row is not None and window[row] == self._selected_row
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
        if self._selected_row is not None and self._selected_row in window:
            top = rect.y() + line + 4 + window.index(self._selected_row) * band
            self.write(painter, QRect(xs[2] + column + 4, top, 64, max(6, band - 3)),
                       f'row {self._selected_row}', colour=self.accent(), points=7,
                       bold=True)
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   f'{_number(self._positions)} token positions stay separate rows '
                   'through both stages', colour=self.muted())

    def describe(self) -> list[str]:
        followed = ([f'row {self._selected_row}'] if self._selected_row is not None
                    and self._selected_row in self.drawn_rows() else [])
        return self._stage_labels() + followed + super().describe()


class KVShare(Diagram):
    """Grouped-query attention: a few key/value heads serving many query heads."""

    BODY_HEIGHT = 120

    def __init__(self, kv_heads: int, query_heads: int, *, highlight_query_head=None,
                 tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._kv, self._queries = max(1, int(kv_heads)), max(1, int(query_heads))
        self._head = (None if highlight_query_head is None
                      else max(0, min(int(highlight_query_head), self._queries - 1)))

    def _group(self) -> int:
        return max(1, self._queries // self._kv)

    def _group_text(self) -> str:
        return (f'{_count(self._kv, "key/value head")} → '
                f'{_count(self._queries, "query head")} '
                f'({self._group()} query heads share each one)')

    def _reads_text(self) -> str:
        return (f'query head {self._head} reads key/value head {self._head // self._group()}'
                if self._head is not None else '')

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
        # The drawn boxes stand for evenly spaced real heads, so the highlighted pair is
        # the drawn query box nearest the real head and the box of the head it reads; the
        # exact numbers are written underneath rather than left to the picture.
        hot_q = (None if self._head is None
                 else min(shown_q - 1, self._head * shown_q // self._queries))
        hot_kv = (None if self._head is None
                  else min(shown_kv - 1, (self._head // self._group()) * shown_kv // self._kv))
        for index in range(shown_kv):
            box = QRect(left.x(), int(left.y() + index * kv_step) + 1, left.width(),
                        max(6, int(kv_step) - 3))
            hot = index == hot_kv
            self.outline(painter, box, self.accent() if hot else self.role('k'), radius=2,
                         width=1.6 if hot else 1.0,
                         fill=self.wash(self.role('k'), 90 if hot else 40))
        for index in range(shown_q):
            box = QRect(right.x(), int(right.y() + index * q_step) + 1, right.width(),
                        max(5, int(q_step) - 3))
            hot = index == hot_q
            self.outline(painter, box, self.accent() if hot else self.role('q'), radius=2,
                         width=1.6 if hot else 1.0,
                         fill=self.wash(self.role('q'), 90 if hot else 34))
        painter.save()
        per_group = max(1, shown_q // shown_kv)
        for index in range(shown_q):
            source = min(index // per_group, shown_kv - 1)
            hot = index == hot_q
            painter.setPen(QPen(self.accent() if hot else self.muted(), 1.8 if hot else 1.0))
            y1 = left.y() + source * kv_step + kv_step / 2
            y2 = right.y() + index * q_step + q_step / 2
            painter.drawLine(QRectF(left.right() + 2, y1, 0, 0).topLeft(),
                             QRectF(right.x() - 2, y2, 0, 0).topLeft())
        painter.restore()
        bottom = 'copied, not recomputed'
        if self._head is not None:
            bottom = f'{self._reads_text()}  ·  {bottom}'
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   bottom, colour=self.muted(), align=Qt.AlignCenter)

    def describe(self) -> list[str]:
        reads = [self._reads_text()] if self._head is not None else []
        return ['K / V heads', 'Q heads', self._group_text()] + reads \
            + ['copied, not recomputed'] + super().describe()


class QKCompare(Diagram):
    """One query vector against one key vector, with the shared width bracketed."""

    BODY_HEIGHT = 104

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
        # Level with the gap between the two rows it combines.
        middle = rect.y() + line + strip_h + (line + 6) // 2
        out = QRect(rect.right() - 92, middle - 16, 40, 32)
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
        # Tall enough for square cells: a one-query pass is a single row, a 15-position
        # prefill is a square.
        shown_q, _ = self._drawn_counts()
        line = self.text_height(8)
        self.set_body_height(max(108, shown_q * 15 + 3 * line + 20))

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

    # Every position is drawn when the prompt is short enough for square cells to stay
    # readable -- which covers every saved run -- so the triangle a reader sees is the
    # real causal boundary, row for row.  Longer prompts fall back to a sample.
    MAX_QUERY_ROWS = 16
    MAX_KEY_COLUMNS = 24
    MAX_CELL = 20

    def _drawn_counts(self) -> tuple[int, int]:
        return min(self._queries, self.MAX_QUERY_ROWS), min(self._keys, self.MAX_KEY_COLUMNS)

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
        shown_q, shown_k = self._drawn_counts()
        side = 58
        top = rect.y() + line + 8
        legend_width = 200 if rect.width() >= side + 160 + 200 else 0
        cell = max(4.0, min((rect.width() - side - legend_width - 24) / shown_k,
                            (rect.bottom() - 2 * line - 6 - top) / shown_q,
                            float(self.MAX_CELL)))
        # Centre the grid and its key as one block inside the figure.
        content = side + int(cell * shown_k) + (24 + legend_width if legend_width else 0)
        rect = QRect(rect.x() + max(0, (rect.width() - content) // 2), rect.y(),
                     min(rect.width(), content), rect.height())
        self._grid = QRect(rect.x() + side, top, int(cell * shown_k), int(cell * shown_q))
        self.write(painter, QRect(self._grid.x(), rect.y(), rect.width() - side, line),
                   self._shape_text(), colour=self.role('scores'), points=9, bold=True)
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
                picked = picked_row is not None and row == picked_row
                cell_hit = picked and picked_col is not None and col == picked_col
                self._mask_cell(painter, box, may_read, picked, cell_hit, allowed)
        # Row numbers when there is room for them, the followed row always.
        label_box = QRect(rect.x(), 0, side - 8, int(cell_h))
        small = cell_h >= 11
        for row in range(shown_q):
            real_q = self._to_real(row, shown_q, self._queries)
            mine = picked_row is not None and row == picked_row
            if not (small or mine):
                continue
            box = label_box.translated(0, int(self._grid.y() + row * cell_h))
            self.write(painter, box, (f'query {real_q}' if mine else str(real_q)),
                       colour=self.accent() if mine else self.muted(),
                       points=8 if mine else 7, bold=mine,
                       align=Qt.AlignRight | Qt.AlignVCenter)
        if picked_col is not None:
            real_k = self._to_real(picked_col, shown_k, self._keys)
            x = int(self._grid.x() + picked_col * cell_w)
            self.write(painter, QRect(x - 20, self._grid.bottom() + 2, int(cell_w) + 40, line),
                       f'key {real_k}', colour=self.accent(), points=7, bold=True,
                       align=Qt.AlignHCenter | Qt.AlignTop)
        if self._cached:
            band = QRect(self._grid.x(), self._grid.y() - 4,
                         int(cell_w * min(shown_k, max(1, round(
                             self._cached * shown_k / self._keys)))), 2)
            self.outline(painter, band, self.role('k'), radius=1, width=1.0,
                         fill=self.wash(self.role('k'), 170))
        self.write(painter, QRect(self._grid.x(), rect.bottom() - line,
                                  max(self._grid.width(), 120), line),
                   'key position  →', colour=self.muted())
        if not small and picked_row is None:
            self.write(painter, QRect(rect.x(), self._grid.y(), side - 8, self._grid.height()),
                       'query', colour=self.muted(), align=Qt.AlignRight | Qt.AlignVCenter)
        # The key, beside the grid: what each kind of cell means, in words.
        if legend_width:
            x = self._grid.right() + 24
            y = self._grid.y()
            for may_read, name in ((True, 'allowed'), (False, 'blocked')):
                self._mask_cell(painter, QRect(x, y + 2, 14, 14), may_read, False, False,
                                allowed)
                self.write(painter, QRect(x + 22, y, legend_width - 22, 18), name,
                           colour=self.ink(), points=8, bold=True)
                y += 22
            notes = [self._mask_text()]
            if self._cached:
                notes.insert(0, f'{_count(self._cached, "key")} came from the cache')
            for note in notes:
                height = max(line, rect.bottom() - y - line)
                painter.save()
                painter.setFont(self.font_at(7))
                bounds = painter.boundingRect(QRectF(x, y + 4, legend_width, height),
                                              int(Qt.TextWordWrap), note)
                painter.restore()
                self.write(painter, QRectF(x, y + 4, legend_width, bounds.height()), note,
                           colour=self.role('k') if 'cache' in note else self.muted(),
                           points=7, align=Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap)
                y += int(bounds.height()) + 6
        else:
            self.write(painter, QRect(rect.x(), rect.bottom() - line, side - 8, line),
                       'allowed / blocked', colour=self.muted(),
                       align=Qt.AlignRight | Qt.AlignVCenter)

    def _mask_cell(self, painter, box, may_read: bool, picked: bool, cell: bool,
                   allowed: QColor) -> None:
        self.outline(painter, box, allowed if may_read else self.faint(), radius=2,
                     width=1.8 if cell else 1.3 if picked else 1.0,
                     fill=self.wash(allowed, 150 if (may_read and cell) else
                                    95 if (may_read and picked) else
                                    55 if may_read else 0))
        if not may_read:
            painter.save()
            painter.setPen(QPen(self.faint(), 0.8))
            painter.drawLine(QRectF(box.x(), box.bottom(), 0, 0).topLeft(),
                             QRectF(box.right(), box.y(), 0, 0).topLeft())
            painter.restore()

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
        return [self._shape_text(), 'query', 'key position', 'allowed', 'blocked',
                self._mask_text()] + extra + super().describe()


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

    BODY_HEIGHT = 214
    MAX_COLUMN = 76

    def __init__(self, scores, weights, *, labels=None, masked_indices=(),
                 selected: int | None = None, drawn: int = DRAWN_CELLS,
                 indices=None, sublabels=None,
                 tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._scores = np.asarray(scores).reshape(-1)
        self._weights = np.asarray(weights).reshape(-1)
        self._masked = set(int(index) for index in masked_indices)
        self._selected_row = selected
        self._drawn = drawn
        size = int(min(self._scores.size, self._weights.size))
        # `indices` are the real key positions to draw, chosen by the caller -- the ones
        # that carry the weight, say -- and drawn in position order.  Without them, the
        # first `drawn` keys, as before.
        if indices is None:
            self._indices = list(range(min(drawn, size)))
        else:
            self._indices = sorted({int(i) for i in indices if 0 <= int(i) < size})
        # `labels` / `sublabels` are per real key position (index into the full row).
        self._labels = list(labels or [])
        self._sublabels = list(sublabels or [])

    def _shown(self) -> int:
        return len(self._indices)

    def _label_for(self, real: int) -> str:
        return str(self._labels[real]) if real < len(self._labels) else str(real)

    def _sublabel_for(self, real: int) -> str:
        return str(self._sublabels[real]) if real < len(self._sublabels) else ''

    def drawn_keys(self) -> list:
        return list(self._indices)

    def _rows(self) -> list[str]:
        return ['captured scores · already scaled and masked',
                'captured weights · the softmax output the model used']

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        shown = self._shown()
        if shown == 0:
            self.write(painter, rect, 'no values to draw', colour=self.muted())
            return
        # A bar per key, at a width that still reads as a bar: five keys spread over a
        # wide window had become 250-pixel slabs.
        column = min(float(self.MAX_COLUMN), rect.width() / shown)
        # Centre the chart as one block: its bars, or its band titles if those are wider.
        metrics = QFontMetrics(self.font_at(8, bold=True))
        content = int(max(column * shown,
                          max(metrics.horizontalAdvance(name) for name in self._rows()) + 4))
        content = min(rect.width(), content)
        rect = QRect(rect.x() + (rect.width() - content) // 2, rect.y(), content,
                     rect.height())
        finite = self._scores[np.isfinite(self._scores)]
        # Masked entries are a large negative additive value, so including them in the
        # range would flatten every real score into one pixel.  They are still drawn and
        # still labelled -- excluded from the *scale* only, exactly as the attention table
        # model does.
        visible = np.array([float(self._scores[index]) for index in self._indices
                            if index not in self._masked
                            and np.isfinite(self._scores[index])]) if finite.size else finite
        low = float(visible.min()) if visible.size else 0.0
        high = float(visible.max()) if visible.size else 1.0
        span = (high - low) or 1.0
        weight_high = (float(max(self._weights[index] for index in self._indices))
                       if self._indices else 1.0)
        weight_high = weight_high or 1.0

        has_sub = any(self._sublabel_for(real) for real in self._indices)
        footer = (2 if has_sub else 1) * line + 4
        band_height = (rect.height() - footer) // 2
        for band, (name, colour) in enumerate(
                zip(self._rows(), (self.role('scores'), self.role('weights')))):
            top = rect.y() + band * band_height
            self.write(painter, QRect(rect.x(), top, rect.width(), line), name,
                       colour=colour, points=8, bold=True)
            plot = QRect(rect.x(), top + line + 3, int(column * shown),
                         max(16, band_height - 2 * line - 8))
            painter.save()
            painter.setPen(QPen(self.faint(), 1.0))
            painter.drawLine(QRectF(plot.x(), plot.bottom() + 0.5, 0, 0).topLeft(),
                             QRectF(plot.right(), plot.bottom() + 0.5, 0, 0).topLeft())
            painter.restore()
            for index in range(shown):
                real = self._indices[index]
                value = float(self._scores[real] if band == 0 else self._weights[real])
                fraction = ((value - low) / span) if band == 0 else (value / weight_high)
                fraction = max(0.04, min(1.0, fraction))
                height = max(3, int(plot.height() * fraction))
                x = int(rect.x() + index * column)
                bar = QRect(x + 4, plot.bottom() - height, max(4, int(column) - 8), height)
                masked = real in self._masked
                picked = self._selected_row is not None and real == self._selected_row
                self.outline(painter, bar, self.accent() if picked else
                             (colour if not masked else self.faint()), radius=2,
                             width=1.8 if picked else 1.0,
                             fill=self.wash(self.accent() if picked else colour,
                                            0 if masked else 150 if picked else 110))
                if masked:
                    painter.save()
                    painter.setPen(QPen(self.faint(), 0.8, Qt.DashLine))
                    painter.drawLine(QRectF(bar.x(), bar.bottom(), 0, 0).topLeft(),
                                     QRectF(bar.right(), bar.y(), 0, 0).topLeft())
                    painter.restore()
                self.write(painter, QRect(x, plot.bottom() + 3, int(column), line),
                           f'{value:.3g}', colour=self.ink() if picked else self.muted(),
                           points=7, bold=picked, align=Qt.AlignHCenter | Qt.AlignTop)
        for index in range(shown):
            real = self._indices[index]
            cell = QRect(int(rect.x() + index * column),
                         rect.bottom() - (2 * line if has_sub else line),
                         int(column), line)
            self.write(painter, cell, self._label_for(real)[:6], colour=self.muted(),
                       points=7, align=Qt.AlignCenter)
            if has_sub:
                self.write(painter, cell.translated(0, line), self._sublabel_for(real)[:7],
                           colour=self.ink(), points=7, align=Qt.AlignCenter)

    def describe(self) -> list[str]:
        shown = self._shown()
        spelled = [text for text in (self._sublabel_for(real) for real in self._indices)
                   if text.strip()]
        return self._rows() + [f'{shown} of {self._scores.size} key positions drawn'] \
            + spelled + super().describe()


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
        window = self.row_window(rows, 24, self._selected_row)
        shown_r, shown_c = len(window), min(cols, 32)
        grid = QRect(rect.x() + 40, rect.y() + line + 2, max(40, rect.width() - 44),
                     max(24, rect.height() - 2 * line - 8))
        cell_w, cell_h = grid.width() / shown_c, grid.height() / shown_r
        finite = self._matrix[np.isfinite(self._matrix)]
        low = float(finite.min()) if finite.size else 0.0
        high = float(finite.max()) if finite.size else 1.0
        span = (high - low) or 1.0
        base = self.role('weights')
        for row in range(shown_r):
            if window[row] is None:
                continue
            for col in range(shown_c):
                value = float(self._matrix[window[row], col])
                strength = 0.0 if not np.isfinite(value) else (value - low) / span
                box = QRect(int(grid.x() + col * cell_w), int(grid.y() + row * cell_h),
                            max(2, int(cell_w)), max(2, int(cell_h)))
                painter.fillRect(box, self.wash(base, int(18 + 210 * max(0.0, min(1.0, strength)))))
        self.outline(painter, grid, self.faint(), radius=2)
        if self._selected_row is not None and self._selected_row in window:
            top = int(grid.y() + window.index(self._selected_row) * cell_h)
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

    def __init__(self, branches, *, trunk_symbols=('X',), active: int | None = None,
                 tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        # branches: a sequence of (branch_label, output_symbol, result_symbol)
        self._branches = [tuple(item) for item in branches]
        self._trunk = list(trunk_symbols)
        # The branch the surrounding operation is about.  The other stays drawn, faded, so
        # the reader sees where this step sits in the whole layer.
        self._active = active

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
            if self._active is not None and index != self._active:
                colour = self.faint()
            hot = self._active is not None and index == self._active
            self.outline(painter, box, colour, radius=4, width=2.0 if hot else 1.3,
                         fill=self.wash(colour, 60 if hot else 26))
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
                 marked: int = 0, focus_position: int | None = None, tokens=None,
                 caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._hidden, self._vocab = int(hidden), int(vocab)
        self._positions, self._final = int(positions), int(final_position)
        self._marked = int(marked)
        # The position the reader has been following.  When it is not the final one the
        # picture says so: that row was computed too, and it is not the row that chose.
        self._focus = None if focus_position is None else int(focus_position)

    def drawn_rows(self) -> list:
        return self.row_window(self._positions, 7, self._focus, self._final)

    def _followed_text(self) -> str:
        if self._focus is None or self._focus == self._final:
            return ''
        return f'you followed position {self._focus}; it did not choose'


    def _scores_label(self) -> str:
        return (f'{_number(self._vocab)} scores, one per vocabulary entry' if self._vocab
                else 'one score per vocabulary entry (count not recorded in this run)')

    def _lines(self) -> list[str]:
        marked = [f'{_number(self._marked)} of them were recorded by name'] if self._marked else []
        followed = [self._followed_text()] if self._followed_text() else []
        return [f"{_count(self._positions, 'token position')}",
                f'only position {self._final} chose the next token',
                self._scores_label()] + marked + followed

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        window = self.drawn_rows() or [0]
        shown = len(window)
        column = 96
        rows = QRect(rect.x() + 30, rect.y() + line, column, rect.height() - 2 * line - 4)
        step = rows.height() / shown
        final_row = window.index(self._final) if self._final in window else shown - 1
        for index in range(shown):
            box = QRect(rows.x(), int(rows.y() + index * step) + 1, column,
                        max(5, int(step) - 3))
            real = window[index]
            if real is None:
                self.write(painter, box, ELLIPSIS, colour=self.muted(), align=Qt.AlignCenter)
                continue
            self.write(painter, QRect(rect.x(), box.y(), 26, box.height()), str(real),
                       colour=self.muted(), points=7, align=Qt.AlignRight | Qt.AlignVCenter)
            if index == final_row:
                self.outline(painter, box, self.accent(), radius=2, width=1.4,
                             fill=self.wash(self.accent(), 90))
            elif self._focus is not None and real == self._focus:
                self.outline(painter, box, self.ink(), radius=2, width=1.2, dashed=True,
                             fill=self.wash(self.muted(), 30))
            else:
                self.outline(painter, box, self.faint(), radius=2,
                             fill=self.wash(self.muted(), 16))
        self.write(painter, QRect(rows.x(), rect.y(), column + 40, line),
                   _count(self._positions, 'position'), colour=self.muted())
        mid = int(rows.y() + final_row * step + step / 2)
        self.arrow(painter, rows.right() + 4, mid, rows.right() + 40, self.accent())
        bar = QRect(rows.right() + 48, mid - 9, max(50, rect.right() - rows.right() - 52), 18)
        self.outline(painter, bar, self.role('scores'), radius=2, width=1.3,
                     fill=self.wash(self.role('scores'), 70))
        under = (f'{_number(self._vocab)} scores' if self._vocab
                 else 'score count not recorded')
        if self._marked:
            under += f'  ·  {_number(self._marked)} of them were recorded by name'
        self.write(painter, QRect(bar.x(), bar.bottom() + 2, bar.width(), line), under,
                   colour=self.muted())
        self.write(painter, QRect(bar.x(), bar.y() - line - 2, bar.width(), line),
                   f'position {self._final} only', colour=self.accent(), points=8, bold=True)
        if self._marked:
            width = max(6, bar.width() // 12)
            self.outline(painter, QRect(bar.x(), bar.y(), width, bar.height()),
                         self.accent(), radius=2, width=1.4,
                         fill=self.wash(self.accent(), 120))
        if self._followed_text():
            self.write(painter, QRect(rect.x(), rect.bottom() - line, column + 30, line),
                       f'dashed: position {self._focus}', colour=self.muted(), points=7)

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


class DecisionChain(Diagram):
    """From the final row to a token: captured steps solid, discarded steps dashed.

    Numberless on purpose.  The point is which links of the chain a capture holds -- the
    final layer output and the logits -- and which it does not: the final normalization,
    the vocabulary projection's weights.  The text says it as well as the line style.
    """

    BODY_HEIGHT = 92

    def __init__(self, stages, *, tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        # stages: a sequence of (label, captured: bool)
        self._stages = [(str(text), bool(kept)) for text, kept in stages]

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        count = max(1, len(self._stages))
        gap = 16
        width = max(58, (rect.width() - gap * (count - 1)) // count)
        top = rect.y() + 4
        height = max(28, rect.height() - line - 14)
        for index, (text, kept) in enumerate(self._stages):
            box = QRect(rect.x() + index * (width + gap), top, width, height)
            colour = self.role('scores') if kept else self.muted()
            self.outline(painter, box, colour, radius=4, width=1.3, dashed=not kept,
                         fill=self.wash(colour, 40 if kept else 0))
            self.write(painter, box.adjusted(4, 2, -4, -2), text, colour=self.ink() if kept
                       else self.muted(), points=7, align=Qt.AlignCenter | Qt.TextWordWrap)
            if index < count - 1:
                self.arrow(painter, box.right() + 2, top + height // 2,
                           box.right() + gap - 2, self.muted())
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   'solid: captured  ·  dashed: happened in the model, not kept',
                   colour=self.muted())

    def describe(self) -> list[str]:
        return ([text for text, _ in self._stages]
                + ['solid: captured  ·  dashed: happened in the model, not kept']
                + super().describe())


class PositionScope(Diagram):
    """What attention reads versus what the feed-forward network reads.

    Two panels of the same five rows.  On the left, arrows run from earlier rows into the
    followed row: attention gathers across positions.  On the right each row is processed
    in its own box with nothing crossing between rows.
    """

    BODY_HEIGHT = 124
    ROWS = 5

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        half = (rect.width() - 24) // 2
        rows = self.ROWS
        top = rect.y() + line + 4
        step = max(10, (rect.height() - 2 * line - 12) // rows)
        focus = rows - 2
        panels = (('attention: reads across positions', self.role('weights')),
                  ('feed-forward: one position at a time', self.role('norm')))
        for panel, (title, colour) in enumerate(panels):
            x = rect.x() + panel * (half + 24)
            self.write(painter, QRect(x, rect.y(), half, line), title, colour=colour,
                       points=8, bold=True)
            bar_w = max(40, half // 2)
            for row in range(rows):
                y = top + row * step
                box = QRect(x + 20, y, bar_w, max(6, step - 4))
                hot = row == focus
                self.outline(painter, box, colour if hot else self.faint(), radius=2,
                             width=1.5 if hot else 1.0,
                             fill=self.wash(colour, 70 if hot else 18))
                if panel == 1 and row <= focus:
                    loop = QRect(box.right() + 6, y, 18, max(6, step - 4))
                    self.outline(painter, loop, colour, radius=3, width=1.0,
                                 fill=self.wash(colour, 30 if hot else 10))
            if panel == 0:
                painter.save()
                target_y = top + focus * step + (step - 4) / 2
                for row in range(focus):
                    painter.setPen(QPen(colour, 1.1))
                    source_y = top + row * step + (step - 4) / 2
                    path = QPainterPath()
                    path.moveTo(x + 20 + bar_w + 2, source_y)
                    path.cubicTo(x + 20 + bar_w + 34, source_y, x + 20 + bar_w + 34,
                                 target_y, x + 20 + bar_w + 2, target_y)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path)
                painter.restore()
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   'the highlighted row is one position; later rows are never read',
                   colour=self.muted())

    def describe(self) -> list[str]:
        return ['attention: reads across positions', 'feed-forward: one position at a time',
                'the highlighted row is one position; later rows are never read'] \
            + super().describe()


# ── Level 0 and Level 1 ──────────────────────────────────────────────────────

class PipelineStrip(Diagram):
    """The run as a pipeline: the shape a token's data takes from prompt to answer.

    One box per stage, arrows between them, the real shape printed under each.  Each box is
    clickable and emits `stageClicked` with its key, so the notebook can jump the reader to
    the operation behind it.  The stages are `(key, title, shape_text)` -- the shape the
    data has at that point in this run.
    """

    BODY_HEIGHT = 120

    stageClicked = pyqtSignal(str)

    def __init__(self, stages, *, tokens=None, caption: str = '', parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        # stages: a sequence of (key, title, shape_text) or (key, title, shape_text,
        # discarded).  A discarded stage -- a shape the model computed and then threw away,
        # such as the full vocabulary projection -- is drawn with a dashed outline so the
        # reader sees it is not a value this capture kept.
        self._stages = [
            (str(row[0]), str(row[1]), str(row[2]), bool(row[3]) if len(row) > 3 else False)
            for row in stages
        ]
        self._boxes: list[QRect] = []

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        count = max(1, len(self._stages))
        gap = 16
        width = max(48, (rect.width() - gap * (count - 1)) // count)
        top = rect.y() + 4
        height = max(26, rect.height() - 2 * line - 14)
        self._boxes = []
        for index, (key, text, shape, discarded) in enumerate(self._stages):
            box = QRect(rect.x() + index * (width + gap), top, width, height)
            self._boxes.append(box)
            colour = self.role('scores')
            self.outline(painter, box, colour, radius=4, width=1.3,
                         fill=None if discarded else self.wash(colour, 40),
                         dashed=discarded)
            self.write(painter, box.adjusted(4, 2, -4, -2), text, colour=self.ink(),
                       points=7, align=Qt.AlignCenter | Qt.TextWordWrap)
            self.write(painter, QRect(box.x(), box.bottom() + 2, box.width(), line),
                       shape, colour=self.muted(), points=7,
                       align=Qt.AlignCenter | Qt.TextWordWrap)
            if index < count - 1:
                self.arrow(painter, box.right() + 2, top + height // 2,
                           box.right() + gap - 2, self.muted())
        self.write(painter, QRect(rect.x(), rect.bottom() - line, rect.width(), line),
                   'the shape the data took, from prompt to the first token',
                   colour=self.muted())

    def handle_click(self, point) -> bool:
        for index, (key, *_rest) in enumerate(self._stages):
            if index < len(self._boxes) and self._boxes[index].contains(point):
                self.stageClicked.emit(key)
                return True
        return False

    def describe(self) -> list[str]:
        out = []
        for _key, text, shape, _discarded in self._stages:
            out.append(text)
            out.append(shape)
        out.append('the shape the data took, from prompt to the first token')
        return out + super().describe()


class LayerDepthView(Diagram):
    """One row per layer: how much of this position's attention weight went to each key.

    The head-mean of the captured `attention_weights` row for the deciding position, over
    every key, one row per layer.  Typed positions are emphasised and the highlighted layer
    is outlined, so the reader sees at a glance where the position's attention settled and
    which layer put the most of it on the prompt.  Hovering a cell shows the exact head-mean.
    """

    BODY_HEIGHT = 180
    MAX_ROWS = 16
    MAX_COLS = 24

    def __init__(self, matrix, layer_labels, key_labels, *, typed=frozenset(),
                 highlight_row: int | None = None, tokens=None, caption: str = '',
                 parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._matrix = np.asarray(matrix, dtype=np.float64)
        if self._matrix.ndim != 2:
            self._matrix = self._matrix.reshape(1, -1) if self._matrix.size else self._matrix[:0].reshape(0, 0)
        self._layer_labels = list(layer_labels)
        self._key_labels = list(key_labels)
        self._typed = set(typed)
        self._highlight = None if highlight_row is None else int(highlight_row)
        self._grid = QRect()
        self._row_rects: list[QRect] = []
        line = self.text_height(8)
        self.set_body_height(max(120, min(self._matrix.shape[0], self.MAX_ROWS) * 16
                                  + 3 * line + 24))

    def _drawn_counts(self) -> tuple[int, int]:
        return (min(self._matrix.shape[0], self.MAX_ROWS),
                min(self._matrix.shape[1], self.MAX_COLS))

    def highlighted_row(self) -> int | None:
        """The layer the outline is on, or None.  Read by the test that checks the Level 1
        highlight equals an independently computed argmax."""
        return self._highlight

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        rows, cols = self._matrix.shape
        shown_r, shown_c = self._drawn_counts()
        side = 48
        top = rect.y() + line + 6
        self._grid = QRect(rect.x() + side, top,
                           max(40, rect.width() - side - 4),
                           max(30, rect.height() - 3 * line - 16))
        self.write(painter, QRect(rect.x(), rect.y(), side, line),
                   'layer', colour=self.muted(), points=8, bold=True,
                   align=Qt.AlignLeft | Qt.AlignVCenter)
        self.write(painter, QRect(self._grid.x(), rect.y(), self._grid.width(), line),
                   'key position  →', colour=self.muted())
        cell_h = self._grid.height() / max(1, shown_r)
        cell_w = self._grid.width() / max(1, shown_c)
        self._row_rects = []
        typed_cols = [c for c in range(min(shown_c, len(self._key_labels)))
                      if c in self._typed]
        for r in range(shown_r):
            real_r = r if rows <= self.MAX_ROWS else round(r * (rows - 1) / (shown_r - 1))
            y = int(self._grid.y() + r * cell_h)
            box = QRect(self._grid.x(), y, int(cell_w), int(cell_h))
            self._row_rects.append(box)
            is_hot = self._highlight is not None and real_r == self._highlight
            self.outline(painter, box, self.role('norm') if is_hot else self.faint(),
                         radius=2, width=2.0 if is_hot else 1.0)
            for c in range(shown_c):
                real_c = c if cols <= self.MAX_COLS else round(c * (cols - 1) / (shown_c - 1))
                value = float(self._matrix[real_r, real_c]) if real_r < rows and real_c < cols else 0.0
                cell = QRect(int(box.x() + c * cell_w) + 1, y + 1,
                             max(2, int(cell_w) - 2), max(2, int(cell_h) - 2))
                shade = int(40 + min(215, value * 260))
                fill = QColor(shade, shade, shade)
                painter.fillRect(cell, fill)
                if c in typed_cols:
                    self.outline(painter, cell, self.role('k'), radius=1, width=1.4)
        # Row labels down the side, the highlighted one always.
        small = cell_h >= 12
        for r in range(shown_r):
            real_r = r if rows <= self.MAX_ROWS else round(r * (rows - 1) / (shown_r - 1))
            label_text = (self._layer_labels[real_r] if real_r < len(self._layer_labels)
                          else str(real_r))
            box = QRect(rect.x(), int(self._grid.y() + r * cell_h), side - 6, int(cell_h))
            hot = self._highlight is not None and real_r == self._highlight
            self.write(painter, box, label_text, colour=self.role('norm') if hot else self.muted(),
                       points=7, bold=hot, align=Qt.AlignRight | Qt.AlignVCenter)
        self.write(painter, QRect(self._grid.x(), self._grid.bottom() + 4,
                                  self._grid.width(), line),
                   'attention weight, averaged over heads (darker = more)',
                   colour=self.muted())

    def tooltip_at(self, point) -> str:
        if not self._grid.contains(point):
            return ''
        rows, cols = self._matrix.shape
        shown_r, shown_c = self._drawn_counts()
        cell_h = self._grid.height() / max(1, shown_r)
        cell_w = self._grid.width() / max(1, shown_c)
        r = int((point.y() - self._grid.y()) / cell_h)
        c = int((point.x() - self._grid.x()) / cell_w)
        if not (0 <= r < shown_r and 0 <= c < shown_c):
            return ''
        real_r = r if rows <= self.MAX_ROWS else round(r * (rows - 1) / (shown_r - 1))
        real_c = c if cols <= self.MAX_COLS else round(c * (cols - 1) / (shown_c - 1))
        if not (0 <= real_r < rows and 0 <= real_c < cols):
            return ''
        layer = self._layer_labels[real_r] if real_r < len(self._layer_labels) else str(real_r)
        key = self._key_labels[real_c] if real_c < len(self._key_labels) else str(real_c)
        return f'Layer {layer} · key {key} · head-mean weight {float(self._matrix[real_r, real_c]):.6f}'

    def describe(self) -> list[str]:
        return (['layer'] + ['key position  →',
                             'attention weight, averaged over heads (darker = more)']
                + [f'{label}' for label in self._layer_labels]
                + super().describe())


class ScoreDotPlot(Diagram):
    """The top scores on a raw score axis -- the logits that chose the token, as scores.

    Never percentages: these are logits, and the axis is labelled 'score (logit)'.  The
    selected token's dot is marked, so the reader sees which entry won and by how much.
    """

    BODY_HEIGHT = 120

    def __init__(self, scores, labels, selected, *, tokens=None, caption: str = '',
                 parent=None):
        super().__init__(tokens=tokens, caption=caption, parent=parent)
        self._scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        self._labels = [str(label) for label in labels]
        self._selected = None if selected is None else int(selected)
        self._dots: list[tuple[float, QRect]] = []

    def draw(self, painter: QPainter, rect: QRect) -> None:
        line = self.text_height(8)
        n = self._scores.size
        if n == 0:
            self.write(painter, rect, 'No scores recorded.', colour=self.muted(),
                       align=Qt.AlignCenter)
            return
        lo, hi = float(np.min(self._scores)), float(np.max(self._scores))
        span = (hi - lo) or 1.0
        axis_top = rect.y() + line + 6
        axis_bottom = rect.bottom() - line - 4
        axis_left = rect.x() + 44
        axis_right = rect.right() - 8
        # The axis line.
        pen = QPen(self.muted(), 1.0)
        painter.setPen(pen)
        painter.drawLine(axis_left, axis_top, axis_left, axis_bottom)
        painter.drawLine(axis_left, axis_bottom, axis_right, axis_bottom)
        self.write(painter, QRect(rect.x(), rect.y(), axis_right - rect.x(), line),
                   'score (logit)', colour=self.role('scores'), points=8, bold=True)
        # Ticks at the ends.
        self.write(painter, QRect(axis_left - 24, axis_bottom + 2, 48, line),
                   f'{lo:.1f}', colour=self.muted(), points=7,
                   align=Qt.AlignHCenter)
        self.write(painter, QRect(axis_right - 24, axis_bottom + 2, 48, line),
                   f'{hi:.1f}', colour=self.muted(), points=7,
                   align=Qt.AlignHCenter)
        self._dots = []
        step = max(10, (axis_bottom - axis_top - 6) / n)
        for index, value in enumerate(self._scores):
            y = axis_bottom - (index + 0.5) * step
            x = axis_left + int((value - lo) / span * (axis_right - axis_left))
            is_selected = self._selected is not None and index == self._selected
            radius = 5 if is_selected else 3
            painter.setPen(QPen(self.role('scores') if is_selected else self.muted(),
                                1.6 if is_selected else 1.0))
            painter.setBrush(self.wash(self.role('scores'), 90 if is_selected else 50))
            painter.drawEllipse(QPoint(x, int(y)), radius, radius)
            label_text = self._labels[index] if index < len(self._labels) else ''
            if label_text:
                self.write(painter, QRect(x - 34, int(y) - 6, 72, 12), label_text,
                           colour=self.ink() if is_selected else self.muted(), points=7,
                           align=Qt.AlignHCenter | Qt.AlignVCenter)
            self._dots.append((float(value), QRect(x - radius - 4, int(y) - 8,
                                                   radius * 2 + 8, 16)))

    def tooltip_at(self, point) -> str:
        for value, box in self._dots:
            if box.contains(point):
                return f'score (logit) {value:.6f}'
        return ''

    def describe(self) -> list[str]:
        if self._scores.size == 0:
            return ['No scores recorded.'] + super().describe()
        lo, hi = float(np.min(self._scores)), float(np.max(self._scores))
        return ([f'{lo:.1f}', f'{hi:.1f}', 'score (logit)']
                + [self._labels[i] for i in range(self._scores.size)
                   if i < len(self._labels)]
                + super().describe())


# Every concrete diagram, for the gallery below and for the test that paints all of them.
DIAGRAMS = (VectorStrip, MatrixGrid, EmbeddingLookupRow, MatMulShapes, HeadSplit, KVShare,
            QKCompare, CausalMaskTriangle, SoftmaxFlow, SoftmaxBars, AttentionGrid,
            ResidualBypass, GatedFFNPath, VocabularyProjection, LayerStack,
            TrainingVsInference, StoppingLoop, DecisionChain, PositionScope,
            PipelineStrip, LayerDepthView, ScoreDotPlot)


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
        HeadSplit(2560, 4096, 32, 128, 15, highlight_head=1, highlight_row=4, tokens=tokens,
                  caption='Projection computes; reshaping only regroups.'),
        KVShare(8, 32, highlight_query_head=5, tokens=tokens,
                caption='Grouped-query attention.'),
        QKCompare(128, query_pos=14, key_pos=4, query_label='ĊĊ', key_label='Ġcolor',
                  tokens=tokens, caption='One query against one key.'),
        CausalMaskTriangle(15, 15, selected_query=3, clickable=True, tokens=tokens,
                           caption='Each query may read itself and earlier positions.'),
        SoftmaxFlow(tokens=tokens, caption='Deliberately numberless.'),
        SoftmaxBars(scores, weights, labels=[str(i) for i in range(9)],
                    sublabels=['<|im', 'user', '\\n', 'fav', ' color', '?', '<|im', '', ''],
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
                             focus_position=4, tokens=tokens,
                             caption='Only the final row was stored.'),
        LayerStack(36, 17, tokens=tokens, caption='One of thirty-six.'),
        TrainingVsInference(tokens=tokens, caption='Weights are written once, then read.'),
        StoppingLoop(tokens=tokens, caption='One loop, one exit.'),
        DecisionChain([('final row of X_final', True), ('final RMSNorm', False),
                       ('× W_Uᵀ', False), ('logits', True), ('argmax', True),
                       ('token', True)], tokens=tokens,
                      caption='Captured links solid, discarded links dashed.'),
        PositionScope(tokens=tokens, caption='Across positions, then within one.'),
        PipelineStrip([('ids', 'token IDs', '[P]'),
                       ('embed', 'embed', '[P × 2560]'),
                       ('layers', 'N × layer', '[P × 2560]'),
                       ('vocab', 'project', '[151,936]', True),
                       ('token', 'token', '1', True)],
                      tokens=tokens, caption='The shape this run took, prompt to answer.'),
        LayerDepthView(np.triu(np.linspace(0.1, 0.9, 36).reshape(36, 1)
                               * np.linspace(1.0, 0.2, 15).reshape(1, 15),
                               k=0).astype(np.float64),
                       layer_labels=[f'{i}' for i in range(36)],
                       key_labels=[str(i) for i in range(15)],
                       typed={0, 1, 2, 3, 4}, highlight_row=23, tokens=tokens,
                       caption='Head-mean attention weight per layer, per key.'),
        ScoreDotPlot(np.array([2.5, 3.1, 1.2, 4.8, 0.4, -1.1, 2.2, 1.9, 0.7, -0.3],
                              dtype=np.float64),
                     labels=[str(i) for i in range(10)], selected=3, tokens=tokens,
                     caption='The top scores; the marked dot chose the token.'),
    ]



# ── Figure widths ────────────────────────────────────────────────────────────
# The widest each sketch is drawn.  Set in one table so the proportions of the whole
# set of figures can be judged together; a figure in a wider window grows margins.
_FIGURE_WIDTHS = {
    VectorStrip: 560,
    MatrixGrid: 760,
    EmbeddingLookupRow: 780,
    MatMulShapes: 780,
    HeadSplit: 780,
    KVShare: 700,
    QKCompare: 700,
    CausalMaskTriangle: 620,
    SoftmaxFlow: 640,
    SoftmaxBars: 760,
    AttentionGrid: 560,
    ResidualBypass: 780,
    GatedFFNPath: 760,
    VocabularyProjection: 840,
    LayerStack: 700,
    TrainingVsInference: 700,
    StoppingLoop: 660,
    DecisionChain: 840,
    PositionScope: 700,
    PipelineStrip: 800,
    LayerDepthView: 760,
    ScoreDotPlot: 620,
}
for _cls, _width in _FIGURE_WIDTHS.items():
    _cls.MAX_WIDTH = _width
del _cls, _width

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
