"""Indexed palette swatch grid.

Custom-painted rather than a grid of QPushButtons: a palette can run to several
hundred slots, and rebuilding that many widgets on every edit is both slow and
awkward to keep in sync with slot indices that shift when a slot is merged away.

The first cell is a checkerboard "no paint" pseudo-slot, so erase can be bound to
a mouse button exactly like any color.
"""

from PyQt6.QtCore import Qt, QRect, QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QWidget, QSizePolicy

from ...core.indexed import EMPTY

_PAD = 2


class PaletteGrid(QWidget):
    """Left-click sets the left paint color, right-click the right one."""

    slot_picked = pyqtSignal(int, int)     # slot (EMPTY for erase), Qt button value
    slot_selected = pyqtSignal(int)        # selection changed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._indexed = None
        self._counts = None
        self._barred = None     # drawn, but refuses to be picked
        self._selected = 0
        self._size = 24
        self._cols = 1
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ---------- data ----------
    def set_indexed(self, indexed):
        self._indexed = indexed
        self._counts = None if indexed is None else indexed.used_counts()
        if indexed is not None and self._selected >= len(indexed.slots):
            self._selected = len(indexed.slots) - 1 if indexed.slots else EMPTY
        self._relayout()
        self.update()

    def set_barred(self, slot):
        """Mark one slot as visible but unpickable.
        
        Replace-with needs you to see which color you are replacing. Leaving it out
        of the grid shifts every swatch after it and you lose your place, while never
        letting you pick it as its own target.
        """
        self._barred = slot
        self.update()

    def set_swatch_size(self, n):
        self._size = max(12, int(n))
        self._relayout()
        self.update()

    def selected(self):
        return self._selected

    def set_selected(self, k):
        self._selected = int(k)
        self.update()
        self.slot_selected.emit(self._selected)

    # ---------- geometry ----------
    def _cell_count(self):
        return 1 + (len(self._indexed.slots) if self._indexed else 0)   # +1 for erase

    def _relayout(self):
        step = self._size + _PAD
        self._cols = max(1, (self.width() - _PAD) // step)
        rows = (self._cell_count() + self._cols - 1) // self._cols
        self.setMinimumHeight(rows * step + _PAD)

    def resizeEvent(self, e):
        self._relayout()
        super().resizeEvent(e)

    def sizeHint(self):
        return QSize(300, self.minimumHeight())

    def _cell_rect(self, i):
        step = self._size + _PAD
        r, c = divmod(i, self._cols)
        return QRect(_PAD + c * step, _PAD + r * step, self._size, self._size)

    def _cell_at(self, pos):
        step = self._size + _PAD
        c = (pos.x() - _PAD) // step
        r = (pos.y() - _PAD) // step
        if c < 0 or c >= self._cols or r < 0:
            return None
        i = r * self._cols + c
        return i if 0 <= i < self._cell_count() else None

    def _slot_for_cell(self, i):
        return EMPTY if i == 0 else i - 1

    # ---------- events ----------
    def mousePressEvent(self, e):
        i = self._cell_at(e.position().toPoint())
        if i is None:
            return
        slot = self._slot_for_cell(i)
        if self._barred is not None and slot == self._barred:
            return
        self.set_selected(slot)
        self.slot_picked.emit(slot, int(e.button().value))

    # ---------- paint ----------
    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().window())
        if self._indexed is None:
            p.end()
            return
        for i in range(self._cell_count()):
            slot = self._slot_for_cell(i)
            r = self._cell_rect(i)
            if slot == EMPTY:
                half = self._size // 2
                p.fillRect(r, QColor("#6a6a6a"))
                p.fillRect(r.left(), r.top(), half, half, QColor("#585858"))
                p.fillRect(r.left() + half, r.top() + half,
                           r.width() - half, r.height() - half, QColor("#585858"))
            else:
                p.fillRect(r, QColor(self._indexed.slots[slot].hex))
                if self._counts is not None and slot < len(self._counts) and self._counts[slot] == 0:
                    # unused slots exist only for painting; mark them so it is
                    # obvious which ones are safe to remove
                    p.setPen(QPen(QColor(0, 0, 0, 160), 1))
                    p.setBrush(QColor(255, 255, 255, 200))
                    d = max(4, self._size // 5)
                    p.drawEllipse(r.right() - d - 1, r.bottom() - d - 1, d, d)
                    p.setBrush(Qt.BrushStyle.NoBrush)
            if self._barred is not None and slot == self._barred:
                # Hatched and ringed: reads as "this one, and not available".
                p.setPen(QPen(QColor(0, 0, 0, 170), 1))
                for o in range(-self._size, self._size, 4):
                    p.drawLine(r.left() + max(0, o), r.top() + max(0, -o),
                               r.left() + min(self._size, o + self._size),
                               r.top() + min(self._size, self._size - o))
                p.setPen(QPen(QColor("#d04545"), 2))
                p.drawRect(r.adjusted(1, 1, -1, -1))
            elif slot == self._selected:
                p.setPen(QPen(QColor("#000000"), 3))
                p.drawRect(r.adjusted(1, 1, -1, -1))
                p.setPen(QPen(QColor("#ffffff"), 1))
                p.drawRect(r.adjusted(1, 1, -1, -1))
            else:
                p.setPen(QPen(QColor(0, 0, 0, 90), 1))
                p.drawRect(r)
        p.end()
