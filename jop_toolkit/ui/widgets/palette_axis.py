"""The palette number line: a strip of the image's own colors ordered along a
selectable axis, with draggable markers that own the stretch nearest to them.

Markers are anchors, not fences: a color joins whichever marker is closest along
the axis, so the boundary between two markers is their midpoint and dragging one
moves both its anchor and its edges. Each marker's core color is read from the
strip at its position unless the user pins an override, which is what makes the
line behave like a curve whose output is color.
"""

from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPen, QBrush, QColor, QPolygon
from PyQt6.QtWidgets import QWidget, QSizePolicy
import numpy as np

_STRIP_H = 46
_HANDLE_H = 18
_PAD = 10


class PaletteAxis(QWidget):
    """Strip + markers. Positions are normalized 0..1 so an axis switch keeps them."""

    markers_changed = pyqtSignal()
    selection_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._strip = None
        self._positions = []
        self._colors = []
        self._pinned = []
        self._selected = -1
        self._drag = -1
        self.setMinimumHeight(_STRIP_H + _HANDLE_H + 2 * _PAD)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    # ---------- data ----------
    def set_strip(self, strip_rgb):
        self._strip = None if strip_rgb is None else np.ascontiguousarray(strip_rgb, dtype=np.uint8)
        self.update()

    def set_markers(self, positions, colors, pinned=None):
        order = np.argsort(np.asarray(positions, dtype=float)) if len(positions) else []
        pinned = list(pinned) if pinned is not None else [False] * len(positions)
        self._positions = [float(positions[i]) for i in order]
        self._colors = [colors[i] for i in order]
        self._pinned = [bool(pinned[i]) for i in order]
        self._selected = min(self._selected, len(self._positions) - 1)
        self.update()

    def markers(self):
        return list(self._positions), list(self._colors), list(self._pinned)

    def selected(self):
        return self._selected

    def set_selected(self, idx):
        self._selected = int(idx) if 0 <= idx < len(self._positions) else -1
        self.update()
        self.selection_changed.emit(self._selected)

    def set_marker_color(self, idx, hex_color, pin=True):
        """Pinning stops the color being re-read from the strip on the next drag."""
        if 0 <= idx < len(self._colors):
            self._colors[idx] = hex_color
            self._pinned[idx] = bool(pin)
            self.update()

    def unpin(self, idx):
        if 0 <= idx < len(self._pinned):
            self._pinned[idx] = False
            self._colors[idx] = self.strip_color_at(self._positions[idx])
            self.update()

    # ---------- geometry ----------
    def _strip_rect(self):
        return QRect(_PAD, _PAD, max(1, self.width() - 2 * _PAD), _STRIP_H)

    def _x_for(self, t):
        r = self._strip_rect()
        return r.left() + int(round(t * (r.width() - 1)))

    def _t_for(self, x):
        r = self._strip_rect()
        return min(1.0, max(0.0, (x - r.left()) / max(1, r.width() - 1)))

    def _hit(self, x, y):
        if not self._positions:
            return -1
        r = self._strip_rect()
        if y < r.top() - 4 or y > r.bottom() + _HANDLE_H:
            return -1
        dists = [abs(self._x_for(t) - x) for t in self._positions]
        i = int(np.argmin(dists))
        return i if dists[i] <= 9 else -1

    # ---------- events ----------
    def mousePressEvent(self, event):
        i = self._hit(event.pos().x(), event.pos().y())
        if event.button() == Qt.MouseButton.RightButton:
            if i >= 0 and len(self._positions) > 1:
                del self._positions[i]
                del self._colors[i]
                del self._pinned[i]
                self._selected = -1
                self.update()
                self.markers_changed.emit()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if i >= 0:
                self._drag = i
                self.set_selected(i)
            else:
                self.set_selected(-1)

    def mouseMoveEvent(self, event):
        if self._drag >= 0:
            self._positions[self._drag] = self._t_for(event.pos().x())
            self.update()
            return
        self.setCursor(Qt.CursorShape.SizeHorCursor
                       if self._hit(event.pos().x(), event.pos().y()) >= 0
                       else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._drag >= 0:
            # Re-sort on release, not during the drag: reordering mid-drag would
            # swap the handle out from under the cursor.
            # An unpinned marker re-reads its color from the strip, so the line
            # behaves like a curve whose output is color; a pinned one keeps it.
            if not self._pinned[self._drag]:
                self._colors[self._drag] = self.strip_color_at(self._positions[self._drag])
            pos, col, pin = self._positions, self._colors, self._pinned
            order = list(np.argsort(np.asarray(pos, dtype=float)))
            moved = pos[self._drag]
            self._positions = [pos[i] for i in order]
            self._colors = [col[i] for i in order]
            self._pinned = [pin[i] for i in order]
            self._selected = self._positions.index(moved)
            self._drag = -1
            self.update()
            self.markers_changed.emit()
            self.selection_changed.emit(self._selected)

    def mouseDoubleClickEvent(self, event):
        if self._hit(event.pos().x(), event.pos().y()) >= 0:
            return
        t = self._t_for(event.pos().x())
        self._positions.append(t)
        self._colors.append(self.strip_color_at(t))
        self._pinned.append(False)
        self.set_markers(self._positions, self._colors, self._pinned)
        self.markers_changed.emit()

    def strip_color_at(self, t):
        if self._strip is None or len(self._strip) == 0:
            return "#808080"
        i = min(len(self._strip) - 1, max(0, int(t * (len(self._strip) - 1))))
        r, g, b = (int(v) for v in self._strip[i])
        return f"#{r:02x}{g:02x}{b:02x}"

    # ---------- paint ----------
    def paintEvent(self, _event):
        p = QPainter(self)
        r = self._strip_rect()
        if self._strip is not None and len(self._strip):
            img = QImage(self._strip.tobytes(), len(self._strip), 1, 3 * len(self._strip),
                         QImage.Format.Format_RGB888)
            p.drawImage(r, img)
        else:
            p.fillRect(r, QColor("#3a3a3a"))
        p.setPen(QPen(QColor("#202020")))
        p.drawRect(r)

        for i, t in enumerate(self._positions):
            if i + 1 < len(self._positions):          # boundary = midpoint
                bx = (self._x_for(t) + self._x_for(self._positions[i + 1])) // 2
                p.setPen(QPen(QColor(255, 255, 255, 140), 1, Qt.PenStyle.DashLine))
                p.drawLine(bx, r.top(), bx, r.bottom())

        for i, t in enumerate(self._positions):
            x = self._x_for(t)
            sel = (i == self._selected)
            p.setPen(QPen(QColor("#ffffff" if sel else "#101010"), 2 if sel else 1))
            p.drawLine(x, r.top(), x, r.bottom())
            tri = QPolygon([QPoint(x, r.bottom() + 2),
                            QPoint(x - 7, r.bottom() + _HANDLE_H),
                            QPoint(x + 7, r.bottom() + _HANDLE_H)])
            p.setBrush(QBrush(QColor(self._colors[i])))
            p.setPen(QPen(QColor("#ffffff" if sel else "#202020"), 2 if sel else 1))
            p.drawPolygon(tri)
        p.end()
