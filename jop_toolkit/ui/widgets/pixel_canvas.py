"""Pixel-accurate editing canvas.

Deliberately not a QGraphicsView: ZoomView's ScrollHandDrag and compare swipe
both claim the left button, which a brush needs. This widget owns its own
zoom/pan and reports integer pixel coordinates, so tools never deal with scene
transforms.

Panning is middle-drag or space-drag only. Left and right stay free for tools.
"""

from PyQt6.QtCore import Qt, QPoint, QRect, QLine, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QColor, QPen, QPolygon
from PyQt6.QtWidgets import QWidget, QSizePolicy
import numpy as np

from ...core import selection as sel_core

_CHECKER = 8            # checkerboard square size, in screen pixels
_MIN_ZOOM = 1
_MAX_ZOOM = 64
_ANT_LIMIT = 8000       # segments above which the ants stop marching


class PixelCanvas(QWidget):
    """Displays an IndexedImage and reports pixel-space mouse events."""

    pixel_pressed = pyqtSignal(int, int, int)    # x, y, Qt.MouseButton value
    pixel_dragged = pyqtSignal(int, int, int)
    pixel_released = pyqtSignal(int, int, int)
    hover_changed = pyqtSignal(int, int)         # -1, -1 when outside
    zoom_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = None            # QImage, ARGB, alpha 0 where empty
        self._buf = None
        self._w = self._h = 0
        self._zoom = 8
        self._origin = QPoint(0, 0) # top-left of the image in widget coords
        self._panning = False
        self._pan_from = QPoint()
        self._space = False
        self._hover = (-1, -1)
        self._brush_size = 1
        self._grid_on = False
        self._grid_step = 16
        self._sel_segs = None
        self._sel_box = None     # selection bounds in image coords
        self._pending = None
        self._ant_phase = 0
        self._ants = QTimer(self)
        self._ants.setInterval(150)
        self._ants.timeout.connect(self._tick_ants)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ---------- content ----------
    def set_indexed(self, indexed, fit=False):
        """Rebuild the display image. Empty pixels become transparent."""
        if indexed is None:
            self._img = None
            self._w = self._h = 0
            self.update()
            return
        rgb = indexed.to_rgb()
        h, w = rgb.shape[:2]
        argb = np.empty((h, w, 4), dtype=np.uint8)
        argb[..., 0] = rgb[..., 2]           # QImage ARGB32 is BGRA in memory
        argb[..., 1] = rgb[..., 1]
        argb[..., 2] = rgb[..., 0]
        argb[..., 3] = np.where(indexed.indices < 0, 0, 255).astype(np.uint8)
        self._buf = np.ascontiguousarray(argb)   # keep a reference: QImage does not copy
        self._img = QImage(self._buf.data, w, h, 4 * w, QImage.Format.Format_ARGB32)
        first = (self._w, self._h) != (w, h)
        self._w, self._h = w, h
        if fit or first:
            self.fit()
        self.update()

    # ---------- view ----------
    def zoom(self):
        return self._zoom

    def set_zoom(self, z, anchor=None):
        z = max(_MIN_ZOOM, min(_MAX_ZOOM, int(z)))
        if z == self._zoom:
            return
        if anchor is None:
            anchor = QPoint(self.width() // 2, self.height() // 2)
        # keep the image point under `anchor` fixed across the zoom
        ix = (anchor.x() - self._origin.x()) / self._zoom
        iy = (anchor.y() - self._origin.y()) / self._zoom
        self._zoom = z
        self._origin = QPoint(int(anchor.x() - ix * z), int(anchor.y() - iy * z))
        self.zoom_changed.emit(self._zoom)
        self.update()

    def fit(self):
        if not self._w or not self._h:
            return
        z = max(_MIN_ZOOM, min(_MAX_ZOOM,
                               int(min(self.width() / self._w, self.height() / self._h))))
        self._zoom = z
        self._origin = QPoint((self.width() - self._w * z) // 2,
                              (self.height() - self._h * z) // 2)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_to(self, x0, y0, x1, y1, margin=2):
        """Frame an image-space box, the way fit() frames the whole image."""
        if not self._w or not self._h:
            return
        x0 = max(0, int(x0) - margin)
        y0 = max(0, int(y0) - margin)
        x1 = min(self._w, int(x1) + 1 + margin)
        y1 = min(self._h, int(y1) + 1 + margin)
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        z = max(_MIN_ZOOM, min(_MAX_ZOOM,
                               int(min(self.width() / bw, self.height() / bh))))
        self._zoom = z
        self._origin = QPoint(int(self.width() / 2 - (x0 + bw / 2) * z),
                              int(self.height() / 2 - (y0 + bh / 2) * z))
        self.zoom_changed.emit(self._zoom)
        self.update()

    def set_grid(self, on, step=None):
        self._grid_on = bool(on)
        if step:
            self._grid_step = max(1, int(step))
        self.update()

    def set_brush_size(self, n):
        self._brush_size = max(1, int(n))
        self.update()

    # ---------- selection ----------
    def set_selection(self, mask):
        """Outline edges are precomputed here rather than in paintEvent, which
        runs again on every ant-animation frame."""
        if mask is None or not mask.any():
            self._sel_segs = None
            self._sel_box = None
            self._ants.stop()
        else:
            self._sel_segs = sel_core.outline_segments(mask)
            rows = np.flatnonzero(mask.any(1))
            cols = np.flatnonzero(mask.any(0))
            self._sel_box = (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))
            if self.isVisible() and len(self._sel_segs) <= _ANT_LIMIT:
                self._ants.start()
            else:
                self._ants.stop()
        self.update()

    def set_pending(self, kind, points):
        """Live outline while a marquee or lasso drag is in progress."""
        self._pending = None if kind is None else (kind, list(points))
        self.update()

    def _tick_ants(self):
        # Repaint only where the outline is. A selection in one corner should
        # not cost a full-canvas repaint several times a second.
        self._ant_phase = (self._ant_phase + 1) % 8
        if self._sel_box is None:
            self.update()
            return
        x0, y0, x1, y1 = self._sel_box
        z = self._zoom
        self.update(QRect(self._origin.x() + x0 * z, self._origin.y() + y0 * z,
                          (x1 - x0 + 1) * z, (y1 - y0 + 1) * z).adjusted(-2, -2, 2, 2))

    def hideEvent(self, e):
        self._ants.stop()
        super().hideEvent(e)

    def showEvent(self, e):
        if self._sel_segs is not None and len(self._sel_segs) <= _ANT_LIMIT:
            self._ants.start()
        super().showEvent(e)

    # ---------- coordinate mapping ----------
    def to_pixel(self, pos):
        """Widget point -> (x, y) image coords, or (-1, -1) outside."""
        if not self._w:
            return -1, -1
        x = int((pos.x() - self._origin.x()) // self._zoom)
        y = int((pos.y() - self._origin.y()) // self._zoom)
        if 0 <= x < self._w and 0 <= y < self._h:
            return x, y
        return -1, -1

    # ---------- events ----------
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._space = False
            self.setCursor(Qt.CursorShape.CrossCursor)
        super().keyReleaseEvent(e)

    def wheelEvent(self, e):
        d = e.angleDelta().y()
        if d:
            self.set_zoom(self._zoom * 2 if d > 0 else self._zoom // 2, e.position().toPoint())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton or self._space:
            self._panning = True
            self._pan_from = e.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        x, y = self.to_pixel(e.position().toPoint())
        if x >= 0:
            self.pixel_pressed.emit(x, y, int(e.button().value))

    def mouseMoveEvent(self, e):
        p = e.position().toPoint()
        if self._panning:
            self._origin += p - self._pan_from
            self._pan_from = p
            self.update()
            return
        x, y = self.to_pixel(p)
        if (x, y) != self._hover:
            self._hover = (x, y)
            self.hover_changed.emit(x, y)
            self.update()
        if x >= 0 and e.buttons():
            self.pixel_dragged.emit(x, y, int(e.buttons().value))

    def mouseReleaseEvent(self, e):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space
                           else Qt.CursorShape.CrossCursor)
            return
        x, y = self.to_pixel(e.position().toPoint())
        self.pixel_released.emit(x, y, int(e.button().value))

    def leaveEvent(self, _e):
        self._hover = (-1, -1)
        self.hover_changed.emit(-1, -1)
        self.update()

    # ---------- paint ----------
    def paintEvent(self, e):
        # Everything below is clipped to the damaged region, not the whole
        # widget: the ant animation repaints only the selection's bounds.
        damage = e.rect()
        p = QPainter(self)
        p.fillRect(damage, QColor("#2b2b2b"))
        if self._img is None:
            p.end()
            return
        z = self._zoom
        area = QRect(self._origin.x(), self._origin.y(), self._w * z, self._h * z)
        vis = area.intersected(self.rect())
        if vis.isEmpty():
            p.end()
            return

        # Checkerboard, clipped to what is on screen. Iterating the whole image
        # area costs ~890k fillRect calls at 64x on a 96x144 image; the visible
        # region is bounded by the viewport no matter how far in you zoom.
        p.save()
        p.setClipRect(vis)
        c0, c1 = QColor("#6a6a6a"), QColor("#585858")
        x0 = area.left() + ((vis.left() - area.left()) // _CHECKER) * _CHECKER
        y0 = area.top() + ((vis.top() - area.top()) // _CHECKER) * _CHECKER
        for gy in range(y0, vis.bottom() + 1, _CHECKER):
            for gx in range(x0, vis.right() + 1, _CHECKER):
                odd = ((gx - area.left()) // _CHECKER + (gy - area.top()) // _CHECKER) & 1
                p.fillRect(gx, gy, _CHECKER, _CHECKER, c1 if odd else c0)
        p.restore()

        # Blit only the visible sub-rectangle of the source image.
        sx0 = (vis.left() - area.left()) // z
        sy0 = (vis.top() - area.top()) // z
        sx1 = (vis.right() - area.left()) // z
        sy1 = (vis.bottom() - area.top()) // z
        src = QRect(sx0, sy0, sx1 - sx0 + 1, sy1 - sy0 + 1)
        dst = QRect(area.left() + sx0 * z, area.top() + sy0 * z,
                    src.width() * z, src.height() * z)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        p.drawImage(dst, self._img, src)

        if self._grid_on and z >= 3:
            pen = QPen(QColor(255, 255, 255, 70))
            pen.setCosmetic(True)
            p.setPen(pen)
            step = self._grid_step
            for gx in range(sx0 - sx0 % step, sx1 + step + 1, step):
                if 0 <= gx <= self._w:
                    x = self._origin.x() + gx * z
                    p.drawLine(x, vis.top(), x, vis.bottom())
            for gy in range(sy0 - sy0 % step, sy1 + step + 1, step):
                if 0 <= gy <= self._h:
                    y = self._origin.y() + gy * z
                    p.drawLine(vis.left(), y, vis.right(), y)

        self._paint_selection(p, z, sx0, sy0, sx1, sy1)
        self._paint_pending(p, z)

        hx, hy = self._hover
        if hx >= 0:
            n = self._brush_size
            off = (n - 1) // 2
            r = QRect(self._origin.x() + (hx - off) * z, self._origin.y() + (hy - off) * z,
                      n * z, n * z)
            p.setPen(QPen(QColor(0, 0, 0, 200), 3))
            p.drawRect(r)
            p.setPen(QPen(QColor(255, 255, 255, 230), 1))
            p.drawRect(r)
        p.end()

    def _paint_selection(self, p, z, sx0, sy0, sx1, sy1):
        if self._sel_segs is None or not len(self._sel_segs):
            return
        s = self._sel_segs
        # A speckled selection can run to tens of thousands of edges; only the
        # ones inside the viewport are worth turning into QLine objects.
        v = s[(s[:, 0] >= sx0) & (s[:, 0] <= sx1 + 1) &
              (s[:, 1] >= sy0) & (s[:, 1] <= sy1 + 1)]
        ox, oy = self._origin.x(), self._origin.y()
        lines = [QLine(ox + int(a) * z, oy + int(b) * z,
                       ox + int(c) * z, oy + int(d) * z) for a, b, c, d in v]
        if not lines:
            return
        for color, phase in ((QColor(0, 0, 0), 0), (QColor(255, 255, 255), 4)):
            pen = QPen(color, 1)
            pen.setCosmetic(True)
            pen.setDashPattern([4, 4])
            pen.setDashOffset(phase + self._ant_phase)
            p.setPen(pen)
            p.drawLines(lines)

    def _paint_pending(self, p, z):
        if self._pending is None:
            return
        kind, pts = self._pending
        pen = QPen(QColor(255, 255, 255, 220), 1)
        pen.setCosmetic(True)
        pen.setDashPattern([3, 3])
        p.setPen(pen)
        ox, oy = self._origin.x(), self._origin.y()
        if kind == 'rect' and len(pts) == 2:
            (ax, ay), (bx, by) = pts
            xa, xb = sorted((ax, bx))
            ya, yb = sorted((ay, by))
            p.drawRect(QRect(ox + xa * z, oy + ya * z,
                             (xb - xa + 1) * z, (yb - ya + 1) * z))
        elif kind == 'lasso' and len(pts) > 1:
            half = z // 2
            p.drawPolyline(QPolygon([QPoint(ox + px * z + half, oy + py * z + half)
                                     for px, py in pts]))
