"""Pan/zoom QGraphicsView with crisp nearest-neighbor pixel scaling."""

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPixmap, QPen, QWheelEvent, QMouseEvent
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
import numpy as np


def numpy_to_qimage(arr):
    """(H, W, 3) uint8 RGB -> QImage (copied so the buffer stays alive)."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    img = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()


class ZoomView(QGraphicsView):
    """Scrollable view with mouse-wheel zoom and middle-click pan."""

    cell_clicked = pyqtSignal(int, int, int)  # x, y, button

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_paint = None
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self._user_zoomed = False
        self._compare_pix = None
        self._compare_on = False
        self._split = 0.5
        self._dragging_split = False

    def set_image(self, arr, fit=False):
        """Display an (H, W, 3) uint8 array. fit=True rescales to fit on first show."""
        img = numpy_to_qimage(arr)
        pix = QPixmap.fromImage(img)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pix)
        else:
            self._pixmap_item.setPixmap(pix)
        self._scene.setSceneRect(0, 0, pix.width(), pix.height())
        if fit or not self._user_zoomed:
            self._fit()

    def set_overlay_paint(self, cb):
        """Register a callback(painter, scene_rect) to draw on top of the image."""
        self._overlay_paint = cb
        self.viewport().update()

    # ---------- before/after compare ----------
    def set_compare_image(self, arr):
        """Set the image shown left of the swipe divider (None clears)."""
        self._compare_pix = None if arr is None else QPixmap.fromImage(numpy_to_qimage(arr))
        self.viewport().update()

    def set_compare_enabled(self, on):
        # Leave ScrollHandDrag alone: mousePressEvent only claims the press when
        # it lands on the divider, so ordinary drags still pan.
        self._compare_on = bool(on)
        self.viewport().update()

    def _split_scene_x(self):
        return self._scene.sceneRect().width() * self._split

    def _near_split(self, scene_x):
        w = max(1.0, self._scene.sceneRect().width())
        return abs(scene_x - self._split_scene_x()) <= max(2.0, w * 0.03)

    def _fit(self):
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1 / 1.25
        self.scale(factor, factor)
        self._user_zoomed = True

    def mousePressEvent(self, event: QMouseEvent):
        if (self._compare_on and self._compare_pix is not None
                and event.button() == Qt.MouseButton.LeftButton
                and self._near_split(self.mapToScene(event.pos()).x())):
            self._dragging_split = True
            return
        if self._pixmap_item is not None and event.button() in (
            Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton
        ):
            scene_pt = self.mapToScene(event.pos())
            ix = int(scene_pt.x())
            iy = int(scene_pt.y())
            pix = self._pixmap_item.pixmap()
            if 0 <= ix < pix.width() and 0 <= iy < pix.height():
                self.cell_clicked.emit(ix, iy, int(event.button().value))
                # Suppress drag-pan on middle/right so they're click-only
                if event.button() != Qt.MouseButton.LeftButton:
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging_split:
            w = max(1.0, self._scene.sceneRect().width())
            self._split = min(1.0, max(0.0, self.mapToScene(event.pos()).x() / w))
            self.viewport().update()
            return
        if self._compare_on and self._compare_pix is not None:
            near = self._near_split(self.mapToScene(event.pos()).x())
            self.viewport().setCursor(Qt.CursorShape.SplitHCursor if near
                                      else Qt.CursorShape.OpenHandCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging_split:
            self._dragging_split = False
            return
        super().mouseReleaseEvent(event)

    def drawForeground(self, painter, rect):
        if self._compare_on and self._compare_pix is not None:
            sx = self._split_scene_x()
            sr = self._scene.sceneRect()
            painter.save()
            painter.setClipRect(QRectF(sr.left(), sr.top(), sx - sr.left(), sr.height()),
                                Qt.ClipOperation.IntersectClip)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            painter.drawPixmap(sr, self._compare_pix, QRectF(self._compare_pix.rect()))
            painter.restore()
            pen = QPen(Qt.GlobalColor.white)
            pen.setCosmetic(True)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(QPointF(sx, sr.top()), QPointF(sx, sr.bottom()))
        if self._overlay_paint is not None:
            self._overlay_paint(painter, rect)
