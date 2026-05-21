"""Pan/zoom QGraphicsView with crisp nearest-neighbor pixel scaling."""

from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPixmap, QWheelEvent, QMouseEvent
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

    def drawForeground(self, painter, rect):
        if self._overlay_paint is not None:
            self._overlay_paint(painter, rect)
