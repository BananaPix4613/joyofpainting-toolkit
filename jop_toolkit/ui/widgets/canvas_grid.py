"""Per-tile painting view: faded full grid + pulsing highlight on the active color.

Ported from the original overlay.paintEvent rendering, decoupled from any
OS-level transparent window — it's now just a regular QWidget.
"""

from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QMouseEvent
from PyQt6.QtWidgets import QWidget


class CanvasGridWidget(QWidget):
    cell_picked = pyqtSignal(str)  # emits the hex color of the clicked cell

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixels = None         # 2D list of hex strings
        self._highlight_hex = None
        self._pulse = 0.0
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_tile(self, pixel_grid):
        self._pixels = pixel_grid
        self.update()

    def set_highlight(self, hex_color):
        self._highlight_hex = hex_color
        self.update()

    def _tick(self):
        self._pulse = (self._pulse + 0.08) % 1.0
        if self._highlight_hex:
            self.update()

    def _cell_size(self):
        if not self._pixels:
            return 0, 0, 0, 0, 0, 0
        rows = len(self._pixels)
        cols = len(self._pixels[0])
        # Fit grid into widget keeping aspect, centered.
        ws = self.width() / cols
        hs = self.height() / rows
        s = min(ws, hs)
        cw = s
        ch = s
        ox = (self.width() - cw * cols) / 2
        oy = (self.height() - ch * rows) / 2
        return rows, cols, cw, ch, ox, oy

    def _cell_at(self, x, y):
        if not self._pixels:
            return None
        rows, cols, cw, ch, ox, oy = self._cell_size()
        if cw == 0:
            return None
        cx = int((x - ox) / cw)
        cy = int((y - oy) / ch)
        if 0 <= cx < cols and 0 <= cy < rows:
            return cx, cy
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._pixels:
            pos = self._cell_at(event.pos().x(), event.pos().y())
            if pos:
                cx, cy = pos
                hex_color = self._pixels[cy][cx]
                self.cell_picked.emit(hex_color)
        super().mousePressEvent(event)

    def paintEvent(self, _ev):
        if not self._pixels:
            return
        rows, cols, cw, ch, ox, oy = self._cell_size()

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        p.fillRect(self.rect(), QColor('#202020'))

        # Background: faded full grid (alpha 70 — matches original overlay look).
        p.setPen(Qt.PenStyle.NoPen)
        for ry, row in enumerate(self._pixels):
            for cx, hex_color in enumerate(row):
                c = QColor(hex_color)
                c.setAlpha(70)
                p.setBrush(QBrush(c))
                p.drawRect(QRectF(ox + cx * cw, oy + ry * ch, cw, ch))

        # Pulsing highlight on the active color.
        if self._highlight_hex:
            pulse_alpha = int(120 + 100 * abs(0.5 - self._pulse) * 2)
            fill = QColor(self._highlight_hex)
            fill.setAlpha(pulse_alpha)
            # Pick an outline that contrasts the highlight fill: black for bright
            # colors, white for dark ones. Otherwise near-white highlights are
            # invisible against a white outline.
            r, g, b, _ = fill.getRgb()
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            outline = QColor('#000000' if luma > 160 else '#ffffff')
            outline.setAlpha(255)
            for ry, row in enumerate(self._pixels):
                for cx, hex_color in enumerate(row):
                    if hex_color != self._highlight_hex:
                        continue
                    p.setBrush(QBrush(fill))
                    p.setPen(QPen(outline, 1.5))
                    p.drawRect(QRectF(ox + cx * cw + 0.5, oy + ry * ch + 0.5, cw - 1, ch - 1))
