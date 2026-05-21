"""Stage 3: Paint — active canvas view + mini overview + palette list."""

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSplitter,
    QRadioButton, QButtonGroup,
)

from .widgets.canvas_grid import CanvasGridWidget
from .widgets.palette_list import PaletteListWidget
from .widgets.zoom_view import ZoomView


class PaintStage(QWidget):
    status_changed = pyqtSignal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left column: large canvas grid
        left = QWidget()
        llay = QVBoxLayout(left)
        self.canvas = CanvasGridWidget()
        self.canvas.cell_picked.connect(self._on_canvas_cell_pick)
        llay.addWidget(self.canvas, stretch=1)
        nav = QHBoxLayout()
        self.btn_prev_tile = QPushButton("◀ Tile")
        self.btn_next_tile = QPushButton("Tile ▶")
        self.btn_prev_tile.clicked.connect(lambda: self._step_tile(-1))
        self.btn_next_tile.clicked.connect(lambda: self._step_tile(+1))
        nav.addWidget(self.btn_prev_tile); nav.addStretch(1); nav.addWidget(self.btn_next_tile)
        llay.addLayout(nav)
        splitter.addWidget(left)

        # Right column: mini overview + palette list
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.addWidget(QLabel("Overview (click a tile):"))
        self.overview = ZoomView()
        self.overview.setFixedHeight(220)
        self.overview.cell_clicked.connect(self._on_overview_click)
        self.overview.set_overlay_paint(self._paint_overview)
        rlay.addWidget(self.overview)
        # Palette filter mode
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Show:"))
        self.radio_all = QRadioButton("All colors in image"); self.radio_all.setChecked(True)
        self.radio_tile = QRadioButton("Colors in this canvas only")
        group = QButtonGroup(self); group.addButton(self.radio_all); group.addButton(self.radio_tile)
        self.radio_all.toggled.connect(self._refresh_palette_list)
        filt_row.addWidget(self.radio_all); filt_row.addWidget(self.radio_tile); filt_row.addStretch(1)
        rlay.addLayout(filt_row)
        self.palette = PaletteListWidget()
        self.palette.color_selected.connect(self._on_color_selected)
        rlay.addWidget(self.palette, stretch=1)
        cnav = QHBoxLayout()
        self.btn_prev_color = QPushButton("◀ Color")
        self.btn_next_color = QPushButton("Color ▶")
        self.btn_prev_color.clicked.connect(lambda: self._step_color(-1))
        self.btn_next_color.clicked.connect(lambda: self._step_color(+1))
        cnav.addWidget(self.btn_prev_color); cnav.addStretch(1); cnav.addWidget(self.btn_next_color)
        rlay.addLayout(cnav)
        splitter.addWidget(right)
        splitter.setSizes([700, 400])

    def controls_help(self) -> str:
        return ("Left-click canvas cell: pick that color  •  "
                "↑/↓ or ◀/▶ Color: cycle palette  •  "
                "PgUp/PgDn or ◀/▶ Tile: switch canvas  •  "
                "Click overview tile: switch canvas")

    def enter(self):
        if not self.state.tiles or not self.state.palette_entries:
            return
        self.overview.set_image(self.state.processed_image, fit=True)
        self._refresh_palette_list()
        if self.state.active_color_hex:
            self.palette.select_hex(self.state.active_color_hex)
        self._show_tile(self.state.active_tile_index)

    def _refresh_palette_list(self):
        if not self.state.palette_entries:
            return
        if self.radio_tile.isChecked() and 0 <= self.state.active_tile_index < len(self.state.tiles):
            tile = self.state.tiles[self.state.active_tile_index]
            in_tile = {c for row in tile['pixels'] for c in row}
            entries = [e for e in self.state.palette_entries if e['hex'] in in_tile]
        else:
            entries = self.state.palette_entries
        self.palette.set_entries(entries)
        if self.state.active_color_hex:
            self.palette.select_hex(self.state.active_color_hex)

    # ---------- tile selection ----------
    def _show_tile(self, idx):
        if not (0 <= idx < len(self.state.tiles)):
            return
        self.state.active_tile_index = idx
        tile = self.state.tiles[idx]
        self.canvas.set_tile(tile['pixels'])
        # If filtering by current canvas, the palette list contents change with the tile.
        if self.radio_tile.isChecked():
            self._refresh_palette_list()
        self.canvas.set_highlight(self.state.active_color_hex)
        self.overview.viewport().update()
        self._emit_status()

    def _step_tile(self, delta):
        n = len(self.state.tiles)
        if n:
            self._show_tile((self.state.active_tile_index + delta) % n)

    def _on_overview_click(self, x, y, button):
        for i, t in enumerate(self.state.tiles):
            tx, ty = t['pixel_pos']; tw, th = t['tile_size']
            if tx <= x < tx + tw and ty <= y < ty + th:
                self._show_tile(i)
                return

    # ---------- color selection ----------
    def _on_color_selected(self, hex_color):
        self.state.active_color_hex = hex_color
        self.canvas.set_highlight(hex_color)
        self._emit_status()

    def _on_canvas_cell_pick(self, hex_color):
        self.state.active_color_hex = hex_color
        # If the color isn't in the filtered list, switch back to "all" so it can show.
        idx_in_list = None
        for i in range(self.palette.count()):
            if self.palette.item(i).data(Qt.ItemDataRole.UserRole) == hex_color:
                idx_in_list = i; break
        if idx_in_list is None and self.radio_tile.isChecked():
            self.radio_all.setChecked(True)  # triggers _refresh_palette_list
        self.palette.select_hex(hex_color)

    def _step_color(self, delta):
        n = self.palette.count()
        if not n:
            return
        row = (self.palette.currentRow() + delta) % n
        self.palette.setCurrentRow(row)

    # ---------- overlay ----------
    def _paint_overview(self, painter, _rect):
        if not self.state.tiles:
            return
        painter.save()
        for i, t in enumerate(self.state.tiles):
            x, y = t['pixel_pos']; w, h = t['tile_size']
            is_active = (i == self.state.active_tile_index)
            pen = QPen(QColor('#ffd400' if is_active else '#ffffff'))
            pen.setCosmetic(True)
            pen.setWidth(3 if is_active else 1)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(x, y, w, h))
        painter.restore()

    def _emit_status(self):
        tiles = len(self.state.tiles)
        colors = len(self.state.palette_entries)
        ci = (self.palette.currentRow() + 1) if self.palette.currentRow() >= 0 else 0
        # Count occurrences of active color in active tile
        count = 0
        if self.state.active_color_hex and 0 <= self.state.active_tile_index < tiles:
            for row in self.state.tiles[self.state.active_tile_index]['pixels']:
                count += sum(1 for c in row if c == self.state.active_color_hex)
        self.status_changed.emit(
            f"Tile {self.state.active_tile_index + 1}/{tiles}  •  "
            f"Color {ci}/{colors}  •  Pixels of this color in tile: {count}"
        )

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Right):
            self._step_color(+1); return
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Left):
            self._step_color(-1); return
        if key == Qt.Key.Key_PageDown:
            self._step_tile(+1); return
        if key == Qt.Key.Key_PageUp:
            self._step_tile(-1); return
        super().keyPressEvent(event)
