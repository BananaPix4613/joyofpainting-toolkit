"""Stage 2: Layout — arrange canvas tiles over the processed image grid."""

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QPen, QFont, QBrush
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QRadioButton, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QMessageBox,
)
import numpy as np

from ..core import tiler, preprocess
from .widgets.zoom_view import ZoomView


CANVAS_SIZE_OPTIONS = ["mixed", "16x16", "16x32", "32x16", "32x32"]


def _parse_size(s):
    w, h = s.split("x")
    return (int(w), int(h))


class LayoutStage(QWidget):
    layout_finalized = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._padded_size = (0, 0)
        self._manual_brush = (32, 32)
        self._highlight_index: int | None = None

        root = QHBoxLayout(self)
        self.view = ZoomView()
        self.view.cell_clicked.connect(self._on_canvas_click)
        self.view.set_overlay_paint(self._paint_overlay)
        root.addWidget(self.view, stretch=1)

        right = QWidget()
        right.setFixedWidth(280)
        rcol = QVBoxLayout(right)
        root.addWidget(right)

        mode_box = QGroupBox("Mode")
        mlay = QVBoxLayout(mode_box)
        self.radio_auto = QRadioButton("Auto"); self.radio_auto.setChecked(True)
        self.radio_manual = QRadioButton("Manual")
        mlay.addWidget(self.radio_auto); mlay.addWidget(self.radio_manual)
        self.radio_auto.toggled.connect(self._on_mode_changed)
        rcol.addWidget(mode_box)

        # Auto controls
        self.auto_box = QGroupBox("Auto packing")
        alay = QVBoxLayout(self.auto_box)
        self.cmb_size = QComboBox()
        self.cmb_size.addItems(CANVAS_SIZE_OPTIONS)
        self.cmb_size.currentTextChanged.connect(lambda _: self._recompute_auto())
        alay.addWidget(self.cmb_size)
        rcol.addWidget(self.auto_box)

        # Manual controls
        self.manual_box = QGroupBox("Manual brush")
        man_l = QVBoxLayout(self.manual_box)
        man_l.addWidget(QLabel("Click on the image to place a canvas.\nRight-click to remove."))
        self.cmb_brush = QComboBox()
        self.cmb_brush.addItems(["16x16", "16x32", "32x16", "32x32"])
        self.cmb_brush.setCurrentText("32x32")
        self.cmb_brush.currentTextChanged.connect(lambda t: setattr(self, "_manual_brush", _parse_size(t)))
        man_l.addWidget(QLabel("Canvas size to place:"))
        man_l.addWidget(self.cmb_brush)
        self.btn_clear = QPushButton("Clear all")
        self.btn_clear.clicked.connect(self._clear_manual)
        man_l.addWidget(self.btn_clear)
        self.manual_box.setVisible(False)
        rcol.addWidget(self.manual_box)

        # Tile list
        rcol.addWidget(QLabel("Canvases:"))
        self.lst = QListWidget()
        self.lst.currentRowChanged.connect(self._on_select_row)
        rcol.addWidget(self.lst, stretch=1)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        rcol.addWidget(self.lbl_status)

    def controls_help(self) -> str:
        if self.radio_manual.isChecked():
            return ("Manual mode  •  Left-click image: place canvas at brush size  •  "
                    "Right-click: remove canvas  •  Mouse-wheel: zoom  •  Drag: pan")
        return ("Auto mode  •  Pick a packing size to repack  •  "
                "Mouse-wheel: zoom  •  Drag: pan")

    # ---------- entered from main window ----------
    def enter(self):
        """Called when this stage becomes visible."""
        if self.state.matched_hex_grid is None:
            self.lbl_status.setText("Load and process an image first.")
            return
        self._refresh_image()
        if self.state.layout_mode == "manual":
            self.radio_manual.setChecked(True)
            self._refresh_list()
        else:
            self.radio_auto.setChecked(True)
            self._recompute_auto()

    def _refresh_image(self):
        # Pad processed image to multiples of 16 so layouts cover everything.
        padded = preprocess.pad_to_multiple(self.state.processed_image, 16)
        self._padded_size = (padded.shape[1], padded.shape[0])  # (W, H)
        # Also re-pad the matched grid so finalize step can slice with the layout.
        h, w = self.state.matched_hex_grid.shape
        pad_h = (-h) % 16
        pad_w = (-w) % 16
        if pad_h or pad_w:
            grid = np.pad(self.state.matched_hex_grid,
                          ((0, pad_h), (0, pad_w)), mode='edge')
        else:
            grid = self.state.matched_hex_grid
        self.state.matched_hex_grid = grid
        self.state.processed_image = padded
        self.view.set_image(padded, fit=True)

    # ---------- mode ----------
    def _on_mode_changed(self):
        if self.radio_auto.isChecked():
            self.state.layout_mode = "auto"
            self.auto_box.setVisible(True)
            self.manual_box.setVisible(False)
            self._recompute_auto()
        else:
            self.state.layout_mode = "manual"
            self.auto_box.setVisible(False)
            self.manual_box.setVisible(True)
            # Seed from current auto layout if manual is empty
            if not self.state.layout:
                self._recompute_auto()
            self._refresh_list()

    def _recompute_auto(self):
        w, h = self._padded_size
        if not w or not h:
            return
        mode = self.cmb_size.currentText() if self.radio_auto.isChecked() else "mixed"
        try:
            if mode == "mixed":
                layout = tiler.greedy_mixed_layout(w, h)
            else:
                layout = tiler.uniform_layout(w, h, _parse_size(mode))
        except ValueError as exc:
            self.lbl_status.setText(f"Error: {exc}")
            return
        self.state.layout = layout
        self._refresh_list()
        self.view.viewport().update()

    # ---------- manual placement ----------
    def _on_canvas_click(self, x, y, button):
        if not self.radio_manual.isChecked():
            return
        gx = (x // 16) * 16
        gy = (y // 16) * 16
        if button == int(Qt.MouseButton.RightButton.value):
            self._remove_at(gx, gy)
            return
        if button == int(Qt.MouseButton.LeftButton.value):
            self._place_at(gx, gy, self._manual_brush)

    def _remove_at(self, px, py):
        new_layout = []
        for p in self.state.layout:
            x, y = p['pixel_pos']
            w, h = p['tile_size']
            if x <= px < x + w and y <= py < y + h:
                continue
            new_layout.append(p)
        if len(new_layout) != len(self.state.layout):
            self.state.layout = new_layout
            self._reindex(); self._refresh_list(); self.view.viewport().update()

    def _place_at(self, px, py, size):
        w, h = size
        pw, ph = self._padded_size
        if px + w > pw or py + h > ph:
            self.lbl_status.setText("Canvas would extend past the image.")
            return
        # Reject overlaps
        for p in self.state.layout:
            ex, ey = p['pixel_pos']
            ew, eh = p['tile_size']
            if not (px + w <= ex or ex + ew <= px or py + h <= ey or ey + eh <= py):
                self.lbl_status.setText("Overlaps an existing canvas.")
                return
        self.state.layout.append({'pixel_pos': [px, py], 'tile_size': [w, h]})
        self._reindex(); self._refresh_list(); self.view.viewport().update()
        self.lbl_status.setText("")

    def _clear_manual(self):
        self.state.layout = []
        self._refresh_list(); self.view.viewport().update()

    def _reindex(self):
        # Sort top-to-bottom, left-to-right and renumber
        self.state.layout.sort(key=lambda p: (p['pixel_pos'][1], p['pixel_pos'][0]))

    # ---------- list ----------
    def _refresh_list(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        for i, p in enumerate(self.state.layout):
            x, y = p['pixel_pos']; w, h = p['tile_size']
            self.lst.addItem(QListWidgetItem(f"#{i}  {w}×{h}  @ ({x},{y})"))
        self.lst.blockSignals(False)
        # Coverage stats
        w, h = self._padded_size
        covered = sum(p['tile_size'][0] * p['tile_size'][1] for p in self.state.layout)
        total = w * h
        if total:
            pct = 100 * covered / total
            self.lbl_status.setText(f"{len(self.state.layout)} canvases • coverage {pct:.0f}%")

    def _on_select_row(self, row):
        self._highlight_index = row if row >= 0 else None
        self.view.viewport().update()

    # ---------- overlay ----------
    def _paint_overlay(self, painter, _rect):
        if not self.state.layout:
            return
        painter.save()
        font = QFont(); font.setPointSize(8); font.setBold(True)
        painter.setFont(font)
        for i, p in enumerate(self.state.layout):
            x, y = p['pixel_pos']; w, h = p['tile_size']
            is_active = (i == self._highlight_index)
            pen = QPen(QColor('#ffffff' if not is_active else '#ffd400'))
            pen.setCosmetic(True)
            pen.setWidth(2 if is_active else 1)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(x, y, w, h))
            # Index badge
            painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRectF(x, y, 14, 10))
            painter.setPen(QPen(QColor('#ffffff')))
            painter.drawText(QRectF(x, y, 14, 10), Qt.AlignmentFlag.AlignCenter, str(i))
        painter.restore()

    # ---------- finalize ----------
    def finalize(self) -> bool:
        """Build per-tile hex grids on state.tiles. Returns False if not ready."""
        if not self.state.layout:
            QMessageBox.warning(self, "No canvases", "Add at least one canvas first.")
            return False
        from ..core.sidecar import slice_tiles, build_palette_entries
        self.state.tiles = slice_tiles(self.state.matched_hex_grid, self.state.layout)
        self.state.palette_entries = build_palette_entries(
            self.state.tiles, self.state.palette, self.state.recipes)
        self.state.active_tile_index = 0
        self.state.active_color_hex = (
            self.state.palette_entries[0]['hex'] if self.state.palette_entries else None)
        return True
