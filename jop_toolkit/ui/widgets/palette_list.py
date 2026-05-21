"""Scrollable color list showing swatch, hex, and dye recipe; emits selection."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QIcon
from PyQt6.QtWidgets import QListWidget, QListWidgetItem


def _swatch(hex_color, size=18):
    pix = QPixmap(size, size)
    pix.fill(QColor(hex_color))
    return QIcon(pix)


class PaletteListWidget(QListWidget):
    color_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(self.iconSize().scaled(20, 20, Qt.AspectRatioMode.IgnoreAspectRatio))
        self.itemSelectionChanged.connect(self._emit_selection)
        self._entries = []

    def set_entries(self, entries):
        """entries = [{'hex': ..., 'rgb': [r,g,b], 'recipe': [dye, ...]}, ...]."""
        self.blockSignals(True)
        self.clear()
        self._entries = list(entries)
        for e in self._entries:
            recipe = ' → '.join(e['recipe'])
            text = f"{e['hex']}   {recipe}"
            item = QListWidgetItem(_swatch(e['hex']), text)
            item.setData(Qt.ItemDataRole.UserRole, e['hex'])
            self.addItem(item)
        self.blockSignals(False)
        if self.count():
            self.setCurrentRow(0)

    def select_hex(self, hex_color):
        for i in range(self.count()):
            if self.item(i).data(Qt.ItemDataRole.UserRole) == hex_color:
                self.setCurrentRow(i)
                return

    def current_hex(self):
        item = self.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _emit_selection(self):
        h = self.current_hex()
        if h:
            self.color_selected.emit(h)
