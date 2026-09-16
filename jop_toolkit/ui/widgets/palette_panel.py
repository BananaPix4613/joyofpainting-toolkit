"""Indexed palette panel: swatch grid plus the operations on the selected slot.

Every color that lands in a slot is snapped to something the mod can actually
mix, so a slot always carries a usable recipe. Two operations look similar but
differ in what happens to the slot count:

    Change color   recolors the slot in place; pixel count and slot count hold
    Replace with   merges this slot into another; this slot disappears

History is requested from the host before each mutation, so the panel never has
to know how undo is implemented.
"""

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QColorDialog, QMessageBox, QFrame,
)

from ...core.colors import hex_to_rgb
from ...core.colorspace import srgb_to_lab
from ...core.indexed import EMPTY
from .palette_grid import PaletteGrid
from .slot_picker import SlotPicker

SORTS = ("usage", "lightness", "hue", "recipe length")


class PalettePanel(QWidget):
    request_history = pyqtSignal(str)      # host snapshots before we mutate
    changed = pyqtSignal(object)           # we mutated; host should refresh; carries bool mask
    color_picked = pyqtSignal(int, int)    # slot, Qt button value

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Palette")
        blay = QVBoxLayout(box)
        self.grid = PaletteGrid()
        blay.addWidget(self.grid)

        row = QHBoxLayout()
        self.cmb_size = QComboBox()
        self.cmb_size.addItems(["16", "24", "32"])
        self.cmb_size.setCurrentText("24")
        self.cmb_sort = QComboBox()
        self.cmb_sort.addItems(SORTS)
        self.btn_sort = QPushButton("Sort")
        row.addWidget(QLabel("Swatch"))
        row.addWidget(self.cmb_size)
        row.addWidget(self.cmb_sort)
        row.addWidget(self.btn_sort)
        blay.addLayout(row)

        row2 = QHBoxLayout()
        self.btn_add = QPushButton("Add color")
        self.btn_dup = QPushButton("Duplicate")
        self.btn_clean = QPushButton("Drop unused")
        for b in (self.btn_add, self.btn_dup, self.btn_clean):
            row2.addWidget(b)
        blay.addLayout(row2)
        lay.addWidget(box)

        self.det = QGroupBox("Selected color")
        dlay = QFormLayout(self.det)
        self.sw = QLabel()
        self.sw.setFrameShape(QFrame.Shape.StyledPanel)
        self.sw.setMinimumHeight(36)
        self.lbl_hex = QLabel("-")
        self.lbl_use = QLabel("-")
        self.lbl_recipe = QLabel("-")
        self.lbl_recipe.setWordWrap(True)
        self.btn_change = QPushButton("Change color...")
        self.btn_replace = QPushButton("Replace with...")
        self.btn_remove = QPushButton("Remove")
        dlay.addRow(self.sw)
        dlay.addRow("Hex", self.lbl_hex)
        dlay.addRow("Used by", self.lbl_use)
        dlay.addRow("Recipe", self.lbl_recipe)
        dlay.addRow(self.btn_change)
        dlay.addRow(self.btn_replace)
        dlay.addRow(self.btn_remove)
        lay.addWidget(self.det)

        self.lbl_totals = QLabel("-")
        self.lbl_totals.setFrameShape(QFrame.Shape.StyledPanel)
        lay.addWidget(self.lbl_totals)

        self.grid.slot_picked.connect(self.color_picked)
        self.grid.slot_selected.connect(lambda _k: self._refresh_detail())
        self.cmb_size.currentTextChanged.connect(lambda t: self.grid.set_swatch_size(int(t)))
        self.btn_sort.clicked.connect(self._sort)
        self.btn_add.clicked.connect(self._add)
        self.btn_dup.clicked.connect(self._duplicate)
        self.btn_clean.clicked.connect(self._drop_unused)
        self.btn_change.clicked.connect(self._change_color)
        self.btn_replace.clicked.connect(self._replace_with)
        self.btn_remove.clicked.connect(self._remove)

    # ---------- public ----------
    def refresh(self):
        self.grid.set_indexed(self.state.indexed)
        self._refresh_detail()
        self._refresh_totals()

    def selected(self):
        return self.grid.selected()

    def select_slot(self, k):
        """Point the panel at a slot without treating it as a fresh pick.

        Used when the selection is driven from outside the grid, e.g. the Edit
        stage's eyedropper landing on a color that is already in the palette.
        The guard keeps the grid from re-emitting slot_selected when the grid
        itself is what originated the pick.
        """
        if self.grid.selected() != k:
            self.grid.set_selected(k)

    # ---------- snapping ----------
    def snap(self, rgb):
        """Nearest mixable color, with its recipe. A slot with no recipe is
        useless downstream, so nothing enters the palette unsnapped."""
        pal = self.state.palette or {}
        if not pal:
            from ...core.colors import rgb_to_hex
            return rgb_to_hex(tuple(int(v) for v in rgb)), []
        from scipy.spatial import cKDTree
        keys = list(pal.keys())
        tree = cKDTree(srgb_to_lab(np.array(list(pal.values()), dtype=np.float64)))
        _, i = tree.query(srgb_to_lab(np.asarray(rgb, dtype=np.float64)), k=1)
        hx = keys[int(i)]
        rec = self.state.recipes.get(hx, []) if self.state.recipes else []
        return hx, list(rec)

    # ---------- operations ----------
    def _ix(self):
        return self.state.indexed

    def _change_color(self):
        ix, k = self._ix(), self.grid.selected()
        if ix is None or k < 0 or k >= len(ix.slots):
            return
        c = QColorDialog.getColor(QColor(ix.slots[k].hex), self, "Change this color")
        if not c.isValid():
            return
        hx, rec = self.snap((c.red(), c.green(), c.blue()))
        if hx == ix.slots[k].hex:
            return
        touched = ix.indices == k          # captured before the slot moves
        self.request_history.emit("change color")
        ix.set_slot_color(k, hx, rec)
        self.changed.emit(touched)

    def _replace_with(self):
        ix, k = self._ix(), self.grid.selected()
        if ix is None or k < 0 or k >= len(ix.slots) or len(ix.slots) < 2:
            return
        dlg = SlotPicker(ix, exclude=k, parent=self,
                         prompt=f"Replace slot {k} with which color?")
        if not dlg.exec() or dlg.chosen() is None:
            return
        target = dlg.chosen()
        n = int(ix.used_counts()[k])
        touched = ix.indices == k
        self.request_history.emit("replace color")
        ix.merge_slot(k, target)
        self.changed.emit(touched)
        QMessageBox.information(self, "Replaced",
                                f"{n} pixels now use {ix.slots[min(target, len(ix.slots) - 1)].hex}.")

    def _remove(self):
        ix, k = self._ix(), self.grid.selected()
        if ix is None or k < 0 or k >= len(ix.slots):
            return
        n = int(ix.used_counts()[k])
        if n and QMessageBox.question(
                self, "Remove color",
                f"{n} pixels use this color. They become empty. Continue?") \
                != QMessageBox.StandardButton.Yes:
            return
        touched = ix.indices == k
        self.request_history.emit("remove color")
        ix.remove_slot(k)
        self.changed.emit(touched)

    def _duplicate(self):
        ix, k = self._ix(), self.grid.selected()
        if ix is None or k < 0 or k >= len(ix.slots):
            return
        self.request_history.emit("duplicate color")
        ix.add_slot(ix.slots[k].hex, ix.slots[k].recipe)
        self.changed.emit(None)

    def _add(self):
        ix = self._ix()
        if ix is None:
            return
        c = QColorDialog.getColor(QColor("#808080"), self, "Add a color")
        if not c.isValid():
            return
        hx, rec = self.snap((c.red(), c.green(), c.blue()))
        self.request_history.emit("add color")
        ix.add_slot(hx, rec)
        self.changed.emit(None)

    def _drop_unused(self):
        ix = self._ix()
        if ix is None:
            return
        dead = [i for i, c in enumerate(ix.used_counts()) if c == 0]
        if not dead:
            QMessageBox.information(self, "Nothing to drop", "Every color is in use.")
            return
        self.request_history.emit("drop unused colors")
        ix.compact()
        self.changed.emit(None)

    def _sort(self):
        ix = self._ix()
        if ix is None or not ix.slots:
            return
        mode = self.cmb_sort.currentText()
        counts = ix.used_counts()
        lab = srgb_to_lab(np.array([hex_to_rgb(s.hex) for s in ix.slots], dtype=np.float64))
        if mode == "usage":
            key = -counts.astype(float)
        elif mode == "lightness":
            key = lab[:, 0]
        elif mode == "hue":
            key = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
        else:
            key = np.array([len(s.recipe) for s in ix.slots], dtype=float)
        self.request_history.emit(f"sort by {mode}")
        ix.reorder(list(np.argsort(key, kind="stable")))
        self.changed.emit(None)

    # ---------- display ----------
    def _refresh_detail(self):
        ix, k = self._ix(), self.grid.selected()
        on = ix is not None and 0 <= k < len(ix.slots)
        for wdg in (self.btn_change, self.btn_replace, self.btn_remove, self.btn_dup):
            wdg.setEnabled(on)
        if not on:
            self.sw.setStyleSheet("")
            self.lbl_hex.setText("erase (no paint)" if k == EMPTY else "-")
            self.lbl_use.setText("-")
            self.lbl_recipe.setText("-")
            return
        s = ix.slots[k]
        n = int(ix.used_counts()[k])
        total = int(ix.indices.size)
        self.sw.setStyleSheet(f"background-color: {s.hex};")
        self.lbl_hex.setText(f"slot {k} · {s.hex}")
        self.lbl_use.setText(f"{n} px ({100.0 * n / max(total, 1):.1f}%)"
                             + ("  - unused" if n == 0 else ""))
        self.lbl_recipe.setText(" ".join(s.recipe) + f"  ({len(s.recipe)} drops)"
                                if s.recipe else "no recipe")

    def _refresh_totals(self):
        ix = self._ix()
        if ix is None:
            self.lbl_totals.setText("-")
            return
        counts = ix.used_counts()
        used = int((counts > 0).sum())
        drops = sum(len(s.recipe) for s, c in zip(ix.slots, counts) if c > 0)
        self.lbl_totals.setText(f"{len(ix.slots)} slots · {used} in use · {drops} drops to mix")
