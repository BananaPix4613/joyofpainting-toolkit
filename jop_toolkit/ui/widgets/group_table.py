"""Editable table of palette groups: core color, budget floor/ceiling, coverage.

Kept as its own widget because the interesting part is bookkeeping, not layout:
rows own live spin boxes and color buttons, and every edit has to round-trip
through ColorGroup without the table becoming the source of truth.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QColorDialog, QCheckBox,
)
import numpy as np

from ...core.colors import hex_to_rgb, rgb_to_hex
from ...core.colorspace import srgb_to_lab
from ...core.groups import ColorGroup, auto_groups, assign_groups

_COLS = ("On", "Core", "Cover", "Min", "Max", "Got")


class GroupTable(QWidget):
    """Rows of (enabled, core color, coverage %, min, max, allocated)."""

    changed = pyqtSignal()
    pick_requested = pyqtSignal(int)      # row index, or -1 to disarm

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette = {}
        self._loading = False
        self._budget = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (0, 2, 3, 4, 5):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(150)
        lay.addWidget(self.table)

        row = QHBoxLayout()
        self.btn_auto = QPushButton("Auto-detect")
        self.btn_add = QPushButton("Add")
        self.btn_remove = QPushButton("Remove")
        self.btn_pick = QPushButton("Pick from image")
        self.btn_pick.setCheckable(True)
        for b in (self.btn_auto, self.btn_add, self.btn_remove, self.btn_pick):
            row.addWidget(b)
        lay.addLayout(row)

        self.lbl_budget = QLabel("")
        self.lbl_budget.setWordWrap(True)
        lay.addWidget(self.lbl_budget)

        self.btn_add.clicked.connect(self._add_blank)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_pick.toggled.connect(self._on_pick_toggled)

    # ---------- palette ----------
    def set_palette(self, palette):
        """Needed so a picked color snaps to something actually mixable."""
        self._palette = palette or {}

    def snap_color(self, rgb):
        if not self._palette:
            return rgb_to_hex(tuple(int(v) for v in rgb))
        from scipy.spatial import cKDTree
        keys = list(self._palette.keys())
        tree = cKDTree(srgb_to_lab(np.array(list(self._palette.values()), dtype=np.float64)))
        _, i = tree.query(srgb_to_lab(np.asarray(rgb, dtype=np.float64)), k=1)
        return keys[int(i)]

    # ---------- rows ----------
    def _make_row(self, group):
        r = self.table.rowCount()
        self.table.insertRow(r)

        chk = QCheckBox()
        chk.setChecked(bool(group.enabled))
        chk.toggled.connect(self._emit_changed)
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(chk, alignment=Qt.AlignmentFlag.AlignCenter)
        self.table.setCellWidget(r, 0, holder)

        btn = QPushButton(group.core_hex)
        btn.clicked.connect(lambda _=False, row=r: self._choose_color(row))
        self._paint_button(btn, group.core_hex)
        self.table.setCellWidget(r, 1, btn)

        cover = QTableWidgetItem("-")
        cover.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(r, 2, cover)

        for col, val in ((3, group.min_colors), (4, group.max_colors)):
            sp = QSpinBox()
            sp.setRange(0, 512)
            sp.setValue(int(val))
            sp.valueChanged.connect(self._emit_changed)
            self.table.setCellWidget(r, col, sp)

        got = QTableWidgetItem("-")
        got.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.table.setItem(r, 5, got)

    @staticmethod
    def _paint_button(btn, hex_color):
        r, g, b = hex_to_rgb(hex_color)
        fg = "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#ffffff"
        btn.setStyleSheet(f"background-color: {hex_color}; color: {fg};")
        btn.setText(hex_color)

    def _add_blank(self):
        self._make_row(ColorGroup(core_hex="#808080"))
        self._emit_changed()

    def _remove_selected(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._rebind_color_buttons()
            self._emit_changed()

    def _rebind_color_buttons(self):
        """Row indices shift after a removal, so the lambdas must be re-bound or
        the color button edits the wrong row."""
        for r in range(self.table.rowCount()):
            btn = self.table.cellWidget(r, 1)
            if btn is not None:
                try:
                    btn.clicked.disconnect()
                except TypeError:
                    pass
                btn.clicked.connect(lambda _=False, row=r: self._choose_color(row))

    def _choose_color(self, row):
        btn = self.table.cellWidget(row, 1)
        if btn is None:
            return
        col = QColorDialog.getColor(QColor(btn.text()), self, "Core color for this group")
        if col.isValid():
            self.set_row_color(row, self.snap_color((col.red(), col.green(), col.blue())))

    def set_row_color(self, row, hex_color):
        btn = self.table.cellWidget(row, 1)
        if btn is not None:
            self._paint_button(btn, hex_color)
            self._emit_changed()

    def _on_pick_toggled(self, on):
        self.pick_requested.emit(self.table.currentRow() if on else -1)

    # ---------- data ----------
    def groups(self):
        out = []
        for r in range(self.table.rowCount()):
            holder = self.table.cellWidget(r, 0)
            btn = self.table.cellWidget(r, 1)
            mn = self.table.cellWidget(r, 3)
            mx = self.table.cellWidget(r, 4)
            if None in (holder, btn, mn, mx):
                continue
            chk = holder.findChild(QCheckBox)
            out.append(ColorGroup(core_hex=btn.text(),
                                  min_colors=mn.value(),
                                  max_colors=mx.value(),
                                  enabled=chk.isChecked() if chk else True))
        return out

    def set_groups(self, groups):
        self._loading = True
        try:
            self.table.setRowCount(0)
            for g in groups:
                self._make_row(g)
        finally:
            self._loading = False
        self._update_budget_label()

    def detect(self, src_rgb, palette, n_groups=6):
        self.set_palette(palette)
        self.set_groups(auto_groups(src_rgb, palette, n_groups))
        self._emit_changed()

    # ---------- stats ----------
    def refresh_coverage(self, src_rgb):
        """Coverage is what makes min/max meaningful otherwise you budget blind."""
        gs = [g for g in self.groups() if g.enabled]
        if src_rgb is None or not gs:
            for r in range(self.table.rowCount()):
                self.table.item(r, 2).setText("-")
            return
        src = np.asarray(src_rgb)
        uq, cnt = np.unique(src.reshape(-1, 3).astype(np.int64), axis=0, return_counts=True)
        gi = assign_groups(srgb_to_lab(uq.astype(np.float64)), gs)
        total = float(cnt.sum())
        share = {k: (100.0 * float(cnt[gi == k].sum()) / total if total else 0.0)
                 for k in range(len(gs))}
        k = 0
        for r in range(self.table.rowCount()):
            holder = self.table.cellWidget(r, 0)
            chk = holder.findChild(QCheckBox) if holder else None
            if chk is not None and chk.isChecked():
                self.table.item(r, 2).setText(f"{share.get(k, 0.0):.0f}%")
                k += 1
            else:
                self.table.item(r, 2).setText("off")

    def set_allocation(self, alloc):
        k = 0
        for r in range(self.table.rowCount()):
            holder = self.table.cellWidget(r, 0)
            chk = holder.findChild(QCheckBox) if holder else None
            if chk is not None and chk.isChecked() and k < len(alloc):
                self.table.item(r, 5).setText(str(int(alloc[k])))
                k += 1
            else:
                self.table.item(r, 5).setText("-")

    def set_budget(self, budget):
        self._budget = int(budget) if budget else 0
        self._update_budget_label()

    def _update_budget_label(self):
        gs = [g for g in self.groups() if g.enabled]
        if not gs:
            self.lbl_budget.setText("No active groups - falling back to automatic selection.")
            return
        lo = sum(g.min_colors for g in gs)
        hi = sum(g.max_colors for g in gs)
        msg = f"Groups ask for {lo}-{hi} colors; budget is {self._budget}."
        if self._budget and lo > self._budget:
            msg += f"  Minimums exceed the budget by {lo - self._budget}; they will be scaled down."
        elif self._budget and hi < self._budget:
            msg += f"  Maximums cap out {self._budget - hi} below the budget; those go unused."
        self.lbl_budget.setText(msg)

    def _emit_changed(self):
        if self._loading:
            return
        self._update_budget_label()
        self.changed.emit()
