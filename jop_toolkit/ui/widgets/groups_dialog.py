"""Non-modal window for palette group control.

Lives outside the Source panel because none of this fits in a 340px column. The
number line is the primary editor; the table is an advanced tab for people who
want explicit anchors rather than positions along an axis.

The budget is not owned here. It instead mirrors the Source panel's color limit,
and edits push back to it, so there is exactly one number in the app that means
"how many colors am I willing to mix".
"""

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QTabWidget, QWidget, QGroupBox, QColorDialog, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal
import numpy as np

from ...core.colorspace import srgb_to_lab
from ...core import axis_core as AX
from ...core.colors import hex_to_rgb
from ...core.groups import ColorGroup
from .palette_axis import PaletteAxis
from .group_table import GroupTable


class GroupsDialog(QDialog):
    """Emits `applied` when anything that affects selection changes."""

    applied = pyqtSignal()
    budget_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Palette groups")
        self.setModal(False)
        self.resize(700, 460)
        self._palette = {}
        self._uq = self._uq_lab = self._w = None
        self._loading = False
        self._mins = []
        self._maxs = []
        self._src = None

        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Total palette:"))
        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(2, 4096)
        self.spin_budget.valueChanged.connect(self._on_budget)
        top.addWidget(self.spin_budget)
        top.addWidget(QLabel("colors  (shared with the Source panel's color limit)"))
        top.addStretch(1)
        lay.addLayout(top)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        page = QWidget()
        play = QVBoxLayout(page)
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Order colors by:"))
        self.cmb_axis = QComboBox()
        self.cmb_axis.addItems(list(AX.AXES))
        self.cmb_axis.currentIndexChanged.connect(self._rebuild_strip)
        arow.addWidget(self.cmb_axis)
        self.lbl_axis_q = QLabel("")
        arow.addWidget(self.lbl_axis_q)
        arow.addStretch(1)
        play.addLayout(arow)

        self.axis_widget = PaletteAxis()
        self.axis_widget.markers_changed.connect(self._on_markers_changed)
        self.axis_widget.selection_changed.connect(self._on_marker_selected)
        play.addWidget(self.axis_widget)
        play.addWidget(QLabel("Drag a marker to move it. Double-click the strip to add one, "
                              "right-click a marker to remove it.\nDashed lines are the "
                              "boundaries between markers."))

        self.box_marker = QGroupBox("Selected marker")
        mform = QFormLayout(self.box_marker)
        self.btn_marker_color = QPushButton("-")
        self.btn_marker_color.clicked.connect(self._choose_marker_color)
        self.chk_pinned = QCheckBox("Keep this color when the marker moves")
        self.chk_pinned.toggled.connect(self._on_pin_toggled)
        self.spin_min = QSpinBox(); self.spin_min.setRange(0, 512)
        self.spin_max = QSpinBox(); self.spin_max.setRange(0, 512)
        for s in (self.spin_min, self.spin_max):
            s.valueChanged.connect(self._on_markers_changed)
        self.lbl_marker_stats = QLabel("-")
        mform.addRow("Core color", self.btn_marker_color)
        mform.addRow(self.chk_pinned)
        mform.addRow("Min colors", self.spin_min)
        mform.addRow("Max colors", self.spin_max)
        mform.addRow("Coverage", self.lbl_marker_stats)
        self.box_marker.setEnabled(False)
        play.addWidget(self.box_marker)
        self.tabs.addTab(page, "Number line")

        self.table = GroupTable()
        self.table.changed.connect(lambda: self.applied.emit())
        self.table.btn_auto.clicked.connect(self._auto_detect)
        self.tabs.addTab(self.table, "Advanced (table)")

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)

    # ---------- data in ----------
    def set_source(self, src_rgb, palette):
        self._palette = palette or {}
        self.table.set_palette(self._palette)
        if src_rgb is None:
            self._uq = self._uq_lab = self._w = None
            self._src = None
            return
        self._src = np.asarray(src_rgb)
        src = self._src
        uq, cnt = np.unique(src.reshape(-1, 3).astype(np.int64), axis=0, return_counts=True)
        self._uq = uq.astype(np.float64)
        self._uq_lab = srgb_to_lab(self._uq)
        self._w = cnt.astype(np.float64)
        self._rebuild_strip()

    def set_budget(self, n):
        self._loading = True
        self.spin_budget.setValue(int(n or 0) or 2)
        self._loading = False
        self.table.set_budget(self.spin_budget.value())
        self._update_summary()

    def set_groups(self, groups, axis='lightness'):
        self._loading = True
        try:
            self.cmb_axis.setCurrentText(axis if axis in AX.AXES else 'lightness')
            self._mins = [g.min_colors for g in groups]
            self._maxs = [g.max_colors for g in groups]
            self.axis_widget.set_markers([g.position for g in groups],
                                         [g.core_hex for g in groups],
                                         [g.pinned for g in groups])
            self.table.set_groups(groups)
        finally:
            self._loading = False
        self._rebuild_strip()

    # ---------- data out ----------
    def groups(self):
        if self.tabs.currentIndex() == 1:
            return self.table.groups(), 'cores', self.cmb_axis.currentText()
        pos, col, pin = self.axis_widget.markers()
        self._sync_budget_lists(len(pos))
        out = [ColorGroup(core_hex=col[i], min_colors=self._mins[i],
                          max_colors=self._maxs[i], position=float(t),
                          pinned=bool(pin[i]))
               for i, t in enumerate(pos)]
        return out, 'axis', self.cmb_axis.currentText()

    def _sync_budget_lists(self, n):
        while len(self._mins) < n:
            self._mins.append(1)
        while len(self._maxs) < n:
            self._maxs.append(64)
        del self._mins[n:]
        del self._maxs[n:]

    # ---------- internals ----------
    def _snap_unpinned(self):
        """A derived color is a strip average, which may not be mixable. Snap it
        so the swatch shows the color the painting will actually use."""
        if not self._palette:
            return
        _, col, pin = self.axis_widget.markers()
        for i, (c, pinned) in enumerate(zip(col, pin)):
            if pinned or c in self._palette:
                continue
            snapped = self.table.snap_color(hex_to_rgb(c))
            if snapped != c:
                self.axis_widget.set_marker_color(i, snapped, pin=False)

    def _rebuild_strip(self):
        if self._uq is None:
            return
        axis = self.cmb_axis.currentText()
        self.axis_widget.set_strip(AX.build_strip(self._uq, self._uq_lab, self._w, axis, n=512))
        q = AX.axis_quality(self._uq_lab, self._w, axis)
        self.lbl_axis_q.setText(f"carries {100*q:.0f}% of this image's color variance")
        self.lbl_axis_q.setStyleSheet("color: #cc8800;" if q < 0.35 else "")
        if not self._loading:
            self._snap_unpinned()
            self._refresh_coverage()
            self._update_summary()

    def _auto_detect(self):
        if self._src is None or not self._palette:
            self.lbl_summary.setText("Apply once before auto-detecting groups.")
            return
        from ...core.groups import auto_groups
        self.table.set_palette(self._palette)
        self.table.set_groups(auto_groups(self._src, self._palette, 6))
        self.table.refresh_coverage(self._src)
        self.applied.emit()

    def _on_budget(self, v):
        self.table.set_budget(v)
        self._update_summary()
        if not self._loading:
            self.budget_changed.emit(int(v))
            self.applied.emit()

    def _on_markers_changed(self):
        if self._loading:
            return
        self._snap_unpinned()
        pos, _, _ = self.axis_widget.markers()
        self._sync_budget_lists(len(pos))
        i = self.axis_widget.selected()
        if 0 <= i < len(self._mins):
            self._mins[i] = self.spin_min.value()
            self._maxs[i] = self.spin_max.value()
        self._refresh_coverage()
        self._update_summary()
        self.applied.emit()

    def _on_marker_selected(self, idx):
        pos, col, pin = self.axis_widget.markers()
        self._sync_budget_lists(len(pos))
        ok = 0 <= idx < len(pos)
        self.box_marker.setEnabled(ok)
        self._loading = True
        try:
            if ok:
                c = QColor(col[idx])
                fg = "#000000" if (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) > 140 else "#ffffff"
                self.btn_marker_color.setText(col[idx])
                self.btn_marker_color.setStyleSheet(f"background-color: {col[idx]}; color: {fg};")
                self.chk_pinned.setChecked(bool(pin[idx]))
                self.spin_min.setValue(int(self._mins[idx]))
                self.spin_max.setValue(int(self._maxs[idx]))
            else:
                self.btn_marker_color.setText("-")
                self.btn_marker_color.setStyleSheet("")
        finally:
            self._loading = False
        self._refresh_coverage()

    def _choose_marker_color(self):
        i = self.axis_widget.selected()
        if i < 0:
            return
        c = QColorDialog.getColor(QColor(self.btn_marker_color.text()), self,
                                  "Core color for this marker")
        if not c.isValid():
            return
        self.axis_widget.set_marker_color(i, self.table.snap_color((c.red(), c.green(), c.blue())),
                                          pin=True)
        self._on_marker_selected(i)
        self.applied.emit()

    def _on_pin_toggled(self, on):
        if self._loading:
            return
        i = self.axis_widget.selected()
        if i < 0:
            return
        if on:
            _, col, _ = self.axis_widget.markers()
            self.axis_widget.set_marker_color(i, col[i], pin=True)
        else:
            self.axis_widget.unpin(i)
            self._on_marker_selected(i)
        self.applied.emit()

    def _refresh_coverage(self):
        if self._uq_lab is None:
            return
        pos, _, _ = self.axis_widget.markers()
        if not pos:
            self.lbl_marker_stats.setText("-")
            return
        gi = AX.assign_by_markers(self._uq_lab, self._w, self.cmb_axis.currentText(), pos)
        i = self.axis_widget.selected()
        if 0 <= i < len(pos):
            self.lbl_marker_stats.setText(
                f"{100.0 * self._w[gi == i].sum() / self._w.sum():.1f}% of the image")
        else:
            self.lbl_marker_stats.setText("-")

    def _update_summary(self):
        pos, _, _ = self.axis_widget.markers()
        self._sync_budget_lists(len(pos))
        budget = self.spin_budget.value()
        lo, hi = sum(self._mins), sum(self._maxs)
        if not pos:
            self.lbl_summary.setText("No markers - selection falls back to automatic.")
            return
        msg = f"{len(pos)} markers asking for {lo}-{hi} of {budget} colors."
        if lo > budget:
            msg += f"  Minimums exceed the budget by {lo - budget}; they will be scaled down."
        elif hi < budget:
            msg += f"  Maximums leave {budget - hi} colors unused."
        self.lbl_summary.setText(msg)
