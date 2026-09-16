"""Step-by-step mixing guide for the Paint stage.

Steps are grouped by the color they finish, because that is how painting goes:
pick a color, mix it, paint it. Moving to a color in the palette list jumps here
to the first step of that color, and finishing a color here selects it there.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from ...core import mixing
from .palette_card import PaletteCard


class MixingPanel(QWidget):
    color_changed = pyqtSignal(str)      # the color the current step works toward

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plan = mixing.MixPlan()
        self.index = 0
        self._group = (0, -1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.lbl_summary = QLabel("-")
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)

        head = QHBoxLayout()
        self.swatch = QLabel()
        self.swatch.setFixedSize(28, 28)
        self.swatch.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_color = QLabel("-")
        head.addWidget(self.swatch)
        head.addWidget(self.lbl_color, 1)
        lay.addLayout(head)

        self.card = PaletteCard()
        lay.addWidget(self.card, 1)

        self.lbl_step = QLabel("-")
        self.lbl_step.setWordWrap(True)
        f = self.lbl_step.font()
        f.setPointSizeF(f.pointSizeF() + 2)
        f.setBold(True)
        self.lbl_step.setFont(f)
        lay.addWidget(self.lbl_step)

        self.list = QListWidget()
        self.list.setMaximumHeight(120)
        self.list.currentRowChanged.connect(self._on_list_row)
        lay.addWidget(self.list)

        nav = QHBoxLayout()
        self.btn_prev_color = QPushButton("◀ Color")
        self.btn_prev = QPushButton("◀ Step")
        self.btn_next = QPushButton("Step ▶")
        self.btn_next_color = QPushButton("Color ▶")
        for b in (self.btn_prev_color, self.btn_prev, self.btn_next, self.btn_next_color):
            nav.addWidget(b)
        lay.addLayout(nav)
        self.btn_prev.clicked.connect(lambda: self.go(self.index - 1))
        self.btn_next.clicked.connect(lambda: self.go(self.index + 1))
        self.btn_prev_color.clicked.connect(lambda: self.jump_color(-1))
        self.btn_next_color.clicked.connect(lambda: self.jump_color(+1))
        self._refresh()

    # ---------- data ----------
    def set_plan(self, plan):
        self.plan = plan
        self.card.set_used_dyes({s.dye for s in plan.steps if s.action == 'add'})
        self.lbl_summary.setText(plan.summary())
        self._group = (0, -1)
        self.go(0, announce=False)

    def show_color(self, hex_color):
        """Jump to the first step of a color, unless already working on it."""
        t = self.plan.step_for(hex_color) if hex_color else None
        if t is None:
            return
        first, last = self.plan.group(t)
        if first <= self.index <= last:
            return
        self.go(first, announce=False)

    # ---------- navigation ----------
    def go(self, i, announce=True):
        if not self.plan.steps:
            self._refresh()
            return
        i = max(0, min(int(i), len(self.plan.steps) - 1))
        old = self._group
        self.index = i
        self._group = self.plan.group(i)
        self._refresh(rebuild_list=self._group != old)
        if announce and self._group != old:
            self.color_changed.emit(self.plan.steps[self._group[1]].hex)

    def jump_color(self, delta):
        targets = self.plan.targets
        if not targets:
            return
        first, last = self._group
        if delta > 0:
            nxt = [t for t in targets if t > last]
            if nxt:
                self.go(self.plan.group(nxt[0])[0])
        else:
            prv = [t for t in targets if t < first]
            if prv:
                self.go(self.plan.group(prv[-1])[0])

    def _on_list_row(self, row):
        if row >= 0:
            target = self._group[0] + row
            if target != self.index:
                self.go(target)

    # ---------- display ----------
    def _refresh(self, rebuild_list=True):
        steps = self.plan.steps
        has = bool(steps)
        for b in (self.btn_prev, self.btn_next, self.btn_prev_color, self.btn_next_color):
            b.setEnabled(has)
        if not has:
            self.card.clear_step()
            self.lbl_color.setText("Nothing to mix yet.")
            self.lbl_step.setText("")
            self.swatch.setStyleSheet("")
            self.list.clear()
            return

        st = steps[self.index]
        first, last = self._group
        goal = steps[last].hex
        targets = self.plan.targets
        n_color = targets.index(last) + 1 if last in targets else 0
        self.swatch.setStyleSheet(f"background-color: {goal};")
        self.lbl_color.setText(f"Color {n_color} of {len(targets)}  ·  {goal}")
        self.card.set_step(st.spots, dye=st.dye or None,
                           water=st.action == 'wash', spot=st.spot)
        done = "  -  this color is ready, paint with it" if st.target else ""
        self.lbl_step.setText(f"Step {self.index + 1} of {len(steps)}: {st.describe()}{done}")

        if rebuild_list:
            self.list.blockSignals(True)
            self.list.clear()
            for k in range(first, last + 1):
                s = steps[k]
                text = s.describe().replace("the highlighted spot", "the spot")
                item = QListWidgetItem(f"{k - first + 1}. {text}")
                self.list.addItem(item)
            self.list.blockSignals(False)
        self.list.blockSignals(True)
        self.list.setCurrentRow(self.index - first)
        self.list.blockSignals(False)
        self.btn_prev.setEnabled(self.index > 0)
        self.btn_next.setEnabled(self.index < len(steps) - 1)
        self.btn_prev_color.setEnabled(first > 0)
        self.btn_next_color.setEnabled(last < len(steps) - 1)
