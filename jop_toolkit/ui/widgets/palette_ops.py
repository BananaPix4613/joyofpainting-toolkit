"""Add detail / Reduce colors / Select lines, as Edit-menu dialogs.

Both live here rather than in the right-hand panel: they are occasional, and the
panel is for what you reference constantly. Both follow the same contract, which
is why they share a base:

  * nothing is computed until you ask for it. An earlier design planned the whole
    curve so a slider could scrub it for free; on an image processed with no
    color limit that is a 3265-color palette and the plan never returns.
  * preview is off by default, and when on it computes only the target you have
    actually set, only when the slider is release rather than while dragging.
  * every computation goes behind a busy popup, because several hundred
    milliseconds of frozen window reads as a crash.
  * closing by any route puts the image back; only Apply keeps anything.
"""

from contextlib import contextmanager

import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QFormLayout, QLabel, QCheckBox,
    QComboBox, QSlider, QSpinBox, QPushButton, QProgressBar, QDialogButtonBox,
    QFrame,
)

from ...core import refine, reduce as red, lines as lines_mod


class SettleSlider(QSlider):
    """A slider that says when its value has finished changing.

    Preview is expensive, so it must run once per intent rather than once per
    event. The three ways to move a slider end differently: a drag finishes with
    sliderReleased, a key press is one discrete change, and a wheel spin is a
    burst with no end marker at all. Only the wheel needs a timer, restarted on
    every notch so it fires once the spin stops.

    Without this, wheel and arrow changes produced no sliderReleased, so the
    canvas kept showing the previous target while the label showed the new one.
    """

    settled = pyqtSignal()
    WHEEL_SETTLE_MS = 250

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(self.WHEEL_SETTLE_MS)
        self._settle.timeout.connect(self.settled)
        self.sliderReleased.connect(self._settle_now)

    def _settle_now(self):
        self._settle.stop()
        self.settled.emit()

    def wheelEvent(self, e):
        super().wheelEvent(e)
        self._settle.start()          # restarts: fires once the spin stops

    def keyPressEvent(self, e):
        before = self.value()
        super().keyPressEvent(e)
        if self.value() != before:
            self._settle_now()


class BusyDialog(QDialog):
    """Indeterminate 'working' popup. No cancel: these operations are short
    enough that a half-finished one would be more confusing than waiting."""

    def __init__(self, parent=None, text="Working..."):
        super().__init__(parent)
        self.setWindowTitle("Working")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(text))
        bar = QProgressBar()
        bar.setRange(0, 0)           # indeterminate
        bar.setTextVisible(False)
        lay.addWidget(bar)
        self.setMinimumWidth(300)


@contextmanager
def busy(parent, text):
    dlg = BusyDialog(parent, text)
    dlg.show()
    QApplication.processEvents()     # let it actually paint before we block
    try:
        yield
    finally:
        dlg.close()
        dlg.deleteLater()


class _PreviewDialog(QDialog):
    """Shared shell: target controls, optional preview, Apply/Cancel."""

    BUSY_TEXT = "Working..."

    def __init__(self, stage, title, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.state = stage.state
        self.setWindowTitle(title)
        self.setModal(True)
        self._base = stage.state.indexed.copy() if stage.state.indexed else None
        self._result = None
        self._shown_for = None       # the target the preview currently shows

        self.form = QFormLayout()
        root = QVBoxLayout(self)
        root.addLayout(self.form)

        self.lbl_info = QLabel("-")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_info.setMinimumHeight(56)
        root.addWidget(self.lbl_info)

        self.chk_preview = QCheckBox("Preview on canvas")
        root.addWidget(self.chk_preview)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply |
                                        QDialogButtonBox.StandardButton.Cancel)
        self.btn_apply = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.btn_apply.clicked.connect(self._on_apply)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.chk_preview.toggled.connect(self._on_preview_toggled)
        self.setMinimumWidth(380)

    # ---- subclass contract ----
    def _target(self):
        raise NotImplementedError

    def _compute(self):
        """Return a result object for the current target, or None."""
        raise NotImplementedError

    def _apply_result(self, result):
        """Mutate state.indexed. Return a changed mask."""
        raise NotImplementedError

    def _describe(self, result):
        raise NotImplementedError

    def _history_label(self):
        return "edit"

    # ---- shared behavior ----
    def _restore(self):
        if self._base is not None:
            self.state.indexed = self._base.copy()
            self.stage.refresh()

    def _invalidate(self):
        """The target changed: whatever is on screen no longer matches it."""
        self._result = None
        if self._shown_for is not None:
            self._restore()
            self._shown_for = None
        self._refresh_text()

    def _ensure(self):
        if self._result is None:
            with busy(self, self.BUSY_TEXT):
                self._result = self._compute()
        return self._result

    def _do_preview(self):
        if not self.chk_preview.isChecked() or self._base is None:
            return
        target = self._target()
        if self._shown_for == target and self._result is not None:
            return
        self._restore()
        result = self._ensure()
        if result:
            with busy(self, "Rendering preview..."):
                self._apply_result(result)
            self.stage.refresh()
            self._shown_for = target
        self._refresh_text()

    def _on_preview_toggled(self, on):
        if on:
            self._do_preview()
        else:
            self._restore()
            self._shown_for = None
            self._refresh_text()

    def _on_apply(self):
        if self._base is None:
            self.reject()
            return
        result = self._ensure()
        if not result:
            self._refresh_text()
            return
        # Rewind first so the undo snapshot is the pre-operation image rather
        # than whatever the preview left on screen.
        self.state.indexed = self._base.copy()
        self.stage.commit_edit(self._history_label(), lambda: self._apply_result(result))
        self._base = None            # nothing to restore: the change is kept
        self.accept()

    def _refresh_text(self):
        self.lbl_info.setText(self._describe(self._result))

    def reject(self):
        self._restore()
        super().reject()

    def closeEvent(self, e):
        self._restore()
        super().closeEvent(e)


class ReduceDialog(_PreviewDialog):
    """Reduce to N / merge similar / limit the selection."""

    BUSY_TEXT = "Working out the reduction..."
    SIZE, SIMILAR, REGION = 0, 1, 2

    def __init__(self, stage, parent=None):
        super().__init__(stage, "Reduce colors", parent)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Reduce to N colors",
                                "Merge similar colors",
                                "Limit selection to N colors"])
        self.chk_protect = QCheckBox("Preserve custom colors")
        self.chk_protect.setChecked(True)
        self.sld = SettleSlider(Qt.Orientation.Horizontal)
        self.lbl_target = QLabel("-")
        self.form.addRow("Mode", self.cmb_mode)
        self.form.addRow(self.chk_protect)
        self.form.addRow("Target", self.sld)
        self.form.addRow("", self.lbl_target)

        self.cmb_mode.currentIndexChanged.connect(self._on_mode)
        self.chk_protect.toggled.connect(lambda _c: self._invalidate())
        # valueChanged only moves the label; the work waits for the mouse to
        # come up, so dragging never queues a pile of computations.
        self.sld.valueChanged.connect(self._on_value)
        self.sld.settled.connect(self._do_preview)
        self._on_mode(0)

    def _on_mode(self, _i=0):
        ix = self.state.indexed
        mode = self.cmb_mode.currentIndex()
        self.chk_protect.setEnabled(mode != self.REGION)
        n = len(ix.slots) if ix else 2
        self.sld.blockSignals(True)
        if mode == self.SIZE:
            self.sld.setRange(2, max(2, n))
            self.sld.setValue(max(2, min(n, 32)))
        elif mode == self.SIMILAR:
            self.sld.setRange(1, 30)
            self.sld.setValue(3)
        else:
            sel = self.stage.selection
            used = 0
            if ix is not None and sel is not None:
                u = np.unique(ix.indices[sel])
                used = int((u >= 0).sum())
            self.sld.setRange(1, max(1, used))
            self.sld.setValue(max(1, min(used, 3)))
        self.sld.blockSignals(False)
        self._invalidate()

    def _on_value(self, _v):
        # Label only. The work waits for SettleSlider.settled, so neither
        # dragging nor spinning the wheel queues a pile of computations.
        self._result = None
        self._refresh_text()

    def _target(self):
        return (self.cmb_mode.currentIndex(), self.sld.value(),
                self.chk_protect.isChecked())

    def _protect(self):
        ix = self.state.indexed
        if ix is None or not self.chk_protect.isChecked():
            return None
        # Slot.locked means "came from the pipeline", so a color you added by
        # hand or that refinement introduced is exactly the un-locked set.
        return {s.hex for s in ix.slots if not s.locked}

    def _compute(self):
        ix, src = self.state.indexed, self.state.match_source
        if ix is None:
            return None
        mode = self.cmb_mode.currentIndex()
        if mode == self.SIZE:
            return red.reduce_to(ix, self.sld.value(), src, self._protect())
        if mode == self.SIMILAR:
            return red.merge_similar(ix, float(self.sld.value()), src, self._protect())
        if self.stage.selection is None:
            return None
        return red.limit_region(ix, self.stage.selection, self.sld.value(), src)

    def _apply_result(self, result):
        return red.apply_reduction(self.state.indexed, result)

    def _history_label(self):
        return "reduce colors"

    def _describe(self, result):
        mode = self.cmb_mode.currentIndex()
        v = self.sld.value()
        ix = self.state.indexed
        now = len(ix.slots) if ix else 0
        if mode == self.SIZE:
            self.lbl_target.setText(f"{v} colors")
        elif mode == self.SIMILAR:
            self.lbl_target.setText(f"within dE {v}")
        else:
            self.lbl_target.setText(f"{v} colors in the selection")
        if mode == self.REGION and self.stage.selection is None:
            return "Select an area first."
        if result is None:
            return f"Palette is {now} colors.\nApply, or tick preview, to work it out."
        if not result:
            return "Nothing to merge at this target."
        return result.summary()


class LinesDialog(QDialog):
    """Select the outlines, so limit_region can flatten them.
    
    Not a _PreviewDialog: this one never touches a pixel. Its whole output is a
    selection, and a selection is already its own preview, so there is nothing to
    rewind and nothing to compute twice. Detection is one morphological pass,
    cheap enough to rerun on every change.
    """

    def __init__(self, stage, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.state = stage.state
        self.setWindowTitle("Select lines")
        self.setModal(True)
        # Detection reads the selection you already had, so you can confine it
        # to one area; Cancel has to put that back.
        self._before = None if stage.selection is None else stage.selection.copy()

        form = QFormLayout()
        root = QVBoxLayout(self)
        root.addLayout(form)
        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 8)
        self.spin_width.setValue(1)
        self.spin_width.setToolTip("The widest line to catch, in pixels.")
        self.spin_thr = QSpinBox()
        self.spin_thr.setRange(1, 60)
        self.spin_thr.setValue(6)
        self.spin_thr.setToolTip("How far a line dips below its surroundings, in L*.")
        self.spin_min = QSpinBox()
        self.spin_min.setRange(1, 64)
        self.spin_min.setValue(3)
        self.spin_min.setToolTip("Discard runs shorter than this, which are speckle.")
        self.chk_light = QCheckBox("Light lines on a dark background")
        self.chk_within = QCheckBox("Only inside the current selection")
        self.chk_within.setEnabled(self._before is not None)
        form.addRow("Line width", self.spin_width)
        form.addRow("Darkness", self.spin_thr)
        form.addRow("Ignore runs under", self.spin_min)
        form.addRow(self.chk_light)
        form.addRow(self.chk_within)

        self.lbl = QLabel("-")
        self.lbl.setWordWrap(True)
        self.lbl.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl.setMinimumHeight(48)
        root.addWidget(self.lbl)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                        QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        for w in (self.spin_width, self.spin_thr, self.spin_min):
            w.valueChanged.connect(lambda _v: self._run())
        self.chk_light.toggled.connect(lambda _c: self._run())
        self.chk_within.toggled.connect(lambda _c: self._run())
        self.setMinimumWidth(360)
        self._run()

    def _run(self):
        src = self.state.match_source
        ix = self.state.indexed
        if ix is None or src is None or src.shape[:2] != ix.shape:
            self.lbl.setText("Apply changes on the Source stage first.")
            return
        within = self._before if self.chk_within.isChecked() else None
        mask = lines_mod.detect_lines(
            src, width=self.spin_width.value(), threshold=float(self.spin_thr.value()),
            min_size=self.spin_min.value(), within=within,
            light=self.chk_light.isChecked())
        self.stage.set_selection(mask)
        self.lbl.setText(lines_mod.describe(mask, ix))

    def reject(self):
        self.stage.set_selection(self._before)
        super().reject()

    def closeEvent(self, e):
        if self.result() != QDialog.DialogCode.Accepted:
            self.stage.set_selection(self._before)
        super().closeEvent(e)


class DetailDialog(_PreviewDialog):
    """Add colors for the selected area."""

    BUSY_TEXT = "Looking for colors..."

    def __init__(self, stage, parent=None):
        super().__init__(stage, "Add detail", parent)
        self.chk_range = QCheckBox("Also where these colors appear elsewhere")
        self.chk_dither = QCheckBox("Dither, using the Source stage settings")
        self.sld = SettleSlider(Qt.Orientation.Horizontal)
        self.sld.setRange(1, refine.MAX_ADD)
        self.sld.setValue(12)
        self.lbl_target = QLabel("-")
        self.form.addRow(self.chk_range)
        self.form.addRow(self.chk_dither)
        self.form.addRow("Colors to add", self.sld)
        self.form.addRow("", self.lbl_target)

        self.chk_range.toggled.connect(lambda _c: self._invalidate())
        self.chk_dither.toggled.connect(lambda _c: self._invalidate())
        self.sld.valueChanged.connect(self._on_value)
        self.sld.settled.connect(self._do_preview)
        self._refresh_text()

    def _on_value(self, _v):
        # Label only. The work waits for SettleSlider.settled, so neither
        # dragging nor spinning the wheel queues a pile of computations.
        self._result = None
        self._refresh_text()

    def _target(self):
        return (self.sld.value(), self.chk_range.isChecked(),
                self.chk_dither.isChecked())

    def _region(self):
        sel = self.stage.selection
        if sel is None:
            return None
        if self.chk_range.isChecked():
            return refine.color_range_mask(self.state.indexed, sel)
        return sel

    def _dither_opts(self):
        if not self.chk_dither.isChecked():
            return None
        return {'strength': self.state.dither_strength,
                'method': self.state.dither_method,
                'serpentine': self.state.dither_serpentine,
                'matrix_size': self.state.dither_matrix,
                'flat': getattr(self.state, 'dither_flat', 1.25)}

    def _compute(self):
        ix, src = self.state.indexed, self.state.match_source
        region = self._region()
        if ix is None or region is None or src is None or src.shape[:2] != ix.shape:
            return None
        plan = refine.plan(src, ix, region, self.state.palette, self.state.recipes,
                           max_add=self.sld.value())
        if not len(plan):
            return None
        return (plan, region)

    def _apply_result(self, result):
        plan, region = result
        return refine.apply_plan(self.state.indexed, self.state.match_source,
                                 region, plan, len(plan), self._dither_opts())

    def _history_label(self):
        return "add detail"

    def _describe(self, result):
        self.lbl_target.setText(f"up to {self.sld.value()}")
        ix, src = self.state.indexed, self.state.match_source
        if self.stage.selection is None:
            return "Select the area that lacks definition first."
        if ix is None or src is None or src.shape[:2] != ix.shape:
            return "Apply changes on the Source stage first."
        if result is None:
            n = int(self.stage.selection.sum())
            return (f"{n} px selected, palette is {len(ix.slots)} colors.\n"
                    "Apply, or tick preview, to work it out.")
        plan, region = result
        m0, q0 = plan.at(0)
        m1, q1 = plan.at(len(plan))
        return (f"+{len(plan)} colors over {plan.region_px} px\n"
                f"error here: {m0:.2f} -> {m1:.2f}  (worst {q0:.2f} -> {q1:.2f})\n"
                f"palette: {len(ix.slots)} -> {len(ix.slots) + len(plan)} colors")
