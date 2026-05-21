"""Stage 1: Source — load image, downscale, directional smear, color-match."""

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGridLayout, QPushButton,
    QSpinBox, QCheckBox, QLabel, QGroupBox, QScrollArea, QFrame, QFileDialog,
    QProgressDialog,
)
import numpy as np

from ..core import preprocess, palette as palette_mod, tiler
from ..core.colors import BASE_COLORS
from .widgets.zoom_view import ZoomView


class _Worker(QObject):
    finished = pyqtSignal(object, object, object, object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)  # done, total, label

    def __init__(self, source_image, target_resolution, margins, depth,
                 selected_dyes, color_limit):
        super().__init__()
        self.source_image = source_image
        self.target_resolution = target_resolution
        self.margins = margins
        self.depth = depth
        self.selected_dyes = selected_dyes
        self.color_limit = color_limit

    def run(self):
        try:
            # Steps: 1 downscale+pad, depth-1 palette-gen passes, 1 match, 1 limit (optional), 1 render
            n_gen = max(0, self.depth - 1)
            extra = 3 + (1 if self.color_limit else 0)
            total = n_gen + extra
            step = 0
            self.progress.emit(step, total, "Preprocessing image…")
            img = preprocess.downscale(self.source_image, self.target_resolution)
            arr = np.asarray(img, dtype=np.uint8)
            t, r, b, l = self.margins
            arr = preprocess.directional_margin(arr, t, r, b, l)
            step += 1; self.progress.emit(step, total, "Generating palette…")

            def gen_cb(done, _of, label):
                self.progress.emit(step + done, total, label)
            palette, recipes = palette_mod.generate_palette(
                self.selected_dyes, self.depth, progress_cb=gen_cb)
            step += n_gen
            self.progress.emit(step, total, "Matching colors…")
            matcher = palette_mod.Matcher(palette)
            matched = matcher.match_pixels(arr)
            step += 1
            if self.color_limit:
                self.progress.emit(step, total, "Reducing palette…")
                matched = palette_mod.limit_used_palette(matched, palette, self.color_limit)
                step += 1
            self.progress.emit(step, total, "Rendering preview…")
            processed = preprocess.hex_grid_to_rgb(matched, palette)
            self.progress.emit(total, total, "Done")
            self.finished.emit(processed, palette, recipes, matched)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SourceStage(QWidget):
    state_changed = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._aspect_locked = True

        root = QHBoxLayout(self)
        self.view = ZoomView()
        root.addWidget(self.view, stretch=1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(340)
        right = QWidget()
        right_scroll.setWidget(right)
        rcol = QVBoxLayout(right)
        root.addWidget(right_scroll)

        # Open
        self.btn_open = QPushButton("Open Image…")
        self.btn_open.clicked.connect(self._open_image)
        rcol.addWidget(self.btn_open)

        # Resolution
        res_box = QGroupBox("Target resolution")
        res_form = QFormLayout(res_box)
        self.spin_w = QSpinBox(); self.spin_w.setRange(8, 4096); self.spin_w.setValue(64)
        self.spin_h = QSpinBox(); self.spin_h.setRange(8, 4096); self.spin_h.setValue(64)
        self.chk_lock = QCheckBox("Lock aspect ratio"); self.chk_lock.setChecked(True)
        res_form.addRow("Width", self.spin_w)
        res_form.addRow("Height", self.spin_h)
        res_form.addRow(self.chk_lock)
        rcol.addWidget(res_box)
        self.spin_w.valueChanged.connect(self._on_w_changed)
        self.spin_h.valueChanged.connect(self._on_h_changed)
        self.chk_lock.toggled.connect(self._on_lock_toggled)

        # Margin (directional)
        margin_box = QGroupBox("Margin (smear / crop)")
        mgrid = QGridLayout(margin_box)
        self.spin_top = self._make_margin_spin()
        self.spin_right = self._make_margin_spin()
        self.spin_bottom = self._make_margin_spin()
        self.spin_left = self._make_margin_spin()
        mgrid.addWidget(QLabel("Top"), 0, 0); mgrid.addWidget(self.spin_top, 0, 1)
        mgrid.addWidget(QLabel("Right"), 1, 0); mgrid.addWidget(self.spin_right, 1, 1)
        mgrid.addWidget(QLabel("Bottom"), 2, 0); mgrid.addWidget(self.spin_bottom, 2, 1)
        mgrid.addWidget(QLabel("Left"), 3, 0); mgrid.addWidget(self.spin_left, 3, 1)
        mgrid.addWidget(QLabel("Negative shrinks; positive smears edge outward."), 4, 0, 1, 2)
        rcol.addWidget(margin_box)

        # Depth (with apply)
        depth_box = QGroupBox("Palette depth")
        dlay = QFormLayout(depth_box)
        self.spin_depth = QSpinBox(); self.spin_depth.setRange(1, 8); self.spin_depth.setValue(4)
        self.btn_apply_depth = QPushButton("Apply depth")
        self.btn_apply_depth.clicked.connect(self._apply_depth)
        self.lbl_depth_warn = QLabel("")
        self.lbl_depth_warn.setStyleSheet("color: #cc8800;")
        dlay.addRow("Depth", self.spin_depth)
        dlay.addRow(self.btn_apply_depth)
        dlay.addRow(self.lbl_depth_warn)
        rcol.addWidget(depth_box)

        # Color limit
        limit_box = QGroupBox("Palette color limit")
        llay = QFormLayout(limit_box)
        self.chk_limit = QCheckBox("Limit total colors")
        self.spin_limit = QSpinBox(); self.spin_limit.setRange(2, 4096); self.spin_limit.setValue(32)
        self.spin_limit.setEnabled(False)
        self.chk_limit.toggled.connect(self._on_limit_toggled)
        self.spin_limit.valueChanged.connect(self._schedule_recompute)
        llay.addRow(self.chk_limit)
        llay.addRow("Max colors", self.spin_limit)
        rcol.addWidget(limit_box)

        # Dye selection
        dyes_box = QGroupBox("Base dyes")
        dlay2 = QVBoxLayout(dyes_box)
        self.dye_checks = {}
        for name in BASE_COLORS:
            cb = QCheckBox(name); cb.setChecked(True)
            cb.toggled.connect(self._schedule_recompute)
            dlay2.addWidget(cb)
            self.dye_checks[name] = cb
        rcol.addWidget(dyes_box)

        # Stats
        self.lbl_stats = QLabel("No image loaded")
        self.lbl_stats.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_stats.setWordWrap(True)
        rcol.addWidget(self.lbl_stats)
        rcol.addStretch(1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(250)
        self._debounce.timeout.connect(self._recompute)
        self._progress_dialog: QProgressDialog | None = None
        self._show_progress = False

    def _make_margin_spin(self):
        s = QSpinBox()
        s.setRange(-256, 256)
        s.setValue(0)
        s.valueChanged.connect(self._schedule_recompute)
        return s

    # ---------- public ----------
    def controls_help(self) -> str:
        return ("Mouse-wheel: zoom preview  •  Click-drag: pan  •  "
                "Edit dyes/resolution/margins to refresh live  •  "
                "Apply depth to regenerate the palette")

    def refresh_from_state(self):
        if self.state.source_image is None:
            return
        self.spin_w.blockSignals(True); self.spin_h.blockSignals(True)
        self.spin_w.setValue(self.state.target_resolution[0])
        self.spin_h.setValue(self.state.target_resolution[1])
        self.spin_w.blockSignals(False); self.spin_h.blockSignals(False)
        t, r, b, l = getattr(self.state, "margins", (0, 0, 0, 0))
        for s, v in [(self.spin_top, t), (self.spin_right, r), (self.spin_bottom, b), (self.spin_left, l)]:
            s.blockSignals(True); s.setValue(int(v)); s.blockSignals(False)
        self.spin_depth.setValue(self.state.depth)
        for name, cb in self.dye_checks.items():
            cb.blockSignals(True); cb.setChecked(name in self.state.selected_dyes); cb.blockSignals(False)
        limit = getattr(self.state, "color_limit", None)
        self.chk_limit.blockSignals(True)
        self.chk_limit.setChecked(bool(limit))
        self.spin_limit.setEnabled(bool(limit))
        if limit:
            self.spin_limit.setValue(int(limit))
        self.chk_limit.blockSignals(False)
        self._recompute()

    # ---------- handlers ----------
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
        if not path:
            return
        from pathlib import Path
        img = preprocess.load_image(path)
        self.state.source_path = Path(path)
        self.state.source_image = img
        w, h = img.size
        self.spin_w.blockSignals(True); self.spin_h.blockSignals(True)
        self.spin_w.setValue(w); self.spin_h.setValue(h)
        self.spin_w.blockSignals(False); self.spin_h.blockSignals(False)
        self.state.target_resolution = (w, h)
        self._recompute()

    def _on_w_changed(self, v):
        if self._aspect_locked and self.state.source_image is not None:
            sw, sh = self.state.source_image.size
            new_h = max(8, round(v * sh / sw))
            self.spin_h.blockSignals(True); self.spin_h.setValue(new_h); self.spin_h.blockSignals(False)
        self._schedule_recompute()

    def _on_h_changed(self, v):
        if self._aspect_locked and self.state.source_image is not None:
            sw, sh = self.state.source_image.size
            new_w = max(8, round(v * sw / sh))
            self.spin_w.blockSignals(True); self.spin_w.setValue(new_w); self.spin_w.blockSignals(False)
        self._schedule_recompute()

    def _on_lock_toggled(self, on):
        self._aspect_locked = on

    def _on_limit_toggled(self, on):
        self.spin_limit.setEnabled(on)
        self._schedule_recompute()

    def _schedule_recompute(self):
        if self.state.source_image is None:
            return
        self._debounce.start()

    def _apply_depth(self):
        if self.state.source_image is None:
            return
        self._show_progress = True
        self._recompute()

    # ---------- pipeline ----------
    def _recompute(self):
        if self.state.source_image is None:
            return
        self.state.target_resolution = (self.spin_w.value(), self.spin_h.value())
        margins = (self.spin_top.value(), self.spin_right.value(),
                   self.spin_bottom.value(), self.spin_left.value())
        self.state.margins = margins
        # Back-compat: state.margin = the (now-meaningless) uniform value
        self.state.margin = max(margins)
        self.state.depth = self.spin_depth.value()
        self.state.selected_dyes = [n for n, cb in self.dye_checks.items() if cb.isChecked()]
        self.state.color_limit = self.spin_limit.value() if self.chk_limit.isChecked() else None
        self.lbl_depth_warn.setText("Slower; large palette" if self.state.depth >= 6 else "")
        if not self.state.selected_dyes:
            self.lbl_stats.setText("Select at least one dye.")
            return
        if self._thread is not None:
            self._thread.quit(); self._thread.wait(); self._thread = None
        self.lbl_stats.setText("Working…")
        self._worker = _Worker(
            self.state.source_image, self.state.target_resolution, margins,
            self.state.depth, list(self.state.selected_dyes), self.state.color_limit,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        if self._show_progress:
            dlg = QProgressDialog("Applying depth…", None, 0, 0, self)
            dlg.setWindowTitle("Generating palette")
            dlg.setMinimumDuration(0)
            dlg.setCancelButton(None)
            dlg.setAutoClose(True)
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            self._progress_dialog = dlg
            dlg.show()
            self._show_progress = False
        self._thread.start()

    def _on_progress(self, done, total, label):
        # Pin a local reference: setValue() pumps events, so _on_done can fire
        # mid-call and null self._progress_dialog before the next line runs.
        dlg = self._progress_dialog
        if dlg is None:
            return
        dlg.setMaximum(max(1, total))
        dlg.setLabelText(label)
        dlg.setValue(done)

    def _on_done(self, processed, palette, recipes, matched):
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        self.state.processed_image = processed
        self.state.palette = palette
        self.state.recipes = recipes
        self.state.matched_hex_grid = matched
        self.view.set_image(processed)
        h, w = processed.shape[:2]
        pad_w = (w + 15) // 16 * 16
        pad_h = (h + 15) // 16 * 16
        try:
            est_canvases = len(tiler.greedy_mixed_layout(pad_w, pad_h))
        except Exception:
            est_canvases = "?"
        # Distinct colors actually in the matched image
        unique_used = len({h_ for h_ in matched.reshape(-1).tolist()})
        self.lbl_stats.setText(
            f"Output: {w}×{h} px\n"
            f"Palette generated: {len(palette)} colors\n"
            f"Colors used in image: {unique_used}\n"
            f"Est. canvases (mixed): {est_canvases}"
        )
        self.state_changed.emit()

    def _on_failed(self, msg):
        if self._progress_dialog is not None:
            self._progress_dialog.close()
            self._progress_dialog = None
        self.lbl_stats.setText(f"Error: {msg}")
