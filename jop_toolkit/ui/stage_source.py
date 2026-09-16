"""Stage 1: Source - load image, downscale, directional smear, color-match."""

import time
from math import comb

from PyQt6.QtCore import Qt, QObject, QThread, QEvent, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGridLayout, QPushButton,
    QSpinBox, QCheckBox, QLabel, QGroupBox, QScrollArea, QFrame, QFileDialog,
    QDialog, QProgressBar, QComboBox,
)
import numpy as np

from ..core import (preprocess, palette as palette_mod, tiler, gamut,
                    dither as dither_mod, artifacts)
from ..core.colors import BASE_COLORS
from ..core.indexed import IndexedImage
from .widgets.zoom_view import ZoomView

# Measured palette-build cost, 16 dyes, one-time per settings change.
_DEPTH_COST = {9: "~1 s", 10: "~3 s", 11: "~8 s", 12: "~19 s"}


class _ProgressDialog(QDialog):
    """Drop-in for QProgressDialog without its reentrancy.

    QProgressDialog.setValue() calls processEvents() while modal, so a fast
    progress stream makes setValue re-enter itself; the queued finished signal
    lands mid-call and the dialog re-shows itself as that call unwinds. A plain
    QProgressBar has no such behaviour.
    """

    def __init__(self, parent=None, on_cancel=None):
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        lay = QVBoxLayout(self)
        self._label = QLabel("Starting...")
        self._label.setWordWrap(True)
        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self.btn_cancel = QPushButton("Cancel")
        lay.addWidget(self._label)
        lay.addWidget(self._bar)
        lay.addWidget(self.btn_cancel)
        self.setMinimumWidth(340)
        if on_cancel is not None:
            self.btn_cancel.clicked.connect(on_cancel)

    def setLabelText(self, text):
        self._label.setText(text)

    def setValue(self, permille):
        self._bar.setValue(permille)

    def reject(self):
        pass          # Esc must not dismiss it; use Cancel


class _Cancelled(Exception):
    """Raised out of a progress callback when the user cancels."""


class _NoWheelFilter(QObject):
    """Spin boxes inside a scroll area swallow wheel events and silently change
    values while the user is only trying to scroll the panel. Ignore the wheel
    unless the widget has been deliberately focused."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return False


def _phase_weights(depth, n_pixels, n_dyes, color_limit, dither_strength,
                   n_source_px=0, dither_method='floyd-steinberg'):
    """Relative cost of each pipeline phase, so the bar tracks work done rather
    than steps completed. Which phase dominates swings enormously with settings,
    so a flat step count would sit still for most of a long run. Only the ratios
    matter, the ETA is extrapolated from the observed rate."""
    multisets = sum(comb(n_dyes - 1 + d, d) for d in range(1, depth + 1))
    per_px = dither_mod.METHOD_COST.get(dither_method, 2.0e-5)
    return {
        'preprocess': n_pixels * 1e-6 + n_source_px * 4e-7,
        'palette': multisets * 1.5e-6,
        'gamut': n_pixels * 3e-6,
        'select': (1.0 + multisets * 2e-7) if color_limit else 0.0,
        'match': n_pixels * 1e-6,
        'dither': n_pixels * per_px if dither_strength > 0 else 0.0,
        'render': n_pixels * 3e-6,
    }


class _Worker(QObject):
    finished = pyqtSignal(object, object, object, object, object, object, object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, str)  # permille 0..1000, label

    def __init__(self, source_image, target_resolution, margins, depth,
                 selected_dyes, color_limit, gamut_strength, dither_strength, pre,
                 dither_method='floyd-steinberg', dither_serpentine=True, dither_matrix=8,
                 dither_flat=1.25):
        super().__init__()
        self.source_image = source_image
        self.target_resolution = target_resolution
        self.margins = margins
        self.depth = depth
        self.selected_dyes = selected_dyes
        self.color_limit = color_limit
        self.gamut_strength = gamut_strength
        self.dither_strength = dither_strength
        self.dither_method = dither_method
        self.dither_serpentine = dither_serpentine
        self.dither_matrix = dither_matrix
        self.dither_flat = dither_flat
        self.pre = pre
        self._abort = False
        self._weights = {}
        self._total = 1.0
        self._done = 0.0
        self._cur = ''
        self._label = ''

    def abort(self):
        self._abort = True

    # ---------- progress plumbing ----------
    def _phase(self, name, label):
        self._cur = name
        self._label = label
        self._sub(0.0)

    def _sub(self, frac):
        if self._abort:
            raise _Cancelled()
        w = self._weights.get(self._cur, 0.0)
        done = self._done + w * max(0.0, min(1.0, frac))
        pm = int(1000 * done / self._total) if self._total > 0 else 0
        self.progress.emit(max(0, min(1000, pm)), self._label)

    def _end_phase(self):
        self._done += self._weights.get(self._cur, 0.0)

    def run(self):
        try:
            tw, th = self.target_resolution
            n_px = max(1, int(tw) * int(th))
            sw, sh = self.source_image.size
            n_dyes = len(self.selected_dyes) if self.selected_dyes else len(BASE_COLORS)
            self._weights = _phase_weights(self.depth, n_px, n_dyes, self.color_limit,
                                           self.dither_strength, n_source_px=sw * sh,
                                           dither_method=self.dither_method)
            self._total = sum(self._weights.values()) or 1.0
            self._done = 0.0

            self._phase('preprocess', "Preprocessing image...")
            p = self.pre
            full = artifacts.preprocess_full(
                self.source_image,
                deblock_strength=p['deblock'], denoise_mode=p['denoise_mode'],
                denoise_radius=p['denoise_radius'], denoise_strength=p['denoise_strength'],
                median_radius=p['median_radius'])
            img = artifacts.downscale_to(full, self.target_resolution, p['resample'])
            arr = np.asarray(img, dtype=np.uint8)
            arr = artifacts.snap_colors(arr, p['snap'])
            # At target resolution, after the downscale: the soft pixels it acts
            # on are created by the downscale. Before the margins, so the edge
            # smear they pad with is not itself mistaken for an edge.
            arr = artifacts.harden_edges(arr, p.get('harden', 0.0),
                                         p.get('harden_threshold', 12.0))
            t, r, b, l = self.margins
            arr = preprocess.directional_margin(arr, t, r, b, l)
            sup = max(1, min(4, int(min(sw / max(1, tw), sh / max(1, th)))))
            comp = preprocess.downscale(self.source_image, (tw * sup, th * sup),
                                        artifacts.RESAMPLE_FILTERS['box'])
            comp_arr = np.asarray(comp, dtype=np.uint8)
            comp_arr = preprocess.directional_margin(
                comp_arr, t * sup, r * sup, b * sup, l * sup)
            self._end_phase()

            self._phase('palette', f"Generating depth-{self.depth} palette...")

            def gen_cb(done_units, total_units, label):
                self._label = label
                self._sub(done_units / max(1, total_units))

            palette, recipes = palette_mod.generate_palette(
                self.selected_dyes, self.depth, progress_cb=gen_cb)
            self._end_phase()

            self._phase('gamut', "Mapping to reachable gamut...")
            l_min, l_max = gamut.palette_l_range(palette)
            src = gamut.compress_lightness(arr, l_min, l_max,
                                           strength=self.gamut_strength)
            src = np.clip(src, 0, 255).astype(np.int64)
            self._end_phase()

            if self.color_limit:
                self._phase('select', f"Selecting {self.color_limit} colors...")
                # Palette groups used to branch here. They redistributed the
                # global budget, so tuning one group shifted every other area;
                # the Edit-menu Add detail and Reduce colors replace them without
                # that coupling. `alloc` stays in the signal as an empty list so
                # the arity _on_done expects does not change.
                used, alloc = palette_mod.select_palette(
                    src, palette, self.color_limit), []
                sub = {h: palette[h] for h in used}
                self._end_phase()
            else:
                used, sub, alloc = None, palette, []

            self._phase('match', "Matching colors...")
            matched = palette_mod.Matcher(sub).match_pixels(src)
            self._end_phase()

            if self.dither_strength > 0:
                self._phase('dither', "Dithering...")
                keys = used if used is not None else list(sub.keys())
                sub_rgb = np.array([sub[h] for h in keys], dtype=np.float64)
                di = dither_mod.dither(
                    src, sub_rgb, strength=self.dither_strength,
                    mask=dither_mod.ramp_mask(src, self.dither_flat),
                    method=self.dither_method,
                    serpentine=self.dither_serpentine,
                    matrix_size=self.dither_matrix,
                    progress_cb=lambda d, t_: self._sub(d / max(1, t_)))
                matched = np.array(keys, dtype=object)[di]
                self._end_phase()

            self._phase('render', "Rendering preview...")
            processed = preprocess.hex_grid_to_rgb(matched, palette)
            self._end_phase()

            self.progress.emit(1000, "Done")
            self.finished.emit(processed, palette, recipes, matched, comp_arr, alloc,
                               src.astype(np.uint8))
        except _Cancelled:
            self.cancelled.emit()
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
        self._no_wheel = _NoWheelFilter(self)
        root.addWidget(right_scroll)

        # Open
        self.btn_open = QPushButton("Open Image…")
        self.btn_open.clicked.connect(self._open_image)
        rcol.addWidget(self.btn_open)

        # Resolution
        res_box = QGroupBox("Target resolution")
        res_form = QFormLayout(res_box)
        self.spin_w = self._guard(QSpinBox()); self.spin_w.setRange(8, 4096); self.spin_w.setValue(64)
        self.spin_h = self._guard(QSpinBox()); self.spin_h.setRange(8, 4096); self.spin_h.setValue(64)
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

        # Depth
        depth_box = QGroupBox("Palette depth")
        dlay = QFormLayout(depth_box)
        self.spin_depth = self._guard(QSpinBox()); self.spin_depth.setRange(1, 12); self.spin_depth.setValue(6)
        self.spin_depth.valueChanged.connect(self._mark_dirty)
        self.lbl_depth_warn = QLabel("")
        self.lbl_depth_warn.setStyleSheet("color: #cc8800;")
        dlay.addRow("Depth", self.spin_depth)
        dlay.addRow(self.lbl_depth_warn)
        rcol.addWidget(depth_box)

        # Color limit
        limit_box = QGroupBox("Palette color limit")
        llay = QFormLayout(limit_box)
        self._limit_form = llay
        self.chk_limit = QCheckBox("Limit total colors")
        self.spin_limit = self._guard(QSpinBox()); self.spin_limit.setRange(2, 4096); self.spin_limit.setValue(32)
        self.spin_limit.setEnabled(False)
        self.chk_limit.toggled.connect(self._on_limit_toggled)
        self.spin_limit.valueChanged.connect(self._mark_dirty)
        llay.addRow(self.chk_limit)
        llay.addRow("Max colors", self.spin_limit)
        rcol.addWidget(limit_box)
        self._on_limit_toggled(self.chk_limit.isChecked())

        # Gamut + dither
        tone_box = QGroupBox("Tone handling")
        tlay = QFormLayout(tone_box)
        self._tone_form = tlay
        self.spin_gamut = self._guard(QSpinBox()); self.spin_gamut.setRange(0, 100); self.spin_gamut.setValue(0)
        self.spin_gamut.setSuffix(" %")
        self.spin_dither = self._guard(QSpinBox()); self.spin_dither.setRange(0, 100); self.spin_dither.setValue(0)
        self.spin_dither.setSuffix(" %")
        self.cmb_dither = QComboBox(); self.cmb_dither.addItems(list(dither_mod.METHODS))
        self.cmb_dither.setCurrentText("none")
        self.chk_serpentine = QCheckBox("Alternate row direction"); self.chk_serpentine.setChecked(False)
        self.cmb_matrix = QComboBox(); self.cmb_matrix.addItems(["2", "4", "8"])
        self.cmb_matrix.setCurrentText("8")
        self.cmb_dither.currentIndexChanged.connect(self._mark_dirty)
        self.cmb_dither.currentIndexChanged.connect(self._sync_dither_rows)
        self.spin_dither.valueChanged.connect(self._sync_dither_rows)
        self.chk_serpentine.toggled.connect(self._mark_dirty)
        self.cmb_matrix.currentIndexChanged.connect(self._mark_dirty)
        tlay.addRow("Method", self.cmb_dither)
        tlay.addRow("Serpentine", self.chk_serpentine)
        tlay.addRow("Matrix size", self.cmb_matrix)
        self.spin_flat = self._guard(QSpinBox()); self.spin_flat.setRange(0, 100)
        self.spin_flat.setValue(125); self.spin_flat.setSuffix(" /100")
        self.spin_flat.setRange(0, 1000)
        self.spin_flat.valueChanged.connect(self._mark_dirty)
        tlay.addRow("Flat threshold", self.spin_flat)
        self.spin_gamut.valueChanged.connect(self._mark_dirty)
        self.spin_dither.valueChanged.connect(self._mark_dirty)
        tlay.addRow("Shadow lift", self.spin_gamut)
        tlay.addRow("Dither", self.spin_dither)
        tlay.addRow(QLabel("Shadow lift rescues detail below black dye.\n"
                           "Dither trades per-pixel accuracy for smooth gradients."))
        rcol.addWidget(tone_box)
        self._sync_dither_rows()

        # Source cleanup
        clean_box = QGroupBox("Source cleanup")
        clay = QFormLayout(clean_box)
        self._clean_form = clay
        self.cmb_resample = QComboBox()
        self.cmb_resample.addItems(["box", "mode", "nearest", "bilinear", "lanczos"])
        self.cmb_denoise = QComboBox(); self.cmb_denoise.addItems(["none", "guided", "median"])
        self.spin_dn_radius = self._guard(QSpinBox()); self.spin_dn_radius.setRange(1, 8); self.spin_dn_radius.setValue(2)
        self.spin_dn_strength = self._guard(QSpinBox()); self.spin_dn_strength.setRange(1, 50)
        self.spin_dn_strength.setValue(6); self.spin_dn_strength.setSuffix(" %")
        self.spin_med_radius = self._guard(QSpinBox()); self.spin_med_radius.setRange(1, 4); self.spin_med_radius.setValue(1)
        self.spin_deblock = self._guard(QSpinBox()); self.spin_deblock.setRange(0, 100)
        self.spin_deblock.setValue(0); self.spin_deblock.setSuffix(" %")
        self.spin_snap = self._guard(QSpinBox()); self.spin_snap.setRange(0, 32); self.spin_snap.setValue(0)
        self.spin_harden = self._guard(QSpinBox()); self.spin_harden.setRange(0, 100)
        self.spin_harden.setValue(0); self.spin_harden.setSuffix(" %")
        self.spin_harden.setToolTip("0 keeps the soft anti-aliased edges the downscale makes;\n"
                                    "100 snaps edge pixels to one side, like pixel art.")
        self.spin_harden_thr = self._guard(QSpinBox()); self.spin_harden_thr.setRange(1, 60)
        self.spin_harden_thr.setValue(12)
        self.spin_harden_thr.setToolTip("Contrast (L*) below which an area counts as a gradient\n"
                                        "and is left soft.")
        self.chk_compare = QCheckBox("Compare with cleaned source")
        for w in (self.cmb_resample, self.cmb_denoise):
            w.currentIndexChanged.connect(self._mark_dirty)
        self.cmb_denoise.currentIndexChanged.connect(self._sync_denoise_rows)
        for w in (self.spin_dn_radius, self.spin_dn_strength, self.spin_med_radius,
                  self.spin_deblock, self.spin_snap, self.spin_harden, self.spin_harden_thr):
            w.valueChanged.connect(self._mark_dirty)
        self.chk_compare.toggled.connect(self._on_compare_toggled)
        clay.addRow("Downscale", self.cmb_resample)
        clay.addRow("Denoise", self.cmb_denoise)
        clay.addRow("Denoise radius", self.spin_dn_radius)
        clay.addRow("Denoise strength", self.spin_dn_strength)
        clay.addRow("Median radius", self.spin_med_radius)
        clay.addRow("JPEG deblock", self.spin_deblock)
        clay.addRow("Color snap", self.spin_snap)
        clay.addRow("Edge hardness", self.spin_harden)
        clay.addRow("Keep gradients under", self.spin_harden_thr)
        clay.addRow(self.chk_compare)
        clay.addRow(QLabel("Denoise and deblock run at source resolution, before\n"
                           "downscale. Edge hardness runs after it, on the pixels\n"
                           "you will actually paint.\n"
                           "Drag the divider in the preview to compare."))
        rcol.addWidget(clean_box)
        self._sync_denoise_rows()

        # Dye selection
        dyes_box = QGroupBox("Base dyes")
        dlay2 = QVBoxLayout(dyes_box)
        self.dye_checks = {}
        for name in BASE_COLORS:
            cb = QCheckBox(name); cb.setChecked(True)
            cb.toggled.connect(self._mark_dirty)
            dlay2.addWidget(cb)
            self.dye_checks[name] = cb
        rcol.addWidget(dyes_box)

        # Stats
        self.lbl_stats = QLabel("No image loaded")
        self.lbl_stats.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_stats.setWordWrap(True)
        rcol.addWidget(self.lbl_stats)

        # Apply
        self.btn_apply = QPushButton("Apply Changes  (F5)")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply_changes)
        rcol.addWidget(self.btn_apply)
        rcol.addStretch(1)

        self._apply_sc = QShortcut(QKeySequence("F5"), self)
        self._apply_sc.activated.connect(self._apply_changes)

        self._progress_dialog: QProgressDialog | None = None
        self._progress_started = 0.0
        self._dirty = False

    def _make_margin_spin(self):
        s = self._guard(QSpinBox())
        s.setRange(-256, 256)
        s.setValue(0)
        s.valueChanged.connect(self._mark_dirty)
        return s

    def _guard(self, spin):
        """Wheel-proof a spin box and route its edits through the dirty flag."""
        spin.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        spin.installEventFilter(self._no_wheel)
        return spin

    def _sync_denoise_rows(self):
        """Show only the fields belonging to the selected denoise mode.
        
        Rows are looked up by widget rather than index so reordering the form
        cannot silently hide the wrong control.
        """
        mode = self.cmb_denoise.currentText()
        for spin, want in ((self.spin_dn_radius, mode == "guided"),
                           (self.spin_dn_strength, mode == "guided"),
                           (self.spin_med_radius, mode == "median")):
            row, _ = self._clean_form.getWidgetPosition(spin)
            if row >= 0:
                self._clean_form.setRowVisible(row, want)

    def _sync_dither_rows(self):
        """Show only the controls the selected dither method actually uses."""
        method = self.cmb_dither.currentText()
        active = self.spin_dither.value() > 0
        for widget, want in (
            (self.cmb_dither, active),
            (self.chk_serpentine, active and method in dither_mod.DIFFUSION_KERNELS),
            (self.cmb_matrix, active and method == 'ordered'),
            (self.spin_flat, active),
        ):
            row, _ = self._tone_form.getWidgetPosition(widget)
            if row >= 0:
                self._tone_form.setRowVisible(row, want)

    # ---------- public ----------
    def controls_help(self) -> str:
        return ("Mouse-wheel: zoom preview  •  Click-drag: pan  •  "
                "Edit settings, then Apply Changes or press F5")

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
        for spin, val in [(self.spin_gamut, getattr(self.state, "gamut_strength", 0.5)),
                          (self.spin_dither, getattr(self.state, "dither_strength", 0.85))]:
            spin.blockSignals(True); spin.setValue(int(round(val * 100))); spin.blockSignals(False)
        self.cmb_resample.blockSignals(True)
        self.cmb_resample.setCurrentText(getattr(self.state, "resample_filter", "box"))
        self.cmb_resample.blockSignals(False)
        self.cmb_denoise.blockSignals(True)
        self.cmb_denoise.setCurrentText(getattr(self.state, "denoise_mode", "none"))
        self.cmb_denoise.blockSignals(False)
        self._sync_denoise_rows()
        for spin, val in [(self.spin_dn_radius, getattr(self.state, "denoise_radius", 2)),
                          (self.spin_dn_strength, round(getattr(self.state, "denoise_strength", 0.06) * 100)),
                          (self.spin_med_radius, getattr(self.state, "median_radius", 1)),
                          (self.spin_deblock, round(getattr(self.state, "deblock_strength", 0.0) * 100)),
                          (self.spin_snap, getattr(self.state, "color_snap", 0)),
                          (self.spin_harden, round(getattr(self.state, "edge_hardness", 0.0) * 100)),
                          (self.spin_harden_thr, getattr(self.state, "edge_threshold", 12.0))]:
            spin.blockSignals(True); spin.setValue(int(val)); spin.blockSignals(False)
        self.cmb_dither.blockSignals(True)
        self.cmb_dither.setCurrentText(getattr(self.state, "dither_method", "floyd-steinberg"))
        self.cmb_dither.blockSignals(False)
        self.chk_serpentine.blockSignals(True)
        self.chk_serpentine.setChecked(bool(getattr(self.state, "dither_serpentine", True)))
        self.chk_serpentine.blockSignals(False)
        self.cmb_matrix.blockSignals(True)
        self.cmb_matrix.setCurrentText(str(getattr(self.state, "dither_matrix", 8)))
        self.cmb_matrix.blockSignals(False)
        self.spin_flat.blockSignals(True)
        self.spin_flat.setValue(int(round(getattr(self.state, "dither_flat", 1.25) * 100)))
        self.spin_flat.blockSignals(False)
        self._sync_dither_rows()
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
        # Overlay keys are (y, x): a different image at the same size would
        # silently inherit the previous image's edits.
        self.state.edit_overlay = {}
        self.state.edit_overlay_size = None
        self.state.indexed_pristine = None
        self.state.match_source = None
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
        self._mark_dirty()

    def _on_h_changed(self, v):
        if self._aspect_locked and self.state.source_image is not None:
            sw, sh = self.state.source_image.size
            new_w = max(8, round(v * sw / sh))
            self.spin_w.blockSignals(True); self.spin_w.setValue(new_w); self.spin_w.blockSignals(False)
        self._mark_dirty()

    def _on_lock_toggled(self, on):
        self._aspect_locked = on

    def _on_compare_toggled(self, on):
        self.state.compare_source = bool(on)
        self.view.set_compare_enabled(bool(on))

    def _on_limit_toggled(self, on):
        self.spin_limit.setEnabled(on)
        self._mark_dirty()

    def _mark_dirty(self):
        """Edits no longer recompute on their own."""
        if self.state.source_image is None:
            return
        self._dirty = True
        self.btn_apply.setEnabled(True)
        self.btn_apply.setText("Apply Changes *  (F5)")

    def _apply_changes(self):
        if self.state.source_image is None or self._progress_dialog is not None:
            return
        self._recompute()

    def _cancel_work(self):
        if self._worker is not None:
            self._worker.abort()

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
        self.state.gamut_strength = self.spin_gamut.value() / 100.0
        self.state.dither_strength = self.spin_dither.value() / 100.0
        self.state.dither_method = self.cmb_dither.currentText()
        self.state.dither_serpentine = self.chk_serpentine.isChecked()
        self.state.dither_matrix = int(self.cmb_matrix.currentText())
        self.state.dither_flat = self.spin_flat.value() / 100.0
        self.state.resample_filter = self.cmb_resample.currentText()
        self.state.denoise_mode = self.cmb_denoise.currentText()
        self.state.denoise_radius = self.spin_dn_radius.value()
        self.state.denoise_strength = self.spin_dn_strength.value() / 100.0
        self.state.median_radius = self.spin_med_radius.value()
        self.state.deblock_strength = self.spin_deblock.value() / 100.0
        self.state.color_snap = self.spin_snap.value()
        self.state.edge_hardness = self.spin_harden.value() / 100.0
        self.state.edge_threshold = float(self.spin_harden_thr.value())
        self.lbl_depth_warn.setText(
            f"Depth {self.state.depth}: palette build takes {_DEPTH_COST[self.state.depth]}"
            if self.state.depth in _DEPTH_COST else "")
        if not self.state.selected_dyes:
            self.lbl_stats.setText("Select at least one dye.")
            return
        if self._thread is not None:
            self._thread.quit(); self._thread.wait(); self._thread = None
        self.lbl_stats.setText("Working...")
        self._worker = _Worker(
            self.state.source_image, self.state.target_resolution, margins,
            self.state.depth, list(self.state.selected_dyes), self.state.color_limit,
            self.state.gamut_strength, self.state.dither_strength,
            {'deblock': self.state.deblock_strength,
             'denoise_mode': self.state.denoise_mode,
             'denoise_radius': self.state.denoise_radius,
             'denoise_strength': self.state.denoise_strength,
             'median_radius': self.state.median_radius,
             'resample': self.state.resample_filter,
             'snap': self.state.color_snap,
             'harden': self.state.edge_hardness,
             'harden_threshold': self.state.edge_threshold},
            dither_method=self.state.dither_method,
            dither_serpentine=self.state.dither_serpentine,
            dither_matrix=self.state.dither_matrix,
            dither_flat=self.state.dither_flat,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._thread.finished.connect(self._on_thread_finished)
        self._worker.finished.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)

        dlg = _ProgressDialog(self, on_cancel=self._cancel_work)
        self._progress_dialog = dlg
        self._progress_started = time.monotonic()
        dlg.show()
        self._thread.start()

    def _on_progress(self, permille, label):
        # Pin a local reference: setValue() pumps events, so _on_done can fire
        # mid-call and null self._progress_dialog before the next line runs.
        dlg = self._progress_dialog
        if dlg is None:
            return
        elapsed = time.monotonic() - self._progress_started
        eta = ""
        if permille >= 30 and elapsed > 0.7:
            remain = max(0.0, elapsed * 1000.0 / permille - elapsed)
            eta = f"   ~{remain:.0f}s remaining" if remain >= 1.0 else "   almost done"
        dlg.setLabelText(f"{label}\n{permille / 10:.0f}%  -  {elapsed:.0f}s elapsed{eta}")
        dlg.setValue(permille)

    def _close_progress(self):
        # Never wait() here: this runs on the main thread from a queued signal,
        # and _thread.quit() is still in this thread's own event queue.
        dlg, self._progress_dialog = self._progress_dialog, None
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()

    def _clear_dirty(self):
        self._dirty = False
        self.btn_apply.setEnabled(False)
        self.btn_apply.setText("Apply Changes  (F5)")

    def _on_thread_finished(self):
        t, self._thread = self._thread, None
        self._worker = None
        self._close_progress()      # backstop: never leave the dialog up
        if t is not None:
            t.deleteLater()

    def _on_cancelled(self):
        self._close_progress()
        self.lbl_stats.setText("Cancelled. Adjust settings and press F5 to retry.")
        self._mark_dirty()

    def _on_done(self, processed, palette, recipes, matched, source_pre, alloc,
                 match_src=None):
        self._close_progress()
        self._clear_dirty()
        self.state.preprocessed_source = source_pre
        self.state.match_source = match_src
        self.state.group_allocation = list(alloc or [])
        self.view.set_compare_image(source_pre)
        self.view.set_compare_enabled(self.chk_compare.isChecked())
        self.state.processed_image = processed
        self.state.palette = palette
        self.state.recipes = recipes
        ix = IndexedImage.from_matched(matched, palette, recipes)
        note = ""
        if self.state.edit_overlay and self.state.edit_overlay_size != ix.shape:
            # (y, x) keys mean nothing at a new resolution.
            note = f"Discarded {len(self.state.edit_overlay)} manual edits: the image size changed."
            self.state.edit_overlay = {}
            self.state.edit_overlay_size = None
        self.state.indexed_pristine = ix.copy()   # what "revert edits" goes back to
        stamped = ix.apply_overlay(self.state.edit_overlay, recipes)
        self.state.indexed = ix
        if stamped:
            note = f"Re-applied {stamped} manual edits."
            self.state.processed_image = ix.to_rgb()
        if note:
            self.lbl_stats.setText(note)
        self.view.set_image(processed)
        h, w = processed.shape[:2]
        pad_w = (w + 15) // 16 * 16
        pad_h = (h + 15) // 16 * 16
        try:
            est_canvases = len(tiler.greedy_mixed_layout(pad_w, pad_h))
        except Exception:
            est_canvases = "?"
        # Distinct colors actually in the matched image
        unique_used = int((ix.used_counts() > 0).sum())
        self.lbl_stats.setText(
            f"Output: {w}×{h} px\n"
            f"Palette generated: {len(palette)} colors\n"
            f"Colors used in image: {unique_used}\n"
            f"Est. canvases (mixed): {est_canvases}"
        )
        self.state_changed.emit()

    def _on_failed(self, msg):
        self._close_progress()
        self.lbl_stats.setText(f"Error: {msg}")
        self._mark_dirty()
