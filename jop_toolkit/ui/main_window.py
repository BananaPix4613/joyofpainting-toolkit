"""Main window with stage stack and File menu."""

from pathlib import Path

from PyQt6.QtCore import QEvent, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QToolBar, QFileDialog, QMessageBox, QLabel,
)
import numpy as np

from ..core.project import ProjectState
from ..core import sidecar, preprocess
from . import icons, theme
from .stage_source import SourceStage
from .widgets.palette_ops import DetailDialog, LinesDialog, ReduceDialog
from .stage_edit import EditStage
from .stage_layout import LayoutStage
from .stage_paint import PaintStage


STAGE_SOURCE = 0
STAGE_EDIT = 1
STAGE_LAYOUT = 2
STAGE_PAINT = 3


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Joy of Painting Toolkit")
        self.resize(1280, 800)
        self.state = ProjectState()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stage_source = SourceStage(self.state)
        self.stage_edit = EditStage(self.state)
        self.stage_layout = LayoutStage(self.state)
        self.stage_paint = PaintStage(self.state)
        self.stack.addWidget(self.stage_source)
        self.stack.addWidget(self.stage_edit)
        self.stack.addWidget(self.stage_layout)
        self.stack.addWidget(self.stage_paint)

        self.stage_paint.status_changed.connect(self.statusBar().showMessage)
        self.stage_edit.status_changed.connect(self.statusBar().showMessage)
        self.stage_edit.history_changed.connect(self._refresh_edit_actions)
        self.stage_edit.selection_changed.connect(self._refresh_edit_actions)
        self.stage_edit.clipboard_changed.connect(self._refresh_edit_actions)

        self._footer = QLabel("")
        self._footer.setStyleSheet("color: #888;")
        self.statusBar().addPermanentWidget(self._footer, 1)
        # Stage 2 controls help depends on Auto/Manual radio state.
        self.stage_layout.radio_auto.toggled.connect(self._refresh_footer)
        self.stage_layout.radio_manual.toggled.connect(self._refresh_footer)
        self.stack.currentChanged.connect(lambda _i: self._refresh_footer())

        self._build_menus()
        self._build_edit_view_menus(self.menuBar())
        self._build_toolbar()
        self._update_nav_actions()
        self._refresh_footer()

    # ---------- menus ----------
    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        act_open_img = QAction("Open &Image…", self)
        act_open_img.triggered.connect(self.stage_source._open_image)
        act_open_img.setShortcut(QKeySequence("Ctrl+I"))
        m.addAction(act_open_img)

        act_open_proj = QAction("Open &Project…", self)
        act_open_proj.triggered.connect(self._open_project)
        act_open_proj.setShortcut(QKeySequence.StandardKey.Open)
        m.addAction(act_open_proj)

        act_save = QAction("&Save Project…", self)
        act_save.triggered.connect(self._save_project)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        m.addAction(act_save)

        m.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.triggered.connect(self.close)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        m.addAction(act_quit)

    def _build_edit_view_menus(self, mb):
        """Edit and View. Both act on whichever stage is showing, so entries that
        only make sense on one stage are enabled from _update_nav_actions."""
        e = mb.addMenu("&Edit")
        self.act_undo = QAction("&Undo", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.stage_edit.undo)
        e.addAction(self.act_undo)
        self.act_redo = QAction("&Redo", self)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_redo.triggered.connect(self.stage_edit.redo)
        e.addAction(self.act_redo)
        e.addSeparator()
        self.act_revert = QAction("Re&vert All Edits", self)
        self.act_revert.triggered.connect(self.stage_edit.revert_edits)
        e.addAction(self.act_revert)

        e.addSeparator()
        self.act_cut = QAction("Cu&t", self)
        self.act_cut.setShortcut(QKeySequence.StandardKey.Cut)
        self.act_cut.triggered.connect(lambda: self.stage_edit.cut_selection())
        e.addAction(self.act_cut)
        self.act_copy = QAction("&Copy", self)
        self.act_copy.setShortcut(QKeySequence.StandardKey.Copy)
        self.act_copy.triggered.connect(lambda: self.stage_edit.copy_selection())
        e.addAction(self.act_copy)
        self.act_paste = QAction("&Paste", self)
        self.act_paste.setShortcut(QKeySequence.StandardKey.Paste)
        self.act_paste.triggered.connect(lambda: self.stage_edit.paste_clip())
        e.addAction(self.act_paste)

        e.addSeparator()
        # triggered(bool) would land in the first positional argument of any
        # slot that takes one, so every entry here goes through a no-arg lambda.
        self.act_sel_all = QAction("Select &All", self)
        self.act_sel_all.setShortcut(QKeySequence.StandardKey.SelectAll)
        self.act_sel_all.triggered.connect(lambda: self.stage_edit.select_all())
        e.addAction(self.act_sel_all)
        self.act_sel_none = QAction("&Deselect", self)
        self.act_sel_none.setShortcut(QKeySequence("Ctrl+D"))
        self.act_sel_none.triggered.connect(lambda: self.stage_edit.select_none())
        e.addAction(self.act_sel_none)
        self.act_sel_inv = QAction("&Invert Selection", self)
        self.act_sel_inv.setShortcut(QKeySequence("Ctrl+Shift+I"))
        self.act_sel_inv.triggered.connect(lambda: self.stage_edit.invert_selection())
        e.addAction(self.act_sel_inv)
        self.act_sel_lines = QAction("Select &Lines…", self)
        self.act_sel_lines.triggered.connect(self._open_lines)
        e.addAction(self.act_sel_lines)

        e.addSeparator()
        self.act_fill_sel = QAction("&Fill Selection", self)
        self.act_fill_sel.setShortcut(QKeySequence("Ctrl+Return"))
        self.act_fill_sel.triggered.connect(lambda: self.stage_edit.fill_selection())
        e.addAction(self.act_fill_sel)
        self.act_erase_sel = QAction("&Erase Selected Pixels", self)
        self.act_erase_sel.setShortcut(QKeySequence("Del"))
        self.act_erase_sel.triggered.connect(lambda: self.stage_edit.erase_selection())
        e.addAction(self.act_erase_sel)

        e.addSeparator()
        self.act_despeckle = QAction("Clean Up &Speckles…", self)
        self.act_despeckle.triggered.connect(lambda: self.stage_edit.clean_speckles())
        e.addAction(self.act_despeckle)

        e.addSeparator()
        self.act_detail = QAction("Add &Detail…", self)
        self.act_detail.triggered.connect(self._open_detail)
        e.addAction(self.act_detail)
        self.act_reduce = QAction("Reduce &Colors…", self)
        self.act_reduce.triggered.connect(self._open_reduce)
        e.addAction(self.act_reduce)

        v = mb.addMenu("&View")
        act_zin = QAction("Zoom &In", self)
        act_zin.setShortcut(QKeySequence.StandardKey.ZoomIn)
        act_zin.triggered.connect(lambda: self._canvas_zoom(2))
        v.addAction(act_zin)
        act_zout = QAction("Zoom &Out", self)
        act_zout.setShortcut(QKeySequence.StandardKey.ZoomOut)
        act_zout.triggered.connect(lambda: self._canvas_zoom(0.5))
        v.addAction(act_zout)
        act_fit = QAction("&Fit to Window", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(lambda: self.stage_edit.canvas.fit())
        v.addAction(act_fit)
        self.act_zoom_sel = QAction("Zoom to &Selection", self)
        self.act_zoom_sel.setShortcut(QKeySequence("Ctrl+Shift+0"))
        self.act_zoom_sel.triggered.connect(lambda: self.stage_edit.zoom_to_selection())
        v.addAction(self.act_zoom_sel)
        v.addSeparator()

        self.act_grid = QAction("Show &Grid", self)
        self.act_grid.setCheckable(True)
        self.act_grid.setShortcut(QKeySequence("Ctrl+'"))
        self.act_grid.toggled.connect(self.stage_edit.chk_grid.setChecked)
        self.stage_edit.chk_grid.toggled.connect(self.act_grid.setChecked)
        v.addAction(self.act_grid)
        v.addSeparator()

        # Source-stage preview controls live here too, so the right panel does
        # not have to carry them.
        self.act_compare = QAction("&Compare With Source", self)
        self.act_compare.setCheckable(True)
        self.act_compare.toggled.connect(self.stage_source.chk_compare.setChecked)
        self.stage_source.chk_compare.toggled.connect(self.act_compare.setChecked)
        v.addAction(self.act_compare)
        v.addSeparator()

        tm = v.addMenu("&Theme")
        group = QActionGroup(self)
        group.setExclusive(True)
        current = theme.load()
        for mode in theme.MODES:
            act = QAction(theme.LABELS[mode], self)
            act.setCheckable(True)
            act.setChecked(mode == current)
            act.triggered.connect(lambda _c=False, m=mode: self._set_theme(m))
            group.addAction(act)
            tm.addAction(act)
        # In System mode the OS can change under us. The palette catches up
        # after this signal, so re-check once it has.
        QApplication.styleHints().colorSchemeChanged.connect(
            lambda _s: QTimer.singleShot(0, lambda: theme.apply(theme.load())))

    def _open_lines(self):
        if self.state.indexed is None:
            return
        LinesDialog(self.stage_edit, self).exec()

    def _open_detail(self):
        if self.state.indexed is None:
            return
        DetailDialog(self.stage_edit, self).exec()

    def _open_reduce(self):
        if self.state.indexed is None:
            return
        ReduceDialog(self.stage_edit, self).exec()

    def _set_theme(self, mode):
        theme.save(mode)
        theme.apply(mode)

    def changeEvent(self, event):
        # Icons are drawn once in the text color, so a new palette needs them
        # drawn again.
        if event.type() == QEvent.Type.PaletteChange:
            icons.retint()
        super().changeEvent(event)

    def _canvas_zoom(self, factor):
        c = self.stage_edit.canvas
        c.set_zoom(int(c.zoom() * factor) if factor > 1 else max(1, int(c.zoom() * factor)))

    def _build_toolbar(self):
        tb = QToolBar("Stages")
        self.addToolBar(tb)
        self.act_source = QAction("1. Source", self)
        self.act_edit = QAction("2. Edit", self)
        self.act_layout = QAction("3. Layout", self)
        self.act_paint = QAction("4. Paint", self)
        for i, act in enumerate([self.act_source, self.act_edit, self.act_layout, self.act_paint]):
            act.setCheckable(True)
            act.triggered.connect(lambda _checked, i=i: self._goto(i))
            tb.addAction(act)
        tb.addSeparator()
        self.act_back = QAction("◀ Back", self)
        self.act_next = QAction("Next ▶", self)
        self.act_back.triggered.connect(lambda: self._goto(self.stack.currentIndex() - 1))
        self.act_next.triggered.connect(lambda: self._goto(self.stack.currentIndex() + 1))
        tb.addAction(self.act_back)
        tb.addAction(self.act_next)

    # ---------- navigation ----------
    def _goto(self, idx):
        if idx < 0 or idx >= self.stack.count():
            self._update_nav_actions()
            return
        # Gate transitions.
        if idx >= STAGE_EDIT and self.state.indexed is None:
            QMessageBox.information(self, "Load an image", "Open an image on the Source stage first.")
            self._update_nav_actions()
            return
        if idx == STAGE_PAINT and not self.stage_layout.finalize():
            self._update_nav_actions()
            return
        if idx == STAGE_EDIT:
            self.stage_edit.enter()
        if idx == STAGE_LAYOUT:
            self.stage_layout.enter()
        if idx == STAGE_PAINT:
            self.stage_paint.enter()
        self.stack.setCurrentIndex(idx)
        self._update_nav_actions()

    def _refresh_edit_actions(self):
        """Undo/redo availability changes with every edit, not just when the
        stage changes. Without this the actions stay disabled and both the
        menu item and Ctrl+Z silently do nothing."""
        on_edit = self.stack.currentIndex() == STAGE_EDIT
        self.act_undo.setEnabled(on_edit and self.stage_edit.history.can_undo())
        self.act_redo.setEnabled(on_edit and self.stage_edit.history.can_redo())
        self.act_revert.setEnabled(on_edit and bool(self.state.edit_overlay))
        live = on_edit and self.state.indexed is not None
        has_sel = live and self.stage_edit.selection is not None
        for act in (self.act_sel_all, self.act_sel_inv, self.act_despeckle,
                    self.act_detail, self.act_reduce, self.act_sel_lines):
            act.setEnabled(live)
        for act in (self.act_sel_none, self.act_fill_sel, self.act_erase_sel,
                    self.act_cut, self.act_copy, self.act_zoom_sel):
            act.setEnabled(has_sel)
        self.act_paste.setEnabled(live and self.stage_edit.can_paste())

    def _refresh_footer(self):
        stage = self.stack.currentWidget()
        helper = getattr(stage, "controls_help", None)
        self._footer.setText(helper() if helper else "")

    def _update_nav_actions(self):
        i = self.stack.currentIndex()
        for j, act in enumerate([self.act_source, self.act_edit, self.act_layout, self.act_paint]):
            act.setChecked(j == i)
        self._refresh_edit_actions()
        self.act_back.setEnabled(i > 0)
        self.act_next.setEnabled(i < self.stack.count() - 1)

    # ---------- persistence ----------
    def _save_project(self):
        if self.state.matched_hex_grid is None:
            QMessageBox.information(self, "Nothing to save", "Process an image first.")
            return
        # Always re-enter: _refresh_image() is what pads the matched grid to a
        # multiple of 16, and a layout left over from a previous Apply will not
        # match a freshly-processed grid.
        self.stage_layout.enter()
        if not self.stage_layout.finalize():
            return
        default = (self.state.source_path.with_suffix('.jop.json')
                   if self.state.source_path else 'project.jop.json')
        path, _ = QFileDialog.getSaveFileName(self, "Save project", str(default),
                                              "JoP Sidecar (*.jop.json *.json)")
        if not path:
            return
        data = sidecar.build_sidecar_from_state(self.state)
        sidecar.write_sidecar(data, path)
        self.statusBar().showMessage(f"Saved {path}", 5000)

    def _confirm_discard_edits(self):
        """Opening a project rebuilds the whole session from the file, so any
        manual edits not already saved are gone. Say so first."""
        n = len(self.state.edit_overlay)
        if not n:
            return True
        return QMessageBox.question(
            self, "Discard edits?",
            f"You have {n} unsaved manual edit{'' if n == 1 else 's'} on the "
            f"current image.\nOpening a project will discard that work. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def _open_project(self):
        if not self._confirm_discard_edits():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "",
                                              "JoP Sidecar (*.jop.json *.json)")
        if not path:
            return
        data = sidecar.load_sidecar(path)
        src = data.get('source_image')
        src_path = Path(src) if src else None
        if not src_path or not src_path.exists():
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Locate source image (original could not be found)", "",
                "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
            if not chosen:
                return
            src_path = Path(chosen)

        toolkit = data.get('toolkit', {})
        self.state.source_path = src_path
        self.state.source_image = preprocess.load_image(str(src_path))
        if 'target_resolution' in toolkit:
            self.state.target_resolution = tuple(toolkit['target_resolution'])
        else:
            self.state.target_resolution = tuple(data.get('image_size', list(self.state.source_image.size)))
        self.state.margin = int(toolkit.get('margin', 0))
        m = toolkit.get('margins')
        if m and len(m) == 4:
            self.state.margins = tuple(int(x) for x in m)
        else:
            self.state.margins = (self.state.margin,) * 4
        self.state.depth = int(toolkit.get('depth', 6))
        self.state.color_snap = int(toolkit.get('color_snap', 0))
        self.state.edge_hardness = float(toolkit.get('edge_hardness', 0.0))
        self.state.edge_threshold = float(toolkit.get('edge_threshold', 12.0))
        self.state.gamut_strength = float(toolkit.get('gamut_strength', 0.5))
        self.state.dither_strength = float(toolkit.get('dither_strength', 0.85))
        self.state.dither_method = toolkit.get('dither_method', 'floyd-steinberg')
        self.state.dither_serpentine = bool(toolkit.get('dither_serpentine', True))
        self.state.dither_matrix = int(toolkit.get('dither_matrix', 8))
        self.state.resample_filter = toolkit.get('resample_filter', 'box')
        self.state.denoise_mode = toolkit.get('denoise_mode', 'none')
        self.state.denoise_radius = int(toolkit.get('denoise_radius', 2))
        self.state.denoise_strength = float(toolkit.get('denoise_strength', 0.06))
        self.state.median_radius = int(toolkit.get('median_radius', 1))
        self.state.deblock_strength = float(toolkit.get('deblock_strength', 0.0))
        self.state.color_snap = int(toolkit.get('color_snap', 0))
        if 'selected_dyes' in toolkit:
            self.state.selected_dyes = list(toolkit['selected_dyes'])
        self.state.layout_mode = toolkit.get('layout_mode', 'auto')

        # Restore layout from tiles
        self.state.layout = [
            {'pixel_pos': list(t['pixel_pos']), 'tile_size': list(t['tile_size'])}
            for t in data.get('tiles', [])
        ]
        self.state.edit_overlay = {(int(y), int(x)): hx
                                   for y, x, hx in toolkit.get('edit_overlay', [])}
        size = toolkit.get('edit_overlay_size')
        if size:
            self.state.edit_overlay_size = (int(size[0]), int(size[1]))
        elif self.state.edit_overlay:
            tw, th = self.state.target_resolution
            t, r, b, l = self.state.margins
            self.state.edit_overlay_size = (th + t + b, tw + l + r)
        else:
            self.state.edit_overlay_size = None

        # Trigger Stage 1 recompute (palette/matched grid are derived).
        self.stage_source.refresh_from_state()
        self._goto(STAGE_SOURCE)
        self.statusBar().showMessage(f"Loaded {path}", 5000)
