"""Main window with stage stack and File menu."""

from pathlib import Path

from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QStackedWidget, QToolBar, QFileDialog, QMessageBox, QLabel,
)
import numpy as np

from ..core.project import ProjectState
from ..core import sidecar, preprocess
from .stage_source import SourceStage
from .stage_layout import LayoutStage
from .stage_paint import PaintStage


STAGE_SOURCE = 0
STAGE_LAYOUT = 1
STAGE_PAINT = 2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Joy of Painting Toolkit")
        self.resize(1280, 800)
        self.state = ProjectState()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stage_source = SourceStage(self.state)
        self.stage_layout = LayoutStage(self.state)
        self.stage_paint = PaintStage(self.state)
        self.stack.addWidget(self.stage_source)
        self.stack.addWidget(self.stage_layout)
        self.stack.addWidget(self.stage_paint)

        self.stage_paint.status_changed.connect(self.statusBar().showMessage)

        self._footer = QLabel("")
        self._footer.setStyleSheet("color: #888;")
        self.statusBar().addPermanentWidget(self._footer, 1)
        # Stage 2 controls help depends on Auto/Manual radio state.
        self.stage_layout.radio_auto.toggled.connect(self._refresh_footer)
        self.stage_layout.radio_manual.toggled.connect(self._refresh_footer)
        self.stack.currentChanged.connect(lambda _i: self._refresh_footer())

        self._build_menus()
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

    def _build_toolbar(self):
        tb = QToolBar("Stages")
        self.addToolBar(tb)
        self.act_source = QAction("1. Source", self)
        self.act_layout = QAction("2. Layout", self)
        self.act_paint = QAction("3. Paint", self)
        for i, act in enumerate([self.act_source, self.act_layout, self.act_paint]):
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
            return
        # Gate transitions
        if idx >= STAGE_LAYOUT and self.state.matched_hex_grid is None:
            QMessageBox.information(self, "Load an image", "Open an image on the Source stage first.")
            return
        if idx == STAGE_PAINT and not self.stage_layout.finalize():
            return
        if idx == STAGE_LAYOUT:
            self.stage_layout.enter()
        if idx == STAGE_PAINT:
            self.stage_paint.enter()
        self.stack.setCurrentIndex(idx)
        self._update_nav_actions()

    def _refresh_footer(self):
        stage = self.stack.currentWidget()
        helper = getattr(stage, "controls_help", None)
        self._footer.setText(helper() if helper else "")

    def _update_nav_actions(self):
        i = self.stack.currentIndex()
        for j, act in enumerate([self.act_source, self.act_layout, self.act_paint]):
            act.setChecked(j == i)
        self.act_back.setEnabled(i > 0)
        self.act_next.setEnabled(i < self.stack.count() - 1)

    # ---------- persistence ----------
    def _save_project(self):
        if self.state.matched_hex_grid is None:
            QMessageBox.information(self, "Nothing to save", "Process an image first.")
            return
        # Ensure layout/tiles are filled. If user hasn't visited Stage 2, do an auto pass.
        if not self.state.layout:
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

    def _open_project(self):
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
        self.state.depth = int(toolkit.get('depth', 4))
        self.state.color_limit = toolkit.get('color_limit')
        if 'selected_dyes' in toolkit:
            self.state.selected_dyes = list(toolkit['selected_dyes'])
        self.state.layout_mode = toolkit.get('layout_mode', 'auto')

        # Restore layout from tiles
        self.state.layout = [
            {'pixel_pos': list(t['pixel_pos']), 'tile_size': list(t['tile_size'])}
            for t in data.get('tiles', [])
        ]

        # Trigger Stage 1 recompute (palette/matched grid are derived).
        self.stage_source.refresh_from_state()
        self._goto(STAGE_SOURCE)
        self.statusBar().showMessage(f"Loaded {path}", 5000)
