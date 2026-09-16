"""Stage 2: Edit - manual touch-up of the indexed result.

Deliberately a stage rather than a tab on Source: Source owns Apply Changes,
which regenerates the whole pipeline. Editing has to live somewhere that never
re-runs it, or a stray Apply silently discards the work. Edits are recorded
sparsely into state.edit_overlay so a later Apply can stamp them back on.

Left and right mouse buttons carry independent colors, so the two tones you are
working between are both a click away without returning to the palette.
"""

import numpy as np
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence, QColor, QImage
from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel,
    QGroupBox, QCheckBox, QSpinBox, QPushButton, QScrollArea, QFrame, QComboBox,
    QToolBar, QInputDialog, QSlider,
)

from ..core import tools, selection as sel, refine, reduce as red
from ..core.history import History
from ..core.colors import hex_to_rgb
from ..core.indexed import EMPTY
from . import icons
from .widgets.pixel_canvas import PixelCanvas
from .widgets.palette_panel import PalettePanel

BRUSH, ERASER, BUCKET, DROPPER = 'brush', 'eraser', 'bucket', 'dropper'
RECT, LASSO, WAND, MOVE = 'rect', 'lasso', 'wand', 'move'
_LEFT = int(Qt.MouseButton.LeftButton.value)
_RIGHT = int(Qt.MouseButton.RightButton.value)


class EditStage(QWidget):
    status_changed = pyqtSignal(str, int)   # message, timeout ms
    clipboard_changed = pyqtSignal()        # something was copied
    history_changed = pyqtSignal()          # undo/redo availability changed
    selection_changed = pyqtSignal()        # a selection apeared or went away

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.history = History()
        self.tool = BRUSH
        self.fg = 0            # slot index, or EMPTY for "no paint"
        self.bg = EMPTY
        self._stroke = None    # accumulating change mask during a drag
        self._last = None      # last pixel of the current stroke
        self.selection = None  # bool mask, or None meaning "the whole image"
        self._clip = None      # copied pixels, as hexes so they survive a re-Apply
        self._shape = None     # points of a marquee/lasso drag in progress
        self._move = None      # lifted pixels during a move drag

        root = QHBoxLayout(self)
        self.canvas = PixelCanvas()
        center = QVBoxLayout()
        center.setContentsMargins(0, 0, 0, 0)
        center.setSpacing(0)
        center.addWidget(self._build_toolbar())
        center.addWidget(self.canvas, stretch=1)
        center_holder = QWidget()
        center_holder.setLayout(center)
        root.addWidget(center_holder, stretch=1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFixedWidth(340)
        right = QWidget()
        right_scroll.setWidget(right)
        rcol = QVBoxLayout(right)
        root.addWidget(right_scroll)

        # ---- colors ----
        col_box = QGroupBox("Colors")
        clay = QFormLayout(col_box)
        row = QHBoxLayout()
        self.btn_fg = QPushButton("left")
        self.btn_bg = QPushButton("right")
        self.btn_swap = icons.apply(QPushButton(""), 'swap')
        self.btn_swap.setToolTip("Swap the two colors  (X)")
        for b in (self.btn_fg, self.btn_bg):
            b.setMinimumHeight(30)
        row.addWidget(self.btn_fg)
        row.addWidget(self.btn_bg)
        holder = QWidget()
        holder.setLayout(row)
        clay.addRow(holder)
        clay.addRow(self.btn_swap)
        clay.addRow(QLabel("Alt-click the canvas picks a color.\n"
                           "Right-click paints with the right color."))
        rcol.addWidget(col_box)

        # ---- palette ----
        self.palette_panel = PalettePanel(self.state)
        rcol.addWidget(self.palette_panel)

        # ---- selection ----
        sel_box = QGroupBox("Selection")
        slay = QFormLayout(sel_box)
        self.lbl_sel = QLabel("-")
        srow = QHBoxLayout()
        self.btn_sel_all = icons.apply(QPushButton(""), 'sel_all')
        self.btn_sel_all.setToolTip("Select all  (Ctrl+A)")
        self.btn_sel_none = icons.apply(QPushButton(""), 'sel_none')
        self.btn_sel_none.setToolTip("Deselect  (Ctrl+D)")
        self.btn_sel_inv = icons.apply(QPushButton(""), 'sel_inv')
        self.btn_sel_inv.setToolTip("Invert the selection  (Ctrl+Shift+I)")
        for b in (self.btn_sel_all, self.btn_sel_none, self.btn_sel_inv):
            srow.addWidget(b)
        sholder = QWidget()
        sholder.setLayout(srow)
        self.btn_fill_sel = QPushButton("Fill with left color")
        self.btn_erase_sel = QPushButton("Erase selected pixels")
        slay.addRow(self.lbl_sel)
        slay.addRow(sholder)
        slay.addRow(self.btn_fill_sel)
        slay.addRow(self.btn_erase_sel)
        slay.addRow(QLabel("Shift-drag adds, Ctrl-drag subtracts.\n"
                           "Every tool stays inside the selection."))
        rcol.addWidget(sel_box)

        # ---- edits ----
        edit_box = QGroupBox("Edits")
        elay = QFormLayout(edit_box)
        self.btn_revert = QPushButton("Revert all edits")
        self.btn_revert.clicked.connect(self.revert_edits)
        elay.addRow(self.btn_revert)
        elay.addRow(QLabel("Goes back to the pipeline result.\nUndoable."))
        rcol.addWidget(edit_box)

        # ---- view ----
        view_box = QGroupBox("View")
        vlay = QFormLayout(view_box)
        self.chk_grid = QCheckBox("Show grid")
        self.spin_grid = QSpinBox()
        self.spin_grid.setRange(1, 64)
        self.spin_grid.setValue(16)
        self.spin_grid.setSuffix(" px")
        self.btn_fit = icons.apply(QPushButton("  Fit to window"), 'fit')
        self.lbl_zoom = QLabel("-")
        vlay.addRow(self.chk_grid)
        vlay.addRow("Grid size", self.spin_grid)
        vlay.addRow("Zoom", self.lbl_zoom)
        vlay.addRow(self.btn_fit)
        rcol.addWidget(view_box)

        self.lbl_info = QLabel("-")
        self.lbl_info.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_info.setWordWrap(True)
        rcol.addWidget(self.lbl_info)

        # ---- tool options ----
        self.opt_box = QGroupBox("Tool options")
        olay = QFormLayout(self.opt_box)
        self._opt_form = olay
        self.spin_size = QSpinBox()
        self.spin_size.setRange(1, 32)
        self.spin_size.setValue(1)
        self.cmb_shape = QComboBox()
        self.cmb_shape.addItems([tools.SQUARE, tools.ROUND])
        self.cmb_conn = QComboBox()
        self.cmb_conn.addItems(["4-connected", "8-connected"])
        self.chk_global = QCheckBox("Replace across whole image")
        self.spin_tol = QSpinBox()
        self.spin_tol.setRange(0, 100)
        self.spin_tol.setToolTip("Lab distance. 0 selects one color exactly.")
        self.chk_wand_all = QCheckBox("Select across whole image")
        olay.addRow("Size", self.spin_size)
        olay.addRow("Shape", self.cmb_shape)
        olay.addRow("Spread", self.cmb_conn)
        olay.addRow(self.chk_global)
        olay.addRow("Tolerance", self.spin_tol)
        olay.addRow(self.chk_wand_all)
        rcol.addWidget(self.opt_box)
        rcol.addStretch(1)

        self.chk_grid.toggled.connect(self._on_grid)
        self.spin_grid.valueChanged.connect(self._on_grid)
        self.btn_fit.clicked.connect(self.canvas.fit)
        self.spin_size.valueChanged.connect(self.canvas.set_brush_size)
        self.btn_swap.clicked.connect(self.swap_colors)
        self.canvas.zoom_changed.connect(lambda z: self.lbl_zoom.setText(f"{z}x"))
        self.canvas.hover_changed.connect(self._on_hover)
        self.canvas.pixel_pressed.connect(self._on_press)
        self.canvas.pixel_dragged.connect(self._on_drag)
        self.canvas.pixel_released.connect(self._on_release)
        self.btn_sel_all.clicked.connect(self.select_all)
        self.btn_sel_none.clicked.connect(self.select_none)
        self.btn_sel_inv.clicked.connect(self.invert_selection)
        self.btn_fill_sel.clicked.connect(self.fill_selection)
        self.btn_erase_sel.clicked.connect(self.erase_selection)
        self.palette_panel.color_picked.connect(self._on_palette_pick)
        self.palette_panel.request_history.connect(self.push_history)
        self.palette_panel.changed.connect(self._on_palette_changed)
        self._sync_tool_options()
        self._refresh_swatches()
        self._refresh_selection_info()

    # ---------- toolbar ----------
    def _build_toolbar(self):
        tb = QToolBar()
        tb.setOrientation(Qt.Orientation.Horizontal)
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        group = QActionGroup(self)
        group.setExclusive(True)
        self._tool_actions = {}
        rows = ((BRUSH, "Brush", "B", 'brush'), (ERASER, "Eraser", "E", 'eraser'),
                (BUCKET, "Fill", "G", 'bucket'), (DROPPER, "Pick", "I", 'dropper'),
                None,
                (RECT, "Marquee", "M", 'rect'), (LASSO, "Lasso", "Q", 'lasso'),
                (WAND, "Wand", "W", 'wand'), (MOVE, "Move", "V", 'move'))
        for row in rows:
            if row is None:
                tb.addSeparator()
                continue
            tool, text, key, art = row
            # The name still lives on the action for the menu and for screen
            # readers; only the button hides it. The shortcut shows on hover.
            act = icons.apply(QAction(text, self), art)
            act.setToolTip(f"{text}  ({key})")
            act.setCheckable(True)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(lambda _c=False, t=tool: self.set_tool(t))
            group.addAction(act)
            tb.addAction(act)
            self._tool_actions[tool] = act
        self._tool_actions[BRUSH].setChecked(True)
        act_swap = QAction("Swap", self)
        act_swap.setShortcut(QKeySequence("X"))
        act_swap.triggered.connect(self.swap_colors)
        self.addAction(act_swap)
        return tb

    # ---------- public ----------
    def controls_help(self) -> str:
        return ("Left/right click paints  •  Alt-click picks  •  "
                "Shift-drag adds to a selection, Ctrl-drag subtracts  •  "
                "Wheel: zoom  •  Middle or Space drag: pan  •  Ctrl+Z: undo")

    def enter(self):
        if self.state.indexed is None:
            self.lbl_info.setText("Process an image on the Source stage first.")
            return
        self.history.clear()
        self.history_changed.emit()
        if self.fg >= len(self.state.indexed.slots):
            self.fg = 0
        if self.selection is not None and self.selection.shape != self.state.indexed.shape:
            # A re-Apply at a new resolution leaves the old mask the wrong size.
            self.selection = None
        self.canvas.set_indexed(self.state.indexed, fit=True)
        self.canvas.set_selection(self.selection)
        self.canvas.set_brush_size(self.spin_size.value())
        self._refresh_selection_info()
        self._on_grid()
        self.palette_panel.refresh()
        self._refresh_swatches()
        self._refresh_info()

    def _on_palette_pick(self, slot, button):
        self.set_color(slot, 'bg' if button == _RIGHT else 'fg')

    def _on_palette_changed(self, touched=None):
        # Recolors and merges repaint real pixels, so they belong in the edit
        # overlay like any other edit; otherwise the next Apply reverts them.
        if touched is not None and self.state.indexed is not None:
            self._record(touched)
        # Merging or removing a slot shifts every index above it, so the two
        # paint colors have to be brought back into range.
        ix = self.state.indexed
        if ix is not None:
            n = len(ix.slots)
            self.fg = min(self.fg, n - 1) if self.fg >= 0 else EMPTY
            self.bg = min(self.bg, n - 1) if self.bg >= 0 else EMPTY
        self.refresh()
        self.history_changed.emit()

    def refresh(self):
        self.canvas.set_indexed(self.state.indexed)
        # Layout and Paint both read processed_image, so an edit that does not
        # update it leaves those stages showing the pre-edit render.
        if self.state.indexed is not None:
            self.state.processed_image = self.state.indexed.to_rgb()
        self._refresh_swatches()
        self.palette_panel.refresh()
        self._refresh_info()

    def set_tool(self, tool):
        self.tool = tool
        if tool in self._tool_actions:
            self._tool_actions[tool].setChecked(True)
        self._sync_tool_options()

    def swap_colors(self):
        self.fg, self.bg = self.bg, self.fg
        self._refresh_swatches()

    def set_color(self, slot, which='fg'):
        if which == 'fg':
            self.fg = slot
        else:
            self.bg = slot
        # Picking a color off the canvas should land on the same swatch the
        # palette shows as selected, so its details and operations follow along.
        self.palette_panel.select_slot(slot)
        self._refresh_swatches()

    # ---------- selection ----------
    def set_selection(self, mask):
        """An empty mask is stored as None: `within=` treats None as no
        restriction, so an all-False mask would silently freeze every tool."""
        if mask is not None and not mask.any():
            mask = None
        self.selection = mask
        self.canvas.set_selection(mask)
        self._refresh_selection_info()
        self.selection_changed.emit()

    def _commit_selection(self, mask):
        mods = QApplication.keyboardModifiers()
        if self.selection is not None:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                mask = self.selection | mask
            elif mods & Qt.KeyboardModifier.ControlModifier:
                mask = self.selection & ~mask
        self.set_selection(mask)

    def select_all(self):
        if self.state.indexed is not None:
            self.set_selection(np.ones(self.state.indexed.shape, dtype=bool))

    def select_none(self):
        self.set_selection(None)

    def invert_selection(self):
        ix = self.state.indexed
        if ix is None:
            return
        self.set_selection(np.ones(ix.shape, dtype=bool) if self.selection is None
                           else ~self.selection)

    def fill_selection(self):
        self._fill_selection(self.fg, "fill selection")

    def erase_selection(self):
        self._fill_selection(EMPTY, "erase selection")

    def _fill_selection(self, value, label):
        ix = self.state.indexed
        if ix is None or self.selection is None:
            return
        mask = self.selection & (ix.indices != value)
        if not mask.any():
            return
        self.push_history(label)
        ix.indices[mask] = value
        self._record(mask)
        self.refresh()
        self.status_changed.emit(f"{label.capitalize()}: {int(mask.sum())} pixels", 3000)

    def clean_speckles(self):
        """Despeckle, restricted to the selection when there is one."""
        ix = self.state.indexed
        if ix is None:
            return
        n, ok = QInputDialog.getInt(
            self, "Clean up speckles",
            "Replace a pixel when at least this many of its 8 neighbors agree:",
            7, 4, 8, 1)
        if not ok:
            return
        self.push_history("clean up speckles")
        mask = tools.despeckle(ix.indices, n, within=self.selection)
        self._record(mask)
        self.refresh()
        self.status_changed.emit(f"Cleaned {int(mask.sum())} speckled pixels", 4000)

    def zoom_to_selection(self):
        """Frame the selection. Refining a 15x20 face on a 96x144 grid otherwise
        means panning to it by hand every single time."""
        if self.selection is None or self.state.indexed is None:
            return
        ys, xs = np.nonzero(self.selection)
        self.canvas.zoom_to(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    # ---------- clipboard ----------
    def can_paste(self):
        return self._clip is not None

    def copy_selection(self):
        """Copy the selection.

        Stored as hex strings rather than slot indices: indices mean nothing once
        the palette changes, and this way a clip can be pasted into a different
        image entirely. An erased pixel copies as None so it pastes back erased
        rather than filling in.
        """
        ix = self.state.indexed
        if ix is None or self.selection is None:
            return 0
        ys, xs = np.nonzero(self.selection)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        mask = self.selection[y0:y1, x0:x1].copy()
        idx = ix.indices[y0:y1, x0:x1]
        hexes = np.full(mask.shape, None, dtype=object)
        for k in np.unique(idx[mask]):
            if k >= 0:
                hexes[mask & (idx == k)] = ix.slots[int(k)].hex
        self._clip = {'mask': mask, 'hex': hexes, 'origin': (y0, x0)}
        self._to_system_clipboard(mask, hexes)
        n = int(mask.sum())
        self.status_changed.emit(f"Copied {n} pixels", 3000)
        self.clipboard_changed.emit()
        return n

    def cut_selection(self):
        if self.copy_selection():
            self._fill_selection(EMPTY, "cut")

    def paste_clip(self):
        """Paste at the position it was copied from, clamped into the image.

        Landing it back where it came from is the predictable choice; the Move
        tool is right there to shift it, and the paste is left selected so Move
        can pick it up immediately.
        """
        ix = self.state.indexed
        if ix is None or self._clip is None:
            return
        mask, hexes = self._clip['mask'], self._clip['hex']
        h, w = mask.shape
        H, W = ix.shape
        if h > H or w > W:
            mask, hexes = mask[:H, :W], hexes[:H, :W]
            h, w = mask.shape
        y0, x0 = self._clip['origin']
        y0 = min(max(0, int(y0)), H - h)
        x0 = min(max(0, int(x0)), W - w)

        self.push_history("paste")
        by_hex = {s.hex: i for i, s in enumerate(ix.slots)}
        target = ix.indices[y0:y0 + h, x0:x0 + w]
        for hx in {v for v in hexes[mask].tolist() if v is not None}:
            if hx not in by_hex:
                # A clip from another image can carry colors this palette lacks;
                # bring the recipe along or it arrives unmixable.
                by_hex[hx] = ix.add_slot(hx, (self.state.recipes or {}).get(hx))
        for hx in {v for v in hexes[mask].tolist()}:
            spot = mask & (hexes == hx)
            target[spot] = EMPTY if hx is None else by_hex[hx]

        placed = np.zeros(ix.shape, dtype=bool)
        placed[y0:y0 + h, x0:x0 + w] = mask
        self._record(placed)
        self.refresh()
        self.set_selection(placed)
        self.status_changed.emit(f"Pasted {int(mask.sum())} pixels", 3000)

    def _to_system_clipboard(self, mask, hexes):
        """Also hand the clip to the OS as a transparent PNG, so it can go into
        another program. Pasting back in always uses the internal copy, which
        keeps exact palette colors instead of re-matching an image."""
        h, w = mask.shape
        argb = np.zeros((h, w, 4), dtype=np.uint8)
        for hx in {v for v in hexes[mask].tolist() if v is not None}:
            r, g, b = hex_to_rgb(hx)
            spot = mask & (hexes == hx)
            argb[spot] = (b, g, r, 255)          # QImage ARGB32 is BGRA in memory
        buf = np.ascontiguousarray(argb)
        img = QImage(buf.data, w, h, 4 * w, QImage.Format.Format_ARGB32).copy()
        QApplication.clipboard().setImage(img)

    # ---------- history ----------
    def commit_edit(self, label, fn):
        """Run a mutation as one undoable step and record it into the overlay.

        The Edit-menu dialogs own their own preview and rewind, so all they need
        from the stage is this: snapshot, mutate, record, redraw.
        """
        self.push_history(label)
        mask = fn()
        self._record(mask)
        self.refresh()
        return mask

    def push_history(self, label):
        if self.state.indexed is not None:
            self.history.push(self.state.indexed, label,
                              self.state.edit_overlay, self.state.edit_overlay_size)
            self.history_changed.emit()

    def _restore_overlay(self, overlay, size):
        """The overlay rides along with every snapshot; without this an undone
        stroke stays in the project file and comes back on the next load."""
        if overlay is not None:
            self.state.edit_overlay = overlay
            self.state.edit_overlay_size = size

    def undo(self):
        if self.state.indexed is None:
            return
        got = self.history.undo(self.state.indexed, self.state.edit_overlay,
                                self.state.edit_overlay_size)
        if got is not None:
            label, overlay, size = got
            self._restore_overlay(overlay, size)
            self.refresh()
            self.history_changed.emit()
            self.status_changed.emit(f"Undo {label}", 3000)

    def redo(self):
        if self.state.indexed is None:
            return
        got = self.history.redo(self.state.indexed, self.state.edit_overlay,
                                self.state.edit_overlay_size)
        if got is not None:
            label, overlay, size = got
            self._restore_overlay(overlay, size)
            self.refresh()
            self.history_changed.emit()
            self.status_changed.emit(f"Redo {label}", 3000)

    def revert_edits(self):
        """Throw away every manual edit and go back to the pipeline result."""
        if self.state.indexed_pristine is None:
            return
        n = len(self.state.edit_overlay)
        self.push_history("revert edits")
        self.state.indexed = self.state.indexed_pristine.copy()
        self.state.edit_overlay = {}
        self.state.edit_overlay_size = None
        self.refresh()
        self.history_changed.emit()
        self.status_changed.emit(f"Reverted {n} manual edits", 4000)

    # ---------- painting ----------
    def _value_for(self, buttons):
        """Which slot a button paints. Eraser always clears."""
        if self.tool == ERASER:
            return EMPTY
        return self.fg if (buttons & _LEFT) else self.bg

    def _record(self, mask):
        """Remember changed pixels so a later Apply can stamp them back on."""
        if mask is None or not mask.any():
            return
        ix = self.state.indexed
        self.state.edit_overlay_size = ix.shape
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            k = int(ix.indices[y, x])
            self.state.edit_overlay[(y, x)] = None if k < 0 else ix.slots[k].hex

    def _on_press(self, x, y, button):
        ix = self.state.indexed
        if ix is None:
            return
        # QWidget has no keyboardModifiers(); ask the application instead.
        alt = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
        if self.tool == DROPPER or alt:
            k = int(ix.indices[y, x])
            self.set_color(k, 'bg' if button == _RIGHT else 'fg')
            return
        if self.tool == RECT:
            self._shape = [(x, y), (x, y)]
            self.canvas.set_pending('rect', self._shape)
            return
        if self.tool == LASSO:
            self._shape = [(x, y)]
            self.canvas.set_pending('lasso', self._shape)
            return
        if self.tool == WAND:
            self._commit_selection(sel.wand_mask(
                ix, x, y, float(self.spin_tol.value()),
                connectivity=4 if self.cmb_conn.currentIndex() == 0 else 8,
                contiguous=not self.chk_wand_all.isChecked()))
            return
        if self.tool == MOVE:
            self._start_move(x, y)
            return
        if self.tool == BUCKET:
            self.push_history("fill")
            mask = tools.flood_fill(
                ix.indices, x, y, self._value_for(button),
                connectivity=4 if self.cmb_conn.currentIndex() == 0 else 8,
                whole_image=self.chk_global.isChecked(), within=self.selection)
            self._record(mask)
            self.refresh()
            return
        self.push_history("stroke")
        self._stroke = tools.stamp(ix.indices, x, y, self._value_for(button),
                                   self.spin_size.value(), self.cmb_shape.currentText(),
                                   within=self.selection)
        self._last = (x, y)
        self.refresh()

    # ---------- move ----------
    def _start_move(self, x, y):
        """Lift the selected pixels off the canvas so the drag can preview."""
        ix = self.state.indexed
        if self.selection is None or not self.selection[y, x]:
            self.status_changed.emit("Select something first, then drag it.", 3000)
            return
        self.push_history("move selection")
        base = ix.indices.copy()
        base[self.selection] = EMPTY
        self._move = {'from': (x, y), 'vals': ix.indices.copy(),
                      'mask': self.selection.copy(), 'base': base,
                      'touched': self.selection.copy()}

    def _drag_move(self, x, y):
        m = self._move
        fx, fy = m['from']
        moved = sel.move_pixels(self.state.indexed.indices, m['mask'], m['vals'],
                                m['base'], x - fx, y - fy)
        m['touched'] |= moved
        self.set_selection(moved)
        self.refresh()

    def _on_drag(self, x, y, buttons):
        ix = self.state.indexed
        if ix is None:
            return
        if self._move is not None:
            self._drag_move(x, y)
            return
        if self._shape is not None:
            if self.tool == RECT:
                self._shape[1] = (x, y)
            elif (x, y) != self._shape[-1]:
                self._shape.append((x, y))
            self.canvas.set_pending('rect' if self.tool == RECT else 'lasso', self._shape)
            return
        if self._stroke is None or self._last is None:
            return
        lx, ly = self._last
        mask = tools.line(ix.indices, lx, ly, x, y, self._value_for(buttons),
                          self.spin_size.value(), self.cmb_shape.currentText(),
                          within=self.selection)
        self._stroke |= mask
        self._last = (x, y)
        self.refresh()

    def _on_release(self, _x, _y, _button):
        if self._move is not None:
            self._record(self._move['touched'])
            self._move = None
            self.refresh()
            return
        if self._shape is not None:
            pts, self._shape = self._shape, None
            self.canvas.set_pending(None, None)
            shape = self.state.indexed.shape
            if self.tool == RECT:
                (x0, y0), (x1, y1) = pts
                self._commit_selection(sel.rect_mask(shape, x0, y0, x1, y1))
            else:
                self._commit_selection(sel.lasso_mask(shape, pts))
            return
        if self._stroke is not None:
            self._record(self._stroke)
        self._stroke = None
        self._last = None

    # ---------- handlers ----------
    def _sync_tool_options(self):
        painting = self.tool in (BRUSH, ERASER)
        wand = self.tool == WAND
        for widget, want in ((self.spin_size, painting), (self.cmb_shape, painting),
                             (self.cmb_conn, self.tool == BUCKET or wand),
                             (self.chk_global, self.tool == BUCKET),
                             (self.spin_tol, wand), (self.chk_wand_all, wand)):
            row, _ = self._opt_form.getWidgetPosition(widget)
            if row >= 0:
                self._opt_form.setRowVisible(row, want)
        self.opt_box.setVisible(self.tool not in (DROPPER, RECT, LASSO, MOVE))

    def _refresh_selection_info(self):
        on = self.selection is not None
        if on:
            n = int(self.selection.sum())
            pct = 100.0 * n / max(self.selection.size, 1)
            self.lbl_sel.setText(f"{n} px selected ({pct:.1f}%)")
        else:
            self.lbl_sel.setText("Nothing selected.")
        for b in (self.btn_sel_none, self.btn_fill_sel, self.btn_erase_sel):
            b.setEnabled(on)

    def _swatch_text(self, slot):
        ix = self.state.indexed
        if slot is None or slot < 0 or ix is None or slot >= len(ix.slots):
            return "erase", "#000000", "#ffffff"
        hx = ix.slots[slot].hex
        c = QColor(hx)
        fg = "#000000" if (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) > 140 else "#ffffff"
        return hx, hx, fg

    def _refresh_swatches(self):
        for btn, slot, side in ((self.btn_fg, self.fg, "left"), (self.btn_bg, self.bg, "right")):
            text, bg, fg = self._swatch_text(slot)
            btn.setText(f"{side}\n{text}")
            btn.setStyleSheet(f"background-color: {bg}; color: {fg};")

    def _on_grid(self):
        self.canvas.set_grid(self.chk_grid.isChecked(), self.spin_grid.value())

    def _on_hover(self, x, y):
        ix = self.state.indexed
        if ix is None or x < 0:
            self._refresh_info()
            return
        k = int(ix.indices[y, x])
        if k < 0:
            self.lbl_info.setText(f"{x}, {y}\nempty (no paint)")
            return
        s = ix.slots[k]
        recipe = " ".join(s.recipe) if s.recipe else "no recipe"
        self.lbl_info.setText(f"{x}, {y}\nslot {k} · {s.hex}\n{recipe}")

    def _refresh_info(self):
        ix = self.state.indexed
        if ix is None:
            self.lbl_info.setText("-")
            return
        counts = ix.used_counts()
        used = int((counts > 0).sum())
        drops = sum(len(s.recipe) for s, c in zip(ix.slots, counts) if c > 0)
        empty = int((ix.indices < 0).sum())
        self.lbl_info.setText(
            f"{ix.shape[1]}×{ix.shape[0]} px\n"
            f"{used} colors · {drops} drops to mix\n"
            f"{empty} empty px · {len(self.state.edit_overlay)} manual edits")
        