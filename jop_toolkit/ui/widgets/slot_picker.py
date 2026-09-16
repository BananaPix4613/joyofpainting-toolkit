"""Modal 'pick a target slot' dialog, used by Replace with.

A dropdown of hex strings is unreadable for a palette of colors, so this reuses
the same grid the panel shows.
"""

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QScrollArea

from .palette_grid import PaletteGrid


class SlotPicker(QDialog):
    """Returns the chosen slot index, or None if cancelled."""

    def __init__(self, indexed, exclude=None, parent=None, prompt="Pick a color"):
        super().__init__(parent)
        self.setWindowTitle(prompt)
        self.resize(360, 320)
        self._chosen = None
        self._exclude = exclude
        self._indexed = indexed

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(prompt))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.grid = PaletteGrid()
        self.grid.set_indexed(indexed)
        self.grid.set_barred(exclude)
        scroll.setWidget(self.grid)
        lay.addWidget(scroll, 1)

        self.lbl = QLabel("-")
        lay.addWidget(self.lbl)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                        QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        lay.addWidget(self.buttons)

        self.grid.slot_picked.connect(self._on_pick)

    def _on_pick(self, slot, _button):
        # The barred slot no longer emits at all; erase still can.
        if slot < 0:
            self._chosen = None
            self.lbl.setText("Erase is not a merge target.")
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        self._chosen = slot
        self.lbl.setText(f"slot {slot} · {self._indexed.slots[slot].hex}")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def chosen(self):
        return self._chosen
