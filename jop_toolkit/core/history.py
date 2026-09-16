"""Undo/redo for the Edit stage.

Snapshot-based rather than command-based on purpose: an index grid at painting
resolution is ~27 KB, so 200 levels costs about 6 MB and every operation (brush
stroke, bucket fill, palette merge, slot reorder) is undoable through one code
path with no per-command inverse to get wrong.

The sparse edit overlay travels in the snapshot too. It is what the sidecar is
built from, so rolling back the pixels without it would leave the project file
describing edits that are no longer on the image, so they would reappear on the
next load.
"""

import numpy as np

from .indexed import Slot

# Rough per-entry cost of the overlay dict, for the memory cap. A filled
# selection can put one entry per pixel in there, which is not free.
_OVERLAY_ENTRY_BYTES = 120


def _snapshot(indexed, overlay=None, overlay_size=None):
    return (indexed.indices.copy(),
            [Slot(s.hex, list(s.recipe), s.locked) for s in indexed.slots],
            None if overlay is None else dict(overlay),
            overlay_size)


def _restore(indexed, snap):
    """Returns the overlay and its size, for the caller to assign back."""
    indexed.indices = snap[0].copy()
    indexed.slots = [Slot(s.hex, list(s.recipe), s.locked) for s in snap[1]]
    return (None if snap[2] is None else dict(snap[2])), snap[3]


def _cost(snap):
    return snap[0].nbytes + len(snap[2] or ()) * _OVERLAY_ENTRY_BYTES


class History:
    """Call push() with the state as it is *before* an edit, then mutate."""

    def __init__(self, limit=200, max_bytes=192 * 1024 * 1024):
        self._undo = []
        self._redo = []
        self._limit = int(limit)
        self._max_bytes = int(max_bytes)
        self._bytes = 0

    def clear(self):
        self._undo.clear()
        self._redo.clear()
        self._bytes = 0

    def push(self, indexed, label="", overlay=None, overlay_size=None):
        """Record the pre-edit state. Any new edit invalidates the redo branch."""
        snap = _snapshot(indexed, overlay, overlay_size)
        self._undo.append((snap, label))
        self._bytes += _cost(snap)
        self._redo.clear()
        while len(self._undo) > self._limit or (self._bytes > self._max_bytes and len(self._undo) > 1):
            old, _ = self._undo.pop(0)
            self._bytes -= _cost(old)

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def undo_label(self):
        return self._undo[-1][1] if self._undo else ""

    def redo_label(self):
        return self._redo[-1][1] if self._redo else ""

    def undo(self, indexed, overlay=None, overlay_size=None):
        """Returns (label, overlay, overlay_size), or None if there is nothing
        to undo. The overlay comes back for the caller to assign."""
        if not self._undo:
            return None
        snap, label = self._undo.pop()
        self._bytes -= _cost(snap)
        self._redo.append((_snapshot(indexed, overlay, overlay_size), label))
        ov, size = _restore(indexed, snap)
        return label, ov, size

    def redo(self, indexed, overlay=None, overlay_size=None):
        if not self._redo:
            return None
        snap, label = self._redo.pop()
        self._undo.append((_snapshot(indexed, overlay, overlay_size), label))
        self._bytes += _cost(snap)
        ov, size = _restore(indexed, snap)
        return label, ov, size
