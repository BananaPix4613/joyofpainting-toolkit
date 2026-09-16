"""Indexed image: an ordered palette plus a grid of slot numbers.

The pipeline used to hand downstream stages an object array of hex strings, which
made palette and pixels the same thing. "Change this palette entry" had nowhere to
happen. Splitting them means a palette edit is an array operation:

    change slot k's color   -> slots[k].hex = new           (pixels untouched)
    replace slot k with j   -> indices[indices == k] = j    (slot k removed)
    add a color             -> append a slot                (affects nothing yet)

Index -1 means "no pixel here": the Paint stage skips it and the sidecare writes
null. This is what an eraser produces.
"""

from dataclasses import dataclass, field

import numpy as np

EMPTY = -1


@dataclass
class Slot:
    """One palette entry. `hex` is always a color the mod can actually mix."""
    hex: str
    recipe: list = field(default_factory=list)   # dye names
    locked: bool = True                          # came from the pipeline, not the user

    def to_dict(self):
        return {'hex': self.hex, 'recipe': list(self.recipe), 'locked': bool(self.locked)}

    @staticmethod
    def from_dict(d):
        return Slot(hex=d['hex'], recipe=list(d.get('recipe', [])),
                    locked=bool(d.get('locked', True)))


class IndexedImage:
    """Ordered slots + an (H, W) int16 index grid. Slot order is paint order."""

    def __init__(self, slots, indices):
        self.slots = list(slots)
        self.indices = np.asarray(indices, dtype=np.int16)

    # ---------- construction ----------
    @staticmethod
    def from_matched(matched_hex_grid, palette, recipes=None):
        """Build from the pipeline's hex grid. Slot order follows first use, so
        the palette reads in roughly the order the image introduces colors."""
        flat = matched_hex_grid.reshape(-1)
        order, seen = [], {}
        for h in flat.tolist():
            if h not in seen:
                seen[h] = len(order)
                order.append(h)
        slots = [Slot(hex=h, recipe=list(recipes[h]) if recipes else [], locked=True)
                 for h in order]
        idx = np.array([seen[h] for h in flat.tolist()], dtype=np.int16)
        return IndexedImage(slots, idx.reshape(matched_hex_grid.shape))

    def copy(self):
        return IndexedImage([Slot(s.hex, list(s.recipe), s.locked) for s in self.slots],
                            self.indices.copy())

    # ---------- views ----------
    @property
    def shape(self):
        return self.indices.shape

    def hex_array(self):
        """Slot colors as an object array, for indexing by the grid."""
        return np.array([s.hex for s in self.slots], dtype=object)

    def to_hex_grid(self, empty='#000000'):
        """(H, W) object array of hex strings. Legacy consumers still want this."""
        lut = np.append(self.hex_array(), np.array([empty], dtype=object))
        return lut[np.where(self.indices < 0, len(self.slots), self.indices)]

    def to_rgb(self, empty=(64, 64, 64)):
        """(H, W, 3) uint8 for display. Empty pixels get `empty`."""
        from .colors import hex_to_rgb
        lut = np.array([hex_to_rgb(s.hex) for s in self.slots] + [tuple(empty)], dtype=np.uint8)
        return lut[np.where(self.indices < 0, len(self.slots), self.indices)]

    def used_counts(self):
        """Pixels per slot. Index -1 is not counted."""
        flat = self.indices.reshape(-1)
        return np.bincount(flat[flat >= 0], minlength=len(self.slots))

    # ---------- palette edits ----------
    def set_slot_color(self, k, hex_color, recipe=None):
        """Recolor a slot in place. Every pixel using it changes; count is unchanged."""
        self.slots[k].hex = hex_color
        if recipe is not None:
            self.slots[k].recipe = list(recipe)
        self.slots[k].locked = False

    def add_slot(self, hex_color, recipe=None):
        """Append a paint-only slot. Nothing in the image references it yet."""
        self.slots.append(Slot(hex=hex_color, recipe=list(recipe or []), locked=False))
        return len(self.slots) - 1

    def merge_slot(self, k, into):
        """Every pixel of slot k becomes slot `into`, then k is removed."""
        if k == into or not (0 <= k < len(self.slots)) or not (0 <= into < len(self.slots)):
            return
        self.indices[self.indices == k] = into
        self._drop([k])

    def remove_slot(self, k):
        """Remove a slot; pixels using it become empty."""
        self.indices[self.indices == k] = EMPTY
        self._drop([k])

    def compact(self):
        """Drop every unused slot and renumber. Returns the removed slot indices."""
        counts = self.used_counts()
        dead = [i for i, c in enumerate(counts) if c == 0]
        if dead:
            self._drop(dead)
        return dead

    def reorder(self, order):
        """`order` lists old slot indices in their new positions."""
        remap = np.full(len(self.slots), EMPTY, dtype=np.int16)
        for new, old in enumerate(order):
            remap[old] = new
        self.slots = [self.slots[o] for o in order]
        m = self.indices >= 0
        self.indices[m] = remap[self.indices[m]]

    def _drop(self, dead):
        """Remove slots by index and renumber the grid to match."""
        dead = set(dead)
        keep = [i for i in range(len(self.slots)) if i not in dead]
        remap = np.full(len(self.slots), EMPTY, dtype=np.int16)
        for new, old in enumerate(keep):
            remap[old] = new
        m = self.indices >= 0
        self.indices[m] = remap[self.indices[m]]
        self.slots = [self.slots[i] for i in keep]

    # ---------- geometry ----------
    def pad_to_multiple(self, mult=16):
        """Edge-pad the grid so a canvas layout covers it. Slots are unchanged."""
        h, w = self.indices.shape
        ph, pw = (-h) % mult, (-w) % mult
        if ph or pw:
            self.indices = np.pad(self.indices, ((0, ph), (0, pw)), mode='edge')
        return self

    # ---------- edit overlay ----------
    def apply_overlay(self, overlay, recipes=None):
        """Stamp sparse manual edits back on after a re-run of the pipeline.
        
        `overlay` maps (y, x) -> slot hex, or None to erase. Keyed by hex, not
        index: indices are meaningless across a regenerated palette. Colors
        missing from the new palette are appended so an edit is never silently
        dropped. `recipes` (hex -> [dye, ...]) supplies the mix for those, without
        which an edited-in color comes back unmixable after a re-Apply.
        Returns how many landed.
        """
        if not overlay:
            return 0
        by_hex = {s.hex: i for i, s in enumerate(self.slots)}
        h, w = self.indices.shape
        n = 0
        for (y, x), hx in overlay.items():
            if not (0 <= y < h and 0 <= x < w):
                continue
            if hx is None:
                self.indices[y, x] = EMPTY
            else:
                if hx not in by_hex:
                    by_hex[hx] = self.add_slot(hx, (recipes or {}).get(hx))
                self.indices[y, x] = by_hex[hx]
            n += 1
        return n

    # ---------- persistence ----------
    def to_dict(self):
        return {'slots': [s.to_dict() for s in self.slots],
                'indices': self.indices.astype(np.int16).tolist()}

    @staticmethod
    def from_dict(d):
        return IndexedImage([Slot.from_dict(s) for s in d['slots']],
                            np.array(d['indices'], dtype=np.int16))
