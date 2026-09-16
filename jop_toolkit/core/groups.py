"""Deliberate palette allocation: split the color budget across named groups.

Plain weighted k-means minimizes squared error, so it allocates by
variance x mass instead of by what matters. On a dark/red image with a small
face it spends 36% of a 28-color budget on roses and gives skin one color. No
statistical objective recovers that, because "theface matters more than a rose"
is not in the pixels. This module lets the user say so.

A group is a *selector* plus a budget. Today the selector is a core color and
membership is nearest-core in Lab. The partition step is deliberately isolated in
`assign_groups` so a painted mask can later become an additional selector without
disturbing budget distribution, k-means, or snapping.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import axis_core as AX
from .colors import hex_to_rgb, rgb_to_hex
from .colorspace import srgb_to_lab
from .palette import _weighted_kmeans


@dataclass
class ColorGroup:
    """One row of the palette-groups table."""
    core_hex: str
    min_colors: int = 1
    max_colors: int = 64
    enabled: bool = True
    label: str = ""
    position: float = 0.5      # normalized spot on the number line (axis mode)
    pinned: bool = False       # core_hex was chosen, not read off the strip

    def to_dict(self):
        return {'core_hex': self.core_hex, 'min_colors': int(self.min_colors),
                'max_colors': int(self.max_colors), 'enabled': bool(self.enabled),
                'label': self.label, 'position': float(self.position),
                'pinned': bool(self.pinned)}

    @staticmethod
    def from_dict(d):
        return ColorGroup(core_hex=d['core_hex'],
                          min_colors=int(d.get('min_colors', 1)),
                          max_colors=int(d.get('max_colors', 64)),
                          enabled=bool(d.get('enabled', True)),
                          label=d.get('label', ''),
                          position=float(d.get('position', 0.5)),
                          pinned=bool(d.get('pinned', False)))


def auto_groups(src_rgb, palette, n_groups=6, seed=0):
    """Seed groups from the image: k-means at small k, each centroid snapped to a
    reachable mix. A starting point for the user to edit, not an answer."""
    src = np.asarray(src_rgb)
    uq, cnt = np.unique(src.reshape(-1, 3).astype(np.int64), axis=0, return_counts=True)
    cen = _weighted_kmeans(srgb_to_lab(uq.astype(np.float64)), cnt.astype(np.float64),
                           min(n_groups, len(uq)), seed=seed)
    keys = list(palette.keys())
    ptree = cKDTree(srgb_to_lab(np.array(list(palette.values()), dtype=np.float64)))
    _, nn = ptree.query(cen, k=1, workers=-1)
    out, seen = [], set()
    for i in nn:
        h = keys[int(i)]
        if h not in seen:
            seen.add(h)
            out.append(ColorGroup(core_hex=h))
    return out


def assign_groups(uq_lab, groups, weights=None, mode='cores', axis='lightness'):
    """Map each unique color to a group index.

    'cores' -- nearest core color in 3-D Lab (Voronoi over the anchors).
    'axis'  -- nearest marker along the chosen 1-D axis. A projection, so colors
               differing only off-axis land together; the axis-quality readout in
               the UI exists to make that visible.

    This is the single partition step, kept isolated so a painted mask can later
    claim its pixels here without touching budgets, k-means, or snapping.
    """
    if mode == 'axis':
        w = np.ones(len(uq_lab)) if weights is None else weights
        return AX.assign_by_markers(uq_lab, w, axis, [g.position for g in groups])
    cores = srgb_to_lab(np.array([hex_to_rgb(g.core_hex) for g in groups], dtype=np.float64))
    _, idx = cKDTree(cores).query(uq_lab, k=1, workers=-1)
    return idx.astype(np.int32)


def distribute_budget(groups, weights, sse, total):
    """Hand out `total` colors:every group starts at its min, then each extra
    goes to whichever group currently has the largest error-per-color."""
    n = len(groups)
    alloc = np.array([max(0, int(g.min_colors)) for g in groups], dtype=np.int64)
    caps = np.array([max(0, int(g.max_colors)) for g in groups], dtype=np.int64)
    alloc = np.minimum(alloc, caps)
    alloc[weights <= 0] = 0                     # nothing assigned to this group
    remaining = int(total) - int(alloc.sum())
    while remaining > 0:
        gain = np.where((alloc < caps) & (weights > 0), sse / np.maximum(alloc, 1), -1.0)
        if not np.any(gain > 0):
            break
        alloc[int(np.argmax(gain))] += 1
        remaining -= 1
    if remaining < 0:                           # mins oversubscribed the budget
        order = np.argsort(-alloc)
        i = 0
        while remaining < 0 and np.any(alloc > 0):
            j = order[i % n]
            if alloc[j] > 0:
                alloc[j] -= 1
                remaining += 1
            i += 1
    return alloc


def select_palette_grouped(src_rgb, palette, groups, max_colors, seed=0,
                           mode='cores', axis='lightness'):
    """Per-group weighted k-means with pinned core colors.
    
    Returns (hex_keys, allocation) where allocation[i] is how many colors group i
    actually received. The UI shows it back so the split is visible.
    """
    active = [g for g in groups if g.enabled]
    if not active:
        from .palette import select_palette
        return select_palette(src_rgb, palette, max_colors, seed=seed), []

    src = np.asarray(src_rgb)
    uq, cnt = np.unique(src.reshape(-1, 3).astype(np.int64), axis=0, return_counts=True)
    uq_lab = srgb_to_lab(uq.astype(np.float64))
    w = cnt.astype(np.float64)
    gi = assign_groups(uq_lab, active, weights=w, mode=mode, axis=axis)

    weights = np.zeros(len(active))
    sse = np.zeros(len(active))
    for k in range(len(active)):
        m = gi == k
        if not m.any():
            continue
        weights[k] = w[m].sum()
        mean = (uq_lab[m] * w[m, None]).sum(0) / weights[k]
        sse[k] = float((((uq_lab[m] - mean) ** 2).sum(1) * w[m]).sum())

    alloc = distribute_budget(active, weights, sse, max_colors)

    keys = list(palette.keys())
    ptree = cKDTree(srgb_to_lab(np.array(list(palette.values()), dtype=np.float64)))
    chosen = []
    for k, g in enumerate(active):
        if alloc[k] <= 0:
            continue
        if g.core_hex in palette:
            chosen.append(g.core_hex)
        else:
            _, ci = ptree.query(srgb_to_lab(np.array(hex_to_rgb(g.core_hex), dtype=np.float64)), k=1)
            chosen.append(keys[int(ci)])
        m = gi == k
        free = int(alloc[k]) - 1
        if free <= 0 or not m.any():
            continue
        pts, wts = uq_lab[m], w[m]
        cen = _weighted_kmeans(pts, wts, min(free, len(pts)), seed=seed)
        _, nn = ptree.query(cen, k=1, workers=-1)
        chosen.extend(keys[int(i)] for i in nn)

    out, seen = [], set()
    for h in chosen:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out, alloc.tolist()
