"""Pixel operations on an index grid.

Every function mutates `indices` in place and returns a bool mask of what it
changed, so the Edit stage can record exactly those pixels into the sparse edit
overlay without diffing whole arrays.

Kept free of Qt so the behavior is testable on its own.
"""

import numpy as np
from scipy.ndimage import label

SQUARE = 'square'
ROUND = 'round'


def brush_offsets(size, shape=SQUARE):
    """Offsets covered by a brush of `size`, centered on the cursor pixel.

    Even sizes cannot be centered exactly; they extend one further right/down,
    which is what every pixel editor does.
    """
    size = max(1, int(size))
    lo = -((size - 1) // 2)
    hi = size // 2
    ys, xs = np.mgrid[lo:hi + 1, lo:hi + 1]
    if shape == ROUND and size > 2:
        r = size / 2.0
        keep = (xs ** 2 + ys ** 2) <= r * r
        return ys[keep], xs[keep]
    return ys.ravel(), xs.ravel()


def stamp(indices, x, y, value, size=1, shape=SQUARE, within=None):
    """Paint one brush dab. `within` optionally restricts to a selection mask."""
    h, w = indices.shape
    oy, ox = brush_offsets(size, shape)
    inside = ((y + oy >= 0) & (y + oy < h) & (x + ox >= 0) & (x + ox < w))
    ys = np.clip(y + oy, 0, h - 1)[inside]
    xs = np.clip(x + ox, 0, w - 1)[inside]
    changed = np.zeros((h, w), dtype=bool)
    if len(ys) == 0:
        return changed
    sel = np.zeros((h, w), dtype=bool)
    sel[ys, xs] = True
    if within is not None:
        sel &= within
    sel &= indices != value
    indices[sel] = value
    return sel


def line(indices, x0, y0, x1, y1, value, size=1, shape=SQUARE, within=None):
    """Bresenham between two points, stamping at each step.
    
    Mouse move events arrive at screen resolution, so a fast drag skips whole
    pixels; without this a stroke comes out as dots.
    """
    changed = np.zeros(indices.shape, dtype=bool)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        changed |= stamp(indices, x0, y0, value, size, shape, within)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return changed


def flood_fill(indices, x, y, value, connectivity=4, whole_image=False,
               match_values=None, within=None):
    """Fill from (x, y).

    whole_image ignores connectivity and replaces every matching pixel, which is
    the same thing as a palette-level replace but reachable from the canvas.
    match_values lets a tolerance setting treat several slots as equal.
    """
    h, w = indices.shape
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), dtype=bool)
    target = int(indices[y, x])
    match = np.isin(indices, list(match_values)) if match_values else (indices == target)
    if not whole_image:
        struct = (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool) if connectivity == 4
                  else np.ones((3, 3), bool))
        lab, _ = label(match, structure=struct)
        match = lab == lab[y, x]
    if within is not None:
        match &= within
    match &= indices != value
    indices[match] = value
    return match


def despeckle(indices, min_neighbors=7, within=None):
    """Remove isolated pixels: any pixel where at least `min_neighbors` of its 8
    neighbors share one other value becomes that value.

    This is the automated form of the hand-cleanup that flat-color sources need
    after quantization.

    Counted per palette value over the whole grid rather than per pixel over its
    neighborhood: the per-pixel form runs one np.unique per pixel, which is
    minutes on anything larger than a thumbnail.
    """
    h, w = indices.shape
    pad = np.pad(indices, 1, mode='edge')
    best_count = np.zeros((h, w), dtype=np.int8)
    best_val = indices.copy()
    for v in np.unique(indices):
        eq = (pad == v)
        c = np.zeros((h, w), dtype=np.int8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    c += eq[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        # Ascending values, strict >: ties go to the lowest slot index.
        better = c > best_count
        best_count = np.where(better, c, best_count)
        best_val = np.where(better, v, best_val)
    changed = (best_count >= min_neighbors) & (best_val != indices)
    if within is not None:
        changed &= within
    indices[changed] = best_val[changed]
    return changed
