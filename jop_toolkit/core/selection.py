"""Selection masks for the Edit stage.

A selection is a plain bool array the size of the index grid. Every function in
tools.py already accepts a `within=` mask, so restricting an operation to the
selection costs nothing once the mask exists.

Kept free of Qt so the shapes are testable on their own.
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

from .colors import hex_to_rgb
from .colorspace import srgb_to_lab


def rect_mask(shape, x0, y0, x1, y1):
    """Axis-aligned marquee. Both corners are inclusive."""
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    xa, xb = sorted((int(x0), int(x1)))
    ya, yb = sorted((int(y0), int(y1)))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w - 1, xb), min(h - 1, yb)
    if xb < xa or yb < ya:
        return m
    m[ya:yb + 1, xa:xb + 1] = True
    return m


def lasso_mask(shape, points):
    """Even-odd polygon fill of a freehand path.

    Pillow is already a dependency and its polygon rasterizer handles
    self-intersecting paths, which a hand-rolled scanline fill does not.
    """
    h, w = shape
    pts = [(int(x), int(y)) for x, y in points]
    if len(pts) < 3:
        return np.zeros((h, w), dtype=bool)
    img = Image.new('1', (w, h), 0)
    ImageDraw.Draw(img).polygon(pts, fill=1, outline=1)
    return np.asarray(img, dtype=bool)


def slot_labs(indexed):
    """Lab coordinates for every slot, for tolerance comparisons."""
    if not indexed.slots:
        return np.zeros((0, 3), dtype=np.float64)
    rgb = np.array([hex_to_rgb(s.hex) for s in indexed.slots], dtype=np.float64)
    return srgb_to_lab(rgb)


def wand_mask(indexed, x, y, tolerance=0.0, connectivity=8, contiguous=True):
    """Pixels similar to the one under the cursor.

    Tolerance is a Lab distance, not a distance between slot numbers: once the
    palette has been sorted, neighboring indices have nothing to do with
    neighboring colors.
    """
    ind = indexed.indices
    h, w = ind.shape
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), dtype=bool)
    target = int(ind[y, x])
    if target < 0:
        match = ind < 0                       # empty selects every empty pixel
    elif tolerance <= 0:
        match = ind == target
    else:
        labs = slot_labs(indexed)
        d = np.linalg.norm(labs - labs[target], axis=1)
        match = np.isin(ind, np.nonzero(d <= tolerance)[0])
    if contiguous:
        struct = (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool) if connectivity == 4
                  else np.ones((3, 3), bool))
        comp, _ = label(match, structure=struct)
        match = comp == comp[y, x]
    return match


def outline_segments(mask):
    """Boundary edges of `mask` as an (n, 4) array of x0, y0, x1, y1.

    Left in image-space integers rather than a painter path: the canvas scales
    them by the current zoom at paint time, and the outline would otherwise have
    to be rebuilt on every zoom step as well as every selection change.
    """
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    segs = []
    sides = ((mask & ~pad[0:h, 1:w + 1], (0, 0, 1, 0)),       # top
             (mask & ~pad[2:h + 2, 1:w + 1], (0, 1, 1, 1)),   # bottom
             (mask & ~pad[1:h + 1, 0:w], (0, 0, 0, 1)),       # left
             (mask & ~pad[1:h + 1, 2:w + 2], (1, 0, 1, 1)))   # right
    for m, (ax, ay, bx, by) in sides:
        ys, xs = np.nonzero(m)
        if len(ys):
            segs.append(np.stack([xs + ax, ys + ay, xs + bx, ys + by], axis=1))
    if not segs:
        return np.zeros((0, 4), dtype=np.int32)
    return np.concatenate(segs).astype(np.int32)


def move_pixels(indices, mask, values, base, dx, dy):
    """Stamp the lifted `values` at an offset over `base` and return where they
    landed.
    
    Rewritten from `base` on every drag step rather than shifted incrementally:
    a drag that wanders back and forth would otherwise smear the lifted pixels
    across the canvas.
    """
    h, w = indices.shape
    indices[:] = base
    ys, xs = np.nonzero(mask)
    ny, nx = ys + int(dy), xs + int(dx)
    keep = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    indices[ny[keep], nx[keep]] = values[ys[keep], xs[keep]]
    moved = np.zeros((h, w), dtype=bool)
    moved[ny[keep], nx[keep]] = True
    return moved
