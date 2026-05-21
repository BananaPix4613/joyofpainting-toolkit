"""Image preprocessing: downscale + edge-smear padding."""

import numpy as np
from PIL import Image


def load_image(path):
    return Image.open(path).convert('RGB')


def downscale(img, target_size):
    """Resize a PIL.Image to target_size = (W, H) using high-quality Lanczos."""
    if img.size == tuple(target_size):
        return img
    return img.resize(tuple(target_size), Image.LANCZOS)


def smear_pad(arr, margin):
    """Pad an (H, W, 3) uint8 array by `margin` px on every side, repeating edge pixels."""
    if margin <= 0:
        return arr
    return np.pad(arr, ((margin, margin), (margin, margin), (0, 0)), mode='edge')


def directional_margin(arr, top, right, bottom, left):
    """Apply per-side margins. Positive values smear-pad the edge outward;
    negative values crop that many pixels inward from that side.
    """
    h, w = arr.shape[:2]
    # Crop first (negative sides)
    y0 = max(0, -top)
    y1 = h - max(0, -bottom)
    x0 = max(0, -left)
    x1 = w - max(0, -right)
    if y1 <= y0 or x1 <= x0:
        # Would crop everything; keep a single edge row/col so the pipeline still works.
        y0 = min(y0, h - 1); y1 = max(y1, y0 + 1)
        x0 = min(x0, w - 1); x1 = max(x1, x0 + 1)
    cropped = arr[y0:y1, x0:x1]
    # Then pad (positive sides)
    pt = max(0, top); pr = max(0, right); pb = max(0, bottom); pl = max(0, left)
    if pt or pr or pb or pl:
        return np.pad(cropped, ((pt, pb), (pl, pr), (0, 0)), mode='edge')
    return cropped


def pad_to_multiple(arr, mult=16):
    """Pad an (H, W, 3) array on right/bottom to the next multiple of `mult`, smearing the edge."""
    h, w = arr.shape[:2]
    pad_w = (-w) % mult
    pad_h = (-h) % mult
    if pad_w == 0 and pad_h == 0:
        return arr
    return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')


def hex_grid_to_rgb(hex_grid, palette):
    """Convert a 2D object array of hex strings into an (H, W, 3) uint8 RGB array."""
    h, w = hex_grid.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    cache = {}
    for y in range(h):
        for x in range(w):
            hx = hex_grid[y, x]
            rgb = cache.get(hx)
            if rgb is None:
                rgb = palette[hx]
                cache[hx] = rgb
            out[y, x] = rgb
    return out
