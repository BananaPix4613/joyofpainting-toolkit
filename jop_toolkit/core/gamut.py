"""Map source colors into the lightness range the dye palette can reach.

No mix can be darker than black dye (#1D1D21, L*=10.92) or lighter than white
dye, at any depth. Nearest-neighbor matching crushes everything below that
floor onto a single color, which is what destroys dark gradients. Compressing
the source lightness into the reachable span first keeps those tones distinct.
"""

import numpy as np

from .colorspace import srgb_to_lab, lab_to_srgb


def palette_l_range(palette):
    """(min L*, max L*) actually reachable by this palette."""
    lab = srgb_to_lab(np.array(list(palette.values()), dtype=np.float64))
    return float(lab[:, 0].min()), float(lab[:, 0].max())


def compress_lightness(rgb, l_min, l_max, knee=0.35, strength=1.0):
    """Lift crushed shadows and pull in blown highlights.
    
    Only the ends move: L* between the two knees is untouched, so midtones keep
    their exact values and only out-of-gamut extremes are remapped. `strength`
    blends between no mapping (0.0) and full mapping (1.0).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if strength <= 0:
        return rgb
    lab = srgb_to_lab(rgb)
    L = lab[..., 0]
    span = l_max - l_min
    lo_k = l_min + knee * span
    hi_k = l_max - knee * span
    out = L.copy()
    m = L < lo_k
    if m.any():
        out[m] = l_min + (L[m] / lo_k) * (lo_k - l_min)
    m = L > hi_k
    if m.any():
        out[m] = hi_k + ((L[m] - hi_k) / (100.0 - hi_k)) * (l_max - hi_k)
    lab = lab.copy()
    lab[..., 0] = L + strength * (out - L)
    return lab_to_srgb(lab)
