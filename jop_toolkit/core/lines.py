"""Find the outlines in an image, as a selection.

Noisy anti-aliased line art is the case reduce.limit_region exists for: a line
that should be one color, or a color plus two shading steps, instead arrives as
a smear of in-between tones that also bleed into whatever they border. Finding
those pixels by hand with the lasso is miserable, so this does it and hands back
a selection, which then composes with everything else, including Shift-drag to
add and Ctrl-drag to subtract before you act on it.

A black-hat, not a darkness threshold. grey_closing fills anything narrower than
its structuring element, so closing minus the original is exactly what is both
dark and thin. Plain "darker than X" takes every shadow and every dark garment
with it; on a test image that was the difference between tracing the hair
strands and selecting a fifth of the picture.
"""

import numpy as np
from scipy import ndimage

from .colorspace import srgb_to_lab


def detect_lines(source_rgb, width=1, threshold=6.0, min_size=3,
                 within=None, light=False):
    """Pixels belonging to thin dark (or light) structures.

    width      the widest line to catch, in pixels
    threshold  how far the pixel dips below its surroundings, in L*
    min_size   drop connected runs smaller than this, which are speckle
    within     restrict to an existing selection
    light      find light lines on dark instead of the reverse
    """
    src = np.asarray(source_rgb)
    L = srgb_to_lab(src.astype(np.float64).reshape(-1, 3))[:, 0].reshape(src.shape[:2])
    size = 2 * max(1, int(width)) + 1
    if light:
        relief = L - ndimage.grey_opening(L, size=size)
    else:
        relief = ndimage.grey_closing(L, size=size) - L
    mask = relief > float(threshold)
    if within is not None:
        mask &= within
    if min_size > 1 and mask.any():
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
        if n:
            sizes = np.bincount(lab.ravel())
            keep = np.zeros(n + 1, dtype=bool)
            keep[1:] = sizes[1:] >= int(min_size)
            mask = keep[lab]
    return mask


def describe(mask, indexed=None):
    """One line about what was found, for the dialog."""
    n = int(mask.sum())
    if not n:
        return "Nothing found at these settings."
    pct = 100.0 * n / mask.size
    if indexed is None:
        return f"{n} px ({pct:.1f}% of the image)"
    used = np.unique(indexed.indices[mask])
    return (f"{n} px ({pct:.1f}% of the image)\n"
            f"using {int((used >= 0).sum())} of {len(indexed.slots)} colors")
