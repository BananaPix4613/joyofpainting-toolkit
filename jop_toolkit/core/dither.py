"""Dithering: error-diffusion kernels and ordered/mask methods.

Error diffusion pushes each pixel's quantization error into its neighbors,
which dissolves banding in smooth gradients, but on a grainy source it
propagates the grain instead. Ordered and mask methods never propagate, so they
cannot amplify noise. That trade is why this is a choice and not a constant.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .colorspace import srgb_to_lab

# name -> (divisor, [(dx, dy, weight), ...])
DIFFUSION_KERNELS = {
    'floyd-steinberg': (16, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]),
    # Atkinson distributes only 6/8 of the error and discards the rest, which
    # keeps contrast and stops grain compounding. Good on noisy, high-contrast art.
    'atkinson': (8, [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]),
    'jarvis': (48, [(1, 0, 7), (2, 0, 5),
                    (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
                    (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)]),
    'stucki': (42, [(1, 0, 8), (2, 0, 4),
                    (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
                    (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)]),
    'sierra-lite': (4, [(1, 0, 2), (-1, 1, 1), (0, 1, 1)]),
}
ORDERED_METHODS = ('ordered', 'blue-noise')
METHODS = ('none',) + tuple(DIFFUSION_KERNELS) + ORDERED_METHODS

# Rough seconds per pixel, measured at 112x150 with 28 colors. Used only to
# weight the progress bar. Ordered methods are ~10x cheaper than diffusion.
METHOD_COST = {
    'none': 1e-6, 'ordered': 2e-6, 'blue-noise': 4e-6,
    'floyd-steinberg': 2.0e-5, 'atkinson': 2.4e-5,
    'sierra-lite': 5.7e-5, 'jarvis': 5.5e-5, 'stucki': 7.8e-5,
}


def _to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _from_linear(v):
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return np.where(v > 0.0031308, 1.055 * v ** (1 / 2.4) - 0.055, 12.92 * v) * 255.0


def ramp_mask(rgb, flat_threshold=1.25, edge_percentiles=(70.0, 92.0)):
    """Where dithering helps: 0 on constant areas, 1 on gentle ramps, 0 on edges.

    This replaces detail_mask, which was a low-pass and had exactly the wrong
    polarity. It protected edges but left large flat regions at full strength.
    A flat region whose color sits between two palette entries gets d1~=d2, so
    the ordered dither runs at ~50% across the whole area, which reads as a
    checkerboard over what should be clean paint. Meanwhile the banding in a
    gradient was being suppressed, because a ramp has real local gradient.

    The edge cutoffs are percentiles of this image's own gradient, because what
    counts as "an edge" is image-dependent: median |grad L*| is 0.42 on a soft
    portrait background and 4.93 on a dense illustration.
    """
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64))
    gy, gx = np.gradient(lab[..., 0])
    g = np.hypot(gx, gy)
    lo, hi = np.percentile(g, edge_percentiles[0]), np.percentile(g, edge_percentiles[1])
    hi = max(hi, lo + 1e-6)
    rise = np.clip(g / max(float(flat_threshold), 1e-6), 0.0, 1.0)
    fall = np.clip(1.0 - (g - lo) / (hi - lo), 0.0, 1.0)
    return rise * fall


def bayer_matrix(size=8):
    """Normalized Bayer threshold matrix in [-0.5, 0.5]."""
    m = np.array([[0, 2], [3, 1]], dtype=np.float64)
    while m.shape[0] < size:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m / m.size - 0.5


def blue_noise_matrix(size=64, seed=0):
    """Void-and-cluster-ish blue noise: high-pass then re-rank, repeatedly.
    
    Unlike Bayer it has no visible crosshatch, so it reads as texture not grid.
    """
    rng = np.random.default_rng(seed)
    m = rng.random((size, size))
    for _ in range(24):
        hp = m - gaussian_filter(m, 1.6, mode='wrap')
        r = np.argsort(np.argsort(hp, axis=None)).reshape(size, size)
        m = (r + 0.5) / (size * size)
    return m - 0.5


def _nearest_lut(pal_lab, tree, bits=6):
    """Nearest palette index for every cell of a 2^bits per-channel sRGB grid.
    
    Built vectorized in one shot (~0.08 s at 64^3), after which the sequential
    diffusion loop resolves a color with a single array index.
    """
    n = 1 << bits
    axis = (np.arange(n) + 0.5) * (256.0 / n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    _, idx = tree.query(srgb_to_lab(grid), k=1, workers=-1)
    return idx.reshape(n, n, n).astype(np.int32), 8 - bits


def _ordered(rgb, pal_lab, tree, method, strength, mask, matrix_size, seed=0):
    """Dither between the two nearest palette colors, but only where they
    actually bracket the target.

    Dithering can only represent a color that lies *between* two available ones.
    Where the palette is sparse the second-nearest color is nowhere near: on a
    real 32-color palette, a smooth run through the pale region had d1 ~= 2.3 and
    d2 ~= 13.4, with the two nearest bracketing the target only 6% of the time.
    A naive d1/(d1+d2) still fires there ~15% of the time, flipping isolated
    pixels to a color 13 dE away.

    Gating on projection onto the segment between the two colors makes the
    dither decline where it cannot help. If a region still bands afterwards, the
    palette is too sparse there and the fix is to allocate it more colors, not
    more dithering.
    """
    h, w = np.asarray(rgb).shape[:2]
    if len(pal_lab) < 2:
        return np.zeros((h, w), dtype=np.int32)
    m = bayer_matrix(matrix_size) if method == 'ordered' else blue_noise_matrix(64, seed)
    tile = np.tile(m, (h // m.shape[0] + 1, w // m.shape[1] + 1))[:h, :w] + 0.5   # -> [0,1)
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64)).reshape(-1, 3)
    d, idx = tree.query(lab, k=2, workers=-1)
    c1 = pal_lab[idx[:, 0]]
    seg = pal_lab[idx[:, 1]] - c1
    den = np.einsum('ij,ij->i', seg, seg)
    u = np.where(den > 1e-9, np.einsum('ij,ij->i', lab - c1, seg) / np.maximum(den, 1e-9), 0.0)
    perp = np.linalg.norm((lab - c1) - u[:, None] * seg, axis=1)
    between = (u > 0.0) & (u < 1.0) & (perp < 0.35 * np.sqrt(np.maximum(den, 1e-9)))
    w2 = np.where(between, np.clip(u, 0.0, 1.0) * float(strength), 0.0)
    if mask is not None:
        w2 = w2 * np.asarray(mask).reshape(-1)
    return np.where(tile.reshape(-1) < w2, idx[:, 1], idx[:, 0]).reshape(h, w).astype(np.int32)


def dither(rgb, pal_rgb, strength=0.85, mask=None, method='floyd-steinberg',
           serpentine=True, matrix_size=8, progress_cb=None):
    """rgb: (H, W, 3). pal_rgb: (K, 3) allowed colors.

    method: 'none', a DIFFUSION_KERNELS name, 'ordered', or 'blue-noise'.
    mask: optional (H, W) in [0,1] scaling the effect per pixel (see detail_mask).
    progress_cb(done_rows, total_rows) is called every 8 rows; it may raise to abort.
    
    Returns (H, W) int indices into pal_rgb.
    """
    from scipy.spatial import cKDTree
    pal_rgb = np.asarray(pal_rgb, dtype=np.float64)
    pal_lab = srgb_to_lab(pal_rgb)
    tree = cKDTree(pal_lab)
    h, w = np.asarray(rgb).shape[:2]

    if method == 'none' or strength <= 0:
        _, idx = tree.query(srgb_to_lab(np.asarray(rgb, dtype=np.float64)).reshape(-1, 3),
                            k=1, workers=-1)
        return idx.reshape(h, w).astype(np.int32)

    if method in ORDERED_METHODS:
        if progress_cb is not None:
            progress_cb(0, h)
        out = _ordered(rgb, pal_lab, tree, method, strength, mask, matrix_size)
        if progress_cb is not None:
            progress_cb(h, h)
        return out

    divisor, taps = DIFFUSION_KERNELS[method]
    # Normalize the tap weights once, and resolve nearest-color through a coarse
    # sRGB lookup table instead of a per-pixel KD-tree query plus Lab conversion.
    # The loop is inherently sequential, so these constant-factor cuts are the
    # only ones available. Together they took Floyd-Steinberg from 3.08s to 0.29s.
    ntaps = tuple((dx, dy, wt / divisor) for dx, dy, wt in taps)
    lut, shift = _nearest_lut(pal_lab, tree)
    pal_lin = _to_linear(pal_rgb)
    work = _to_linear(np.asarray(rgb, dtype=np.float64))
    out = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        if progress_cb is not None and y % 8 == 0:
            progress_cb(y, h)
        forward = (not serpentine) or (y % 2 == 0)
        xs = range(w) if forward else range(w - 1, -1, -1)
        step = 1 if forward else -1
        mrow = None if mask is None else mask[y]   # hoisted: mask[y, x] per pixel was ~3x
        wrow = work[y]
        for x in xs:
            old = wrow[x]
            srgb = _from_linear(old)
            k = int(lut[int(srgb[0]) >> shift, int(srgb[1]) >> shift, int(srgb[2]) >> shift])
            out[y, x] = k
            s = strength if mrow is None else strength * mrow[x]
            if s <= 0.0:
                continue
            err = (old - pal_lin[k]) * s
            for dx, dy, wt in ntaps:
                nx, ny = x + dx * step, y + dy
                if 0 <= nx < w and ny < h:
                    work[ny, nx] += err * wt
    return out
