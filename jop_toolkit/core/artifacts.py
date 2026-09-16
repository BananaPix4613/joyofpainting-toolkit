"""Source-image cleanup: grain, JPEG blocking, and near-duplicate colors.

These run at *source* resolution, before the downscale, because that is where
the artifacts live. Denoising a 112px result cannot undo grain that has
already been averaged into it.

None of this scores well by dE against the original: the original *contains*
the grain being removed, so a metric rewarding fidelity to it punishes every
setting here. These are deliberately user controls, not auto-tuned defaults.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, median_filter

RESAMPLE_FILTERS = {
    'box': Image.BOX,            # straight area average, no ringing
    'bilinear': Image.BILINEAR,
    'lanczos': Image.LANCZOS,    # sharpest, but overshoots and invents edge colors
    'nearest': Image.NEAREST,    # keeps every line, at the cost of jaggies
}
DEFAULT_RESAMPLE = 'box'


def mode_downscale(img, size):
    """Each output pixel takes the most common source color in its footprint.
    
    Averaging manufactures a blend at every anti-aliased line edge, and each
    blend becomes a palette entry to mix and hand-clean. On flat-color art at
    96x96 this took 526 unique colors down to 8, and a region that is one flat
    color in the source from 11 distinct colors to 2, while keeping *more*
    linework than BOX (68% vs 59%).

    Only worth it when the source actually has flat regions; on a painterly
    source it just picks an arbitrary winner instead of the average.
    """
    a = np.asarray(img.convert('RGB'), dtype=np.int64)
    H, W = a.shape[:2]
    ow, oh = int(size[0]), int(size[1])
    ys = (np.arange(oh + 1) * H) // oh
    xs = (np.arange(ow + 1) * W) // ow
    packed = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    out = np.zeros((oh, ow), dtype=np.int64)
    for j in range(oh):
        row = packed[ys[j]:ys[j + 1]]
        for i in range(ow):
            v, c = np.unique(row[:, xs[i]:xs[i + 1]].ravel(), return_counts=True)
            out[j, i] = v[int(c.argmax())]
    rgb = np.stack([(out >> 16) & 255, (out >> 8) & 255, out & 255], -1).astype(np.uint8)
    return Image.fromarray(rgb)


def downscale_to(img, size, method='box'):
    """Resize by name, including the methods PIL has no filter for."""
    if tuple(img.size) == tuple(size):
        return img
    if method == 'mode':
        return mode_downscale(img, size)
    return img.resize(tuple(size), RESAMPLE_FILTERS.get(method, Image.BOX))


def guided_denoise(rgb, radius=2, strength=0.06):
    """Edge-preserving denoise (guided filter, each channel its own guide).
    
    O(1) per pixel via box means, so it stays fast at source resolution and adds
    no dependency. Measured on a grainy 736x981 source: removes ~61% of flat-area
    grain while keeping ~88% of edge contrast, in 0.22 s. `strength` is the
    regularization term so larger amounts smooth more; detail below roughly that
    contrast is treated as noise.
    """
    I = np.asarray(rgb, dtype=np.float64) / 255.0
    size = (2 * int(radius) + 1, 2 * int(radius) + 1)
    eps = float(strength) ** 2
    out = np.empty_like(I)
    for c in range(I.shape[2]):
        Ic = I[..., c]
        mean = uniform_filter(Ic, size)
        var = np.maximum(uniform_filter(Ic * Ic, size) - mean * mean, 0.0)
        a = var / (var + eps)
        out[..., c] = uniform_filter(a, size) * Ic + uniform_filter(mean - a * mean, size)
    return np.clip(out, 0.0, 1.0) * 255.0


def median_denoise(rgb, radius=1):
    """Speckle / salt-and-pepper removal. Blunter and slower than guided_denoise,
    but it erases isolated outlier pixels that box means only average."""
    if radius <= 0:
        return np.asarray(rgb, dtype=np.float64)
    r = int(radius)
    return median_filter(np.asarray(rgb), size=(2 * r + 1, 2 * r + 1, 1)).astype(np.float64)


def deblock(rgb, strength=0.5, block=8):
    """Pull JPEG block seams toward their local average, leaving interiors alone.

    Only the pixel pairs straddling each 8x8 boundary are touched, so this does
    not soften real detail the way a global blur would. Must run before any
    resize, while the block grid is still aligned.
    """
    a = np.asarray(rgb, dtype=np.float64)
    if strength <= 0:
        return a
    out = a.copy()
    h, w = a.shape[:2]
    s = float(strength)
    for x in range(block, w, block):
        left, right = a[:, x - 1], a[:, x]
        avg = (left + right) / 2.0
        out[:, x - 1] = left + (avg - left) * s
        out[:, x] = right + (avg - right) * s
    for y in range(block, h, block):
        top, bot = a[y - 1], a[y]
        avg = (top + bot) / 2.0
        out[y - 1] = top + (avg - top) * s
        out[y] = bot + (avg - bot) * s
    return out


def harden_edges(rgb, strength=0.0, threshold=12.0, radius=1):
    """Contiuous control between soft anti-aliased edges and hard aliased ones.

    Toggle mapping (Kramer-Bruckner): where a pixel sits on an edge, move it
    toward whichever side of that edge it is already nearer to. The target is
    always a color some neighbor actually has.

    That only removes tones at full strength on a clear edge, where a pixel lands
    exactly on its neighbor. Below full strength it moves part-way, which is a
    new blend. Measured, 97-99% of changed pixels at 0.25-0.5 match no
    neighbor. At 1.0 the fully-gated zone invents nothing; the remainder all sit
    in the ramp between gradient and edge. So low settings soften the transition
    to hard edges, and only the top of the range actually cuts palette colors.

    Why not just offer nearest or mode for "hard": they are hard everywhere, and
    they alias. On a test portrait at 96x144 they left 3211 and 2169 blended
    edge pixels respectively, because the aliasing itself scatters new mid-tone
    speckle; this pass at full strength left 1478 and kept every gradient that
    was not actually an edge.

    strength   0 leaves the image alone, 1 snaps edge pixels fully to a side
    threshold  L* contrast below which a neighborhood counts as a gradient and
               is left alone; the effect ramps in over one threshold's width so
               a slider does not pop.
    """
    if strength <= 0:
        return np.asarray(rgb)
    from numpy.lib.stride_tricks import sliding_window_view
    from .colorspace import srgb_to_lab
    src = np.asarray(rgb, dtype=np.float64)
    H, W, _ = src.shape
    k = 2 * int(radius) + 1
    lab = srgb_to_lab(src.reshape(-1, 3)).reshape(H, W, 3)
    pad = ((radius, radius), (radius, radius), (0, 0))
    win_rgb = sliding_window_view(np.pad(src, pad, mode='edge'), (k, k),
                                  axis=(0, 1)).reshape(H, W, 3, k * k)
    win_lab = sliding_window_view(np.pad(lab, pad, mode='edge'), (k, k),
                                  axis=(0, 1)).reshape(H, W, 3, k * k)
    L = win_lab[:, :, 0, :]
    lo, hi = L.argmin(-1), L.argmax(-1)
    ii, jj = np.indices((H, W))
    lo_lab, hi_lab = win_lab[ii, jj, :, lo], win_lab[ii, jj, :, hi]
    lo_rgb, hi_rgb = win_rgb[ii, jj, :, lo], win_rgb[ii, jj, :, hi]
    contrast = hi_lab[..., 0] - lo_lab[..., 0]
    nearer_lo = (np.linalg.norm(lab - lo_lab, axis=-1)
                 <= np.linalg.norm(lab - hi_lab, axis=-1))
    target = np.where(nearer_lo[..., None], lo_rgb, hi_rgb)
    t = max(float(threshold), 1e-6)
    gate = np.clip((contrast - t) / t, 0.0, 1.0)[..., None]
    out = src + min(float(strength), 1.0) * gate * (target - src)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def snap_colors(rgb, step=0):
    """Quantize to a coarse RGB lattice so grain stops fragmenting the histogram.
    
    Grain turns one intent-color into hundreds of near-identical ones, skewing
    how the palette selector weights regions. Snapping at step 4 cut a grainy
    image from 14,056 unique colors to 3,992 with no measurable dE cost.
    """
    if step <= 1:
        return np.asarray(rgb)
    a = np.asarray(rgb, dtype=np.int32)
    return np.clip(np.round(a / step) * step, 0, 255).astype(np.uint8)


def preprocess_full(img, *, deblock_strength=0.0, denoise_mode='none',
                    denoise_radius=2, denoise_strength=0.06, median_radius=1):
    """Run the source-resolution cleanup stages. img: PIL.Image -> PIL.Image.
    
    Order matters: deblock first (needs the untouched 8x8 grid), then denoise.
    """
    arr = np.asarray(img.convert('RGB'), dtype=np.float64)
    if deblock_strength > 0:
        arr = deblock(arr, deblock_strength)
    if denoise_mode == 'guided':
        arr = guided_denoise(arr, denoise_radius, denoise_strength)
    elif denoise_mode == 'median':
        arr = median_denoise(arr, median_radius)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
