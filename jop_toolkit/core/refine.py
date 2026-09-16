"""Regional palette refinement: give one part of the image more colors.

The problem this solves: palette selection is weighted by pixel mass, so a small
region whose colors differ from the bulk of the image gets almost nothing.

The fix is additive rather than redistributive. Nothing competes for a budget:
the colors a region needs are computed from the pre-quantization source and
appended to the palette, and only that region's pixels are re-matched. Pixels
outside it cannot move, so fixing one area can never break another.

Planning is separated from applying because greedy selection is incremental: one
plan() call yields the colors in the order they help most, plus the error after
each one, so a UI can scrub through counts without recomputing.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import dither as dither_mod
from .colors import hex_to_rgb
from .colorspace import srgb_to_lab

MAX_ADD = 32
_SEARCH_PX = 1200      # pixels sampled for the search itself
_CURVE_PX = 8000       # pixels used for the reported error curve
_RADIUS = 14.0         # Lab ball around the region's colors to draw candidates from
_CAND_CAP = 4000


@dataclass
class RefinePlan:
    """An ordered list of colors to add, with the error each one buys."""
    colors: list = field(default_factory=list)      # hex, best first
    recipes: list = field(default_factory=list)     # parallel to colors
    mean: list = field(default_factory=list)        # mean dE after k adds, len == n+1
    p95: list = field(default_factory=list)
    region_px: int = 0

    def __len__(self):
        return len(self.colors)

    def at(self, k):
        """(mean, p95) after adding the first k colors."""
        k = max(0, min(int(k), len(self.mean) - 1))
        return self.mean[k], self.p95[k]


def _pairwise(a, b):
    """Euclidean distances between every row of `a` and every row of `b`."""
    d2 = ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :]
          - 2.0 * (a @ b.T))
    return np.sqrt(np.maximum(d2, 0.0, out=d2), out=d2).astype(np.float32)


def _slot_labs(indexed):
    if not indexed.slots:
        return np.zeros((0, 3), dtype=np.float64)
    return srgb_to_lab(np.array([hex_to_rgb(s.hex) for s in indexed.slots], dtype=np.float64))


def color_range_mask(indexed, region):
    """Every pixel that uses a color the region uses.
    
    The 'and everything else that looks like this' scope. Measurably weaker than
    refining the selection alone since the added colors get spread over every other
    pixel sharing these slots, so it is not the default.
    """
    used = np.unique(indexed.indices[region])
    return np.isin(indexed.indices, used[used >= 0])


def plan(source_rgb, indexed, region, palette, recipes, max_add=MAX_ADD, rng=None):
    """Work out which colors would most help `region`, best first.

    `source_rgb` is the pre-quantization image at the index grid's resolution
    state.match_source, which is what Matcher itself consumed. Not
    state.preprocessed_source: that one is supersampled for the swipe view and
    does not line up with the grid. Using the already-quantized result instead
    would be circular, so the gradient that needs recovering is exactly what
    quantization destroyed.
    """
    rng = rng or np.random.default_rng(0)
    ys, xs = np.nonzero(region)
    empty = RefinePlan(mean=[0.0], p95=[0.0])
    if len(ys) == 0 or not palette or source_rgb is None:
        return empty
    if source_rgb.shape[:2] != indexed.shape:
        return empty

    px_all = srgb_to_lab(source_rgb[ys, xs].astype(np.float64))
    curve = (px_all if len(px_all) <= _CURVE_PX
             else px_all[rng.choice(len(px_all), _CURVE_PX, replace=False)])
    search = (px_all if len(px_all) <= _SEARCH_PX
              else px_all[rng.choice(len(px_all), _SEARCH_PX, replace=False)])

    have_lab = _slot_labs(indexed)
    if len(have_lab) == 0:
        return empty
    have_tree = cKDTree(have_lab)
    d_search, _ = have_tree.query(search, k=1)
    d_curve, _ = have_tree.query(curve, k=1)

    cand_hex = list(palette.keys())
    cand_lab = srgb_to_lab(np.array(list(palette.values()), dtype=np.float64))
    # Only colors near what this region actually contains are worth scoring.
    # Asked as "how far is each candidate from this region" rather than
    # query_ball_point per pixel: the ball form returns a Python list per probe,
    # which on a 70k palette costs hundreds of megabytes before anything is
    # scored. This form allocates one float array.
    near_d, _ = cKDTree(search).query(cand_lab, k=1)
    idx = np.nonzero(near_d < _RADIUS)[0]
    if len(idx) == 0:
        return empty
    if len(idx) > _CAND_CAP:
        # Keep the candidates nearest the region's own colors. Drawing at random
        # measurably costs quality: the useful colors are a small fraction of a
        # 70k palette, and a random half of them is a random half of the answer.
        idx = idx[np.argsort(near_d[idx])[:_CAND_CAP]]

    cl = cand_lab[idx]
    # (candidates x search px), needed in full because every step rescores every
    # candidate. Built from the dot-product identity rather than broadcasting:
    # `cl[:, None, :] - search[None, :, :]` would allocate a candidates x px x 3
    # float64 intermediate, which is hundreds of megabytes at these sizes.
    D = _pairwise(cl, search)
    have = {s.hex for s in indexed.slots}

    out = RefinePlan(region_px=int(len(ys)),
                     mean=[float(d_curve.mean())],
                     p95=[float(np.percentile(d_curve, 95))])
    alive = np.ones(len(idx), dtype=bool)
    for _ in range(int(max_add)):
        # Error reduction, weighted by how badly each pixel is currently served:
        # this favors closing the worst gaps over shaving already-good areas.
        gain = (np.clip(d_search.astype(np.float32) - D, 0, None)
                * d_search.astype(np.float32)).sum(1)
        gain[~alive] = -1.0
        k = int(gain.argmax())
        if gain[k] <= 0:
            break
        hx = cand_hex[idx[k]]
        alive[k] = False
        if hx in have:
            continue
        have.add(hx)
        d_search = np.minimum(d_search, D[k])
        # One row at a time: the full curve matrix would be candidates x pixels,
        # which is hundreds of megabytes once the region is large.
        d_curve = np.minimum(d_curve, np.linalg.norm(curve - cl[k], axis=1))
        out.colors.append(hx)
        out.recipes.append(list(recipes.get(hx, [])))
        out.mean.append(float(d_curve.mean()))
        out.p95.append(float(np.percentile(d_curve, 95)))
    return out


def apply_plan(indexed, source_rgb, region, plan_obj, k, dither_opts=None):
    """Add the first k planned colors, then re-match the region against the
    enlarged palette. Returns a bool mask of the pixels that changed.

    Only `region` is written, so this cannot disturb the rest of the image,
    which is the whole point, and is what the tests assert.
    """
    k = max(0, min(int(k), len(plan_obj)))
    have = {s.hex for s in indexed.slots}
    for hx, rcp in zip(plan_obj.colors[:k], plan_obj.recipes[:k]):
        if hx not in have:
            indexed.add_slot(hx, rcp)
            have.add(hx)
    return rematch_region(indexed, source_rgb, region, dither_opts)


def rematch_region(indexed, source_rgb, region, dither_opts=None):
    """Re-match `region` against the current palette. Returns a changed mask.

    `dither_opts` mirrors the Source stage's settings (strength, method,
    serpentine, matrix_size, flat). Error diffusion runs over the region's
    bounding box rather than the whole image, so its cost is bounded by the
    selection; pixels in the box but outside the region are carried at zero
    strength and never written back.
    """
    changed = np.zeros(indexed.shape, dtype=bool)
    ys, xs = np.nonzero(region)
    if len(ys) == 0 or not indexed.slots or source_rgb is None:
        return changed
    if source_rgb.shape[:2] != indexed.shape:
        return changed

    if dither_opts and dither_opts.get('strength', 0) > 0 \
            and dither_opts.get('method', 'none') != 'none':
        best = _dither_region(indexed, source_rgb, region, ys, xs, dither_opts)
    else:
        px = srgb_to_lab(source_rgb[ys, xs].astype(np.float64))
        _, best = cKDTree(_slot_labs(indexed)).query(px, k=1)
    best = np.asarray(best).astype(indexed.indices.dtype)

    before = indexed.indices[ys, xs]
    # An erased pixel is a deliberate "no paint here"; refining the area around
    # it must not quietly fill it back in.
    best = np.where(before < 0, before, best)
    moved = before != best
    indexed.indices[ys, xs] = best
    changed[ys[moved], xs[moved]] = True
    return changed


def _dither_region(indexed, source_rgb, region, ys, xs, opts):
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = source_rgb[y0:y1, x0:x1]
    sub = region[y0:y1, x0:x1]
    pal_rgb = np.array([hex_to_rgb(s.hex) for s in indexed.slots], dtype=np.float64)
    mask = sub.astype(np.float64)
    flat = opts.get('flat')
    if flat is not None:
        mask = mask * dither_mod.ramp_mask(crop, flat)
    idx = dither_mod.dither(
        crop, pal_rgb,
        strength=float(opts.get('strength', 0.85)),
        mask=mask,
        method=opts.get('method', 'floyd-steinberg'),
        serpentine=bool(opts.get('serpentine', True)),
        matrix_size=int(opts.get('matrix_size', 8)))
    return idx[ys - y0, xs - x0]
