"""Palette reduction: fewer colors, chosen by what they cost.

The counterpart to refine.py. Refinement only ever adds, so a session that fixes
four areas can land well past any budget with no way back except undoing the
work. This gives back the half that palette grouping genuinely did by taking
colors from where they are not earning their place.

Every operation computes one result for one target. An earlier version built the
whole merge curve up front so a slider could scrub it; that is fine at 50 colors
and fatal at 3265, which is what an image processed with no color limit actually
produces. Greedy pairwise merging needs a K x K cost matrix rebuilt every step:
85 MB per step, 3263 steps, 35 billion operations. None of these do that.

    reduce_to       weighted k-means over the palette, so the split is global
                    rather than a chain of myopic pair merges
    merge_similar   a KD-tree radius query, which finds near-duplicates without
                    ever forming the full distance matrix
    limit_region    reduce_to confined to one area, for cleaning up a noisy line

All three return a plain victim -> survivor mapping, so applying one is the same
code path and nothing new is invented: every surviving color is a color the
image already had, and keeps its recipe.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .colors import hex_to_rgb
from .colorspace import srgb_to_lab
from .palette import _weighted_kmeans


@dataclass
class Reduction:
    """One computed reduction, ready to apply or throw away."""
    mapping: dict = field(default_factory=dict)   # victim_hex -> survivor_hex
    region: object = None          # bool mask, or None for the whole image
    before: int = 0
    after: int = 0
    mean_before: float = 0.0
    mean_after: float = 0.0
    moved: int = 0

    def __bool__(self):
        return bool(self.mapping)

    def summary(self):
        lines = [f"{self.before} -> {self.after} colors"]
        if self.mean_before or self.mean_after:
            lines.append(f"error: {self.mean_before:.2f} -> {self.mean_after:.2f}")
        lines.append(f"{self.moved} pixels repainted")
        return "\n".join(lines)


def _labs(indexed):
    if not indexed.slots:
        return np.zeros((0, 3), dtype=np.float64)
    return srgb_to_lab(np.array([hex_to_rgb(s.hex) for s in indexed.slots], dtype=np.float64))


def _resolve(mapping):
    """Collapse a -> b -> c chains so every victim points at a final survivor."""
    out = {}
    for victim in mapping:
        seen, cur = {victim}, mapping[victim]
        while cur in mapping and cur not in seen:
            seen.add(cur)
            cur = mapping[cur]
        out[victim] = cur
    return {v: s for v, s in out.items() if v != s}


def _score(indexed, labs, mapping, source_rgb, region=None):
    """Mean dE before and after, and how many pixels move. One pass, no loop."""
    idx = indexed.indices
    sel = np.ones(idx.shape, dtype=bool) if region is None else region
    live = sel & (idx >= 0)
    if not live.any():
        return 0.0, 0.0, 0
    by_hex = {s.hex: i for i, s in enumerate(indexed.slots)}
    target = np.arange(len(indexed.slots))
    for victim, survivor in mapping.items():
        a, b = by_hex.get(victim), by_hex.get(survivor)
        if a is not None and b is not None:
            target[a] = b
    old = idx[live]
    new = target[old]
    moved = int((old != new).sum())
    if source_rgb is None or source_rgb.shape[:2] != idx.shape:
        return 0.0, 0.0, moved
    px = srgb_to_lab(source_rgb[live].astype(np.float64))
    before = float(np.linalg.norm(px - labs[old], axis=1).mean())
    after = float(np.linalg.norm(px - labs[new], axis=1).mean())
    return before, after, moved


def reduce_to(indexed, n, source_rgb=None, protect=None):
    """Cut the palette to `n` colors.

    Weighted k-means over the palette itself rather than a chain of pair merges:
    one global split beats a sequence of locally-cheapest choices, and it runs in
    k x K time instead of K^3. Cluster representatives are snapped to a color the
    image already has, so nothing new is introduced and every survivor keeps its
    recipe.
    """
    out = Reduction(before=len(indexed.slots), after=len(indexed.slots))
    n = max(1, int(n))
    if len(indexed.slots) <= n:
        return out
    labs = _labs(indexed)
    counts = indexed.used_counts().astype(np.float64)
    hexes = [s.hex for s in indexed.slots]
    guard = set(protect or ())
    keep = np.array([h in guard for h in hexes])
    free = np.nonzero(~keep)[0]
    n_free = max(1, n - int(keep.sum()))
    if len(free) <= n_free:
        return out

    w = np.maximum(counts[free], 1e-6)
    centers = _weighted_kmeans(labs[free], w, n_free)
    lbl = cKDTree(centers).query(labs[free], k=1, workers=-1)[1]
    mapping = {}
    for c in range(len(centers)):
        members = free[lbl == c]
        if len(members) < 2:
            continue
        # The survivor is whichever member costs least to move everyone else
        # onto.
        w_m = np.maximum(counts[members], 1e-6)
        dd = np.linalg.norm(labs[members][:, None, :] - labs[members][None, :, :], axis=2)
        winner = members[int(np.argmin((dd * w_m[:, None]).sum(0)))]
        for m in members:
            if m != winner:
                mapping[hexes[m]] = hexes[winner]
    out.mapping = _resolve(mapping)
    out.after = out.before - len(out.mapping)
    out.mean_before, out.mean_after, out.moved = _score(
        indexed, labs, out.mapping, source_rgb)
    return out


def merge_similar(indexed, max_de, source_rgb=None, protect=None):
    """Fold together colors within `max_de` of each other.

    Leader clustering, not a transitive closure: the busiest colors become
    representatives and every other color joins one within `max_de` of itself.
    Union-find was the obvious choice and is wrong here because "similar" chains,
    so on a dense palette a - b - c - ... fuses the whole spectrum into one group.
    Measured on a 3265-color image, a dE 3 threshold that way collapsed it to 324
    colors and drove mean error from 2.6 to 19.6. This form guarantees no pixel
    moves further than `max_de`.
    """
    out = Reduction(before=len(indexed.slots), after=len(indexed.slots))
    if len(indexed.slots) < 2 or max_de <= 0:
        return out
    labs = _labs(indexed)
    counts = indexed.used_counts().astype(np.float64)
    hexes = [s.hex for s in indexed.slots]
    guard = set(protect or ())
    tree = cKDTree(labs)

    taken = np.zeros(len(labs), dtype=bool)
    mapping = {}
    # Busiest first, and protected colors ahead of everything so they lead their
    # own group rather than being absorbed into someone else's.
    order = sorted(range(len(labs)),
                   key=lambda i: (hexes[i] not in guard, -counts[i]))
    for lead in order:
        if taken[lead]:
            continue
        taken[lead] = True
        for other in tree.query_ball_point(labs[lead], float(max_de)):
            if other == lead or taken[other] or hexes[other] in guard:
                continue
            taken[other] = True
            mapping[hexes[other]] = hexes[lead]
    out.mapping = _resolve(mapping)
    out.after = out.before - len(out.mapping)
    out.mean_before, out.mean_after, out.moved = _score(
        indexed, labs, out.mapping, source_rgb)
    return out


def limit_region(indexed, region, n, source_rgb=None):
    """Cut the colors used inside `region` to `n`, touching nothing outside.
    
    For a noisy anti-aliased line: select it, ask for one color, or for a line
    color plus a couple of shading steps.
    """
    out = Reduction(before=len(indexed.slots), after=len(indexed.slots), region=region)
    n = max(1, int(n))
    used = np.unique(indexed.indices[region])
    used = used[used >= 0]
    if len(used) <= n:
        return out
    labs = _labs(indexed)
    hexes = [s.hex for s in indexed.slots]
    # Weight by use inside the region only: a color that is everywhere else in
    # the image but barely here should not win this vote.
    here = indexed.indices[region]
    counts = np.array([float((here == u).sum()) for u in used])
    centers = _weighted_kmeans(labs[used], np.maximum(counts, 1e-6), n)
    lbl = cKDTree(centers).query(labs[used], k=1, workers=-1)[1]
    mapping = {}
    for c in range(len(centers)):
        members = used[lbl == c]
        if len(members) < 2:
            continue
        local = np.maximum(counts[np.searchsorted(used, members)], 1e-6)
        dd = np.linalg.norm(labs[members][:, None, :] - labs[members][None, :, :], axis=2)
        winner = members[int(np.argmin((dd * local[:, None]).sum(0)))]
        for m in members:
            if m != winner:
                mapping[hexes[m]] = hexes[winner]
    out.mapping = _resolve(mapping)
    out.mean_before, out.mean_after, out.moved = _score(
        indexed, labs, out.mapping, source_rgb, region)
    # The palette only shrinks if those colors appear nowhere else.
    survivors = set(hexes) - set(out.mapping)
    outside = np.unique(indexed.indices[~region])
    still = {hexes[i] for i in outside if i >= 0}
    out.after = len(survivors | still)
    return out


def apply_reduction(indexed, reduction):
    """Apply a mapping. Returns a bool mask of the pixels that changed.

    Whole-image reductions merge slots so the palette actually shrinks. Region
    reductions only repoint pixels inside the region, then compact away anything
    that is left unused.
    """
    changed = np.zeros(indexed.shape, dtype=bool)
    if not reduction.mapping:
        return changed
    if reduction.region is None:
        for victim, survivor in reduction.mapping.items():
            where = {}
            for i, slot in enumerate(indexed.slots):
                where.setdefault(slot.hex, []).append(i)
            vs, ss = where.get(victim, []), where.get(survivor, [])
            if not vs or not ss:
                continue
            a, b = (vs[-1], vs[0]) if victim == survivor else (vs[0], ss[0])
            if a == b:
                continue
            changed |= indexed.indices == a
            indexed.merge_slot(a, b)
        return changed
    by_hex = {s.hex: i for i, s in enumerate(indexed.slots)}
    region = reduction.region
    for victim, survivor in reduction.mapping.items():
        a, b = by_hex.get(victim), by_hex.get(survivor)
        if a is None or b is None or a == b:
            continue
        hit = region & (indexed.indices == a)
        if hit.any():
            indexed.indices[hit] = b
            changed |= hit
    indexed.compact()
    return changed
