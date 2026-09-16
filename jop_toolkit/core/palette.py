"""Generate the dye-mix palette and run vectorized Lab-space matching."""

import itertools
from collections.abc import Mapping

import numpy as np

from .colors import BASE_COLORS, hex_to_rgb, rgb_to_hex
from .colorspace import srgb_to_lab

_HEX2 = np.array(['%02x' % i for i in range(256)], dtype='U2')


def pack_rgb(a):
    a = np.asarray(a, dtype=np.int64)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def unpack_rgb(p):
    p = np.asarray(p, dtype=np.int64)
    return np.stack([(p >> 16) & 255, (p >> 8) & 255, p & 255], axis=-1)


def hex_array(rgb):
    """Vectorized (..., 3) int -> (...) object array of '#rrggbb'."""
    rgb = np.asarray(rgb, dtype=np.int64)
    return np.char.add('#', np.char.add(np.char.add(
        _HEX2[rgb[..., 0]], _HEX2[rgb[..., 1]]), _HEX2[rgb[..., 2]])).astype(object)


class _LazyRecipes(Mapping):
    """hex -> [dye, ...]. Rows are stored packed and built on first access, so a
    500k-color palette costs no Python lists until something actually asks."""

    def __init__(self, hexes, rows, base_names):
        self._idx = {h: i for i, h in enumerate(hexes)}
        self._rows = rows
        self._names = base_names
        self._cache = {}

    def __getitem__(self, h):
        v = self._cache.get(h)
        if v is None:
            v = [self._names[k] for k in self._rows[self._idx[h]] if k >= 0]
            self._cache[h] = v
        return v

    def __iter__(self):
        return iter(self._idx)

    def __len__(self):
        return len(self._idx)


_CHUNK = 1_000_000


def generate_palette(selected_colors=None, max_depth=4, progress_cb=None):
    """Return ({hex: rgb}, {hex: [dye, ...]}) for every reachable mix.
    
    Mixes use the mod's exact arithmetic: floor(sum(channel) / n). Combinations
    are consumed in chunks and summed column-wise, so peak memory stays flat
    (~250 MB at depth 12) instead of materializing an (n, depth, 3) expansion.
    
    progress_cb(done_units, total_units, label) is called once per chunk, where
    units are multisets processed. Cost per depth grows ~4x per level, so unit
    counts, not depth counts, are what a progress bar should track. The callback
    may raise to abort.
    """
    from math import comb
    selected = selected_colors or list(BASE_COLORS.keys())
    base_names = [n for n in selected if n in BASE_COLORS]
    if not base_names:
        raise ValueError("No valid base colors selected")
    nb = len(base_names)
    base_rgb = np.array([hex_to_rgb(BASE_COLORS[n]) for n in base_names], dtype=np.int64)

    total_units = sum(comb(nb - 1 + d, d) for d in range(1, max_depth + 1))
    done_units = 0

    seen = np.zeros(1 << 24, dtype=bool)   # 16 MB membership bitmap
    packs, rows = [], []
    for depth in range(1, max_depth + 1):
        it = itertools.combinations_with_replacement(range(nb), depth)
        while True:
            buf = list(itertools.islice(it, _CHUNK))
            if not buf:
                break
            combos = np.array(buf, dtype=np.int8)
            n_combos = len(combos)
            del buf
            mixed = np.zeros((len(combos), 3), dtype=np.int64)
            for j in range(depth):          # column-wise: O(chunk*3), not O(chunk*depth*3)
                mixed += base_rgb[combos[:, j]]
            mixed //= depth                 # mod-exact floor
            p = (mixed[:, 0] << 16) | (mixed[:, 1] << 8) | mixed[:, 2]
            del mixed
            uq, first = np.unique(p, return_index=True)
            del p
            fresh = ~seen[uq]               # keep the shallowest recipe
            uq, first = uq[fresh], first[fresh]
            seen[uq] = True
            padded = np.full((len(first), max_depth), -1, dtype=np.int8)
            padded[:, :depth] = combos[first]
            packs.append(uq)
            rows.append(padded)
            del combos
            done_units = min(total_units, done_units + n_combos)
            if progress_cb:
                progress_cb(done_units, total_units, f"Depth {depth} of {max_depth}...")

    all_packs = np.concatenate(packs)
    all_rows = np.concatenate(rows)
    rgbs = unpack_rgb(all_packs)
    hexes = hex_array(rgbs)
    palette = dict(zip(hexes.tolist(), map(tuple, rgbs.tolist())))
    return palette, _LazyRecipes(hexes.tolist(), all_rows, base_names)


def _weighted_kmeans(X, w, k, iters=24, seed=0):
    """Lloyd's algorithm in Lab, weighted by pixel count. k-means++ seeding."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=np.float64)
    k = min(k, len(X))
    first = int(rng.choice(len(X), p=w / w.sum()))
    idx = [first]
    d2 = ((X - X[first]) ** 2).sum(1)
    for _ in range(k - 1):
        p = d2 * w
        s = p.sum()
        idx.append(int(rng.choice(len(X), p=p / s)) if s > 0 else int(rng.integers(len(X))))
        d2 = np.minimum(d2, ((X - X[idx[-1]]) ** 2).sum(1))
    C = X[idx].copy()
    for _ in range(iters):
        lbl = cKDTree(C).query(X, k=1, workers=-1)[1]
        moved = False
        for j in range(k):
            m = lbl == j
            if m.any():
                nc = (X[m] * w[m, None]).sum(0) / w[m].sum()
                if not np.allclose(nc, C[j]):
                    moved = True
                C[j] = nc
        if not moved:
            break
    return C


def select_palette(src_rgb, palette, max_colors, smooth_boost=0.0, seed=0):
    """Pick <= max_colors reachable mixes that minimize pixel-weighted Lab error.
    
    The old farthest-point sampling maximized spectral *coverage*, which on a
    mostly-neutral image spends ~78% of its picks on the few saturated pixels and
    starves the gray ramp that most of the canvas actually needs. Weighted
    k-means allocates by where the pixels are, then snaps each centroid to the
    nearest mix the mod can actually produce.

    Returns a list of hex keys (may be shorter than max_colors when distinct
    centroids snap to the same reachable mix).
    """
    from scipy.spatial import cKDTree
    src = np.asarray(src_rgb)
    flat = src.reshape(-1, 3).astype(np.int64)
    uq, cnt = np.unique(flat, axis=0, return_counts=True)
    keys = list(palette.keys())
    ptree = cKDTree(srgb_to_lab(np.array(list(palette.values()), dtype=np.float64)))
    if len(uq) <= max_colors:
        _, nn = ptree.query(srgb_to_lab(uq.astype(np.float64)), k=1, workers=-1)
        return [keys[i] for i in sorted(set(int(v) for v in nn))]
    w = cnt.astype(np.float64)
    if smooth_boost > 0:
        lab = srgb_to_lab(src.astype(np.float64))
        gy, gx = np.gradient(lab[..., 0])
        sm = (np.hypot(gx, gy) < 1.0).reshape(-1).astype(np.float64)
        _, inv = np.unique(flat, axis=0, return_inverse=True)
        frac = np.bincount(inv, weights=sm, minlength=len(uq)) / np.maximum(cnt, 1)
    cen = _weighted_kmeans(srgb_to_lab(uq.astype(np.float64)), w, max_colors, seed=seed)
    _, nn = ptree.query(cen, k=1, workers=-1)
    return [keys[i] for i in sorted(set(int(v) for v in nn))]


def limit_used_palette(matched_grid, palette, max_colors):
    """Reduce the matched image to `max_colors` colors that span the widest spectrum.

    Selection is farthest-point sampling in Lab space, seeded with the most-used
    color so the dominant tone is preserved. Each subsequent pick is the color
    whose minimum Lab distance to the already-picked set is greatest — this keeps
    rare outlier colors (the few oranges in a sea of grays) instead of letting
    them get clustered out.

    Remaining pixels are remapped to the nearest survivor in Lab space.
    """
    from collections import Counter
    flat = matched_grid.reshape(-1)
    counts = Counter(flat.tolist())
    if len(counts) <= max_colors:
        return matched_grid

    used_hexes = list(counts.keys())
    used_rgb = np.array([palette[h] for h in used_hexes], dtype=np.float32)
    used_lab = srgb_to_lab(used_rgb).astype(np.float32)

    # Seed with the most-used color.
    seed_hex = counts.most_common(1)[0][0]
    seed_idx = used_hexes.index(seed_hex)
    picked = [seed_idx]
    # min_dist[i] = distance from candidate i to the closest already-picked color
    min_dist = np.linalg.norm(used_lab - used_lab[seed_idx], axis=1)
    while len(picked) < max_colors:
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] == 0:
            break  # all remaining candidates already coincide with a pick
        picked.append(nxt)
        d = np.linalg.norm(used_lab - used_lab[nxt], axis=1)
        min_dist = np.minimum(min_dist, d)

    keep = [used_hexes[i] for i in picked]
    keep_set = set(keep)
    sub_palette = {h: palette[h] for h in keep}
    sub_matcher = Matcher(sub_palette)
    drop_hexes = [h for h in used_hexes if h not in keep_set]
    remap = {}
    if drop_hexes:
        rgbs = np.array([palette[h] for h in drop_hexes], dtype=np.int16)
        out = sub_matcher.match_pixels(rgbs.reshape(-1, 1, 3))
        for src, dst in zip(drop_hexes, out.reshape(-1)):
            remap[src] = dst
    if not remap:
        return matched_grid
    new_flat = np.array([remap.get(h, h) for h in flat.tolist()], dtype=object)
    return new_flat.reshape(matched_grid.shape)


class Matcher:
    """Vectorized closest-match in Lab color space."""

    def __init__(self, palette):
        from scipy.spatial import cKDTree
        self._hex_keys = np.array(list(palette.keys()), dtype=object)
        self._rgb = np.array(list(palette.values()), dtype=np.float64)
        self._tree = cKDTree(srgb_to_lab(self._rgb).astype(np.float32))

    def match_indices(self, pixel_array):
        """Map an (H, W, 3) or (N, 3) RGB to integer palette indices."""
        arr = np.asarray(pixel_array)
        uq, inv = np.unique(pack_rgb(arr.reshape(-1, 3)), return_inverse=True)
        _, near = self._tree.query(
            srgb_to_lab(unpack_rgb(uq).astype(np.float64)).astype(np.float32),
            k=1, workers=-1)
        return near[inv].reshape(arr.shape[:-1])

    def match_pixels(self, pixel_array):
        """Map an (H, W, 3) or (N, 3) RGB array to matched hex strings."""
        return self._hex_keys[self.match_indices(pixel_array)]
