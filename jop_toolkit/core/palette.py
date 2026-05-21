"""Generate the dye-mix palette and run vectorized Lab-space matching."""

import itertools

import numpy as np

from .colors import BASE_COLORS, hex_to_rgb, rgb_to_hex
from .colorspace import srgb_to_lab


def generate_palette(selected_colors=None, max_depth=4, progress_cb=None):
    """Return ({hex: rgb}, {hex: [dye, ...]}) for every reachable mix.

    progress_cb(done, total, label) is called once per depth if provided.
    """
    selected = selected_colors or list(BASE_COLORS.keys())
    base_names = [n for n in selected if n in BASE_COLORS]
    if not base_names:
        raise ValueError("No valid base colors selected")
    base_rgb = np.array([hex_to_rgb(BASE_COLORS[n]) for n in base_names], dtype=np.float64)

    palette = {}
    recipes = {}
    for name, rgb in zip(base_names, base_rgb):
        rgb_t = tuple(int(v) for v in rgb)
        h = rgb_to_hex(rgb_t)
        palette[h] = rgb_t
        recipes[h] = [name]

    depths = list(range(2, max_depth + 1))
    for i, depth in enumerate(depths, start=1):
        combos = np.fromiter(
            itertools.chain.from_iterable(
                itertools.combinations_with_replacement(range(len(base_names)), depth)
            ),
            dtype=np.int16,
        ).reshape(-1, depth)
        if combos.size == 0:
            continue
        rgbs = base_rgb[combos].mean(axis=1)
        rounded = np.round(rgbs).astype(np.int16)
        unique_rows, first_idx = np.unique(rounded, axis=0, return_index=True)
        for row_idx, color in zip(first_idx, unique_rows):
            color_t = tuple(int(c) for c in color)
            h = rgb_to_hex(color_t)
            if h in palette:
                continue
            palette[h] = color_t
            recipes[h] = [base_names[k] for k in combos[row_idx]]
        if progress_cb:
            progress_cb(i, len(depths), f"Depth {depth}: {len(palette)} colors")

    return palette, recipes


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
    """Vectorized closest-match in Lab color space, with unique-pixel caching."""

    def __init__(self, palette):
        from scipy.spatial import cKDTree
        self._hex_keys = list(palette.keys())
        self._rgb = np.array([palette[h] for h in self._hex_keys], dtype=np.float32)
        self._lab = srgb_to_lab(self._rgb).astype(np.float32)
        self._tree = cKDTree(self._lab)
        self._cache = {}

    def match_pixels(self, pixel_array):
        """Map an (H, W, 3) or (N, 3) RGB array to matched hex strings (object dtype)."""
        arr = np.asarray(pixel_array)
        flat = arr.reshape(-1, 3).astype(np.int16)
        uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
        new_mask = np.array([tuple(p) not in self._cache for p in uniq])
        new_uniq = uniq[new_mask]
        if len(new_uniq):
            new_lab = srgb_to_lab(new_uniq.astype(np.float32)).astype(np.float32)
            _, nearest = self._tree.query(new_lab, k=1, workers=-1)
            for j, n in enumerate(nearest):
                self._cache[tuple(int(v) for v in new_uniq[j])] = self._hex_keys[n]
        result_flat = np.array([self._cache[tuple(p)] for p in flat], dtype=object)
        return result_flat.reshape(arr.shape[:-1])
