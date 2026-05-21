"""Build, load, and save the .jop.json sidecar."""

import json
from collections import Counter

import numpy as np


def sequence_recipe(recipe):
    """Reorder dyes so the first drop is a singleton (JoP starts mixing from 1 drop)."""
    if len(recipe) <= 1:
        return list(recipe)
    counts = Counter(recipe)
    starter = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    rest = list(recipe)
    rest.remove(starter)
    interleaved = []
    pool = Counter(rest)
    last = starter
    while sum(pool.values()):
        candidates = [d for d, n in pool.items() if n > 0 and d != last] or \
                     [d for d, n in pool.items() if n > 0]
        pick = sorted(candidates, key=lambda d: -pool[d])[0]
        interleaved.append(pick)
        pool[pick] -= 1
        last = pick
    return [starter] + interleaved


def order_palette(palette_entries):
    """Sort palette entries by (recipe length, dye signature, hex) so similar mixes cluster."""
    return sorted(
        palette_entries,
        key=lambda e: (len(e['recipe']), tuple(sorted(set(e['recipe']))), e['hex']),
    )


def slice_tiles(matched_hex_grid, layout):
    """Extract per-tile 2D hex grids from a full matched image and a layout list."""
    tiles = []
    for idx, p in enumerate(layout):
        x, y = p['pixel_pos']
        tw, th = p['tile_size']
        sub = matched_hex_grid[y:y + th, x:x + tw]
        pixels = [[str(sub[r, c]) for c in range(tw)] for r in range(th)]
        tiles.append({
            'index': idx,
            'pixel_pos': [int(x), int(y)],
            'tile_size': [int(tw), int(th)],
            'pixels': pixels,
        })
    return tiles


def build_palette_entries(tiles, palette, recipes):
    """Build sequenced palette entries containing only colors actually used in the tiles."""
    used = {}
    for tile in tiles:
        for row in tile['pixels']:
            for h in row:
                if h not in used:
                    used[h] = sequence_recipe(recipes[h])
    entries = [{'hex': h, 'rgb': list(palette[h]), 'recipe': used[h]} for h in used]
    return order_palette(entries)


def build_sidecar_from_state(state):
    """Compose the sidecar dict from a fully-populated ProjectState."""
    h, w = state.matched_hex_grid.shape
    tiles = slice_tiles(state.matched_hex_grid, state.layout)
    palette_entries = build_palette_entries(tiles, state.palette, state.recipes)
    return {
        'source_image': str(state.source_path) if state.source_path else '',
        'image_size': [int(w), int(h)],
        'palette': palette_entries,
        'tiles': tiles,
        'toolkit': {
            'target_resolution': list(state.target_resolution),
            'margin': int(state.margin),
            'margins': list(state.margins),
            'depth': int(state.depth),
            'color_limit': state.color_limit,
            'selected_dyes': list(state.selected_dyes),
            'layout_mode': state.layout_mode,
        },
    }


def write_sidecar(sidecar, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, indent=2)


def load_sidecar(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
