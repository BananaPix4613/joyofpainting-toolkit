"""Build, load, and save the .jop.json sidecar."""

import json
from collections import Counter

import numpy as np

from .colors import hex_to_rgb


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


def slice_tiles(indexed, layout):
    """Per-tile hex grids from an IndexedImage and a layout list.

    Erased pixels serialize as null. A canvas that overhangs the grid edge-pads
    rather than failing, so a half-covered bottom row streaks the last colors.
    """
    hexes = indexed.hex_array()
    tiles = []
    for idx, p in enumerate(layout):
        x, y = p['pixel_pos']
        tw, th = p['tile_size']
        sub = indexed.indices[y:y + th, x:x + tw]
        if sub.size == 0:
            continue
        if sub.shape[0] < th or sub.shape[1] < tw:
            sub = np.pad(sub, ((0, th - sub.shape[0]), (0, tw - sub.shape[1])), mode='edge')
        pixels = [[None if sub[r, c] < 0 else str(hexes[sub[r, c]]) for c in range(tw)]
                  for r in range(th)]
        tiles.append({
            'index': idx,
            'pixel_pos': [int(x), int(y)],
            'tile_size': [int(tw), int(th)],
            'pixels': pixels,
        })
    return tiles


def build_palette_entries(indexed, tiles):
    """Sequenced entries for the colors actually present in the tiles.

    Driven by slot order, so the palette lists in paint order rather than
    whatever order a dict happened to produce.
    """
    used = set()
    for tile in tiles:
        for row in tile['pixels']:
            for h in row:
                if h not in used:
                    used.add(h)
    entries = [{'hex': s.hex, 'rgb': list(hex_to_rgb(s.hex)),
                'recipe': sequence_recipe(s.recipe)}
                for s in indexed.slots if s.hex in used]
    return order_palette(entries)


def build_sidecar_from_state(state):
    """Compose the sidecar dict from a fully-populated ProjectState."""
    h, w = state.indexed.shape
    tiles = slice_tiles(state.indexed, state.layout)
    palette_entries = build_palette_entries(state.indexed, tiles)
    return {
        'format': 3,
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
            'gamut_strength': float(state.gamut_strength),
            'dither_strength': float(state.dither_strength),
            'dither_method': state.dither_method,
            'dither_serpentine': bool(state.dither_serpentine),
            'dither_matrix': int(state.dither_matrix),
            'resample_filter': state.resample_filter,
            'denoise_mode': state.denoise_mode,
            'denoise_radius': int(state.denoise_radius),
            'denoise_strength': float(state.denoise_strength),
            'median_radius': int(state.median_radius),
            'deblock_strength': float(state.deblock_strength),
            'color_snap': int(state.color_snap),
            'edge_hardness': float(state.edge_hardness),
            'edge_threshold': float(state.edge_threshold),
            'edit_overlay': [[int(y), int(x), hx] for (y, x), hx in state.edit_overlay.items()],
            'edit_overlay_size': (list(state.edit_overlay_size)
                                  if state.edit_overlay_size else None),
        },
    }


def write_sidecar(sidecar, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, indent=2)


def load_sidecar(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
