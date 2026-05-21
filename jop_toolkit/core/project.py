"""Shared mutable project state for all three stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .colors import BASE_COLORS


@dataclass
class ProjectState:
    source_path: Path | None = None
    source_image: Any = None  # PIL.Image
    # Stage 1
    target_resolution: tuple[int, int] = (64, 64)
    margin: int = 0  # legacy/uniform — kept for sidecar back-compat
    margins: tuple[int, int, int, int] = (0, 0, 0, 0)  # top, right, bottom, left
    depth: int = 4
    color_limit: int | None = None
    selected_dyes: list[str] = field(default_factory=lambda: list(BASE_COLORS.keys()))
    processed_image: np.ndarray | None = None     # (H, W, 3) uint8 — what the user sees in the preview
    palette: dict = field(default_factory=dict)     # hex -> (r,g,b)
    recipes: dict = field(default_factory=dict)     # hex -> [dye, ...]
    matched_hex_grid: np.ndarray | None = None      # (H, W) object dtype, hex strings
    # Stage 2
    layout: list = field(default_factory=list)      # [{pixel_pos: [x,y], tile_size: [w,h]}]
    layout_mode: str = "auto"                       # "auto" | "manual"
    auto_size_mode: str = "mixed"                   # "mixed" | "16x16" | "16x32" | "32x16" | "32x32"
    # Stage 3
    active_tile_index: int = 0
    active_color_hex: str | None = None
    # Cached palette entries (built when entering Stage 3)
    palette_entries: list = field(default_factory=list)
    tiles: list = field(default_factory=list)       # per-tile sliced hex grids (after layout is finalized)
