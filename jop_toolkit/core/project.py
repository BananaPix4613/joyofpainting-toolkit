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
    depth: int = 6
    color_limit: int | None = None
    gamut_strength: float = 0.5     # 0 disables shadow/highlight remapping
    dither_strength: float = 0.85   # 0 disables error diffusion
    dither_method: str = "floyd-steinberg"
    dither_serpentine: bool = True
    dither_matrix: int = 8            # ordered/Bayer only
    # Source cleanup (all run at source resolution, before downscale)
    resample_filter: str = "box"      # "box" | "bilinear" | "lanczos"
    denoise_mode: str = "none"        # "none" | "guided" | "median"
    denoise_radius: int = 2
    denoise_strength: float = 0.06
    median_radius: int = 1
    deblock_strength: float = 0.0
    color_snap: int = 0               # 0/1 disables
    edge_hardness: float = 0.0        # 0 soft (as downscaled) .. 1 fully hard edges
    edge_threshold: float = 12.0      # L* contrast below which a gradient is kept
    compare_source: bool = False
    preprocessed_source: np.ndarray | None = None   # (H, W, 3) supersampled, for the swipe view
    match_source: np.ndarray | None = None          # (H, W, 3) at the index grid
    selected_dyes: list[str] = field(default_factory=lambda: list(BASE_COLORS.keys()))
    processed_image: np.ndarray | None = None       # (H, W, 3) uint8 — what the user sees in the preview
    palette: dict = field(default_factory=dict)     # hex -> (r,g,b)
    recipes: dict = field(default_factory=dict)     # hex -> [dye, ...]
    indexed: Any = None                               # IndexedImage -- the source of truth
    indexed_pristine: Any = None                      # as the pipeline produced it, pre-edit
    edit_overlay: dict = field(default_factory=dict)  # (y, x) -> hex, or None to erase
    edit_overlay_size: tuple | None = None            # (H, W) the overlay was recorded at
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
    # DEPRECATED -- palette groups. Nothing reads these any more; the worker no
    # longer accepts groups and the sidecar no longer writes them. Kept so any
    # code still touching the attributes does not raise while the modules below
    # remain in the tree: core/groups.py, core/axis_core.py, and
    # ui/widgets/{groups_dialog,group_table,palette_axis}.py.
    use_palette_groups: bool = False
    palette_groups: list = field(default_factory=list)   # [ColorGroup, ...]
    group_allocation: list = field(default_factory=list) # read-back for the UI
    group_mode: str = "axis"          # "axis" | "cores"
    group_axis: str = "lightness"

    @property
    def matched_hex_grid(self):
        """Derived view for consumers not yet migrated to the indexed model.
        Read-only on purpose: writing it would let palette and pixels diverge."""
        return None if self.indexed is None else self.indexed.to_hex_grid()
