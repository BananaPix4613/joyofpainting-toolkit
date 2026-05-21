# Joy of Painting Toolkit

A standalone desktop app for translating any image into a Joy of Painting
(Minecraft mod) painting plan: downscale, color-match to dye-mixable colors,
arrange canvases, and view each canvas with a pulsing highlight on the active
color so you know exactly which pixels to paint.

The painting itself stays manual — this is a visual guide, not an input
automator.

## Install

```powershell
pip install -e .
```

## Run

```powershell
python -m jop_toolkit
```

## Workflow

1. **Source** — Open an image; adjust target resolution, edge-smear margin,
   palette depth, and which dyes you have available. Preview updates live.
2. **Layout** — Default is auto mixed-size packing. Switch to Manual to
   click-place individual canvases.
3. **Paint** — Pick a tile from the overview; iterate through the color
   recipes with the arrow keys (or `◀ Color` / `Color ▶`). The active color
   pulses on the canvas grid. Middle-click a cell to jump to its color.

Save your work with `Ctrl+S` (writes a `.jop.json` sidecar, compatible with
the original `joyofpainting-tools` CLI plus a small `toolkit` block for
restoring UI state).

## Package

```powershell
.\scripts\build_exe.ps1
```

Produces `dist/JoyOfPaintingToolkit.exe`.
