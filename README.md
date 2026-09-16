# Joy of Painting Toolkit

A standalone desktop app for translating any image into a Joy of Painting
(Minecraft mod) painting plan: downscale, color-match to dye-mixable colors,
arrange canvases, and view each canvas with a pulsing highlight on the active
color so you know exactly which pixels to paint.

## Install

```powershell
pip install -e .
```

## Run

```powershell
python -m jop_toolkit
```

## Workflow

1. **Source** - Open an image (`Ctrl+I`) and set the target resolution,
   edge-smear margins, palette depth, color limit, and which dyes you have.
   Tone handling covers shadow lift and dithering; source cleanup covers
   downscale filter, denoise, JPEG deblock, color snap, and edge hardness.
   Press **Apply Changes** (`F5`) to process.
2. **Edit** - Clean up the result pixel by pixel.
   - Tools: Brush `B`, Eraser `E`, Fill `G`, Pick `I`, Marquee `M`,
     Lasso `Q`, Wand `W`, Move `V`. Left and right click paint with two
     colors (`X` swaps them); Alt-click picks.
   - Selections: Shift-drag adds, Ctrl-drag subtracts, and every tool stays
     inside the selection. Cut, copy, and paste work on selections.
   - The **Edit** menu has Reduce Colors, Add Detail, Select Lines, and
     Clean Up Speckles, each with an optional preview before you apply.
   - The palette panel recolors, replaces, adds, and removes colors; every
     color stays mixable from the dyes you have.
   - Edits are kept when you re-apply the Source stage, as long as the image
     size does not change.
3. **Layout** - Default is auto mixed-size packing. Switch to Manual to
   click-place individual canvases; right-click removes one.
4. **Paint** - Pick a canvas from the overview (or `PgUp`/`PgDn`), then step
   through the colors and their dye recipes with the arrow keys (or
   `◀ Color` / `Color ▶`). The active color pulses on the canvas grid; click
   a cell to jump to its color. Show all colors in the image, or only the
   ones in the current canvas.

Save your work with `Ctrl+S`. It writes a `.jop.json` sidecar with the
canvas pixels and dye recipes, plus a `toolkit` block that restores your
settings and manual edits when you open it again with `Ctrl+O`.


## Package

```powershell
.\scripts\build_exe.ps1
```

Produces `dist/JoyOfPaintingToolkit.exe`.
