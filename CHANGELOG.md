# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-16-9

### Added
- **Edit stage** between Source and Layout for pixel-level cleanup of the
  color-matched image.
  - Tools: Brush (B), Eraser (E), Fill (G), Pick (I), Marquee (M), Lasso (Q),
    Wand (W), Move (V). Left and right mouse buttons paint with two colors;
    X swaps them, and Alt-click picks. Picking a color that is in the palette
    selects it there.
  - Selections: Shift-drag adds, Ctrl-drag subtracts, and every tool stays
    inside the selection. Select All (Ctrl+A), Deselect (Ctrl+D), Invert
    (Ctrl+Shift+I), Fill Selection (Ctrl+Enter), Erase Selected Pixels (Del).
  - Cut, Copy and Paste of selections, including between images.
  - Undo and Redo, plus Revert All Edits to go back to the pipeline result.
  - Palette panel: change a color in place, replace one color with another,
    add, duplicate, remove, sort and drop unused colors. Every color snaps to
    something the mod can mix, so each one keeps a usable recipe.
- **Edit menu operations**, each in a dialog with Apply/Cancel and an optional
  canvas preview:
  - Reduce Colors: reduce to N colors, merge similar colors, or limit the
    selection to N colors, with "Preserve custom colors".
  - Add Detail: find extra colors for the selected area, optionally where those
    colors appear elsewhere, and optionally dithered with the Source settings.
  - Select Lines: detect thin dark (or light) lines to refine them.
  - Clean Up Speckles, restricted to the selection when there is one.
- **View menu**: Zoom In/Out, Fit to Window (Ctrl+0), Zoom to Selection
  (Ctrl+Shift+0), Show Grid (Ctrl+'), Compare With Source.
- **Dark mode**: View → Theme → System, Light, or Dark, remembered
  between runs.
- **Source stage options**:
  - Tone handling: shadow lift, dithering (Floyd-Steinberg, Atkinson, Jarvis,
    Stucki, ordered, blue noise) with serpentine, matrix size and flat
    threshold.
  - Source cleanup: downscale filter, guided or median denoise, JPEG deblock,
    color snap, edge hardness with a gradient threshold, and a comparison view
    against the cleaned source.
  - Progress dialog with time remaining and Cancel.
- Toolbar icons with shortcuts in their tooltips, and a new application icon.

### Changed
- The workflow has four stages: Source, Edit, Layout, Paint.
- Manual edits are saved in the project and re-applied when the Source stage
  is applied again, as long as the image size does not change.
- Sidecars are written as format 3. Pixels erased in the Edit stage are
  written as `null`.
- Requires PyQt6 6.8 or newer.

### Fixed
- Layout and Paint now show manual edits right after a project is loaded or
  Source is re-applied, instead of the image from before the edits.

## [0.1.0] - 2026-05-21
- Initial public release.
- Three-stage workflow: Source (import + preprocess), Layout (auto/manual canvas packing), Paint (per-canvas guide).
- Directional smear/crop margins, palette depth with Apply, optional spectrum-coverage color limit.
- Sidecar `.jop.json` save/load (forward-compatible with the original CLI tool).
- Single-file Windows .exe via PyInstaller.
