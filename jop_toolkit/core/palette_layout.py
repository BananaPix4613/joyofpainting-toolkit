"""Where things are on the mod's palette screen.

Positions come from the mod's own layout data (BasePalette.BASIC_COLOR_CENTERS,
CUSTOM_COLOR_CENTERS and WATER_CENTER), in palette pixels. Only the layout is used:
the mod is GPL-3.0, so its palette artwork is not reproduced here, and the guide
draws its own.

Checked against a hand calibration of the real game rather than taken on trust:
at GUI scale 3, screen = 3 * position + origin placed all 16 dye wells within
2.6 px on average of where they were clicked, against a 33 px well radius.

Spot numbering is internal. In game the spots are unlabeled so a guide has to
point at a position rather than saying "spot 8".
"""

PALETTE_SIZE = (157, 193)

# BasePalette.BASIC_COLORS order. Names match core.colors.BASE_COLORS; the hex is
# the mod's own value, kept so a test can prove the two tables agree.
DYE_ORDER = ("Black", "Red", "Green", "Brown", "Blue", "Purple", "Cyan", "Light Gray",
             "Gray", "Pink", "Lime", "Yellow", "Light Blue", "Magenta", "Orange", "White")
DYE_HEX = ("#1D1D21", "#B02E26", "#5E7C16", "#835432", "#3C44AA", "#8932B8", "#169C9C",
           "#9D9D97", "#474F52", "#F38BAA", "#80C71F", "#FED83D", "#3AB3DA", "#C74EBD",
           "#F9801D", "#F9FFFE")

DYE_WELLS = ((23.5, 172.5), (18.5, 145.5), (16.5, 117.5), (17.5, 89.5), (23.5, 62.5),
             (38.5, 39.5), (61.5, 24.5), (87.5, 17.5), (114.5, 15.5), (44.5, 154.5),
             (41.5, 127.5), (42.5, 100.5), (48.5, 74.5), (64.5, 52.5), (90.5, 44.5),
             (117.5, 42.5))

SPOTS = ((101.5, 132.0), (113.5, 118.0), (120.5, 102.0), (124.5, 84.0), (126.5, 66.0),
         (97.5, 152.0), (114.5, 146.0), (127.5, 133.0), (134.5, 116.0), (139.5, 98.0),
         (142.5, 80.0), (144.5, 62.0))

WATER = (140.5, 28.0)

DYE_RADIUS = 11.0
SPOT_RADIUS = 6.5             # the water well uses the same radius

EMPTY_SPOT = (255, 236, 229)  # PaletteUtil.EMPTINESS_COLOR


def dye_index(name):
    return DYE_ORDER.index(name)
