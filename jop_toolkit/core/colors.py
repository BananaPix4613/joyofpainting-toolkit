"""The 16 Minecraft dye colors and small RGB helpers."""

BASE_COLORS = {
    'White': '#F9FFFE',
    'Orange': '#F9801D',
    'Magenta': '#C74EBD',
    'Light Blue': '#3AB3DA',
    'Yellow': '#FED83D',
    'Lime': '#80C71F',
    'Pink': '#F38BAA',
    'Gray': '#474F52',
    'Light Gray': '#9D9D97',
    'Cyan': '#169C9C',
    'Purple': '#8932B8',
    'Blue': '#3C44AA',
    'Brown': '#835432',
    'Green': '#5E7C16',
    'Red': '#B02E26',
    'Black': '#1D1D21',
}


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def color_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def mix_colors(colors):
    """Mix dye colors exactly as the mod does.

    xercapaint's PaletteUtil.CustomColor.calculateResult computes
        int gainFactor = averageMaximum / maximumOfAverage;
    in *integer* arithmetic. Across the 16 dyes that ratio peaks at 1.7485 and
    never reaches 2.0, so the division always truncates to 1 and vanilla's
    leather-armor gain correction never fires. What remains is the plain
    per-channel mean, floored by Java's integer division.
    """

    n = len(colors)
    if n == 0:
        return (0, 0, 0)
    return tuple(sum(c[i] for c in colors) // n for i in range(3))
