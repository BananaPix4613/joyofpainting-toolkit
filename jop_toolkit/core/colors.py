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
    n = len(colors)
    if n == 0:
        return (0, 0, 0)
    r = round(sum(c[0] for c in colors) / n)
    g = round(sum(c[1] for c in colors) / n)
    b = round(sum(c[2] for c in colors) / n)
    return (r, g, b)
