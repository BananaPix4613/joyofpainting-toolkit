"""Vector icons, defined in source rather than shipped as files.

The build is `pyinstaller --onefile`, which bundles nothing it is not told about,
so an icons/ directory would quietly vanish from the packaged exe and every
button would come up blank. Path data in a module cannot go missing, needs no
--add-data, and behaves identically in a dev checkout and a frozen build.

Drawn as strokes on a 24x24 grid so one definition serves every size and recolors
cleanly for light and dark themes.
"""

from PyQt6 import sip
from PyQt6.QtCore import QByteArray, QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QPalette
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication

_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{c}" stroke-width="{w}" stroke-linecap="round" '
        'stroke-linejoin="round">{body}</svg>')

# Each entry is the body of that 24x24 drawing.
PATHS = {
    # --- tools ---
    'brush':   '<path d="M20 3 12 11"/>'
               '<path d="M13.5 9.5 14.5 10.5"/>'
               '<path d="M11 12c-1.7 0-3 1.3-3 3 0 1.7-1.5 2.5-3.5 2.5 1 1.5 2.6 2.5 4.5 2.5'
               ' 2.8 0 5-2.2 5-5 0-1.7-1.3-3-3-3z"/>',
    'eraser':  '<path d="M7.5 20H20"/>'
               '<path d="M16 4 4.5 15.5a2 2 0 0 0 0 2.8l2.2 2.2a2 2 0 0 0 2.8 0L21 9z"/>'
               '<path d="M10 10 15 15"/>',
    'bucket':  '<path d="M9 3 19 13a1.5 1.5 0 0 1 0 2.1l-5.4 5.4a1.5 1.5 0 0 1-2.1 0'
               'L3 12.1a1.5 1.5 0 0 1 0-2.1L8 5"/>'
               '<path d="M5 10h13"/>'
               '<path d="M21 16c0 1.7-.9 3-2 3s-2-1.3-2-3 2-4 2-4 2 2.3 2 4z"/>',
    'dropper': '<path d="M18 2.5a2.5 2.5 0 0 1 3.5 3.5L17 10.5 13.5 7z"/>'
               '<path d="M15 8.5 5.5 18a2 2 0 0 0-.5 1.2L4.5 21l1.8-.5A2 2 0 0 0 7.5 20L17 10.5z"/>',
    'rect':    '<path d="M3 8V5a2 2 0 0 1 2-2h3"/><path d="M16 3h3a2 2 0 0 1 2 2v3"/>'
               '<path d="M21 16v3a2 2 0 0 1-2 2h-3"/><path d="M8 21H5a2 2 0 0 1-2-2v-3"/>'
               '<path d="M11 3h2"/><path d="M11 21h2"/><path d="M3 11v2"/><path d="M21 11v2"/>',
    'lasso':   '<path d="M12 4c5 0 9 2.7 9 6 0 2.4-2.1 4.5-5.2 5.4"/>'
               '<path d="M12 4C7 4 3 6.7 3 10c0 1.8 1.2 3.4 3 4.5"/>'
               '<path d="M6 14.5c0 2 .7 3.5 0 5"/>'
               '<circle cx="5.5" cy="20.5" r="1.6"/>'
               '<path d="M15.8 15.4c-1.2.4-2.5.6-3.8.6"/>',
    'wand':    '<path d="M4 20 15 9"/><path d="M13.5 7.5 16.5 10.5"/>'
               '<path d="M18 3v3"/><path d="M21.5 4.5 19.5 6.5"/><path d="M21 10h-3"/>'
               '<path d="M14.5 3.5 16 5"/>',
    'move':    '<path d="M12 3v18"/><path d="M3 12h18"/>'
               '<path d="M9 6 12 3l3 3"/><path d="M9 18l3 3 3-3"/>'
               '<path d="M6 9 3 12l3 3"/><path d="M18 9l3 3-3 3"/>',
    # --- small buttons ---
    'swap':    '<path d="M4 8h13"/><path d="M13 4l4 4-4 4"/>'
               '<path d="M20 16H7"/><path d="M11 12l-4 4 4 4"/>',
    'fit':     '<path d="M3 8V4a1 1 0 0 1 1-1h4"/><path d="M16 3h4a1 1 0 0 1 1 1v4"/>'
               '<path d="M21 16v4a1 1 0 0 1-1 1h-4"/><path d="M8 21H4a1 1 0 0 1-1-1v-4"/>'
               '<rect x="8" y="8" width="8" height="8" rx="1"/>',
    'sel_all': '<rect x="3" y="3" width="18" height="18" rx="2"/>'
               '<path d="M7.5 12.5 11 16l5.5-6.5"/>',
    # Corners plus a slash reads as "expand" -- the two arcs land where an
    # arrowhead would. A centered cross is unambiguous.
    'sel_none': '<path d="M3 8V5a2 2 0 0 1 2-2h3"/><path d="M16 3h3a2 2 0 0 1 2 2v3"/>'
                '<path d="M21 16v3a2 2 0 0 1-2 2h-3"/><path d="M8 21H5a2 2 0 0 1-2-2v-3"/>'
                '<path d="M9 9 15 15"/><path d="M15 9 9 15"/>',
    'sel_inv': '<rect x="3" y="3" width="18" height="18" rx="2"/>'
               '<path d="M3 3h18v18z" fill="{c}" stroke="none"/>',
}

_cache = {}
_bound = []      # (button or action, name, size) that follow theme changes


def _stroke_color():
    app = QApplication.instance()
    if app is None:
        return "#d0d0d0"
    return app.palette().color(QPalette.ColorRole.WindowText).name()


def icon(name, size=20, color=None):
    """QIcon for `name`, tinted to the current theme's text color."""
    body = PATHS.get(name)
    if body is None:
        return QIcon()
    c = color or _stroke_color()
    key = (name, size, c)
    if key in _cache:
        return _cache[key]
    # Thinner strokes at small sizes, or a 16px button turns into a blob.
    width = 2.0 if size >= 20 else 1.7
    svg = _SVG.format(c=c, w=width, body=body.replace("{c}", c))
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(p)
    p.end()
    ic = QIcon(pm)
    _cache[key] = ic
    return ic


def clear_cache():
    """Call if the palette changes; icons are baked at one color."""
    _cache.clear()


def apply(target, name, size=20):
    """Give a button or action its icon, and re-tint it on every retint()."""
    target.setIcon(icon(name, size))
    _bound.append((target, name, size))
    return target


def retint():
    """Redraw every applied icon in the current palette's text color."""
    clear_cache()
    live = []
    for target, name, size in _bound:
        if sip.isdeleted(target):
            continue
        target.setIcon(icon(name, size))
        live.append((target, name, size))
    _bound[:] = live
