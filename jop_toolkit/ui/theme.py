"""Light, dark, or follow the system.

Qt does the recoloring: setting the style hints' color scheme swaps the whole
application palette. Not every style listens, though. The windowsvista style,
which is the default on Windows 10, stays light whatever the scheme says, so when
dark is wanted and the palette did not follow, this switches to Fusion, which does.
The original style comes back when the scheme returns to light.

Fusion on a dark Windows palette fills unchecked checkboxes and radio buttons with
the accent blue, so they look checked. Setting the Base color explicitly, even to
the value it already has, makes Fusion draw them empty again.

The choice is remembered with QSettings, which on Windows is a registry key
under HKEY_CURRENT_USER.
"""

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QApplication

MODES = ("system", "light", "dark")
LABELS = {"system": "&System", "light": "&Light", "dark": "&Dark"}

_KEY = "appearance/theme"
_native_style = None


def _settings():
    return QSettings("JoyOfPaintingToolkit", "JoyOfPaintingToolkit")


def load():
    mode = _settings().value(_KEY, "system")
    return mode if mode in MODES else "system"


def save(mode):
    _settings().setValue(_KEY, mode)


def _looks_light(palette):
    return palette.color(QPalette.ColorRole.Window).lightness() > 128


def _pin_base(app):
    pal = app.palette()
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive,
                  QPalette.ColorGroup.Disabled):
        pal.setColor(group, QPalette.ColorRole.Base, pal.color(group, QPalette.ColorRole.Base))
    app.setPalette(pal)


def apply(mode):
    """Switch the running application to `mode`. Safe to call repeatedly."""
    global _native_style
    app = QApplication.instance()
    if app is None:
        return
    if _native_style is None:
        _native_style = app.style().name()
    hints = app.styleHints()
    # A pinned palette would stop Qt from following the scheme, so start clean.
    app.setPalette(QPalette())
    if mode == "light":
        hints.setColorScheme(Qt.ColorScheme.Light)
    elif mode == "dark":
        hints.setColorScheme(Qt.ColorScheme.Dark)
    else:
        hints.unsetColorScheme()

    # In "system" mode this is the OS setting, so a dark Windows 10 desktop
    # takes the Fusion fallback too.
    want_dark = hints.colorScheme() == Qt.ColorScheme.Dark
    if want_dark:
        if _looks_light(app.palette()):
            app.setStyle("fusion")
        if app.style().name() == "fusion":
            _pin_base(app)
    elif app.style().name() != _native_style:
        app.setStyle(_native_style)
