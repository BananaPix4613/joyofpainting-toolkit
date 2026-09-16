"""Application bootstrap."""

import sys
import traceback
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

from .ui import theme
from .ui.main_window import MainWindow


def _asset(name):
    """Locate a bundled file in both a checkout and a PyInstaller --onefile exe.

    The frozen build unpacks its data into a temporary directory named by
    sys._MEIPASS; in a checkout the same fil esits beside the package. Returns
    None rather than raising, because a missing icon is not worth a crash.
    """
    roots = []
    base = getattr(sys, "_MEIPASS", None)
    if base:
        roots.append(Path(base))
    roots.append(Path(__file__).resolve().parent.parent)
    for r in roots:
        p = r / name
        if p.exists():
            return str(p)
    return None


def _install_crash_guard():
    """PyQt6 calls qFatal() on an unhandled exception in a slot, which on Windows
    surfaces as a bare 0xC0000409 and takes unsaved work with it. Overriding
    sys.excepthook keeps the process alive and shows what actually went wrong."""
    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        try:
            QMessageBox.critical(None, "Unexpected error",
                                 f"{exc_type.__name__}: {exc}\n\n{text[-2000:]}")
        except Exception:
            pass
    sys.excepthook = hook


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Joy of Painting Toolkit")
    art = _asset("icon.png")
    if art:
        app.setWindowIcon(QIcon(art))
    _install_crash_guard()
    theme.apply(theme.load())
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
