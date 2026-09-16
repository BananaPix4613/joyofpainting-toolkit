"""A drawing of the mod's palette with one mixing step marked on it.

Spots are unlabeled in game, so the card never names one: it shows every spot's
current color, rings the dye (or the water) to pick up and the spot to drop it on,
and joins them with an arrow. Positions come from core.palette_layout; the drawing
itself is ours, since the mod's palette art is GPL-3.0.
"""

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ...core import palette_layout as L

_RING = QColor('#ffd23c')
_WATER = QColor('#3a78c8')


class PaletteCard(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spots = (None,) * len(L.SPOTS)
        self._dye = None          # index into DYE_ORDER to pick up from
        self._water = False       # pick up water instead
        self._spot = None         # spot to drop onto
        self._used_dyes = None    # dim wells the plan never touches
        self.setMinimumSize(180, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_used_dyes(self, names):
        self._used_dyes = None if names is None else {L.dye_index(n) for n in names}
        self.update()

    def set_step(self, spots, dye=None, water=False, spot=None):
        """spots: hex or None per spot, as they look after this step."""
        self._spots = tuple(spots) if spots else (None,) * len(L.SPOTS)
        self._dye = None if dye is None else L.dye_index(dye)
        self._water = bool(water)
        self._spot = spot
        self.update()

    def clear_step(self):
        self.set_step(None)

    def _geometry(self):
        pw, ph = L.PALETTE_SIZE
        margin = 8
        scale = min((self.width() - 2 * margin) / pw, (self.height() - 2 * margin) / ph)
        ox = (self.width() - pw * scale) / 2
        oy = (self.height() - ph * scale) / 2
        return scale, ox, oy

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        s, ox, oy = self._geometry()

        def pt(xy):
            return QPointF(ox + xy[0] * s, oy + xy[1] * s)

        def circle(xy, r, fill, ring=None, ring_w=1.0):
            c = pt(xy)
            p.setBrush(fill)
            p.setPen(QPen(ring or QColor(0, 0, 0, 90), ring_w))
            p.drawEllipse(c, r * s, r * s)

        # the board
        board = self.palette().color(self.palette().ColorRole.Mid)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(board)
        pw, ph = L.PALETTE_SIZE
        p.drawRoundedRect(QRectF(ox, oy, pw * s, ph * s), 14 * s, 14 * s)

        for i, xy in enumerate(L.DYE_WELLS):
            col = QColor(L.DYE_HEX[i])
            if self._used_dyes is not None and i not in self._used_dyes and i != self._dye:
                col.setAlpha(70)
            circle(xy, L.DYE_RADIUS, col)
        circle(L.WATER, L.SPOT_RADIUS, _WATER)
        for i, xy in enumerate(L.SPOTS):
            h = self._spots[i] if i < len(self._spots) else None
            circle(xy, L.SPOT_RADIUS, QColor(h) if h else QColor(*L.EMPTY_SPOT))

        src = None
        if self._dye is not None:
            src, src_r = L.DYE_WELLS[self._dye], L.DYE_RADIUS
        elif self._water:
            src, src_r = L.WATER, L.SPOT_RADIUS
        if self._spot is not None:
            dst = L.SPOTS[self._spot]
            if src is not None:
                self._arrow(p, pt(src), pt(dst), src_r * s, L.SPOT_RADIUS * s)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(_RING, max(2.0, 1.4 * s)))
                p.drawEllipse(pt(src), src_r * s + 2, src_r * s + 2)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(_RING, max(2.0, 1.4 * s)))
            p.drawEllipse(pt(dst), L.SPOT_RADIUS * s + 2, L.SPOT_RADIUS * s + 2)
        p.end()


    @staticmethod
    def _arrow(p, a, b, ra, rb):
        dx, dy = b.x() - a.x(), b.y() - a.y()
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        start = QPointF(a.x() + ux * (ra + 3), a.y() + uy * (ra + 3))
        tip = QPointF(b.x() - ux * (rb + 4), b.y() - uy * (rb + 4))
        pen = QPen(_RING, 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(start, tip)
        head = 9.0
        left = QPointF(tip.x() - ux * head - uy * head * 0.55, tip.y() - uy * head + ux * head * 0.55)
        right = QPointF(tip.x() - ux * head + uy * head * 0.55, tip.y() - uy * head - ux * head * 0.55)
        p.setBrush(_RING)
        p.drawPolygon(QPolygonF([tip, left, right]))
