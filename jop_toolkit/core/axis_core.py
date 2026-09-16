"""Axis machinery for the palette number line: order an image's colors along a
single selectable axis, render that ordering as a strip, and partition it with
markers.

A marker owns the stretch of axis nearest to it (1-D Voronoi), so boundaries are
implicit midpoints and dragging a marker moves both its anchor and its edges. Its
core color is derived from its position unless the user pins an override, which
is what makes the strip behave like a curve whose output is color.
"""

import numpy as np

AXES = ('lightness', 'chroma', 'hue', 'spine')


def _pc1(lab, w):
    mean = (lab * w[:, None]).sum(0) / w.sum()
    X = lab - mean
    cov = (X * w[:, None]).T @ X / w.sum()
    val, vec = np.linalg.eigh(cov)
    order = np.argsort(-val)
    return X @ vec[:, order[0]], float(val[order][0] / val.sum())


def _spine(lab, w, n_knots=24):
    """Order along a coarse principal curve: bin by PC1, take weighted bin
    centroids, then project each color onto the nearest segment of that path.
    Follows a bent color route (dark -> red -> skin -> light) that a straight
    line cannot."""
    t, _ = _pc1(lab, w)
    edges = np.quantile(t, np.linspace(0, 1, n_knots + 1))
    knots = []
    for i in range(n_knots):
        m = (t >= edges[i]) & (t <= edges[i + 1])
        if m.any() and w[m].sum() > 0:
            knots.append((lab[m] * w[m, None]).sum(0) / w[m].sum())
    K = np.array(knots)
    if len(K) < 2:
        return t
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(K, axis=0), axis=1))]
    pos = np.zeros(len(lab))
    best = np.full(len(lab), np.inf)
    for i in range(len(K) - 1):
        a, d = K[i], K[i + 1] - K[i]
        denom = float(d @ d) or 1.0
        u = np.clip((lab - a) @ d / denom, 0.0, 1.0)
        dist = np.linalg.norm(lab - (a + u[:, None] * d), axis=1)
        hit = dist < best
        best[hit] = dist[hit]
        pos[hit] = seg[i] + u[hit] * np.linalg.norm(d)
    return pos


def axis_positions(uq_lab, weights, axis='lightness'):
    """Raw axis coordinate per color. Hue is an angle in degrees; the rest are
    linear coordinates."""
    L, A, B = uq_lab[:, 0], uq_lab[:, 1], uq_lab[:, 2]
    if axis == 'lightness':
        return L.copy()
    if axis == 'chroma':
        return np.hypot(A, B)
    if axis == 'hue':
        return np.degrees(np.arctan2(B, A)) % 360.0
    if axis == 'spine':
        return _spine(uq_lab, weights)
    return _pc1(uq_lab, weights)[0]


def normalize_positions(pos, axis='lightness'):
    """Map to 0..1 so marker positions are axis-independent and survive a switch."""
    if axis == 'hue':
        return pos / 360.0
    lo, hi = float(np.min(pos)), float(np.max(pos))
    if hi - lo < 1e-9:
        return np.zeros_like(pos)
    return (pos - lo) / (hi - lo)


def build_strip(uq_rgb, uq_lab, weights, axis='lightness', n=256):
    """(n, 3) uint8 strip: the image's own colors ordered along the axis.

    Each column is the weighted mean of the colors falling in that slice, so the
    strip shows what the image actually contains rather than a synthetic ramp.
    Empty slices inherit their nearest filled neighbor.
    """
    pos = normalize_positions(axis_positions(uq_lab, weights, axis), axis)
    idx = np.clip((pos * n).astype(int), 0, n - 1)
    out = np.zeros((n, 3), dtype=np.float64)
    wsum = np.bincount(idx, weights=weights, minlength=n)
    for c in range(3):
        out[:, c] = np.bincount(idx, weights=weights * uq_rgb[:, c], minlength=n)
    nz = wsum > 0
    out[nz] /= wsum[nz, None]
    if not nz.any():
        return np.zeros((n, 3), dtype=np.uint8)
    known = np.flatnonzero(nz)
    for i in np.flatnonzero(~nz):
        out[i] = out[known[np.argmin(np.abs(known - i))]]
    return np.clip(out, 0, 255).astype(np.uint8)


def color_at(uq_rgb, uq_lab, weights, axis, t, window=0.04):
    """Representative RGB at normalized position t: the weighted mean of colors
    within +/-window of t, widening until something is found."""
    pos = normalize_positions(axis_positions(uq_lab, weights, axis), axis)
    for wdt in (window, window * 2, window * 4, 1.0):
        m = np.abs(pos - t) <= wdt
        if m.any() and weights[m].sum() > 0:
            return (uq_rgb[m] * weights[m, None]).sum(0) / weights[m].sum()
    return uq_rgb[0].astype(np.float64)


def assign_by_markers(uq_lab, weights, axis, marker_positions):
    """Each color joins the nearest marker along the axis (1-D Voronoi)."""
    pos = normalize_positions(axis_positions(uq_lab, weights, axis), axis)
    mk = np.asarray(marker_positions, dtype=np.float64)
    if len(mk) == 0:
        return np.zeros(len(pos), dtype=np.int32)
    return np.abs(pos[:, None] - mk[None, :]).argmin(axis=1).astype(np.int32)


def axis_quality(uq_lab, weights, axis):
    """How much color variance this axis actually carries. Shown in the UI so a
    user can see when a strip is hiding structure. Hue scores ~3% on both test
    images, which is exactly the case where a 1-D line misleads."""
    pos = axis_positions(uq_lab, weights, axis).astype(np.float64)
    if pos.std() < 1e-9:
        return 0.0
    mean = (uq_lab * weights[:, None]).sum(0) / weights.sum()
    total = float((((uq_lab - mean) ** 2).sum(1) * weights).sum())
    if total <= 0:
        return 0.0
    x = (pos - pos.mean()) / (pos.std() or 1.0)
    resid = 0.0
    for c in range(3):
        y = uq_lab[:, c]
        ybar = (weights * y).sum() / weights.sum()
        beta = float((weights * x * (y - ybar)).sum() / max((weights * x * x).sum(), 1e-9))
        resid += float((weights * (y - (ybar + beta * x)) ** 2).sum())
    return max(0.0, min(1.0, 1.0 - resid / total))
