"""sRGB -> CIE Lab (D65) conversion, vectorized."""

import numpy as np

_M_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_D65 = np.array([0.95047, 1.0, 1.08883])
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def srgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = linear @ _M_RGB_TO_XYZ.T
    xyz = xyz / _D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)

_M_XYZ_TO_RGB = np.linalg.inv(_M_RGB_TO_XYZ)

def lab_to_srgb(lab):
    """CIE Lab (D65) -> sRGB floats in [0, 255]. Inverse of srgb_to_lab."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def finv(f):
        f3 = f ** 3
        return np.where(f3 > _EPS, f3, (116.0 * f - 16.0) / _KAPPA)

    yr = np.where(L > _KAPPA * _EPS, ((L + 16.0) / 116.0) ** 3, L / _KAPPA)
    xyz = np.stack([finv(fx), yr, finv(fz)], axis=-1) * _D65
    linear = xyz @ _M_XYZ_TO_RGB.T
    srgb = np.where(linear > 0.0031308,
                    1.055 * np.maximum(linear, 0.0) ** (1 / 2.4) - 0.055,
                    12.92 * linear)
    return np.clip(srgb, 0.0, 1.0) * 255.0
