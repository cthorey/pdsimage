"""Pure projection math shared by the binary tables and the high-level Area.

These functions are a faithful port of the original ``lambert_window`` /
``cylindrical_window`` / ``_kp_func`` methods that were duplicated in both
``BinaryTable`` and ``Area``. The arithmetic is preserved verbatim (only the
hardcoded ``1734.4`` is replaced by :data:`MOON_RADIUS_KM`, an identical value)
so results match the historical Python 2 implementation.
"""

from __future__ import annotations

import numpy as np

from .constants import MOON_RADIUS_KM

Window = tuple[float, float, float, float]


def kp_func(lat: float, lon: float, lat0: float, long0: float) -> float:
    """Lambert azimuthal equal-area ``k'`` scaling factor (radians in)."""
    kp = 1.0 + np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - long0)
    return np.sqrt(2.0 / kp)


def lambert_window(radius: float, lat0: float, long0: float) -> Window:
    """Square Lambert azimuthal equal-area window around ``(lat0, long0)``.

    Args:
        radius: Radius of the window (km).
        lat0: Latitude of the center (degree).
        long0: Longitude of the center (degree).

    Returns:
        ``(longll, longtr, latll, lattr)`` corner coordinates in degree.
    """
    radius = radius * 360.0 / (np.pi * 2 * MOON_RADIUS_KM)
    radius = radius * np.pi / 180.0
    lat0 = lat0 * np.pi / 180.0
    long0 = long0 * np.pi / 180.0

    bot = kp_func(lat0 - radius, long0, lat0, long0)
    bot = bot * (np.cos(lat0) * np.sin(lat0 - radius) - np.sin(lat0) * np.cos(lat0 - radius))
    x = bot
    y = bot
    rho = np.sqrt(x**2 + y**2)
    c = 2.0 * np.arcsin(rho / 2.0)
    latll = np.arcsin(np.cos(c) * np.sin(lat0) + y * np.sin(c) * np.cos(lat0) / rho) * 180.0 / np.pi
    lon = long0 + np.arctan2(
        x * np.sin(c), rho * np.cos(lat0) * np.cos(c) - y * np.sin(lat0) * np.sin(c)
    )
    longll = lon * 180.0 / np.pi

    x = -bot
    y = -bot
    rho = np.sqrt(x**2 + y**2)
    c = 2.0 * np.arcsin(rho / 2.0)
    lattr = np.arcsin(np.cos(c) * np.sin(lat0) + y * np.sin(c) * np.cos(lat0) / rho) * 180.0 / np.pi
    lon = long0 + np.arctan2(
        x * np.sin(c), rho * np.cos(lat0) * np.cos(c) - y * np.sin(lat0) * np.sin(c)
    )
    longtr = lon * 180.0 / np.pi

    return longll, longtr, latll, lattr


def cylindrical_window(radius: float, lat0: float, long0: float) -> Window:
    """Cylindrical (equirectangular) window around ``(lat0, long0)``.

    Args:
        radius: Radius of the window (km).
        lat0: Latitude of the center (degree).
        long0: Longitude of the center (degree).

    Returns:
        ``(longll, longtr, latll, lattr)`` corner coordinates in degree.
    """
    radi = radius * 2 * np.pi / (2 * MOON_RADIUS_KM * np.pi)
    lamb0 = long0 * np.pi / 180.0
    phi0 = lat0 * np.pi / 180.0

    longll = -radi / np.cos(phi0) + lamb0
    latll = np.arcsin((-radi + np.sin(phi0) / np.cos(phi0)) * np.cos(phi0))
    if np.isnan(latll):
        latll = -90 * np.pi / 180.0
    longtr = radi / np.cos(phi0) + lamb0
    lattr = np.arcsin((radi + np.tan(phi0)) * np.cos(phi0))

    return (
        longll * 180 / np.pi,
        longtr * 180 / np.pi,
        latll * 180 / np.pi,
        lattr * 180 / np.pi,
    )


def deg_per_km(radius_km: float) -> float:
    """Angular size (degree) subtended by ``radius_km`` on the Moon surface."""
    return radius_km * 360.0 / (2 * np.pi * MOON_RADIUS_KM)
