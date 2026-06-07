"""Cartopy coordinate reference systems built on a spherical Moon.

Cartopy ships Earth-based CRSs by default. For the Moon we build every
projection on a custom :class:`cartopy.crs.Globe` whose radius matches the
value used throughout the library.
"""

from __future__ import annotations

import cartopy.crs as ccrs

from .constants import MOON_RADIUS_M

#: A perfectly spherical Moon globe.
MOON_GLOBE = ccrs.Globe(
    semimajor_axis=MOON_RADIUS_M,
    semiminor_axis=MOON_RADIUS_M,
    ellipse=None,
)


def plate_carree() -> ccrs.PlateCarree:
    """Lon/lat CRS on the Moon globe (the CRS the raw arrays live in)."""
    return ccrs.PlateCarree(globe=MOON_GLOBE)


def laea(lon0: float, lat0: float) -> ccrs.LambertAzimuthalEqualArea:
    """Lambert azimuthal equal-area CRS centered on ``(lon0, lat0)``."""
    return ccrs.LambertAzimuthalEqualArea(
        central_longitude=lon0,
        central_latitude=lat0,
        globe=MOON_GLOBE,
    )
