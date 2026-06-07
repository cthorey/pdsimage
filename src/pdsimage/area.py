"""High-level region of interest on the lunar surface (data only).

``Area`` gathers the LOLA/WAC arrays for a square window centered on a point.
Rendering lives in :mod:`pdsimage.plotting` (functions taking an ``Area``),
keeping data extraction and plotting cleanly separated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage

from . import projections
from .constants import DEFAULT_PPD_LOLA, DEFAULT_PPD_WAC
from .maps import LolaMap, WacMap


class Area:
    """A square region of interest centered on ``(lon0, lat0)``.

    Args:
        lon0: Center longitude (0-360 degree).
        lat0: Center latitude (degree).
        size: Radius of the region of interest (km).
        root: Cache root for PDS files; defaults to the OS cache dir.
        allow_download: Whether missing tiles may be fetched from the network.

    Example:
        >>> area = Area(10, 10, 20)
        >>> X, Y, Z = area.get_arrays("lola")
    """

    def __init__(self, lon0, lat0, size, root=None, allow_download=True):
        self.root = Path(root) if root else None
        self.allow_download = allow_download
        self.lat0 = lat0
        self.lon0 = lon0
        self.ppdlola = DEFAULT_PPD_LOLA
        self.ppdwac = DEFAULT_PPD_WAC
        assert 0.0 < self.lon0 < 360.0, "Longitude has to span 0-360 !!!"
        self.change_window(size)

    def change_window(self, size_window: float) -> None:
        """Recompute the window for a new radius (km)."""
        self.size_window = size_window
        self.window = self.lambert_window(self.size_window, self.lat0, self.lon0)

    def lambert_window(self, radius, lat0, long0):
        """See :func:`pdsimage.projections.lambert_window`."""
        return projections.lambert_window(radius, lat0, long0)

    def cylindrical_window(self, radius, lat0, long0):
        """See :func:`pdsimage.projections.cylindrical_window`."""
        return projections.cylindrical_window(radius, lat0, long0)

    def get_arrays(self, type_img: str):
        """Return ``(X, Y, Z)`` arrays for ``"lola"`` or ``"wac"``."""
        kind = type_img.lower()
        if kind == "lola":
            return LolaMap(
                self.ppdlola, *self.window, root=self.root, allow_download=self.allow_download
            ).image()
        elif kind == "wac":
            return WacMap(
                self.ppdwac, *self.window, root=self.root, allow_download=self.allow_download
            ).image()
        raise ValueError('The img type has to be either "lola" or "wac"')

    def get_profile(self, img_type: str, coordinate, num_points: int):
        """Interpolate ``Z`` along the segment given by ``coordinate``.

        Args:
            img_type: ``"lola"`` or ``"wac"``.
            coordinate: ``(lon0, lon1, lat0, lat1)`` endpoints (degree).
            num_points: Number of samples along the profile.
        """
        lon0, lon1, lat0, lat1 = coordinate
        X, Y, Z = self.get_arrays(img_type)
        y0, x0 = np.argmin(np.abs(X[0, :] - lon0)), np.argmin(np.abs(Y[:, 0] - lat0))
        y1, x1 = np.argmin(np.abs(X[0, :] - lon1)), np.argmin(np.abs(Y[:, 0] - lat1))
        x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)
        return scipy.ndimage.map_coordinates(Z, np.vstack((x, y)))

    # -- convenience renderers (delegate to pdsimage.plotting, return a Figure) --
    # Plotting is imported lazily to avoid a circular import (plotting imports
    # Area). Unlike the original library these return a matplotlib Figure rather
    # than saving to disk; call ``fig.savefig(...)`` yourself.
    def lola_image(self):
        """Topography map of the region (see :func:`pdsimage.plotting.lola_image`)."""
        from . import plotting

        return plotting.lola_image(self)

    def wac_image(self):
        """WAC image of the region (see :func:`pdsimage.plotting.wac_image`)."""
        from . import plotting

        return plotting.wac_image(self)

    def overlay(self):
        """Topography over WAC (see :func:`pdsimage.plotting.overlay`)."""
        from . import plotting

        return plotting.overlay(self)

    def draw_profile(self, coordinates, num_points: int = 500):
        """Elevation profiles (see :func:`pdsimage.plotting.draw_profile`)."""
        from . import plotting

        return plotting.draw_profile(self, coordinates, num_points)
