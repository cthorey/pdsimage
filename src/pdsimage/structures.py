"""Named lunar structures backed by the bundled catalogs.

``Crater`` and ``Dome`` are thin ``Area`` subclasses that look a structure up
by name (or table index) and default the window to 80% of its diameter.
"""

from __future__ import annotations

from pathlib import Path

from .area import Area
from .catalogs import load_craters, load_domes, lookup


def _switchtype(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)


class _CatalogStructure(Area):
    """Shared loader for catalog-backed structures."""

    _csv_loader = None  # set by subclasses

    def __init__(self, ide, idx, root=None, allow_download=True):
        self.root = Path(root) if root else None
        self.allow_download = allow_download
        self.ppdlola = 512
        self.ppdwac = 128

        df = lookup(type(self)._csv_loader(), ide, idx)
        if len(df) == 0:
            raise ValueError(
                f"The tuple ({ide},{idx}) does not correspond to any structure in the dataset."
            )
        for f in df.columns:
            setattr(self, f, _switchtype(df[f].iloc[0]))

        assert 0.0 < self.lon0 < 360.0, "Longitude has to span 0-360 !!!"
        self.name = df.name.iloc[0]
        self.change_window(0.8 * self.diameter)


class Crater(_CatalogStructure):
    """An impact crater from the bundled crater catalog.

    Args:
        ide: ``"name"`` or ``"index"`` (the column to match on).
        idx: The crater name or table index.

    Example:
        >>> from pdsimage.plotting import overlay
        >>> overlay(Crater("name", "Copernicus"))
    """

    _csv_loader = staticmethod(load_craters)


class Dome(_CatalogStructure):
    """A low-slope dome from the bundled dome catalog.

    Example:
        >>> Dome("name", "M13")
    """

    _csv_loader = staticmethod(load_domes)
