"""Cartopy Moon globe + a tiny render to PNG (headless)."""

from __future__ import annotations

import numpy as np

from pdsimage.constants import MOON_RADIUS_M
from pdsimage.globe import MOON_GLOBE, laea, plate_carree


def test_globe_radius_is_moon():
    assert MOON_GLOBE.semimajor_axis == MOON_RADIUS_M
    assert MOON_GLOBE.semiminor_axis == MOON_RADIUS_M


def test_projections_instantiate():
    assert plate_carree() is not None
    assert laea(180.0, 0.0) is not None


def test_small_pcolormesh_renders_png():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pdsimage.plotting import fig_to_png

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=laea(10.0, 10.0))
    ax.set_extent([8, 12, 8, 12], crs=plate_carree())
    x, y = np.meshgrid(np.linspace(8, 12, 5), np.linspace(8, 12, 5))
    z = np.random.rand(5, 5)
    ax.pcolormesh(x, y, z, shading="auto", transform=plate_carree())
    png = fig_to_png(fig)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
