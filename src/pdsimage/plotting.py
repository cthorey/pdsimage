"""Cartopy renderers for an :class:`~pdsimage.area.Area`.

Every function takes an ``Area`` and returns a :class:`matplotlib.figure.Figure`
(it does not call ``plt.show``), which makes them equally usable from a Jupyter
notebook and from the web backend (via :func:`fig_to_png`).

The original Basemap-based plots are reproduced on a spherical Moon globe
(:mod:`pdsimage.globe`). There are no coastlines or land features — this is the
Moon — and the Basemap ``drawmapscale`` is replaced by a simple km scale bar.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: safe for servers and notebooks alike
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

from .area import Area  # noqa: E402
from .globe import laea, plate_carree  # noqa: E402

Figure = matplotlib.figure.Figure


def _new_axis(area: Area, fig):
    """Create a Lambert-azimuthal axis covering the area's window."""
    lon_m, lon_M, lat_m, lat_M = area.window
    ax = fig.add_subplot(1, 1, 1, projection=laea(area.lon0, area.lat0))
    ax.set_extent([lon_m, lon_M, lat_m, lat_M], crs=plate_carree())
    ax.gridlines(draw_labels=True, color="0.6", linewidth=0.5)
    return ax


def _add_center(area: Area, ax) -> None:
    ax.scatter(
        area.lon0, area.lat0, s=200, marker="v", color="red", zorder=3, transform=plate_carree()
    )


def _add_scale_bar(area: Area, ax, bar_km: float = 10.0) -> None:
    """Draw a small km scale bar in the lower-left of a projected (meters) axis."""
    x0, x1, y0, y1 = ax.get_extent(crs=ax.projection)
    length = bar_km * 1000.0  # laea units are meters
    px, py = x0 + 0.08 * (x1 - x0), y0 + 0.08 * (y1 - y0)
    ax.plot([px, px + length], [py, py], color="k", lw=3, zorder=4, transform=ax.projection)
    ax.text(
        px + length / 2.0,
        py + 0.02 * (y1 - y0),
        f"{bar_km:.0f} km",
        ha="center",
        va="bottom",
        fontsize=12,
        transform=ax.projection,
        zorder=4,
    )


def lola_image(area: Area) -> Figure:
    """Topography map of the region (LOLA)."""
    fig = plt.figure(figsize=(10, 8))
    ax = _new_axis(area, fig)
    X, Y, Z = area.get_arrays("lola")
    mesh = ax.pcolormesh(X, Y, Z, cmap="gist_earth", shading="auto", transform=plate_carree())
    _add_center(area, ax)
    _add_scale_bar(area, ax)
    cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Topography (m)")
    return fig


def wac_image(area: Area) -> Figure:
    """Wide-angle camera image of the region (WAC)."""
    fig = plt.figure(figsize=(10, 8))
    ax = _new_axis(area, fig)
    X, Y, Z = area.get_arrays("wac")
    ax.pcolormesh(X, Y, Z, cmap=cm.gray, shading="auto", transform=plate_carree())
    _add_center(area, ax)
    _add_scale_bar(area, ax)
    return fig


def overlay(area: Area) -> Figure:
    """Topography (LOLA) overlaid on the wide-angle image (WAC)."""
    fig = plt.figure(figsize=(10, 8))
    ax = _new_axis(area, fig)

    Xw, Yw, Zw = area.get_arrays("wac")
    ax.pcolormesh(Xw, Yw, Zw, cmap=cm.gray, shading="auto", transform=plate_carree(), zorder=1)

    Xl, Yl, Zl = area.get_arrays("lola")
    cs = ax.contourf(
        Xl, Yl, Zl, 100, cmap="gist_earth", alpha=0.4, zorder=2, transform=plate_carree()
    )
    _add_center(area, ax)
    _add_scale_bar(area, ax)
    cb = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Topography (m)")
    return fig


def draw_profile(area: Area, coordinates, num_points: int = 500) -> Figure:
    """Render one or more elevation profiles next to a topography map.

    Args:
        area: The region of interest.
        coordinates: An iterable of ``(lon0, lon1, lat0, lat1)`` segments.
        num_points: Samples per profile.
    """
    coordinates = list(coordinates)
    fig = plt.figure(figsize=(18, max(1, len(coordinates)) * 5))
    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(len(coordinates), 4)

    for i, coordinate in enumerate(coordinates):
        ax_map = fig.add_subplot(gs[i, :2], projection=laea(area.lon0, area.lat0))
        lon_m, lon_M, lat_m, lat_M = area.window
        ax_map.set_extent([lon_m, lon_M, lat_m, lat_M], crs=plate_carree())
        ax_map.gridlines(draw_labels=True, color="0.6", linewidth=0.5)

        X, Y, Z = area.get_arrays("lola")
        ax_map.pcolormesh(X, Y, Z, cmap="gist_earth", shading="auto", transform=plate_carree())

        lon1, lon0, lat1, lat0 = coordinate
        ax_map.plot(
            [lon1, lon0], [lat1, lat0], "ro-", transform=plate_carree(), zorder=3
        )

        z = area.get_profile("lola", coordinate, num_points)
        ax_prof = fig.add_subplot(gs[i, 2:])
        dist = np.linspace(0, 1, len(z))
        ax_prof.plot(dist, z, lw=2, marker="o", ms=3)
        ax_prof.set_ylabel("Topographic profile (m)")
        ax_prof.set_xlabel("Normalized distance")
    return fig


def fig_to_png(fig: Figure, dpi: int = 100) -> bytes:
    """Render a figure to PNG bytes and close it."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
