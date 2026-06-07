"""Catalog loading, lookup and Crater/Dome construction (no network)."""

from __future__ import annotations

import pytest

from pdsimage import Crater, Dome
from pdsimage.catalogs import load_craters, load_domes, lookup, search


def test_crater_catalog_columns():
    df = load_craters()
    for col in ("name", "lat0", "lon0", "diameter", "type", "radius", "index"):
        assert col in df.columns


def test_dome_catalog_columns():
    df = load_domes()
    for col in ("lat0", "lon0", "name", "diameter", "thickness", "index"):
        assert col in df.columns


def test_search_filters_by_name():
    rows = search(load_craters(), q="copernicus")
    assert any("copernicus" in str(r["name"]).lower() for r in rows)


def test_lookup_by_name():
    df = lookup(load_craters(), "name", "Copernicus")
    assert len(df) == 1


def test_crater_construction_sets_window():
    c = Crater("name", "Copernicus")
    assert c.name == "Copernicus"
    assert 0.0 < c.lon0 < 360.0
    # window defaults to 80% of diameter
    assert c.size_window == pytest.approx(0.8 * c.diameter)
    assert len(c.window) == 4


def test_dome_construction():
    d = Dome("name", "M13")
    assert d.name == "M13"
    assert d.size_window == pytest.approx(0.8 * d.diameter)
