"""Tile-naming tests for WacMap / LolaMap (no download)."""

from __future__ import annotations

from pdsimage.maps import LolaMap, WacMap


def test_lola_512_name_for_copernicus():
    # Copernicus: lon ~339.87, lat ~9.62 at 512 ppd -> a single NE tile.
    m = LolaMap(512, 270.0, 360.0, 0.0, 45.0)
    lon = m._format_lon(339.87)
    lat = m._format_lat(9.62)
    assert m._format_name_map(lon, lat) == "LDEM_512_00N_45N_270_360"


def test_lola_lowres_name():
    m = LolaMap(128, 10.0, 20.0, 10.0, 20.0)
    assert m._format_name_map(None, None) == "LDEM_128"


def test_wac_128_name():
    m = WacMap(128, 10.0, 20.0, 10.0, 20.0)
    lonc = m._format_lon(15.0)
    latc = m._format_lat(15.0)
    name = m._format_name_map(lonc, latc)
    assert name.startswith("WAC_GLOBAL_E")
    assert name.endswith("128P")


def test_wac_define_case_single_tile():
    # A small window well inside one 128 ppd tile -> case 1 path is selected.
    m = WacMap(128, 12.0, 14.0, 12.0, 14.0)
    assert m._map_center("long", 12.0) == m._map_center("long", 14.0)
    assert m._map_center("lat", 12.0) == m._map_center("lat", 14.0)
