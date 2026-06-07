"""Coordinate round-trips and binary reading for BinaryTable (no network)."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import make_lola_table


@pytest.mark.parametrize("lat", [-30.0, -5.0, 0.0, 12.5, 40.0])
def test_lola_lat_line_roundtrip(lola_table, lat):
    line = lola_table.line_id(lat)
    back = lola_table.lat_id(line)
    assert back == pytest.approx(lat, abs=0.25)  # 4 ppd -> 0.25 deg grid


@pytest.mark.parametrize("lon", [10.0, 90.0, 180.5, 300.0])
def test_lola_lon_sample_roundtrip(lola_table, lon):
    sample = lola_table.sample_id(lon)
    back = lola_table.long_id(sample)
    assert back == pytest.approx(lon, abs=0.25)


def test_array_applies_scaling_factor(tmp_path):
    raw = np.array([0, 2, 4, 6, 8], dtype=np.int16)
    bt = make_lola_table(tmp_path, values=raw)
    out = bt.array(len(raw), 0, bt.bytesize)
    # SCALING_FACTOR is "0.5".
    np.testing.assert_allclose(out, raw * 0.5)


def test_array_seek_offset(tmp_path):
    raw = np.arange(10, dtype=np.int16)
    bt = make_lola_table(tmp_path, values=raw)
    out = bt.array(3, 4, bt.bytesize)  # values at index 4,5,6
    np.testing.assert_allclose(out, np.array([4, 5, 6]) * 0.5)
