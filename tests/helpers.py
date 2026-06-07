"""Test helpers: build BinaryTable instances without any network IO."""

from __future__ import annotations

import numpy as np

from pdsimage.binarytable import BinaryTable


def make_lola_table(tmp_path, values: np.ndarray | None = None) -> BinaryTable:
    """A LOLA BinaryTable backed by a tiny synthetic .IMG, no download."""
    bt = object.__new__(BinaryTable)
    bt.fname = "LDEM_TEST"
    bt.grid = "LOLA"
    bt.start_byte = 0
    bt.bytesize = 2
    bt.dtype = np.int16
    bt.SCALING_FACTOR = "0.5"
    # Header for a simple cylindrical grid: 4 ppd, centered at (0, 0).
    bt.MAP_RESOLUTION = "4"
    bt.CENTER_LATITUDE = "0"
    bt.CENTER_LONGITUDE = "0"
    # A realistic line offset (half a 4 ppd global height) keeps line numbers
    # positive for the latitudes exercised in the tests.
    bt.LINE_PROJECTION_OFFSET = "360"
    bt.SAMPLE_PROJECTION_OFFSET = "0"
    bt.SAMPLE_FIRST_PIXEL = "1"
    bt.SAMPLE_LAST_PIXEL = "100000"
    bt.LINE_FIRST_PIXEL = "1"
    bt.LINE_LAST_PIXEL = "100000"
    bt.WESTERNMOST_LONGITUDE = "0"
    bt.EASTERNMOST_LONGITUDE = "360"
    bt.MINIMUM_LATITUDE = "-90"
    bt.MAXIMUM_LATITUDE = "90"
    bt.MAP_PROJECTION_TYPE = '"SIMPLE'

    if values is None:
        values = np.arange(10, dtype=np.int16)
    img = tmp_path / "LDEM_TEST.IMG"
    img.write_bytes(values.astype(np.int16).tobytes())
    bt.img = img
    return bt
