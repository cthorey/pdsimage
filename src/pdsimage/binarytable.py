"""Read binary PDS image tiles (LOLA topography, LROC WAC imagery).

A faithful Python 3 port of the original ``BinaryTable``. The numerical
coordinate/pixel conversions and the extraction algorithm are preserved
verbatim; only Python-2-isms (``map`` as list, ``np.fromstring``,
``.iteritems``) and the interactive download path were modernized.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pvl

from . import projections
from .download import ensure_tile


class BinaryTable:
    """Read an LRO LOLA or LROC WAC binary PDS tile.

    Args:
        fname: Tile name without extension (e.g. ``LDEM_512_00N_45N_270_360``
            or ``WAC_GLOBAL_E300N0450_128P``).
        root: Cache root where PDS files live; defaults to the OS cache dir.
        allow_download: Whether missing tiles may be fetched from the network.

    Note:
        Only cylindrical projections are supported
        (WAC ``EQUIRECTANGULAR`` / LOLA ``"SIMPLE``).
    """

    def __init__(self, fname: str, root: str | Path | None = None, allow_download: bool = True):
        self.fname = fname.upper()
        spec = ensure_tile(self.fname, Path(root) if root else None, allow_download)
        self.grid = spec.grid
        self.img = spec.img
        self.lbl = spec.lbl
        self._load_info_lbl()
        assert self.MAP_PROJECTION_TYPE in ['"SIMPLE', "EQUIRECTANGULAR"], (
            "Only cylindrical projection is possible - %s NOT IMPLEMENTED"
            % (self.MAP_PROJECTION_TYPE)
        )

    # ------------------------------------------------------------------ header
    def _load_info_lbl(self) -> None:
        """Populate header attributes from the WAC ``.IMG`` or LOLA ``.LBL``."""
        if self.grid == "WAC":
            label = pvl.load(str(self.img))
            for key, val in label.items():
                if isinstance(val, Mapping):
                    for k2, v2 in val.items():
                        try:
                            setattr(self, k2, v2.value)
                        except AttributeError:
                            setattr(self, k2, v2)
                else:
                    setattr(self, key, val)
            self.start_byte = self.RECORD_BYTES
            self.bytesize = 4
            self.projection = str(label["IMAGE_MAP_PROJECTION"]["MAP_PROJECTION_TYPE"])
            self.dtype = np.float32
        else:
            with open(self.lbl) as f:
                for line in f:
                    attr = [a.strip() for a in line.split("=")]
                    if len(attr) == 2:
                        setattr(self, attr[0], attr[1].split(" ")[0])
            self.start_byte = 0
            self.bytesize = 2
            self.projection = ""
            self.dtype = np.int16

    # -------------------------------------------------------------- coordinates
    def lat_id(self, line: int) -> float:
        """Latitude (degree) of a given image line."""
        if self.grid == "WAC":
            lat = (
                (1 + self.LINE_PROJECTION_OFFSET - line)
                * self.MAP_SCALE
                * 1e-3
                / self.A_AXIS_RADIUS
            )
            return lat * 180 / np.pi
        lat = float(self.CENTER_LATITUDE) - (
            line - float(self.LINE_PROJECTION_OFFSET) - 1
        ) / float(self.MAP_RESOLUTION)
        return lat

    def long_id(self, sample: int) -> float:
        """Longitude (degree) of a given sample on a line."""
        if self.grid == "WAC":
            lon = self.CENTER_LONGITUDE + (sample - self.SAMPLE_PROJECTION_OFFSET - 1) * (
                self.MAP_SCALE * 1e-3 / (self.A_AXIS_RADIUS * np.cos(self.CENTER_LATITUDE * np.pi / 180.0))
            )
            return lon * 180 / np.pi
        lon = float(self.CENTER_LONGITUDE) + (
            sample - float(self.SAMPLE_PROJECTION_OFFSET) - 1
        ) / float(self.MAP_RESOLUTION)
        return lon

    def _control_sample(self, sample):
        if sample > float(self.SAMPLE_LAST_PIXEL):
            return int(self.SAMPLE_LAST_PIXEL)
        elif sample < float(self.SAMPLE_FIRST_PIXEL):
            return int(self.SAMPLE_FIRST_PIXEL)
        return sample

    def sample_id(self, lon: float):
        """Sample number for a given longitude (degree)."""
        if self.grid == "WAC":
            sample = np.rint(
                float(self.SAMPLE_PROJECTION_OFFSET)
                + 1.0
                + (lon * np.pi / 180.0 - float(self.CENTER_LONGITUDE))
                * self.A_AXIS_RADIUS
                * np.cos(self.CENTER_LATITUDE * np.pi / 180.0)
                / (self.MAP_SCALE * 1e-3)
            )
        else:
            sample = (
                np.rint(
                    float(self.SAMPLE_PROJECTION_OFFSET)
                    + float(self.MAP_RESOLUTION) * (lon - float(self.CENTER_LONGITUDE))
                )
                + 1
            )
        return self._control_sample(sample)

    def _control_line(self, line):
        if line > float(self.LINE_LAST_PIXEL):
            return int(self.LINE_LAST_PIXEL)
        elif line < float(self.LINE_FIRST_PIXEL):
            return int(self.LINE_FIRST_PIXEL)
        return line

    def line_id(self, lat: float):
        """Line number for a given latitude (degree)."""
        if self.grid == "WAC":
            line = np.rint(
                1.0
                + self.LINE_PROJECTION_OFFSET
                - self.A_AXIS_RADIUS * np.pi * lat / (self.MAP_SCALE * 1e-3 * 180)
            )
        else:
            line = (
                np.rint(
                    float(self.LINE_PROJECTION_OFFSET)
                    - float(self.MAP_RESOLUTION) * (lat - float(self.CENTER_LATITUDE))
                )
                + 1
            )
        return self._control_line(line)

    # --------------------------------------------------------------- extraction
    def array(self, size_chunk: int, start: int, bytesize: int) -> np.ndarray:
        """Read ``size_chunk`` values starting at value index ``start``."""
        with open(self.img, "rb") as f1:
            f1.seek(self.start_byte + start * self.bytesize)
            data = f1.read(size_chunk * self.bytesize)
            Z = np.frombuffer(data, dtype=self.dtype, count=size_chunk)
            if self.grid == "LOLA":
                return Z * float(self.SCALING_FACTOR)
            return Z

    def extract_all(self):
        """Extract the whole tile as ``(X, Y, Z)`` arrays (degree, degree, value)."""
        sample_min, sample_max = (int(self.SAMPLE_FIRST_PIXEL), int(self.SAMPLE_LAST_PIXEL))
        line_min, line_max = (int(self.LINE_FIRST_PIXEL), int(self.LINE_LAST_PIXEL))

        X = np.array([self.long_id(s) for s in range(sample_min, sample_max + 1, 1)])
        Y = np.array([self.lat_id(line) for line in range(line_min, line_max + 1, 1)])
        Z = None
        for i, line in enumerate(range(int(line_min), int(line_max) + 1)):
            start = (line - 1) * int(self.SAMPLE_LAST_PIXEL) + sample_min
            chunk_size = int(sample_max - sample_min)
            Za = self.array(chunk_size, start, self.bytesize)
            Z = Za if i == 0 else np.vstack((Z, Za))

        X, Y = np.meshgrid(X, Y)
        return X, Y, Z

    def extract_grid(self, longmin: float, longmax: float, latmin: float, latmax: float):
        """Extract a lon/lat window as ``(X, Y, Z)`` arrays."""
        sample_min, sample_max = (int(self.sample_id(longmin)), int(self.sample_id(longmax)))
        line_min, line_max = (int(self.line_id(latmax)), int(self.line_id(latmin)))
        X = np.array([self.long_id(s) for s in range(sample_min, sample_max, 1)])
        Y = np.array([self.lat_id(line) for line in range(line_min, line_max + 1, 1)])

        Z = None
        for i, line in enumerate(range(int(line_min), int(line_max) + 1)):
            start = (line - 1) * int(self.SAMPLE_LAST_PIXEL) + sample_min
            chunk_size = int(sample_max - sample_min)
            Za = self.array(chunk_size, start, self.bytesize)
            Z = Za if i == 0 else np.vstack((Z, Za))

        X, Y = np.meshgrid(X, Y)
        return X, Y, Z

    def boundary(self):
        """Return ``(west_lon, east_lon, min_lat, max_lat)`` integer bounds."""
        return (
            int(self.WESTERNMOST_LONGITUDE),
            int(self.EASTERNMOST_LONGITUDE),
            int(self.MINIMUM_LATITUDE),
            int(self.MAXIMUM_LATITUDE),
        )

    # ------------------------------------------------------------- projections
    def lambert_window(self, radius, lat0, long0):
        """See :func:`pdsimage.projections.lambert_window`."""
        return projections.lambert_window(radius, lat0, long0)

    def cylindrical_window(self, radius, lat0, long0):
        """See :func:`pdsimage.projections.cylindrical_window`."""
        return projections.cylindrical_window(radius, lat0, long0)
