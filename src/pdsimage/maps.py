"""Assemble a lon/lat window from one or more PDS tiles.

``WacMap`` / ``LolaMap`` figure out which tile(s) cover a requested window and
stitch them together. Four cases are handled: the window fits in one tile, it
straddles a longitude boundary (2 tiles), a latitude boundary (2 tiles), or
both (4 tiles). Tile-naming and stitching logic is a faithful port of the
original; only ``map``-as-list constructs were modernized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .binarytable import BinaryTable
from .constants import LOLA_RESOLUTIONS, WAC_RESOLUTIONS


class WacMap:
    """Build an LROC WAC GLOBAL window of a given resolution.

    Example:
        >>> X, Y, Z = WacMap(128, 10, 20, 10, 20).image()
    """

    implemented_res = WAC_RESOLUTIONS

    def __init__(self, ppd, lonm, lonM, latm, latM, root=None, allow_download=True):
        self.root = Path(root) if root else None
        self.allow_download = allow_download
        self.ppd = ppd
        self.lonm = lonm
        self.lonM = lonM
        self.latm = latm
        self.latM = latM
        self._control_longitude()
        self._confirm_resolution(self.implemented_res)

    def _binary(self, name):
        return BinaryTable(name, self.root, self.allow_download)

    def _control_longitude(self):
        if self.lonm < 0.0:
            self.lonm = 360.0 + self.lonm
        if self.lonM < 0.0:
            self.lonM = 360.0 + self.lonM
        if self.lonm > 360.0:
            self.lonm = self.lonm - 360.0
        if self.lonM > 360.0:
            self.lonM = self.lonM - 360.0

    def _confirm_resolution(self, implemented_res):
        assert self.ppd in implemented_res, (
            " Resolution %d ppd not implemented yet.\n"
            " Consider using one of the implemented resolutions %s"
            % (self.ppd, ", ".join([str(f) + " ppd" for f in implemented_res]))
        )
        if self.ppd == 256:
            assert (np.abs(self.latM) < 60) and (np.abs(self.latm) < 60), (
                "This resolution is available in cylindrical geometry only for -60<latitude<60"
            )

    def _map_center(self, coord, val):
        if self.ppd in [4, 8, 16, 32, 64]:
            res = {"lat": 0, "long": 360}
            return res[coord] / 2.0
        elif self.ppd in [128]:
            res = {"lat": 90, "long": 90}
            return (val // res[coord] + 1) * res[coord] - res[coord] / 2.0
        elif self.ppd in [256]:
            res = {"lat": 60, "long": 90}
            return (val // res[coord] + 1) * res[coord] - res[coord] / 2.0

    def _define_case(self):
        lonBool = self._map_center("long", self.lonM) != self._map_center("long", self.lonm)
        latBool = self._map_center("lat", self.latM) != self._map_center("lat", self.latm)

        if not lonBool and not latBool:
            return self._cas_1()
        elif lonBool and not latBool:
            return self._cas_2()
        elif not lonBool and latBool:
            return self._cas_3()
        return self._cas_4()

    def _format_lon(self, lon):
        lonf = self._map_center("long", lon)
        st = str(lonf).split(".")
        return "".join(("{0:0>3}".format(st[0]), st[1]))

    def _format_lat(self, lat):
        if self.ppd in [4, 8, 16, 32, 64]:
            return "000N"
        elif self.ppd in [128]:
            return "450S" if lat < 0 else "450N"

    def _format_name_map(self, lonc, latc):
        return "_".join(["WAC", "GLOBAL"] + ["E" + latc + lonc, "{0:0>3}".format(self.ppd) + "P"])

    def _cas_1(self):
        lonc = self._format_lon(self.lonm)
        latc = self._format_lat(self.latm)
        img_map = self._binary(self._format_name_map(lonc, latc))
        return img_map.extract_grid(self.lonm, self.lonM, self.latm, self.latM)

    def _cas_2(self):
        lonc_left = self._format_lon(self.lonm)
        lonc_right = self._format_lon(self.lonM)
        latc = self._format_lat(self.latm)

        img_left = self._binary(self._format_name_map(lonc_left, latc))
        X_left, Y_left, Z_left = img_left.extract_grid(
            self.lonm, float(img_left.EASTERNMOST_LONGITUDE), self.latm, self.latM
        )
        img_right = self._binary(self._format_name_map(lonc_right, latc))
        X_right, Y_right, Z_right = img_right.extract_grid(
            float(img_right.WESTERNMOST_LONGITUDE), self.lonM, self.latm, self.latM
        )
        return (
            np.hstack((X_left, X_right)),
            np.hstack((Y_left, Y_right)),
            np.hstack((Z_left, Z_right)),
        )

    def _cas_3(self):
        lonc = self._format_lon(self.lonm)
        latc_top = self._format_lat(self.latM)
        latc_bot = self._format_lat(self.latm)

        img_top = self._binary(self._format_name_map(lonc, latc_top))
        X_top, Y_top, Z_top = img_top.extract_grid(
            self.lonm, self.lonM, float(img_top.MINIMUM_LATITUDE), self.latM
        )
        img_bottom = self._binary(self._format_name_map(lonc, latc_bot))
        X_bottom, Y_bottom, Z_bottom = img_bottom.extract_grid(
            self.lonm, self.lonM, self.latm, float(img_bottom.MAXIMUM_LATITUDE)
        )
        return (
            np.vstack((X_top, X_bottom)),
            np.vstack((Y_top, Y_bottom)),
            np.vstack((Z_top, Z_bottom)),
        )

    def _cas_4(self):
        lonc_left = self._format_lon(self.lonm)
        lonc_right = self._format_lon(self.lonM)
        latc_top = self._format_lat(self.latM)
        latc_bot = self._format_lat(self.latm)

        img_00 = self._binary(self._format_name_map(lonc_left, latc_top))
        X_00, Y_00, Z_00 = img_00.extract_grid(
            self.lonm, float(img_00.EASTERNMOST_LONGITUDE), float(img_00.MINIMUM_LATITUDE), self.latM
        )
        img_01 = self._binary(self._format_name_map(lonc_right, latc_top))
        X_01, Y_01, Z_01 = img_01.extract_grid(
            float(img_01.WESTERNMOST_LONGITUDE), self.lonM, float(img_01.MINIMUM_LATITUDE), self.latM
        )
        img_10 = self._binary(self._format_name_map(lonc_left, latc_bot))
        X_10, Y_10, Z_10 = img_10.extract_grid(
            self.lonm, float(img_10.EASTERNMOST_LONGITUDE), self.latm, float(img_10.MAXIMUM_LATITUDE)
        )
        img_11 = self._binary(self._format_name_map(lonc_right, latc_bot))
        X_11, Y_11, Z_11 = img_11.extract_grid(
            float(img_11.WESTERNMOST_LONGITUDE), self.lonM, self.latm, float(img_11.MAXIMUM_LATITUDE)
        )

        X_new = np.vstack((np.hstack((X_00, X_01)), np.hstack((X_10, X_11))))
        Y_new = np.vstack((np.hstack((Y_00, Y_01)), np.hstack((Y_10, Y_11))))
        Z_new = np.vstack((np.hstack((Z_00, Z_01)), np.hstack((Z_10, Z_11))))
        return X_new, Y_new, Z_new

    def image(self):
        """Return ``(X, Y, Z)`` arrays covering the requested window."""
        return self._define_case()


class LolaMap(WacMap):
    """Build an LRO LOLA window of a given resolution.

    Example:
        >>> X, Y, Z = LolaMap(512, 10, 20, 10, 20).image()
    """

    implemented_res = LOLA_RESOLUTIONS

    def _confirm_resolution(self, implemented_res):
        assert self.ppd in implemented_res, (
            " Resolution %d ppd not implemented yet.\n"
            " Consider using one of the implemented resolutions %s"
            % (self.ppd, ", ".join([str(f) + " ppd" for f in implemented_res]))
        )

    def _map_center(self, coord, val):
        if self.ppd in [4, 16, 64, 128]:
            res = {"lat": 0, "long": 360}
            return res[coord] / 2.0
        elif self.ppd in [256]:
            res = {"lat": 90, "long": 180}
            c = (val // res[coord] + 1) * res[coord]
            return c - res[coord], c
        elif self.ppd in [512]:
            res = {"lat": 45, "long": 90}
            c = (val // res[coord] + 1) * res[coord]
            return c - res[coord], c
        elif self.ppd in [1024]:
            res = {"lat": 15, "long": 30}
            c = (val // res[coord] + 1) * res[coord]
            return c - res[coord], c

    def _format_lon(self, lon):
        if self.ppd in [4, 16, 64, 128]:
            return None
        return ["{0:0>3}".format(int(x)) for x in self._map_center("long", lon)]

    def _format_lat(self, lat):
        if self.ppd in [4, 16, 64, 128]:
            return None
        if lat < 0:
            return ["{0:0>2}".format(int(np.abs(x))) + "S" for x in self._map_center("lat", lat)]
        return ["{0:0>2}".format(int(x)) + "N" for x in self._map_center("lat", lat)]

    def _format_name_map(self, lon, lat):
        if self.ppd in [4, 16, 64, 128]:
            return "_".join(["LDEM", str(self.ppd)])
        elif self.ppd in [512]:
            return "_".join(["LDEM", str(self.ppd), lat[0], lat[1], lon[0], lon[1]])
        raise NotImplementedError(
            f"LOLA tile naming for ppd={self.ppd} is not implemented; use 4/16/64/128/512."
        )
