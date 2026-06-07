"""Projection-math tests: geometric properties + regression pins."""

from __future__ import annotations

import numpy as np
import pytest

from pdsimage import projections


def test_lambert_window_symmetric_at_equator():
    # Centered on the equator the window is symmetric in lon and lat.
    longll, longtr, latll, lattr = projections.lambert_window(50.0, 0.0, 180.0)
    assert longll < 180.0 < longtr
    assert latll < 0.0 < lattr
    assert longtr - 180.0 == pytest.approx(180.0 - longll, rel=1e-9)
    assert lattr == pytest.approx(-latll, rel=1e-9)


def test_cylindrical_window_symmetric_at_equator():
    longll, longtr, latll, lattr = projections.cylindrical_window(50.0, 0.0, 180.0)
    assert longtr - 180.0 == pytest.approx(180.0 - longll, rel=1e-9)
    assert lattr == pytest.approx(-latll, rel=1e-9)


def test_deg_per_km_roundtrip():
    # 2*pi*R km == 360 degrees.
    from pdsimage.constants import MOON_RADIUS_KM

    assert projections.deg_per_km(2 * np.pi * MOON_RADIUS_KM) == pytest.approx(360.0)


def test_lambert_window_regression():
    # Regression pin (values from the faithful port); guards future edits.
    longll, longtr, latll, lattr = projections.lambert_window(100.0, 9.62, 339.92)
    assert longll == pytest.approx(336.5977853028346, abs=1e-6)
    assert longtr == pytest.approx(343.30779382945184, abs=1e-6)
    assert latll == pytest.approx(6.301861682734138, abs=1e-6)
    assert lattr == pytest.approx(12.905780569378784, abs=1e-6)
