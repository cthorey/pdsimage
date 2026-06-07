"""API tests using FastAPI's TestClient; data extraction is monkeypatched."""

from __future__ import annotations

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pdsimage import area as area_mod  # noqa: E402
from pdsimage.api.app import app  # noqa: E402

client = TestClient(app)


def _fake_arrays(self, type_img):
    lon = np.linspace(self.window[0], self.window[1], 8)
    lat = np.linspace(self.window[2], self.window[3], 8)
    X, Y = np.meshgrid(lon, lat)
    Z = np.hypot(X - self.lon0, Y - self.lat0)
    return X, Y, Z


@pytest.fixture(autouse=True)
def patch_arrays(monkeypatch):
    monkeypatch.setattr(area_mod.Area, "get_arrays", _fake_arrays)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_catalog_craters_search():
    r = client.get("/catalog/craters", params={"q": "Copernicus"})
    assert r.status_code == 200
    assert any("copernicus" in str(row["name"]).lower() for row in r.json())


def test_render_returns_png():
    r = client.post(
        "/render",
        json={"lon0": 339.92, "lat0": 9.62, "size_km": 100, "img_type": "lola", "ppd": 512},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_validation_error():
    r = client.post("/render", json={"lon0": 400, "lat0": 9.62, "size_km": 100})
    assert r.status_code == 422  # pydantic validation


def test_profile_ppd_snapping():
    from pdsimage.api.routes import _profile_ppd

    assert _profile_ppd(None) == 16  # default
    assert _profile_ppd(4) == 4
    assert _profile_ppd(16) == 16
    assert _profile_ppd(128) == 64  # capped for speed
    assert _profile_ppd(512) == 64  # never the multi-GB tile


def test_profile_returns_series():
    r = client.post(
        "/profile",
        json={
            "lon0": 339.92,
            "lat0": 9.62,
            "size_km": 100,
            "p1": {"lon": 339.5, "lat": 9.4},
            "p2": {"lon": 340.3, "lat": 9.8},
            "num_points": 50,
            "img_type": "lola",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["z"]) == 50
    assert len(body["distance"]) == 50
