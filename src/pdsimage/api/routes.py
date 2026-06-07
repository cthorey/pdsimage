"""API routes: catalog browsing, image rendering and elevation profiles."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import scipy.ndimage
from fastapi import APIRouter, HTTPException, Query, Response
from starlette.concurrency import run_in_threadpool

from .. import plotting
from ..area import Area
from ..catalogs import load_craters, load_domes, search
from ..download import DownloadError
from .schemas import ProfileRequest, ProfileResponse, RenderRequest

router = APIRouter()

_RENDERERS = {
    "lola": plotting.lola_image,
    "wac": plotting.wac_image,
    "overlay": plotting.overlay,
}


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/catalog/craters")
def craters(q: str | None = Query(None), limit: int = Query(200, ge=1, le=2000)) -> list[dict]:
    return search(load_craters(), q, limit)


@router.get("/catalog/domes")
def domes(q: str | None = Query(None), limit: int = Query(200, ge=1, le=2000)) -> list[dict]:
    return search(load_domes(), q, limit)


def _render_png(req: RenderRequest) -> tuple[bytes, tuple]:
    area = Area(req.lon0, req.lat0, req.size_km)
    if req.ppd is not None:
        # ppd applies to the relevant product; overlay uses both defaults.
        if req.img_type in ("lola", "overlay"):
            area.ppdlola = req.ppd
        if req.img_type in ("wac", "overlay"):
            area.ppdwac = req.ppd
    fig = _RENDERERS[req.img_type](area)
    return plotting.fig_to_png(fig), tuple(area.window)


@router.post("/render")
async def render(req: RenderRequest) -> Response:
    try:
        png, window = await run_in_threadpool(_render_png, req)
    except DownloadError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (AssertionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # window = (lon_ll, lon_tr, lat_ll, lat_tr); used by the client profile tool.
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Window": json.dumps(list(window))},
    )


# Profiles use a single global LOLA file (fast, no multi-GB tiles). We snap the
# requested resolution to one of these and cap it for snappy interactive use.
_PROFILE_RESOLUTIONS = [4, 16, 64]


def _profile_ppd(ppd: int | None) -> int:
    if not ppd:
        return 16
    return min(_PROFILE_RESOLUTIONS, key=lambda r: abs(r - ppd))


@lru_cache(maxsize=16)
def _cached_lola_arrays(lon0: float, lat0: float, size_km: float, ppd: int):
    """Cache extracted LOLA arrays so repeated profiles on a region are instant."""
    area = Area(lon0, lat0, size_km)
    area.ppdlola = ppd
    return area.get_arrays("lola")


def _profile(req: ProfileRequest) -> ProfileResponse:
    ppd = _profile_ppd(req.ppd)
    X, Y, Z = _cached_lola_arrays(req.lon0, req.lat0, req.size_km, ppd)
    # Mirrors Area.get_profile, but on cached arrays at an interactive resolution.
    y0, x0 = np.argmin(np.abs(X[0, :] - req.p1.lon)), np.argmin(np.abs(Y[:, 0] - req.p1.lat))
    y1, x1 = np.argmin(np.abs(X[0, :] - req.p2.lon)), np.argmin(np.abs(Y[:, 0] - req.p2.lat))
    xs, ys = np.linspace(x0, x1, req.num_points), np.linspace(y0, y1, req.num_points)
    z = scipy.ndimage.map_coordinates(Z, np.vstack((xs, ys)))
    dist = np.linspace(0.0, 1.0, len(z))
    return ProfileResponse(distance=dist.tolist(), z=np.asarray(z, dtype=float).tolist())


@router.post("/profile", response_model=ProfileResponse)
async def profile(req: ProfileRequest) -> ProfileResponse:
    try:
        return await run_in_threadpool(_profile, req)
    except DownloadError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (AssertionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
