"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..constants import LOLA_RESOLUTIONS, WAC_RESOLUTIONS

ImgType = str  # "lola" | "wac" | "overlay"


class RenderRequest(BaseModel):
    lon0: float = Field(..., gt=0, lt=360, description="Center longitude (0-360).")
    lat0: float = Field(..., ge=-90, le=90, description="Center latitude.")
    size_km: float = Field(..., gt=0, description="Window radius in km.")
    img_type: str = Field("lola", description="lola | wac | overlay.")
    ppd: int | None = Field(None, description="Pixels-per-degree resolution.")

    @field_validator("img_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in ("lola", "wac", "overlay"):
            raise ValueError("img_type must be one of lola, wac, overlay")
        return v

    @field_validator("ppd")
    @classmethod
    def _check_ppd(cls, v, info):
        if v is None:
            return v
        allowed = set(LOLA_RESOLUTIONS) | set(WAC_RESOLUTIONS)
        if v not in allowed:
            raise ValueError(f"ppd must be one of {sorted(allowed)}")
        return v


class Point(BaseModel):
    lon: float = Field(..., gt=0, lt=360)
    lat: float = Field(..., ge=-90, le=90)


class ProfileRequest(BaseModel):
    lon0: float = Field(..., gt=0, lt=360)
    lat0: float = Field(..., ge=-90, le=90)
    size_km: float = Field(..., gt=0)
    p1: Point
    p2: Point
    num_points: int = Field(500, gt=1, le=5000)
    img_type: str = Field("lola")
    ppd: int | None = Field(None, description="Preferred resolution; snapped to a fast LOLA grid.")


class ProfileResponse(BaseModel):
    distance: list[float]
    z: list[float]
