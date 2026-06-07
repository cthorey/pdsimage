"""FastAPI backend for pdsimage (optional ``[api]`` extra)."""

from __future__ import annotations

from .app import app, create_app

__all__ = ["app", "create_app"]
