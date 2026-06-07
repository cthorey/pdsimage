"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="pdsimage API",
        version="2.0.0",
        description="Extract and visualize NASA LRO LOLA/WAC lunar data.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Window"],
    )
    app.include_router(router)
    return app


app = create_app()
