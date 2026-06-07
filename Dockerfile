# Backend image: pdsimage library + FastAPI API.
FROM python:3.12-slim

# System libraries required by Cartopy (GEOS / PROJ) and matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgeos-dev \
        libproj-dev \
        proj-data \
        proj-bin \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_LINK_MODE=copy \
    PDSIMAGE_CACHE_DIR=/cache \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first (better layer caching), then the project.
COPY pyproject.toml README.md LICENSE.txt ./
COPY src ./src
RUN uv sync --extra api --group dev

COPY tests ./tests

RUN mkdir -p /cache
EXPOSE 8000

CMD ["uvicorn", "pdsimage.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
