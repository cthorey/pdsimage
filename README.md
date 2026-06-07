# pdsimage

Extract and visualize NASA **Lunar Reconnaissance Orbiter** data over an
arbitrary region of the Moon:

- **LOLA** (Lunar Orbiter Laser Altimeter) — topography
- **LROC WAC** (Wide Angle Camera) — imagery

It can pull data for any lat/lon window, render topography maps, WAC images,
overlays and elevation profiles, and ships catalogs of named impact craters and
low-slope domes for one-line lookup. Originally written in 2015 for Python 2.7
and `mpl_toolkits.basemap`; now Python 3.10+, packaged with **uv**, plotting on
**Cartopy**, with a **FastAPI** backend and a **React** frontend.

> Everything runs in Docker — you should not need a host Python/Node toolchain.

## Running the app

**Prerequisites:** Docker with the Compose plugin (`docker compose`). No host
Python or Node toolchain is needed — everything runs in containers.

```bash
git clone git@github.com:cthorey/pdsimage.git
cd pdsimage

# Build + start backend (API) and frontend (UI) together
docker compose up --build backend frontend
```

Then open **http://localhost:5173** for the UI. The API is on
**http://localhost:8000** (interactive docs at
**http://localhost:8000/docs**). The frontend dev server proxies `/api` to the
backend over the compose network.

Stop everything with `Ctrl-C`, or from another terminal:

```bash
docker compose down            # stop + remove containers
docker compose down -v         # also wipe the pds-cache volume (re-downloads tiles)
```

Downloaded PDS tiles are cached in the `pds-cache` volume, so they persist
across restarts and are shared by every service.

### Other commands

```bash
# Run the test suite (network mocked) — one-shot
docker compose run --rm test

# Start only the API, detached, on http://localhost:8000
docker compose up -d backend

# Tail logs / check status
docker compose logs -f backend
docker compose ps
```

Smoke-test the API (LOLA data server is the reliable one):

```bash
curl -s localhost:8000/health
curl -s "localhost:8000/catalog/craters?q=Copernicus"
curl -s -X POST localhost:8000/render -H 'content-type: application/json' \
  -d '{"lon0":339.92,"lat0":9.62,"size_km":100,"img_type":"lola","ppd":512}' \
  -o copernicus.png && file copernicus.png
```

## Library usage

```python
from pdsimage import Area, Crater
from pdsimage import plotting

# A named crater (window defaults to 80% of its diameter)
fig = plotting.overlay(Crater("name", "Copernicus"))
fig.savefig("copernicus.png")

# An arbitrary 100 km region around (lon, lat) = (339.92, 9.62)
area = Area(339.92, 9.62, 100)
X, Y, Z = area.get_arrays("lola")
```

Renderers (`lola_image`, `wac_image`, `overlay`, `draw_profile`) return a
`matplotlib.figure.Figure` — display it in a notebook or `savefig` it.

## CLI

```bash
docker compose run --rm backend pdsimage render --crater Copernicus --type lola -o /cache/c.png
```

## Architecture

| Module | Responsibility |
|---|---|
| `constants` | Moon radius, server URLs, resolution tables |
| `projections` | Lambert / cylindrical window math (pure) |
| `globe` | Cartopy CRSs on a spherical Moon |
| `download` | Cached, non-interactive PDS tile fetch |
| `binarytable` | Binary PDS IO, header parse, pixel↔coord |
| `maps` | `WacMap` / `LolaMap` tile stitching |
| `catalogs` | Crater / dome CSV loading + search |
| `area` | Region of interest (data) |
| `plotting` | Cartopy renderers → `Figure` |
| `structures` | `Crater`, `Dome` |
| `api/` | FastAPI app (optional `[api]` extra) |

## Notes & caveats

- The **WAC** data server (`lroc.sese.asu.edu`) currently serves an expired TLS
  certificate; downloads from it fall back to HTTP / host-scoped
  `verify=False`. The **LOLA** server (`imbrium.mit.edu`) is unaffected.
- The first request for an uncached tile downloads it (WAC global tiles are
  large and can take a while); subsequent requests are served from the cache
  volume.

## License

MIT — see `LICENSE.txt`.
