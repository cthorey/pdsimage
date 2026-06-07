"""Shared physical constants, data-server URLs and resolution tables.

The Moon radius value (1734.4 km) is the one used by the original library and
is preserved verbatim so that all projection math reproduces historical output
byte-for-byte.
"""

from __future__ import annotations

# Moon radius as used by the original pdsimage code (kept for numerical parity).
MOON_RADIUS_KM: float = 1734.4
MOON_RADIUS_M: float = 1734400.0

# Remote PDS data servers.
# NOTE: the WAC server (ASU) currently serves an *expired* TLS certificate, so
# downloads fall back to plain HTTP / host-scoped verify=False (see download.py).
WAC_BASE_URL: str = (
    "http://lroc.sese.asu.edu/data/LRO-L-LROC-5-RDR-V1.0/"
    "LROLRC_2001/DATA/BDR/WAC_GLOBAL/"
)
LOLA_BASE_URL: str = "http://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/"

# Resolutions (pixels per degree) implemented for each product.
WAC_RESOLUTIONS: list[int] = [4, 8, 16, 32, 64, 128, 256]
LOLA_RESOLUTIONS: list[int] = [4, 16, 64, 128, 256, 512, 1024]

# Default resolutions used by Area / Crater / Dome.
DEFAULT_PPD_LOLA: int = 512
DEFAULT_PPD_WAC: int = 128
