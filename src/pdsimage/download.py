"""Non-interactive, cached download of PDS tiles (LOLA ``.IMG/.LBL``, WAC ``.IMG``).

This replaces the original interactive ``_maybe_download`` / ``_downloadfile``
helpers. Tiles are cached on disk so a given file is only fetched once. The
cache root defaults to the OS user-cache directory and can be overridden with
the ``PDSIMAGE_CACHE_DIR`` environment variable.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from pathlib import Path

import httpx
from platformdirs import user_cache_dir

from .constants import LOLA_BASE_URL, WAC_BASE_URL


class DownloadError(RuntimeError):
    """Raised when a tile is missing locally and could not be downloaded."""


def cache_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the cache directory, creating the ``LOLA``/``LROC_WAC`` subdirs."""
    if path is not None:
        root = Path(path)
    elif os.environ.get("PDSIMAGE_CACHE_DIR"):
        root = Path(os.environ["PDSIMAGE_CACHE_DIR"])
    else:
        root = Path(user_cache_dir("pdsimage"))
    (root / "LOLA").mkdir(parents=True, exist_ok=True)
    (root / "LROC_WAC").mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class TileSpec:
    """Where a tile lives, locally and remotely."""

    grid: str  # "WAC" or "LOLA"
    img: Path
    lbl: Path | None  # only LOLA has a separate label file
    img_url: str
    lbl_url: str | None


def categorize(fname: str, root: Path) -> TileSpec:
    """Resolve a tile name to local paths and remote URLs.

    Args:
        fname: Tile name *without* extension, e.g. ``LDEM_512_00N_45N_270_360``.
        root: Cache root (see :func:`cache_root`).
    """
    name = fname.upper()
    prefix = name.split("_")[0]
    if prefix == "WAC":
        img = root / "LROC_WAC" / f"{name}.IMG"
        return TileSpec("WAC", img, None, WAC_BASE_URL + f"{name}.IMG", None)
    if prefix == "LDEM":
        img = root / "LOLA" / f"{name}.IMG"
        lbl = root / "LOLA" / f"{name}.LBL"
        return TileSpec("LOLA", img, lbl, LOLA_BASE_URL + f"{name}.IMG", LOLA_BASE_URL + f"{name}.LBL")
    raise ValueError(
        f"{fname!r}: unrecognized image type. Names must start with 'WAC' or 'LDEM'."
    )


def _client_for(url: str) -> httpx.Client:
    """HTTP client tolerant of the WAC server's expired certificate.

    Verification is disabled *only* for the ASU host that serves an expired
    cert; every other host keeps normal TLS verification.
    """
    verify: bool | ssl.SSLContext = True
    if "lroc.sese.asu.edu" in url:
        verify = False
    timeout = httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=30.0)
    return httpx.Client(verify=verify, follow_redirects=True, timeout=timeout)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with _client_for(url) as client, client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Failed to download {url}: {exc}") from exc


def ensure_tile(fname: str, root: Path | None = None, allow_download: bool = True) -> TileSpec:
    """Ensure a tile (and its ``.LBL`` for LOLA) exists locally.

    Args:
        fname: Tile name without extension.
        root: Cache root; defaults to :func:`cache_root`.
        allow_download: If ``False``, never hit the network — raise if missing.

    Returns:
        The resolved :class:`TileSpec` with local paths guaranteed to exist.
    """
    root = cache_root(root)
    spec = categorize(fname, root)

    if not spec.img.is_file():
        if not allow_download:
            raise DownloadError(f"{spec.img} is missing and downloads are disabled.")
        _download(spec.img_url, spec.img)

    if spec.grid == "LOLA" and spec.lbl is not None and not spec.lbl.is_file():
        if not allow_download:
            raise DownloadError(f"{spec.lbl} is missing and downloads are disabled.")
        _download(spec.lbl_url, spec.lbl)  # type: ignore[arg-type]

    return spec
