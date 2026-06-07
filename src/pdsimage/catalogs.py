"""Load the bundled crater and dome catalogs.

The CSVs ship inside the package (``pdsimage/data``) and are read through
:mod:`importlib.resources` so they work from an installed wheel. Both files
were written with a leading unnamed index column, so :func:`pandas.read_csv`
uses it as the DataFrame index (preserving the original column semantics:
crater ``name`` = ``SegnerC`` etc.).
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import pandas as pd

CRATER_CSV = "Data_Crater.csv"
DOME_CSV = "Data_Dome.csv"


@lru_cache(maxsize=2)
def load_craters() -> pd.DataFrame:
    """Return the crater catalog as a DataFrame."""
    with (files("pdsimage.data") / CRATER_CSV).open("r") as fh:
        return pd.read_csv(fh)


@lru_cache(maxsize=2)
def load_domes() -> pd.DataFrame:
    """Return the dome catalog as a DataFrame."""
    with (files("pdsimage.data") / DOME_CSV).open("r") as fh:
        return pd.read_csv(fh)


def lookup(df: pd.DataFrame, ide: str, idx) -> pd.DataFrame:
    """Return rows of ``df`` where column ``ide`` equals ``idx``.

    Args:
        df: A catalog DataFrame.
        ide: Column to match on, typically ``"name"`` or ``"index"``.
        idx: Value to match.
    """
    return df[df[ide] == idx]


def search(df: pd.DataFrame, q: str | None = None, limit: int | None = None) -> list[dict]:
    """Return named catalog rows (optionally name-filtered) as JSON-friendly dicts.

    Rows without a usable name are dropped: this is a name-based browse list, and
    unnamed entries can only be reached by table index anyway.
    """
    out = df[df["name"].notna()]
    out = out[out["name"].astype(str).str.strip() != ""]
    if q:
        out = out[out["name"].astype(str).str.contains(q, case=False, na=False)]
    if limit is not None:
        out = out.head(limit)
    return out.to_dict(orient="records")
