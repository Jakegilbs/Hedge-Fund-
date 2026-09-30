"""The trading universe: the core list from config plus (optionally) the S&P 500.

The S&P 500 list is downloaded at most once a week and cached in data/universe/.
If the download fails, the cached copy is used; with no cache, just the core list.
"""
from __future__ import annotations

import time

import pandas as pd

from .config import DATA_DIR, Settings

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
CACHE = DATA_DIR / "universe" / "sp500.csv"
MAX_AGE_SECONDS = 7 * 24 * 3600


def _normalise(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")  # BRK.B -> BRK-B (Yahoo style)


def sp500_symbols() -> list[str]:
    fresh = CACHE.is_file() and time.time() - CACHE.stat().st_mtime < MAX_AGE_SECONDS
    if not fresh:
        try:
            df = pd.read_csv(SP500_URL)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            df[["Symbol"]].to_csv(CACHE, index=False)
        except Exception:
            pass  # fall back to the cached copy, however old
    if CACHE.is_file():
        return sorted({_normalise(s) for s in pd.read_csv(CACHE)["Symbol"].dropna()})
    return []


def load_universe(settings: Settings) -> list[str]:
    symbols = set(settings.allowlist)
    if settings.include_sp500:
        symbols |= set(sp500_symbols())
    return sorted(symbols)
