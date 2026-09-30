"""The trading universe: the core list, the S&P 500 and (options mode) cheap Nasdaq stocks.

Lists are refreshed at most once a week and cached in data/universe/. If a
download fails, the cached copy is used; with no cache, that part is skipped.

Cheap Nasdaq stocks: every common stock on Nasdaq's Global Select and Global
Market tiers (from Nasdaq's own symbol file) is screened once a week on a month
of prices, keeping those in the configured price band with enough daily dollar
volume. Cheap, liquid stocks are where a small account can afford options.
"""
from __future__ import annotations

import io
import time

import pandas as pd

from .config import DATA_DIR, Settings

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
NASDAQ_CACHE = DATA_DIR / "universe" / "nasdaq_listed.csv"
NASDAQ_CHEAP_CACHE = DATA_DIR / "universe" / "nasdaq_cheap.csv"
NOT_COMMON = ("warrant", "unit", " right", "preferred", "depositary", "notes due", "debenture", "acquisition corp")

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
CACHE = DATA_DIR / "universe" / "sp500.csv"
MAX_AGE_SECONDS = 7 * 24 * 3600


def _normalise(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")  # BRK.B -> BRK-B (Yahoo style)


def sp500_symbols() -> list[str]:
    if not _fresh(CACHE):
        try:
            df = pd.read_csv(SP500_URL)
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            df[["Symbol"]].to_csv(CACHE, index=False)
        except Exception:
            pass  # fall back to the cached copy, however old
    if CACHE.is_file():
        return sorted({_normalise(s) for s in pd.read_csv(CACHE)["Symbol"].dropna()})
    return []


def _fresh(path) -> bool:
    return path.is_file() and time.time() - path.stat().st_mtime < MAX_AGE_SECONDS


def parse_nasdaq_listed(text: str) -> list[str]:
    """Common stocks from Nasdaq's pipe-delimited symbol file (Global Select / Global Market tiers)."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")
    keep = ((df["Test Issue"] == "N") & (df["ETF"] == "N") & (df["Financial Status"] == "N")
            & df["Market Category"].isin(["Q", "G"]))
    df = df[keep]
    name = df["Security Name"].str.lower()
    df = df[~name.apply(lambda n: any(w in n for w in NOT_COMMON))]
    syms = df["Symbol"].str.strip()
    return sorted(s for s in syms if s.isalpha() and len(s) <= 5)


def nasdaq_symbols() -> list[str]:
    if not _fresh(NASDAQ_CACHE):
        try:
            import requests
            r = requests.get(NASDAQ_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            NASDAQ_CACHE.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"Symbol": parse_nasdaq_listed(r.text)}).to_csv(NASDAQ_CACHE, index=False)
        except Exception:
            pass
    if NASDAQ_CACHE.is_file():
        return pd.read_csv(NASDAQ_CACHE)["Symbol"].dropna().astype(str).tolist()
    return []


def _yahoo_month(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """One month of daily bars, not cached (screening only)."""
    import yfinance as yf

    from .market_data import _extract
    raw = yf.download(symbols, period="1mo", interval="1d", auto_adjust=True, group_by="ticker",
                      progress=False, threads=False)
    return _extract(raw, symbols, min_rows=10)


def screen_cheap(symbols: list[str], min_price: float, max_price: float, min_dollar_volume: float,
                 fetch=_yahoo_month, batch: int = 50, pause: float = 1.5, log=print) -> list[str]:
    """Keep symbols whose last close is in [min_price, max_price] with 20-day average
    dollar volume of at least min_dollar_volume."""
    keep = []
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        try:
            bars = fetch(chunk)
        except Exception:
            bars = {}
        for sym, df in bars.items():
            close = float(df["Close"].iloc[-1])
            dollar_vol = float((df["Close"] * df["Volume"]).tail(20).mean())
            if min_price <= close <= max_price and dollar_vol >= min_dollar_volume:
                keep.append(sym)
        if (i // batch) % 10 == 9:
            log(f"  screened {i + len(chunk)}/{len(symbols)} Nasdaq stocks...")
        time.sleep(pause)
    return sorted(keep)


def cheap_nasdaq(settings: Settings, log=print, fetch=_yahoo_month) -> list[str]:
    """Weekly-cached list of cheap, liquid Nasdaq stocks."""
    if not _fresh(NASDAQ_CHEAP_CACHE):
        listed = nasdaq_symbols()
        if listed:
            log(f"Weekly screen of {len(listed)} Nasdaq stocks for ${settings.nasdaq_min_price:g}-"
                f"${settings.nasdaq_max_price:g} and ${settings.nasdaq_min_dollar_volume / 1e6:g}M+/day "
                "(a few minutes, once a week)...")
            found = screen_cheap(listed, settings.nasdaq_min_price, settings.nasdaq_max_price,
                                 settings.nasdaq_min_dollar_volume, fetch=fetch, log=log)
            if found:  # never replace a good list with an empty one after a failed screen
                NASDAQ_CHEAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"Symbol": found}).to_csv(NASDAQ_CHEAP_CACHE, index=False)
    if NASDAQ_CHEAP_CACHE.is_file():
        return pd.read_csv(NASDAQ_CHEAP_CACHE)["Symbol"].dropna().astype(str).tolist()
    return []


def load_universe(settings: Settings, log=print) -> list[str]:
    symbols = set(settings.allowlist)
    if settings.include_sp500:
        symbols |= set(sp500_symbols())
    if settings.uses_options:
        symbols |= set(settings.options_extra)
        if settings.include_nasdaq_cheap:
            symbols |= set(cheap_nasdaq(settings, log=log))
    return sorted(symbols)
