"""Earnings history for the earnings-drift signal (free, from Yahoo via yfinance).

Stocks that beat expectations and rise on the report tend to keep drifting up
for weeks ("post-earnings announcement drift"). For each stock this module
keeps the past report dates with the EPS surprise, cached in
data/cache/earnings/ and refreshed monthly, and turns them into a day-by-day
flag: True for about three months after a report that beat estimates AND that
the market rewarded.

The market's reaction is measured from the close BEFORE the report date to the
close of the first trading day AFTER it, which covers reports before the open
and after the close alike. The flag only starts after that day, so a backtest
never uses a reaction before it happened.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import DATA_DIR

EARNINGS_CACHE = DATA_DIR / "cache" / "earnings"
MAX_AGE_SECONDS = 30 * 24 * 3600
DRIFT_DAYS = 63            # a report's signal lasts about one quarter (63 trading days)
COLUMNS = ["date", "eps_estimate", "eps_actual", "surprise_pct"]


def _path(sym: str):
    return EARNINGS_CACHE / f"{sym}.csv"


def normalise(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Yahoo's table -> date (report day, New York time), eps_estimate, eps_actual, surprise_pct.
    Future reports (no reported EPS yet) are dropped."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).tz_convert("America/New_York")
    cols = {c.lower().replace(" ", ""): c for c in df.columns}

    def col(*keys):
        for k in keys:
            for low, orig in cols.items():
                if k in low:
                    return pd.to_numeric(df[orig], errors="coerce").to_numpy()
        return np.full(len(df), np.nan)

    out = pd.DataFrame({"date": idx.tz_localize(None).normalize(),
                        "eps_estimate": col("epsestimate", "estimate"),
                        "eps_actual": col("reportedeps", "epsactual", "reported"),
                        "surprise_pct": col("surprise")})
    out = out[out.eps_actual.notna()].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    missing = out.surprise_pct.isna() & out.eps_estimate.notna() & (out.eps_estimate != 0)
    out.loc[missing, "surprise_pct"] = ((out.eps_actual - out.eps_estimate) / out.eps_estimate.abs() * 100)[missing]
    return out


def _fetch_yahoo(sym: str) -> pd.DataFrame | None:
    import logging

    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        return yf.Ticker(sym).get_earnings_dates(limit=60)
    except Exception:
        return None


def load_earnings(symbols: list[str], fetch=_fetch_yahoo, pause: float = 0.4, log=print) -> dict[str, pd.DataFrame]:
    """Earnings history per symbol. Cached for a month; a symbol with no data is cached
    as empty so it is not asked again every run."""
    EARNINGS_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    todo = []
    for sym in symbols:
        p = _path(sym)
        if p.is_file() and time.time() - p.stat().st_mtime < MAX_AGE_SECONDS:
            try:
                df = pd.read_csv(p, parse_dates=["date"])
                out[sym] = df[COLUMNS] if len(df) else pd.DataFrame(columns=COLUMNS)
                continue
            except Exception:
                pass
        todo.append(sym)
    if todo:
        log(f"Downloading earnings history for {len(todo)} stocks (about {len(todo) * (pause + 0.6) / 60:.0f} "
            f"minutes the first time; cached for a month)...")
    for n, sym in enumerate(todo, 1):
        df = normalise(fetch(sym))
        df.to_csv(_path(sym), index=False, date_format="%Y-%m-%d")
        out[sym] = df
        if n % 50 == 0:
            log(f"  earnings: {n}/{len(todo)}")
        time.sleep(pause)
    return out


def drift_flags(earnings: dict[str, pd.DataFrame], close: pd.DataFrame, min_surprise: float = 0.0,
                days: int = DRIFT_DAYS) -> pd.DataFrame:
    """True on the days a stock is in a positive earnings drift: its latest report beat
    estimates (surprise above min_surprise, when known) and the stock rose on it.
    Starts the day after the reaction day and lasts `days` trading days or until
    the next report."""
    flags = pd.DataFrame(False, index=close.index, columns=close.columns)
    idx = close.index
    for sym in close.columns:
        ev = earnings.get(sym)
        if ev is None or len(ev) == 0:
            continue
        c = close[sym].to_numpy(float)
        col = flags.columns.get_loc(sym)
        arr = np.zeros(len(idx), dtype=bool)
        for _, e in ev.sort_values("date").iterrows():
            day = pd.Timestamp(e["date"])
            before = idx.searchsorted(day) - 1                 # last close before the report day
            after = idx.searchsorted(day, side="right")        # first trading day after it
            if before < 0 or after >= len(idx) or not (np.isfinite(c[before]) and np.isfinite(c[after])):
                continue
            reaction = c[after] / c[before] - 1
            surprise = e["surprise_pct"]
            good = reaction > 0 and (not np.isfinite(surprise) or surprise > min_surprise)
            arr[after + 1: after + 1 + days] = good             # a newer report overrides an older one
        flags.iloc[:, col] = arr
    return flags
