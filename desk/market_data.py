"""Market data collection (free sources via yfinance).

All network access lives here so the rest of the desk can be tested offline.
Robinhood quotes replace these prices once the MCP connection is built.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import logging
import time

import pandas as pd
import yfinance as yf

from .config import DATA_DIR
from .indicators import snapshot

# Keep yfinance's cache inside the project folder instead of the system cache.
(DATA_DIR / "cache").mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(DATA_DIR / "cache"))
# yfinance prints lookup misses (e.g. "no fundamentals for SPY") that the desk already handles.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _extract(raw: pd.DataFrame, symbols: list[str], min_rows: int = 30) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df[[c for c in OHLCV if c in df.columns]].dropna()
        if len(df) >= min_rows and list(df.columns) == OHLCV:
            df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else df.index
            bars[sym] = df
    return bars


BAR_CACHE = DATA_DIR / "cache" / "bars"
FRESH_SECONDS = 20 * 60        # a re-run within 20 minutes downloads nothing


def _cache_path(sym: str, cache_dir=None):
    return (cache_dir or BAR_CACHE) / f"{sym.replace('^', '_')}.csv"


def _load_cached(sym: str, cache_dir=None) -> pd.DataFrame | None:
    p = _cache_path(sym, cache_dir)
    if not p.is_file():
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        return df[OHLCV] if len(df) >= 30 else None
    except Exception:
        return None


def _save_cached(sym: str, df: pd.DataFrame, cache_dir=None) -> None:
    (cache_dir or BAR_CACHE).mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(sym, cache_dir), date_format="%Y-%m-%d")


def download_bars(symbols: list[str], period: str = "2y", batch: int = 40, pause: float = 2.0,
                  priority: list[str] | None = None, log=print,
                  cache_dir=None) -> dict[str, pd.DataFrame]:
    """Daily OHLCV bars per symbol, oldest first, cached in data/cache/bars/.

    Yahoo throttles large bursts, so this downloads gently: priority symbols
    first (market context), then small sequential batches with pauses; symbols
    with a cache only fetch the last month; symbols cached in the last 20
    minutes are not downloaded at all; failures fall back to the cached copy.
    `cache_dir` keeps a separate cache (the backtester stores longer histories).
    """
    order = list(dict.fromkeys([*(priority or []), *symbols]))
    bars: dict[str, pd.DataFrame] = {}
    need_full, need_recent = [], []
    for sym in order:
        cached = _load_cached(sym, cache_dir)
        if cached is not None and time.time() - _cache_path(sym, cache_dir).stat().st_mtime < FRESH_SECONDS:
            bars[sym] = cached
        elif cached is not None:
            bars[sym] = cached              # fallback if the update fails
            need_recent.append(sym)
        else:
            need_full.append(sym)

    def fetch(syms: list[str], per: str) -> dict[str, pd.DataFrame]:
        try:
            raw = yf.download(syms, period=per, interval="1d", auto_adjust=True,
                              group_by="ticker", progress=False, threads=False)
        except Exception:
            return {}
        return _extract(raw, syms, min_rows=1)

    jobs = [(need_recent, "1mo"), (need_full, period)]
    total = len(need_recent) + len(need_full)
    if total:
        log(f"Downloading {total} symbols ({len(bars) - len(need_recent)} already fresh in cache)...")
    done = 0
    for syms, per in jobs:
        for i in range(0, len(syms), batch):
            chunk = syms[i:i + batch]
            got = fetch(chunk, per)
            for sym, df in got.items():
                if per == "1mo" and sym in bars:
                    df = pd.concat([bars[sym], df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                if len(df) >= 30:
                    bars[sym] = df
                    _save_cached(sym, df, cache_dir)
            done += len(chunk)
            if len(got) < len(chunk) * 0.5:
                log(f"  Yahoo returned {len(got)}/{len(chunk)}; slowing down...")
                time.sleep(pause * 5)
            else:
                time.sleep(pause)
    return bars


MARKET_OPEN, MARKET_CLOSE = (9, 30), (16, 0)


def split_incomplete_bar(bars: dict[str, pd.DataFrame], now: datetime) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """During market hours today's daily bar is still forming: drop it from the
    indicators (its volume is partial) and report its price separately as live.

    Returns (completed bars, live prices). Outside market hours nothing changes.
    """
    in_session = now.weekday() < 5 and MARKET_OPEN <= (now.hour, now.minute) < MARKET_CLOSE
    if not in_session:
        return bars, {}
    today = now.date()
    done: dict[str, pd.DataFrame] = {}
    live: dict[str, float] = {}
    for sym, df in bars.items():
        if df.index[-1].date() == today:
            live[sym] = round(float(df["Close"].iloc[-1]), 2)
            df = df.iloc[:-1]
        done[sym] = df
    return done, live


def mark_stale(snapshots: dict[str, dict], reference_date: str | None) -> None:
    """Flag any ticker whose last bar is older than the reference (usually SPY's) date."""
    for snap in snapshots.values():
        snap["stale"] = reference_date is None or snap["last_date"] < reference_date


def technical_packet(bars: dict[str, pd.DataFrame], tickers: list[str], reference_date: str | None,
                     live_prices: dict[str, float] | None = None) -> dict:
    """Indicators use completed daily bars; `live_price` is today's price so far, if the market is open."""
    snaps = {t: snapshot(bars[t]) for t in tickers if t in bars}
    mark_stale(snaps, reference_date)
    for t, snap in snaps.items():
        snap["live_price"] = (live_prices or {}).get(t)
    missing = [t for t in tickers if t not in bars]
    note = ("Market is open: indicators use completed daily bars through reference_date; "
            "live_price is the current intraday price.") if live_prices else \
           "Market is closed: indicators include the latest completed session."
    return {"reference_date": reference_date, "session_note": note, "tickers": snaps, "missing_data": missing}


def _news_items(ticker: yf.Ticker, limit: int = 6) -> list[dict]:
    items = []
    for raw in (ticker.news or [])[:limit]:
        body = raw.get("content", raw)  # yfinance changed this shape between versions
        provider = body.get("provider") or {}
        items.append({
            "title": body.get("title"),
            "publisher": provider.get("displayName") if isinstance(provider, dict) else body.get("publisher"),
            "published": body.get("pubDate") or body.get("providerPublishTime"),
            "summary": (body.get("summary") or "")[:300] or None,
        })
    return [i for i in items if i["title"]]


def _next_earnings(ticker: yf.Ticker) -> str | None:
    try:
        cal = ticker.calendar
    except Exception:
        return None
    dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
    if not dates:
        return None
    upcoming = sorted(d for d in dates if isinstance(d, date) and d >= date.today())
    return str(upcoming[0]) if upcoming else None


def catalyst_packet(tickers: list[str], etfs: frozenset[str] = frozenset()) -> dict:
    """Headlines and next earnings date per ticker. Failures are recorded, not raised.

    ETFs have no earnings, so no earnings lookup is attempted for them.
    """
    out: dict[str, dict] = {}
    for sym in tickers:
        t = yf.Ticker(sym)
        entry: dict = {"is_etf": sym in etfs, "next_earnings": None, "headlines": [], "errors": []}
        if sym not in etfs:
            try:
                entry["next_earnings"] = _next_earnings(t)
            except Exception as e:  # data gaps must not stop the run
                entry["errors"].append(f"earnings: {e}")
        try:
            entry["headlines"] = _news_items(t)
        except Exception as e:
            entry["errors"].append(f"news: {e}")
        out[sym] = entry
    return {"today": str(date.today()), "tickers": out}


def regime_packet(bars: dict[str, pd.DataFrame], regime_symbols: list[str], sector_etfs: list[str],
                  allowlist: list[str]) -> dict:
    """Market-wide context: index trends, volatility, rates, sector leadership, breadth."""
    def brief(sym: str) -> dict | None:
        if sym not in bars:
            return None
        s = snapshot(bars[sym])
        keys = ("last_date", "close", "change_pct", "ema20", "ema50", "sma200", "rsi14", "pct_from_52w_high")
        return {k: s[k] for k in keys}

    sectors = {}
    for etf in sector_etfs:
        if etf in bars and len(bars[etf]) > 20:
            c = bars[etf]["Close"]
            sectors[etf] = round((float(c.iloc[-1]) / float(c.iloc[-21]) - 1) * 100, 2)

    above_50 = [t for t in allowlist if t in bars and len(bars[t]) >= 50
                and bars[t]["Close"].iloc[-1] > bars[t]["Close"].ewm(span=50, adjust=False).mean().iloc[-1]]
    return {
        "indexes_and_rates": {sym: brief(sym) for sym in regime_symbols},
        "sector_20d_return_pct": dict(sorted(sectors.items(), key=lambda kv: -kv[1])),
        "breadth_pct_allowlist_above_ema50": round(100 * len(above_50) / max(1, len(allowlist)), 1),
    }


def now_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M")
