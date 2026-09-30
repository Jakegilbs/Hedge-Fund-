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


def _extract(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df[[c for c in OHLCV if c in df.columns]].dropna()
        if len(df) >= 30:
            bars[sym] = df
    return bars


def download_bars(symbols: list[str], period: str = "2y", batch: int = 100,
                  pause: float = 1.0) -> dict[str, pd.DataFrame]:
    """Daily OHLCV bars per symbol, oldest first. Symbols with no data are omitted.

    Downloads in batches (one big request gets throttled by Yahoo), then retries
    whatever came back empty once.
    """
    def fetch(syms: list[str]) -> dict[str, pd.DataFrame]:
        raw = yf.download(syms, period=period, interval="1d", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        return _extract(raw, syms)

    bars: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), batch):
        bars.update(fetch(symbols[i:i + batch]))
        time.sleep(pause)
    missing = [s for s in symbols if s not in bars]
    if missing:
        time.sleep(pause * 3)
        for i in range(0, len(missing), batch // 2):
            bars.update(fetch(missing[i:i + batch // 2]))
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
