"""Market data collection (free sources via yfinance).

All network access lives here so the rest of the desk can be tested offline.
Robinhood quotes replace these prices once the MCP connection is built.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .indicators import snapshot

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def download_bars(symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Daily OHLCV bars per symbol, oldest first. Symbols with no data are omitted."""
    raw = yf.download(symbols, period=period, interval="1d", auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
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


def mark_stale(snapshots: dict[str, dict], reference_date: str | None) -> None:
    """Flag any ticker whose last bar is older than the reference (usually SPY's) date."""
    for snap in snapshots.values():
        snap["stale"] = reference_date is None or snap["last_date"] < reference_date


def technical_packet(bars: dict[str, pd.DataFrame], tickers: list[str], reference_date: str | None) -> dict:
    snaps = {t: snapshot(bars[t]) for t in tickers if t in bars}
    mark_stale(snaps, reference_date)
    missing = [t for t in tickers if t not in bars]
    return {"reference_date": reference_date, "tickers": snaps, "missing_data": missing}


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


def catalyst_packet(tickers: list[str]) -> dict:
    """Headlines and next earnings date per ticker. Failures are recorded, not raised."""
    out: dict[str, dict] = {}
    for sym in tickers:
        t = yf.Ticker(sym)
        entry: dict = {"next_earnings": None, "headlines": [], "errors": []}
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
