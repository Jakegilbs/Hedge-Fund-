"""Technical indicators and swing-trade setups computed in code.

Analysts receive these numbers instead of raw price history, which keeps
prompts small (cheap) and the arithmetic exact. Setup rules are ported from
the earlier ai-trade scanner.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def _r(x: float | None, nd: int = 2) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), nd)


def setup_breakout(df: pd.DataFrame) -> dict | None:
    """Close at or above the prior 52-week closing high on above-average volume."""
    if len(df) < 200:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    prior_high = float(c.shift(1).rolling(252, min_periods=120).max().iloc[-1])
    vol_x = float(v.iloc[-1] / (v.rolling(20, min_periods=10).mean().iloc[-1] or 1))
    if not (c.iloc[-1] >= prior_high * 0.995 and vol_x >= 1.15):
        return None
    a = float(atr(df).iloc[-1])
    entry = max(float(h.iloc[-1]), prior_high)
    stop = min(float(l.iloc[-1]), entry - 1.1 * a)
    return {"setup": "breakout", "pivot": _r(prior_high), "entry": _r(entry), "stop": _r(stop),
            "volume_vs_20d": _r(vol_x)}


def setup_pullback(df: pd.DataFrame) -> dict | None:
    """Uptrend (close > EMA50 > SMA200) pulled back to within 2% of EMA20 and turning up."""
    if len(df) < 220:
        return None
    c, h, l = df["Close"], df["High"], df["Low"]
    e20, e50, s200 = ema(c, 20), ema(c, 50), sma(c, 200)
    uptrend = c.iloc[-1] > e50.iloc[-1] > s200.iloc[-1]
    near20 = abs(c.iloc[-1] - e20.iloc[-1]) / e20.iloc[-1] < 0.02
    reclaim = c.iloc[-1] > c.iloc[-2]
    if not (uptrend and near20 and reclaim):
        return None
    a = float(atr(df).iloc[-1])
    entry = float(h.iloc[-1])
    stop = min(float(l.rolling(5).min().iloc[-1]), entry - 1.25 * a)
    return {"setup": "pullback", "entry": _r(entry), "stop": _r(stop), "ema20": _r(float(e20.iloc[-1]))}


def setup_vcp(df: pd.DataFrame) -> dict | None:
    """Volatility contraction near the 100-day high."""
    if len(df) < 120:
        return None
    c, h = df["Close"], df["High"]
    atrp = atr(df) / c
    atrp_rank = float((atrp / atrp.rolling(120, min_periods=60).max()).iloc[-1])
    near_high = float(c.iloc[-1] / c.rolling(100, min_periods=60).max().iloc[-1])
    if not (atrp_rank < 0.45 and near_high > 0.95):
        return None
    a = float(atr(df).iloc[-1])
    pivot = float(h.rolling(20).max().iloc[-1])
    return {"setup": "vcp", "pivot": _r(pivot), "entry": _r(pivot), "stop": _r(pivot - 1.5 * a),
            "atr_pct_rank": _r(atrp_rank)}


SETUPS = (setup_breakout, setup_pullback, setup_vcp)


def trend_facts(c: pd.Series) -> dict:
    """Moving-average comparisons done in code, so analysts never have to do the arithmetic."""
    last = float(c.iloc[-1])
    e50 = float(ema(c, 50).iloc[-1])
    s200 = float(sma(c, 200).iloc[-1]) if len(c) >= 200 else None
    up_20d = len(c) > 20 and last > float(c.iloc[-21])
    facts = {
        "close_above_ema20": last > float(ema(c, 20).iloc[-1]),
        "close_above_ema50": last > e50,
        "close_above_sma200": None if s200 is None else last > s200,
        "ema50_above_sma200": None if s200 is None else e50 > s200,
        "up_over_20d": up_20d,
    }
    if s200 is not None and last > e50 > s200 and up_20d:
        facts["trend"] = "up"
    elif s200 is not None and last < e50 < s200:
        facts["trend"] = "down"
    else:
        facts["trend"] = "sideways"
    return facts


def snapshot(df: pd.DataFrame) -> dict:
    """Compact technical snapshot of one ticker's daily bars (oldest first)."""
    c, v = df["Close"], df["Volume"]
    last = float(c.iloc[-1])

    def change(n: int) -> float | None:
        return _r((last / float(c.iloc[-1 - n]) - 1) * 100) if len(c) > n else None

    hi_252 = float(df["High"].tail(252).max())
    lo_252 = float(df["Low"].tail(252).min())
    a = float(atr(df).iloc[-1]) if len(df) >= 14 else float("nan")
    vol20 = float(v.rolling(20, min_periods=10).mean().iloc[-1]) if len(v) >= 10 else float("nan")
    return {
        "trend_facts": trend_facts(c),
        "last_date": str(df.index[-1].date()),
        "close": _r(last),
        "change_pct": {"1d": change(1), "5d": change(5), "20d": change(20), "60d": change(60)},
        "ema20": _r(float(ema(c, 20).iloc[-1])),
        "ema50": _r(float(ema(c, 50).iloc[-1])),
        "sma200": _r(float(sma(c, 200).iloc[-1])) if len(c) >= 200 else None,
        "rsi14": _r(float(rsi(c).iloc[-1]), 1),
        "atr14": _r(a),
        "atr_pct": _r(a / last * 100),
        "high_52w": _r(hi_252),
        "low_52w": _r(lo_252),
        "pct_from_52w_high": _r((last / hi_252 - 1) * 100),
        "volume_vs_20d": _r(float(v.iloc[-1]) / vol20) if vol20 else None,
        "avg_dollar_volume_20d": _r(float((c * v).tail(20).mean()), 0),
        "last_10_closes": [_r(float(x)) for x in c.tail(10)],
        "setups": [s for s in (fn(df) for fn in SETUPS) if s],
    }
