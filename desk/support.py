"""Support bounces: when a stock dips back to a level that held before, does it hold again?

    python -m desk.support            # study, exploration years (2016-2021)
    python -m desk.support --confirm  # the locked years (2022 on), once the rule is written down
    python -m desk.support --scan     # today: market regime + stocks sitting at tested support

Support levels are found the way a chart reader sees them, without hindsight:
- price floors: a swing low (the lowest low of 21 days) only counts once 10 more
  days have passed, i.e. once it is visible on the chart. A retest is the first
  dip back to within 1% of it after the stock rose 5%+ above it. "Touches" is how
  many confirmed swing lows sit within 3% of that level over the past year
  (1 = the first retest of a single low; 3+ = a floor tested again and again);
- moving averages: a dip to a rising 50- or 200-day average from above.

Each retest is bought with a limit order just above the level (filled at the
level, or at the open if the stock gaps below it). It counts as a BOUNCE if the
stock then rises 2 daily ranges (ATR) before it closes 1 ATR below the level (a
BREAK), within 40 trading days; every trade, support or random, uses the same
stop distance (1 ATR plus 1% below the fill). About 35-40% bounces is break-even.
The same rule applied to random days in uptrending stocks is the baseline.

Market regime (from SPY): trend = above or below its 200-day average;
volatility = calm or volatile (20-day volatility below or above 20% a year).
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .indicators import atr

K = 10                  # a swing low is the lowest low of K days either side
WINDOW = 250            # levels older than about a year are ignored
ZONE = 0.03             # swing lows within 3% are the same level
NEAR = 0.01             # a retest = the low comes within 1% of the level
RISE = 0.05             # ... after the stock had risen 5% above it
MAX_DAYS = 40
TARGET_ATR, STOP_ATR = 2.0, 1.0
HORIZONS = (5, 20, 60)
LOCK_FROM = pd.Timestamp("2022-01-01")
VOL_CALM = 0.20


def regimes(spy: pd.DataFrame) -> pd.Series:
    """'bull calm', 'bull volatile', 'bear calm' or 'bear volatile' for each day."""
    c = spy["Close"]
    trend = np.where(c > c.rolling(200).mean(), "bull", "bear")
    vol = np.log(c).diff().rolling(20).std() * math.sqrt(252)
    calm = np.where(vol < VOL_CALM, "calm", "volatile")
    out = pd.Series([f"{a} {b}" for a, b in zip(trend, calm)], index=c.index)
    out[c.rolling(200).mean().isna() | vol.isna()] = None
    return out


def swing_lows(low: np.ndarray, k: int = K) -> np.ndarray:
    s = pd.Series(low)
    m = s.rolling(2 * k + 1, center=True).min()
    return np.nonzero((s == m).to_numpy() & m.notna().to_numpy())[0]


def find_events(df: pd.DataFrame) -> list[dict]:
    """Every support retest in one stock: day index, level, limit price, kind, touches."""
    low, close = df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    n = len(df)
    sl = swing_lows(low)
    events = []
    for s in sl:
        level, conf = low[s], s + K
        end = min(n, s + WINDOW)
        if conf + 2 >= end or not level > 0:
            continue
        rose = np.nonzero(close[conf + 1:end] >= level * (1 + RISE))[0]
        if not len(rose):
            continue
        start = conf + 1 + rose[0]
        hit = np.nonzero(low[start + 1:end] <= level * (1 + NEAR))[0]
        if not len(hit):
            continue
        t = start + 1 + hit[0]
        if close[conf + 1:t].min() < level * (1 - ZONE):        # the level already broke on the way
            continue
        touches = int(sum(1 for s2 in sl if s2 + K < t and t - s2 <= WINDOW
                          and abs(low[s2] / level - 1) <= ZONE))
        events.append({"t": t, "level": level, "limit": level * (1 + NEAR), "kind": "price floor",
                       "touches": touches})
    for w in (50, 200):
        sma = df["Close"].rolling(w).mean().to_numpy()
        for t in range(w + 21, n):
            lvl = sma[t - 1]
            if (low[t] <= lvl * 1.005 and close[t - 1] > lvl * 1.02 and sma[t - 1] > sma[t - 21]):
                events.append({"t": t, "level": lvl, "limit": lvl * 1.005, "kind": f"rising {w}-day avg",
                               "touches": 0})
    events.sort(key=lambda e: (e["kind"], e["t"]))
    kept, last = [], {}
    for e in events:                                              # one event per kind per 5 days
        if e["t"] - last.get(e["kind"], -99) > 5:
            kept.append(e)
            last[e["kind"]] = e["t"]
    return kept


def outcome(df: pd.DataFrame, t: int, fill: float, stop: float, a: float) -> str:
    """'bounce' (up TARGET_ATR first), 'break' (close below the stop first) or 'neither'."""
    high, close = df["High"].to_numpy(float), df["Close"].to_numpy(float)
    target = fill + TARGET_ATR * a
    for d in range(t, min(len(df), t + MAX_DAYS)):
        if close[d] < stop:
            return "break"
        if d > t and high[d] >= target:
            return "bounce"
    return "neither"


def stop_for(fill: float, a: float) -> float:
    """The same stop distance for every trade (support or random): 1 ATR plus 1% below the fill,
    so a support trade never wins on the bounce measure just because its stop is further away."""
    return fill * (1 - NEAR) - STOP_ATR * a


def study_ticker(ticker: str, df: pd.DataFrame, spy: pd.DataFrame, regime: pd.Series,
                 member_on=None, baseline_every: int = 10) -> list[dict]:
    if len(df) < 260:
        return []
    o, c = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    a14 = atr(df).to_numpy()
    sma200 = df["Close"].rolling(200).mean().to_numpy()
    hi252 = df["High"].rolling(252, min_periods=120).max().to_numpy()
    spy_o, spy_c = spy["Open"].reindex(df.index).to_numpy(float), spy["Close"].reindex(df.index).to_numpy(float)
    reg = regime.reindex(df.index).to_numpy(object)
    rows = []

    def row(t, kind, touches, fill, stop):
        a = a14[t - 1]
        day = df.index[t]
        if not (a > 0 and fill > 0 and c[t - 1] >= 5) or (member_on and ticker not in member_on(day)):
            return None
        up = bool(c[t - 1] > sma200[t - 1] and sma200[t - 1] > sma200[t - 21]) if t > 220 else False
        r = {"ticker": ticker, "date": day, "kind": kind, "touches": touches, "uptrend": up,
             "regime": reg[t - 1], "off_high_pct": (c[t - 1] / hi252[t - 1] - 1) * 100,
             "outcome": outcome(df, t, fill, stop, a)}
        for h in HORIZONS:
            e = t + h
            if e < len(df) and spy_o[t] > 0 and np.isfinite(spy_c[e]):
                r[f"excess_{h}"] = ((c[e] / fill - 1) - (spy_c[e] / spy_o[t] - 1)) * 100
            else:
                r[f"excess_{h}"] = np.nan
        return r

    for e in find_events(df):
        t = e["t"]
        if t < 221 or not np.isfinite(e["level"]):
            continue
        fill = min(o[t], e["limit"])                       # a resting limit buy just above the level
        r = row(t, e["kind"], e["touches"], fill, stop_for(fill, a14[t - 1]))
        if r:
            rows.append(r)
    for t in range(230, len(df) - 1, baseline_every):      # baseline: random days in uptrends
        if c[t - 1] > sma200[t - 1] and sma200[t - 1] > sma200[t - 21]:
            r = row(t, "random uptrend day", 0, o[t], stop_for(o[t], a14[t - 1]))
            if r:
                rows.append(r)
    return rows


def _table(ev: pd.DataFrame, by: pd.Series, title: str, order: list[str]) -> list[str]:
    lines = ["", f"--- {title}",
             f"{'group':<34}{'events':>8}{'bounce':>8}{'break':>8}{'5d xs':>9}{'20d xs':>9}{'60d xs':>9}{'t (20d)':>9}"]
    for name in order:
        part = ev[by == name]
        if part.empty:
            continue
        x = part["excess_20"].dropna()
        t = x.mean() / (x.std() / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan
        few = "  (few)" if len(part) < 100 else ""
        lines.append(f"{name:<34}{len(part):>8}{(part.outcome == 'bounce').mean() * 100:>7.0f}%"
                     f"{(part.outcome == 'break').mean() * 100:>7.0f}%"
                     + "".join(f"{part[f'excess_{h}'].mean():>+8.2f}%" for h in HORIZONS) + f"{t:>9.1f}{few}")
    return lines


def report(ev: pd.DataFrame, locked: bool) -> str:
    lines = ["=" * 104, f"SUPPORT BOUNCE STUDY: {len(ev)} events, {ev.ticker.nunique()} stocks, "
             f"{ev.date.min().date()} to {ev.date.max().date()}"
             + ("  [EXPLORATION YEARS ONLY; 2022+ locked]" if locked else "  [LOCKED YEARS]"), "=" * 104,
             f"bounce = rose {TARGET_ATR:g} ATR before closing {STOP_ATR:g} ATR + 1% below the buy price (within "
             f"{MAX_DAYS} days); about 35-40% is break-even.",
             "xs = return minus SPY after buying at the level. Compare every group with the random uptrend days."]
    lines += _table(ev, ev.kind, "By kind of support (baseline first)",
                    ["random uptrend day", "price floor", "rising 50-day avg", "rising 200-day avg"])
    floor = ev[ev.kind == "price floor"]
    touches = pd.Series(np.select([floor.touches >= 3, floor.touches == 2], ["3+ touches", "2 touches"],
                                  "1 (first retest)"), index=floor.index)
    lines += _table(floor, touches, "Price floors by how often the level held before",
                    ["1 (first retest)", "2 touches", "3+ touches"])
    trend = pd.Series(np.where(floor.uptrend, "stock in uptrend", "stock not in uptrend"), index=floor.index)
    lines += _table(floor, trend, "Price floors: is the stock itself trending up?",
                    ["stock in uptrend", "stock not in uptrend"])
    regs = ["bull calm", "bull volatile", "bear calm", "bear volatile"]
    lines += _table(ev, ev.regime.where(ev.kind != "random uptrend day"), "All support events by market regime", regs)
    lines += _table(ev, ev.regime.where(ev.kind == "random uptrend day"), "Baseline (random uptrend days) by regime", regs)
    best = floor[(floor.touches >= 2) & floor.uptrend]
    lines += _table(best, best.regime, "Tested floors (2+ touches) in uptrending stocks, by regime", regs)
    years = pd.Series([str(d.year) for d in best.date], index=best.index)
    lines += _table(best, years, "Tested floors (2+ touches) in uptrending stocks, year by year", sorted(set(years)))
    lines += ["", "Look for: bounce well above the baseline's, positive 20d/60d excess vs the baseline, |t| > 3,",
              "100+ events, steady across years. Then write the rule down and run --confirm once."]
    return "\n".join(lines)


def current_levels(df: pd.DataFrame) -> list[dict]:
    """Support levels the stock is sitting on today (within 3% above them)."""
    low, close = df["Low"].to_numpy(float), df["Close"].to_numpy(float)
    n, last = len(df), close[-1]
    sl = [s for s in swing_lows(low) if s + K < n and n - s <= WINDOW]
    out, seen = [], []
    for s in sl:
        level = low[s]
        if not (0 <= last / level - 1 <= ZONE) or any(abs(level / x - 1) <= ZONE for x in seen):
            continue
        if close[s + K:].min() < level * (1 - ZONE):              # already broken since
            continue
        touches = sum(1 for s2 in sl if abs(low[s2] / level - 1) <= ZONE)
        seen.append(level)
        out.append({"kind": "price floor", "level": level, "touches": touches})
    for w in (50, 200):
        sma = df["Close"].rolling(w).mean()
        if len(sma.dropna()) > 21 and sma.iloc[-1] > sma.iloc[-21] and 0 <= last / sma.iloc[-1] - 1 <= 0.02:
            out.append({"kind": f"rising {w}-day avg", "level": float(sma.iloc[-1]), "touches": 0})
    return out


def scan(bars: dict[str, pd.DataFrame], tickers: list[str], breadth_list: list[str]) -> str:
    spy = bars["SPY"]
    reg = regimes(spy).iloc[-1]
    c = spy["Close"]
    vol = float(np.log(c).diff().tail(20).std() * math.sqrt(252) * 100)
    above = [t for t in breadth_list if t in bars and len(bars[t]) > 50
             and bars[t]["Close"].iloc[-1] > bars[t]["Close"].tail(50).mean()]
    lines = [f"MARKET REGIME {c.index[-1].date()}: {reg.upper() if reg else 'unknown'}",
             f"  SPY {c.iloc[-1]:.2f} vs 200-day avg {c.tail(200).mean():.2f}; 20-day volatility {vol:.0f}%/yr; "
             f"{len(above) / max(1, len(breadth_list)) * 100:.0f}% of stocks above their 50-day avg",
             "", "STOCKS AT SUPPORT TODAY (uptrending stocks within 3% above a level; unproven until the study "
             "confirms it):", f"{'stock':<7}{'price':>9}{'level':>9}{'above':>8}  kind (touches)"]
    rows = []
    for t in tickers:
        df = bars.get(t)
        if df is None or len(df) < 260:
            continue
        cl = df["Close"]
        s200 = cl.rolling(200).mean()
        if not (cl.iloc[-1] > s200.iloc[-1] > s200.iloc[-21]):
            continue
        for lv in current_levels(df):
            rows.append((lv["touches"], t, cl.iloc[-1], lv))
    rows.sort(key=lambda r: (-r[0], r[1]))
    for touches, t, price, lv in rows[:40]:
        extra = f" ({touches} touches)" if lv["kind"] == "price floor" else ""
        lines.append(f"{t:<7}{price:>9.2f}{lv['level']:>9.2f}{(price / lv['level'] - 1) * 100:>7.1f}%  "
                     f"{lv['kind']}{extra}")
    if not rows:
        lines.append("  none")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true", help="show the locked years (2022 on); use once")
    ap.add_argument("--scan", action="store_true", help="today's regime and stocks at support (no study)")
    args = ap.parse_args()

    from . import market_data
    from .backtest import BACKTEST_CACHE
    from .config import load_settings
    from .membership import load_membership
    from .universe import sp500_symbols

    settings = load_settings()
    today = sorted(set(sp500_symbols()) - settings.etfs)
    if args.scan:
        bars = market_data.download_bars(sorted(set(today) | {"SPY"}), priority=["SPY"])
        print(scan(bars, today, today))
        return
    hist = load_membership()
    end = pd.Timestamp.today().normalize()
    tickers = sorted(((hist.ever_between(end - pd.Timedelta(days=12 * 365), end) if hist else set())
                      | set(today)) - settings.etfs)
    print(f"Support study on {len(tickers)} stocks that were in the S&P 500 (prices cached from earlier tests)...")
    bars = market_data.download_bars(sorted(set(tickers) | {"SPY"}), period="12y", priority=["SPY"],
                                     cache_dir=BACKTEST_CACHE)
    spy = bars["SPY"]
    reg = regimes(spy)
    rows = []
    for n, t in enumerate(tickers, 1):
        if t in bars:
            rows += study_ticker(t, bars[t], spy, reg, hist.on if hist else None)
        if n % 100 == 0:
            print(f"  {n}/{len(tickers)} stocks...")
    ev = pd.DataFrame(rows)
    ev = ev[ev.date >= end - pd.Timedelta(days=int(10 * 365.25))]
    part = ev[ev.date >= LOCK_FROM] if args.confirm else ev[ev.date < LOCK_FROM]
    text = report(part.reset_index(drop=True), locked=not args.confirm)
    print(text)
    out = DATA_DIR / "backtests"
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    (out / f"support_study_{stamp}.txt").write_text(text)
    part.to_csv(out / f"support_events_{stamp}.csv", index=False)
    print(f"\nSaved: data/backtests/support_study_{stamp}.txt and support_events_{stamp}.csv")


if __name__ == "__main__":
    main()
