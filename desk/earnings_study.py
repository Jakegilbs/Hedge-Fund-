"""Earnings event study: how did stocks react to their reports, and what happened next?

    python -m desk.earnings_study            # explore: reports up to the end of 2021 only
    python -m desk.earnings_study --confirm  # AFTER a strategy is written down: the locked years

For every quarterly report of every stock that was in the S&P 500 at the time,
it measures the surprise (reported EPS vs the estimate), the market's reaction
(close before the report to the close after it) and what the stock did NEXT:
bought at the following open and held 5, 20 or 60 trading days. Returns are
shown against SPY over the same days ("excess"), so a rising market does not
look like a discovery.

The years from 2022 on are kept locked by default. Searching many tables will
always turn up something that looks good by chance; a pattern found in the
exploration years is only believed if it also holds in the locked years, tested
once, after the strategy is written down.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR

HORIZONS = (5, 20, 60)
LOCK_FROM = pd.Timestamp("2022-01-01")
MIN_EVENTS = 100          # smaller groups are shown but marked as too small to trust
SURPRISE_BUCKETS = [(-np.inf, 0, "miss (<0%)"), (0, 5, "small beat (0-5%)"), (5, 15, "beat (5-15%)"),
                    (15, np.inf, "big beat (>15%)")]
REACTION_BUCKETS = [(-np.inf, -5, "fell >5%"), (-5, -1, "fell 1-5%"), (-1, 1, "flat (+/-1%)"),
                    (1, 5, "rose 1-5%"), (5, np.inf, "rose >5%")]


def build_events(bars: dict[str, pd.DataFrame], earnings: dict[str, pd.DataFrame],
                 member_on=None) -> pd.DataFrame:
    """One row per report. member_on(date) -> set of index members (None: every stock counts)."""
    spy = bars["SPY"]
    spy_o, spy_c = spy["Open"], spy["Close"]
    rows = []
    for sym, ev in earnings.items():
        df = bars.get(sym)
        if df is None or ev is None or len(ev) == 0 or len(df) < 260:
            continue
        idx = df.index
        o, c, v = (df[k].to_numpy(float) for k in ("Open", "Close", "Volume"))
        sma200 = df["Close"].rolling(200).mean().to_numpy()
        vol20 = df["Volume"].rolling(20).mean().to_numpy()
        for _, e in ev.iterrows():
            day = pd.Timestamp(e["date"])
            if member_on is not None and sym not in member_on(day):
                continue
            before = idx.searchsorted(day) - 1
            after = idx.searchsorted(day, side="right")
            if before < 200 or after + 1 >= len(idx) or not (c[before] > 0 and c[after] > 0):
                continue
            entry_i = after + 1                                   # the reaction is known: buy next open
            row = {"ticker": sym, "date": day.date(), "surprise_pct": e["surprise_pct"],
                   "reaction_pct": (c[after] / c[before] - 1) * 100,
                   "volume_x": max(v[before + 1: after + 1].max(initial=0) / vol20[before], 0) if vol20[before] else np.nan,
                   "runup_20d_pct": (c[before] / c[before - 20] - 1) * 100,
                   "above_200d": bool(c[before] > sma200[before]),
                   "entry_date": idx[entry_i].date()}
            spy_entry = spy_o.get(idx[entry_i])
            for h in HORIZONS:
                exit_i = after + h
                if exit_i >= len(idx) or not (o[entry_i] > 0):
                    row[f"ret_{h}"] = row[f"excess_{h}"] = np.nan
                    continue
                r = c[exit_i] / o[entry_i] - 1
                s = spy_c.get(idx[exit_i])
                row[f"ret_{h}"] = r * 100
                row[f"excess_{h}"] = (r - (s / spy_entry - 1)) * 100 if spy_entry and s else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def bucket(values: pd.Series, buckets) -> pd.Series:
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    for lo, hi, name in buckets:
        out[(values >= lo) & (values < hi)] = name
    return out


def summarize(ev: pd.DataFrame, h: int) -> dict:
    x = ev[f"excess_{h}"].dropna()
    n = len(x)
    if n == 0:
        return {"n": 0}
    sd = x.std()
    return {"n": n, "mean": x.mean(), "median": x.median(), "beat_spy": (x > 0).mean() * 100,
            "t": x.mean() / (sd / math.sqrt(n)) if n > 1 and sd > 0 else np.nan}


def table(ev: pd.DataFrame, by: pd.Series, title: str, order: list[str], horizons=HORIZONS) -> list[str]:
    lines = ["", f"--- {title}",
             f"{'group':<34}{'reports':>8}" + "".join(f"{f'{h}d excess':>12}{'beat':>7}" for h in horizons)
             + f"{f't ({horizons[1]}d)':>10}"]
    for name in order:
        part = ev[by == name]
        cells = [summarize(part, h) for h in horizons]
        n = cells[1].get("n", 0)
        if not n:
            continue
        flag = "" if n >= MIN_EVENTS else "  (few)"
        lines.append(f"{name:<34}{n:>8}" + "".join(
            f"{c.get('mean', np.nan):>+11.2f}%{c.get('beat_spy', np.nan):>6.0f}%" for c in cells)
            + f"{cells[1].get('t', np.nan):>10.1f}{flag}")
    return lines


def report(ev: pd.DataFrame, locked: bool) -> str:
    period = (f"{ev.date.min()} to {ev.date.max()}")
    lines = ["=" * 118, f"EARNINGS EVENT STUDY: {len(ev)} reports, {ev.ticker.nunique()} stocks, {period}"
             + ("  [EXPLORATION YEARS ONLY; 2022+ locked]" if locked else "  [LOCKED YEARS]"), "=" * 118,
             "Excess = the stock's return minus SPY's over the same days, buying at the open after the reaction.",
             "beat SPY = share of reports where the stock beat SPY. t: roughly, |t| above 3 is unlikely to be luck "
             "(reports cluster in the same weeks, so be stricter than usual)."]
    everyone = pd.Series("all reports", index=ev.index)
    lines += table(ev, everyone, "All reports (the baseline)", ["all reports"])
    s_b = bucket(ev.surprise_pct, SURPRISE_BUCKETS)
    r_b = bucket(ev.reaction_pct, REACTION_BUCKETS)
    lines += table(ev, s_b, "By EPS surprise", [b[2] for b in SURPRISE_BUCKETS])
    lines += table(ev, r_b, "By the market's reaction (close before -> close after the report)",
                   [b[2] for b in REACTION_BUCKETS])
    beat_miss = np.select([ev.surprise_pct > 0, ev.surprise_pct <= 0], ["beat", "miss"], "unknown")
    combo = pd.Series(beat_miss, index=ev.index) + " & " + r_b.astype(str)
    lines += table(ev, combo, "Beat or miss x reaction", [f"{a} & {b[2]}" for a in ("beat", "miss")
                                                              for b in REACTION_BUCKETS])
    big = ev.reaction_pct > 5
    vol = pd.Series(np.select([big & (ev.volume_x >= 3), big & (ev.volume_x < 3)],
                              ["rose >5%, volume 3x+", "rose >5%, volume <3x"], "other"), index=ev.index)
    lines += table(ev, vol, "Big rises: with or without heavy volume", ["rose >5%, volume 3x+", "rose >5%, volume <3x"])
    run = pd.Series(np.select([big & (ev.runup_20d_pct > 10), big & (ev.runup_20d_pct.between(-5, 10)),
                               big & (ev.runup_20d_pct < -5)],
                              ["rose >5%, after a 20d run-up >10%", "rose >5%, run-up -5..10%",
                               "rose >5%, after a 20d drop >5%"], "other"), index=ev.index)
    lines += table(ev, run, "Big rises: did the stock already run up before the report?",
                   ["rose >5%, after a 20d run-up >10%", "rose >5%, run-up -5..10%", "rose >5%, after a 20d drop >5%"])
    trend = pd.Series(np.select([big & ev.above_200d, big & ~ev.above_200d],
                                ["rose >5%, above 200-day avg", "rose >5%, below 200-day avg"], "other"),
                      index=ev.index)
    lines += table(ev, trend, "Big rises: in an uptrend or not", ["rose >5%, above 200-day avg",
                                                                   "rose >5%, below 200-day avg"])
    years = pd.Series([str(d.year) for d in ev.date], index=ev.index)
    beat_up = (ev.surprise_pct > 0) & (ev.reaction_pct > 5)
    lines += table(ev[beat_up], years[beat_up], "Beat & rose >5%, year by year (is it steady or one lucky year?)",
                   sorted(set(years[beat_up])))
    lines += ["", "Next: pick the pattern(s) that are strong (big excess, |t| > 3, 100+ reports) AND steady year by",
              "year, write the strategy down, then run --confirm once to test it on the locked years."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true",
                    help="show the locked years (2022 on). Use once, after the strategy is written down.")
    ap.add_argument("--years", type=float, default=10.0, help="years of reports (default 10)")
    args = ap.parse_args()

    from . import market_data
    from .backtest import BACKTEST_CACHE
    from .config import load_settings
    from .earnings import load_earnings
    from .membership import load_membership
    from .universe import sp500_symbols

    settings = load_settings()
    hist = load_membership()
    end = pd.Timestamp.today().normalize()
    first = end - pd.Timedelta(days=int(args.years * 365.25) + 420)
    tickers = sorted(((hist.ever_between(first, end) if hist else set()) | set(sp500_symbols())) - settings.etfs)
    print(f"Earnings study over {args.years:g} years: {len(tickers)} stocks (prices and earnings are cached "
          f"from the rotation test)...")
    bars = market_data.download_bars(sorted(set(tickers) | {"SPY"}), period=f"{math.ceil(args.years) + 2}y",
                                     priority=["SPY"], cache_dir=BACKTEST_CACHE)
    if "SPY" not in bars:
        raise SystemExit("No SPY data: Yahoo is throttling. Wait 15-30 minutes and run again.")
    earn = load_earnings([t for t in tickers if t in bars])
    ev = build_events(bars, earn, hist.on if hist else None)
    if ev.empty:
        raise SystemExit("No earnings reports found: check the earnings download (data/cache/earnings).")
    start = end - pd.Timedelta(days=int(args.years * 365.25))
    dates = pd.to_datetime(ev.date)
    ev = ev[dates >= start]
    dates = pd.to_datetime(ev.date)
    part = ev[dates >= LOCK_FROM] if args.confirm else ev[dates < LOCK_FROM]
    text = report(part.reset_index(drop=True), locked=not args.confirm)
    print(text)
    out = DATA_DIR / "backtests"
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    (out / f"earnings_study_{stamp}.txt").write_text(text)
    part.to_csv(out / f"earnings_events_{stamp}.csv", index=False)
    print(f"\nSaved: data/backtests/earnings_study_{stamp}.txt and earnings_events_{stamp}.csv "
          f"(one row per report{'' if args.confirm else ', exploration years only'})")


if __name__ == "__main__":
    main()
