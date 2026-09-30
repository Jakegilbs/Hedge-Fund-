"""Insider buying study: when company insiders buy their own stock, what happens next?

    python -m desk.insider_study            # explore: purchases filed up to the end of 2021
    python -m desk.insider_study --confirm  # AFTER a strategy is written down: the locked years

Every open-market purchase by an officer, director or 10% owner of any US-listed
company (SEC Form 4) is grouped into buying events: for each company, the
purchases filed in the previous 30 days (how many different insiders, how many
dollars, who). The stock is bought at the first open AFTER the filing date (when
the purchase became public) and held 20, 60 or 120 trading days.

Returns are measured against a size-matched benchmark: SPY for large, heavily
traded companies (over $100M a day), IWM (small caps) for everything else, so a
small-cap rally does not look like an insider edge.

As in the earnings study, the years from 2022 on stay locked until a strategy is
written down. Prices come from Yahoo, which has no history for most companies
that were later acquired or went bankrupt; the coverage line shows how many
events have prices. The missing ones include failures, so results lean optimistic.
"""
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .earnings_study import bucket, table

HORIZONS = (20, 60, 120)
LOCK_FROM = pd.Timestamp("2022-01-01")
WINDOW_DAYS = 30               # purchases filed within 30 days count as one buying event
MIN_PRICE = 1.0                # below $1 the price data is too noisy to trust
TOP_TITLE = re.compile(r"\bCEO\b|CHIEF EXEC|\bCFO\b|CHIEF FIN|PRESIDENT")


def build_events(purchases: pd.DataFrame, window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """One row per buying event. A new event starts when a company has had no event for
    `window_days`, or when more insiders join (1 -> 2 -> 3+), so clusters are captured."""
    p = purchases.dropna(subset=["filing_date"]).copy()
    p["filing_date"] = pd.to_datetime(p["filing_date"]).dt.normalize()
    p["owner_key"] = p["owner_cik"].fillna(p["owner"]).astype(str)
    title = p["title"].fillna("").astype(str).str.upper()
    rel = p["relationship"].fillna("").astype(str).str.upper()
    p["is_top"] = title.str.contains(TOP_TITLE)
    p["is_officer"] = rel.str.contains("OFFICER") | p["is_top"]
    p["is_director"] = rel.str.contains("DIRECTOR")
    p["is_owner10"] = rel.str.contains("TEN") | rel.str.contains("10")
    rows = []
    for ticker, g in p.sort_values("filing_date").groupby("ticker"):
        g = g.reset_index(drop=True)
        last_date, last_level = None, 0
        for day in g["filing_date"].unique():
            day = pd.Timestamp(day)
            w = g[(g.filing_date > day - pd.Timedelta(days=window_days)) & (g.filing_date <= day)]
            n = w["owner_key"].nunique()
            level = min(n, 3)
            fresh = last_date is None or (day - last_date).days > window_days
            if not (fresh or level > last_level):
                continue
            if w["is_top"].any():
                who = "CEO/CFO/President bought"
            elif w["is_officer"].any():
                who = "other officers bought"
            elif w["is_director"].any():
                who = "directors only"
            elif w["is_owner10"].any():
                who = "10% owners only"
            else:
                who = "other"
            rows.append({"ticker": ticker, "date": day, "issuer": g["issuer"].iloc[0], "insiders": n,
                         "value": float(w["value"].sum()), "purchases": len(w), "who": who,
                         "avg_price": float((w["value"].sum() / w["shares"].sum()) if w["shares"].sum() else np.nan)})
            last_date, last_level = day, level
    return pd.DataFrame(rows)


def add_returns(events: pd.DataFrame, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Entry at the first open after the filing date; returns vs the size-matched benchmark."""
    spy, iwm = bars["SPY"], bars.get("IWM", bars["SPY"])
    out = []
    for e in events.itertuples(index=False):
        df = bars.get(e.ticker)
        if df is None or len(df) < 80:
            continue
        idx = df.index
        entry_i = idx.searchsorted(pd.Timestamp(e.date), side="right")        # strictly after the filing day
        sig_i = entry_i - 1
        if entry_i >= len(idx) or sig_i < 60:
            continue
        o, c, v = df["Open"].to_numpy(float), df["Close"].to_numpy(float), df["Volume"].to_numpy(float)
        if not (o[entry_i] >= MIN_PRICE and c[sig_i] > 0):
            continue
        dvol = float(np.nanmean(c[max(0, sig_i - 19): sig_i + 1] * v[max(0, sig_i - 19): sig_i + 1]))
        bench = spy if dvol >= 100e6 else iwm
        row = {**e._asdict(), "entry_date": idx[entry_i].date(), "entry_price": o[entry_i],
               "dollar_volume": dvol, "benchmark": "SPY" if bench is spy else "IWM",
               "prior_60d_pct": (c[sig_i] / c[sig_i - 60] - 1) * 100,
               "value_vs_daily_volume": e.value / dvol if dvol > 0 else np.nan}
        b_open = bench["Open"].get(idx[entry_i])
        for h in HORIZONS:
            exit_i = entry_i + h - 1
            if exit_i >= len(idx):
                row[f"ret_{h}"] = row[f"excess_{h}"] = np.nan
                continue
            r = c[exit_i] / o[entry_i] - 1
            b_close = bench["Close"].get(idx[exit_i])
            row[f"ret_{h}"] = r * 100
            row[f"excess_{h}"] = (r - (b_close / b_open - 1)) * 100 if b_open and b_close else np.nan
        out.append(row)
    return pd.DataFrame(out)


SIZE_BUCKETS = [(0, 1e6, "under $1M/day (micro)"), (1e6, 10e6, "$1-10M/day (small)"),
                (10e6, 100e6, "$10-100M/day (mid)"), (100e6, np.inf, "over $100M/day (large)")]
VALUE_BUCKETS = [(0, 50e3, "under $50k"), (50e3, 500e3, "$50k-500k"), (500e3, 5e6, "$500k-5M"),
                 (5e6, np.inf, "over $5M")]
PRIOR_BUCKETS = [(-np.inf, -20, "fell >20% in 60d"), (-20, 0, "fell 0-20%"), (0, np.inf, "rose")]


def report(ev: pd.DataFrame, locked: bool, coverage: str) -> str:
    h = HORIZONS
    lines = ["=" * 120, f"INSIDER BUYING STUDY: {len(ev)} buying events, {ev.ticker.nunique()} companies, "
             f"{ev.date.min().date()} to {ev.date.max().date()}"
             + ("  [EXPLORATION YEARS ONLY; 2022+ locked]" if locked else "  [LOCKED YEARS]"), "=" * 120,
             coverage,
             "Excess = return minus SPY (companies trading over $100M/day) or IWM (all others), buying at the first",
             "open after the filing became public. beat = share of events that beat the benchmark. |t| above 3 is",
             "unlikely to be luck."]
    insiders = pd.Series(np.select([ev.insiders >= 3, ev.insiders == 2], ["3+ insiders", "2 insiders"],
                                   "1 insider"), index=ev.index)
    size = bucket(ev.dollar_volume, SIZE_BUCKETS)
    lines += table(ev, pd.Series("all buying events", index=ev.index), "All buying events (the baseline)",
                   ["all buying events"], h)
    lines += table(ev, insiders, "By how many insiders bought (within 30 days)",
                   ["1 insider", "2 insiders", "3+ insiders"], h)
    lines += table(ev, bucket(ev.value, VALUE_BUCKETS), "By dollars bought (30 days)", [b[2] for b in VALUE_BUCKETS], h)
    lines += table(ev, ev.who, "By who bought", ["CEO/CFO/President bought", "other officers bought",
                                                 "directors only", "10% owners only"], h)
    lines += table(ev, size, "By company size (average dollars traded per day)", [b[2] for b in SIZE_BUCKETS], h)
    lines += table(ev, bucket(ev.prior_60d_pct, PRIOR_BUCKETS), "By what the stock did in the 60 days before",
                   [b[2] for b in PRIOR_BUCKETS], h)
    cluster = ev.insiders >= 2
    lines += table(ev[cluster], size[cluster], "Clusters (2+ insiders) by company size", [b[2] for b in SIZE_BUCKETS], h)
    top = ev.who == "CEO/CFO/President bought"
    lines += table(ev[top], bucket(ev.value, VALUE_BUCKETS)[top], "CEO/CFO/President buys by dollars",
                   [b[2] for b in VALUE_BUCKETS], h)
    years = pd.Series([str(d.year) for d in ev.date], index=ev.index)
    lines += table(ev[cluster], years[cluster], "Clusters (2+ insiders), year by year", sorted(set(years[cluster])), h)
    lines += ["", "Next: choose the pattern that is strong (big excess, |t| > 3, 100+ events) AND steady year by",
              "year, write it down, then run --confirm once to test it on the locked years."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true", help="show the locked years (2022 on); use once")
    ap.add_argument("--start", type=int, default=2016, help="first year of filings (default 2016)")
    args = ap.parse_args()

    from . import market_data
    from .backtest import BACKTEST_CACHE
    from .config import load_env_file
    from .insiders import load_purchases

    load_env_file()
    purchases = load_purchases(args.start)
    if purchases.empty:
        raise SystemExit("No insider purchases loaded: check the SEC download messages above.")
    events = build_events(purchases)
    tickers = sorted(set(events.ticker))
    years = datetime.now().year - args.start + 2
    print(f"{len(purchases)} insider purchases -> {len(events)} buying events in {len(tickers)} companies. "
          f"Downloading {years} years of prices for them (the first time: 15-30 minutes, then cached)...")
    bars = market_data.download_bars(sorted(set(tickers) | {"SPY", "IWM"}), period=f"{years}y",
                                     priority=["SPY", "IWM"], cache_dir=BACKTEST_CACHE)
    if "SPY" not in bars:
        raise SystemExit("No SPY data: Yahoo is throttling. Wait 15-30 minutes and run again.")
    ev = add_returns(events, bars)
    have = len(ev)
    coverage = (f"Coverage: {have} of {len(events)} events ({have / max(1, len(events)) * 100:.0f}%) have prices. "
                "Missing ones are mostly delisted, renamed or sub-$1 stocks.")
    ev["date"] = pd.to_datetime(ev["date"])
    part = ev[ev.date >= LOCK_FROM] if args.confirm else ev[ev.date < LOCK_FROM]
    text = report(part.reset_index(drop=True), locked=not args.confirm, coverage=coverage)
    print(text)
    out = DATA_DIR / "backtests"
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    (out / f"insider_study_{stamp}.txt").write_text(text)
    part.to_csv(out / f"insider_events_{stamp}.csv", index=False)
    print(f"\nSaved: data/backtests/insider_study_{stamp}.txt and insider_events_{stamp}.csv")


if __name__ == "__main__":
    main()
