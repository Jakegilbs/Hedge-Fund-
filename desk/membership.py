"""Historical S&P 500 membership: which stocks were in the index on each date.

A backtest that ranks TODAY's S&P 500 members in the past includes stocks that
only joined later because they had already risen, which flatters momentum most
of all. This module loads the free historical list maintained at
github.com/fja05680/sp500 (index changes since 1996), cached weekly in
data/universe/, so a backtest can rank only the stocks that were members then.

Old tickers are kept as they were (FB, ANTM); renames are mapped to today's
Yahoo tickers below so their price history is found. Companies that were
acquired or went bankrupt usually have no price data left on Yahoo: they are
counted as "no price data" in the coverage report.
"""
from __future__ import annotations

import io
from bisect import bisect_right
from dataclasses import dataclass

import pandas as pd

from .config import DATA_DIR

HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
CACHE = DATA_DIR / "universe" / "sp500_history.csv"
MAX_AGE_SECONDS = 7 * 24 * 3600

# Ticker changes (same company, new symbol): the historical list keeps the old one.
RENAMES = {
    "FB": "META", "ANTM": "ELV", "FISV": "FI", "ABC": "COR", "RE": "EG", "PKI": "RVTY",
    "FLT": "CPAY", "WLTW": "WTW", "CDAY": "DAY", "BLL": "BALL", "HFC": "DINO", "NLOK": "GEN",
    "SYMC": "GEN", "COG": "CTRA", "ADS": "BFH", "PEAK": "DOC", "GPS": "GAP", "CTL": "LUMN",
    "FBHS": "FBIN",
}


def yahoo_symbol(ticker: str) -> str:
    t = str(ticker).strip().upper()
    return RENAMES.get(t, t).replace(".", "-")        # BRK.B -> BRK-B


@dataclass
class Membership:
    dates: list[pd.Timestamp]
    members: list[frozenset[str]]

    @classmethod
    def from_csv_text(cls, text: str) -> "Membership":
        df = pd.read_csv(io.StringIO(text))
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return cls(list(df["date"]),
                   [frozenset(yahoo_symbol(t) for t in str(s).split(",") if t.strip()) for s in df["tickers"]])

    def on(self, day) -> frozenset[str]:
        """Members on `day` (the latest list dated on or before it)."""
        i = bisect_right(self.dates, pd.Timestamp(day)) - 1
        return self.members[i] if i >= 0 else frozenset()

    def ever_between(self, start, end) -> set[str]:
        """Every stock that was a member at some point between start and end."""
        out = set(self.on(start))
        for d, m in zip(self.dates, self.members):
            if pd.Timestamp(start) < d <= pd.Timestamp(end):
                out |= m
        return out

    def mask(self, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
        """True where the column's stock was a member on that trading day."""
        cols = list(columns)
        rows = []
        for day in index:
            m = self.on(day)
            rows.append([c in m for c in cols])
        return pd.DataFrame(rows, index=index, columns=cols)


def load_membership(log=print) -> Membership | None:
    """The historical list, downloaded at most weekly; None if it was never downloaded."""
    import time
    import urllib.request

    fresh = CACHE.is_file() and time.time() - CACHE.stat().st_mtime < MAX_AGE_SECONDS
    if not fresh:
        try:
            with urllib.request.urlopen(HISTORY_URL, timeout=30) as r:
                text = r.read().decode("utf-8")
            if not text.startswith("date,tickers"):
                raise ValueError("unexpected file format")
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(text)
        except Exception as e:
            log(f"Could not download the historical S&P 500 list ({e}); using the cached copy if any.")
    if not CACHE.is_file():
        return None
    return Membership.from_csv_text(CACHE.read_text())
