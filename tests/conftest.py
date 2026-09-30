import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 300, start: float = 100.0, drift: float = 0.001, seed: int = 0,
              end: str = "2026-09-29") -> pd.DataFrame:
    """Synthetic daily OHLCV bars ending on `end` (business days)."""
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + drift + rng.normal(0, 0.01, n))
    idx = pd.bdate_range(end=end, periods=n)
    high = close * (1 + rng.uniform(0.004, 0.015, n))
    low = close * (1 - rng.uniform(0.004, 0.015, n))
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float)}, index=idx)


def make_breakout(n: int = 300) -> pd.DataFrame:
    """Steady uptrend, then the last bar closes 1% above the prior closing high on double volume."""
    df = make_bars(n, drift=0.0008, seed=11)
    prior_high = df["Close"].iloc[:-1].max()
    df.iloc[-1, df.columns.get_loc("Close")] = prior_high * 1.01
    df.iloc[-1, df.columns.get_loc("High")] = prior_high * 1.015
    df.iloc[-1, df.columns.get_loc("Volume")] = df["Volume"].iloc[-21:-1].mean() * 2.0
    return df


@pytest.fixture
def bars():
    # Only NVDA is a candidate: the others drift down, so they are neither setups nor leaders.
    return {"SPY": make_bars(drift=-0.001, seed=2), "QQQ": make_bars(drift=-0.001, seed=3),
            "AAPL": make_bars(drift=-0.002, seed=4),
            "NVDA": make_breakout(), "^VIX": make_bars(start=15, drift=0, seed=5)}
