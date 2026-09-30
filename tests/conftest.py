import numpy as np
import pandas as pd
import pytest


def make_bars(n: int = 300, start: float = 100.0, drift: float = 0.001, seed: int = 0,
              end: str = "2026-09-29") -> pd.DataFrame:
    """Synthetic daily OHLCV bars ending on `end` (business days)."""
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + drift + rng.normal(0, 0.01, n))
    idx = pd.bdate_range(end=end, periods=n)
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float)}, index=idx)


def make_breakout(n: int = 300) -> pd.DataFrame:
    """Flat base, then the last bar closes at a new high on double volume."""
    df = make_bars(n, drift=0.0, seed=1)
    base_high = df["Close"].iloc[:-1].max()
    df.iloc[-1, df.columns.get_loc("Close")] = base_high * 1.03
    df.iloc[-1, df.columns.get_loc("High")] = base_high * 1.04
    df.iloc[-1, df.columns.get_loc("Volume")] = df["Volume"].iloc[-21:-1].mean() * 2.0
    return df


@pytest.fixture
def bars():
    return {"SPY": make_bars(seed=2), "QQQ": make_bars(seed=3), "AAPL": make_bars(seed=4),
            "NVDA": make_breakout(), "^VIX": make_bars(start=15, drift=0, seed=5)}
