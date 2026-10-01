import numpy as np
import pandas as pd
import pytest

from desk import support as sp

from .conftest import make_bars


def series_to_bars(closes, start="2020-01-01"):
    c = np.asarray(closes, float)
    idx = pd.bdate_range(start, periods=len(c))
    return pd.DataFrame({"Open": c, "High": c * 1.005, "Low": c * 0.995, "Close": c,
                         "Volume": 1e6}, index=idx)


def floor_chart():
    """Uptrend to 120, a floor at ~100 made twice, a rally to 115, then a dip back to 100."""
    up = np.linspace(80, 120, 230)
    dip1 = np.r_[np.linspace(120, 100.5, 20), np.linspace(100.5, 112, 20)]
    dip2 = np.r_[np.linspace(112, 100.6, 15), np.linspace(100.6, 115, 25)]
    dip3 = np.linspace(115, 100.2, 15)
    after = np.linspace(100.2, 112, 30)
    return series_to_bars(np.r_[up, dip1, dip2, dip3, after])


def test_swing_lows_and_retests_without_hindsight():
    df = floor_chart()
    lows = sp.swing_lows(df["Low"].to_numpy())
    assert len(lows) >= 2
    ev = [e for e in sp.find_events(df) if e["kind"] == "price floor"]
    assert ev, "the dip back to the floor should be found"
    last = ev[-1]
    assert abs(last["level"] / 100 - 1) < 0.02 and last["touches"] >= 2
    # every level used was confirmed (K days old) before the retest
    assert all(any(s + sp.K < e["t"] for s in lows) for e in ev)


def test_outcome_bounce_and_break():
    df = series_to_bars(np.r_[[100.0] * 30, np.linspace(100, 110, 20)])
    assert sp.outcome(df, 30, fill=100.0, stop=98.0, a=2.0) == "bounce"
    df2 = series_to_bars(np.r_[[100.0] * 30, np.linspace(100, 90, 20)])
    assert sp.outcome(df2, 30, fill=100.0, stop=98.0, a=2.0) == "break"


def test_regimes_labels():
    bull = sp.regimes(make_bars(400, drift=0.002, seed=1)).dropna()
    bear = sp.regimes(make_bars(400, drift=-0.003, seed=1)).dropna()
    assert set(bull.iloc[-50:]) <= {"bull calm", "bull volatile"}
    assert set(bear.iloc[-50:]) <= {"bear calm", "bear volatile"}


def test_study_and_report_and_scan():
    spy = make_bars(800, drift=0.0005, seed=1, end="2021-06-30")
    df = make_bars(800, drift=0.0008, seed=7, end="2021-06-30")
    rows = sp.study_ticker("X", df, spy, sp.regimes(spy))
    ev = pd.DataFrame(rows)
    assert {"random uptrend day"} <= set(ev.kind) and set(ev.outcome) <= {"bounce", "break", "neither"}
    text = sp.report(ev, locked=True)
    assert "By kind of support" in text and "by regime" in text
    chart = floor_chart()
    out = sp.scan({"SPY": spy, "FLR": chart}, ["FLR"], ["FLR"])
    assert "MARKET REGIME" in out


def test_current_levels_finds_the_floor_we_sit_on():
    df = floor_chart().iloc[:-30]                          # today = the bottom of the third dip
    kinds = [lv for lv in sp.current_levels(df) if lv["kind"] == "price floor"]
    assert kinds and abs(kinds[0]["level"] / 100 - 1) < 0.02 and kinds[0]["touches"] >= 2
