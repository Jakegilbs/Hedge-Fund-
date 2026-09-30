import math

import numpy as np
import pandas as pd
import pytest

from desk import backtest as bt
from desk.indicators import SETUPS
from desk.options import bs_delta

from .conftest import make_bars, make_breakout

CFG = bt.BTConfig(min_price=0, min_dollar_volume=0)


@pytest.mark.parametrize("df", [make_bars(420, drift=0.0015, seed=1), make_bars(420, drift=-0.0015, seed=2),
                                make_bars(420, drift=0.0, seed=3), make_breakout(300)])
def test_signal_frame_matches_live_setup_rules(df):
    """Vectorised rules must fire on exactly the bars (and levels) the live scanner would."""
    sig = bt.signal_frame(df)
    fired = 0
    for t in range(150, len(df)):
        live = {s["setup"]: s for s in (fn(df.iloc[:t + 1]) for fn in SETUPS) if s}
        vec = {name for name in bt.SETUP_NAMES if sig[name].iloc[t]}
        assert vec == set(live), (t, vec, set(live))
        for name, s in live.items():
            assert sig[f"{name}_entry"].iloc[t] == pytest.approx(s["entry"], abs=0.011)
            assert sig[f"{name}_stop"].iloc[t] == pytest.approx(s["stop"], abs=0.011)
        fired += len(live)
    assert fired > 0


def arrays(rows):
    o, h, l, c = (np.array(x, float) for x in zip(*rows))
    return o, h, l, c


def test_trigger_fills_only_when_it_trades():
    # day 0 signal: close 100, trigger 102. Day 1 high 101 (no fill), trigger_days=2: day 2 trades 103.
    o, h, l, c = arrays([(100, 100, 99, 100), (100, 101, 99, 100), (101, 103, 100, 102.5)])
    cfg = bt.BTConfig(trigger_days=1)
    assert bt.find_fill(o, h, l, c, 0, "bullish", 102.0, cfg) is None
    fill = bt.find_fill(o, h, l, c, 0, "bullish", 102.0, bt.BTConfig(trigger_days=2))
    assert (fill.day, fill.price) == (2, 102.0)


def test_gap_past_the_stop_limit_does_not_fill():
    # Opens 3% above the trigger and never comes back to the 0.5% limit.
    o, h, l, c = arrays([(100, 100, 99, 100), (105, 106, 104, 105)])
    assert bt.find_fill(o, h, l, c, 0, "bullish", 102.0, CFG) is None


def test_bearish_trigger_fills_on_breakdown():
    o, h, l, c = arrays([(100, 100, 99, 100), (99.5, 99.6, 97, 97.5)])
    fill = bt.find_fill(o, h, l, c, 0, "bearish", 98.0, CFG)
    assert (fill.day, fill.price) == (1, 98.0)


def test_shares_exit_at_stop_target_and_time():
    fill = bt.Fill(0, 100.0)
    cfg = bt.BTConfig(target_r=2.0, max_hold_days=3, slippage_pct=0.0)
    # stop 95, target 110
    stop = bt.simulate_shares(*arrays([(100, 101, 99, 100), (99, 100, 94, 95)]), fill, "bullish", 100, 95, cfg)
    assert stop["exit_reason"] == "stop" and stop["return_pct"] == pytest.approx(-5.0)
    assert stop["r_multiple"] == pytest.approx(-1.0)
    win = bt.simulate_shares(*arrays([(100, 101, 99, 100), (104, 111, 103, 110)]), fill, "bullish", 100, 95, cfg)
    assert win["exit_reason"] == "target" and win["r_multiple"] == pytest.approx(2.0)
    rows = [(100, 101, 99, 100)] + [(101, 102, 100, 101)] * 4
    timed = bt.simulate_shares(*arrays(rows), fill, "bullish", 100, 95, cfg)
    assert timed["exit_reason"] == "time" and timed["days_held"] == 3
    # Bearish: short 100, stop 105, target 90.
    short = bt.simulate_shares(*arrays([(100, 101, 99, 100), (96, 97, 89, 90)]), fill, "bearish", 100, 105, cfg)
    assert short["exit_reason"] == "target" and short["return_pct"] == pytest.approx(10.0)


def test_stop_is_assumed_when_both_levels_trade_the_same_day():
    fill = bt.Fill(0, 100.0)
    out = bt.simulate_shares(*arrays([(100, 101, 99, 100), (100, 112, 94, 100)]), fill, "bullish", 100, 95,
                             bt.BTConfig(slippage_pct=0.0))
    assert out["exit_reason"] == "stop"


def test_black_scholes_and_strike_for_delta():
    call = bt.bs_price(100, 100, 0.25, 0.3, 0.045, True)
    put = bt.bs_price(100, 100, 0.25, 0.3, 0.045, False)
    assert call - put == pytest.approx(100 - 100 * math.exp(-0.045 * 0.25), abs=1e-6)   # put-call parity
    for is_call in (True, False):
        k = bt.strike_for_delta(50, 35 / 365, 0.6, 0.045, 0.55, is_call)
        assert abs(bs_delta(50, k, 35 / 365, 0.6, 0.045, is_call)) == pytest.approx(0.55, abs=1e-6)


def test_option_take_profit_and_stop():
    dates = list(pd.bdate_range("2026-01-05", periods=10))
    up = [(20, 20.2, 19.9, 20)] + [(20 + i, 20.5 + i, 19.8 + i, 20.4 + i) for i in range(1, 10)]
    out = bt.simulate_option(*arrays(up), dates, bt.Fill(0, 20.0), "bullish", 0.5, CFG)
    assert out["opt_exit_reason"].startswith("take profit") and out["opt_return_pct"] >= 99.9
    down = [(20, 20.2, 19.9, 20), (19.95, 20.0, 18.5, 18.8)] + [(18.8, 19, 18.6, 18.8)] * 8
    out = bt.simulate_option(*arrays(down), dates, bt.Fill(0, 20.0), "bullish", 0.5, CFG)
    assert out["opt_exit_reason"] == "stop" and out["opt_return_pct"] == pytest.approx(-15.0)
    # A put profits from the same fall.
    put = bt.simulate_option(*arrays(down), dates, bt.Fill(0, 20.0), "bearish", 0.5, CFG)
    assert put["opt_return_pct"] > 0


def _signal(day, ticker, score, filled=True, ret=10.0, cost=50.0, exit_day=None, direction="bullish"):
    d = pd.Timestamp(day).date()
    return {"ticker": ticker, "setup": "breakout", "direction": direction, "signal_date": d,
            "rank_score": score, "market_up": True, "shares_ok": True, "opt_cost_est": cost,
            "filled": filled, "opt_cost": cost if filled else np.nan,
            "opt_return_pct": ret if filled else np.nan, "opt_exit_reason": "take profit" if filled else None,
            "opt_exit_date": pd.Timestamp(exit_day or day).date() if filled else None,
            "return_pct": ret / 5 if filled else np.nan, "exit_reason": "target" if filled else None,
            "exit_date": pd.Timestamp(exit_day or day).date() if filled else None}


def test_portfolio_never_peeks_at_which_orders_filled():
    # The top-ranked signal never triggered: that day is a no-trade, not a jump to the next one.
    trades = pd.DataFrame([_signal("2026-01-05", "AAA", 9.0, filled=False),
                           _signal("2026-01-05", "BBB", 1.0, ret=100.0)])
    res = bt.run_portfolio(trades, bt.BTConfig(ai_cost_per_run=0.0))
    assert res.trades == 0 and res.final_equity == 100.0


def test_portfolio_is_all_in_one_at_a_time_and_compounds():
    trades = pd.DataFrame([
        _signal("2026-01-05", "AAA", 5.0, ret=100.0, cost=40.0, exit_day="2026-01-09"),
        _signal("2026-01-07", "BBB", 9.0, ret=-15.0, cost=40.0),              # busy: skipped
        _signal("2026-01-12", "CCC", 5.0, ret=-15.0, cost=300.0),            # unaffordable call -> shares
    ])
    res = bt.run_portfolio(trades, bt.BTConfig(ai_cost_per_run=0.1))
    # 2 contracts x $40 = $80 invested, +100% -> $180; then shares -3% -> $174.60
    assert res.log[0]["instrument"] == "option" and res.log[0]["equity"] == 180.0
    assert res.log[1]["instrument"] == "shares" and res.final_equity == pytest.approx(174.6)
    assert res.trades == 2 and res.ai_cost == pytest.approx(0.3)


def test_bearish_needs_an_affordable_put():
    trades = pd.DataFrame([_signal("2026-01-05", "AAA", 5.0, cost=500.0, direction="bearish")])
    assert bt.run_portfolio(trades, bt.BTConfig()).trades == 0


def test_end_to_end_on_synthetic_history():
    bars = {"SPY": make_bars(700, drift=0.0005, seed=2),
            "UPP": make_bars(700, drift=0.0015, seed=5), "DWN": make_bars(700, drift=-0.0015, seed=6)}
    trades, filtered = bt.run_backtest(bars, ["UPP", "DWN"], CFG, years=2, log=lambda *a: None)
    assert len(trades) and trades.filled.any()
    assert {"return_pct", "opt_return_pct", "rank_score"} <= set(trades.columns)
    ports = [bt.run_portfolio(trades, CFG, "hybrid"), bt.run_portfolio(trades, CFG, "shares", True)]
    text = bt.report(trades, filtered, CFG, ports, 2)
    assert "EXPECTANCY" in text and "THE $100 ACCOUNT" in text


def test_option_stop_can_follow_the_stock_stop():
    dates = list(pd.bdate_range("2026-01-05", periods=10))
    # A 3% dip: the -15% option stop fires, the stock stop at 18.5 does not.
    dip = [(20, 20.2, 19.9, 20), (19.9, 20.0, 19.4, 19.6)] + [(19.6 + i * 0.4, 20 + i * 0.4, 19.5 + i * 0.4, 19.9 + i * 0.4)
                                                              for i in range(8)]
    pct = bt.simulate_option(*arrays(dip), dates, bt.Fill(0, 20.0), "bullish", 0.5, CFG, stock_stop=18.5)
    assert pct["opt_exit_reason"] == "stop"
    cfg = bt.BTConfig(option_stop_on_stock=True)
    stock = bt.simulate_option(*arrays(dip), dates, bt.Fill(0, 20.0), "bullish", 0.5, cfg, stock_stop=18.5)
    assert stock["opt_exit_reason"] != "stop" and stock["opt_return_pct"] > 0
    crash = [(20, 20.2, 19.9, 20), (19.9, 20.0, 18.0, 18.2)] + [(18.2, 18.3, 18.1, 18.2)] * 8
    out = bt.simulate_option(*arrays(crash), dates, bt.Fill(0, 20.0), "bullish", 0.5, cfg, stock_stop=18.5)
    assert out["opt_exit_reason"] == "stop" and out["opt_return_pct"] < -15


def test_select_trades_filters_setups_market_and_half():
    rows = [_signal("2026-01-05", "AAA", 1.0), _signal("2026-07-05", "BBB", 1.0, direction="bearish"),
            _signal("2026-12-05", "CCC", 1.0)]
    rows[2]["setup"] = "pullback"
    t = pd.DataFrame(rows)
    assert list(bt.select_trades(t, ["pullback"]).ticker) == ["CCC"]
    assert list(bt.select_trades(t, with_market=True).ticker) == ["AAA", "CCC"]   # bearish while market up: out
    assert list(bt.select_trades(t, half="first").ticker) == ["AAA"]
    assert list(bt.select_trades(t, half="second").ticker) == ["BBB", "CCC"]
