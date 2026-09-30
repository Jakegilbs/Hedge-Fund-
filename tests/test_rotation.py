import numpy as np
import pandas as pd
import pytest

from desk import rotation as rot

from .conftest import make_bars

CFG = rot.RotationConfig(min_price=0, min_dollar_volume=0, cost_pct=0.0)


def panel(n=600, spy_drift=0.0008, drifts=None):
    drifts = drifts or {"WIN": 0.003, "MID": 0.001, "FLAT": 0.0, "LOSE": -0.002}
    bars = {"SPY": make_bars(n, drift=spy_drift, seed=1)}
    for k, (t, d) in enumerate(drifts.items()):
        bars[t] = make_bars(n, drift=d, seed=10 + k)
    return rot.Panel.from_bars(bars, list(drifts))


def test_rebalance_days_are_first_trading_days_of_the_month():
    idx = pd.bdate_range("2026-01-01", "2026-06-30")
    days = rot.rebalance_days(idx, 1)
    assert [idx[d].strftime("%m-%d") for d in days] == ["02-02", "03-02", "04-01", "05-01", "06-01"]
    assert len(rot.rebalance_days(idx, 1, months=2)) == 3


def test_picks_the_strongest_recent_gainers_with_positive_gains_only():
    p = panel()
    w = rot.target_weights(p, len(p.close) - 1, rot.RotationConfig(**{**CFG.__dict__, "top": 2}))
    assert list(w.index)[0] == "WIN" and "LOSE" not in w.index
    assert w.sum() == pytest.approx(1.0) or w.sum() < 1.0          # never more than fully invested
    # Only positive gainers: asking for all 4 leaves the losers' share in cash.
    w4 = rot.target_weights(p, len(p.close) - 1, rot.RotationConfig(**{**CFG.__dict__, "top": 4}))
    assert "LOSE" not in w4.index and w4.sum() <= 0.75 + 1e-9


def test_market_filter_goes_to_cash_below_spy_200_day_average():
    p = panel(spy_drift=-0.003)
    assert rot.target_weights(p, len(p.close) - 1, CFG).empty
    no_filter = rot.RotationConfig(**{**CFG.__dict__, "market_filter": False})
    assert not rot.target_weights(p, len(p.close) - 1, no_filter).empty


def test_one_stock_held_all_period_matches_its_price_change():
    bars = {"SPY": make_bars(400, drift=0.001, seed=1), "ONE": make_bars(400, drift=0.002, seed=3)}
    p = rot.Panel.from_bars(bars, ["ONE"])
    cfg = rot.RotationConfig(**{**CFG.__dict__, "top": 1, "lookback": 20, "market_filter": False,
                                "positive_only": False})
    r = rot.simulate(p, 250, cfg)
    first = rot.rebalance_days(p.close.index, 250)[0]
    expected = 100 * p.close["ONE"].iloc[-1] / p.open["ONE"].iloc[first]
    assert r.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_trading_costs_reduce_the_result():
    p = panel()
    free = rot.simulate(p, 300, rot.RotationConfig(**{**CFG.__dict__, "top": 2}))
    paid = rot.simulate(p, 300, rot.RotationConfig(**{**CFG.__dict__, "top": 2, "cost_pct": 0.01}))
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]


def test_no_lookahead_signal_uses_only_past_prices():
    """Changing prices after the signal day must not change that day's picks."""
    p = panel()
    s = 400
    before = rot.target_weights(p, s, CFG)
    p.close.iloc[s + 1:] = p.close.iloc[s + 1:] * np.linspace(0.1, 3, p.close.shape[1])
    assert rot.target_weights(p, s, CFG).equals(before)


def test_metrics_and_halves():
    idx = pd.bdate_range("2024-01-01", periods=500)
    e = pd.Series(np.linspace(100, 200, 500), index=idx)
    e.iloc[250] = 50                                   # a one-day crash
    m = rot.metrics(e)
    assert m["final"] == pytest.approx(200) and m["max_drawdown_pct"] < -45
    first, second = rot.halves(e)
    assert first.index[-1] == second.index[0] and len(first) + len(second) == 501


def test_grid_and_report_end_to_end():
    p = panel(800)
    configs = [c for c in rot.default_grid(CFG) if c.top in (1, 3) and c.lookback == 60]
    benches, results = rot.run_grid(p, 300, configs, log=lambda *a: None)
    text = rot.report(benches, results)
    assert "SPY buy and hold" in text and "Equal weight" in text and "edge?" in text
    cfg, r = results[0]
    assert "year by year" in rot.detail(cfg, r, benches[0], benches[1], p)


# ---------- historical membership, holdings log, data check ----------

from desk.membership import Membership, yahoo_symbol

HISTORY = '''date,tickers
2020-01-02,"AAPL,FB,BRK.B,OLD"
2023-06-01,"AAPL,FB,BRK.B,NEW"
2024-01-02,"AAPL,META,BRK.B,NEW"
'''


def test_membership_on_a_date_and_renames():
    m = Membership.from_csv_text(HISTORY)
    assert yahoo_symbol("FB") == "META" and yahoo_symbol("BRK.B") == "BRK-B"
    assert m.on("2021-05-05") == {"AAPL", "META", "BRK-B", "OLD"}      # FB mapped to today's ticker
    assert "NEW" in m.on("2023-06-01") and "OLD" not in m.on("2023-06-01")
    assert m.on("2019-01-01") == frozenset()
    assert m.ever_between("2021-01-01", "2024-06-01") == {"AAPL", "META", "BRK-B", "OLD", "NEW"}


def test_only_members_at_the_time_can_be_picked():
    p = panel(drifts={"WIN": 0.004, "MID": 0.003, "LOSE": -0.002})
    s = len(p.close) - 1
    top1 = rot.RotationConfig(**{**CFG.__dict__, "top": 1})
    assert list(rot.target_weights(p, s, top1).index) == ["WIN"]
    member = pd.DataFrame(True, index=p.close.index, columns=p.close.columns)
    member["WIN"] = False                                  # WIN was not in the index yet
    p2 = rot.Panel(p.close, p.open, p.dollar_volume, p.spy, member)
    assert list(rot.target_weights(p2, s, top1).index) == ["MID"]
    assert "WIN" not in rot.equal_weight_all(p2, s, CFG).index


def test_holdings_table_adds_up_and_flags_bad_data():
    p = panel()
    cfg = rot.RotationConfig(**{**CFG.__dict__, "top": 2})
    r = rot.simulate(p, 300, cfg)
    t = rot.holdings_table(p, r)
    assert set(t.columns) == {"date", "ticker", "weight", "return_pct", "contribution_pct"}
    assert len(t) and (t.groupby("date").weight.sum() <= 1.0001).all()
    first = t[t.date == t.date.iloc[0]]
    assert first.contribution_pct.sum() == pytest.approx((first.weight * first.return_pct).sum(), abs=0.01)
    assert rot.suspicious_moves(p, ["WIN"]) == []
    p.close.iloc[350, p.close.columns.get_loc("WIN")] *= 3              # a bad print
    assert any("WIN" in f for f in rot.suspicious_moves(p, ["WIN"]))
