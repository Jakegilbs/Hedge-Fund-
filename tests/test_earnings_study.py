import numpy as np
import pandas as pd
import pytest

from desk import earnings_study as es

from .conftest import make_bars


def flat_bars(n=400, price=100.0, end="2021-06-30"):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price,
                         "Volume": 1_000_000.0}, index=idx)


def test_event_measures_reaction_and_what_happened_next():
    spy = flat_bars()
    stock = flat_bars()
    idx = stock.index
    report = idx[300]
    stock.loc[idx[301]:, ["Open", "High", "Low", "Close"]] = 110.0     # +10% reaction on the day after
    stock.loc[idx[302]:, "Open"] = 110.0
    stock.loc[idx[301 + 20]:, ["Close"]] = 121.0                         # then +10% more by 20 days
    stock.loc[idx[301], "Volume"] = 4_000_000.0
    earn = {"UP": pd.DataFrame({"date": [report], "eps_estimate": [1.0], "eps_actual": [1.2],
                                "surprise_pct": [20.0]})}
    ev = es.build_events({"SPY": spy, "UP": stock}, earn)
    [row] = ev.to_dict("records")
    assert row["reaction_pct"] == pytest.approx(10.0)
    assert row["entry_date"] == idx[302].date()                         # buy the open AFTER the reaction
    assert row["ret_5"] == pytest.approx(0.0) and row["excess_20"] == pytest.approx(10.0)
    assert row["volume_x"] == pytest.approx(4.0) and row["above_200d"] is False


def test_only_index_members_count_and_short_history_is_skipped():
    spy, stock = flat_bars(), flat_bars()
    earn = {"X": pd.DataFrame({"date": [stock.index[300]], "eps_estimate": [1.0], "eps_actual": [1.1],
                               "surprise_pct": [10.0]})}
    assert es.build_events({"SPY": spy, "X": stock}, earn, member_on=lambda d: set()).empty
    early = {"X": pd.DataFrame({"date": [stock.index[50]], "eps_estimate": [1.0], "eps_actual": [1.1],
                                "surprise_pct": [10.0]})}
    assert es.build_events({"SPY": spy, "X": stock}, early).empty     # not enough history before it


def test_report_tables_and_buckets():
    rng = np.random.default_rng(0)
    n = 400
    ev = pd.DataFrame({"ticker": [f"T{i % 50}" for i in range(n)],
                       "date": [d.date() for d in pd.bdate_range("2017-01-02", periods=n, freq="5B")],
                       "surprise_pct": rng.normal(3, 10, n), "reaction_pct": rng.normal(0, 6, n),
                       "volume_x": rng.uniform(1, 5, n), "runup_20d_pct": rng.normal(0, 8, n),
                       "above_200d": rng.random(n) > 0.4})
    for h in es.HORIZONS:
        ev[f"excess_{h}"] = rng.normal(0, 5, n)
    text = es.report(ev, locked=True)
    for part in ("All reports", "By EPS surprise", "Beat or miss x reaction", "year by year", "2022+ locked"):
        assert part in text
    b = es.bucket(pd.Series([-10.0, 0.5, 7.0]), es.REACTION_BUCKETS)
    assert list(b) == ["fell >5%", "flat (+/-1%)", "rose >5%"]
