from desk.indicators import rsi, setup_breakout, snapshot
from desk.market_data import technical_packet
from desk.scanner import pick_candidates

from .conftest import make_bars, make_breakout


def test_breakout_detected_with_levels():
    hit = setup_breakout(make_breakout())
    assert hit and hit["setup"] == "breakout"
    assert hit["stop"] < hit["entry"]


def test_no_breakout_in_random_walk():
    assert setup_breakout(make_bars(drift=-0.002, seed=9)) is None


def test_rsi_bounds():
    r = rsi(make_bars()["Close"])
    assert r.between(0, 100).all()


def test_snapshot_fields_are_json_friendly():
    s = snapshot(make_bars())
    assert s["last_date"] == "2026-09-29"
    assert len(s["last_10_closes"]) == 10
    assert isinstance(s["close"], float)


def test_stale_ticker_is_flagged_and_not_a_candidate():
    bars = {"SPY": make_bars(), "NVDA": make_breakout(), "OLD": make_bars(end="2026-09-01")}
    packet = technical_packet(bars, ["NVDA", "OLD", "MISSING"], "2026-09-29")
    assert packet["tickers"]["OLD"]["stale"] is True
    assert packet["tickers"]["NVDA"]["stale"] is False
    assert packet["missing_data"] == ["MISSING"]
    assert pick_candidates(packet["tickers"], held=[]).candidates == ["NVDA"]


def test_held_positions_always_reviewed():
    packet = technical_packet({"AAPL": make_bars()}, ["AAPL"], "2026-09-29")
    assert pick_candidates(packet["tickers"], held=["AAPL"]).candidates == ["AAPL"]


def _snap(chg60, price=100.0, dollar_vol=1e9, setups=1):
    return {"setups": [{"setup": "breakout"}] * setups, "stale": False, "close": price,
            "avg_dollar_volume_20d": dollar_vol, "change_pct": {"60d": chg60}}


def test_scanner_ranks_by_strength_and_caps_the_list():
    snaps = {f"T{i}": _snap(chg60=i) for i in range(12)}
    scan = pick_candidates(snaps, held=[], max_candidates=3)
    assert scan.candidates == ["T10", "T11", "T9"]
    assert len(scan.with_setups) == 12 and len(scan.cut_by_limit) == 9


def test_scanner_filters_illiquid_and_cheap():
    snaps = {"CHEAP": _snap(50, price=4), "THIN": _snap(50, dollar_vol=1e6), "OK": _snap(1)}
    scan = pick_candidates(snaps, held=[], min_price=10, min_dollar_volume=5e7)
    assert scan.candidates == ["OK"] and sorted(scan.filtered_illiquid) == ["CHEAP", "THIN"]


def test_trend_facts_computed_in_code():
    from desk.indicators import trend_facts
    rising = make_bars(drift=0.004, seed=3)["Close"]
    facts = trend_facts(rising)
    assert facts["close_above_sma200"] is True and facts["trend"] == "up"


def test_incomplete_bar_dropped_during_market_hours():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from desk.market_data import split_incomplete_bar
    bars = {"SPY": make_bars(end="2026-09-30")}
    et = ZoneInfo("America/New_York")
    done, live = split_incomplete_bar(bars, datetime(2026, 9, 30, 14, 12, tzinfo=et))
    assert str(done["SPY"].index[-1].date()) == "2026-09-29" and "SPY" in live
    done, live = split_incomplete_bar(bars, datetime(2026, 9, 30, 16, 30, tzinfo=et))
    assert str(done["SPY"].index[-1].date()) == "2026-09-30" and live == {}
