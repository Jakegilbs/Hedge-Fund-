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
    assert pick_candidates(packet["tickers"], held=[]) == ["NVDA"]


def test_held_positions_always_reviewed():
    packet = technical_packet({"AAPL": make_bars()}, ["AAPL"], "2026-09-29")
    assert pick_candidates(packet["tickers"], held=["AAPL"]) == ["AAPL"]
