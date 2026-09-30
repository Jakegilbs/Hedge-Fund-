"""Options mode: contract selection, the options Gatekeeper and the full team run (all offline)."""
from datetime import date
from types import SimpleNamespace

import pandas as pd

from desk.config import load_settings
from desk.options import OptionsConfig, bs_delta, select_contract
from desk.schemas import (CatalystReport, CatalystView, OptionOrder, OptionsDecision, RegimeReport,
                          TechnicalReport, TechnicalView)
from desk.team import AccountState, run_team

TODAY = date(2026, 9, 30)
CFG = OptionsConfig()


def chain(spot=20.0, expiry="2026-11-06"):
    """A small call/put chain around `spot` with realistic prices and 40% IV."""
    rows = []
    for k in (spot * 0.85, spot * 0.95, spot, spot * 1.05, spot * 1.2):
        intrinsic = max(spot - k, 0)
        price = round(intrinsic + spot * 0.04, 2)
        rows.append({"contractSymbol": f"X{expiry}{k:.1f}", "strike": round(k, 1), "bid": round(price * 0.97, 2),
                     "ask": round(price * 1.03, 2), "openInterest": 500, "impliedVolatility": 0.40})
    return {expiry: pd.DataFrame(rows)}


def test_black_scholes_delta_sane():
    assert 0.5 < bs_delta(100, 100, 30 / 365, 0.3, 0.045, True) < 0.6
    assert -0.5 < bs_delta(100, 100, 30 / 365, 0.3, 0.045, False) < -0.4


def test_picks_near_the_money_affordable_call():
    c, note = select_contract(chain(), "bullish", 20.0, cash=100, cfg=CFG, today=TODAY)
    assert note == "ok" and c["type"] == "call"
    assert CFG.min_delta <= c["delta"] <= CFG.max_delta and c["cost_per_contract"] <= 99.5


def test_too_expensive_explains_why():
    c, note = select_contract(chain(spot=300.0), "bullish", 300.0, cash=100, cfg=CFG, today=TODAY)
    assert c is None and "per contract" in note


def test_skips_expiries_after_earnings():
    c, note = select_contract(chain(), "bullish", 20.0, cash=100, cfg=CFG, today=TODAY, earnings="2026-10-20")
    assert c is None and "expiry after earnings" in note


def test_skips_wide_spreads_and_thin_open_interest():
    df = chain()["2026-11-06"].assign(openInterest=5)
    c, _ = select_contract({"2026-11-06": df}, "bullish", 20.0, cash=100, cfg=CFG, today=TODAY)
    assert c is None


# ---------- full team run in options mode ----------

def reports(direction="bullish", posture="cautious", conviction=4, event_risk="low", contract=None,
            mode="options", instrument="option", spot=20.0, entry_mult=1.0, stop_mult=None, target_mult=None):
    tech = TechnicalReport(views=[TechnicalView(
        ticker="CHEAP", data_ok=True, trend="up" if direction == "bullish" else "down", direction=direction,
        setup="breakout" if direction == "bullish" else "breakdown", setup_quality=4,
        entry=round(spot * entry_mult, 2),
        stop=round(spot * (stop_mult or (0.95 if direction == "bullish" else 1.05)), 2),
        target=round(spot * (target_mult or (1.15 if direction == "bullish" else 0.85)), 2),
        key_levels=[], evidence="e", risks="r", recommendation="candidate")], warnings=[])
    news = CatalystReport(views=[CatalystView(ticker="CHEAP", data_ok=True, next_earnings=None,
                                              days_to_earnings=None, event_risk=event_risk, sentiment="neutral",
                                              catalysts=[], red_flags=[], summary="")], warnings=[])
    regime = RegimeReport(regime="neutral", posture=posture, max_new_positions_today=1, evidence=[],
                          leading_sectors=[], lagging_sectors=[], summary="", warnings=[])
    if mode == "hybrid":
        from desk.schemas import HybridDecision, HybridOrder
        pm = HybridDecision(market_view="", orders=[HybridOrder(
            ticker="CHEAP", direction=direction, instrument=instrument,
            contract_symbol=None if instrument == "shares" else (contract or "PLACEHOLDER"), thesis="t",
            bear_case="b", conviction=conviction)], position_updates=[], warnings=[], honest_assessment="")
    else:
        pm = OptionsDecision(market_view="", orders=[OptionOrder(
            ticker="CHEAP", direction=direction, contract_symbol=contract or "PLACEHOLDER", thesis="t",
            bear_case="b", conviction=conviction)], position_updates=[], warnings=[], honest_assessment="")
    return {m.__class__.__name__: m for m in (tech, news, regime, pm)}


class FakeClaude:
    def __init__(self, by_title):
        self.by_title, self.calls = by_title, []
        self.beta = SimpleNamespace(messages=self)

    def create(self, **kw):
        title = kw["output_config"]["format"]["schema"]["title"]
        self.calls.append(title)
        return SimpleNamespace(model=kw["model"], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=50),
                               content=[SimpleNamespace(type="text", text=self.by_title[title].model_dump_json())])


def legacy(mode):
    """The rules before the backtest: puts allowed, -15% option stop, no market filter."""
    from dataclasses import replace
    s = load_settings()
    return replace(s, instrument=mode, bullish_only=False, market_filter=False,
                   options=replace(s.options, stop_on_stock=False))


def run_options(bars, mode="options", fetch_empty=False, settings=None, **kw):
    from desk.llm import ClaudeRunner
    settings = settings or legacy(mode)
    spot = float(bars["CHEAP"]["Close"].iloc[-1])

    def fake_chains(ticker, cfg, today, max_expiries=None):
        c = {} if fetch_empty else chain(spot=spot)
        return {"calls": c, "puts": c}

    # Find the contract code will offer, then have the PM pick it.
    offered, note = select_contract(chain(spot=spot), kw.get("direction", "bullish"), spot, 100,
                                    settings.options, TODAY)
    assert offered, note
    if fetch_empty:
        offered = {"contract_symbol": None}
    claude = FakeClaude(reports(contract=kw.pop("contract", offered["contract_symbol"]), mode=mode,
                                spot=round(spot, 2), **kw))
    news = lambda tickers, etfs=frozenset(): {"today": "2026-09-30", "tickers": {}}
    from datetime import datetime
    from zoneinfo import ZoneInfo
    run = run_team(settings, ClaudeRunner(client=claude), AccountState(100, 100), bars, fetch_catalysts=news,
                   tickers=["CHEAP"], universe=list(settings.allowlist) + ["CHEAP"],
                   now=datetime(2026, 9, 30, 17, 0, tzinfo=ZoneInfo("America/New_York")),
                   fetch_option_chains=fake_chains)
    return run, claude


def cheap_bars():
    from .conftest import make_bars
    return {"SPY": make_bars(seed=2), "CHEAP": make_bars(start=15.0, drift=0.0, seed=7)}


def test_options_run_approves_one_all_in_call():
    run, claude = run_options(cheap_bars())
    assert claude.calls[-1] == "OptionsDecision"
    [o] = run.gate.approved
    assert o.option_type == "call" and o.contracts >= 1 and o.cost_usd <= 99.5
    assert o.stop_price == round(o.limit_price * 0.85, 2)          # 15% option stop
    assert o.take_profit_price == round(o.limit_price * 2.0, 2)    # +100%
    assert o.exit_by < o.expiry


def test_puts_allowed_in_flat_regime_but_calls_are_not():
    run, _ = run_options(cheap_bars(), direction="bearish", posture="flat")
    assert run.gate.approved and run.gate.approved[0].option_type == "put"
    run, _ = run_options(cheap_bars(), direction="bullish", posture="flat")
    assert not run.gate.approved and "flat" in " ".join(run.gate.rejected[0].reasons)


def test_invented_contract_rejected():
    run, _ = run_options(cheap_bars(), contract="MADEUP123")
    assert not run.gate.approved and "not the one on the menu" in " ".join(run.gate.rejected[0].reasons)


def test_low_conviction_and_event_risk_rejected():
    run, _ = run_options(cheap_bars(), conviction=2)
    assert not run.gate.approved
    run, _ = run_options(cheap_bars(), event_risk="high")
    assert not run.gate.approved and "event risk" in " ".join(run.gate.rejected[0].reasons)


def test_blank_yahoo_fields_are_ignored_not_crashing():
    df = chain()["2026-11-06"].copy()
    df.loc[0, "openInterest"] = float("nan")
    df.loc[1, "bid"] = float("nan")
    c, note = select_contract({"2026-11-06": df}, "bullish", 20.0, cash=100, cfg=CFG, today=TODAY)
    assert c is None or c["contract_symbol"] not in (df.loc[0, "contractSymbol"], df.loc[1, "contractSymbol"])


def test_estimated_atm_cost_tracks_real_prices():
    from desk.options import estimated_atm_cost
    # A $20 stock moving ~3% a day: roughly $100 for a 5-week near-the-money option.
    assert 60 <= estimated_atm_cost(20.0, 3.0) <= 140
    # A $190 stock moving ~3.6% a day: roughly $1,000+.
    assert estimated_atm_cost(190.0, 3.6) > 800


def test_scanner_skips_tickers_whose_options_are_unaffordable():
    from desk.scanner import pick_candidates
    from desk.options import estimated_atm_cost
    base = {"setups": [{"setup": "pullback", "direction": "bullish"}], "stale": False,
            "avg_dollar_volume_20d": 1e9, "change_pct": {"20d": 2, "60d": 5}, "rsi14": 55.0,
            "trend_facts": {"trend": "up"}, "pct_from_52w_high": -2.0}
    snaps = {"PRICEY": {**base, "close": 190.0, "atr_pct": 3.6}, "CHEAP": {**base, "close": 12.0, "atr_pct": 3.0}}
    ok = lambda s, d="bullish": estimated_atm_cost(s["close"], s["atr_pct"]) <= 100
    scan = pick_candidates(snaps, held=[], affordable=ok)
    assert scan.candidates == ["CHEAP"] and scan.filtered_unaffordable == ["PRICEY"]


def test_options_universe_adds_cheap_names_and_lower_price_floor(monkeypatch):
    from dataclasses import replace
    from desk import universe
    monkeypatch.setattr(universe, "sp500_symbols", lambda: ["AAPL"])
    monkeypatch.setattr(universe, "cheap_nasdaq", lambda settings, log=print: ["CHEAPQ"])
    s = load_settings()
    assert "CHEAPQ" in universe.load_universe(s)
    assert "SOFI" in universe.load_universe(s) and s.scan_min_price == 5.0
    stock = replace(s, instrument="stock")
    assert "SOFI" not in universe.load_universe(stock) and stock.scan_min_price == stock.min_price


NASDAQ_FILE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
SOFI|SoFi Technologies, Inc. - Common Stock|Q|N|N|100|N|N
TQQQ|ProShares UltraPro QQQ|G|N|N|100|Y|N
ZZZW|Some Co - Warrant|G|N|N|100|N|N
TINY|Tiny Corp - Common Stock|S|N|N|100|N|N
TEST|Test Issue Corp - Common Stock|Q|Y|N|100|N|N
PLUG|Plug Power, Inc. - Common Stock|G|N|N|100|N|N
File Creation Time: 1001202612:00|||||||"""


def test_nasdaq_file_keeps_only_common_stocks_on_main_tiers():
    from desk.universe import parse_nasdaq_listed
    assert parse_nasdaq_listed(NASDAQ_FILE) == ["PLUG", "SOFI"]


def test_cheap_screen_price_band_and_liquidity():
    from desk.universe import screen_cheap
    from .conftest import make_bars
    fake = {"CHEAP": make_bars(n=22, start=12.0, drift=0, seed=1),
            "PRICEY": make_bars(n=22, start=150.0, drift=0, seed=2),
            "THIN": make_bars(n=22, start=12.0, drift=0, seed=3).assign(Volume=1000.0)}
    kept = screen_cheap(list(fake), 5, 30, 5_000_000, fetch=lambda syms: {s: fake[s] for s in syms},
                        pause=0, log=lambda *a: None)
    assert kept == ["CHEAP"]


def test_cheap_nasdaq_uses_weekly_cache(tmp_path, monkeypatch):
    from desk import universe
    monkeypatch.setattr(universe, "NASDAQ_CHEAP_CACHE", tmp_path / "cheap.csv")
    monkeypatch.setattr(universe, "nasdaq_symbols", lambda: ["AAA", "BBB"])
    calls = []
    monkeypatch.setattr(universe, "screen_cheap", lambda syms, *a, **k: calls.append(syms) or ["AAA"])
    s = load_settings()
    assert universe.cheap_nasdaq(s, log=lambda *a: None) == ["AAA"]
    assert universe.cheap_nasdaq(s, log=lambda *a: None) == ["AAA"] and len(calls) == 1


def test_empty_quotes_and_missing_expiries_are_explained():
    empty = {"2026-11-06": pd.DataFrame()}
    c, note = select_contract(empty, "bearish", 20.0, cash=100, cfg=CFG, today=TODAY)
    assert c is None and "no put quotes" in note
    c, note = select_contract({"2026-10-02": pd.DataFrame()}, "bearish", 20.0, cash=100, cfg=CFG, today=TODAY)
    assert c is None and "expiries offered: 2026-10-02" in note


def test_precheck_finds_liquid_contract_or_not():
    from desk.options import has_tradeable_contract
    good = lambda t, cfg, today, max_expiries=None: {"calls": chain(), "puts": chain()}
    thin = lambda t, cfg, today, max_expiries=None: {"calls": {k: v.assign(openInterest=1) for k, v in chain().items()},
                                                     "puts": {}}
    assert has_tradeable_contract("X", "bullish", 20.0, 100, CFG, TODAY, fetch=good)
    assert not has_tradeable_contract("X", "bullish", 20.0, 100, CFG, TODAY, fetch=thin)
    boom = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("yahoo down"))
    assert not has_tradeable_contract("X", "bullish", 20.0, 100, CFG, TODAY, fetch=boom)


def test_scanner_skips_names_without_liquid_options_in_rank_order():
    from desk.scanner import pick_candidates
    base = {"setups": [{"setup": "pullback", "direction": "bullish"}], "stale": False, "close": 12.0,
            "avg_dollar_volume_20d": 1e9, "rsi14": 55.0, "atr_pct": 3.0,
            "trend_facts": {"trend": "up"}, "pct_from_52w_high": -2.0}
    snaps = {f"S{i}": {**base, "change_pct": {"20d": i, "60d": i}} for i in range(6)}
    checked = []
    tradeable = lambda t, d: checked.append(t) or t not in ("S5", "S3")
    scan = pick_candidates(snaps, held=[], max_candidates=2, tradeable=tradeable)
    assert scan.candidates == ["S2", "S4"]
    assert checked == ["S5", "S4", "S3", "S2"] and scan.no_liquid_options == ["S5", "S3"]
    capped = pick_candidates(snaps, held=[], max_candidates=2, tradeable=lambda t, d: False, max_checks=3)
    assert capped.candidates == [] and len(capped.no_liquid_options) == 3



# ---------- hybrid mode ----------

def test_hybrid_uses_the_call_when_one_is_on_the_menu():
    run, claude = run_options(cheap_bars(), mode="hybrid")
    assert claude.calls[-1] == "HybridDecision"
    [o] = run.gate.approved
    assert o.option_type == "call" and o.contracts >= 1


def test_hybrid_falls_back_to_all_in_shares_without_a_call():
    run, _ = run_options(cheap_bars(), mode="hybrid", fetch_empty=True, instrument="shares")
    assert run.options_menu["CHEAP"]["instrument"] == "shares"
    [o] = run.gate.approved
    assert o.action == "buy" and not hasattr(o, "contract_symbol")
    assert 98.0 <= o.notional_usd <= 100.0 and o.stop_price < o.limit_price


def test_hybrid_bearish_without_a_put_is_not_tradeable():
    run, claude = run_options(cheap_bars(), mode="hybrid", fetch_empty=True, direction="bearish")
    assert run.options_menu["CHEAP"]["instrument"] is None
    assert run.pm is None and "nothing tradeable" in run.stopped_reason


def test_hybrid_rejects_shares_when_a_call_exists():
    run, _ = run_options(cheap_bars(), mode="hybrid", instrument="shares")
    assert not run.gate.approved and "only allowed" in " ".join(run.gate.rejected[0].reasons)



def test_pending_trigger_rules():
    from desk.gatekeeper import pending_trigger
    assert pending_trigger("bullish", 15.00, 14.85) == 15.00       # breakout not cleared yet
    assert pending_trigger("bullish", 14.86, 14.85) is None        # within 0.2%: already there
    assert pending_trigger("bullish", 14.50, 14.85) is None        # price already above entry
    assert pending_trigger("bearish", 9.00, 9.30) == 9.00          # breakdown not broken yet
    assert pending_trigger("bearish", 9.40, 9.30) is None


def test_hybrid_shares_wait_for_the_breakout_trigger():
    # Entry 1% above the current price, like ABCL at 14.85 with a 15.00 breakout level.
    run, _ = run_options(cheap_bars(), mode="hybrid", fetch_empty=True, instrument="shares", entry_mult=1.01)
    [o] = run.gate.approved
    spot = run.data["technical"]["tickers"]["CHEAP"]["close"]
    assert o.trigger_price == round(spot * 1.01, 2)
    assert o.limit_price == round(o.trigger_price * 1.005, 2)      # stop-limit caps what it pays
    assert o.risk_usd <= 100 * 0.10 + 0.01                          # still within the 10% stop rule


def test_hybrid_shares_already_triggered_buy_at_market_price():
    run, _ = run_options(cheap_bars(), mode="hybrid", fetch_empty=True, instrument="shares")
    [o] = run.gate.approved
    assert o.trigger_price is None


def test_option_order_flags_an_untriggered_setup():
    run, _ = run_options(cheap_bars(), mode="hybrid", entry_mult=1.01)
    [o] = run.gate.approved
    assert o.option_type == "call" and o.trigger_price is not None



def test_trigger_up_to_5pct_away_is_allowed_beyond_that_rejected():
    # MGNI-like: breakout trigger ~4% above the live price.
    run, _ = run_options(cheap_bars(), mode="hybrid", fetch_empty=True, instrument="shares",
                         entry_mult=1.04, stop_mult=0.99, target_mult=1.25)
    [o] = run.gate.approved
    assert o.trigger_price is not None
    run, _ = run_options(cheap_bars(), mode="hybrid", fetch_empty=True, instrument="shares",
                         entry_mult=1.08, stop_mult=1.03, target_mult=1.30)
    assert not run.gate.approved and "5%" in " ".join(run.gate.rejected[0].reasons)


# ---------- rules from the backtest: stock stop, bullish only, market filter ----------

def live_rules(mode="hybrid"):
    from dataclasses import replace
    s = replace(load_settings(), instrument=mode)
    assert s.bullish_only and s.market_filter and s.options.stop_on_stock   # what config/desk.toml says
    return s


def market_bars(spy_drift):
    from .conftest import make_bars
    return {"SPY": make_bars(drift=spy_drift, seed=2), "CHEAP": make_bars(start=15.0, drift=0.0, seed=7)}


def test_market_trend_compares_spy_with_its_50_day_average():
    from desk.market_data import market_trend
    assert market_trend(market_bars(0.004))["up"] is True
    assert market_trend(market_bars(-0.004))["up"] is False
    assert market_trend({}) is None


def test_option_exit_is_the_stock_stop():
    run, _ = run_options(market_bars(0.004), mode="hybrid", settings=live_rules())
    [o] = run.gate.approved
    assert o.option_type == "call" and o.stop_price is None
    assert o.stock_stop < o.stock_entry            # the exit is the stock's own stop level


def test_market_below_its_average_blocks_new_trades():
    run, _ = run_options(market_bars(-0.004), mode="hybrid", settings=live_rules())
    assert run.market["up"] is False and not run.gate.approved
    assert "market filter" in " ".join(run.gate.rejected[0].reasons)


def test_market_filter_skips_the_ai_when_nothing_is_held():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from desk.llm import ClaudeRunner
    claude = FakeClaude({})
    run = run_team(live_rules(), ClaudeRunner(client=claude), AccountState(100, 100), market_bars(-0.004),
                   universe=["SPY", "CHEAP"], now=datetime(2026, 9, 30, 17, 0, tzinfo=ZoneInfo("America/New_York")),
                   fetch_catalysts=lambda *a, **k: {}, fetch_option_chains=lambda *a, **k: {})
    assert claude.calls == [] and "market filter" in run.stopped_reason and run.total_cost_usd == 0


def test_bullish_only_never_trades_a_put():
    s = live_rules()
    assert s.directions == ("bullish",)
    run, claude = run_options(market_bars(0.004), mode="hybrid", settings=s, direction="bearish")
    assert run.pm is None and "HybridDecision" not in claude.calls     # dropped before the PM
