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
    assert c is None and "before earnings" in note


def test_skips_wide_spreads_and_thin_open_interest():
    df = chain()["2026-11-06"].assign(openInterest=5)
    c, _ = select_contract({"2026-11-06": df}, "bullish", 20.0, cash=100, cfg=CFG, today=TODAY)
    assert c is None


# ---------- full team run in options mode ----------

def reports(direction="bullish", posture="cautious", conviction=4, event_risk="low", contract=None):
    tech = TechnicalReport(views=[TechnicalView(
        ticker="CHEAP", data_ok=True, trend="up" if direction == "bullish" else "down", direction=direction,
        setup="breakout" if direction == "bullish" else "breakdown", setup_quality=4,
        entry=20.0, stop=19.0 if direction == "bullish" else 21.0, target=23.0 if direction == "bullish" else 17.0,
        key_levels=[], evidence="e", risks="r", recommendation="candidate")], warnings=[])
    news = CatalystReport(views=[CatalystView(ticker="CHEAP", data_ok=True, next_earnings=None,
                                              days_to_earnings=None, event_risk=event_risk, sentiment="neutral",
                                              catalysts=[], red_flags=[], summary="")], warnings=[])
    regime = RegimeReport(regime="neutral", posture=posture, max_new_positions_today=1, evidence=[],
                          leading_sectors=[], lagging_sectors=[], summary="", warnings=[])
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


def run_options(bars, **kw):
    from desk.llm import ClaudeRunner
    settings = load_settings()
    assert settings.instrument == "options"
    spot = float(bars["CHEAP"]["Close"].iloc[-1])

    def fake_chains(ticker, cfg, today):
        c = chain(spot=spot)
        return {"calls": c, "puts": c}

    # Find the contract code will offer, then have the PM pick it.
    offered, note = select_contract(chain(spot=spot), kw.get("direction", "bullish"), spot, 100,
                                    settings.options, TODAY)
    assert offered, note
    claude = FakeClaude(reports(contract=kw.pop("contract", offered["contract_symbol"]), **kw))
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
