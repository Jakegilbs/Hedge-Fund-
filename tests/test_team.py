"""Team pipeline tests with a fake Claude client: no network, no API cost."""
from types import SimpleNamespace

import pytest

from desk.config import load_settings
from desk.llm import ClaudeRunner, cost_usd
from desk.schemas import (CatalystReport, CatalystView, PMDecision, RegimeReport, TechnicalReport,
                          TechnicalView)
from desk.team import AccountState, run_team

REPORTS = {
    TechnicalReport: TechnicalReport(views=[TechnicalView(
        ticker="NVDA", data_ok=True, trend="up", setup="breakout", setup_quality=4, entry=110.0,
        stop=105.0, target=122.0, key_levels=["105 base"], evidence="New high on 2x volume.",
        risks="Close back inside base.", recommendation="candidate")], warnings=[]),
    CatalystReport: CatalystReport(views=[CatalystView(
        ticker="NVDA", data_ok=True, next_earnings=None, days_to_earnings=None, event_risk="low",
        sentiment="neutral", catalysts=[], red_flags=["earnings date unknown"], summary="Quiet.")],
        warnings=[]),
    RegimeReport: RegimeReport(regime="neutral", posture="cautious", max_new_positions_today=1,
                               evidence=["SPY above ema50"], leading_sectors=["XLK"], lagging_sectors=["XLE"],
                               summary="Mixed.", warnings=[]),
    PMDecision: PMDecision(market_view="Cautious.", orders=[], position_updates=[], warnings=[],
                           honest_assessment="First run."),
}


SCHEMA_BY_TITLE = {m.__name__: m for m in REPORTS}


class FakeMessages:
    def __init__(self, fail_schema=None, stop_reason="end_turn"):
        self.calls = []
        self.fail_schema = fail_schema
        self.stop_reason = stop_reason

    def create(self, **kw):
        schema = SCHEMA_BY_TITLE[kw["output_config"]["format"]["schema"]["title"]]
        self.calls.append({**kw, "schema": schema})
        text = '{"not": "valid"}' if schema is self.fail_schema else REPORTS[schema].model_dump_json()
        return SimpleNamespace(model=kw["model"] + "-20251001", stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
                               content=[SimpleNamespace(type="text", text=text)])


def fake_runner(**kw):
    messages = FakeMessages(**kw)
    return ClaudeRunner(client=SimpleNamespace(beta=SimpleNamespace(messages=messages))), messages


def no_news(tickers):
    return {"today": "2026-09-29", "tickers": {t: {"next_earnings": None, "headlines": [], "errors": []}
                                                for t in tickers}}


def test_full_run_calls_three_analysts_then_pm(bars):
    runner, msgs = fake_runner()
    run = run_team(load_settings(), runner, AccountState(100, 100), bars, fetch_catalysts=no_news,
                   time_et="2026-09-29 10:00")
    assert run.candidates == ["NVDA"]
    assert run.stopped_reason is None and run.pm.ok
    roles = [c["schema"].__name__ for c in msgs.calls]
    assert sorted(roles[:3]) == ["CatalystReport", "RegimeReport", "TechnicalReport"] and roles[3] == "PMDecision"
    pm_call = msgs.calls[3]
    assert pm_call["model"] == "claude-opus-5-5" and pm_call["fallbacks"] == "default"
    assert pm_call["output_config"]["effort"] == "high"
    assert "$100.00" in pm_call["messages"][0]["content"]
    assert run.total_cost_usd > 0
    assert run.gate is not None and run.gate.approved == []


def test_haiku_analysts_get_no_effort_or_fallback(bars):
    runner, msgs = fake_runner()
    run_team(load_settings(), runner, AccountState(100, 100), bars, fetch_catalysts=no_news)
    haiku = [c for c in msgs.calls if c["model"] == "claude-haiku-4-5"]
    assert len(haiku) == 3
    assert all("fallbacks" not in c and "effort" not in c["output_config"] for c in haiku)


def test_quiet_day_costs_nothing(bars):
    runner, msgs = fake_runner()
    bars = {k: v for k, v in bars.items() if k != "NVDA"}
    run = run_team(load_settings(), runner, AccountState(100, 100), bars, fetch_catalysts=no_news)
    assert run.stopped_reason.startswith("quiet day") and msgs.calls == []


def test_failed_analyst_stops_before_pm_but_cost_is_counted(bars):
    runner, msgs = fake_runner(fail_schema=RegimeReport)
    run = run_team(load_settings(), runner, AccountState(100, 100), bars, fetch_catalysts=no_news)
    assert run.pm is None and "regime_analyst" in run.stopped_reason
    assert run.analysts["regime_analyst"].cost_usd > 0
    assert all(c["schema"] is not PMDecision for c in msgs.calls)


def test_cut_off_reply_is_rejected(bars):
    runner, _ = fake_runner(stop_reason="max_tokens")
    run = run_team(load_settings(), runner, AccountState(100, 100), bars, fetch_catalysts=no_news)
    assert run.pm is None and "max_tokens" in run.stopped_reason


def test_tickers_outside_allowlist_rejected(bars):
    runner, _ = fake_runner()
    with pytest.raises(ValueError):
        run_team(load_settings(), runner, AccountState(100, 100), bars, fetch_catalysts=no_news, tickers=["GME"])


def test_cost_math():
    assert cost_usd("claude-haiku-4-5", 1_000_000, 0) == 1.0
    assert cost_usd("claude-opus-5-5", 0, 1_000_000) == 20.0
    assert cost_usd("claude-haiku-4-5-20251001", 1_000_000, 0) == 1.0


def test_small_ranges_are_enforced_by_the_schema_itself():
    import anthropic
    from desk.schemas import TechnicalReport
    schema = anthropic.transform_schema(TechnicalReport.model_json_schema())
    view = schema["$defs"]["TechnicalView"]["properties"]["setup_quality"]
    assert view["enum"] == [0, 1, 2, 3, 4, 5]


def test_account_file_or_paper(tmp_path):
    from desk.team import load_account
    assert load_account(tmp_path / "missing.toml", 100).equity == 100
    f = tmp_path / "account.toml"
    f.write_text('equity = 101.5\ncash = 86.5\n[[positions]]\nticker = "nvda"\nshares = 0.15\n')
    acct = load_account(f, 100)
    assert acct.cash == 86.5 and acct.shares_by_ticker == {"NVDA": 0.15}
