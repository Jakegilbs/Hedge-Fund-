"""The analyst team: three analysts report, then the Portfolio Manager decides.

Fail-safe by design: if any analyst fails, the PM is not called and the run
proposes nothing. A missed trade costs little; a trade on a missing report can
cost a lot.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Callable

import pandas as pd

from . import gatekeeper, market_data
from .config import Settings
from .llm import AgentResult, ClaudeRunner
from .prompts import load_prompt
from .scanner import ScanResult, pick_candidates
from .schemas import CatalystReport, PMDecision, RegimeReport, TechnicalReport

ANALYST_MAX_TOKENS = 8000
PM_MAX_TOKENS = 16000


@dataclass
class AccountState:
    equity: float
    cash: float
    pnl_today: float = 0.0
    positions: list[dict] = field(default_factory=list)      # ticker, shares, entry, stop, target, thesis
    trade_history: list[dict] = field(default_factory=list)  # last 20 closed trades

    @property
    def held(self) -> list[str]:
        return [p["ticker"] for p in self.positions]

    @property
    def shares_by_ticker(self) -> dict[str, float]:
        return {p["ticker"].upper(): float(p["shares"]) for p in self.positions}


def load_account(path, paper_equity: float) -> AccountState:
    """Your real account from data/account.toml, or a paper account if the file is absent."""
    import tomllib
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        return AccountState(equity=paper_equity, cash=paper_equity)
    raw = tomllib.loads(path.read_text())
    positions = [dict(p, ticker=str(p["ticker"]).upper()) for p in raw.get("positions", [])]
    return AccountState(equity=float(raw["equity"]), cash=float(raw["cash"]),
                        pnl_today=float(raw.get("pnl_today", 0.0)), positions=positions)


@dataclass
class TeamRun:
    time_et: str
    candidates: list[str]
    analysts: dict[str, AgentResult] = field(default_factory=dict)
    pm: AgentResult | None = None
    gate: gatekeeper.GateResult | None = None
    scan: ScanResult | None = None
    stopped_reason: str | None = None
    data: dict = field(default_factory=dict)

    @property
    def agents(self) -> list[AgentResult]:
        return list(self.analysts.values()) + ([self.pm] if self.pm else [])

    @property
    def total_cost_usd(self) -> float:
        return round(sum(a.cost_usd for a in self.agents), 4)


def run_team(settings: Settings, runner: ClaudeRunner, account: AccountState,
             bars: dict[str, pd.DataFrame],
             fetch_catalysts: Callable[[list[str]], dict] = market_data.catalyst_packet,
             tickers: list[str] | None = None, time_et: str | None = None,
             now: datetime | None = None, universe: list[str] | None = None) -> TeamRun:
    """Run the team once. `tickers` overrides the scanner (for testing prompts)."""
    now = now or datetime.now(ZoneInfo("America/New_York"))
    time_et = time_et or now.strftime("%Y-%m-%d %H:%M")
    bars, live = market_data.split_incomplete_bar(bars, now)
    reference = str(bars["SPY"].index[-1].date()) if "SPY" in bars else None
    allow = list(universe or settings.allowlist)
    tech_all = market_data.technical_packet(bars, allow, reference, live)

    if tickers:
        outside = sorted(set(t.upper() for t in tickers) - set(allow))
        if outside:
            raise ValueError(f"not on the allowlist: {outside}")
        candidates = sorted(set(t.upper() for t in tickers) | set(account.held))
        scan = None
    else:
        scan = pick_candidates(tech_all["tickers"], account.held, settings.max_candidates,
                               settings.min_price, settings.min_dollar_volume)
        candidates = scan.candidates

    run = TeamRun(time_et=time_et, candidates=candidates, scan=scan)
    if reference is None:
        run.stopped_reason = "no SPY data: cannot judge the market or data freshness"
        return run
    if not candidates:
        run.stopped_reason = "quiet day: no setups and no open positions (no AI cost)"
        return run

    why = scan.why if scan else {t: "requested by you" for t in candidates}
    tech = {**tech_all, "tickers": {t: {"why_selected": why.get(t, ""), **tech_all["tickers"][t]}
                                    for t in candidates if t in tech_all["tickers"]},
            "missing_data": [t for t in candidates if t not in tech_all["tickers"]]}
    regime = market_data.regime_packet(bars, list(settings.regime_symbols), list(settings.sector_etfs), allow)
    news = fetch_catalysts(candidates, frozenset(settings.etfs))
    run.data = {"technical": tech, "regime": regime, "news": news}

    jobs = {
        "technical_analyst": (TechnicalReport, {"technical_data": tech}),
        "catalyst_analyst": (CatalystReport, {"news_data": news, "today": news.get("today", time_et[:10])}),
        "regime_analyst": (RegimeReport, {"regime_data": regime}),
    }

    def call(role: str) -> AgentResult:
        schema, values = jobs[role]
        prompt = load_prompt(role, settings.prompt_versions[role])
        return runner.run(prompt, prompt.render(**values), schema, settings.models[role],
                          effort=settings.effort.get(role), max_tokens=ANALYST_MAX_TOKENS)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for role, result in zip(jobs, pool.map(call, jobs)):
            run.analysts[role] = result

    failed = [f"{r}: {a.error}" for r, a in run.analysts.items() if not a.ok]
    if failed:
        run.stopped_reason = "analyst report missing, PM not called: " + "; ".join(failed)
        return run

    risk = settings.risk
    pm_prompt = load_prompt("portfolio_manager", settings.prompt_versions["portfolio_manager"])
    rendered = pm_prompt.render(
        time_et=time_et,
        equity=f"${account.equity:,.2f}",
        cash=f"${account.cash:,.2f}",
        pnl_today=f"${account.pnl_today:,.2f}",
        positions=account.positions or "none",
        trade_history=account.trade_history or "none yet",
        risk_per_trade_pct=f"{risk.risk_per_trade * 100:g}",
        risk_per_trade=f"{risk.risk_per_trade:g}",
        max_position_pct=f"{risk.max_position_pct * 100:g}",
        max_open_positions=str(risk.max_open_positions),
        allowlist=(f"{len(allow)} pre-approved tickers (S&P 500 and major ETFs); every ticker in the "
                   "analyst reports is on it. Any other ticker is rejected by code."),
        daily_halt_pct=f"{risk.daily_halt_pct * 100:g}",
        min_reward_risk=f"{risk.min_reward_risk:g}",
        max_stop_distance_pct=f"{risk.max_stop_distance_pct * 100:g}",
        min_conviction=str(risk.min_conviction),
        regime_report=run.analysts["regime_analyst"].report.model_dump(),
        technical_report=run.analysts["technical_analyst"].report.model_dump(),
        catalyst_report=run.analysts["catalyst_analyst"].report.model_dump(),
    )
    run.pm = runner.run(pm_prompt, rendered, PMDecision, settings.models["portfolio_manager"],
                        effort=settings.effort.get("portfolio_manager"), max_tokens=PM_MAX_TOKENS)
    if not run.pm.ok:
        run.stopped_reason = f"PM decision unusable, no trades: {run.pm.error}"
        return run

    snaps = tech_all["tickers"]
    run.gate = gatekeeper.check(
        run.pm.report, equity=account.equity, cash=account.cash, pnl_today=account.pnl_today,
        positions=account.shares_by_ticker,
        last_prices={t: s["live_price"] or s["close"] for t, s in snaps.items() if s["close"] is not None},
        stale={t for t, s in snaps.items() if s.get("stale")},
        allowlist=set(allow), risk=risk,
        regime=run.analysts["regime_analyst"].report,
        catalysts=run.analysts["catalyst_analyst"].report)
    return run
