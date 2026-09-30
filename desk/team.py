"""The analyst team: three analysts report, then the Portfolio Manager decides.

Fail-safe by design: if any analyst fails, the PM is not called and the run
proposes nothing. A missed trade costs little; a trade on a missing report can
cost a lot.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from . import market_data
from .config import Settings
from .llm import AgentResult, ClaudeRunner
from .prompts import load_prompt
from .scanner import pick_candidates
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


@dataclass
class TeamRun:
    time_et: str
    candidates: list[str]
    analysts: dict[str, AgentResult] = field(default_factory=dict)
    pm: AgentResult | None = None
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
             tickers: list[str] | None = None, time_et: str | None = None) -> TeamRun:
    """Run the team once. `tickers` overrides the scanner (for testing prompts)."""
    time_et = time_et or market_data.now_et()
    reference = str(bars["SPY"].index[-1].date()) if "SPY" in bars else None
    allow = list(settings.allowlist)
    tech_all = market_data.technical_packet(bars, allow, reference)

    if tickers:
        outside = sorted(set(t.upper() for t in tickers) - set(allow))
        if outside:
            raise ValueError(f"not on the allowlist: {outside}")
        candidates = sorted(set(t.upper() for t in tickers) | set(account.held))
    else:
        candidates = pick_candidates(tech_all["tickers"], account.held)

    run = TeamRun(time_et=time_et, candidates=candidates)
    if reference is None:
        run.stopped_reason = "no SPY data: cannot judge the market or data freshness"
        return run
    if not candidates:
        run.stopped_reason = "quiet day: no setups and no open positions (no AI cost)"
        return run

    tech = {**tech_all, "tickers": {t: tech_all["tickers"][t] for t in candidates if t in tech_all["tickers"]},
            "missing_data": [t for t in candidates if t not in tech_all["tickers"]]}
    regime = market_data.regime_packet(bars, list(settings.regime_symbols), list(settings.sector_etfs), allow)
    news = fetch_catalysts(candidates)
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
        allowlist=", ".join(allow),
        daily_halt_pct=f"{risk.daily_halt_pct * 100:g}",
        min_reward_risk=f"{risk.min_reward_risk:g}",
        regime_report=run.analysts["regime_analyst"].report.model_dump(),
        technical_report=run.analysts["technical_analyst"].report.model_dump(),
        catalyst_report=run.analysts["catalyst_analyst"].report.model_dump(),
    )
    run.pm = runner.run(pm_prompt, rendered, PMDecision, settings.models["portfolio_manager"],
                        effort=settings.effort.get("portfolio_manager"), max_tokens=PM_MAX_TOKENS)
    if not run.pm.ok:
        run.stopped_reason = f"PM decision unusable, no trades: {run.pm.error}"
    return run
