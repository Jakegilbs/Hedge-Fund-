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

from . import gatekeeper, market_data, options
from .config import Settings
from .llm import AgentResult, ClaudeRunner
from .prompts import load_prompt
from .scanner import ScanResult, pick_candidates
from .schemas import CatalystReport, OptionsDecision, PMDecision, RegimeReport, TechnicalReport

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
    options_menu: dict | None = None

    @property
    def agents(self) -> list[AgentResult]:
        return list(self.analysts.values()) + ([self.pm] if self.pm else [])

    @property
    def total_cost_usd(self) -> float:
        return round(sum(a.cost_usd for a in self.agents), 4)


def _check_reports(run: "TeamRun", candidates: list[str], min_rr: float) -> None:
    """Code checks on the analysts' reports before the PM sees them.

    - Views for tickers that were not sent are dropped (analysts can invent tickers).
    - Candidates an analyst skipped are flagged.
    - Reward-to-risk is recomputed from the Technical Analyst's own levels; a
      "candidate" below the minimum becomes "watch".
    """
    sent = set(candidates)
    for role in ("technical_analyst", "catalyst_analyst"):
        report = run.analysts[role].report
        extra = sorted({v.ticker.upper() for v in report.views} - sent)
        report.views = [v for v in report.views if v.ticker.upper() in sent]
        missing = sorted(sent - {v.ticker.upper() for v in report.views})
        if extra:
            report.warnings.append(f"code: removed views for tickers that were not sent: {', '.join(extra)}")
        if missing:
            report.warnings.append(f"code: no view returned for: {', '.join(missing)}")
    for v in run.analysts["technical_analyst"].report.views:
        bearish = getattr(v, "direction", "bullish") == "bearish"
        risk = (v.stop - v.entry) if (bearish and v.entry and v.stop) else \
               (v.entry - v.stop) if (v.entry and v.stop) else 0
        reward = (v.entry - v.target) if (bearish and v.entry and v.target) else \
                 (v.target - v.entry) if (v.entry and v.target) else 0
        v.reward_risk_checked = round(reward / risk, 2) if risk > 0 and reward > 0 else None
        if v.recommendation == "candidate" and (v.reward_risk_checked or 0) < min_rr:
            v.recommendation = "watch"
            v.risks += (f" [code: reward-to-risk {v.reward_risk_checked} is below {min_rr:g}, "
                        "downgraded from candidate to watch]")


def run_team(settings: Settings, runner: ClaudeRunner, account: AccountState,
             bars: dict[str, pd.DataFrame],
             fetch_catalysts: Callable[[list[str]], dict] = market_data.catalyst_packet,
             tickers: list[str] | None = None, time_et: str | None = None,
             now: datetime | None = None, universe: list[str] | None = None,
             fetch_option_chains=options.fetch_chains) -> TeamRun:
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
        directions = ("bullish", "bearish") if settings.instrument == "options" else ("bullish",)
        scan = pick_candidates(tech_all["tickers"], account.held, settings.max_candidates,
                               settings.min_price, settings.min_dollar_volume, directions=directions)
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
        "technical_analyst": (TechnicalReport, {"technical_data": tech,
                                                "min_reward_risk": f"{settings.risk.min_reward_risk:g}"}),
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

    _check_reports(run, candidates, settings.risk.min_reward_risk)
    tradeable = [v.ticker for v in run.analysts["technical_analyst"].report.views
                 if v.recommendation == "candidate" and v.data_ok
                 and (v.direction in ("bullish", "bearish") if settings.instrument == "options"
                      else v.direction != "bearish")]
    if not tradeable and not account.held:
        run.stopped_reason = ("no trade possible: the Technical Analyst rated nothing a 'candidate' and "
                              "nothing is held, so the Portfolio Manager was skipped (no PM cost)")
        return run

    if settings.instrument == "options":
        return _options_pm(run, settings, runner, account, allow, tech_all, tradeable, now,
                           fetch_option_chains)

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


def _options_pm(run: TeamRun, settings: Settings, runner: ClaudeRunner, account: AccountState,
                allow: list[str], tech_all: dict, tradeable: list[str], now: datetime,
                fetch_option_chains) -> TeamRun:
    """Options mode: code builds the contract menu, the PM picks at most one long call or put."""
    tech_rep = run.analysts["technical_analyst"].report
    news_rep = run.analysts["catalyst_analyst"].report
    earnings = {v.ticker.upper(): v.next_earnings for v in news_rep.views}
    views = {v.ticker.upper(): v for v in tech_rep.views}
    snaps = tech_all["tickers"]
    picks = [{"ticker": t, "direction": views[t].direction,
              "spot": snaps[t]["live_price"] or snaps[t]["close"], "earnings": earnings.get(t)}
             for t in (x.upper() for x in tradeable) if t in snaps]
    run.options_menu = options.options_menu(picks, account.cash, settings.options, now.date(),
                                            fetch=fetch_option_chains)
    offered = {t: m for t, m in run.options_menu.items() if m["contract"]}
    if not offered and not account.held:
        notes = "; ".join(f"{t}: {m['note']}" for t, m in run.options_menu.items())
        run.stopped_reason = f"no suitable option contract, PM skipped (no PM cost). {notes}"
        return run

    risk, opts = settings.risk, settings.options
    prompt = load_prompt("portfolio_manager", settings.prompt_versions["portfolio_manager_options"])
    rendered = prompt.render(
        time_et=run.time_et, equity=f"${account.equity:,.2f}", cash=f"${account.cash:,.2f}",
        pnl_today=f"${account.pnl_today:,.2f}", positions=account.positions or "none",
        trade_history=account.trade_history or "none yet",
        max_open_positions=str(risk.max_open_positions),
        stop_loss_pct=f"{opts.stop_loss_pct * 100:g}", take_profit_pct=f"{opts.take_profit_pct * 100:g}",
        exit_days=str(opts.exit_days_before_expiry), min_conviction=str(risk.min_conviction),
        min_reward_risk=f"{risk.min_reward_risk:g}", daily_halt_pct=f"{risk.daily_halt_pct * 100:g}",
        allowlist=(f"{len(allow)} pre-approved tickers (S&P 500 and major ETFs); every ticker in the "
                   "analyst reports is on it."),
        regime_report=run.analysts["regime_analyst"].report.model_dump(),
        technical_report=tech_rep.model_dump(), catalyst_report=news_rep.model_dump(),
        options_menu=run.options_menu,
    )
    run.pm = runner.run(prompt, rendered, OptionsDecision, settings.models["portfolio_manager"],
                        effort=settings.effort.get("portfolio_manager"), max_tokens=PM_MAX_TOKENS)
    if not run.pm.ok:
        run.stopped_reason = f"PM decision unusable, no trades: {run.pm.error}"
        return run
    run.gate = gatekeeper.check_options(
        run.pm.report, cash=account.cash, pnl_today=account.pnl_today, equity=account.equity,
        open_positions=len(account.positions), menu=run.options_menu, technical=tech_rep,
        catalysts=news_rep, regime=run.analysts["regime_analyst"].report, risk=risk, opts=opts,
        allowlist=set(allow))
    return run
