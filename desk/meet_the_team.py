"""Run the analyst team once on today's real market data. No orders are placed.

    python -m desk.meet_the_team                  # scanner picks the tickers
    python -m desk.meet_the_team --tickers NVDA,JPM   # analyse these anyway
    python -m desk.meet_the_team --data-only      # show the data packets, no AI cost
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from . import market_data
from .config import DATA_DIR, ROOT, load_env_file, load_settings
from .llm import ClaudeRunner
from .scanner import pick_candidates
from .team import UPSIDE_WEIGHT, TeamRun, load_account, options_affordable, run_team
from .universe import load_universe

RUNS_DIR = DATA_DIR / "runs"
ACCOUNT_FILE = DATA_DIR / "account.toml"


def _print_run(run: TeamRun) -> None:
    line = "=" * 72
    print(f"\n{line}\nTEAM RUN {run.time_et} ET   candidates: {', '.join(run.candidates) or 'none'}\n{line}")
    if run.scan:
        _print_scan(run.scan)
    for role, a in run.analysts.items():
        head = f"\n--- {role.replace('_', ' ').title()}  [{a.model_served or a.model_requested}, prompt {a.prompt_version}, ${a.cost_usd:.4f}]"
        print(head)
        print(json.dumps(a.report.model_dump(), indent=2) if a.report else f"FAILED: {a.error}")
    if run.pm:
        a = run.pm
        print(f"\n--- Portfolio Manager  [{a.model_served or a.model_requested}, prompt {a.prompt_version}, ${a.cost_usd:.4f}]")
        print(json.dumps(a.report.model_dump(), indent=2) if a.report else f"FAILED: {a.error}")
    if run.options_menu is not None:
        print("\n--- Options menu (code picked one contract per candidate)")
        for t, m in run.options_menu.items():
            c = m["contract"]
            if c:
                print(f"  {t} {m['direction']}: {c['type'].upper()} ${c['strike']:g} exp {c['expiry']} "
                      f"({c['days_to_expiry']}d), delta {c['delta']}, ${c['cost_per_contract']:.0f}/contract")
            else:
                print(f"  {t} {m['direction']}: none ({m['note']})")
    if run.gate:
        _print_gate(run.gate)
    if run.stopped_reason:
        print(f"\nSTOPPED: {run.stopped_reason}")
    print(f"\nTotal AI cost this run: ${run.total_cost_usd:.4f}")


def _print_scan(sc, universe_size: int | None = None, downloaded: int | None = None) -> None:
    print("\n--- Scanner (code, free)")
    if universe_size is not None:
        print(f"Universe {universe_size} tickers, price data for {downloaded}"
              f" ({universe_size - downloaded} missing)")
    counts = ", ".join(f"{k} {v}" for k, v in sorted(sc.setup_counts.items())) or "none"
    print(f"Checked {sc.scanned} | stale {len(sc.stale)} | with a setup {len(sc.with_setups)} ({counts})"
          f" | too illiquid {len(sc.filtered_illiquid)}")
    if sc.filtered_unaffordable:
        print(f"Skipped, options too expensive for the account: {len(sc.filtered_unaffordable)} tickers")
    if sc.filtered_extended or sc.filtered_pinned:
        print(f"Skipped as overextended: {', '.join(sorted(sc.filtered_extended)) or 'none'}"
              f" | pinned (too quiet): {', '.join(sorted(sc.filtered_pinned)) or 'none'}")
    for t in sc.candidates:
        print(f"  -> {t}: {sc.why.get(t, '')}")
    if sc.cut_by_limit:
        print(f"Setups not sent (ranked below the top slots): {', '.join(sc.cut_by_limit)}")


def _print_gate(gate) -> None:
    print("\n--- Gatekeeper (code: recomputed sizes, enforced limits)")
    if gate.halted:
        print(f"HALT: {gate.halted}")
    for r in gate.rejected:
        print(f"REJECTED {r.action.upper()} {r.ticker}: " + "; ".join(r.reasons))
    if not gate.approved:
        print("No approved orders. Nothing to do today.")
        return
    print("\nAPPROVED ORDERS (place these yourself in the Robinhood Agentic account):")
    for o in gate.approved:
        if hasattr(o, "contract_symbol"):
            kind = o.option_type.upper()
            print(f"  BUY {o.contracts} x {o.ticker} {o.expiry} ${o.strike:g} {kind}"
                  f"  LIMIT ${o.limit_price:.2f} per share (${o.cost_usd:.2f} total)\n"
                  f"       contract {o.contract_symbol}\n"
                  f"       EXIT PLAN: sell if the option falls to ${o.stop_price:.2f} "
                  f"(-{round((1 - o.stop_price / o.limit_price) * 100)}% stop), "
                  f"take profit at ${o.take_profit_price:.2f}, sell by {o.exit_by} at the latest\n"
                  f"       stock view: entry {o.stock_entry}, wrong below/above {o.stock_stop}, "
                  f"target {o.stock_target}")
        elif o.action == "buy":
            print(f"  BUY  {o.ticker}: {o.shares} shares, LIMIT ${o.limit_price:.2f} (${o.notional_usd:.2f})\n"
                  f"       then set STOP ${o.stop_price:.2f}; target ${o.target_price:.2f}; "
                  f"reward-to-risk {o.reward_risk}; max loss ${o.risk_usd:.2f}")
        elif o.shares:
            print(f"  SELL {o.ticker}: all {o.shares} shares (last price ${o.limit_price:.2f})")
        else:
            print(f"  SELL {o.ticker}: close the whole position ({o.reason})")


def _save(run: TeamRun) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    record = {
        "time_et": run.time_et,
        "candidates": run.candidates,
        "stopped_reason": run.stopped_reason,
        "total_cost_usd": run.total_cost_usd,
        "agents": [{
            "role": a.role, "prompt_version": a.prompt_version, "prompt_fingerprint": a.prompt_fingerprint,
            "model_requested": a.model_requested, "model_served": a.model_served,
            "input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "cost_usd": a.cost_usd,
            "error": a.error, "report": a.report.model_dump() if a.report else None,
        } for a in run.agents],
        "gate": asdict(run.gate) if run.gate else None,
        "data": run.data,
    }
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="comma-separated allowlist tickers to analyse regardless of setups")
    ap.add_argument("--equity", type=float, help="paper account size (default from config)")
    ap.add_argument("--data-only", action="store_true", help="print the data packets and stop (no AI cost)")
    args = ap.parse_args()

    load_env_file()
    if not args.data_only and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("No API key found. Put ANTHROPIC_API_KEY=sk-ant-... in the .env file "
                         "in this folder (see SETUP_GUIDE.md, step 6).")
    settings = load_settings()
    account = load_account(ACCOUNT_FILE, args.equity or settings.paper_equity)
    source = "data/account.toml" if ACCOUNT_FILE.is_file() else "paper account"
    print(f"Account ({source}): equity ${account.equity:.2f}, cash ${account.cash:.2f}, "
          f"positions: {', '.join(account.held) or 'none'}")
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None

    universe = load_universe(settings)
    symbols = sorted(set(universe) | set(settings.regime_symbols) | set(settings.sector_etfs))
    print(f"Universe: {len(universe)} tradeable tickers. Downloading daily bars for {len(symbols)} symbols "
          f"(about a minute)...")
    bars = market_data.download_bars(symbols, priority=[*settings.regime_symbols, *settings.sector_etfs,
                                                        *settings.allowlist])

    if args.data_only:
        done, live = market_data.split_incomplete_bar(bars, datetime.now(ZoneInfo("America/New_York")))
        ref = str(done["SPY"].index[-1].date()) if "SPY" in done else None
        tech = market_data.technical_packet(done, universe, ref, live)
        scan = pick_candidates(tech["tickers"], account.held, settings.max_candidates,
                               settings.scan_min_price, settings.scan_min_dollar_volume,
                               directions=("bullish", "bearish") if settings.instrument == "options"
                               else ("bullish",),
                               affordable=options_affordable(settings, account.cash),
                               upside_weight=UPSIDE_WEIGHT if settings.instrument == "options" else 0.0)
        print(json.dumps(market_data.regime_packet(
            bars, list(settings.regime_symbols), list(settings.sector_etfs), universe), indent=1, default=str))
        _print_scan(scan, len(universe), sum(1 for t in universe if t in bars))
        if len(bars) < len(symbols) * 0.9:
            print("\nWARNING: more than 10% of symbols have no data. Yahoo is throttling this"
                  " computer: wait 15-30 minutes and run again. Downloaded prices are cached, so"
                  " each retry only fetches what is still missing.")
        return

    print(f"Price data for {sum(1 for t in universe if t in bars)} of {len(universe)} tickers.")
    run = run_team(settings, ClaudeRunner(), account, bars, tickers=tickers, universe=universe)
    _print_run(run)
    if run.analysts:
        print(f"Saved: {_save(run).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
