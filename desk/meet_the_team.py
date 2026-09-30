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
from pathlib import Path

from . import market_data
from .config import DATA_DIR, ROOT, load_env_file, load_settings
from .llm import ClaudeRunner
from .scanner import pick_candidates
from .team import TeamRun, load_account, run_team
from .universe import load_universe

RUNS_DIR = DATA_DIR / "runs"
ACCOUNT_FILE = DATA_DIR / "account.toml"


def _print_run(run: TeamRun) -> None:
    line = "=" * 72
    print(f"\n{line}\nTEAM RUN {run.time_et} ET   candidates: {', '.join(run.candidates) or 'none'}\n{line}")
    if run.scan:
        sc = run.scan
        print(f"Scanner: {sc.scanned} tickers checked, {len(sc.with_setups)} with a setup"
              f"{', ' + str(len(sc.filtered_illiquid)) + ' too illiquid' if sc.filtered_illiquid else ''}.")
        if sc.cut_by_limit:
            print(f"Not sent (ranked below the top {len(run.candidates)}): {', '.join(sc.cut_by_limit)}")
    for role, a in run.analysts.items():
        head = f"\n--- {role.replace('_', ' ').title()}  [{a.model_served or a.model_requested}, prompt {a.prompt_version}, ${a.cost_usd:.4f}]"
        print(head)
        print(json.dumps(a.report.model_dump(), indent=2) if a.report else f"FAILED: {a.error}")
    if run.pm:
        a = run.pm
        print(f"\n--- Portfolio Manager  [{a.model_served or a.model_requested}, prompt {a.prompt_version}, ${a.cost_usd:.4f}]")
        print(json.dumps(a.report.model_dump(), indent=2) if a.report else f"FAILED: {a.error}")
    if run.gate:
        _print_gate(run.gate)
    if run.stopped_reason:
        print(f"\nSTOPPED: {run.stopped_reason}")
    print(f"\nTotal AI cost this run: ${run.total_cost_usd:.4f}")


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
        if o.action == "buy":
            print(f"  BUY  {o.ticker}: {o.shares} shares, LIMIT ${o.limit_price:.2f} (${o.notional_usd:.2f})\n"
                  f"       then set STOP ${o.stop_price:.2f}; target ${o.target_price:.2f}; "
                  f"reward-to-risk {o.reward_risk}; max loss ${o.risk_usd:.2f}")
        else:
            print(f"  SELL {o.ticker}: all {o.shares} shares (last price ${o.limit_price:.2f})")


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
    bars = market_data.download_bars(symbols)

    if args.data_only:
        ref = str(bars["SPY"].index[-1].date()) if "SPY" in bars else None
        tech = market_data.technical_packet(bars, universe, ref)
        scan = pick_candidates(tech["tickers"], account.held, settings.max_candidates,
                               settings.min_price, settings.min_dollar_volume)
        print(json.dumps(market_data.regime_packet(
            bars, list(settings.regime_symbols), list(settings.sector_etfs), universe), indent=1, default=str))
        print(f"\nScanned {scan.scanned} tickers. Setups today ({len(scan.with_setups)}): "
              f"{', '.join(scan.with_setups) or 'none'}")
        print(f"Would send to the analysts: {', '.join(scan.candidates) or 'none (quiet day)'}")
        return

    run = run_team(settings, ClaudeRunner(), account, bars, tickers=tickers, universe=universe)
    _print_run(run)
    if run.analysts:
        print(f"Saved: {_save(run).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
