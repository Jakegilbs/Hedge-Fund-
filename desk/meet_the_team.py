"""Run the analyst team once on today's real market data. No orders are placed.

    python -m desk.meet_the_team                  # scanner picks the tickers
    python -m desk.meet_the_team --tickers NVDA,JPM   # analyse these anyway
    python -m desk.meet_the_team --data-only      # show the data packets, no AI cost
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from . import market_data
from .config import DATA_DIR, ROOT, load_env_file, load_settings
from .llm import ClaudeRunner
from .team import AccountState, TeamRun, run_team

RUNS_DIR = DATA_DIR / "runs"


def _print_run(run: TeamRun) -> None:
    line = "=" * 72
    print(f"\n{line}\nTEAM RUN {run.time_et} ET   candidates: {', '.join(run.candidates) or 'none'}\n{line}")
    for role, a in run.analysts.items():
        head = f"\n--- {role.replace('_', ' ').title()}  [{a.model_served or a.model_requested}, prompt {a.prompt_version}, ${a.cost_usd:.4f}]"
        print(head)
        print(json.dumps(a.report.model_dump(), indent=2) if a.report else f"FAILED: {a.error}")
    if run.pm:
        a = run.pm
        print(f"\n--- Portfolio Manager  [{a.model_served or a.model_requested}, prompt {a.prompt_version}, ${a.cost_usd:.4f}]")
        print(json.dumps(a.report.model_dump(), indent=2) if a.report else f"FAILED: {a.error}")
        if a.report and a.report.orders:
            print("\nNOTE: share counts and reward-to-risk above are the PM's own math."
                  " The Gatekeeper (next build step) will recompute and enforce them.")
    if run.stopped_reason:
        print(f"\nSTOPPED: {run.stopped_reason}")
    print(f"\nTotal AI cost this run: ${run.total_cost_usd:.4f}")


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
    equity = args.equity or settings.paper_equity
    account = AccountState(equity=equity, cash=equity)
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None

    symbols = sorted(set(settings.allowlist) | set(settings.regime_symbols) | set(settings.sector_etfs))
    print(f"Downloading daily bars for {len(symbols)} symbols...")
    bars = market_data.download_bars(symbols)

    if args.data_only:
        ref = str(bars["SPY"].index[-1].date()) if "SPY" in bars else None
        tech = market_data.technical_packet(bars, list(settings.allowlist), ref)
        with_setups = [t for t, s in tech["tickers"].items() if s["setups"]]
        print(json.dumps({"technical": tech, "regime": market_data.regime_packet(
            bars, list(settings.regime_symbols), list(settings.sector_etfs), list(settings.allowlist))},
            indent=1, default=str))
        print(f"\nTickers with a setup today: {with_setups or 'none'}")
        return

    run = run_team(settings, ClaudeRunner(), account, bars, tickers=tickers)
    _print_run(run)
    if run.analysts:
        print(f"Saved: {_save(run).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
