"""Candidate scanner: decides which tickers are worth the analysts' (paid) attention.

Every ticker in the universe is checked in code for free. Tickers with a fresh
setup that pass the liquidity filter are ranked, strongest first (60-day
relative strength, then number of setups), and only the top few go on to the
analysts. If fewer than that have a coded setup, the remaining slots go to
"leaders": the strongest liquid stocks in an uptrend near their highs, where
the Technical Analyst decides whether a trade exists. Open positions are always
reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScanResult:
    candidates: list[str]
    scanned: int = 0
    with_setups: list[str] = field(default_factory=list)
    filtered_illiquid: list[str] = field(default_factory=list)
    cut_by_limit: list[str] = field(default_factory=list)
    leaders: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    setup_counts: dict[str, int] = field(default_factory=dict)
    why: dict[str, str] = field(default_factory=dict)


def _strength(snap: dict) -> tuple:
    return (snap["change_pct"].get("60d") or -999.0, len(snap["setups"]))


def pick_candidates(tech_snapshots: dict[str, dict], held: list[str], max_candidates: int = 8,
                    min_price: float = 0.0, min_dollar_volume: float = 0.0) -> ScanResult:
    """A stale ticker is never a new candidate; a stale held position is still
    reviewed so the PM can see and flag the data problem."""
    result = ScanResult(candidates=[], scanned=len(tech_snapshots))

    def liquid(s: dict) -> bool:
        return s["close"] >= min_price and (s.get("avg_dollar_volume_20d") or 0) >= min_dollar_volume

    with_setup, leaders = [], []
    for t, s in tech_snapshots.items():
        if s.get("stale"):
            result.stale.append(t)
            continue
        for st in s["setups"]:
            result.setup_counts[st["setup"]] = result.setup_counts.get(st["setup"], 0) + 1
        if s["setups"]:
            result.with_setups.append(t)
            (with_setup if liquid(s) else result.filtered_illiquid).append(t)
        elif (liquid(s) and (s.get("trend_facts") or {}).get("trend") == "up"
              and (s.get("pct_from_52w_high") or -100) >= -5):
            leaders.append(t)

    rank = lambda ts: sorted(ts, key=lambda t: _strength(tech_snapshots[t]), reverse=True)
    ranked_setups = [t for t in rank(with_setup) if t not in held]
    new = ranked_setups[:max_candidates]
    for t in new:
        result.why[t] = "coded setup: " + ", ".join(st["setup"] for st in tech_snapshots[t]["setups"])
    room = max_candidates - len(new)
    result.leaders = [t for t in rank(leaders) if t not in held][:room]
    for t in result.leaders:
        result.why[t] = "leader: uptrend within 5% of 52-week high, no coded setup; judge if a trade exists"
    for t in held:
        result.why.setdefault(t, "open position: review hold or exit")
    result.cut_by_limit = ranked_setups[max_candidates:]
    result.candidates = sorted(set(new) | set(result.leaders) | set(held))
    result.with_setups.sort()
    return result
