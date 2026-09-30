"""Candidate scanner: decides which tickers are worth the analysts' (paid) attention.

Every ticker in the universe is checked in code for free. Tickers with a fresh
setup that pass the liquidity filter are ranked, strongest first (60-day
relative strength, then number of setups), and only the top few go on to the
analysts. Open positions are always reviewed.
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


def _strength(snap: dict) -> tuple:
    return (snap["change_pct"].get("60d") or -999.0, len(snap["setups"]))


def pick_candidates(tech_snapshots: dict[str, dict], held: list[str], max_candidates: int = 8,
                    min_price: float = 0.0, min_dollar_volume: float = 0.0) -> ScanResult:
    """A stale ticker is never a new candidate; a stale held position is still
    reviewed so the PM can see and flag the data problem."""
    result = ScanResult(candidates=[], scanned=len(tech_snapshots))
    liquid = []
    for t, s in tech_snapshots.items():
        if not s["setups"] or s.get("stale"):
            continue
        result.with_setups.append(t)
        if s["close"] < min_price or (s.get("avg_dollar_volume_20d") or 0) < min_dollar_volume:
            result.filtered_illiquid.append(t)
        else:
            liquid.append(t)
    ranked = sorted(liquid, key=lambda t: _strength(tech_snapshots[t]), reverse=True)
    new = [t for t in ranked if t not in held][:max_candidates]
    result.cut_by_limit = [t for t in ranked if t not in new and t not in held]
    result.candidates = sorted(set(new) | set(held))
    result.with_setups.sort()
    return result
