"""Candidate scanner: decides which tickers are worth the analysts' (paid) attention."""
from __future__ import annotations


def pick_candidates(tech_snapshots: dict[str, dict], held: list[str]) -> list[str]:
    """Tickers with a fresh setup, plus every open position (those always get reviewed).

    A stale ticker is never a new candidate; a stale held position is still
    reviewed so the PM can see and flag the data problem.
    """
    fresh_setups = [t for t, s in tech_snapshots.items() if s["setups"] and not s.get("stale")]
    return sorted(set(fresh_setups) | set(held))
