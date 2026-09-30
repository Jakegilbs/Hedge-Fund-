"""Candidate scanner: decides which tickers are worth the analysts' (paid) attention.

Every ticker in the universe is checked in code for free. Tickers with a fresh
setup that pass the filters (liquid, not overextended, not pinned) are ranked by
strength relative to SPY over 20 and 60 days, and only the top few go on to the
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
    filtered_extended: list[str] = field(default_factory=list)
    filtered_pinned: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    setup_counts: dict[str, int] = field(default_factory=dict)
    why: dict[str, str] = field(default_factory=dict)


MAX_RSI = 80.0            # above this a stock is stretched and prone to pull back
MAX_20D_GAIN = 30.0       # % gain in 20 days treated as parabolic
MIN_ATR_PCT = 0.8         # daily range below this: pinned (e.g. pending buyout), no swing move


def _strength(snap: dict, bench: dict) -> tuple:
    """Relative strength vs SPY: average of 20- and 60-day outperformance, then setup count."""
    ch = snap["change_pct"]
    rs20 = (ch.get("20d") or 0.0) - (bench.get("20d") or 0.0)
    rs60 = (ch.get("60d") or 0.0) - (bench.get("60d") or 0.0)
    return ((rs20 + rs60) / 2, len(snap["setups"]))


def _problem(s: dict) -> str | None:
    if (s.get("rsi14") or 0) > MAX_RSI or (s["change_pct"].get("20d") or 0) > MAX_20D_GAIN:
        return "extended"
    if s.get("atr_pct") is not None and s["atr_pct"] < MIN_ATR_PCT:
        return "pinned"
    return None


def pick_candidates(tech_snapshots: dict[str, dict], held: list[str], max_candidates: int = 8,
                    min_price: float = 0.0, min_dollar_volume: float = 0.0,
                    benchmark: str = "SPY") -> ScanResult:
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
        problem = _problem(s)
        is_setup = bool(s["setups"])
        is_leader = ((s.get("trend_facts") or {}).get("trend") == "up"
                     and (s.get("pct_from_52w_high") or -100) >= -5)
        if not (is_setup or is_leader):
            continue
        if not liquid(s):
            if is_setup:
                result.filtered_illiquid.append(t)
        elif problem == "extended":
            result.filtered_extended.append(t)
        elif problem == "pinned":
            result.filtered_pinned.append(t)
        else:
            (with_setup if is_setup else leaders).append(t)

    bench = (tech_snapshots.get(benchmark) or {}).get("change_pct", {})
    rank = lambda ts: sorted(ts, key=lambda t: _strength(tech_snapshots[t], bench), reverse=True)
    ranked_setups = [t for t in rank(with_setup) if t not in held]
    new = ranked_setups[:max_candidates]
    for t in new:
        result.why[t] = "coded setup: " + ", ".join(st["setup"] for st in tech_snapshots[t]["setups"])
    room = max_candidates - len(new)
    result.leaders = [t for t in rank(leaders) if t not in held][:room]
    for t in result.leaders:
        result.why[t] = ("leader: strong vs SPY, uptrend within 5% of 52-week high, no coded setup; "
                         "judge if a trade exists")
    for t in held:
        result.why.setdefault(t, "open position: review hold or exit")
    result.cut_by_limit = ranked_setups[max_candidates:]
    result.candidates = sorted(set(new) | set(result.leaders) | set(held))
    result.with_setups.sort()
    return result
