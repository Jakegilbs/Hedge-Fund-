"""Candidate scanner: decides which tickers are worth the analysts' (paid) attention.

Every ticker in the universe is checked in code for free. A ticker qualifies as
bullish (a bullish coded setup, or a "leader": uptrend within 5% of its 52-week
high) or, when puts are allowed, bearish (a bearish coded setup, or a "laggard":
downtrend within 5% of its 52-week low). Illiquid, overextended and pinned
tickers are skipped. Bullish names are ranked by strength relative to SPY,
bearish names by weakness, coded setups before leaders/laggards, and only the
top few go on to the analysts. Open positions are always reviewed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

MAX_RSI = 80.0            # above this a stock is stretched and prone to pull back
MIN_RSI = 20.0            # below this a stock is washed out and prone to bounce
MAX_20D_MOVE = 30.0       # % move in 20 days treated as parabolic, either way
MIN_ATR_PCT = 0.8         # daily range below this: pinned (e.g. pending buyout), no swing move
BULL_SHARE = 0.6          # share of slots for bullish names when both directions are allowed


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
    direction: dict[str, str] = field(default_factory=dict)


def _relative(snap: dict, bench: dict) -> float:
    """Average 20- and 60-day performance vs SPY, in percentage points."""
    ch = snap["change_pct"]
    rs20 = (ch.get("20d") or 0.0) - (bench.get("20d") or 0.0)
    rs60 = (ch.get("60d") or 0.0) - (bench.get("60d") or 0.0)
    return (rs20 + rs60) / 2


def _problem(s: dict, direction: str) -> str | None:
    rsi, move = s.get("rsi14") or 50.0, s["change_pct"].get("20d") or 0.0
    if direction == "bullish" and (rsi > MAX_RSI or move > MAX_20D_MOVE):
        return "extended"
    if direction == "bearish" and (rsi < MIN_RSI or move < -MAX_20D_MOVE):
        return "extended"
    if s.get("atr_pct") is not None and s["atr_pct"] < MIN_ATR_PCT:
        return "pinned"
    return None


def pick_candidates(tech_snapshots: dict[str, dict], held: list[str], max_candidates: int = 8,
                    min_price: float = 0.0, min_dollar_volume: float = 0.0,
                    benchmark: str = "SPY", directions: tuple[str, ...] = ("bullish",)) -> ScanResult:
    """A stale ticker is never a new candidate; a stale held position is still
    reviewed so the PM can see and flag the data problem."""
    result = ScanResult(candidates=[], scanned=len(tech_snapshots))
    bench = (tech_snapshots.get(benchmark) or {}).get("change_pct", {})
    pools: dict[str, dict[str, list[str]]] = {d: {"setup": [], "trend": []} for d in directions}

    def liquid(s: dict) -> bool:
        return s["close"] >= min_price and (s.get("avg_dollar_volume_20d") or 0) >= min_dollar_volume

    for t, s in tech_snapshots.items():
        if s.get("stale"):
            result.stale.append(t)
            continue
        setups = [st for st in s["setups"] if st.get("direction", "bullish") in directions]
        for st in setups:
            result.setup_counts[st["setup"]] = result.setup_counts.get(st["setup"], 0) + 1
        if setups:
            result.with_setups.append(t)
        trend = (s.get("trend_facts") or {}).get("trend")
        if setups:
            direction, kind = setups[0].get("direction", "bullish"), "setup"
        elif "bullish" in directions and trend == "up" and (s.get("pct_from_52w_high") or -100) >= -5:
            direction, kind = "bullish", "trend"
        elif "bearish" in directions and trend == "down" and (s.get("pct_from_52w_low") or 100) <= 5:
            direction, kind = "bearish", "trend"
        else:
            continue
        if not liquid(s):
            if kind == "setup":
                result.filtered_illiquid.append(t)
            continue
        problem = _problem(s, direction)
        if problem == "extended":
            result.filtered_extended.append(t)
        elif problem == "pinned":
            result.filtered_pinned.append(t)
        else:
            pools[direction][kind].append(t)
            result.direction[t] = direction

    def ranked(direction: str, kind: str) -> list[str]:
        sign = 1 if direction == "bullish" else -1   # bearish: weakest first
        names = [t for t in pools[direction][kind] if t not in held]
        return sorted(names, key=lambda t: (sign * _relative(tech_snapshots[t], bench),
                                            len(tech_snapshots[t]["setups"])), reverse=True)

    if len(directions) == 1:
        slots = {directions[0]: max_candidates}
    else:
        bull = math.ceil(max_candidates * BULL_SHARE)
        slots = {"bullish": bull, "bearish": max_candidates - bull}

    picked: dict[str, list[str]] = {}
    for d in directions:
        order = ranked(d, "setup") + ranked(d, "trend")
        picked[d] = order[:slots[d]]
        result.cut_by_limit += [t for t in ranked(d, "setup") if t not in picked[d]]
    # Give unused slots on one side to the other side.
    for d in directions:
        spare = sum(slots[o] - len(picked[o]) for o in directions if o != d)
        if spare > 0:
            extra = [t for t in ranked(d, "setup") + ranked(d, "trend") if t not in picked[d]][:spare]
            picked[d] += extra
            result.cut_by_limit = [t for t in result.cut_by_limit if t not in extra]

    for d, names in picked.items():
        for t in names:
            snap = tech_snapshots[t]
            coded = [st["setup"] for st in snap["setups"] if st.get("direction", "bullish") == d]
            if coded:
                result.why[t] = f"{d} coded setup: " + ", ".join(coded)
            elif d == "bullish":
                result.why[t] = ("leader: strong vs SPY, uptrend within 5% of 52-week high, no coded setup; "
                                 "judge if a trade exists")
                result.leaders.append(t)
            else:
                result.why[t] = ("laggard: weak vs SPY, downtrend within 5% of 52-week low, no coded setup; "
                                 "judge if a bearish trade exists")
                result.leaders.append(t)
    for t in held:
        result.why.setdefault(t, "open position: review hold or exit")
    result.candidates = sorted({t for names in picked.values() for t in names} | set(held))
    result.with_setups.sort()
    return result
