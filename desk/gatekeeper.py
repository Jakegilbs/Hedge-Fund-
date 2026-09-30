"""The Gatekeeper: code that checks every order the Portfolio Manager proposes.

The PM's own share counts and reward-to-risk are ignored and recomputed here.
Every limit is enforced from config/desk.toml; the agents cannot change them.
An order either passes every check or is rejected with the reasons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import RiskLimits
from .schemas import CatalystReport, PMDecision, RegimeReport

MAX_LIMIT_DISTANCE = 0.02   # limit price must be within 2% of the last price
MIN_ORDER_USD = 1.00        # Robinhood's minimum fractional order


@dataclass
class ApprovedOrder:
    action: str
    ticker: str
    shares: float
    limit_price: float
    stop_price: float | None = None
    target_price: float | None = None
    reward_risk: float | None = None
    notional_usd: float = 0.0
    risk_usd: float = 0.0
    reason: str = ""


@dataclass
class Rejection:
    ticker: str
    action: str
    reasons: list[str]


@dataclass
class GateResult:
    approved: list[ApprovedOrder] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    halted: str | None = None


def floor_shares(x: float) -> float:
    """Round down to 4 decimal places (fractional shares)."""
    return math.floor(x * 10_000) / 10_000


def check(decision: PMDecision, *, equity: float, cash: float, pnl_today: float,
          positions: dict[str, float], last_prices: dict[str, float], stale: set[str],
          allowlist: set[str], risk: RiskLimits, regime: RegimeReport,
          catalysts: CatalystReport) -> GateResult:
    """Return the orders that pass every rule, with shares recomputed.

    positions: ticker -> shares held. last_prices: ticker -> latest price.
    """
    result = GateResult()
    event_risk = {v.ticker: v.event_risk for v in catalysts.views}

    # Exits first: they are always allowed for held positions, even when halted.
    exits = {o.ticker.upper() for o in decision.orders if o.action == "sell"}
    exits |= {u.ticker.upper() for u in decision.position_updates if u.decision == "exit"}
    for ticker in sorted(exits):
        held = positions.get(ticker, 0.0)
        price = last_prices.get(ticker)
        if held <= 0:
            result.rejected.append(Rejection(ticker, "sell", ["not held: nothing to sell"]))
            continue
        if price is None:
            result.rejected.append(Rejection(ticker, "sell", ["no price: exit manually"]))
            continue
        result.approved.append(ApprovedOrder("sell", ticker, held, round(price, 2),
                                             notional_usd=round(held * price, 2), reason="PM exit"))

    halt_level = risk.daily_halt_pct * equity
    if pnl_today <= halt_level:
        result.halted = f"daily halt: P&L ${pnl_today:.2f} at or below ${halt_level:.2f}"
    elif regime.posture == "flat":
        result.halted = "market regime posture is flat"

    open_count = sum(1 for s in positions.values() if s > 0)
    new_allowed = min(regime.max_new_positions_today, risk.max_open_positions - open_count)
    cash_left = cash
    buys = sorted((o for o in decision.orders if o.action == "buy"), key=lambda o: -o.conviction)

    for o in buys:
        t = o.ticker.upper()
        reasons: list[str] = []
        price = last_prices.get(t)
        entry, stop, target = o.limit_price, o.stop_price, o.target_price

        if result.halted:
            reasons.append(f"no new entries: {result.halted}")
        if t not in allowlist:
            reasons.append("not on the allowlist")
        if positions.get(t, 0) > 0:
            reasons.append("already held: no adding or averaging down")
        if t in stale or price is None:
            reasons.append("price data stale or missing")
        if event_risk.get(t) == "high":
            reasons.append("news analyst rates event risk high")
        elif t not in event_risk:
            reasons.append("no news/catalyst report for this ticker")
        if not (stop < entry < target):
            reasons.append(f"levels out of order: stop {stop} < entry {entry} < target {target} required")
        if price and abs(entry / price - 1) > MAX_LIMIT_DISTANCE:
            reasons.append(f"limit ${entry} is more than {MAX_LIMIT_DISTANCE:.0%} from last price ${price:.2f}")

        if entry > stop and (entry - stop) / entry > risk.max_stop_distance_pct:
            reasons.append(f"stop {(entry - stop) / entry:.1%} below entry; the limit is "
                           f"{risk.max_stop_distance_pct:.0%}")
        if o.conviction < risk.min_conviction:
            reasons.append(f"conviction {o.conviction}/5 below the minimum {risk.min_conviction}")

        rr = (target - entry) / (entry - stop) if entry > stop else 0.0
        if rr < risk.min_reward_risk:
            reasons.append(f"reward-to-risk {rr:.2f} below {risk.min_reward_risk}")

        if not reasons and new_allowed <= 0:
            reasons.append("position count limit reached for today")

        shares = 0.0
        if not reasons:
            per_share_risk = entry - stop
            # A small cash buffer keeps an all-in limit order from exceeding buying power.
            shares = floor_shares(min(risk.risk_per_trade * equity / per_share_risk,
                                      risk.max_position_pct * equity / entry,
                                      cash_left * 0.995 / entry))
            if shares * entry < MIN_ORDER_USD:
                reasons.append(f"size ${shares * entry:.2f} below the ${MIN_ORDER_USD:.2f} minimum")

        if reasons:
            result.rejected.append(Rejection(t, "buy", reasons))
            continue

        notional = round(shares * entry, 2)
        cash_left -= notional
        new_allowed -= 1
        result.approved.append(ApprovedOrder(
            "buy", t, shares, round(entry, 2), round(stop, 2), round(target, 2), round(rr, 2),
            notional, round(shares * (entry - stop), 2),
            reason=f"PM conviction {o.conviction}/5 (PM proposed {o.shares} shares)"))
    return result
