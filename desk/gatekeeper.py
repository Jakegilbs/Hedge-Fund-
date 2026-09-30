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
TRIGGER_GAP = 0.002         # entry more than 0.2% beyond the price = the setup has not triggered yet
TRIGGER_SLIPPAGE = 0.005    # a buy-stop-limit may pay up to 0.5% above its trigger
TRIGGER_MAX_DISTANCE = 0.05 # a stop-limit only fills at its trigger, so it may sit up to 5% away


def pending_trigger(direction: str, entry: float | None, price: float | None) -> float | None:
    """The stock price that must trade before entering, or None if the setup has already triggered.

    Bullish: entry above the current price (e.g. a breakout level not yet cleared).
    Bearish: entry below the current price (a breakdown level not yet broken).
    """
    if not entry or not price:
        return None
    if direction == "bullish" and entry > price * (1 + TRIGGER_GAP):
        return round(entry, 2)
    if direction == "bearish" and entry < price * (1 - TRIGGER_GAP):
        return round(entry, 2)
    return None


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
    trigger_price: float | None = None   # buy-stop: only enter once the stock trades at/above this


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
          catalysts: CatalystReport, max_limit_distance: float = MAX_LIMIT_DISTANCE) -> GateResult:
    """Return the orders that pass every rule, with shares recomputed.

    max_limit_distance: how far the limit may be from the last price. Plain limit orders
    use 2% (catches invented prices); stop-limit orders at a trigger may sit further away.

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
        if price and abs(entry / price - 1) > max_limit_distance:
            reasons.append(f"limit ${entry} is more than {max_limit_distance:.0%} from last price ${price:.2f}")

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


# ---------- options mode: long calls and long puts only ----------

@dataclass
class ApprovedOptionOrder:
    ticker: str
    contract_symbol: str
    option_type: str            # "call" or "put"
    strike: float
    expiry: str
    contracts: int
    limit_price: float          # per share; one contract = 100 shares
    cost_usd: float
    stop_price: float           # sell if the option falls to this
    take_profit_price: float    # sell if the option rises to this
    exit_by: str                # sell by this date regardless
    stock_entry: float | None = None
    stock_stop: float | None = None
    stock_target: float | None = None
    reason: str = ""
    trigger_price: float | None = None   # only buy once the STOCK trades beyond this price


def check_options(decision, *, cash: float, pnl_today: float, equity: float, open_positions: int,
                  menu: dict[str, dict], technical, catalysts: CatalystReport, regime: RegimeReport,
                  risk: RiskLimits, opts, allowlist: set[str],
                  last_prices: dict[str, float] | None = None) -> GateResult:
    """Approve at most one long call or put from the options menu, sized with all available cash."""
    from datetime import date, timedelta

    result = GateResult()
    event_risk = {v.ticker.upper(): v.event_risk for v in catalysts.views}
    views = {v.ticker.upper(): v for v in technical.views}

    for u in decision.position_updates:
        if u.decision == "exit":
            result.approved.append(ApprovedOrder("sell", u.ticker.upper(), 0.0, 0.0,
                                                 reason=f"PM exit: {u.reason}"))

    if pnl_today <= risk.daily_halt_pct * equity:
        result.halted = f"daily halt: P&L ${pnl_today:.2f}"
    slots = max(0, risk.max_open_positions - open_positions)

    for o in sorted(decision.orders, key=lambda o: -o.conviction):
        t = o.ticker.upper()
        reasons: list[str] = []
        entry = menu.get(t) or {}
        contract = entry.get("contract")
        view = views.get(t)
        if result.halted:
            reasons.append(f"no new entries: {result.halted}")
        if t not in allowlist:
            reasons.append("not on the allowlist")
        if contract is None:
            reasons.append("no contract on the options menu for this ticker")
        elif contract["contract_symbol"] != o.contract_symbol:
            reasons.append(f"contract {o.contract_symbol} is not the one on the menu ({contract['contract_symbol']})")
        elif (contract["type"] == "call") != (o.direction == "bullish"):
            reasons.append("direction does not match the contract type")
        if o.conviction < risk.min_conviction:
            reasons.append(f"conviction {o.conviction}/5 below the minimum {risk.min_conviction}")
        if view is None or view.recommendation != "candidate" or view.direction != o.direction:
            reasons.append("the Technical Analyst does not rate this a candidate in this direction")
        elif (view.reward_risk_checked or 0) < risk.min_reward_risk:
            reasons.append(f"stock reward-to-risk {view.reward_risk_checked} below {risk.min_reward_risk}")
        if event_risk.get(t) == "high":
            reasons.append("news analyst rates event risk high")
        elif t not in event_risk:
            reasons.append("no news/catalyst report for this ticker")
        if o.direction == "bullish" and regime.posture == "flat":
            reasons.append("no calls while the market regime posture is flat")
        if not reasons and slots <= 0:
            reasons.append("one position at a time: a position is already open")
        contracts = 0
        if not reasons:
            contracts = int(cash * 0.995 // (contract["limit_price"] * 100))
            if contracts < 1:
                reasons.append(f"one contract costs ${contract['limit_price'] * 100:,.2f}; cash is ${cash:,.2f}")
        if reasons:
            result.rejected.append(Rejection(t, f"buy {o.direction} option", reasons))
            continue
        price = contract["limit_price"]
        exit_by = date.fromisoformat(contract["expiry"]) - timedelta(days=opts.exit_days_before_expiry)
        slots -= 1
        result.approved.append(ApprovedOptionOrder(
            ticker=t, contract_symbol=contract["contract_symbol"], option_type=contract["type"],
            strike=contract["strike"], expiry=contract["expiry"], contracts=contracts, limit_price=price,
            cost_usd=round(contracts * price * 100, 2),
            stop_price=round(price * (1 - opts.stop_loss_pct), 2),
            take_profit_price=round(price * (1 + opts.take_profit_pct), 2),
            exit_by=exit_by.isoformat(), stock_entry=view.entry, stock_stop=view.stop, stock_target=view.target,
            trigger_price=pending_trigger(o.direction, view.entry, (last_prices or {}).get(t)),
            reason=f"PM conviction {o.conviction}/5"))
    return result



# ---------- hybrid mode: an option when one is on the menu, otherwise fractional shares ----------

def check_hybrid(decision, *, cash: float, pnl_today: float, equity: float, positions: dict[str, float],
                 menu: dict[str, dict], technical, catalysts: CatalystReport, regime: RegimeReport,
                 risk: RiskLimits, opts, allowlist: set[str], last_prices: dict[str, float],
                 stale: set[str]) -> GateResult:
    """Approve at most one new position: a long call/put from the menu, or fractional shares
    for a bullish pick the menu marks as shares. Each order goes through the same checks as
    in options or stock mode."""
    from .schemas import OptionOrder, OptionsDecision, Order, PMDecision

    result = GateResult()
    for u in decision.position_updates:
        if u.decision == "exit":
            result.approved.append(ApprovedOrder("sell", u.ticker.upper(), 0.0, 0.0, reason=f"PM exit: {u.reason}"))
    views = {v.ticker.upper(): v for v in technical.views}
    held = dict(positions)

    for o in sorted(decision.orders, key=lambda o: -o.conviction):
        t = o.ticker.upper()
        entry = menu.get(t) or {}
        if o.instrument == "option":
            sub = check_options(
                OptionsDecision(market_view="", position_updates=[], warnings=[], honest_assessment="",
                                orders=[OptionOrder(ticker=t, direction=o.direction,
                                                    contract_symbol=o.contract_symbol or "", thesis=o.thesis,
                                                    bear_case=o.bear_case, conviction=o.conviction)]),
                cash=cash, pnl_today=pnl_today, equity=equity, open_positions=sum(1 for v in held.values() if v),
                menu=menu, technical=technical, catalysts=catalysts, regime=regime, risk=risk, opts=opts,
                allowlist=allowlist, last_prices=last_prices)
        else:
            view = views.get(t)
            reasons = []
            if entry.get("instrument") != "shares":
                reasons.append("shares are only allowed for a bullish pick with no option on the menu")
            if o.direction != "bullish":
                reasons.append("bearish trades need a put; shares cannot be sold short here")
            if view is None or view.recommendation != "candidate" or view.direction != "bullish":
                reasons.append("the Technical Analyst does not rate this a bullish candidate")
            if reasons:
                result.rejected.append(Rejection(t, "buy shares", reasons))
                continue
            price = last_prices.get(t) or view.entry
            # A setup that has not triggered yet (e.g. a breakout still below its level) is
            # entered with a buy-stop-limit at the trigger, sized and risk-checked at that price.
            trigger = pending_trigger("bullish", view.entry, price)
            limit = round(trigger * (1 + TRIGGER_SLIPPAGE), 2) if trigger else round(price, 2)
            sub = check(
                PMDecision(market_view="", position_updates=[], warnings=[], honest_assessment="",
                           orders=[Order(action="buy", ticker=t, shares=0, order_type="limit",
                                         limit_price=limit, stop_price=view.stop,
                                         target_price=view.target, reward_risk=view.reward_risk_checked or 0,
                                         thesis=o.thesis, bear_case=o.bear_case, conviction=o.conviction)]),
                equity=equity, cash=cash, pnl_today=pnl_today, positions=held, last_prices=last_prices,
                stale=stale, allowlist=allowlist, risk=risk, regime=regime, catalysts=catalysts,
                max_limit_distance=TRIGGER_MAX_DISTANCE if trigger else MAX_LIMIT_DISTANCE)
        result.rejected += sub.rejected
        result.halted = result.halted or sub.halted
        for a in sub.approved:
            if getattr(a, "action", "buy") == "sell":
                continue
            if o.instrument == "shares" and trigger:
                a.trigger_price = trigger
            result.approved.append(a)
            held[t] = 1.0          # counts toward the one-position limit for later orders
    return result
