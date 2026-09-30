"""Choosing the option contract: code, not AI.

For each ticker the Technical Analyst rates a candidate, code pulls the option
chain and picks ONE contract to offer the Portfolio Manager: a call for a
bullish view, a put for a bearish one. Rules (config/desk.toml, [options]):
expiry inside the day window and before any earnings date (earnings crush
option prices), delta near the target (moves with the stock, decays slower
than far-out contracts), enough open interest, a tight bid/ask spread, and a
price the account can pay for at least one contract (100 shares each).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class OptionsConfig:
    min_days_to_expiry: int = 21
    max_days_to_expiry: int = 60
    target_delta: float = 0.55
    min_delta: float = 0.35
    max_delta: float = 0.75
    max_spread_pct: float = 0.12
    min_open_interest: int = 100
    stop_loss_pct: float = 0.15
    take_profit_pct: float = 1.00
    exit_days_before_expiry: int = 7
    risk_free_rate: float = 0.045


def _num(x) -> float:
    """yfinance leaves blanks as NaN; treat missing numbers as 0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) else v


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(spot: float, strike: float, years: float, iv: float, rate: float, call: bool) -> float | None:
    """Black-Scholes delta (calls 0..1, puts -1..0). None when inputs are unusable."""
    if spot <= 0 or strike <= 0 or years <= 0 or not iv or iv <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2) * years) / (iv * math.sqrt(years))
    return _norm_cdf(d1) if call else _norm_cdf(d1) - 1


def limit_price(bid: float, ask: float) -> float:
    """Halfway between mid and ask: fills more often than mid, pays less than ask."""
    mid = (bid + ask) / 2
    return round(min(ask, (mid + ask) / 2), 2)


def select_contract(chains: dict[str, pd.DataFrame], direction: str, spot: float, cash: float,
                    cfg: OptionsConfig, today: date, earnings: str | None = None) -> tuple[dict | None, str]:
    """Pick one contract from {expiry "YYYY-MM-DD": calls-or-puts DataFrame}.

    Returns (contract, note). contract is None when nothing qualifies; note says why.
    """
    call = direction == "bullish"
    earn = date.fromisoformat(earnings) if earnings else None
    eligible, cheapest_ok = [], None
    for expiry, df in chains.items():
        exp = date.fromisoformat(expiry)
        dte = (exp - today).days
        if not (cfg.min_days_to_expiry <= dte <= cfg.max_days_to_expiry):
            continue
        if earn and today <= earn <= exp:
            continue
        for row in df.itertuples():
            bid, ask = _num(getattr(row, "bid", 0)), _num(getattr(row, "ask", 0))
            oi = int(_num(getattr(row, "openInterest", 0)))
            iv = _num(getattr(row, "impliedVolatility", 0))
            if bid <= 0 or ask <= 0 or oi < cfg.min_open_interest:
                continue
            mid = (bid + ask) / 2
            if (ask - bid) / mid > cfg.max_spread_pct:
                continue
            delta = bs_delta(spot, _num(row.strike), dte / 365, iv, cfg.risk_free_rate, call)
            if delta is None or not (cfg.min_delta <= abs(delta) <= cfg.max_delta):
                continue
            price = limit_price(bid, ask)
            cost = price * 100
            if cheapest_ok is None or cost < cheapest_ok:
                cheapest_ok = cost
            if cost > cash * 0.995:
                continue
            eligible.append({
                "contract_symbol": str(row.contractSymbol), "type": "call" if call else "put",
                "strike": float(row.strike), "expiry": expiry, "days_to_expiry": dte,
                "bid": bid, "ask": ask, "limit_price": price, "cost_per_contract": round(cost, 2),
                "delta": round(delta, 2), "implied_volatility": round(iv, 3), "open_interest": oi,
                "contracts_affordable": int(cash * 0.995 // cost),
                "score": abs(abs(delta) - cfg.target_delta) + 0.002 * abs(dte - 35),
            })
    if eligible:
        best = min(eligible, key=lambda c: c["score"])
        best.pop("score")
        return best, "ok"
    if cheapest_ok is not None:
        return None, (f"cheapest suitable {'call' if call else 'put'} costs ${cheapest_ok:,.0f} per contract; "
                      f"cash is ${cash:,.2f}")
    return None, ("no liquid contract in the expiry window"
                  + (" before earnings" if earn else "") + " with a usable delta and spread"
                  + " (outside market hours Yahoo often shows no bid/ask: run while the market is open)")


def fetch_chains(ticker: str, cfg: OptionsConfig, today: date) -> dict[str, pd.DataFrame]:
    """Calls and puts per expiry inside the window, from yfinance: {"calls": {...}, "puts": {...}}."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    out: dict[str, dict[str, pd.DataFrame]] = {"calls": {}, "puts": {}}
    for expiry in t.options or ():
        dte = (date.fromisoformat(expiry) - today).days
        if cfg.min_days_to_expiry <= dte <= cfg.max_days_to_expiry:
            chain = t.option_chain(expiry)
            out["calls"][expiry] = chain.calls
            out["puts"][expiry] = chain.puts
    return out


def options_menu(picks: list[dict], cash: float, cfg: OptionsConfig, today: date,
                 fetch=fetch_chains) -> dict[str, dict]:
    """picks: [{"ticker", "direction", "spot", "earnings"}]. Returns ticker -> {"contract", "note"}."""
    menu = {}
    for p in picks:
        try:
            chains = fetch(p["ticker"], cfg, today)
            side = chains["calls" if p["direction"] == "bullish" else "puts"]
            contract, note = select_contract(side, p["direction"], p["spot"], cash, cfg, today, p.get("earnings"))
        except Exception as e:  # a chain that fails to load just means no contract
            contract, note = None, f"option chain unavailable: {e}"
        menu[p["ticker"]] = {"direction": p["direction"], "contract": contract, "note": note}
    return menu
