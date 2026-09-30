"""Backtester: would the scanner's setups have made money? Free: code only, no AI.

    python -m desk.backtest                      # the live universe, about 4 years
    python -m desk.backtest --tickers NVDA,AMD   # just these
    python -m desk.backtest --target-r 3 --hold 10 --spread 0.05

It replays every trading day of history. On each day, the same setup rules the
live scanner uses (desk/indicators.py) are checked, with the scanner's filters
(liquidity, overextended, pinned) and the Gatekeeper's order rules (trigger at
most 5% away, stock stop at most 10% away). Each signal is then traded two ways:

- shares: enter when the trigger trades (a buy-stop-limit, like the live desk),
  exit at the stop, at the target (entry + target_r x risk), or after `hold` days;
- an option: a ~0.55-delta call (put for bearish) about 35 days out, priced with
  Black-Scholes from the stock's recent volatility, with the live exit plan:
  -15% stop, +100% take profit, sell 5 days before expiry (or after `hold` days).

Finally a $100 account is run through it the way the live desk trades: one
all-in position at a time, the best-ranked signal each day, a call when one is
affordable, otherwise shares (hybrid mode).

Limits, stated plainly: the AI analysts and the PM are not simulated (they
would cost money per day of history); the pick is the scanner's top-ranked
signal. Option prices are an estimate, not historical quotes. The universe is
today's list of stocks, which leaves out companies that failed or were
delisted, so results lean optimistic.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .gatekeeper import TRIGGER_GAP, TRIGGER_MAX_DISTANCE, TRIGGER_SLIPPAGE
from .indicators import atr, ema, rsi, sma
from .scanner import MAX_20D_MOVE, MAX_RSI, MIN_ATR_PCT, MIN_RSI

BACKTEST_DIR = DATA_DIR / "backtests"
BACKTEST_CACHE = DATA_DIR / "cache" / "bars_long"
LIVE_SETUP_NAMES = ("breakout", "pullback", "vcp", "breakdown", "bear_rally")   # desk/indicators.py
# Candidate strategies tested here before they go live:
#   momentum: a strong 6-month winner near its high, bought at the close and held
#             while it stays above its 20-day average (max 40 days).
#   dip_buy:  a stock in an uptrend after a sharp 2-day drop (RSI(2) below 10),
#             bought at the close and sold on the first close above its 5-day average (max 10 days).
SETUP_NAMES = LIVE_SETUP_NAMES + ("momentum", "dip_buy")
DIRECTION = {"breakout": "bullish", "pullback": "bullish", "vcp": "bullish",
             "breakdown": "bearish", "bear_rally": "bearish", "momentum": "bullish", "dip_buy": "bullish"}
# Exit styles other than stop / target / max_hold_days: (line column, exit when close is "below"/"above" it,
# max trading days held). These setups have no fixed target; the line is the exit.
EXIT_LINE = {"momentum": ("ema20", "below", 40), "dip_buy": ("sma5", "above", 10)}
MOMENTUM_MIN_6M = 0.20      # at least +20% over 6 months
MOMENTUM_NEAR_HIGH = 0.90   # within 10% of the 52-week high
DIP_RSI2 = 10.0
MIN_TRADES = 30   # fewer trades than this: too few to judge


@dataclass(frozen=True)
class BTConfig:
    target_r: float = 2.0              # target = entry + target_r x (entry - stop)
    max_hold_days: int = 15            # trading days; the desk's swing horizon is 2-15
    trigger_days: int = 1              # days a stop-limit entry order stays working
    slippage_pct: float = 0.0005       # per side, shares
    option_dte: int = 35               # calendar days to expiry at entry
    option_delta: float = 0.55
    option_spread_pct: float = 0.08    # (ask - bid) / mid; the live desk accepts up to 0.12
    iv_premium: float = 1.1            # implied vol = recent realised vol x this
    option_stop_pct: float = 0.15
    option_stop_on_stock: bool = False # exit the option when the STOCK hits the setup's stop instead
    option_take_profit_pct: float = 1.00
    exit_days_before_expiry: int = 5
    risk_free_rate: float = 0.045
    max_stop_distance_pct: float = 0.10
    min_price: float = 5.0
    min_dollar_volume: float = 20_000_000
    upside_weight: float = 3.0
    start_equity: float = 100.0
    ai_cost_per_run: float = 0.10      # charged on each day the analysts would run


# ---------------------------------------------------------------- signals

def signal_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every setup rule from desk/indicators.py evaluated on every bar at once.

    Row t uses only bars up to t, so it equals running the live setup function
    on df.iloc[:t + 1] (tests check this). Columns: `<setup>` (bool),
    `<setup>_entry`, `<setup>_stop`, plus the scanner's filter inputs.
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    n = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    a = atr(df)
    e20, e50, s200 = ema(c, 20), ema(c, 50), sma(c, 200)
    vol_mean = v.rolling(20, min_periods=10).mean()
    vol_x = v / vol_mean.where(vol_mean != 0, 1.0)
    out = pd.DataFrame(index=df.index)

    prior_high = c.shift(1).rolling(252, min_periods=120).max()
    out["breakout"] = (n >= 200) & (c >= prior_high * 0.995) & (vol_x >= 1.15)
    out["breakout_entry"] = np.maximum(h, prior_high)
    out["breakout_stop"] = np.minimum(l, out["breakout_entry"] - 1.1 * a)

    near20 = (c - e20).abs() / e20 < 0.02
    out["pullback"] = (n >= 220) & (c > e50) & (e50 > s200) & near20 & (c > c.shift(1))
    out["pullback_entry"] = h
    out["pullback_stop"] = np.minimum(l.rolling(5).min(), h - 1.25 * a)

    atrp = a / c
    atrp_rank = atrp / atrp.rolling(120, min_periods=60).max()
    near_high = c / c.rolling(100, min_periods=60).max()
    pivot = h.rolling(20).max()
    out["vcp"] = (n >= 120) & (atrp_rank < 0.45) & (near_high > 0.95)
    out["vcp_entry"] = pivot
    out["vcp_stop"] = pivot - 1.5 * a

    prior_low = c.shift(1).rolling(252, min_periods=120).min()
    out["breakdown"] = (n >= 200) & (c <= prior_low * 1.005) & (vol_x >= 1.15)
    out["breakdown_entry"] = np.minimum(l, prior_low)
    out["breakdown_stop"] = np.maximum(h, out["breakdown_entry"] + 1.1 * a)

    out["bear_rally"] = (n >= 220) & (c < e50) & (e50 < s200) & near20 & (c < c.shift(1))
    out["bear_rally_entry"] = l
    out["bear_rally_stop"] = np.maximum(h.rolling(5).max(), l + 1.25 * a)

    ret126 = c / c.shift(126) - 1
    out["momentum"] = ((n >= 252) & (c > s200) & (e50 > s200) & (c > e20) & (ret126 >= MOMENTUM_MIN_6M)
                       & (c >= MOMENTUM_NEAR_HIGH * h.rolling(252, min_periods=200).max()))
    out["momentum_entry"] = c
    out["momentum_stop"] = c - 2.0 * a

    out["dip_buy"] = (n >= 220) & (c > s200) & (e50 > s200) & (rsi(c, 2) < DIP_RSI2)
    out["dip_buy_entry"] = c
    out["dip_buy_stop"] = c - 2.5 * a

    out["ema20"] = e20
    out["sma5"] = c.rolling(5).mean()

    for name in SETUP_NAMES:
        out[f"{name}_entry"] = out[f"{name}_entry"].round(2)
        out[f"{name}_stop"] = out[f"{name}_stop"].round(2)
        out[name] = out[name].fillna(False).astype(bool)

    out["close"] = c
    out["atr_pct"] = a / c * 100
    out["rsi"] = rsi(c)
    out["chg20"] = c.pct_change(20) * 100
    out["chg60"] = c.pct_change(60) * 100
    out["dollar_volume"] = (c * v).rolling(20, min_periods=1).mean()
    out["vol20"] = np.log(c).diff().rolling(20).std() * math.sqrt(252)   # annualised
    return out


def filter_reason(row, direction: str, cfg: BTConfig) -> str | None:
    """The live scanner's filters (desk/scanner.py) on one signal day."""
    if row.close < cfg.min_price or row.dollar_volume < cfg.min_dollar_volume:
        return "illiquid"
    rsi_v = row.rsi if math.isfinite(row.rsi) else 50.0
    move = row.chg20 if math.isfinite(row.chg20) else 0.0
    if direction == "bullish" and (rsi_v > MAX_RSI or move > MAX_20D_MOVE):
        return "extended"
    if direction == "bearish" and (rsi_v < MIN_RSI or move < -MAX_20D_MOVE):
        return "extended"
    if math.isfinite(row.atr_pct) and row.atr_pct < MIN_ATR_PCT:
        return "pinned"
    return None


# ---------------------------------------------------------------- trade simulation

@dataclass
class Fill:
    day: int          # bar index of the fill
    price: float


def find_fill(o, h, l, c, i: int, direction: str, entry: float, cfg: BTConfig) -> Fill | None:
    """How the live desk enters: a buy-stop-limit at the trigger (sell side for bearish)
    when the setup has not triggered yet, otherwise a limit order at the entry next day."""
    bull = direction == "bullish"
    close = c[i]
    pending = entry > close * (1 + TRIGGER_GAP) if bull else entry < close * (1 - TRIGGER_GAP)
    last = min(i + cfg.trigger_days, len(c) - 1)
    for j in range(i + 1, last + 1):
        if pending:
            if bull and h[j] >= entry:
                limit = entry * (1 + TRIGGER_SLIPPAGE)
                if o[j] <= limit:
                    return Fill(j, max(o[j], entry))
                if l[j] <= limit:
                    return Fill(j, limit)
            elif not bull and l[j] <= entry:
                limit = entry * (1 - TRIGGER_SLIPPAGE)
                if o[j] >= limit:
                    return Fill(j, min(o[j], entry))
                if h[j] >= limit:
                    return Fill(j, limit)
        else:
            if bull and l[j] <= entry:
                return Fill(j, min(o[j], entry))
            if not bull and h[j] >= entry:
                return Fill(j, max(o[j], entry))
    return None


def _line_exit(c: float, line: float, side: str) -> bool:
    return math.isfinite(line) and (c < line if side == "below" else c > line)


def simulate_shares(o, h, l, c, fill: Fill, direction: str, entry: float, stop: float,
                    cfg: BTConfig, exit_line=None) -> dict:
    """Exit at the stop, the target or after max_hold_days. Gaps fill at the open.
    When the stop and the target are both touched in one day, the stop is assumed.
    exit_line = (values, "below"/"above", max days): no target; exit at the close
    once the close crosses the line (momentum, dip_buy)."""
    bull = direction == "bullish"
    sign = 1 if bull else -1
    risk = (entry - stop) * sign
    target = entry + sign * cfg.target_r * risk
    max_hold = cfg.max_hold_days
    if exit_line is not None:
        line, side, max_hold = exit_line
        target = math.inf if bull else -math.inf
    j, px = fill.day, fill.price
    best = px
    exit_day, exit_px, reason = None, None, None
    for k in range(j, len(c)):
        lo, hi = (l[k], h[k]) if bull else (h[k], l[k])       # worst, best price of the day
        if k == j:     # entry day: only the stop (and a close-based exit) can be judged after the fill
            if (lo <= stop) if bull else (lo >= stop):
                exit_day, exit_px, reason = k, stop, "stop"
                break
            if exit_line is not None and _line_exit(c[k], line[k], side):
                exit_day, exit_px, reason = k, c[k], "exit line"
                break
            continue
        best = max(best, hi) if bull else min(best, hi)
        if (o[k] <= stop) if bull else (o[k] >= stop):
            exit_day, exit_px, reason = k, o[k], "stop (gap)"
        elif (lo <= stop) if bull else (lo >= stop):
            exit_day, exit_px, reason = k, stop, "stop"
        elif (o[k] >= target) if bull else (o[k] <= target):
            exit_day, exit_px, reason = k, o[k], "target (gap)"
        elif (hi >= target) if bull else (hi <= target):
            exit_day, exit_px, reason = k, target, "target"
        elif exit_line is not None and _line_exit(c[k], line[k], side):
            exit_day, exit_px, reason = k, c[k], "exit line"
        elif k - j >= max_hold:
            exit_day, exit_px, reason = k, c[k], "time"
        if exit_day is not None:
            break
    if exit_day is None:
        exit_day, exit_px, reason = len(c) - 1, c[-1], "open"
    ret = (exit_px / px - 1) * sign - 2 * cfg.slippage_pct
    return {"exit_day": exit_day, "exit_price": round(float(exit_px), 4), "exit_reason": reason,
            "return_pct": round(ret * 100, 3), "r_multiple": round((exit_px - px) * sign / risk, 3),
            "mfe_r": round((best - px) * sign / risk, 3), "days_held": exit_day - j}


_N = NormalDist()


def bs_price(spot: float, strike: float, years: float, iv: float, rate: float, call: bool) -> float:
    """Black-Scholes price per share; intrinsic value at expiry."""
    if years <= 0 or iv <= 0:
        return max(0.0, spot - strike) if call else max(0.0, strike - spot)
    sq = iv * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + iv * iv / 2) * years) / sq
    d2 = d1 - sq
    disc = math.exp(-rate * years)
    if call:
        return spot * _N.cdf(d1) - strike * disc * _N.cdf(d2)
    return strike * disc * _N.cdf(-d2) - spot * _N.cdf(-d1)


def strike_for_delta(spot: float, years: float, iv: float, rate: float, delta: float, call: bool) -> float:
    """The strike whose Black-Scholes delta is `delta` (calls) or `-delta` (puts)."""
    d1 = _N.inv_cdf(delta if call else 1 - delta)
    return spot * math.exp(-d1 * iv * math.sqrt(years) + (rate + iv * iv / 2) * years)


def option_cost_estimate(spot: float, direction: str, vol: float, cfg: BTConfig) -> float | None:
    """What one contract would cost on the signal day (known before any fill)."""
    if not math.isfinite(vol) or vol <= 0:
        return None
    call = direction == "bullish"
    iv = min(max(vol * cfg.iv_premium, 0.10), 3.0)
    t0 = cfg.option_dte / 365
    strike = strike_for_delta(spot, t0, iv, cfg.risk_free_rate, cfg.option_delta, call)
    return round(bs_price(spot, strike, t0, iv, cfg.risk_free_rate, call) * (1 + cfg.option_spread_pct / 4) * 100, 2)


def simulate_option(o, h, l, c, dates, fill: Fill, direction: str, vol: float, cfg: BTConfig,
                    stock_stop: float | None = None, exit_line=None) -> dict | None:
    """Estimated long call/put with the live exit plan. Buys halfway between mid and ask,
    sells halfway between mid and bid (the spread is paid both ways); implied
    volatility is held constant, so volatility crush is not modelled.
    With cfg.option_stop_on_stock the stop is the stock's setup stop, not an option %."""
    if not math.isfinite(vol) or vol <= 0:
        return None
    call = direction == "bullish"
    iv = min(max(vol * cfg.iv_premium, 0.10), 3.0)
    r = cfg.risk_free_rate
    j, spot = fill.day, fill.price
    t0 = cfg.option_dte / 365
    strike = strike_for_delta(spot, t0, iv, r, cfg.option_delta, call)
    half = cfg.option_spread_pct / 4
    pay = bs_price(spot, strike, t0, iv, r, call) * (1 + half)
    if pay <= 0.01:
        return None
    stop_v = pay * (1 - cfg.option_stop_pct)
    tp_v = pay * (1 + cfg.option_take_profit_pct)
    last_day = cfg.option_dte - cfg.exit_days_before_expiry

    def value(s: float, k: int) -> float:
        days = (dates[k] - dates[j]).days
        return bs_price(s, strike, (cfg.option_dte - days) / 365, iv, r, call) * (1 - half)

    on_stock = cfg.option_stop_on_stock and stock_stop is not None

    def stopped(s: float) -> bool:
        if on_stock:
            return s <= stock_stop if call else s >= stock_stop
        return False

    def stop_value(k: int) -> float:
        return value(stock_stop, k) if on_stock else stop_v

    exit_day, exit_v, reason = None, None, None
    for k in range(j, len(c)):
        worst, best = (l[k], h[k]) if call else (h[k], l[k])
        hit = stopped if on_stock else (lambda s, k=k: value(s, k) <= stop_v)
        if k == j:
            if hit(worst):
                exit_day, exit_v, reason = k, stop_value(k), "stop"
                break
            continue
        open_v = value(o[k], k)
        if hit(o[k]):
            exit_day, exit_v, reason = k, open_v, "stop (gap)"
        elif hit(worst):
            exit_day, exit_v, reason = k, stop_value(k), "stop"
        elif open_v >= tp_v:
            exit_day, exit_v, reason = k, open_v, "take profit (gap)"
        elif value(best, k) >= tp_v:
            exit_day, exit_v, reason = k, tp_v, "take profit"
        elif (dates[k] - dates[j]).days >= last_day:
            exit_day, exit_v, reason = k, value(c[k], k), "expiry exit"
        elif exit_line is not None and _line_exit(c[k], exit_line[0][k], exit_line[1]):
            exit_day, exit_v, reason = k, value(c[k], k), "exit line"
        elif k - j >= (exit_line[2] if exit_line is not None else cfg.max_hold_days):
            exit_day, exit_v, reason = k, value(c[k], k), "time"
        if exit_day is not None:
            break
    if exit_day is None:
        exit_day, exit_v, reason = len(c) - 1, value(c[-1], len(c) - 1), "open"
    # How far the stock may move against the position before the option stop hits
    # (delta approximation; the spread paid on entry already uses up part of the room).
    room = pay * (1 - half) / (1 + half) - stop_v
    stop_move = max(0.0, room) / (cfg.option_delta * spot) * 100
    return {"opt_strike": round(strike, 2), "opt_iv": round(iv, 3), "opt_cost": round(pay * 100, 2),
            "opt_stop_stock_move_pct": round(stop_move, 2),
            "opt_exit_reason": reason, "opt_return_pct": round((exit_v / pay - 1) * 100, 3),
            "opt_exit_day": exit_day, "opt_days_held": exit_day - j}


def trades_for_ticker(ticker: str, df: pd.DataFrame, cfg: BTConfig, start: pd.Timestamp | None = None,
                      spy: pd.DataFrame | None = None) -> tuple[list[dict], dict[str, int]]:
    """Every signal of every setup for one ticker, traded. A setup is not re-entered
    while its previous trade is still open. Returns (trades, filtered counts)."""
    sig = signal_frame(df)
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    lines = {col: sig[col].to_numpy(float) for col in ("ema20", "sma5")}
    dates = list(df.index)
    spy_up = None
    if spy is not None:
        s = spy["Close"]
        spy_up = (s > ema(s, 50)).reindex(df.index).ffill()
        spy_chg = s.pct_change(20).reindex(df.index) * 100, s.pct_change(60).reindex(df.index) * 100
    filtered: dict[str, int] = {}
    trades: list[dict] = []
    busy = {name: -1 for name in SETUP_NAMES}
    rows = sig.itertuples()
    for i, row in enumerate(rows):
        if start is not None and dates[i] < start:
            continue
        for name in SETUP_NAMES:
            if not getattr(row, name) or i <= busy[name] or i >= len(c) - 1:
                continue
            direction = DIRECTION[name]
            entry, stop = getattr(row, f"{name}_entry"), getattr(row, f"{name}_stop")
            why = filter_reason(row, direction, cfg)
            if why is None and not (math.isfinite(entry) and math.isfinite(stop)):
                why = "no levels"
            if why is None and abs(entry / c[i] - 1) > TRIGGER_MAX_DISTANCE:
                why = "trigger too far"
            if why is None and ((entry - stop) if direction == "bullish" else (stop - entry)) <= 0:
                why = "no levels"
            if why:
                filtered[why] = filtered.get(why, 0) + 1
                continue
            fill = find_fill(o, h, l, c, i, direction, entry, cfg)
            sign = 1 if direction == "bullish" else -1
            rel = None
            if spy_up is not None:
                rel = ((row.chg20 - spy_chg[0].iloc[i]) + (row.chg60 - spy_chg[1].iloc[i])) / 2
            t = {"ticker": ticker, "setup": name, "direction": direction, "signal_date": dates[i].date(),
                 "close": round(c[i], 4), "entry": entry, "stop": stop,
                 "stop_distance_pct": round(abs(entry - stop) / entry * 100, 2),
                 "atr_pct": round(row.atr_pct, 3),
                 "rank_score": None if rel is None or not math.isfinite(rel)
                 else round(sign * rel + cfg.upside_weight * row.atr_pct, 3),
                 "market_up": None if spy_up is None or pd.isna(spy_up.iloc[i]) else bool(spy_up.iloc[i]),
                 "shares_ok": abs(entry - stop) / entry <= cfg.max_stop_distance_pct,
                 "opt_cost_est": option_cost_estimate(c[i], direction, row.vol20, cfg),
                 "filled": fill is not None}
            if fill is None:
                trades.append(t)
                continue
            t["fill_date"] = dates[fill.day].date()
            t["fill_price"] = round(fill.price, 4)
            end = fill.day
            ex = None
            if name in EXIT_LINE:
                col, side, days = EXIT_LINE[name]
                ex = (lines[col], side, days)
            if t["shares_ok"]:
                sh = simulate_shares(o, h, l, c, fill, direction, entry, stop, cfg, exit_line=ex)
                end = sh.pop("exit_day")
                sh["exit_date"] = dates[end].date()
                t.update(sh)
            opt = simulate_option(o, h, l, c, dates, fill, direction, row.vol20, cfg, stock_stop=stop,
                                  exit_line=ex)
            if opt:
                oe = opt.pop("opt_exit_day")
                opt["opt_exit_date"] = dates[oe].date()
                end = max(end, oe)
                t.update(opt)
            busy[name] = end
            trades.append(t)
    return trades, filtered


# ---------------------------------------------------------------- statistics

def stats(returns: pd.Series) -> dict:
    """Win rate, average win/loss and expectancy (average return per trade), in %."""
    r = returns.dropna()
    if r.empty:
        return {"trades": 0}
    wins, losses = r[r > 0], r[r <= 0]
    gross_loss = -losses.sum()
    return {
        "trades": int(len(r)),
        "win_rate": round(len(wins) / len(r) * 100, 1),
        "avg_win_pct": round(wins.mean(), 2) if len(wins) else 0.0,
        "avg_loss_pct": round(losses.mean(), 2) if len(losses) else 0.0,
        "expectancy_pct": round(r.mean(), 2),
        "profit_factor": round(wins.sum() / gross_loss, 2) if gross_loss > 0 else None,
    }


def closed_returns(trades: pd.DataFrame, return_col: str, exit_col: str) -> pd.DataFrame:
    """Filled trades that have finished (still-open trades at the end of the data are left out)."""
    if return_col not in trades:
        return trades.iloc[0:0].assign(**{return_col: []})
    done = trades[trades.filled & trades[return_col].notna()]
    return done[done[exit_col] != "open"]


def setup_table(trades: pd.DataFrame, return_col: str, exit_col: str) -> dict[str, dict]:
    closed = closed_returns(trades, return_col, exit_col)
    out = {name: stats(closed.loc[closed.setup == name, return_col]) for name in SETUP_NAMES}
    out["all"] = stats(closed[return_col])
    return out


# ---------------------------------------------------------------- the $100 account

@dataclass
class PortfolioResult:
    label: str
    start_equity: float
    final_equity: float = 0.0
    trades: int = 0
    wins: int = 0
    max_drawdown_pct: float = 0.0
    ai_cost: float = 0.0
    by_instrument: dict[str, int] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)

    @property
    def net_after_ai(self) -> float:
        return round(self.final_equity - self.ai_cost, 2)


def _ok(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def run_portfolio(trades: pd.DataFrame, cfg: BTConfig, mode: str = "hybrid",
                  regime_filter: bool = False, label: str = "") -> PortfolioResult:
    """One all-in position at a time. Each day while flat, the best-ranked signal
    (the scanner's ranking) is ordered; if its trigger never trades, the next
    day starts over. mode: "hybrid" (a call if one contract is affordable, else
    shares; bearish needs a put), "shares" (bullish shares only) or "options"."""
    res = PortfolioResult(label or mode, cfg.start_equity)
    equity = peak = cfg.start_equity
    if trades.empty:
        res.final_equity = equity
        return res
    t = trades.copy()
    t = t[t.rank_score.notna()]
    if regime_filter:
        t = t[t.market_up.notna() & ((t.direction == "bullish") == t.market_up.astype(bool))]
    if mode == "shares":
        t = t[t.direction == "bullish"]
    by_day = {d: g.sort_values("rank_score", ascending=False) for d, g in t.groupby("signal_date")}
    active_days = 0
    busy_until = None
    for day in sorted(by_day):
        if busy_until is not None and day <= busy_until:
            active_days += 1          # the PM reviews the open position
            continue
        busy_until = None
        active_days += 1
        for _, s in by_day[day].iterrows():
            # Choose using only what is known on the signal day (never whether it filled).
            instrument = None
            est = s.get("opt_cost_est")
            if mode in ("hybrid", "options") and _ok(est) and est <= equity * 0.995:
                instrument = "option"
            elif mode in ("hybrid", "shares") and s.direction == "bullish" and s.shares_ok:
                instrument = "shares"
            if instrument is None:
                continue          # the live scanner would not offer it: try the next-ranked
            if not s.filled:
                break             # the order was placed but never triggered
            if instrument == "option":
                contracts = int(equity * 0.995 // s.opt_cost) if _ok(s.get("opt_cost")) else 0
                if contracts < 1:
                    break         # the price moved before the fill and one contract no longer fits
                pnl = contracts * s.opt_cost * s.opt_return_pct / 100
                exit_date, reason = s.opt_exit_date, s.opt_exit_reason
            else:
                pnl = equity * s.return_pct / 100
                exit_date, reason = s.exit_date, s.exit_reason
            equity = max(0.0, equity + pnl)
            peak = max(peak, equity)
            res.max_drawdown_pct = min(res.max_drawdown_pct, (equity / peak - 1) * 100)
            res.trades += 1
            res.wins += pnl > 0
            res.by_instrument[instrument] = res.by_instrument.get(instrument, 0) + 1
            res.log.append({"signal_date": str(day), "ticker": s.ticker, "setup": s.setup,
                            "instrument": instrument, "exit_date": str(exit_date), "exit_reason": reason,
                            "pnl": round(pnl, 2), "equity": round(equity, 2)})
            busy_until = exit_date
            break
        if equity < 1:
            break
    res.final_equity = round(equity, 2)
    res.ai_cost = round(active_days * cfg.ai_cost_per_run, 2)
    return res


# ---------------------------------------------------------------- running it

def run_backtest(bars: dict[str, pd.DataFrame], tickers: list[str], cfg: BTConfig,
                 years: float = 4.0, log=print) -> tuple[pd.DataFrame, dict[str, int]]:
    spy = bars.get("SPY")
    last = max(df.index[-1] for df in bars.values())
    start = last - pd.Timedelta(days=int(years * 365))
    all_trades: list[dict] = []
    filtered: dict[str, int] = {}
    for n, t in enumerate(tickers, 1):
        if t not in bars or len(bars[t]) < 220:
            continue
        tr, fl = trades_for_ticker(t, bars[t], cfg, start=start, spy=spy)
        all_trades += tr
        for k, v in fl.items():
            filtered[k] = filtered.get(k, 0) + v
        if n % 100 == 0:
            log(f"  simulated {n}/{len(tickers)} tickers...")
    return pd.DataFrame(all_trades), filtered


def select_trades(trades: pd.DataFrame, setups=SETUP_NAMES, with_market: bool = False,
                  half: str | None = None) -> pd.DataFrame:
    """Keep only some setups, only trades with the market trend, and/or one half of the period."""
    if trades.empty:
        return trades
    t = trades[trades.setup.isin(setups)]
    if with_market:
        t = t[t.market_up.notna() & ((t.direction == "bullish") == t.market_up.astype(bool))]
    if half:
        dates = pd.to_datetime(trades.signal_date)
        mid = dates.min() + (dates.max() - dates.min()) / 2
        d = pd.to_datetime(t.signal_date)
        t = t[d < mid] if half == "first" else t[d >= mid]
    return t


def _fmt(row: dict) -> str:
    if not row.get("trades"):
        return "      0 trades"
    pf = row["profit_factor"]
    note = "  (too few to judge)" if row["trades"] < MIN_TRADES else ""
    return (f"{row['trades']:>6} trades  win {row['win_rate']:>5.1f}%  avg win {row['avg_win_pct']:>+7.2f}%  "
            f"avg loss {row['avg_loss_pct']:>+7.2f}%  EXPECTANCY {row['expectancy_pct']:>+6.2f}%/trade  "
            f"PF {pf if pf is not None else '-'}{note}")


def report(trades: pd.DataFrame, filtered: dict[str, int], cfg: BTConfig, portfolios: list[PortfolioResult],
           years: float) -> str:
    lines = []
    line = "=" * 100
    signals = len(trades)
    filled = int(trades.filled.sum()) if signals else 0
    lines += [line, f"BACKTEST: last {years:g} years, {trades.ticker.nunique() if signals else 0} tickers with signals",
              line,
              f"Signals that passed the scanner filters: {signals}; the trigger traded (order filled): {filled}",
              "Filtered out: " + (", ".join(f"{k} {v}" for k, v in sorted(filtered.items())) or "none"),
              f"Rules: target {cfg.target_r}R, max hold {cfg.max_hold_days} days, stop at most "
              f"{cfg.max_stop_distance_pct:.0%} away (shares); options ~{cfg.option_delta} delta, "
              f"{cfg.option_dte} days, spread {cfg.option_spread_pct:.0%}, stop "
              f"{'at the stock stop' if cfg.option_stop_on_stock else f'-{cfg.option_stop_pct:.0%}'}, "
              f"take profit +{cfg.option_take_profit_pct:.0%}",
              "momentum and dip_buy have no target: momentum exits on a close below its 20-day average "
              "(max 40 days), dip_buy on a close above its 5-day average (max 10 days)"]
    if not filled:
        return "\n".join(lines + ["No trades filled."])

    for title, col, ex in (("SHARES (return per trade, after slippage)", "return_pct", "exit_reason"),
                           ("OPTIONS (estimated; return on the premium paid, spread included)",
                            "opt_return_pct", "opt_exit_reason")):
        lines += ["", f"--- {title}"]
        for name, row in setup_table(trades, col, ex).items():
            lines.append(f"  {name:<11}{_fmt(row)}")

    done = trades[trades.filled]
    known = trades[trades.market_up.notna()]
    if len(known):
        aligned = (known.direction == "bullish") == known.market_up.astype(bool)
        for title, col, ex in (("shares", "return_pct", "exit_reason"), ("options", "opt_return_pct", "opt_exit_reason")):
            lines += ["", f"--- WITH vs AGAINST THE MARKET (SPY above/below its 50-day average), {title}"]
            for label, part in (("with", known[aligned]), ("against", known[~aligned])):
                lines.append(f"  {label:<11}{_fmt(stats(closed_returns(part, col, ex)[col]))}")

    if "exit_reason" in done:
        reasons = done.exit_reason.dropna().str.replace(r" \(gap\)", "", regex=True).value_counts()
        lines += ["", "--- How share trades ended: " + ", ".join(f"{k} {v}" for k, v in reasons.items())]
        mfe = done.mfe_r.dropna()
        if len(mfe):
            lines.append(f"    Best open profit during the trade (median): {mfe.median():.2f}R; "
                         f"reached 1R: {(mfe >= 1).mean() * 100:.0f}%, 2R: {(mfe >= 2).mean() * 100:.0f}%, "
                         f"3R: {(mfe >= 3).mean() * 100:.0f}% of trades")
    if "opt_exit_reason" in done:
        reasons = done.opt_exit_reason.dropna().str.replace(r" \(gap\)", "", regex=True).value_counts()
        lines.append("--- How option trades ended: " + ", ".join(f"{k} {v}" for k, v in reasons.items()))
        move = done.get("opt_stop_stock_move_pct", pd.Series(dtype=float)).dropna()
        if len(move) and not cfg.option_stop_on_stock:
            lines.append(f"    The -{cfg.option_stop_pct:.0%} option stop hits after a stock move of about "
                         f"{move.median():.1f}% (median); these stocks' average daily range is "
                         f"{done.atr_pct.median():.1f}%. A stop inside one day's normal range is mostly noise.")

    lines += ["", f"--- THE ${cfg.start_equity:.0f} ACCOUNT (one all-in position at a time, top-ranked signal)"]
    for p in portfolios:
        inst = ", ".join(f"{k} {v}" for k, v in p.by_instrument.items()) or "none"
        win = f"{p.wins / p.trades * 100:.0f}%" if p.trades else "-"
        lines.append(f"  {p.label:<34} ${p.start_equity:.0f} -> ${p.final_equity:>9.2f}  trades {p.trades:>3} "
                     f"({inst}), win {win}, worst drawdown {p.max_drawdown_pct:.0f}%, "
                     f"AI cost ~${p.ai_cost:.0f} -> net ${p.net_after_ai:.2f}")
    lines += ["", "Read this with care: option prices are estimated, the AI analysts are not simulated,",
              "and today's stock list leaves out companies that failed (results lean optimistic).",
              f"A setup counts as an edge only with {MIN_TRADES}+ trades and a positive SHARES expectancy in both "
              "halves of the period (--half first, --half second)."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="comma-separated tickers (default: the live universe)")
    ap.add_argument("--years", type=float, default=4.0, help="years of signals to test (default 4)")
    ap.add_argument("--target-r", type=float, help="target in multiples of the risk (default 2)")
    ap.add_argument("--hold", type=int, help="max trading days held (default 15)")
    ap.add_argument("--spread", type=float, help="option bid/ask spread as a fraction (default 0.08)")
    ap.add_argument("--equity", type=float, help="starting account (default 100)")
    ap.add_argument("--option-stop", help="option stop: a fraction like 0.15 (live default) or 0.5, "
                                          "or 'stock' to exit when the stock hits the setup's stop")
    ap.add_argument("--setups", help=f"only these setups, comma-separated ({', '.join(SETUP_NAMES)}); "
                                     "default: what the live desk trades")
    ap.add_argument("--with-market", action="store_true",
                    help="only trades with the market: bullish while SPY is above its 50-day average, "
                         "bearish while below (on by default when the live desk's market filter is on)")
    ap.add_argument("--any-market", action="store_true", help="turn the market filter off for this run")
    ap.add_argument("--half", choices=("first", "second"),
                    help="only the first or second half of the period: tune on one, confirm on the other")
    args = ap.parse_args()
    from .config import load_settings as _load
    live = _load()
    live_setups = [n for n in SETUP_NAMES if DIRECTION[n] in live.directions]
    setups = [x.strip() for x in args.setups.split(",")] if args.setups else live_setups
    with_market = (args.with_market or live.market_filter) and not args.any_market
    unknown = set(setups) - set(SETUP_NAMES)
    if unknown:
        raise SystemExit(f"Unknown setup(s): {', '.join(sorted(unknown))}. Choose from {', '.join(SETUP_NAMES)}.")

    from . import market_data
    from .config import load_settings
    from .universe import load_universe

    settings = load_settings()
    cfg = BTConfig(min_price=settings.scan_min_price, min_dollar_volume=settings.scan_min_dollar_volume,
                   max_stop_distance_pct=settings.risk.max_stop_distance_pct,
                   option_delta=settings.options.target_delta,
                   option_stop_pct=settings.options.stop_loss_pct,
                   option_stop_on_stock=settings.options.stop_on_stock,
                   option_take_profit_pct=settings.options.take_profit_pct,
                   exit_days_before_expiry=settings.options.exit_days_before_expiry,
                   risk_free_rate=settings.options.risk_free_rate,
                   start_equity=settings.paper_equity)
    overrides = {"target_r": args.target_r, "max_hold_days": args.hold, "option_spread_pct": args.spread,
                 "start_equity": args.equity}
    if args.option_stop == "stock":
        overrides["option_stop_on_stock"] = True
    elif args.option_stop:
        overrides["option_stop_pct"] = float(args.option_stop)
        overrides["option_stop_on_stock"] = False
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else load_universe(settings)
    period = f"{min(10, math.ceil(args.years) + 2)}y"     # +1 year of warm-up for the 200/252-day rules
    print(f"Backtesting {len(tickers)} tickers over {args.years:g} years. Downloading {period} of daily bars "
          f"(first time: several minutes; cached in data/cache/bars_long afterwards)...")
    bars = market_data.download_bars(sorted(set(tickers) | {"SPY"}), period=period, priority=["SPY"],
                                     cache_dir=BACKTEST_CACHE)
    if "SPY" not in bars:
        raise SystemExit("No SPY data: Yahoo is throttling. Wait 15-30 minutes and run again.")
    print(f"Price data for {sum(1 for t in tickers if t in bars)} of {len(tickers)} tickers. Simulating...")

    trades, filtered = run_backtest(bars, tickers, cfg, years=args.years)
    trades = select_trades(trades, setups, with_market, args.half)
    notes = [f"setups: {', '.join(setups)}"]
    notes += ["only trades with the market (SPY vs its 50-day average)"] if with_market else ["any market"]
    notes += [f"{args.half} half of the period only"] if args.half else []
    if notes:
        print("Filters: " + "; ".join(notes))
    portfolios = []
    if not trades.empty:
        calls = "on" if live.calls else "off in the live desk"
        for mode, label in (("shares", "all setups, shares"),
                            ("hybrid", f"all setups, call if affordable ({calls})")):
            portfolios.append(run_portfolio(trades, cfg, mode, False, label))
        for name in setups:                      # one account per setup, to compare them
            part = trades[trades.setup == name]
            if DIRECTION[name] == "bullish" and not part.empty:
                portfolios.append(run_portfolio(part, cfg, "shares", False, f"{name} only, shares"))
    text = report(trades, filtered, cfg, portfolios, args.years)
    print(text)

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    trades.to_csv(BACKTEST_DIR / f"trades_{stamp}.csv", index=False)
    (BACKTEST_DIR / f"summary_{stamp}.json").write_text(json.dumps({
        "config": asdict(cfg), "years": args.years, "report": text,
        "portfolios": [{**{k: v for k, v in asdict(p).items() if k != "log"}, "trade_log": p.log}
                       for p in portfolios]}, indent=2, default=str))
    print(f"\nSaved: data/backtests/trades_{stamp}.csv (every trade; opens in Excel/Numbers) "
          f"and summary_{stamp}.json")


if __name__ == "__main__":
    main()
