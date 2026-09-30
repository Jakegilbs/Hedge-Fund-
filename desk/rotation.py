"""Momentum rotation backtester: hold the recent winners, rebalance monthly. Free: code only.

    python -m desk.rotation                          # the grid: many versions side by side
    python -m desk.rotation --lookback 126 --top 10 --score smooth --earnings   # one version, in detail
    python -m desk.rotation --rebalance 2            # every two months

Each rebalance day (the first trading day of the month) it ranks the stocks by
their price gain over the lookback (optionally skipping the most recent month),
keeps the top N with a positive gain, and holds them until the next rebalance.
Orders are placed after the ranking, at the next day's open. With the market
filter on, it holds cash whenever SPY closes below its 200-day average.

Two benchmarks, both on the same dates:
- SPY: what $100 in the index would have become.
- Equal weight: every stock in the same universe, equally weighted, rebalanced
  monthly. Momentum only counts as an edge if it beats this.

The universe on each date is the S&P 500 as it was THEN (desk/membership.py),
so stocks are not ranked before they joined the index. Companies that were
later acquired or went bankrupt usually have no price data left and are
missing; the coverage line shows how many. `--membership today` repeats the
old, biased method (today's list for every date) for comparison.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR

BACKTEST_DIR = DATA_DIR / "backtests"
TRADING_DAYS = 252


@dataclass(frozen=True)
class RotationConfig:
    lookback: int = 60                # trading days of price gain used to rank
    skip: int = 0                     # skip the most recent N days (21 = the classic "12-1 month")
    top: int = 10                     # hold this many stocks
    weighting: str = "equal"          # "equal" or "invvol" (steadier stocks get more)
    market_filter: bool = True        # cash while SPY is below its 200-day average
    positive_only: bool = True        # only stocks that actually rose over the lookback
    score: str = "gain"               # "gain": biggest rise; "smooth": rise relative to its volatility,
                                      #   skipping stocks whose rise came mostly from one day
    earnings: bool = False            # only stocks in a positive earnings drift (desk/earnings.py)
    rebalance_months: int = 1
    cost_pct: float = 0.001           # per dollar traded (slippage; Robinhood charges no commission)
    min_price: float = 5.0
    min_dollar_volume: float = 20_000_000
    start_equity: float = 100.0

    @property
    def label(self) -> str:
        look = f"{self.lookback}d" + (f"-skip{self.skip}" if self.skip else "")
        rank = self.score + ("+earn" if self.earnings else "")
        return (f"{look:<12} {rank:<12} top {self.top:<3} {self.weighting:<7} "
                f"{'filter' if self.market_filter else 'no filter'}")


@dataclass
class Panel:
    """Prices for every stock on the same dates (SPY's trading days)."""
    close: pd.DataFrame
    open: pd.DataFrame
    dollar_volume: pd.DataFrame
    spy: pd.Series
    member: pd.DataFrame | None = None     # True where a stock was in the index that day
    earn_ok: pd.DataFrame | None = None    # True where a stock is in a positive earnings drift

    def eligible(self, s: int) -> pd.Series:
        """Stocks that may be ranked on day s: index members at the time (if known)."""
        if self.member is None:
            return pd.Series(True, index=self.close.columns)
        return self.member.iloc[s].reindex(self.close.columns, fill_value=False).astype(bool)

    @classmethod
    def from_bars(cls, bars: dict[str, pd.DataFrame], tickers: list[str]) -> "Panel":
        idx = bars["SPY"].index
        names = [t for t in tickers if t in bars and t != "SPY"]
        close = pd.DataFrame({t: bars[t]["Close"] for t in names}).reindex(idx)
        open_ = pd.DataFrame({t: bars[t]["Open"] for t in names}).reindex(idx)
        vol = pd.DataFrame({t: bars[t]["Volume"] for t in names}).reindex(idx)
        close = close.ffill(limit=3)          # bridge a few missing days, never a delisting
        return cls(close, open_, (close * vol).rolling(20, min_periods=10).mean(), bars["SPY"]["Close"])


def rebalance_days(index: pd.DatetimeIndex, start: int, months: int = 1) -> list[int]:
    """Positions of the first trading day of every `months`-th month, from `start` on."""
    days, last = [], None
    for i in range(max(start, 1), len(index)):
        m = (index[i].year, index[i].month)
        if m != (index[i - 1].year, index[i - 1].month):
            if last is None or (m[0] - last[0]) * 12 + (m[1] - last[1]) >= months:
                days.append(i)
                last = m
    return days


def target_weights(panel: Panel, s: int, cfg: RotationConfig) -> pd.Series:
    """Weights to hold from the next day, using only data up to day s (the signal day)."""
    c = panel.close
    if cfg.market_filter:
        spy = panel.spy
        sma200 = spy.rolling(200).mean()
        if not (spy.iloc[s] > sma200.iloc[s]):
            return pd.Series(dtype=float)
    end, begin = s - cfg.skip, s - cfg.skip - cfg.lookback
    if begin < 0:
        return pd.Series(dtype=float)
    gain = c.iloc[end] / c.iloc[begin] - 1
    ok = ((c.iloc[s] >= cfg.min_price) & (panel.dollar_volume.iloc[s] >= cfg.min_dollar_volume)
          & gain.notna() & panel.eligible(s))
    if cfg.positive_only:
        ok &= gain > 0
    if cfg.earnings:
        if panel.earn_ok is None:
            return pd.Series(dtype=float)
        ok &= panel.earn_ok.iloc[s].reindex(c.columns, fill_value=False).astype(bool)
    rank = gain
    if cfg.score == "smooth":
        rank, smooth_ok = smooth_scores(c, begin, end)
        ok &= smooth_ok
    picks = rank[ok].sort_values(ascending=False).head(cfg.top)
    if picks.empty:
        return pd.Series(dtype=float)
    if cfg.weighting == "invvol":
        vol = c[picks.index].iloc[max(0, s - 60):s + 1].pct_change().std()
        inv = (1 / vol.replace(0, np.nan)).fillna(0)
        w = inv / inv.sum() if inv.sum() > 0 else pd.Series(1.0, index=picks.index)
    else:
        w = pd.Series(1.0, index=picks.index)
    # Fewer than `top` qualifying stocks: each keeps its 1/top share and the rest stays in cash.
    return w / w.sum() * (len(picks) / cfg.top)


JUMP_SHARE = 0.5   # a rise with more than half of it from one day is a jump, not smooth growth


def smooth_scores(c: pd.DataFrame, begin: int, end: int) -> tuple[pd.Series, pd.Series]:
    """Rise relative to its own volatility over the lookback: log gain / (daily volatility x
    sqrt(days)). High = a steady climb. Also returns which stocks did NOT get most of
    their rise from a single day."""
    window = np.log(c.iloc[begin:end + 1])
    daily = window.diff().iloc[1:]
    total = window.iloc[-1] - window.iloc[0]
    vol = daily.std() * math.sqrt(len(daily))
    score = total / vol.replace(0, np.nan)
    no_jump = ~(daily.max() > JUMP_SHARE * total.where(total > 0))
    return score, no_jump.fillna(True)


def equal_weight_all(panel: Panel, s: int, cfg: RotationConfig) -> pd.Series:
    """Benchmark: every liquid stock in the universe, equal weight."""
    c = panel.close
    ok = ((c.iloc[s] >= cfg.min_price) & (panel.dollar_volume.iloc[s] >= cfg.min_dollar_volume)
          & c.iloc[s].notna() & panel.eligible(s))
    names = ok[ok].index
    return pd.Series(1.0 / len(names), index=names) if len(names) else pd.Series(dtype=float)


@dataclass
class RotationResult:
    label: str
    equity: pd.Series
    rebalances: int = 0
    turnover: float = 0.0             # average fraction of the account traded per rebalance
    months_in_cash: int = 0
    last_weights: pd.Series | None = None
    log: list = None                  # (rebalance day index, weights) for every rebalance


def simulate(panel: Panel, start: int, cfg: RotationConfig, weigher=target_weights,
             label: str | None = None) -> RotationResult:
    """Daily mark-to-market. On each rebalance day: mark holdings to the open, trade to the
    targets (paying cost_pct on every dollar traded), then mark to the close."""
    C, O = panel.close.to_numpy(float), panel.open.to_numpy(float)
    cols = {t: k for k, t in enumerate(panel.close.columns)}
    days = set(rebalance_days(panel.close.index, start, cfg.rebalance_months))
    hold = np.zeros(C.shape[1])
    cash = cfg.start_equity
    eq = np.full(len(C), np.nan)
    eq[start - 1] = cash
    res = RotationResult(label or cfg.label, pd.Series(dtype=float), log=[])
    turnovers, w = [], pd.Series(dtype=float)

    def ratio(a, b):
        r = a / b
        return np.where(np.isfinite(r) & (r > 0), r, 1.0)

    for d in range(start, len(C)):
        if d in days:
            hold *= ratio(O[d], C[d - 1])
            total = hold.sum() + cash
            w = weigher(panel, d - 1, cfg)
            target = np.zeros_like(hold)
            for t, x in w.items():
                if np.isfinite(O[d, cols[t]]):
                    target[cols[t]] = x * total
            traded = np.abs(target - hold).sum()
            after = total - traded * cfg.cost_pct          # trading costs come out of the account
            hold = target * (after / total) if total > 0 else target
            cash = after - hold.sum()
            turnovers.append(traded / total if total > 0 else 0.0)
            res.rebalances += 1
            res.months_in_cash += w.empty
            res.log.append((d, w))
            hold *= ratio(C[d], O[d])
        else:
            hold *= ratio(C[d], C[d - 1])
        eq[d] = hold.sum() + cash
    res.equity = pd.Series(eq, index=panel.close.index).iloc[start - 1:]
    res.turnover = float(np.mean(turnovers)) if turnovers else 0.0
    res.last_weights = w
    return res


def spy_buy_and_hold(panel: Panel, start: int, equity: float = 100.0) -> RotationResult:
    s = panel.spy.iloc[start - 1:]
    return RotationResult("SPY buy and hold", s / s.iloc[0] * equity)


def metrics(equity: pd.Series) -> dict:
    e = equity.dropna()
    if len(e) < 2:
        return {}
    years = (e.index[-1] - e.index[0]).days / 365.25
    daily = e.pct_change().dropna()
    dd = (e / e.cummax() - 1).min()
    vol = daily.std() * math.sqrt(TRADING_DAYS)
    monthly = e.resample("ME").last().pct_change().dropna()
    return {
        "final": round(float(e.iloc[-1] / e.iloc[0] * 100), 2),
        "cagr_pct": round(((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1) * 100, 1) if years > 0 else None,
        "max_drawdown_pct": round(float(dd) * 100, 1),
        "vol_pct": round(vol * 100, 1),
        "sharpe": round(daily.mean() / daily.std() * math.sqrt(TRADING_DAYS), 2) if daily.std() > 0 else None,
        "worst_month_pct": round(float(monthly.min()) * 100, 1) if len(monthly) else None,
    }


def halves(equity: pd.Series) -> tuple[pd.Series, pd.Series]:
    e = equity.dropna()
    mid = e.index[0] + (e.index[-1] - e.index[0]) / 2
    return e[e.index <= mid], e[e.index >= e.index[e.index <= mid][-1]]


def default_grid(base: RotationConfig) -> list[RotationConfig]:
    """Decided in advance (to avoid fishing for a lucky version): 3 lookbacks x 3 sizes for
    each of 4 ranking ideas. No market filter: it hurt every version over 10 years."""
    out = []
    for score, earnings in (("gain", False), ("smooth", False), ("gain", True), ("smooth", True)):
        for lookback, skip in ((60, 0), (126, 0), (252, 21)):
            for top in (5, 10, 20):
                out.append(replace(base, lookback=lookback, skip=skip, top=top, score=score,
                                   earnings=earnings, market_filter=False))
    return out


def run_grid(panel: Panel, start: int, configs: list[RotationConfig], log=print) -> tuple[list, list]:
    base = configs[0]
    benches = [spy_buy_and_hold(panel, start, base.start_equity),
               simulate(panel, start, replace(base, market_filter=False), equal_weight_all,
                        label="Equal weight, whole universe")]
    results = []
    for n, cfg in enumerate(configs, 1):
        results.append((cfg, simulate(panel, start, cfg)))
        if n % 20 == 0:
            log(f"  tested {n}/{len(configs)} versions...")
    return benches, results


def report(benches: list[RotationResult], results: list[tuple[RotationConfig, RotationResult]],
           ai_cost_per_rebalance: float = 0.20) -> str:
    line = "=" * 130
    b_first = [metrics(halves(b.equity)[0]) for b in benches]
    b_second = [metrics(halves(b.equity)[1]) for b in benches]
    eq_first, eq_second = b_first[1].get("cagr_pct"), b_second[1].get("cagr_pct")
    e0 = benches[0].equity.dropna()
    lines = [line, f"MOMENTUM ROTATION: {e0.index[0].date()} to {e0.index[-1].date()}, $100 start", line,
             f"{'version':<52}{'$100 ->':>9}{'CAGR':>7}{'max drop':>10}{'Sharpe':>8}{'worst mo':>10}"
             f"{'1st half':>10}{'2nd half':>10}  edge?"]

    def row(label: str, e: pd.Series, mark: str = "") -> str:
        m, (f, s) = metrics(e), (metrics(x) for x in halves(e))
        return (f"{label:<52}{m['final']:>9.2f}{m['cagr_pct']:>6.1f}%{m['max_drawdown_pct']:>9.1f}%"
                f"{(m['sharpe'] or 0):>8.2f}{(m['worst_month_pct'] or 0):>9.1f}%"
                f"{f.get('cagr_pct', 0):>9.1f}%{s.get('cagr_pct', 0):>9.1f}%  {mark}")

    lines.append("--- benchmarks")
    for b in benches:
        lines.append(row(b.label, b.equity))
    lines.append("--- momentum versions (best first). edge? = beat the equal-weight benchmark in BOTH halves")
    scored = sorted(results, key=lambda r: -(metrics(r[1].equity).get("final") or 0))
    wins = 0
    for cfg, r in scored:
        f, s = (metrics(x).get("cagr_pct") for x in halves(r.equity))
        edge = f is not None and s is not None and f > eq_first and s > eq_second
        wins += edge
        lines.append(row(cfg.label, r.equity, "YES" if edge else ""))
    n_reb = results[0][1].rebalances if results else 0
    families: dict[str, list[bool]] = {}
    for cfg, r in results:
        f, s = (metrics(x).get("cagr_pct") for x in halves(r.equity))
        name = cfg.score + (" + earnings drift" if cfg.earnings else "")
        families.setdefault(name, []).append(f is not None and s is not None and f > eq_first and s > eq_second)
    lines += ["", "--- VERDICT per idea (decided in advance: an idea counts only if at least 2 of every 3 "
                  "versions beat equal weight in both halves)"]
    for name, flags in families.items():
        n, k = len(flags), sum(flags)
        verdict = "EDGE" if k * 3 >= 2 * n else "no edge"
        lines.append(f"  {name:<28} {k} of {n} versions  -> {verdict}")
    lines += ["",
              f"{wins} of {len(results)} versions beat equal weight in both halves. With this many versions, "
              "a few will look good by luck:",
              "trust a pattern (e.g. every top-5/top-10 version of one lookback winning), not the single best row.",
              f"Rebalances: {n_reb}. If the AI team reviewed each one (~${ai_cost_per_rebalance:.2f}), that is "
              f"~${n_reb * ai_cost_per_rebalance:.0f} over the period, about the same for every version.",
              "Judge momentum against equal weight (same stocks, same dates), not only against SPY."]
    return "\n".join(lines)


def holdings_table(panel: Panel, r: RotationResult) -> pd.DataFrame:
    """Every position of every rebalance: weight, the stock's return until the next
    rebalance (open to open), and its contribution (weight x return)."""
    O, C = panel.open, panel.close
    rows = []
    for k, (d, w) in enumerate(r.log or []):
        end = r.log[k + 1][0] if k + 1 < len(r.log) else None
        for t, x in w.items():
            buy = O[t].iloc[d]
            sell = O[t].iloc[end] if end is not None else np.nan
            if not np.isfinite(sell):              # last price available before the next rebalance
                held = C[t].iloc[d:end].dropna()
                sell = held.iloc[-1] if len(held) else buy
            ret = sell / buy - 1 if np.isfinite(buy) and buy > 0 else 0.0
            rows.append({"date": O.index[d].date(), "ticker": t, "weight": round(float(x), 4),
                         "return_pct": round(float(ret) * 100, 2),
                         "contribution_pct": round(float(x * ret) * 100, 3)})
    return pd.DataFrame(rows)


def suspicious_moves(panel: Panel, tickers, limit: float = 0.5) -> list[str]:
    """Held stocks with a one-day move above 50%: often a bad split or price in the data."""
    out = []
    for t in sorted(set(tickers)):
        ch = panel.close[t].pct_change(fill_method=None).abs()
        big = ch[ch > limit]
        for day, v in big.items():
            out.append(f"{t} {day.date()} {v * 100:+.0f}% in one day")
    return out


def detail(cfg: RotationConfig, r: RotationResult, spy: RotationResult, ew: RotationResult,
           panel: Panel, biased: tuple[RotationResult, RotationResult] | None = None) -> str:
    """One version: results by year, turnover, and what it would hold now."""
    lines = ["", f"--- {cfg.label}: year by year (return %)",
             f"{'year':<8}{'momentum':>10}{'SPY':>8}{'equal wt':>10}"]
    for y in sorted(set(r.equity.dropna().index.year)):
        def yr(e):
            x = e.dropna()
            x = x[x.index.year == y]
            prev = e.dropna()[e.dropna().index.year < y]
            base = prev.iloc[-1] if len(prev) else x.iloc[0]
            return (x.iloc[-1] / base - 1) * 100 if len(x) else float("nan")
        lines.append(f"{y:<8}{yr(r.equity):>9.1f}%{yr(spy.equity):>7.1f}%{yr(ew.equity):>9.1f}%")
    lines.append(f"Average share of the account traded per rebalance: {r.turnover * 100:.0f}%; "
                 f"rebalances in cash (market filter): {r.months_in_cash} of {r.rebalances}")
    if biased:
        m, e = (metrics(x.equity).get("final") for x in biased)
        lines.append(f"Same version on TODAY's S&P 500 list for every date (the biased way): $100 -> ${m:,.2f} "
                     f"(equal weight ${e:,.2f}). The gap to the numbers above is the survivorship bias.")
    table = holdings_table(panel, r)
    if not table.empty:
        by = table.groupby("ticker").agg(months=("date", "count"), contribution=("contribution_pct", "sum"))
        total = by.contribution.sum()
        best = by.sort_values("contribution", ascending=False).head(10)
        lines += ["", f"--- Where the gains came from ({len(by)} different stocks held; sum of monthly "
                      f"contributions {total:+.0f} points)",
                  f"{'stock':<8}{'months held':>12}{'contribution':>14}"]
        for t, row_ in best.iterrows():
            lines.append(f"{t:<8}{int(row_.months):>12}{row_.contribution:>+13.1f}%")
        top3 = best.contribution.head(3).sum()
        lines.append(f"Top 3 stocks: {top3:+.0f} of {total:+.0f} points"
                     + (" -> the result depends on a few stocks" if total > 0 and top3 > 0.5 * total else ""))
        worst = by.sort_values("contribution").head(3)
        lines.append("Worst: " + ", ".join(f"{t} {v:+.1f}%" for t, v in worst.contribution.items()))
        flags = suspicious_moves(panel, by.index)
        lines.append("Data check (one-day moves over 50% in held stocks): "
                     + ("; ".join(flags[:10]) + (" ..." if len(flags) > 10 else "") if flags else "none found"))
        lines += ["", "--- Last 6 rebalances"]
        for day, g in list(table.groupby("date"))[-6:]:
            lines.append(f"{day}: " + ", ".join(f"{t} {x:+.0f}%" for t, x in zip(g.ticker, g.return_pct)))
    now = target_weights(panel, len(panel.close) - 1, cfg)
    if now.empty:
        lines.append("Holdings today: CASH (market filter, or no stock passes the rules)")
    else:
        c = panel.close.iloc[-1]
        gain = c / panel.close.iloc[-1 - cfg.skip - cfg.lookback] - 1
        lines.append(f"If it rebalanced today it would hold ({cfg.start_equity:.0f} dollars):")
        for t, w in now.items():
            lines.append(f"  {t:<6} {w * 100:5.1f}%  ${w * cfg.start_equity:6.2f}   "
                         f"{cfg.lookback}-day gain {gain[t] * 100:+.1f}%")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=float, default=10.0, help="years to test (default 10)")
    ap.add_argument("--lookback", type=int, help="test one version: lookback in trading days (e.g. 60)")
    ap.add_argument("--skip", type=int, default=0, help="skip the most recent N days (e.g. 21)")
    ap.add_argument("--top", type=int, default=10, help="stocks held (default 10)")
    ap.add_argument("--weighting", choices=("equal", "invvol"), default="equal")
    ap.add_argument("--filter", action="store_true", help="hold cash while SPY is below its 200-day average")
    ap.add_argument("--score", choices=("gain", "smooth"), default="gain",
                    help="gain: biggest rise; smooth: rise relative to volatility, no one-day jumps")
    ap.add_argument("--earnings", action="store_true", help="only stocks in a positive earnings drift")
    ap.add_argument("--rebalance", type=int, default=1, help="rebalance every N months (default 1)")
    ap.add_argument("--membership", choices=("history", "today"), default="history",
                    help="history (default): only stocks in the S&P 500 at the time; "
                         "today: today's list for every date (biased, for comparison)")
    args = ap.parse_args()

    from . import market_data
    from .backtest import BACKTEST_CACHE
    from .config import load_settings
    from .membership import load_membership
    from .universe import sp500_symbols

    settings = load_settings()
    today_list = sorted(set(sp500_symbols()) - settings.etfs)
    hist = load_membership() if args.membership == "history" else None
    if args.membership == "history" and hist is None:
        raise SystemExit("Could not load the historical S&P 500 list. Check the internet connection, or run "
                         "with --membership today (biased).")
    end = pd.Timestamp.today().normalize()
    first_needed = end - pd.Timedelta(days=int(args.years * 365.25) + 420)   # + warm-up for the lookbacks
    tickers = sorted((hist.ever_between(first_needed, end) if hist else set()) | set(today_list))
    tickers = [t for t in tickers if t not in settings.etfs]
    period = f"{math.ceil(args.years) + 2}y"
    print(f"Momentum rotation over {args.years:g} years: {len(tickers)} stocks were in the S&P 500 at some point "
          f"(downloading {period} of prices; the first run takes several minutes, then it is cached)...")
    bars = market_data.download_bars(sorted(set(tickers) | {"SPY"}), period=period, priority=["SPY"],
                                     cache_dir=BACKTEST_CACHE)
    if "SPY" not in bars:
        raise SystemExit("No SPY data: Yahoo is throttling. Wait 15-30 minutes and run again.")
    panel = Panel.from_bars(bars, tickers)
    first = panel.close.index[-1] - pd.Timedelta(days=int(args.years * 365.25))
    start = max(int(panel.close.index.searchsorted(first)), 273 + 1)   # room for a 252+21 day lookback

    today_mask = pd.DataFrame(True, index=panel.close.index,
                              columns=[t for t in today_list if t in panel.close.columns])
    panel_today = replace(panel, member=today_mask)
    if hist:
        panel = replace(panel, member=hist.mask(panel.close.index, panel.close.columns))
        print("Coverage (index members with price data):")
        for y in sorted(set(panel.close.index[start:].year)):
            day = panel.close.index[panel.close.index.year == y][0]
            members = hist.on(day)
            have = sum(1 for t in members if t in panel.close.columns and pd.notna(panel.close[t].loc[day]))
            print(f"  {day.date()}: {have} of {len(members)} members ({have / max(1, len(members)) * 100:.0f}%)"
                  + ("" if have / max(1, len(members)) > 0.9 else "  <- missing ones were mostly acquired or delisted"))
    else:
        panel = panel_today
    needs_earnings = not args.lookback or args.earnings
    if needs_earnings:
        from .earnings import drift_flags, load_earnings
        earn = load_earnings(list(panel.close.columns))
        flags = drift_flags(earn, panel.close)
        panel = replace(panel, earn_ok=flags)
        panel_today = replace(panel_today, earn_ok=flags)
        have = sum(1 for t in panel.close.columns if len(earn.get(t, [])))
        oldest = min((earn[t].date.min() for t in panel.close.columns if len(earn.get(t, []))), default=None)
        print(f"Earnings history for {have} of {panel.close.shape[1]} stocks"
              + (f", back to {pd.Timestamp(oldest).date()}" if oldest is not None else "")
              + ". Stocks without it are never picked by the earnings versions.")
    print(f"Testing from {panel.close.index[start].date()}...")

    base = RotationConfig(rebalance_months=args.rebalance)
    if args.lookback:
        configs = [replace(base, lookback=args.lookback, skip=args.skip, top=args.top,
                           weighting=args.weighting, market_filter=args.filter, score=args.score,
                           earnings=args.earnings)]
    else:
        configs = default_grid(base)
    benches, results = run_grid(panel, start, configs)
    text = report(benches, results)
    cfg, r = (results[0] if args.lookback else
              next((c, x) for c, x in results                  # the idea being tested
                   if c.lookback == 126 and c.top == 10 and c.score == "smooth" and c.earnings))
    biased = None
    if hist:
        biased = (simulate(panel_today, start, cfg),
                  simulate(panel_today, start, replace(cfg, market_filter=False), equal_weight_all))
    text += detail(cfg, r, benches[0], benches[1], panel, biased)
    print(text)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    path = BACKTEST_DIR / f"rotation_{stamp}.txt"
    path.write_text(text)
    holdings_table(panel, r).to_csv(BACKTEST_DIR / f"rotation_holdings_{stamp}.csv", index=False)
    print(f"\nSaved: {path.relative_to(DATA_DIR.parent)} and rotation_holdings_{stamp}.csv "
          f"(every monthly position of: {' '.join(cfg.label.split())})")


if __name__ == "__main__":
    main()
