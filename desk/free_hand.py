"""Free-hand fund: the AI decides everything, code only keeps the books and the score.

    python -m desk.free_hand              # one decision: research, then the portfolio (~$0.20 of AI)
    python -m desk.free_hand --dry-run    # decide but do not update the book
    python -m desk.free_hand --status     # the scoreboard only, no AI cost

Each run:
1. The agent sees its portfolio, its results against SPY, its own past decisions,
   a market snapshot and this week's biggest movers, and asks for up to 15 stocks
   or ETFs to research (any US-listed ones).
2. Code fetches fresh data on those and on everything held: price trend, news,
   the next earnings date and the last earnings surprises.
3. The agent decides the whole portfolio: which stocks, what share of the account
   each, how much cash, with a thesis and an exit plan for every position.

Code does not judge the decision. It only makes it executable: tickers must have
real prices, weights cannot add up to more than 100%, and orders under $1 are
skipped. The book (data/free_hand/book.json) assumes each order fills at the
price shown; place the same orders in Robinhood and edit the book if a fill
differs. Guardrails that stay: long stocks and ETFs only, no margin, no options,
no short selling, so the account can never lose more than it holds.

This cannot be backtested honestly (the model already knows how past years
turned out), so the scoreboard against SPY, week by week, IS the test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from .config import DATA_DIR, Settings

BOOK_FILE = DATA_DIR / "free_hand" / "book.json"
MAX_RESEARCH = 15
MIN_ORDER_USD = 1.00          # Robinhood's minimum fractional order


# ---------------------------------------------------------------- report formats

class ResearchRequest(BaseModel):
    market_view: str = Field(description="2-4 sentences: what the market is doing and what you want to find")
    tickers: list[str] = Field(description=f"Up to {MAX_RESEARCH} US-listed stocks or ETFs to research now "
                                           "(held positions are always included)")
    reasons: str = Field(description="Why these, in a few sentences")


class TargetPosition(BaseModel):
    ticker: str
    weight: float = Field(description="Share of the whole account, 0 to 1 (e.g. 0.25 = 25%)")
    thesis: str = Field(description="Why you own it, citing the data provided")
    risks: str = Field(description="What would prove you wrong")
    exit_plan: str = Field(description="When you would sell: price, event or date")


class PortfolioDecision(BaseModel):
    market_view: str
    positions: list[TargetPosition] = Field(description="The complete portfolio you want to hold after this "
                                                        "run. Anything held and not listed is sold.")
    cash_weight: float = Field(description="Share of the account kept in cash, 0 to 1")
    reasoning: str = Field(description="How you arrived at this portfolio and why it maximises expected return")
    what_would_change_my_mind: str
    confidence: int = Field(description="1-5: how confident you are this beats SPY over the next month")


# ---------------------------------------------------------------- the book

@dataclass
class Book:
    start_date: str
    start_equity: float
    cash: float
    spy_start: float
    positions: dict[str, float] = field(default_factory=dict)        # ticker -> shares
    ai_cost_total: float = 0.0
    history: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = BOOK_FILE) -> "Book | None":
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text()))

    def save(self, path: Path = BOOK_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, default=str))

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(sh * prices.get(t, 0.0) for t, sh in self.positions.items())

    def spy_equivalent(self, spy_price: float) -> float:
        return self.start_equity * spy_price / self.spy_start


@dataclass
class Order:
    side: str           # "sell" or "buy"
    ticker: str
    shares: float
    price: float

    @property
    def usd(self) -> float:
        return round(self.shares * self.price, 2)


def plan_orders(book: Book, decision: PortfolioDecision, prices: dict[str, float]) -> tuple[list[Order], list[str]]:
    """Turn target weights into orders. Returns (orders, notes about anything adjusted)."""
    notes: list[str] = []
    equity = book.equity(prices)
    targets: dict[str, float] = {}
    for p in decision.positions:
        t = p.ticker.strip().upper()
        if t not in prices or not prices[t] > 0:
            notes.append(f"{t}: no price data, left out")
            continue
        targets[t] = targets.get(t, 0.0) + max(0.0, p.weight)
    total = sum(targets.values())
    if total > 1.0:
        notes.append(f"weights added up to {total:.0%}; scaled down to 100%")
        targets = {t: w / total for t, w in targets.items()}
    orders: list[Order] = []
    for t in sorted(set(book.positions) | set(targets)):
        price = prices.get(t)
        if not price:
            notes.append(f"{t}: held but no price today, kept as is")
            continue
        have = book.positions.get(t, 0.0) * price
        want = targets.get(t, 0.0) * equity * 0.995        # a small buffer so buys fit the cash
        diff = want - have
        if t not in targets and book.positions.get(t, 0.0) > 0:
            orders.append(Order("sell", t, book.positions[t], price))          # sell all of it
        elif abs(diff) >= MIN_ORDER_USD:
            shares = math.floor(abs(diff) / price * 10_000) / 10_000
            if shares > 0:
                orders.append(Order("sell" if diff < 0 else "buy", t, shares, price))
    orders.sort(key=lambda o: o.side != "sell")                               # sells first
    return orders, notes


def apply_orders(book: Book, orders: list[Order]) -> None:
    for o in orders:
        if o.side == "sell":
            book.positions[o.ticker] = round(book.positions.get(o.ticker, 0.0) - o.shares, 4)
            book.cash += o.shares * o.price
        else:
            cost = min(o.shares * o.price, book.cash)
            book.positions[o.ticker] = round(book.positions.get(o.ticker, 0.0) + cost / o.price, 4)
            book.cash -= cost
        if book.positions.get(o.ticker, 0.0) <= 1e-4:
            book.positions.pop(o.ticker, None)
    book.cash = round(book.cash, 2)


# ---------------------------------------------------------------- context for the agent

def scoreboard(book: Book, prices: dict[str, float], spy_price: float) -> dict:
    eq = book.equity(prices)
    spy_eq = book.spy_equivalent(spy_price)
    return {"started": book.start_date, "start_equity": book.start_equity, "equity_now": round(eq, 2),
            "return_pct": round((eq / book.start_equity - 1) * 100, 2),
            "same_money_in_spy": round(spy_eq, 2), "spy_return_pct": round((spy_eq / book.start_equity - 1) * 100, 2),
            "ai_cost_so_far": round(book.ai_cost_total, 2),
            "equity_after_ai_cost": round(eq - book.ai_cost_total, 2)}


def holdings_view(book: Book, prices: dict[str, float]) -> list[dict]:
    eq = book.equity(prices) or 1.0
    out = []
    for t, sh in sorted(book.positions.items()):
        p = prices.get(t)
        out.append({"ticker": t, "shares": sh, "price": p, "value": round(sh * (p or 0), 2),
                    "weight_pct": round(sh * (p or 0) / eq * 100, 1)})
    return out


def past_decisions(book: Book, n: int = 5) -> list[dict]:
    return [{k: h.get(k) for k in ("date", "equity", "spy_equivalent", "market_view", "positions", "confidence")}
            for h in book.history[-n:]]


def movers(snaps: dict[str, dict], n: int = 10) -> dict:
    """Biggest 20- and 60-day gainers and losers among the scanned stocks."""
    rows = [(t, s["change_pct"].get("20d"), s["change_pct"].get("60d")) for t, s in snaps.items()
            if not s.get("stale")]
    def top(k, rev):
        valid = [r for r in rows if r[k] is not None]
        return [{"ticker": r[0], "pct": r[k]} for r in sorted(valid, key=lambda r: r[k], reverse=rev)[:n]]
    return {"gainers_20d": top(1, True), "losers_20d": top(1, False),
            "gainers_60d": top(2, True), "losers_60d": top(2, False)}


def earnings_view(earn: dict[str, pd.DataFrame], tickers: list[str]) -> dict:
    out = {}
    for t in tickers:
        df = earn.get(t)
        if df is None or not len(df):
            continue
        last = df.sort_values("date").tail(4)
        out[t] = [{"date": str(pd.Timestamp(r.date).date()), "eps_estimate": r.eps_estimate,
                   "eps_actual": r.eps_actual, "surprise_pct": None if pd.isna(r.surprise_pct)
                   else round(float(r.surprise_pct), 1)} for r in last.itertuples()]
    return out


HOUSE_NOTES = """Findings from this desk's own backtests (2016-2026, S&P 500 members at the time), for your
information only; you are free to ignore them:
- Chart setups (breakouts, pullbacks, volatility contraction, dip-buys) averaged about zero per trade.
- Buying options on those setups lost 4-8% per trade after spreads and time decay.
- Monthly momentum (holding the recent biggest winners) beat SPY in 2021-2026 but lagged in 2016-2021,
  with 40-60% drawdowns when concentrated.
- Buying after earnings beats or big positive earnings reactions did not beat SPY out of sample.
- Holding SPY itself beat every tested strategy over 10 years."""


# ---------------------------------------------------------------- one run

@dataclass
class FreeHandRun:
    research: object = None
    decision: object = None
    orders: list[Order] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    score_before: dict = field(default_factory=dict)
    cost: float = 0.0
    error: str | None = None


def run_free_hand(settings: Settings, runner, book: Book, bars: dict[str, pd.DataFrame], snaps: dict[str, dict],
                  fetch_bars, fetch_news, fetch_earnings, today: str, regime: dict) -> FreeHandRun:
    """Research, then decide. Mutates nothing; the caller applies the orders."""
    from .indicators import snapshot
    from .prompts import load_prompt

    run = FreeHandRun()
    prices = {t: float(df["Close"].iloc[-1]) for t, df in bars.items()}
    spy = prices["SPY"]
    run.score_before = scoreboard(book, prices, spy)
    model = settings.models["portfolio_manager"]
    common = dict(today=today, scoreboard=run.score_before, holdings=holdings_view(book, prices) or "none (all cash)",
                  cash=f"${book.cash:,.2f}", past_decisions=past_decisions(book) or "none yet (first run)",
                  market=regime, movers=movers(snaps), house_notes=HOUSE_NOTES)

    p1 = load_prompt("free_hand", settings.prompt_versions.get("free_hand_research", "research_v1"))
    run.research = runner.run(p1, p1.render(**common, max_research=str(MAX_RESEARCH)), ResearchRequest, model,
                              effort="medium", max_tokens=8000)
    run.cost += run.research.cost_usd
    if not run.research.ok:
        run.error = f"research step failed: {run.research.error}"
        return run
    wanted = [t.strip().upper().replace(".", "-") for t in run.research.report.tickers][:MAX_RESEARCH]
    tickers = sorted(set(wanted) | set(book.positions))

    missing = [t for t in tickers if t not in bars]
    if missing:
        bars.update(fetch_bars(missing))
    data = {}
    for t in tickers:
        if t in bars and len(bars[t]) >= 30:
            s = snapshot(bars[t])
            s.pop("last_10_closes", None)
            data[t] = s
            prices[t] = float(bars[t]["Close"].iloc[-1])
        else:
            run.notes.append(f"{t}: no price data found (not a US-listed ticker?)")
    news = fetch_news(list(data), frozenset(settings.etfs))
    earnings = earnings_view(fetch_earnings(list(data)), list(data))

    p2 = load_prompt("free_hand", settings.prompt_versions.get("free_hand_decide", "decide_v1"))
    rendered = p2.render(**common, research_request=run.research.report.model_dump(), research_data=data,
                         news=news, earnings=earnings)
    run.decision = runner.run(p2, rendered, PortfolioDecision, model,
                              effort=settings.effort.get("portfolio_manager"), max_tokens=16000)
    run.cost += run.decision.cost_usd
    if not run.decision.ok:
        run.error = f"decision step failed: {run.decision.error}"
        return run
    orders, notes = plan_orders(book, run.decision.report, prices)
    run.orders, run.notes = orders, run.notes + notes
    return run


def record(book: Book, run: FreeHandRun, prices: dict[str, float], spy_price: float, today: str) -> None:
    d = run.decision.report
    apply_orders(book, run.orders)
    book.ai_cost_total = round(book.ai_cost_total + run.cost, 4)
    book.history.append({
        "date": today, "equity": round(book.equity(prices), 2),
        "spy_equivalent": round(book.spy_equivalent(spy_price), 2), "ai_cost": round(run.cost, 4),
        "market_view": d.market_view, "confidence": d.confidence,
        "positions": [{"ticker": p.ticker.upper(), "weight": p.weight, "thesis": p.thesis,
                       "exit_plan": p.exit_plan} for p in d.positions],
        "orders": [o.__dict__ for o in run.orders]})


# ---------------------------------------------------------------- command line

def _print_score(s: dict) -> None:
    ahead = s["equity_now"] - s["same_money_in_spy"]
    print(f"\nSCOREBOARD since {s['started']}: fund ${s['equity_now']:.2f} ({s['return_pct']:+.2f}%)  vs  "
          f"same money in SPY ${s['same_money_in_spy']:.2f} ({s['spy_return_pct']:+.2f}%)  -> "
          f"{'AHEAD' if ahead >= 0 else 'BEHIND'} by ${abs(ahead):.2f}")
    print(f"AI cost so far ${s['ai_cost_so_far']:.2f}; fund after AI cost ${s['equity_after_ai_cost']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="decide, but do not update the book")
    ap.add_argument("--status", action="store_true", help="show the scoreboard only (no AI cost)")
    ap.add_argument("--equity", type=float, help="starting money for a new book (default from config)")
    args = ap.parse_args()

    from . import market_data
    from .config import load_env_file, load_settings
    from .earnings import load_earnings
    from .llm import ClaudeRunner
    from .universe import sp500_symbols

    load_env_file()
    settings = load_settings()
    book = Book.load()
    scan = sorted(set(sp500_symbols()) | set(settings.allowlist))
    symbols = sorted(set(scan) | set(settings.regime_symbols) | set(settings.sector_etfs) | {"SPY"}
                     | set(book.positions if book else []))
    print(f"Downloading prices for {len(symbols)} symbols (cached)...")
    bars = market_data.download_bars(symbols, priority=["SPY", *settings.regime_symbols, *settings.sector_etfs])
    if "SPY" not in bars:
        raise SystemExit("No SPY data: Yahoo is throttling. Wait 15-30 minutes and run again.")
    today = str(date.today())
    spy_price = float(bars["SPY"]["Close"].iloc[-1])
    if book is None:
        start = args.equity or settings.paper_equity
        book = Book(start_date=today, start_equity=start, cash=start, spy_start=spy_price)
        print(f"New free-hand book: ${start:.2f} in cash, measured against SPY at ${spy_price:.2f}.")
    prices = {t: float(df["Close"].iloc[-1]) for t, df in bars.items()}
    if args.status:
        _print_score(scoreboard(book, prices, spy_price))
        for h in holdings_view(book, prices):
            print(f"  {h['ticker']:<6} {h['shares']:>10.4f} sh  ${h['value']:>8.2f}  {h['weight_pct']:>5.1f}%")
        print(f"  cash   ${book.cash:.2f}")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("No API key found. Put ANTHROPIC_API_KEY=sk-ant-... in the .env file.")

    snaps = market_data.technical_packet(bars, [t for t in scan if t in bars],
                                         str(bars["SPY"].index[-1].date()))["tickers"]
    regime = market_data.regime_packet(bars, list(settings.regime_symbols), list(settings.sector_etfs),
                                       list(settings.allowlist))

    def fetch_bars(tickers):
        return market_data.download_bars(tickers, log=lambda *a: None)

    run = run_free_hand(settings, ClaudeRunner(), book, bars, snaps, fetch_bars, market_data.catalyst_packet,
                        lambda ts: load_earnings(ts, log=lambda *a: None), today, regime)
    _print_score(run.score_before)
    if run.research and run.research.ok:
        r = run.research.report
        print(f"\n--- Research request\n{r.market_view}\nLooking at: {', '.join(r.tickers)}\n{r.reasons}")
    if run.error:
        print(f"\nSTOPPED: {run.error}. Nothing changes. AI cost ${run.cost:.4f}")
        return
    d = run.decision.report
    print(f"\n--- Decision (confidence {d.confidence}/5)\n{d.market_view}\n\n{d.reasoning}")
    for p in d.positions:
        print(f"\n  {p.ticker.upper():<6} {p.weight * 100:5.1f}%  {p.thesis}\n         risks: {p.risks}\n"
              f"         exit: {p.exit_plan}")
    print(f"\n  cash   {d.cash_weight * 100:5.1f}%\nWhat would change its mind: {d.what_would_change_my_mind}")
    for n in run.notes:
        print(f"note: {n}")
    print("\n--- ORDERS (place these in Robinhood; sells first)" if run.orders else "\n--- No orders: keep as is")
    for o in run.orders:
        print(f"  {o.side.upper():<4} {o.ticker:<6} {o.shares:.4f} shares (~${o.usd:.2f} at ${o.price:.2f})")
    print(f"\nAI cost this run: ${run.cost:.4f}")
    if args.dry_run:
        print("Dry run: the book was not updated.")
        return
    record(book, run, prices, spy_price, today)
    book.save()
    print(f"Book updated: data/free_hand/book.json (cash ${book.cash:.2f}). If a fill in Robinhood differs, "
          f"edit the shares or cash there.")


if __name__ == "__main__":
    main()
