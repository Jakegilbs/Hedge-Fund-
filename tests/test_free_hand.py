from types import SimpleNamespace

import pytest

from desk import free_hand as fh
from desk.config import load_settings
from desk.llm import ClaudeRunner

from .conftest import make_bars


class FakeClaude:
    def __init__(self, replies):
        self.replies, self.calls, self.prompts = replies, [], []
        self.beta = SimpleNamespace(messages=self)

    def create(self, **kw):
        title = kw["output_config"]["format"]["schema"]["title"]
        self.calls.append(title)
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(model=kw["model"], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
                               content=[SimpleNamespace(type="text", text=self.replies[title].model_dump_json())])


def pos(t, w):
    return fh.TargetPosition(ticker=t, weight=w, thesis="t", risks="r", exit_plan="e")


def decision(*positions, cash=0.0):
    return fh.PortfolioDecision(market_view="m", positions=list(positions), cash_weight=cash, reasoning="r",
                                what_would_change_my_mind="w", confidence=3)


def new_book(cash=100.0):
    return fh.Book(start_date="2026-09-30", start_equity=100.0, cash=cash, spy_start=500.0)


def test_orders_from_weights_sells_first_and_skips_tiny_trades():
    book = new_book(cash=20.0)
    book.positions = {"OLD": 2.0, "KEEP": 1.0}                   # OLD $40, KEEP $40, cash $20 -> equity $100
    prices = {"OLD": 20.0, "KEEP": 40.0, "NEW": 10.0}
    orders, notes = fh.plan_orders(book, decision(pos("KEEP", 0.4), pos("NEW", 0.6)), prices)
    assert [(o.side, o.ticker) for o in orders] == [("sell", "OLD"), ("buy", "NEW")]   # KEEP: diff < $1
    assert orders[0].shares == 2.0 and orders[1].usd == pytest.approx(59.7, abs=0.02)
    fh.apply_orders(book, orders)
    assert "OLD" not in book.positions and book.cash >= 0
    assert book.equity(prices) == pytest.approx(100.0, abs=0.01)


def test_weights_over_100_percent_are_scaled_and_unknown_tickers_dropped():
    book = new_book()
    orders, notes = fh.plan_orders(book, decision(pos("A", 0.8), pos("B", 0.8), pos("FAKE", 0.2)),
                                   {"A": 10.0, "B": 10.0})
    assert any("scaled down" in n for n in notes) and any("FAKE" in n for n in notes)
    assert sum(o.usd for o in orders) <= 100.0
    fh.apply_orders(book, orders)
    assert book.cash >= 0


def test_scoreboard_compares_with_spy():
    book = new_book()
    book.positions, book.cash = {"A": 11.0}, 0.0
    s = fh.scoreboard(book, {"A": 10.0}, spy_price=525.0)
    assert s["equity_now"] == 110.0 and s["same_money_in_spy"] == 105.0 and s["return_pct"] == 10.0


def test_full_run_researches_then_decides_and_records(tmp_path):
    settings = load_settings()
    bars = {"SPY": make_bars(300, drift=0.001, seed=1), "XLK": make_bars(300, seed=2)}
    new = {"NVDA": make_bars(300, start=120, seed=3)}
    research = fh.ResearchRequest(market_view="m", tickers=["NVDA", "NOPE"], reasons="r")
    claude = FakeClaude({"ResearchRequest": research,
                         "PortfolioDecision": decision(pos("NVDA", 0.5), cash=0.5)})
    book = new_book()
    run = fh.run_free_hand(settings, ClaudeRunner(client=claude), book, bars, snaps={},
                           fetch_bars=lambda ts: {t: new[t] for t in ts if t in new},
                           fetch_news=lambda ts, etfs: {"tickers": {t: {"headlines": []} for t in ts}},
                           fetch_earnings=lambda ts: {}, today="2026-09-30", regime={"breadth": 50})
    assert claude.calls == ["ResearchRequest", "PortfolioDecision"] and run.error is None
    assert "NVDA" in claude.prompts[1] and any("NOPE" in n for n in run.notes)
    [o] = run.orders
    assert o.side == "buy" and o.ticker == "NVDA" and o.usd == pytest.approx(49.75, abs=0.05)
    assert run.cost > 0
    prices = {"SPY": float(bars["SPY"]["Close"].iloc[-1]), "NVDA": o.price}
    fh.record(book, run, prices, prices["SPY"], "2026-09-30")
    book.save(tmp_path / "book.json")
    again = fh.Book.load(tmp_path / "book.json")
    assert again.positions["NVDA"] > 0 and again.history[-1]["positions"][0]["ticker"] == "NVDA"
    assert again.ai_cost_total == pytest.approx(run.cost)


def test_failed_research_changes_nothing():
    class Broken(FakeClaude):
        def create(self, **kw):
            raise __import__("anthropic").APIConnectionError(request=__import__("httpx").Request("POST", "http://x"))
    run = fh.run_free_hand(load_settings(), ClaudeRunner(client=Broken({})), new_book(),
                           {"SPY": make_bars(300, seed=1)}, {}, None, None, None, "2026-09-30", {})
    assert run.error and "research" in run.error and run.orders == []
