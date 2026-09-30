from desk.config import RiskLimits, load_settings
from desk.gatekeeper import check
from desk.schemas import CatalystReport, CatalystView, Order, PMDecision, PositionUpdate, RegimeReport

# Diversified style (1% risk, 15% cap, 6 positions); the all-in style is tested below.
RISK = RiskLimits(risk_per_trade=0.01, max_position_pct=0.15, max_open_positions=6, min_reward_risk=2.0,
                  daily_halt_pct=-0.03, weekly_halt_pct=-0.06, max_drawdown_pct=-0.15)
ALL_IN = load_settings().risk
ALLOW = {"NVDA", "JPM", "AAPL", "SPY"}


def regime(posture="cautious", max_new=2):
    return RegimeReport(regime="neutral", posture=posture, max_new_positions_today=max_new, evidence=[],
                        leading_sectors=[], lagging_sectors=[], summary="", warnings=[])


def news(*tickers, risk="low"):
    return CatalystReport(views=[CatalystView(ticker=t, data_ok=True, next_earnings=None, days_to_earnings=None,
                                              event_risk=risk, sentiment="neutral", catalysts=[], red_flags=[],
                                              summary="") for t in tickers], warnings=[])


def buy(ticker="NVDA", entry=100.0, stop=95.0, target=112.0, shares=999.0, conviction=3):
    return Order(action="buy", ticker=ticker, shares=shares, order_type="limit", limit_price=entry,
                 stop_price=stop, target_price=target, reward_risk=9.9, thesis="t", bear_case="b",
                 conviction=conviction)


def decide(*orders, updates=()):
    return PMDecision(market_view="", orders=list(orders), position_updates=list(updates), warnings=[],
                      honest_assessment="")


def run(decision, equity=100.0, cash=100.0, pnl=0.0, positions=None, prices=None, stale=(), reg=None, cat=None,
        risk=RISK):
    return check(decision, equity=equity, cash=cash, pnl_today=pnl, positions=positions or {},
                 last_prices=prices or {"NVDA": 100.0, "JPM": 50.0, "AAPL": 200.0}, stale=set(stale),
                 allowlist=ALLOW, risk=risk, regime=reg or regime(), catalysts=cat or news("NVDA", "JPM", "AAPL"))


def test_shares_recomputed_and_capped_by_position_limit():
    # risk sizing: $1 / $5 = 0.2 shares ($20) but 15% cap = $15 -> 0.15 shares
    g = run(decide(buy()))
    [o] = g.approved
    assert o.shares == 0.15 and o.notional_usd == 15.0
    assert o.risk_usd <= 1.0 and o.reward_risk == 2.4


def test_wide_stop_sizes_by_risk():
    # $1 risk / $10 per share = 0.1 shares ($10), under the $15 cap
    [o] = run(decide(buy(entry=100, stop=90, target=125))).approved
    assert o.shares == 0.1 and o.risk_usd == 1.0


def test_low_reward_risk_rejected():
    g = run(decide(buy(target=105)))
    assert not g.approved and "reward-to-risk" in g.rejected[0].reasons[0]


def test_bad_level_order_rejected():
    g = run(decide(buy(entry=100, stop=101, target=110)))
    assert any("levels out of order" in r for r in g.rejected[0].reasons)


def test_limit_far_from_market_rejected():
    g = run(decide(buy(entry=100, stop=95, target=112)), prices={"NVDA": 90.0})
    assert any("from last price" in r for r in g.rejected[0].reasons)


def test_not_allowlisted_stale_and_event_risk_rejected():
    g = run(decide(buy(ticker="GME"), buy(ticker="JPM", entry=50, stop=48, target=55), buy(ticker="AAPL", entry=200, stop=190, target=230)),
            stale={"JPM"}, cat=news("NVDA", "JPM", "AAPL", risk="high"))
    reasons = {r.ticker: " ".join(r.reasons) for r in g.rejected}
    assert "allowlist" in reasons["GME"]
    assert "stale" in reasons["JPM"]
    assert "event risk high" in reasons["AAPL"]
    assert not g.approved


def test_no_averaging_down():
    g = run(decide(buy()), positions={"NVDA": 0.1})
    assert "already held" in g.rejected[0].reasons[0]


def test_daily_halt_blocks_buys_but_allows_exits():
    g = run(decide(buy(), updates=[PositionUpdate(ticker="JPM", decision="exit", reason="stop")]),
            pnl=-3.0, positions={"JPM": 0.2})
    assert g.halted and [o.action for o in g.approved] == ["sell"]
    assert g.approved[0].shares == 0.2


def test_flat_regime_blocks_entries():
    g = run(decide(buy()), reg=regime(posture="flat", max_new=0))
    assert not g.approved and "flat" in g.halted


def test_regime_position_count_respected_by_conviction():
    g = run(decide(buy("JPM", 50, 48, 55, conviction=2), buy("NVDA", conviction=5)), reg=regime(max_new=1))
    assert [o.ticker for o in g.approved] == ["NVDA"]
    assert "position count" in g.rejected[0].reasons[0]


def test_cash_limits_total_spend():
    g = run(decide(buy("NVDA"), buy("JPM", 50, 48, 55)), cash=20.0)
    assert sum(o.notional_usd for o in g.approved) <= 20.0


def test_sell_of_unheld_ticker_rejected():
    o = buy()
    o.action = "sell"
    g = run(decide(o))
    assert "not held" in g.rejected[0].reasons[0]


# ---------- all-in, high-conviction style (the current config) ----------

def test_config_is_all_in():
    assert ALL_IN.all_in and ALL_IN.max_open_positions == 1 and ALL_IN.min_conviction >= 4


def test_all_in_uses_all_cash_on_the_single_best_trade():
    g = run(decide(buy("JPM", 50, 48, 55, conviction=4), buy("NVDA", conviction=5)), risk=ALL_IN)
    [o] = g.approved
    assert o.ticker == "NVDA" and 99.0 <= o.notional_usd <= 100.0
    assert o.risk_usd <= 5.0            # 5% stop on a ~$99.50 position
    assert "position count" in g.rejected[0].reasons[0]


def test_all_in_rejects_low_conviction():
    g = run(decide(buy(conviction=3)), risk=ALL_IN)
    assert not g.approved and "conviction 3/5" in " ".join(g.rejected[0].reasons)


def test_all_in_rejects_wide_stop():
    g = run(decide(buy(entry=100, stop=85, target=140, conviction=5)), risk=ALL_IN)
    assert not g.approved and "15.0% below entry" in " ".join(g.rejected[0].reasons)


def test_all_in_blocked_while_a_position_is_open():
    g = run(decide(buy("JPM", 50, 48, 55, conviction=5)), positions={"AAPL": 0.5}, risk=ALL_IN)
    assert not g.approved and "position count" in g.rejected[0].reasons[0]
