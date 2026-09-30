# Hedge Fund

An AI analyst team that swing-trades (2–15 day holds) a small Robinhood Agentic
account. In the default **hybrid** mode a bullish pick is a long call when a
liquid, affordable contract exists and fractional shares otherwise. Three
analysts research the setups, code picks one liquid, affordable contract per
candidate, a portfolio manager chooses at most one, and code (not AI) sizes it
with all available cash and enforces every limit: one position at a time,
conviction 3+/5, stock reward-to-risk 1.5+, no earnings inside the option's
life, and an exit plan on every trade (sell when the stock breaks the setup's
stop, +100% take profit, out 5 days before expiry).

Rules set from the backtest (`python -m desk.backtest`): **bullish setups only**
(bearish setups lost money), **no new trades while SPY is below its 50-day
average**, and the option stop is the **stock's** stop level (a -15% option stop
fired on ordinary daily noise). Calls are also off for now (`calls = false`):
every trade is fractional shares. Two new strategy types, **momentum** and
**dip_buy**, are in the backtester only; the one that proves an edge goes live.
All of these are switches in `config/desk.toml`.

**Status: Level 1 (you approve and place every trade).** The analyst team runs
on real market data, the Gatekeeper checks and sizes every order, and you place
the approved orders yourself in Robinhood (SETUP_GUIDE.md, Step 8). Automatic
order placement comes later.

## The team

| Member | Model | Question it answers | Report |
| --- | --- | --- | --- |
| Technical Analyst | Haiku 4.5 | Is the chart set up for a trade? | Trend, setup, entry, stop, target |
| News / Catalyst Analyst | Haiku 4.5 | Is there a reason to move, or a danger coming? | Earnings date, event risk, catalysts, red flags |
| Market Regime Analyst | Haiku 4.5 | Is the market friendly right now? | Risk on/off, posture, max new positions |
| Portfolio Manager | Opus 5.5 | Given all reports, what do we do? | Orders, exits, warnings, honest assessment |

How a run works:

1. Code downloads daily prices for the whole universe and computes indicators
   and setups (free). The universe is the S&P 500 and core ETFs; in options mode
   it also includes cheap, liquid Nasdaq stocks ($5-30, $20M+ a day, screened
   weekly from Nasdaq's full symbol list) and a list of popular cheap optionable
   names, and skips stocks whose options the account cannot afford.
2. The scanner keeps tickers with a fresh setup that pass the liquidity filter,
   ranks them by strength relative to SPY, and sends only the top 8 (plus any open
   positions) to the analysts. None means a quiet day and **no AI cost**.
3. The three analysts run in parallel, each receiving one data packet and
   replying in a strict format (`desk/schemas.py`).
4. If any analyst fails, the run stops with no decision. Otherwise the Portfolio
   Manager reads all three reports and decides.

## Quick start

**New here? Follow [SETUP_GUIDE.md](SETUP_GUIDE.md)** for step-by-step
instructions (Mac and Windows). Everything installs into one `HedgeFund`
folder on your Desktop.

Short version, from inside the project folder:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install --no-cache-dir -r requirements.txt
python -m pytest -q                                  # offline tests, no cost
python -m desk.meet_the_team --data-only             # today's data, no AI cost
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env           # key stays in this folder
python -m desk.meet_the_team --tickers NVDA,JPM      # meet the team (< $0.30)
python -m desk.backtest                              # test the setups on ~4 years of history, no AI cost
python -m desk.rotation                              # test monthly momentum rotation, no AI cost
```

Each run prints every report, the PM's decision and the cost, and saves a full
record to `data/runs/` (not committed).

## Changing the team

- **Prompts** live in `prompts/<role>/<version>.md`. To try a change, copy `v1.md`
  to `v2.md`, edit it, and set the version in `config/desk.toml`. Every run logs
  the version and a fingerprint of the exact wording, so results can be compared.
- **Models, universe, candidate limit and risk limits** live in `config/desk.toml`.

## Layout

```
config/desk.toml        allowlist, risk limits, model and prompt version per role
prompts/                versioned prompts for each team member
desk/indicators.py      indicators and setups (breakout, pullback, VCP)
desk/market_data.py     price, news and earnings data (yfinance)
desk/universe.py        S&P 500 + weekly cheap-Nasdaq screen (cached) plus the core list
desk/scanner.py         ranks setups and picks which tickers the analysts see
desk/schemas.py         report format for each team member
desk/llm.py             one Claude call per agent, with cost tracking
desk/team.py            runs the team, fail-safe
desk/options.py         picks the option contract (expiry, strike, liquidity, cost)
desk/gatekeeper.py      recomputes sizes, enforces every limit (stock and options)
desk/meet_the_team.py   command-line entry point
desk/backtest.py        replays history: setups traded as shares and estimated options, plus a $100 account
desk/rotation.py        monthly momentum rotation backtest vs SPY and an equal-weight benchmark
desk/membership.py      historical S&P 500 membership, so backtests only rank stocks that were members then
desk/earnings.py        earnings history (Yahoo, cached) and the positive earnings-drift flag
desk/earnings_study.py  earnings event study: surprise, reaction, and the next 5/20/60 days vs SPY
desk/free_hand.py       free-hand fund: the AI researches and decides the whole portfolio; code keeps score vs SPY
desk/insiders.py        open-market insider purchases from the SEC's quarterly Form 4 data sets (cached)
desk/insider_study.py   insider buying study: clusters, size, who bought; 20/60/120 days vs SPY or IWM
tests/                  offline tests with a fake Claude client
```

## Safety

- Never commit API keys or logins. `.env` and `data/` are git-ignored.
- This repository is public: prompts and strategy are visible to anyone.
- Nothing here can place an order. Order placement arrives later, behind the
  Gatekeeper, and only for gatekeeper-approved orders.

## Next steps

1. Trade journal, weekly halt and drawdown stop (need trade history).
2. Robinhood MCP connection (read-only first), then the Executor and Monitor.
3. n8n workflow and phone alerts.
