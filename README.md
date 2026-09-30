# Hedge Fund

An AI analyst team that swing-trades (2–15 day holds) a small Robinhood Agentic
account. Three analysts research, a portfolio manager decides, and code (not AI)
enforces every risk limit.

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

1. Code downloads daily prices for the whole universe (S&P 500 plus core ETFs,
   about 510 tickers) and computes indicators and setups (free).
2. The scanner keeps tickers with a fresh setup that pass the liquidity filter,
   ranks them by 60-day strength, and sends only the top 8 (plus any open
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
desk/universe.py        S&P 500 list (cached weekly) plus the core list
desk/scanner.py         ranks setups and picks which tickers the analysts see
desk/schemas.py         report format for each team member
desk/llm.py             one Claude call per agent, with cost tracking
desk/team.py            runs the team, fail-safe
desk/gatekeeper.py      recomputes sizes, enforces every limit
desk/meet_the_team.py   command-line entry point
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
