# Hedge Fund

An AI analyst team that swing-trades (2–15 day holds) a small Robinhood Agentic
account. Three analysts research, a portfolio manager decides, and code (not AI)
enforces every risk limit.

**Status: Step 2 of the build.** The analyst team runs on real market data and
prints its reports and decisions. It cannot place orders yet.

## The team

| Member | Model | Question it answers | Report |
| --- | --- | --- | --- |
| Technical Analyst | Haiku 4.5 | Is the chart set up for a trade? | Trend, setup, entry, stop, target |
| News / Catalyst Analyst | Haiku 4.5 | Is there a reason to move, or a danger coming? | Earnings date, event risk, catalysts, red flags |
| Market Regime Analyst | Haiku 4.5 | Is the market friendly right now? | Risk on/off, posture, max new positions |
| Portfolio Manager | Opus 5.5 | Given all reports, what do we do? | Orders, exits, warnings, honest assessment |

How a run works:

1. Code downloads daily prices and computes indicators and setups (free).
2. The scanner keeps only tickers with a setup, plus open positions. None means
   a quiet day and **no AI cost**.
3. The three analysts run in parallel, each receiving one data packet and
   replying in a strict format (`desk/schemas.py`).
4. If any analyst fails, the run stops with no decision. Otherwise the Portfolio
   Manager reads all three reports and decides.

## Quick start (on your computer)

Needs Python 3.11 or newer.

```bash
git clone https://github.com/Jakegilbs/Hedge-Fund-.git
cd Hedge-Fund-
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # offline tests, no cost
```

See the data the analysts would get (no AI cost):

```bash
python -m desk.meet_the_team --data-only
```

Meet the team (costs about $0.05–0.30 per run):

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=sk-ant-...
python -m desk.meet_the_team                        # scanner picks tickers
python -m desk.meet_the_team --tickers NVDA,JPM     # force specific tickers
```

Each run prints every report, the PM's decision and the cost, and saves a full
record to `data/runs/` (not committed).

## Changing the team

- **Prompts** live in `prompts/<role>/<version>.md`. To try a change, copy `v1.md`
  to `v2.md`, edit it, and set the version in `config/desk.toml`. Every run logs
  the version and a fingerprint of the exact wording, so results can be compared.
- **Models, allowlist and risk limits** live in `config/desk.toml`.

## Layout

```
config/desk.toml        allowlist, risk limits, model and prompt version per role
prompts/                versioned prompts for each team member
desk/indicators.py      indicators and setups (breakout, pullback, VCP)
desk/market_data.py     price, news and earnings data (yfinance)
desk/scanner.py         picks which tickers the analysts look at
desk/schemas.py         report format for each team member
desk/llm.py             one Claude call per agent, with cost tracking
desk/team.py            runs the team, fail-safe
desk/meet_the_team.py   command-line entry point
tests/                  offline tests with a fake Claude client
```

## Safety

- Never commit API keys or logins. `.env` and `data/` are git-ignored.
- This repository is public: prompts and strategy are visible to anyone.
- Nothing here can place an order. Order placement arrives later, behind the
  Gatekeeper, and only for gatekeeper-approved orders.

## Next steps

1. Gatekeeper: recompute shares and reward-to-risk, enforce limits and circuit breakers.
2. Trade journal and paper trading with a simulated $100 account.
3. Robinhood MCP connection (read-only first) and the n8n workflow.
