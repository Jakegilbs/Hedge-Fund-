# Setup Guide: Meet Your Analyst Team

Follow these steps in order, once. Total time: about 30 minutes.
Where Mac and Windows differ, both are shown.

## Where everything lives

Everything for this project is saved in **one folder on your Desktop: `HedgeFund`**.

```
Desktop/
└── HedgeFund/                 the whole project (downloaded from GitHub)
    ├── .env                   your Anthropic API key (private, never uploaded)
    ├── .venv/                 the project's own Python packages
    ├── data/
    │   ├── account.toml       your real balance and positions (you edit this)
    │   ├── runs/              a saved record of every team run
    │   └── cache/             downloaded market-data cache
    ├── config/desk.toml       allowlist, risk limits, models
    ├── prompts/               each analyst's instructions
    ├── desk/                  the code
    └── SETUP_GUIDE.md         this guide
```

To remove the project completely, delete the `HedgeFund` folder. The only things
installed outside it are the two apps in Step 1 (Python and Git).

---

## Step 1: Install Python and Git (one time)

**Python 3.11 or newer**

- Mac: download from https://www.python.org/downloads and run the installer.
- Windows: download from https://www.python.org/downloads. On the first
  installer screen, **tick "Add python.exe to PATH"**, then click Install Now.

**Git**

- Mac: nothing to download. The first time you type `git` in Terminal, a
  window offers to install it; click **Install**.
- Windows: download from https://git-scm.com/download/win and click through
  the installer with the default options.

## Step 2: Open a terminal

- Mac: press **Cmd + Space**, type `Terminal`, press Enter.
- Windows: press the **Windows key**, type `PowerShell`, press Enter.

Check that Python works (it should print `Python 3.11` or higher):

| Mac | Windows |
| --- | --- |
| `python3 --version` | `python --version` |

## Step 3: Download the project to your Desktop

Paste each line, press Enter, and wait for it to finish before the next.

**Mac**

```bash
cd ~/Desktop
git clone -b claude/great-lovelace-l6ar1p https://github.com/Jakegilbs/Hedge-Fund-.git HedgeFund
cd HedgeFund
```

**Windows**

```powershell
cd ([Environment]::GetFolderPath("Desktop"))
git clone -b claude/great-lovelace-l6ar1p https://github.com/Jakegilbs/Hedge-Fund-.git HedgeFund
cd HedgeFund
```

A `HedgeFund` folder now appears on your Desktop.

## Step 4: Install the project's packages (inside the folder)

**Mac**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --no-cache-dir -r requirements.txt
```

`(.venv)` now shows at the start of the line. That means you are using the
project's own Python, stored in `HedgeFund/.venv`.

If Windows says *running scripts is disabled*, run this once, type `Y`, then
repeat the Activate line:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Step 5: Run the free checks

```bash
python -m pytest -q
```

Expect a line like **145 passed** with no failures. Then look at today's market data (no AI, no cost):

```bash
python -m desk.meet_the_team --data-only
```

It prints the numbers the analysts receive. The last line lists tickers with a
setup today; `none` is normal.

## Step 6: Add your Anthropic API key (saved in the project folder)

1. Go to https://console.anthropic.com and sign up.
2. **Billing**: add $10 of credit.
3. **Limits**: set a monthly spend limit of $10.
4. **API Keys → Create Key**: copy the key (starts with `sk-ant-`). It is shown
   only once.

Save it into `HedgeFund/.env`, replacing the example text with your key:

| Mac | Windows |
| --- | --- |
| `echo 'ANTHROPIC_API_KEY=sk-ant-paste-your-key' > .env` | `Set-Content .env 'ANTHROPIC_API_KEY=sk-ant-paste-your-key'` |

The key now stays in that file, so you never have to type it again. `.env` is
excluded from GitHub. Never paste the key into GitHub, chat or email.

## Step 7: Meet the team

```bash
python -m desk.meet_the_team --tickers NVDA,JPM
```

`--tickers NVDA,JPM` makes the team analyse those two even if the scanner found
no setups. You will see, in order:

1. Technical Analyst report
2. News / Catalyst Analyst report
3. Market Regime Analyst report
4. Portfolio Manager decision
5. Gatekeeper: rejected orders with reasons, and APPROVED ORDERS with exact
   share counts
6. Total cost (expect under $0.30)

A full record is saved to `HedgeFund/data/runs/`.

Without `--tickers`, the scanner decides. On a quiet day it stops before any AI
call and costs nothing.

## Step 8: Trading live, with you approving (Level 1)

The desk runs in **hybrid** mode (`config/desk.toml`, `[strategy] instrument = "hybrid"`):
a bullish pick is a **long call** when a liquid, affordable one exists, otherwise
**fractional shares** of the stock. **Calls are switched off for now**
(`calls = false`): every trade is fractional shares until a setup proves an edge.
Two more rules come from the backtest (Step 9):
**bullish setups only** (no puts), and **no new trades while SPY is below its
50-day average** (the run says so at the top and costs nothing that day).
Your Robinhood Agentic account needs **options trading enabled** (Robinhood asks
a few questions to approve it).

Until the Robinhood connection is built, you are the executor. Each trading
day:

1. Run the team: `python -m desk.meet_the_team`
2. Read the **APPROVED ORDERS**. Only orders listed there may be placed; never
   place a REJECTED one or change the numbers.
3. To buy the option in the Robinhood app (Agentic account): open the stock,
   **Trade → Trade Options**, pick the **expiration date** and **Call** or
   **Put** exactly as shown, tap the **strike**, choose **Buy**, set the
   number of contracts and a **limit price** equal to the one shown.
4. Once it fills, protect it with the **exit plan** printed under the order:
   - Stop: sell the option if the **stock** trades below the stop level shown
     ("sell the option if TICKER trades below $X"). That is where the setup is
     proven wrong. Robinhood cannot trigger an option sale from the stock price,
     so set a **price alert** on the stock at that level in the app and sell
     when it fires.
   - Take profit: a **limit sell** at the take-profit price (good till canceled).
   - Time: sell by the "sell by" date no matter what.
   Keep the take-profit as the open sell order and handle the stop with the alert.
5. For a **shares** order (`BUY TICKER: 0.1234 shares, LIMIT $…`): open the stock,
   **Buy**, switch the amount to **shares**, enter the exact number and a limit
   price equal to the one shown. If the app only allows a market order for
   fractional shares, place it only while the price is within about 0.5% of the
   limit shown. Then place a **stop loss sell** at the stop price shown (or check
   it each afternoon if the app will not accept one for fractional shares).
6. For a SELL instruction, close the whole position and cancel its open orders.
7. Tell the team what you hold: copy the example file and edit it after every
   trade:

   ```bash
   cp config/account.example.toml data/account.toml
   open -e data/account.toml
   ```

   Set equity and cash to what the app shows. For an option position use the
   contract as the ticker (for example `ticker = "NVDA 2026-11-06 180 CALL"`),
   the number of contracts as shares, and the premium you paid as entry.

Skipping a trade is always allowed. Every trade uses all available cash, and
its cost is the most it can lose. Because the stop is on the stock, an option
usually loses more than 15% when it is hit (often 30-50%). If the stop is not
honoured, an option can expire worthless.

## Step 9: Test the strategy on history (free, no AI)

```bash
python -m desk.backtest
```

It replays about 4 years of history. Every day, it applies the scanner's setup
rules and trades each signal two ways: as shares, and as an estimated option
using the live exit plan. Then it runs a $100 account through it the way the
desk trades (one all-in position, best-ranked signal). The first run downloads
about 6 years of prices and takes several minutes. Later runs use the cache in
`data/cache/bars_long/`.

How to read it:

- **EXPECTANCY** is the average gain per trade after costs. Positive with **30+
  trades** means a setup has an edge; negative means it loses money over time.
- **With vs against the market** shows whether trading only with the market
  trend helps.
- **The $100 account** shows what the balance would have become, the worst
  drop along the way, and the AI cost for the same period.

Try other rules without changing the live desk:

```bash
python -m desk.backtest --option-stop stock    # exit options when the stock hits its stop
python -m desk.backtest --option-stop 0.5      # a -50% option stop instead of -15%
python -m desk.backtest --target-r 3 --hold 10 # bigger target, shorter hold
python -m desk.backtest --spread 0.05          # tighter option spreads
python -m desk.backtest --setups breakout,pullback --with-market   # only some setups, only with the trend
python -m desk.backtest --setups momentum,dip_buy  # the two strategy types being tested
python -m desk.backtest --half first           # tune on the first half of the years...
python -m desk.backtest --half second          # ...then check the same settings on the second half
```

A rule that only works in the half you tuned it on is luck, not an edge.

Every trade is saved to `data/backtests/` as a CSV (opens in Numbers or Excel).
The analysts are not simulated, option prices are estimates, and today's
stock list leaves out companies that went bust, so treat results as optimistic.

## Step 10: Test momentum rotation (free, no AI)

A different kind of strategy: hold the stocks that rose the most recently,
rebalance once a month, and hold cash while SPY is below its 200-day average.

```bash
python -m desk.rotation                              # 36 versions of 4 ranking ideas, last 10 years
python -m desk.rotation --lookback 126 --top 10 --score smooth --earnings   # one version in detail
python -m desk.rotation --rebalance 2                # every two months
```

How to read it:

- **Four ranking ideas** are tested, each with 3 lookbacks x 3 portfolio sizes:
  biggest gain; **smooth** gain (rise relative to volatility, skipping stocks
  whose rise came mostly from one day); and each of those limited to stocks in
  a **positive earnings drift** (last report beat estimates and the stock rose
  on it, within the last ~3 months). The **VERDICT** lines apply a rule set in
  advance: an idea shows an edge only if at least 2 of every 3 of its versions
  beat equal weight in both halves.
- The first run downloads about 15 years of earnings history for each stock
  (10-15 minutes, cached for a month).
- **Only stocks that were in the S&P 500 at the time are ranked**, using a free
  historical membership list (downloaded weekly). The coverage lines show how
  many members have price data; the missing ones were mostly acquired or went
  bust. `--membership today` repeats the old, biased way for comparison, and
  the report shows the gap.
- **Compare with "Equal weight, whole universe"** as well as SPY: it holds the
  same stocks on the same dates, so beating it is what shows a momentum effect.
- **Where the gains came from** lists the stocks that contributed most. If a
  few stocks did all the work, the result is fragile. The **data check** lists
  one-day moves over 50% in held stocks, which are often bad data.
- Every monthly position is saved to `data/backtests/rotation_holdings_*.csv`.
- The first 10-year run downloads about 12 years of prices for ~740 stocks and
  takes several minutes; later runs use the cache.
- **edge? = YES** means the version beat equal weight in both halves of the
  period. Look for a pattern (for example most top-5 and top-10 versions of
  one lookback winning), not the single best row: out of 60 versions, a few
  win by luck.
- **max drop** is the worst fall from a peak. Plan for it to happen again.

## Step 11: Earnings event study (free, no AI)

```bash
python -m desk.earnings_study
```

For every quarterly report of every S&P 500 member since 2016 it shows the EPS
surprise, how the stock reacted, and what it did over the next 5, 20 and 60
trading days compared with SPY (bought at the open after the reaction).

- Only **2016-2021** is shown. The years from 2022 on are **locked** so that a
  pattern found by searching the tables can be tested once on data it never saw.
- Look for groups with a large excess return, **|t| above 3**, **100+ reports**,
  and a steady result year by year.
- Write the strategy down first, then run `python -m desk.earnings_study --confirm`
  once to see the locked years.
- Every report is saved to `data/backtests/earnings_events_*.csv`.

## Step 12: The free-hand fund (the AI decides everything)

```bash
python -m desk.free_hand --dry-run    # see what it would do, without recording it
python -m desk.free_hand              # decide and record it in the book (about $0.20 of AI)
python -m desk.free_hand --status     # scoreboard only, no AI cost
```

The agent is told it is an expert portfolio manager whose only goal is to grow
the account, with no rules on style, number of stocks or holding period. It
picks up to 15 stocks or ETFs to research, gets fresh data on them (trend, news,
earnings), then decides the whole portfolio with a thesis and exit plan for each
position. The only limits are the account's: long stocks and ETFs, no options,
margin or short selling, so it can never lose more than the account holds.

- Run it about **once a week** (for example Monday after the open).
- Place the printed **ORDERS** in Robinhood, sells first. The book in
  `data/free_hand/book.json` assumes each order fills at the price shown; if a
  fill differs, edit the shares or cash there. Some stocks cannot be bought in
  fractional shares on Robinhood; skip those and tell Claude.
- The **SCOREBOARD** compares the fund with the same money in SPY and shows the
  AI cost. That comparison, week after week, is the only honest test: the AI
  already knows how past years turned out, so it cannot be backtested.

## Step 13: Insider buying study (free, no AI)

The SEC requires a contact in every automated request. Add one line to `.env`
(your name and email), once:

```bash
echo 'SEC_USER_AGENT=Your Name your.email@example.com' >> .env
python -m desk.insider_study
```

It downloads the SEC's quarterly insider-filing data since 2016 (the first run
takes a while: 40 quarters of 10-60 MB each, then prices for a few thousand
companies), keeps every open-market **purchase** by an officer, director or
10% owner, and measures what the stock did 20, 60 and 120 trading days after
the filing became public, against SPY (large companies) or IWM (small ones).

- Tables by number of insiders, dollars bought, who bought, company size, and
  the stock's move before the purchase, plus clusters year by year.
- As in Step 11, **2022 onward is locked**: pick a pattern from the exploration
  years, write it down, then run `--confirm` once.

---

## Every time after this

| Mac | Windows |
| --- | --- |
| `cd ~/Desktop/HedgeFund` | `cd (Join-Path ([Environment]::GetFolderPath("Desktop")) HedgeFund)` |
| `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| `python -m desk.meet_the_team` | `python -m desk.meet_the_team` |

To get the latest version of the code: `git pull` inside the folder.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| `python3: command not found` (Windows) | Use `python` or `py` instead |
| `git: command not found` | Finish Step 1, then close and reopen the terminal |
| `No module named desk` | You are not in the folder: go back to the `cd` line in Step 3 |
| `No module named pandas` (or similar) | Activate first (Step 4), then run again |
| `No API key found` | Redo Step 6 from inside the `HedgeFund` folder |
| `API error 401` | The key in `.env` is wrong: redo Step 6 with the full key |
| `Failed download` / all data `null` | Internet or Yahoo Finance issue: wait a minute and retry |
| Anything else | Copy the whole error and send it to Claude |
