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

Expect a line like **68 passed** with no failures. Then look at today's market data (no AI, no cost):

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

The desk trades **only long calls and long puts** (set in `config/desk.toml`,
`[strategy] instrument = "options"`). Your Robinhood Agentic account needs
**options trading enabled** (Robinhood asks a few questions to approve it).

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
   - Stop: place a **stop-limit sell** at the stop price if the app offers it;
     otherwise check the option each afternoon and sell if it is at or below
     the stop.
   - Take profit: a **limit sell** at the take-profit price (good till canceled).
   - Time: sell by the "sell by" date no matter what.
   Robinhood allows only one open sell order per contract, so choose the stop
   or the take-profit order and watch the other level yourself.
5. For a SELL instruction, close the whole option position and cancel its
   open orders.
6. Tell the team what you hold: copy the example file and edit it after every
   trade:

   ```bash
   cp config/account.example.toml data/account.toml
   open -e data/account.toml
   ```

   Set equity and cash to what the app shows. For an option position use the
   contract as the ticker (for example `ticker = "NVDA 2026-11-06 180 CALL"`),
   the number of contracts as shares, and the premium you paid as entry.

Skipping a trade is always allowed. Every trade uses all available cash, and
its cost is the most it can lose: the 15% stop limits a normal loss to about
15%, but an option can gap through the stop, and if the stop is not honoured
an option can expire worthless.

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
