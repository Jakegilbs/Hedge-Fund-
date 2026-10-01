# Strategy under test: executive conviction buys

Written on 2026-10-01 from the insider buying study's exploration years
(2016-2021), **before** the locked years (2022 onward) were looked at. The rules
below are fixed; they are also in `desk/insider_study.py` (`STRATEGY`,
`strategy_mask`, `verdict`).

## Rule

Buy a stock at the first open after an SEC Form 4 filing shows that:

- its **CEO, CFO or President** bought shares **in the open market** (cash
  purchases, not grants or option exercises),
- for **$500,000 to $5,000,000** in total within 30 days,
- and the company trades **at least $1M a day** (no micro caps).

Hold **60 trading days**, then sell.

## Why (exploration years, 2016-2021)

| Group | Events | 60-day excess | Beat | 120-day excess | t |
|---|---|---|---|---|---|
| All insider buys (baseline) | 12,516 | +1.36% | 48% | +2.42% | 5.6 |
| CEO/CFO/President, $500k-5M | 872 | +4.12% | 53% | +5.63% | 4.1 |

- Top executives spending serious personal money is a costly, deliberate signal.
- The effect grows from 20 to 60 to 120 days rather than fading.
- Micro caps (under $1M/day) showed no effect (+0.75%) and are the hardest to
  trade and the most affected by missing data, so they are excluded.
- Purchases over $5M did worse (-0.41%): often controlling holders, not
  conviction buys.

## Pass rules for the locked years (2022 onward), tested once

1. 60-day excess at least **+1.5 points above** that period's baseline.
2. **t above 2.**
3. Beats the benchmark (SPY or IWM by size) in **at least 50%** of events.

All three must pass. Run `python -m desk.insider_study --confirm` once; the
report prints PASS/FAIL for each and a verdict.

## Known weaknesses

- Only about a third of insider events have price data; the missing ones are
  mostly delisted companies, so results lean optimistic.
- Fewer than half of insider events beat the benchmark: returns come from a
  minority of big winners, so the strategy needs many positions at once.
