# 45-DTE NIFTY Short Straddle — LIVE executor deploy

**STATUS: DEPLOYED (unarmed until Arun arms it)** · 2026-09-09 · research/119 · real money

---

## 1. The Ask

> "Manual entry at ~15:30 - pls ensure its automated entry and management and exit"
> — after "option b … for 11 sep trade"

Automate the whole cycle for the book research/119 validated: enter, manage, exit,
with real money, starting with the **11-Sep-2026** campaign. Until now this book
existed only as a paper record — `services/straddle45_paper.py` writes SQLite rows
from bhavcopy and has **no order placement of any kind**.

## 2. What is automated — exactly the ruleset of record, nothing invented

| | |
|---|---|
| Entry | ATM NIFTY straddle, **45 calendar days** before the monthly expiry, rolled off weekends |
| Filter | India VIX percentile rank vs prior 252 sessions **> 25**; below that the cycle is SKIPPED |
| Size | **3 lots = 195 qty**, product NRML (carried) |
| Target | combined premium ≤ **50%** of entry credit |
| Stop | combined premium ≥ **200%** of entry credit |
| Time exit | **21 DTE**, rolled off weekends |
| Decision window | **15:20–15:29 IST**, once per session |

**Why one decision window and not continuous monitoring.** The study strikes entries
and exits on the session CLOSE. Phase D checked whether a finer cadence changes
anything on 28.3M real 1-minute quotes: in the DTE≥21 band the ATM straddle travels
a mean +6.3%/−4.3% around its close, and **zero of 60 sessions travelled ≥50%** either
way, so the target and stop are never approached intraday. Evaluating once near the
close is both faithful to the tested rule and operationally identical. An intraday
trigger would be a *different* rule with no evidence behind it.

**The stop is implemented but is not expected to earn.** Phase G measured it as
harmful when it fires (−130.9 pts, t −2.34) and it has never fired on the VIX>25 book
in 61 campaigns. It ships because it is the ruleset of record, not because it pays.

## 3. Safety design

| Guard | What it does |
|---|---|
| `STRADDLE45_LIVE=1` | **Not set = dry run.** Logs the exact orders, sends nothing. |
| Kill file `backtest_data/straddle45_KILL` | `touch` it → every order refused instantly, no deploy |
| Reconcile-first, BOTH ways | Reads `kite.positions()` and HALTS on any mismatch in either direction - a book leg missing at the broker, OR a broker leg the book does not know about (an order that filled while the DB write failed would otherwise be re-entered on top of). Scoped to NRML so NAS's intraday MIS legs are not mistaken for this book's — the 2026-08-06 SENSEX phantom and 2026-08-14 momentum ledger corruption were both "assumed instead of read" |
| **Both legs or neither** | If one leg fills and the other is rejected, the filled leg is bought back at once. A lone short option is the one outcome this book must never produce; if the unwind *also* fails it HALTs with an explicit manual-action message |
| Fill verification | Every order polled through `order_history` to a terminal state; an unverified order halts rather than being assumed good |
| Margin gate | Refuses to enter unless available ≥ **1.25×** the real basket requirement |
| Liquidity gate | Both legs must show volume ≥ 1,000 and OI ≥ 10,000 before selling |
| `panic` command | Flattens every open leg immediately, whatever the rules say |

Orders route through `services.kite_service.get_kite()`, which auto-injects
`market_protection` — Kite has rejected bare MARKET orders on options since
2026-08-14.

## 4. Three bugs found in pre-flight testing — each would have broken Friday silently

None raised an error. All three returned a plausible wrong answer.

1. **`sessions()` excluded today.** The daily NIFTY bar is only written after the
   close, so at 15:20 on the entry day the session list still ended at *yesterday*.
   Fixed by appending today when the broker confirms the market is actually open
   (a weekday can still be an exchange holiday).
2. **`entry_session()` collapsed on future dates.** `last session ≤ target` returns
   TODAY when the target is in the future — so on 09-Sep it reported 09-Sep as the
   entry day for the 27-Oct expiry and **would have entered two days early**.
3. **`exit_session()` had the same collapse** — it returned TODAY for a 06-Oct exit,
   so the executor **would have closed the position on the day it opened.**

Both (2) and (3) are the same class as the paper book's `prev_session` future-date
collapse fixed in Aug-2026. Entry and exit are now CALENDAR dates rolled off
weekends — the study's own stated convention — and verified against the book's
published forward plan:

```
2026-10   expiry 2026-10-27   entry 2026-09-11   exit 2026-10-06   <- matches the page
```

Entry carries **exactly one session** of grace, to cover an exchange holiday on the
nominal day. It was 4 calendar days on first deploy; Phase I (2026-09-11) measured
entries two or more sessions late at **+12.0 points against +99.5** for an on-time
entry (t 0.10) while taking max drawdown from âˆ’564.8 to âˆ’978.5, so the window was
narrowed the same day. Miss the day and the cycle is gone.

## 5. Status

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-09 11:50 | Executor written | `services/straddle45_live.py` |
| 2026-09-09 11:58 | Dry run OK | connects, reconciles 0 vs broker, respects the window |
| 2026-09-09 12:00 | 3 date bugs found and fixed | see section 4 |
| 2026-09-09 12:02 | Dates verified against the published plan | entry 11-Sep, exit 06-Oct ✓ |

## 6. Crash recovery / manual operation

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd
tail -50 /tmp/straddle45_live.log                      # what it did and why
./venv/bin/python3 services/straddle45_live.py status  # the book
touch backtest_data/straddle45_KILL                    # STOP all orders NOW
set -a; . ./.env; set +a
STRADDLE45_LIVE=1 ./venv/bin/python3 services/straddle45_live.py panic   # flatten
```

A HALT exits with code 2 and writes the reason to the log and the `events` table.
**A HALT never leaves an order in flight** — it stops before placing, or after a
verified fill.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `services/straddle45_live.py` | the executor | yes |
| `backtest_data/straddle45_live.db` | positions + events | no (data) |
| `static/app/straddle45_live.json` | published state | no (generated) |
| this file | deploy record | yes |

## 8. Open before/at go-live

- **11-Sep is a SKIP as things stand**: India VIX 11.15, rank **17.9**, filter needs
  > 25. The executor will correctly do nothing unless volatility rises by Friday.
  `STRADDLE45_OFF_PLAN=1` overrides the filter — deliberately not set.
- **The order path has never placed a real order.** Everything up to placement is
  tested; the fill path itself is first exercised on the first live entry.
- Ring-fence still unresolved (₹11.96L vs ₹13.5L — 3 lots breaches at an 8% move).
- Stress-margin vol axis still dated 2026-11-30.
