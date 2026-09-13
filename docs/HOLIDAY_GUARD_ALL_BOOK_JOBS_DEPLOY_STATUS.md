# NSE holiday guard — every True North, Open Alpha and IPO Base job

**STATUS: DEPLOYED 13-Sep-2026 (Sunday), before the Monday 14-Sep Ganesh Chaturthi holiday**

## The ask

**What Arun asked:** "this has to be the case for all jobs - monthly/weekly etc. for tn, oa, ipo
all, can u pls chk if this is incorporated".

**Why.** Cron and the in-app scheduler fire "mon-fri" regardless of exchange holidays. On a closed
day a book job either sends orders the exchange refuses, records a fake flat day in a value curve,
or repeats the previous session's decisions on stale data.

## The audit (before this change)

| Book | Job | Holiday-safe before? | Now |
|---|---|---|---|
| True North | 14:45 month-end rebalance | yes — `_is_last_trading_day()` asks the calendar | unchanged |
| True North | 15:05 stops + weekly gate | no — fixed earlier on 13-Sep | guarded |
| True North | 15:25 backstop, startup catch-up, 15:35 report, 15:40 monthly report | yes | unchanged |
| True North | 09:20 broker reconcile | no — read-only, but could alert on a closed day | guarded in code |
| True North | page bakes (`gen_momentum_*`) | display only | left alone |
| Open Alpha | 18:50 entry + exit AMOs | **no** — its duplicate check reads the broker's day-scoped order book, so a holiday re-run could re-place yesterday's order | wrapped |
| Open Alpha | 15:18 check, reconciles, marks | no — the 18:46 mark appends a daily value point, a fake flat day on a holiday | wrapped |
| IPO Base | 09:20 order job | **no** — checked weekday and time only | wrapped + guarded in code |
| IPO Base | 18:45 nightly cycle, reconciles, marks | no | wrapped |
| Cash park | 15:10 / 15:11 | **no** — weekday and time only | wrapped + guarded in code |
| Rebalance one-off | Tue 15-Sep 09:45 | yes — gated on 13-Sep | unchanged |
| Book-curve builder, payout declaration | — | safe: read-only / quarter-window idempotent | left alone |

## The change

1. `scripts/on_trading_day.sh` — runs its command only on an NSE trading day; answer cached per
   day; FAILS OPEN (runs, logged) if the calendar cannot be read.
2. Crontab: the wrapper in front of the 13 Open Alpha, IPO, executor and cash-park jobs.
3. In code, so manual runs are covered too: `equity_executor.py --arm`, `cash_park._armable()`,
   True North's in-app 09:20 reconcile.

No trading rule changed.

## Known limit

`services/trading_calendar.py` treats every weekday as a trading day when a year's holiday file is
missing. **`config/nse_holidays_2027.json` must exist before 1-Jan-2027** — review due 15-Dec-2026.

## Crash recovery

- Undo the crontab: the backup path is printed in the deploy log (`/tmp/ct.bak.holiday.*`);
  `crontab <backup>`.
- Undo the code: `git revert` this commit; True North's part loads at the 09:00 restart.
- See what was skipped: grep `skipped - ` in `/tmp/oa_*.log`, `/tmp/ipo_*.log`,
  `/tmp/equity_executor.log`, `/tmp/cash_park.log`.
