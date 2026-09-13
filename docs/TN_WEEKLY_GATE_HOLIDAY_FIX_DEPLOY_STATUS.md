# True North — weekly gate on the last TRADING day of the week (holiday fix)

**STATUS: DEPLOYED 13-Sep-2026 (Sunday) — takes effect at the next app restart (09:00 weekday pre-open)**

## The ask

**What Arun asked:** "pls fix this" — after the review of True North's weekly gate found that it
ignored exchange holidays.

**What was wrong.** The weekly gate check runs inside the 15:05 end-of-day job, only when
`_is_last_trading_day_of_week()` says so. That helper skipped Saturdays and Sundays but not NSE
holidays, and the end-of-day job itself had no holiday guard. In a week whose Friday is a holiday:
- Thursday, the real last session, was judged "not last", so no gate check ran on it;
- the check then fired on the holiday Friday, where every sell or re-entry order is refused.

The ledger stayed correct, because True North records nothing unless the broker fills. But that
week's gate action — a liquidation or a re-entry — slipped a full week. In 2026 that affects the
weeks of Friday 2-Oct (Gandhi Jayanti) and Friday 25-Dec (Christmas).

## The base — what changed

| | Before | After |
|---|---|---|
| Last trading day of the week | next weekday in a different ISO week | today is a trading day AND the next TRADING day is in a different ISO (year, week) |
| 15:05 job on an NSE holiday | ran; its orders were refused | skipped, logged |
| If the calendar cannot be read | — | falls back to the old weekend-only rule / runs as before, logged |

No rule changed: same NIFTYBEES 100-day gate, same exits, same order of steps. Same pattern as
`_is_last_trading_day()`, which already asks `services/trading_calendar.py` for month-end.

## Evidence

Tests on the real 2026 calendar, every step stubbed so nothing trades:
- Thu 1-Oct and Thu 24-Dec are the last trading day of their weeks; the holiday Fridays are not.
- On Fri 2-Oct and Mon 14-Sep nothing in the end-of-day job runs.
- On Thu 1-Oct and Fri 18-Sep the stops and the weekly gate both run; on Wed 16-Sep only the stops.

## Deploy

True North runs inside the `quantifyd` web app, so the code loads at restart. No manual restart
was done. The existing weekday 09:00 `scripts/preopen_restart.sh` loads it, well before the first
affected week (2-Oct). The week of 18-Sep has a normal Friday, so the old code behaves correctly
until then.

## Crash recovery

Revert: `cp /tmp/momentum_paper.pre_weekfix.py services/momentum_paper.py`, then let the 09:00
restart load it (or restart after 15:40 on a trading day). Check the effect on a holiday week in
`journalctl -u quantifyd` for `[MP] EOD skipped` and `[MP] GATE`.

## Status log

| Date/time (IST) | Event |
|---|---|
| 2026-09-13 | Defect found while confirming True North's gate schedule |
| 2026-09-13 | Patched, tested, deployed (loads at the next 09:00 restart) |
