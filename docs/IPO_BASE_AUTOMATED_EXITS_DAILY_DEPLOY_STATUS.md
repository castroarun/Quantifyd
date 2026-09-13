# IPO Base — exits placed automatically, booked only when the broker fills

**STATUS: DEPLOYED 13-Sep-2026 (Sunday) — first live evening Tue 15-Sep (Mon 14-Sep is a holiday)**

## The ask

**What Arun asked:** "IPO Base exits. Nothing places an IPO sell ... pls automate this".

## What was wrong

In live mode, when a stop, target or trail fired on the close, the 18:45 run booked the sale at
that close — removed the position, credited the cash, wrote the trade — and only sent an alert
ending "Place it". No order was placed. The reconcile read only BUY orders, so a late or missed
manual sale was never caught: the book showed the name gone while the account still held it with
no stop watching it, and the next morning's order job treated that cash as free.

## What it does now (LIVE; PAPER is unchanged)

| Step | When | What |
|---|---|---|
| Decide | 18:45 nightly run | a fired exit marks the position `exit_due` and KEEPS it |
| Place | same run, after the day's fills are applied | one after-market SELL for the next open, tag `IPO-EXIT` — MARKET first, LIMIT 2% under the signal close if refused; never a second SELL while one rests |
| Book | 09:20 (the order job reconciles first), 09:35 / 11:35 / 13:35 / 15:35 reconciles, 18:45 run | the sale is recorded only when the broker shows the IPO-EXIT SELL complete, at the broker's price, with the signal close and the slippage |
| Retry | next evening run | a refused, rejected or lapsed SELL is placed again; a refusal raises a CRITICAL alert |
| Slots | 18:45 arming | an exiting position counts as a free slot, as in the backtest |

## Known deviation, being measured

The backtest sells AT the signal close. A decision taken on the official close at 18:45 can only
sell at the next open — which is also what the manual process did. A research run measures what the
overnight gap costs, against a bar pre-registered in its STATUS doc (material if more than 1.0pp of
CAGR or 0.10 Calmar at the median on 20 of 30 paired seeds).

## Evidence

Unit tests with a fake broker: live marks and keeps, paper books; one SELL per exiting name with the
IPO-EXIT tag and the real tick; never twice a day; a resting sell is not re-placed; a refusal is
CRITICAL; a fill books at the broker price with slippage; a partial fill keeps the rest exiting; a
fill for a name not held is refused; exiting positions free their slot. A dry run of the full nightly
cycle on the live book.

## Crash recovery

- Stop all IPO orders: `touch backtest_data/executor_kill.flag` stops the 09:20 buys. Exit sells are
  placed by the 18:45 run: to stop those, comment out the 18:45 `services/ipo_paper.py` cron line
  (crontab backup first).
- Revert: `cp /tmp/ipo_paper.pre_exits.py services/ipo_paper.py` and
  `cp /tmp/equity_executor.pre_exits.py services/equity_executor.py`.
- An exiting position in `backtest_data/ipo_paper_state.json` carries `exit_due` with the orders
  placed for it. Do not hand-edit it while an IPO-EXIT SELL is resting.
