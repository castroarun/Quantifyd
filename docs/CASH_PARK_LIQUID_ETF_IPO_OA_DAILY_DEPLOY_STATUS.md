# Idle-Cash Park — IPO Base and Open Alpha idle cash into a liquid ETF (CASHIETF)

**STATUS: DEPLOYED SWITCHED OFF — live test on IPO Base due 15-Sep-2026**

## 1. The ask

**What Arun asked (13-Sep-2026):** "can we do arb fund for ipo cash?" → after the broker limit was
found, chose **"Automated liquid-ETF sweep (Recommended)"**.

**What is being built:** IPO Base and Open Alpha · Base Age hold their uninvested money as plain
cash, earning nothing, while every backtest credits 5.2% post-tax on idle cash. After the
14-Sep rebalance that is roughly ₹2.8L idle in IPO and ₹2.3L in Open Alpha. This parks the part
neither book can need before the next release point into CASHIETF, the liquid ETF True North
already sweeps into.

**Why not the arbitrage fund Arun first asked for:** Kite Connect cannot place mutual-fund orders
(official docs: "Order placement can't be done, as order placement needs payment from the user's
bank account"). Only exchange-traded instruments can be automated. A liquid ETF earns about 3.5%
post-tax at a 30% slab against about 5.2% for an arbitrage fund, but it can be bought and sold by
the system and its sale proceeds can be spent the same day.

## 2. The base — rules

**The rule everything else serves: never park money a buy could need before the next release.**
Parking must change the book's yield, never its trades.

| | IPO Base | Open Alpha · Base Age |
|---|---|---|
| When buys are placed | 09:20 by `services/equity_executor.py` | 18:50 as AMOs, executed at the next open |
| Can a sale fund today's buys? | Yes — the executor sells ETF first, at 09:20, same session | No — AMOs execute at 09:15, before any sale can settle |
| Reserve kept as plain cash | a small buffer (₹10,000) | (free slots + 1 swap) × 6.25% × marked NAV × 1.10 |
| Park run | 15:10 weekdays | 15:10 weekdays |
| Release | at 09:20 on demand, before IPO orders | at 15:10 if cash is below the reserve |

**Ledger model.** Each book's `cash` keeps meaning ALL uninvested money, parked or not, at cost.
A new `park` block records units, cost and last price. So:
- **Free cash** = `cash` − parked cost. Anything that sends an order to the broker uses free cash.
- **NAV** = positions + `cash` + (units × price − parked cost). The gain is the only new term.
- **Sizing** is unchanged, because `cash` still includes parked money — so parking can never
  shrink a position.

**Orders.** CNC, marketable LIMIT (Kite rejects bare MARKET on equities via API), tick-rounded,
paced through the executor's `send_order`, tagged `IPO-PARK` / `OA-PARK` so neither book's own
fill reconciliation picks them up. Each order waits up to 90s for COMPLETE; an unfilled order is
cancelled and alerted, and the ledger is only updated for what actually filled.

**Safety.**
- `backtest_data/cash_park.json` switches each book on or off. Deployed OFF for both.
- `backtest_data/executor_kill.flag` stops it, the same switch as the executor.
- Arming only on weekdays 09:20–15:20 IST.
- At most one park run per book per day (ledger `backtest_data/cash_park_orders.json`).
- `max_order` caps an order's size for the live test.
- A reconcile checks that the units the three books claim (True North's sweep plus these two)
  never exceed what the broker holds.

## 3. Plan

1. Build `services/cash_park.py` (plan, park, release, status, reconcile).
2. IPO engine: NAV includes the park gain at all three valuation points; the page gets a park block;
   a Capital Desk withdrawal cannot be applied against parked money.
3. Executor: release ETF before placing IPO buys when free cash is short; deduct each placed order.
4. Open Alpha engine: NAV includes the park gain in `mark()` and `ui_only()`; the page gets a park
   block; a withdrawal is checked against free cash.
5. Dry runs on the live states. Deploy with both books OFF.
6. **Live test after the 14-Sep rebalance lands:** IPO only, `max_order` ₹10,000 — one park, one
   release through the executor path — then check fills, ledger, NAV and reconcile.
7. Remove the cap, switch IPO on; switch Open Alpha on once it has a non-zero parkable amount.

## 4. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-13 | Broker limit found; Arun chose the liquid-ETF sweep | Kite docs; memory `kite-mf-orders-not-via-api` |
| 2026-09-13 | Status doc written before any code deployed | — |
| 2026-09-13 | Module and integration deployed SWITCHED OFF | ipo_paper, oa_real, equity_executor patched; line endings kept; all four import |
| 2026-09-13 | Tests 21 / 21 pass | ledger maths, plans, sell accounting, resting-order hold-back, executor release path on a fake broker |
| 2026-09-13 | Read-only dry runs on the live books | IPO would park ₹10,000 (capped); Open Alpha nothing (reserve ₹2,56,191 > free ₹1,88,698); both pages carry the park block; NAVs unchanged; state files byte-identical |
| 2026-09-13 | Found and fixed before any job ran | the 15:10 run would have tried to park cash the broker holds against resting IPO buy-stops; it now subtracts the book's open BUY orders |
| 2026-09-13 | 15:10 / 15:11 park jobs installed | gated by the switches, so they only log a plan until a book is switched on |

## 5. Crash recovery

- **Stop everything:** `touch /home/arun/quantifyd/backtest_data/executor_kill.flag`
  (this also stops the 09:20 IPO executor).
- **Switch one book off:** set its entry to `false` under `enabled` in `backtest_data/cash_park.json`.
- **See what is parked:** `venv/bin/python services/cash_park.py status`
- **Check units against the broker:** `venv/bin/python services/cash_park.py reconcile`
- **Unpark everything for a book during market hours:**
  `venv/bin/python services/cash_park.py release --book ipo-base --amount 99999999 --arm`
- Do not hand-edit the `park` block in a book's state file: the book's NAV and free cash read it.

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `services/cash_park.py` | the sweep | yes |
| `backtest_data/cash_park.json` | per-book switch and limits | yes |
| `backtest_data/cash_park_orders.json` | once-a-day ledger | yes |
| `services/ipo_paper.py`, `services/oa_real.py`, `services/equity_executor.py` | integration | yes |

## 7. Findings

- Kite Connect cannot place mutual-fund orders; an arbitrage-fund tier could only ever be manual.
- Arbitrage fund scores are kept at `backtest_data/arb_fund_scores_20260913.json` for the 15-Dec-2026
  cash-yield review: Tata 7.47% and Kotak 7.46% over three years, no losing month.
- **Open Alpha will park little at first.** With 11 of 16 slots held it must keep cash for 5 free
  slots plus a swap, which is more than it holds. Parking becomes material as the book fills.
