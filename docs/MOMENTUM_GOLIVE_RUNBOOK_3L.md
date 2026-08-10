# MOMENTUM GO-LIVE RUNBOOK — Rs3L in the shared account

Order matters. Every step is deliberate; nothing here is automatic.

## Why a reset is mandatory
The paper book holds 6 positions worth ~Rs15.2L on a Rs20L paper capital. Those shares DO NOT exist at
the broker. Carrying that ledger into a Rs3L live account would make the book believe it owns stock it
does not — and every Donchian stop would try to sell shares that, if they exist at all, are Arun's
personal holdings. The reset re-bases the ledger to Rs3L with zero positions.

## Step 1 — AFTER 15:30 IST (market closed)
    ssh arun@94.136.185.54
    cd /home/arun/quantifyd
    TZ=Asia/Kolkata date          # CONFIRM it is past 15:30 and a weekday
    sudo systemctl restart quantifyd     # loads Rs3L + shared-account guards

Without this restart the running process still has capital=Rs20L and NO sell guard, no capital fence
and no clash detection. Arming before it is dangerous.

## Step 2 — reset the ledger (still after close)
    ./venv/bin/python3 scripts/golive_reset.py            # dry run, shows the plan
    ./venv/bin/python3 scripts/golive_reset.py --confirm  # applies it
Archives the paper history to backtest_data/momentum_paper_archive_<date>.json first.
Leaves the book in PAPER mode, unarmed.

## Step 3 — verify before arming
    curl -s localhost:5000/api/momentum-paper/state | python3 -m json.tool | head -30
Expect: capital 300000, cash 300000, n_holdings 0, mode PAPER, hedge null.

## Step 4 — arm (two keys, both required)
    curl -X POST localhost:5000/api/momentum-paper/toggle-mode \
      -H 'Content-Type: application/json' \
      -d '{"live":true,"arm":true,"capital":300000}'
Neither key alone permits an order.

## Step 5 — seed DURING market hours (next trading day, ~15:00 IST)
    curl -X POST localhost:5000/api/momentum-paper/seed
Places 8 real CNC MARKET BUY orders of ~Rs36,375 each (3% cash reserve held back).
Watch: journalctl -u quantifyd -f | grep -E "MP-LIVE|MP-SHARED"

## What runs by itself after that
| Time | Job |
|---|---|
| 09:20 | reconcile book vs broker; alerts only if book > broker (broker > book is expected) |
| 14:45 | monthly re-rank (last trading day of month only) |
| 15:15 | Donchian stops + weekly gate |
| 15:35 | EOD email |
| 15:40 | monthly report (month-end) |

## Kill switch
    curl -X POST localhost:5000/api/momentum-paper/kill-switch
Flips to PAPER and clears both keys. Positions are LEFT ALONE — square off manually if needed.

## Guards active in a shared account
- Sell guard: never sells more than the system's own qty; refuses + high alert if broker < book
- Capital fence: refuses any order taking the book past Rs3L
- Clash flag: emails when buying a name already held personally (merged broker line)
- Reconcile: broker > book is expected; only book > broker alerts

## Deliberately OFF at this size
- Put hedge: one NIFTY lot is Rs16L notional = 5.3x a Rs3L book. Needs ~Rs8L equity. The book uses the
  cash-exit gate instead, which beat the hedge over the full cycle anyway.
- MTF: starts on cash (CNC). At Rs3L, 1.0x sizes all 8 slots cleanly and pays no interest (~14.6%/yr,
  which would be 3.4% of the book at 1.3x). Switch order_product to "MTF" after a clean month.
- LIQUIDCASE sweep: built but off; idle cash earns nothing in live until it is enabled and tested.
