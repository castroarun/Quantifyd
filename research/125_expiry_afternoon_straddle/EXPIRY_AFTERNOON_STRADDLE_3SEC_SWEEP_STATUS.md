# Expiry-Afternoon Straddle — Fine Time-Grid on DTE0 (NIFTY Tue + SENSEX Thu), 3-sec Real Chain

STATUS: DONE (12:05 IST) - see results/RESULTS.md · research/125 · started 2026-08-25 ~11:55 IST (NIFTY expiry day — results wanted before ~13:00)

## 1. The Ask

**What you asked:** An old AlgoTest backtest (BANKNIFTY monthly expiry, sell strangles at
the ~₹10-premium strikes, entry ~13:45, exit 15:00, SL 250%/leg, expiry days only,
2018–2022) made +₹34.5L over 209 trades, 72% win, ret/DD 4.63. Test the idea with
**straddles** on our real recorded data; adjust/optimize the timeslots; use the
second-level price data to measure calm/variability; goal: identify a **second-half
entry→exit for expiry days** (today = NIFTY expiry).

**What we're actually testing:** On the ~17 recorded NIFTY Tuesdays and ~17 SENSEX
Thursdays (DTE0 only), which afternoon (entry, exit, SL) cell for a spot-ATM short
straddle has the best net-of-cost mean/day and ratio? Plus: a minute-level calmness
map of the expiry afternoon (premium variability by 15-min bucket) to explain WHY a
slot wins. The AlgoTest reference cell (13:45→15:00) is printed explicitly for
comparison.

## 2. The Base

- Signal: none — pure time-slot construction. SELL spot-ATM straddle at entry time
  (strike = ATM from live spot at entry minute; sec-19 validated spot-centering).
- Data: options_data.db `option_chain` raw ~3-sec snapshots (2026-04-20 → today),
  the no-5-min rule satisfied at source. Frozen-chain holiday guard: a day with <50
  distinct spot prints is rejected (r/120s rule; 2026-05-28 was a poisoned SENSEX Thu).
- Exit: combined-premium SL with the accepted dwell mechanic (2 consecutive breaching
  snaps → market exit next snap), else time exit.
- Costs: r/123 model — 0.50 pt slippage/leg-side × 4 sides + ₹30/leg-side/lot bundled
  brokerage. NIFTY 10 lots (qty 650) → ₹2,500/round-trip; SENSEX 5 lots (qty 100) → ₹800.
- Sizes on every figure: NIFTY basis 10L, SENSEX basis 5L (also stated per-lot).
- Differences vs the AlgoTest original, stated up front: venue (BANKNIFTY monthly ⇒
  NIFTY/SENSEX weekly DTE0), instrument (₹10-premium strangle ⇒ ATM straddle), era
  (2018-22 ⇒ 2026), n (209 expiry days ⇒ ~17/venue). This is a *construction* test on
  current data, not a replication.

## 3. Plan / grid

- Universe of days: DTE0 only — NIFTY (Tue), SENSEX (Thu), all recorded.
- ENTRIES: 12:00 12:30 12:45 13:00 13:15 13:30 13:45 14:00 14:15 14:30
- EXITS: 14:00 14:15 14:30 14:45 15:00 15:10 15:20 (min hold 30 min)
- SL: 20 / 25 / 30 / 40 / none (none carries the live 50% disaster backstop)
- ≈ 2 venues × ~55 valid (entry,exit) pairs × 5 SLs ≈ 550 cells, n≥6 to report.
- Calm map: per-15-min bucket (12:00→15:20), mean |1-min Δcombined-premium| as % of
  13:00 premium + bucket drift, averaged across expiry days, per venue.

## 4. Success criterion

Rank by ratio (total/maxDD) among cells with n≥8; a slot must beat the corresponding
full-afternoon hold AND survive the 1.00-pt slippage sensitivity to be called a
candidate. Deployment (if any) = user decision, paper-first per playbook unless user
directs otherwise.

## 5. Status log

| Time | Event |
|---|---|
| 11:55 | STATUS written, runner uploading |

## 6. Crash recovery

- Runner: `research/125_expiry_afternoon_straddle/scripts/expiry_afternoon_sweep.py` (VPS)
- Launch: `cd /home/arun/quantifyd && venv/bin/python3 research/125_expiry_afternoon_straddle/scripts/expiry_afternoon_sweep.py > /tmp/r125.log 2>&1 &`
- Progress: `tail /tmp/r125.log` · Output: `research/125_expiry_afternoon_straddle/results/expiry_afternoon.json` (+ printed top tables in the log)
- Idempotent: rerunning overwrites the JSON; no partial-state risk.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| scripts/expiry_afternoon_sweep.py | runner | yes |
| results/expiry_afternoon.json | all cells + calm map | yes (small) |
| this STATUS | live doc | yes |

## 8. Findings

(pending)
