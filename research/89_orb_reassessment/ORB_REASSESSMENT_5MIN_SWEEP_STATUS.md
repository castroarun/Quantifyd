# ORB Reassessment — Live-Book Autopsy + Decay Characterization + Revival Sweep (5-min, F&O)

STATUS: RUNNING (battery launched)

## 1. The Ask

**What you asked:** "Earlier we did ORB study and live as well, but due to
deteriorating performances in live we stopped it — please reassess this
comprehensively, test and optimize and give me the results."

**What we're actually answering, in three questions:**
1. **Autopsy:** why did the live book (₹3L, Apr-May 2026, −₹16.7k in ~6
   trading days) lose — signal decay, implementation deviation, or ops?
2. **Decay diagnosis:** is the validated research signal (gap-up + OR15
   breakout LONG, ≤4-day hold — r/81's "SIGNAL, decaying") in a drought or
   dead? Per-quarter trace 2015→2026 + regime splits.
3. **Revival sweep:** does ANY disciplined variant retain a recent edge —
   and under what pre-stated criteria would ORB be worth re-arming (paper
   first)?

## 2. Constraints & honesty box (binding)

- **The ORB family's OOS (2024-26) was CONSUMED on 2026-07-16 (r/81).**
  Every cell in this study that touches 2024-26 is therefore IN-SAMPLE BY
  CONSTRUCTION. No configuration found here can be called validated — the
  only path to live for any candidate is a PAPER-FORWARD soak.
- Costs: futures-proxy incl. 3bp slippage (CFG3 of r/81). Engine: r/81
  `engine/` (32 unit assertions), data = VPS market_data.db (current).
- Known priors from r/81 (not re-litigated): shorts lose; equal-notional
  sizing; trade-level t ≠ book; book FAILED OOS.

## 3. Live-book forensics (phase A — findings already in hand)

46 closed trades, 2026-04-27 → 2026-05-05, net **−₹16,713** on ₹3L:
- **Shorts −₹18,419 (20 trades) vs longs +₹1,706 (26)** — the live engine
  traded BOTH directions; the validated research signal was LONG-ONLY.
- **17 trades exited as RECONCILED_KITE_FLAT (−₹22,214)** — operational
  reconciliation exits, not strategy exits.
- V9T_LOCK50_BE trailing exits: 20 trades −₹18k; TARGET_HIT only 4 (+₹16.9k).
- Live horizon was INTRADAY (15:18 squareoff); research signal held ≤4 DAYS.
- => The live book was a materially different system from the validated one:
  added shorts, compressed the horizon, added RSI/VWAP/CPR filters and a
  trailing-BE exit. Phase C quantifies what each deviation cost.

## 4. Plan — battery (pre-registered)

**B. Decay trace (1 locked config, no optimization):** W12 gap0.25% long
ts4 on 77 F&O + NIFTY: net bps per QUARTER 2015→2026-07; regime split by
NIFTY>50DMA and by INDIAVIX tercile. Drought-vs-death verdict criteria:
"drought" if negative stretches of comparable depth/length exist pre-2024
with recovery; "decay" if 2024-26 is unprecedented and monotone.

**C1. Live-replica autopsy (4 cells, window 2024-01→2026-07):**
a) live-replica: OR15 both-dirs, intraday EOD exit, 1.5R target, OR-opposite
   SL, RSI15 filter (>60 long / <40 short) — approximation of prod rules
b) same but LONG-only
c) live-replica long-only WITHOUT RSI filter
d) research config (W12 gap0.25 long ts4) same window
=> decomposes live loss into: shorts / intraday horizon / filters / signal.

**C2. Revival grid (36 cells, reported per period 2015-21 / 2022-23 /
2024-26):** W ∈ {6,12,18} × gap ∈ {0.25%, 0.40%} × exit ∈ {ts2, ts4,
intraday-EOD} × LONG only, stop = OR-low (engine default) × cost CFG3.
Revival-candidate gate (pre-stated): net ≥ +10bps AND t ≥ 2.0 in **each** of
the three periods separately — i.e. the config must never have died, not be
resurrected by fitting. Any passer => 90-day paper soak proposal, NOT live.

Ledger: 1 + 4 + 36 = 41 cells (program running total 350 + 41 = 391).

## 5. Status / event log

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-21 ~22:25 IST | Forensics done; STATUS written | live DB: 46 trades dissected |
| 2026-07-21 (launch row in log) | Battery launched on VPS | results/orb_battery.log |

## 6. Crash recovery

- Progress: `tail research/89_orb_reassessment/results/orb_battery.log`
- Alive: `ps -eo args | grep '[r]un_orb_battery'`
- Resume: rerun `venv/bin/python3 research/89_orb_reassessment/scripts/run_orb_battery.py`
  (incremental CSV, skips done cells).

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `scripts/run_orb_battery.py` | phases B, C1, C2 | yes |
| `results/orb_battery.csv` / `.log` | cells / log | yes |
| `results/decay_quarters.csv` | phase B per-quarter trace | yes |
| `results/RESULTS.md` | verdict + recommendation | yes (after) |

## 8. Findings

(populated after run; phase-A forensics above)
