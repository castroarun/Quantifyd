# ATM2 After the Rupee Stop — Re-center at the New ATM, or Close for the Day?

STATUS: **DONE** (2026-09-01) — verdict **NO EDGE: one-and-done stands**

Research number 141. Venues: NIFTY ATM2 (incumbent, one-and-done) + SENSEX ATM2 (live re-center A/B).

---

## 1. The Ask

**What Arun asked (2026-09-01):** he remembered ATM2 as a *rolling* system — stop out, re-center at
the new ATM, go again — was surprised it did not roll on 2026-09-01, and wants the question settled
on numbers rather than memory.

**What we are actually testing:** research/96 (2026-07-28) replaced ATM2's 0.4% spot-move stop with a
DTE-agnostic **₹2,500/lot rupee MTM stop**. That substitution is properly evidenced — r/96's table
compares rupee-stop vs move-stop vs premium-multiple with per-trade P&L, tails and win rates. But the
*same change set* also flipped `move_stop_reenter: True -> False`, and **that half was never measured**.
It rests on one unmeasured line, r/96 read 4: *"Re-center adds churn on trending/expiry days ->
one-and-done"*, and the methodology note *"one-and-done modeled (matches the deploy)"*.

The precise question:

> After ATM2's ₹2,500/lot rupee stop fires, is closing for the day better or worse than re-entering a
> fresh ATM straddle at the new forward-snapped strike — and if re-entering wins, how many times, and
> with what cooldown?

This is a **null-defending** test. Three prior studies (r/54 "re-entry hurts", r/60 the 2026-06-08
churn incident, r/123 short-horizon re-entry dies on its own round-trip cost) all point at
one-and-done. If we reproduce that, it is a clean fourth reproduction. If we contradict it, that is
the finding and it gets stated loudly.

---

## 2. The Base — exactly what is being replayed

### Incumbent (what runs today, NIFTY)

`NAS_ATM2_DEFAULTS` / `NAS_916_ATM2_DEFAULTS` (config.py ~L483):

- `rupee_stop_per_lot: 2500`, `move_stop_pct: 0`, `move_stop_reenter: False`, `re_enter_on_sl: False`
- per-leg 30% SL gated OFF while the rupee stop is active
- `reentry_cooldown_min: 15` (research/60 churn-breaker, currently inert on ATM2 because nothing re-enters)
- entry 09:16 ATM straddle, squareoff 15:15, NIFTY lot 65, live 2 lots

### The live A/B already running (SENSEX)

`SENSEX_ATM2_DEFAULTS` (config.py ~L564), scope-fixed 2026-07-29 (commit `c95f10a`) because the rupee
stop was calibrated on NIFTY chains only:

- `rupee_stop_per_lot: 0`, `move_stop_pct: 0.004`, **`move_stop_reenter: True`**, `strike_interval: 100`
- SENSEX ATM2 has therefore been running the **re-center arm with real recorded fills since 2026-07-22**.
  That book is direct evidence and is reported as its own section.

### Replay mechanics (this study)

- Data: `backtest_data/options_data.db :: option_chain`, **1-minute**, 2026-04-20 -> 2026-09-01, both venues.
- Guards (r/120/121/122 lineage): reject frozen-chain holidays via <50 distinct spot prints/day
  (catches 2026-05-01 / 05-28 / 06-26); reject partial sessions (last snapshot < 15:15).
- **Strike selection = FORWARD SNAP** — `K = round(F/step)*step` where `F = K_ref + (CE - PE)` read at
  the spot-nearest strike with a +/-1-step cross-check (`common132.read_forward`). NOT `round(spot/step)`.
  research/132 showed spot-rounding mis-strikes 36% of NIFTY minutes and 50% of SENSEX; each re-center
  would otherwise inherit a *fresh accidental delta*, which would contaminate this test badly. Live code
  was fixed for this on 2026-08-27; the replay matches.
- Lots: NIFTY lot 65, SENSEX lot 20 (`option_chain.lot_size` is WRONG — do not read it).
  Headline reported **per lot**; r/96 comparison rows reported at **2 lots** to match its table.
- Costs: the **MEASURED outcome-aware model** from research/122 `stage_a_alldays.cost_per_lot()` —
  exact Zerodha F&O rate card + measured slippage **entry 0 pt / time-exit +0.178 pt / forced-stop
  +6.548 pt per leg-side**. Every re-center pays a *full extra round trip whose exit is a forced stop*,
  so churn is priced, not asserted.
- DTE-era labelling mandatory: trading-DTE (`trading_dte`) as primary, calendar DTE reported alongside
  for r/96 comparability (r/96 bucketed on calendar DTE<=1).

---

## 3. Plan — arms and grid

| Arm | Stop | After the stop |
|---|---|---|
| `ONE_AND_DONE` | ₹2,500/lot rupee MTM | **done for the day** (the incumbent null) |
| `RECENTER_1` | ₹2,500/lot | re-enter fresh ATM straddle, max 1 re-center |
| `RECENTER_2` | ₹2,500/lot | max 2 |
| `RECENTER_3` | ₹2,500/lot | max 3 |
| `RECENTER_5` | ₹2,500/lot | max 5 (the pre-July cap) |
| `RECENTER_2_CD15` | ₹2,500/lot | max 2 + 15-min cooldown (r/60 churn-breaker shape) |
| `RECENTER_3_CD15` | ₹2,500/lot | max 3 + 15-min cooldown |
| `RECENTER_5_CD15` | ₹2,500/lot | max 5 + 15-min cooldown |
| `MOVESTOP_ONE` | 0.4% spot move | done for the day |
| `MOVESTOP_RECENTER` | 0.4% spot move | re-center, max 5 (the **pre-July live behaviour**) |
| `MOVESTOP_RC1` | 0.4% spot move | re-center, max 1 |
| `MOVESTOP_RC_CD15` | 0.4% spot move | re-center max 5 + 15-min cooldown (**the arm SENSEX ATM2 runs live today**) |
| `NOSTOP_HOLD` | none | reference: hold to 15:15 |

Re-entry window closes 14:30 (`ENTRY_WIN_END`, matching the live engine). Base re-center requires the
new snapped strike to DIFFER from the one just closed (live semantics + avoids the r/60 same-strike
close/reopen loop); a no-guard sensitivity is reported.

Grid: 14 arms x 2 venues x ~90 days x 1-min walk. Cheap — single pass, niced.

### Gates the winner must clear

1. Beat `ONE_AND_DONE` **net of the measured cost model**, not gross.
2. Sit on a **plateau** of neighbouring n (not one lucky value).
3. Survive a **family-wise haircut** over the 13 comparisons per venue.
4. Hold on an **OOS split** — r/96's own day set (2026-04-20..07-28) as in-sample, 07-29..09-01 as the
   genuinely out-of-sample period *after* the deploy decision was made.
5. Be checked **per DTE** — the original justification was "churns on trending/**expiry** days", so
   DTE0/DTE1 is the specific claim under test.

### Reconciliation gate (before any interpretation)

Reproduce r/96's headline on its own 68-day NIFTY set, calendar DTE<=1, 2 lots, spot-rounded strike
(r/96's own method), one-and-done:

- **RUPEE ₹2,500/lot = +2,153/trade, -6,972 worst, 69% win**
- **0.4% move-stop = +1,386/trade, -6,887 worst, 62% win**

If these do not reproduce, STOP and report that instead.

---

## 4. Status — live event log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-01 ~12:05 | Folder + STATUS-MD written before compute | research/141_atm2_recenter_ab |
| 2026-09-01 ~12:12 | Reconciliation gate run (`recon96.py`, 68 NIFTY days ≤ 07-28) | tails, win rates and the entire far-DTE bucket match r/96 to the rupee; near-expiry mean +121/+122 high |
| 2026-09-01 ~12:20 | Main sweep v1 done (12 arms × 2 venues × 88 days) | `results/arms_daily.csv`, 2,112 rows |
| 2026-09-01 ~12:26 | Residual chased: `recon96_daydiff.py` re-ran r/96's OWN engine day-by-day | **0 of 68 days differ.** Residual is 2026-07-28 alone — r/96's own run date, calendar DTE0, priced ~27 pt before the true 15:15 close. **GATE: PASS** |
| 2026-09-01 ~12:32 | Live-arm ledgers read (SENSEX ATM2 + both pre-r/96 NIFTY ATM2 books) | 42 real re-centers pooled |
| 2026-09-01 ~12:38 | Gap found: the replay's `MOVESTOP_RECENTER` lacked the live 15-min cooldown | added `MOVESTOP_RC_CD15` + `MOVESTOP_RC1`, re-ran the full sweep (14 arms, 2,464 rows) |
| 2026-09-01 ~12:47 | Aggregation + cycle-mix + RESULTS.md written | **VERDICT: NO EDGE — one-and-done stands** |

## 5. Crash recovery

Everything runs on the VPS at `/home/arun/quantifyd`, venv `venv/bin/python3`, all DBs opened
READ-ONLY (`file:...?mode=ro`). Nothing in this study writes outside `research/141_atm2_recenter_ab/results/`.

- Reconciliation: `venv/bin/python3 research/141_atm2_recenter_ab/scripts/recon96.py`
  → stdout + `results/recon96.txt`. ~4 min.
- Day-by-day diff vs r/96's own engine:
  `venv/bin/python3 research/141_atm2_recenter_ab/scripts/recon96_daydiff.py`
  → `results/recon96_daydiff.txt`. ~8 min (loads pandas per day).
- Main sweep: `nice -n 15 venv/bin/python3 research/141_atm2_recenter_ab/scripts/run_arms.py`
  → `results/arms_daily.csv`, flushed per day; safe to kill and re-run (rewrites from scratch, ~4 min).
- Aggregation: `venv/bin/python3 research/141_atm2_recenter_ab/scripts/aggregate.py`
  → `results/aggregate.md`. Reads only `arms_daily.csv`.
- Mechanism: `venv/bin/python3 research/141_atm2_recenter_ab/scripts/cycle_mix.py`
  → `results/cycle_mix.md`.
- Live arm: `venv/bin/python3 research/141_atm2_recenter_ab/scripts/sensex_live_arm.py`
  → `results/live_arm.md`. Reads the `*_trading.db` books READ-ONLY.

Safe to inspect: everything under `results/`. Do NOT touch anything under `services/`, `config.py`,
or any `backtest_data/*.db` (read-only by construction).

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `ATM2_RECENTER_VS_ONEANDDONE_1MIN_SWEEP_STATUS.md` | This file | yes |
| `scripts/engine141.py` | Forward-snap replay engine + the 14-arm grid | yes |
| `scripts/recon96.py` | research/96 reconciliation gate | yes |
| `scripts/recon96_daydiff.py` | Day-by-day diff against r/96's own engine | yes |
| `scripts/run_arms.py` | Main sweep runner | yes |
| `scripts/aggregate.py` | Per-arm tables, churn cost, per-DTE, plateau, Holm, OOS | yes |
| `scripts/cycle_mix.py` | Re-centered-straddle outcome mix (the mechanism) | yes |
| `scripts/sensex_live_arm.py` | Live re-center evidence from the recorded books | yes |
| `results/arms_daily.csv` | Per venue × day × arm rows (2,464) | yes |
| `results/{recon96,recon96_daydiff}.txt` | Gate output | yes |
| `results/{aggregate,cycle_mix,live_arm}.md` | Generated tables | yes |
| `results/RESULTS.md` | Final findings + verdict | yes |

## 7. Findings — **NO EDGE. One-and-done stands.**

Full write-up in `results/RESULTS.md`. Headline:

1. **Reconciliation PASSED.** Our replica reproduces r/96's engine on **0/68 differing days**; the
   published tails (−6,972 / −6,887), win rates (69% / 62%) and the whole far-DTE bucket match to
   the rupee. The +121 residual on the near-expiry mean is 2026-07-28 alone — r/96's own run date,
   priced mid-session ~27 premium points before the true expiry-day close.
2. **No re-center count beats the incumbent on NIFTY.** n = 1/2/3/5, with or without the 15-min
   cooldown, with or without the strike-change guard: Δ −59 to −126 ₹/lot/day, every one of them
   negative (no plateau), none significant, and each roughly doubles-to-triples the worst day
   (−4,541 → −8,279 → −11,391).
3. **The churn is priced, not asserted.** NIFTY extra cycles are **gross-flat** (−₹2,772 to +₹1,274
   over 88 days) while costing ₹6,334–8,383 per lot — **dealing cost is 300–590% of everything the
   extra trades produce.** The re-center cannot pay for its own round trip.
4. **r/96's stated reason is backwards.** The re-center is *better* on DTE0/DTE1 (+182 and +273 on
   NIFTY) and worse on DTE2/DTE3+ (−178, −332). It fails on cost per round trip, not on expiry-day
   trending. Right answer, wrong premise.
5. **Mechanism:** the re-centered straddle stops out again ~30% of the time after a rupee stop and
   **55–61%** after a 0.4% move stop. A 15-minute cooldown does not change that.
6. **Live money agrees.** Pooled across 42 real re-centers in three recorded books, the arm is
   **−₹8,327/lot (−₹198 per re-center)**. SENSEX's live 6 are +₹3,244/lot but 0-for-6 clean against
   a modelled 61% re-stop rate — a small, kind draw; the two NIFTY books lost ₹11,571/lot over 36.
7. **SENSEX ATM2 is live on the worst arm tested** (`0.4% move-stop + re-center + 15-min cooldown`,
   −334 ₹/lot/day, −₹17,884/lot of churn across 99 extra cycles). **Raised as a separate strategy
   change** — own STATUS-MD, own SENSEX-calibrated stop, Arun's sign-off, after-15:40 deploy.
   Not touched here.

**Recommendation: keep one-and-done on NIFTY ATM2. No code change, no deploy.** Fourth
reproduction after research/54, research/60 and research/123.
