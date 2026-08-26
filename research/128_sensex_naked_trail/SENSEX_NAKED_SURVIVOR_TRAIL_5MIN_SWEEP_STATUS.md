# SENSEX Naked-Survivor Trail — build the NIFTY-equivalent trailing stop and calibrate it

STATUS: **DONE** (commissioned 2026-08-26 by Arun, mid-session; concluded 2026-08-26 12:35 IST)

> **VERDICT: SIGNAL - DEPLOY AS A CORRECTNESS FIX, NOT AS AN ALPHA CLAIM.**
> Recommendation: `CEIL_p7_m3.0_N1_SEED` - ratcheting ST(7,3) **ceiling**, never written into
> `sl_price`, seeded warm-up, ~60 s confirm, breakeven clamp. Full write-up:
> [`results/RESULTS.md`](results/RESULTS.md). **Patch sketched, NOT applied** - deploy after 15:40 IST
> with Arun's sign-off.

## 2. The Ask

**What Arun said:** "u better study this based on all the days where we hv options data and
then suggest, but we must hv the trailing SL for this like the nifty counterpart's"

**The requirement is not open-ended.** The deliverable is a WORKING trailing stop for the
SENSEX naked survivor, matched in mechanism to the NIFTY one. "Leave it as it is" is NOT an
acceptable verdict. The open questions are the DESIGN and the PARAMETERS, not whether to have
one.

## 3. What is broken today (verified live, 2026-08-26)

When a SENSEX straddle leg stops out, the survivor is handed a trail by `app.py` (~line 8148)
via `services/sensex_naked_trail.py`. It computes ST(7,3) on 5-min premium candles and the
caller clamps it to `<= breakeven`, then **writes it into `sl_price`**, where the generic
`check_and_handle_sl` fires on `live >= sl_price`.

There is no check that the ST sits ABOVE the live price. Today:

    11:00:02  naked survivor SENSEX26AUG78000CE -> SENSEX_ST_TRAIL(7,3) SL 90.4
    11:00:02  SL HIT: live=134.00 >= SL=90.40
    11:00:05  [EXIT] @ 132.60

A stop written below the market self-triggers the instant it is set. **Contrast NIFTY**
(`nas_ticker`, ~line 956): it keeps the ST in `_atm_naked_st_val`, exits only when
`ltp > st_val` for `NAKED_TRAIL_CONFIRM_TICKS` consecutive ticks, and **never writes it into
`sl_price`**. NIFTY's log shows `close=105.4, stop=132.0` — the stop correctly ABOVE price.

**Measured consequence over 12 live episodes** (all SL_HIT tagged SENSEX_ST_TRAIL): booked
+Rs69,114 vs +Rs62,133 for a BE-only counterfactual — the accident has NET GAINED Rs6,981,
because on 2 of 12 days (17-Aug, 25-Aug) the option round-tripped to entry and the early exit
banked profit that holding would have surrendered. It hurt on 08-25 ATM by Rs7,114. So the
current behaviour is an accidental, arbitrary profit-take that happens to be net-positive on
n=12. That is not a rule; it is luck with a wide spread (-7,114 to +5,910 per leg).

## 4. The Base — what to build and test

**Episodes.** Do not rely on the ~12 live ones. Replay the deployed SENSEX rules over EVERY
usable chain day to synthesise survivor episodes:
- SENSEX ATM: 09:16 ATM straddle, per-leg 30% SL, first SL closes that leg, survivor trails,
  EOD 15:15. `leg_sl_disabled_dtes=(0,)` — NO per-leg stop on DTE0 (Thursday), so DTE0 produces
  no survivor episodes by that route.
- SENSEX ATM4: same, but the first SL ROLLS to match; the survivor case arises on the second SL.
- Report episode count per DTE and per weekday.

**Arms to sweep (the trail designs):**
1. `INCUMBENT` — today's behaviour exactly (ST written to sl_price, no price check). The null.
2. `BE_ONLY` — breakeven protection alone, no trail.
3. `NIFTY_EQUIV` — **the required design**: ST as a CEILING, exit only when premium holds above
   it for N consecutive polls, clamped `<= breakeven`, never written to `sl_price`. Sweep
   `N = 1,2,3,5` (NIFTY uses `NAKED_TRAIL_CONFIRM_TICKS`).
4. `ST_GRID` — ST period 5/7/10/14 x multiplier 2.0/2.5/3.0/3.5/4.0 on the ceiling design.
5. `GIVEBACK` — trail a fixed % or fixed Rs/lot back from the survivor's BEST (lowest) premium.
6. `DECAY_TP` — flat take-profit: buy back once the survivor reaches X% of entry (sweep X).
   This is the deliberate version of what the bug does accidentally — include it so we learn
   whether the accident's edge is real or noise.

**The metric that matters most:** how often does a decaying survivor ROUND-TRIP back to entry?
That single statistic decides whether an aggressive take-profit or a patient trail is right.
Report its distribution, per DTE.

**Costs:** the MEASURED outcome-aware model — forced/stop exits +6.548 pt per leg-side,
time/EOD exits +0.178, entry 0 (see `research/122_window_risk_atlas/scripts/stage_a_alldays.py`
`cost_per_lot()`), plus the exact Zerodha rate card. The retired flat Rs250/lot is not used.

**Candle construction:** the live trail builds 5-min premium candles from 10s polls. Replay
from the 1-min chain and state the resolution difference honestly; do NOT use 5-min underlying
data for excursions (binding), but 5-min PREMIUM candles are what the live code actually uses,
so mirror that for the ST calculation and note the fidelity gap.

## 5. Success criterion

Recommend ONE design + parameters. It must: beat BE_ONLY and the INCUMBENT net of measured
costs; sit on a PLATEAU of neighbouring parameters (no isolated peak); survive a family-wise
haircut over the full grid; hold on an OOS split; and be implementable in the existing code
path. Report what it would have done on each of the 12 real live episodes as a reconciliation.
If the best design is statistically indistinguishable from BE_ONLY, say so — and still
recommend the NIFTY-equivalent ceiling with the most defensible parameters, because Arun has
specified that a trail must exist. State plainly in that case that the parameters are a
judgement call on thin evidence rather than a measured optimum.

## 6. Process

Read-only DBs, niced, `scripts/` + `results/`, live event log in this file,
`results/RESULTS.md` with a bold verdict, row added to `research/INDEX.md`, commit ONLY this
folder + INDEX and push. **No live code changes** — the recommendation goes to Arun with a
proposed patch for an after-15:40 deploy.

---

## 7. Status log

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-26 11:20 IST | Commissioned; sections 1-6 written before compute | live bug verified; 12-episode quantification attached |
| 2026-08-26 12:04 IST | Stage 1 `build_episodes.py` launched (VPS, nice 15) | replay of deployed ATM + ATM4 rules over every SENSEX chain day |
| 2026-08-26 12:06 IST | Stage 1 DONE (1st pass) | 89 candidate days -> 84 usable -> 86 episodes. Guards fired exactly as pre-registered: 04-20 thin, 05-01/05-28/06-26 frozen chain, 08-26 partial |
| 2026-08-26 12:15 IST | Stage 1 re-run with the pre-naked path stored | needed to test the SEEDED warm-up (NIFTY's `_seed_naked_candles` was never ported to SENSEX) |
| 2026-08-26 12:18 IST | Stage 2 `sweep_arms.py` DONE | 188 arms x 86 episodes; net of the research/122 MEASURED cost model reduced to one leg |
| 2026-08-26 12:20 IST | **Headline finding** | **23.3% of decaying naked survivors round-trip all the way back to entry**; 38.4% give back >=half. Concentrated on DTE1/Wed (29%/31%) |
| 2026-08-26 12:22 IST | Stage 3 `reconcile_live.py` DONE | harness fidelity **9/12 EXACT** to the minute vs the real live exits; 3 fire later in replay (1-min vs 10s polls) |
| 2026-08-26 12:24 IST | Stage 4 `analyze.py` + Stage 5 `diagnostics.py` DONE | plateau map, family-wise haircut, OOS split, cost sensitivity, self-trigger diagnostic |
| 2026-08-26 12:30 IST | **Second finding** | INCUMBENT's ST lands BELOW the live premium on **62%** of episodes -> self-trigger. It books Rs2,480/lot, **worse than BE_ONLY** (Rs2,536). The live +Rs6,981 was small-sample luck |
| 2026-08-26 12:35 IST | `results/RESULTS.md` written; **STATUS: DONE** | Verdict **SIGNAL - DEPLOY AS A CORRECTNESS FIX**. Recommendation `CEIL_p7_m3.0_N1_SEED`, patch sketched, NOT applied |

### Live findings during the run

* **A trailing stop is required and the data says so.** `HOLD_EOD` (no stop at all) is the worst arm
  on the board: Rs2,136/lot mean, worst episode **-Rs15,716/lot**. Arun's requirement is supported by
  the evidence, not just by preference.
* **The incumbent is not a trail.** It is an arbitrary ~40-minute delayed market exit (median 38 min
  from naked to the first ST value; 90.7% fire rate; 69-minute mean hold) that self-triggers 62% of
  the time. It underperforms doing nothing beyond breakeven protection.
* **The ceiling design is structurally correct in 100% of episodes** - `compute_short_trailing_stop`
  ratchets down and by construction sits above the last bar's close. The recommended arm has the
  **best tail of anything tested** (worst -Rs613/lot).
* **The uplift over breakeven-only is NOT statistically distinguishable** (+Rs259/lot, t 1.35; OOS a
  wash) and no cell survives a Bonferroni haircut over 188 arms. The parameters are a judgement call
  on a broad plateau, as the brief anticipated.

---

## 8. Findings summary

Full write-up in `results/RESULTS.md`. Headlines:

| | |
|---|---|
| Verdict | **SIGNAL - DEPLOY AS A CORRECTNESS FIX, NOT AS AN ALPHA CLAIM** |
| Recommended arm | `CEIL_p7_m3.0_N1_SEED` - ratcheting ST(7,3) **ceiling**, never written to `sl_price`, seeded warm-up, ~60 s confirm, breakeven clamp |
| Round-trip statistic | **23.3%** full round-trip to entry; 38.4% give back >=50%; 32.6% decay cleanly one-way |
| vs INCUMBENT | **+Rs315/lot, t = 2.31** (the one comparison that holds) |
| vs BE_ONLY | +Rs259/lot, t = 1.35 - **not significant**; OOS +Rs63/lot (a wash) |
| vs HOLD_EOD | +Rs659/lot, and worst episode -Rs613 vs **-Rs15,716** |
| 12 live episodes | LIVE +Rs64,727 -> recommended **+Rs67,417** (BE-only +Rs60,079) |
| Deploy | after 15:40 IST, with Arun's sign-off. Patch sketched in RESULTS section 8; **not applied** |

---

## 6b. Crash recovery - how to resume WITHOUT Claude

Everything runs on the VPS, read-only on all DBs, and each stage is a single command. Total runtime
is under 5 minutes, so the recovery path is simply "re-run the pipeline in order".

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd
R=research/128_sensex_naked_trail

# 1. episodes (reads options_data.db READ-ONLY; ~90 s) -> results/episodes.csv, results/paths.jsonl
nice -n 15 venv/bin/python3 $R/scripts/build_episodes.py

# 2. arm sweep (~60 s) -> results/arm_episode.csv, arm_summary.csv, roundtrip.csv, random_null.json
nice -n 15 venv/bin/python3 $R/scripts/sweep_arms.py

# 3. plateau / family-wise / OOS / cost sensitivity -> results/analysis.txt, results/plateau.csv
R128_ARM=CEIL_p7_m3.0_N1_SEED nice -n 15 venv/bin/python3 $R/scripts/analyze.py

# 4. mechanism diagnostics -> results/diagnostics.txt
nice -n 15 venv/bin/python3 $R/scripts/diagnostics.py

# 5. reconciliation against the 12 real live episodes -> results/live_reconciliation.csv
R128_ARM=CEIL_p7_m3.0_N1_SEED nice -n 15 venv/bin/python3 $R/scripts/reconcile_live.py
```

**How to check what finished:** `ls -la $R/results/` and `tail $R/results/build_episodes.log`.
Stage 1 prints `DONE: usable days=84 episodes=86` on success; stages 2-5 print `DONE` / write their
`.txt`.

**Safe to inspect / delete and regenerate:** everything in `results/`. Nothing in this study writes
to any database, to any live/paper state file, or to any service. **Do not touch** anything outside
`research/128_sensex_naked_trail/` - in particular `services/sensex_naked_trail.py` and `app.py`
carry only a *proposed* patch in `results/RESULTS.md`, deliberately NOT applied.

---

## 6c. Files

| File | Purpose | Committable? |
|---|---|---|
| `SENSEX_NAKED_SURVIVOR_TRAIL_5MIN_SWEEP_STATUS.md` | This file | yes |
| `results/RESULTS.md` | Verdict, tables, recommendation, patch sketch, caveats | yes |
| `results/analysis.txt` | Plateau map, family-wise, OOS, cost sensitivity (as printed) | yes |
| `results/diagnostics.txt` | Self-trigger / ceiling / round-trip-timing diagnostics | yes |
| `scripts/build_episodes.py` | Stage 1 - synthesise survivor episodes from the chain | yes |
| `scripts/sweep_arms.py` | Stage 2 - 188 arms x 86 episodes, net of measured cost | yes |
| `scripts/analyze.py` | Stage 4 - plateau / family-wise / OOS / cost sensitivity | yes |
| `scripts/diagnostics.py` | Stage 5 - mechanism diagnostics | yes |
| `scripts/reconcile_live.py` | Stage 3 - replay vs the 12 real live episodes | yes |
| `results/episodes.csv` | 86 episodes, metadata (7 KB) | yes |
| `results/roundtrip.csv` | Per-episode round-trip statistics (11 KB) | yes |
| `results/arm_summary.csv` | One row per arm (31 KB) | yes |
| `results/plateau.csv` | The 160-cell ST grid (20 KB) | yes |
| `results/live_reconciliation.csv` | The 12 live episodes vs each arm (2 KB) | yes |
| `results/random_null.json` | Random-exit null, per episode (3 KB) | yes |
| `results/build_episodes.log` | Per-day replay log incl. every skip reason (5 KB) | yes |
| `results/arm_episode.csv` | Arm x episode detail (1.8 MB) | **NO - gitignored**, regenerate with stage 2 |
| `results/paths.jsonl` | Per-episode 1-min premium paths (456 KB) | **NO - gitignored**, regenerate with stage 1 |
