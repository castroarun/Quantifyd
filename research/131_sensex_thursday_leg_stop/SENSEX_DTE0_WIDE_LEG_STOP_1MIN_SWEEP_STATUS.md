# SENSEX DTE0 Per-Leg Stop — Is There a WIDE Level That Caps a Breakaway Without Killing the Expiry Edge?  STATUS: DONE

> **VERDICT: NO EDGE — no per-leg stop at any level. Keep `leg_sl_disabled_dtes=(0,)`. No live change.**
> All 33 stop arms lose to HOLD in both entry sets and both outer-layer families; 0 of 33 positive in
> any OOS half; nothing survives the family-wise haircut (best family-wise p = 0.769) while the *harm*
> from tight stops does (t −6.23). And every level wide enough not to harvest noise is one the
> deployed venue book stop/TP has already pre-empted — `RUP8000` fires **0%** of sessions under the
> live outer layer, for **exactly ₹0/lot**. Full write-up: `results/RESULTS.md`.

File: `SENSEX_DTE0_WIDE_LEG_STOP_1MIN_SWEEP_STATUS.md` · research/131 · started 2026-08-27 07:55 IST
Host: Contabo VPS `94.136.185.54`, repo `/home/arun/quantifyd`, venv `venv/bin/python3`. All DBs READ-ONLY.

---

## 1. The Ask

**What Arun asked (via the research brief):**
> SENSEX ATM and ATM4 currently carry **no per-leg stop on expiry day** (`leg_sl_disabled_dtes=(0,)`
> in `config.py`), on the strength of research/114. But r/114 only ever tested a **30%** leg stop and
> combined-SL variants. It never asked what a **WIDE** leg stop would do. A tight stop harvests
> intraday churn; a wide one only catches a genuine breakaway. Having *no* leg-level protection at
> all on expiry day is uncomfortable, and a 30% stop being a noise-harvester does not prove a 60%
> one is.

**What we are actually testing:**
> Over every recorded SENSEX **DTE0** session on the real 1-minute option chain, replaying the live
> 09:16 ATM straddle: **is there a per-leg stop level — percentage-of-entry or rupee-per-lot — wide
> enough to leave the expiry-day decay edge intact while still capping a genuinely breakaway leg?**
> And if a leg does stop, **what should happen to the survivor** — hold it, trail it with the live
> ST(7,3) ceiling, or close both legs together?
>
> The incumbent null is **HOLD** (no leg stop, exit 15:15). A candidate must beat HOLD **net of
> measured costs**, sit on a **plateau** of neighbouring levels, survive a **family-wise** haircut
> across the whole grid, and hold on an **OOS split**. The **fire rate** at every level is reported
> as the diagnostic that separates a disaster-stop from a noise-harvester.

**Prior art this must reconcile against:**

| Study | Finding |
|---|---|
| research/114 | Per-leg 30% on SENSEX Thursday = **-Rs227/lot, 25% win**; HOLD = **+Rs2,630/lot, 92% win**. Only 30% + combined-SL tested. |
| research/118 | The "92% win / worst -127" was a 12-day artefact. Over 127 real DTE0 days: ~34% losers, 8.7% worse than -500 pts, worst about -21,500/lot. **DTE0 is the fattest-tailed slot in the dataset.** |
| research/122 | A 20% stop fires on ~85% of expiry afternoons. Worst recorded expiry day 2026-06-11 = -Rs54,100 at 10 lots (**-Rs5,410/lot**). |
| research/116, 121, 124 | Tightening has been destructive in every independent cut. |
| research/128 | The live SENSEX naked-survivor trail was broken (lower-band bug); rewritten 2026-08-26 to the ratcheting ST(7,3) **upper** band, ~60 s confirm, clamped to breakeven. |

Prior expectation is therefore **HOLD wins at every level** (a fifth reproduction is a good result).
The study exists to test the one variant nobody ran: a stop so wide it is not a churn-harvester.

---

## 2. The Base — what is being tested

**Construction (fixed across every arm; only the exit rule differs).**

| Element | Value |
|---|---|
| Venue | SENSEX (BFO). **Lot = 20** (`option_chain.lot_size` is WRONG — memory: options data traps r/119) |
| Data | `backtest_data/options_data.db :: option_chain`, **1-minute** snapshots, 2026-04-20 -> 2026-08-26 |
| Session universe | Every recorded **DTE0** day = front expiry (min `expiry_date >= day`) **equals the day itself**. This is DTE-era-safe: SENSEX expiry moved Fri -> Tue -> Thu, so weekday selection (what r/114 used) mislabels. |
| Entry | **09:16**, ATM straddle, strike = `round(spot_at_0916/100)*100`, front expiry, SELL CE + SELL PE |
| Baseline exit | **15:15** |
| Sizing | Per **1 lot** (qty 20). 1 premium point = Rs20. Live book runs 2 lots/system. |
| P&L | `(credit - exit_premium) x 20` per lot, per leg where legs exit separately |

**Guards (mandatory, carried from r/120/121/122):**
- Frozen-chain holiday guard: reject any day with **< 50 distinct spot prints** (known 2026-05-01, 05-28, 06-26 — none of which are DTE0, so this is a belt-and-braces check).
- Partial-session guard: reject any day whose last snapshot is before 15:15.
- Thin-day guard: reject any day with < 200 usable minutes on the ATM strike.
- Both legs must have an LTP in the same minute for that minute to be usable.

**Cost model — the MEASURED, outcome-aware model (r/122 `stage_a_alldays.py :: cost_per_lot`),
NOT r/114's retired `SLIP=1.0 pt/leg-side + CHG=Rs30/lot` heavy assumption.**
Decomposed here to **per leg**, so legs that exit for different reasons are charged differently:

```
per leg, per lot:
  brokerage  = Rs20 x 2 orders / NLOTS_REF(10)      = Rs4
  STT        = 0.001    x (entry_prem x 20)          (sell side only)
  txn        = 0.0003503 x (entry+exit) x 20
  ipft       = 0.0000050 x (entry+exit) x 20
  sebi       = 0.0000010 x (entry+exit) x 20
  stamp      = 0.00003   x (exit x 20)
  gst        = 0.18 x (brok + txn + ipft + sebi)
  slippage   = entry 0.0 pt + exit (STOP: +6.548 pt | TIME/EOD: +0.178 pt), x 20
```
Measured from 443 real live leg-sides (Kite fill vs chain LTP, same minute, live NAS 916 books,
2026-08-25). Slippage lives in the stop-outs — which is precisely the asymmetry that decides this
question. **We will also re-run r/114's LEG30 under BOTH cost models and report whether the cost
change alone moves r/114's conclusion.**

**Arms — the exit rule (34 rules x 2 outer-layer families = 68 cells).**

*Leg-stop levels (11 + null):*

| Family | Levels |
|---|---|
| null | `HOLD` — no leg stop |
| percent of leg entry premium | `LEG30` (r/114 reconciliation) · `LEG40` · `LEG50` · `LEG60` · `LEG75` · `LEG100` |
| rupee per lot on the breached leg | `RUP1500` · `RUP2500` · `RUP4000` · `RUP6000` · `RUP8000` |

Trigger: leg premium `ltp >= entry_leg x (1+p)` (percent), or `(ltp - entry_leg) x 20 >= X` (rupee).

*Survivor treatment when a leg stops (3):*

| Code | Treatment |
|---|---|
| `SBOTH` | close **both** legs at that minute (this is what r/114's LEG30 did) |
| `SHOLD` | close the breached leg, **hold the survivor to 15:15** |
| `STRAIL` | close the breached leg, trail the survivor with the **live** ST(7,3) ceiling |

`STRAIL` replicates `services/sensex_naked_trail.py` as deployed 2026-08-26: 5-minute premium candles
built from the 1-min series, `NasAtm4Executor.compute_short_trailing_stop(bars, period=7, mult=3.0)`
(ratcheting **upper** band, tightens only), ceiling **clamped to the leg's own entry (breakeven)**,
warm-up seeded from the leg's 09:16->stop candles, exit only after the premium holds above the
ceiling for **~60 s** (2 consecutive 1-min prints — the live rule is 6 polls at the 10 s cadence).

*Outer-layer families (2) — the deployed venue book stop/TP must stay in the model:*

| Family | Rule |
|---|---|
| `STANDALONE` | leg rule only — isolates the leg stop's own contribution |
| `VENUE` | plus the **deployed DTE0 outer layer**: book stop **-Rs3,000/lot**, take-profit **+Rs4,000/lot** (`services/nas_portfolio_stop.py`, `_stop_per_lot=3000` on SENSEX DTE0, `tp_per_lot=4000`), evaluated on combined straddle P&L, flattening everything when hit |

The `VENUE` family answers the question that actually matters operationally: **a leg stop that never
fires before the book TP/stop already has is irrelevant.** Caveat recorded up front: the real book
stop is computed across the whole SENSEX venue (3 systems), so a single-straddle per-lot proxy is an
approximation — it will fire *later* than the real one on days where sibling systems are also down.

---

## 3. Plan — grid and gates

| Axis | Values | n |
|---|---|---|
| Leg stop level | HOLD, LEG30/40/50/60/75/100, RUP1500/2500/4000/6000/8000 | 12 |
| Survivor treatment | SBOTH, SHOLD, STRAIL (HOLD has none) | 3 |
| Outer layer | STANDALONE, VENUE | 2 |
| **Total cells** | (11 x 3 + 1) x 2 | **68** |
| Sessions | recorded SENSEX DTE0 days | **17** (2026-04-30 -> 2026-08-20) |
| Entry set (added 07:58, pre-compute) | **A0920** uniform 09:20, all 17 · **B0916** live-exact 09:16, the 12 days that have a 09:16 print | 2 |
| Replays | 68 x (17 + 12) | 1,972 + 116 reconciliation rows = 2,088 |

Plus a **reconciliation cell**: r/114's exact LEG30/SBOTH rule under r/114's own cost model
(`SLIP=1.0 pt/leg-side`, `CHG=Rs30/lot`) and its own weekday-based day selection, to tie back to the
published -Rs227/lot to the rupee.

**Gates a recommendation must clear (pre-registered):**

1. **Beat HOLD** on mean net Rs/lot, net of the measured cost model.
2. **Plateau** — the winning level's immediate neighbours must also beat HOLD. A lone winning level is noise and will be reported as such.
3. **Family-wise** — paired per-day differences vs HOLD; significance judged against a **sign-flip permutation max-|t| null across all 33 stop arms** (10,000 draws), so the whole grid is charged for.
4. **OOS split** — first 8 sessions vs last 9; sign must hold in both halves.
5. **Fire rate** reported at every level (fraction of sessions where any leg breaches, and mean breach time).
6. **Worst-day** — what each level did on 2026-06-11 (r/122's -Rs5,410/lot session) and on the study's own worst HOLD day.

With n = 17, gate 4 is weak by construction; the honest read is that this study can **kill** a level
convincingly but can only ever call a survivor a **candidate**.

---

## 4. Status — live event log

**State:** DONE — sweep + gates + RESULTS.md complete. **No live change made or recommended.**

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-27 07:45 | VPS session opened, r/114 + r/122 harnesses read | r/114 `run_sx_thu_exits.py` variant framework reused; r/122 `cost_per_lot` adopted |
| 2026-08-27 07:50 | Universe query run | 89 recorded SENSEX days; **17 DTE0** sessions 2026-04-30 -> 2026-08-20, incl. **2026-05-27 (Wed)** from the era shift — a day r/114's `weekday()==3` selection **missed** |
| 2026-08-27 07:55 | This STATUS-MD written; grid pre-registered | 68 cells, gates 1-6 fixed before any result is seen |
| 2026-08-27 07:58 | **Diagnosed r/114's five "no clean series" skips** | Not bad days: the SENSEX recorder **started at 09:20** before 2026-06-04, so 2026-04-30 / 05-07 / 05-14 / 05-21 / 05-27 have **no 09:16 print**. Grid amended (pre-compute) to TWO entry sets: **A0920** (uniform 09:20, all 17) and **B0916** (live-exact 09:16, the 12 days that have it = exactly r/114's day set) |
| 2026-08-27 08:02 | Sweep launched (`run_leg_stop_sweep.py`, nice 15) | 17/17 sessions pass all guards; A0920 n=17, B0916 n=12 |
| 2026-08-27 08:03 | Sweep DONE (~40 s) | 2,088 rows -> `results/leg_stop_detail.csv` |
| 2026-08-27 08:06 | Gates run (`analyse_leg_stop.py`) | summary + reconciliation CSVs + `results/analysis.txt` (513 lines) |
| 2026-08-27 08:10 | **VERDICT: NO EDGE** | `results/RESULTS.md` written; recommendation = keep `leg_sl_disabled_dtes=(0,)` |

### Live findings

1. **r/114 reconciles to ~1%.** On its exact 12 days under its own cost model: HOLD +2,660/lot 92%
   win (published +2,630, 92%); LEG30 −207/lot 25% win (published −227, 25%).
2. **The cost correction makes r/114 STRONGER, not weaker.** Swapping to the measured outcome-aware
   model on the same days widens the HOLD-vs-LEG30 gap from 2,867 to **3,103 ₹/lot (+236)** — because
   the slippage lives in stop-outs (6.548 pt vs 0.178 pt) and HOLD never stops. r/114's flat 1.0 pt
   per leg-side over-charged HOLD and under-charged the stop.
3. **A leg doubling is ROUTINE on DTE0, not a tail event.** Fire rates: LEG30/40/50 and RUP1500 fire
   on **100%** of sessions, LEG60/75 on 82%, **LEG100 still on 41%**, RUP4000 47%, RUP6000 29%,
   RUP8000 12%. No percentage level in the tradable band separates a breakaway from ordinary
   straddle asymmetry.
4. **All 33 arms lose to HOLD, monotonically toward zero as the level widens.** Best arm
   (A0920 standalone) = RUP8000 at **−867 ₹/lot, t −1.44**; worst = LEG50/STRAIL at −3,553, t −4.99.
   That is a monotone approach to the null, not a plateau of outperformance.
5. **Family-wise: nothing helps, tight stops significantly HURT.** max|t| sign-flip null p99 = 3.62;
   best arm's family-wise p = 0.769, while LEG30/SBOTH (t −6.23) and LEG50/STRAIL (t −4.99) clear
   p99 in the *negative* direction.
6. **OOS unanimous: 0 of 33 arms positive in either half, in either entry set.**
7. **THE DECISIVE ONE — the wide end is already occupied by the venue book stop.** Under the deployed
   DTE0 outer layer (−₹3,000/lot stop, +₹4,000/lot TP), HOLD's exits are **18 BOOK_TP / 10 BOOK_STOP
   / 6 EOD** — the outer layer resolves 82% of expiry sessions before 15:15. `RUP8000`'s fire rate
   falls **12% → 0%** and its effect on P&L is **exactly ₹0/lot (t 0.00)**. There is no level that is
   both wide enough not to harvest noise and still able to act before the book stop.
8. **The live ST(7,3) survivor trail (r/128) measurably works** — it cuts the naked-survivor tail by
   ₹1,284/lot (worst −3,524 vs SHOLD's −4,808) for ₹871/lot of mean. Real and worth recording, but
   it is a repair on damage the leg stop caused; HOLD's worst is −212.
9. **2026-06-11 closes POSITIVE on this construction: +₹3,502/lot** held to 15:15. What turned it
   into a loss (−₹3,770/lot) was the **book stop** firing on a fully-recovered intraday excursion.
   r/122's −₹5,410/lot is a different construction/window. This points the follow-on study at the
   **book stop level and sizing**, not at leg stops.
10. **Caveat carried forward from r/118:** worst standalone HOLD day here is only −₹212/lot over 17
    sessions. This sample is benign and cannot price the real DTE0 tail (r/118: ~34% losers over 127
    days, worst ≈ −21,500/lot). What it *can* say is that a per-leg stop is not the instrument that
    catches that tail.

---

## 5. Crash Recovery — resuming without Claude

Everything runs on the VPS, single-threaded, ~1-2 minutes total. There is no long-lived background
job to babysit; if it dies, just re-run it.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd

# 1. what finished?
ls -l research/131_sensex_thursday_leg_stop/results/
tail -40 research/131_sensex_thursday_leg_stop/results/run.log

# 2. is anything still alive?
pgrep -af run_leg_stop_sweep.py

# 3. re-run from scratch (idempotent - it truncates its own outputs)
nice -n 15 venv/bin/python3 research/131_sensex_thursday_leg_stop/scripts/run_leg_stop_sweep.py \
     > /tmp/r131.log 2>&1

# 4. stats / gates only, on an existing detail CSV
nice -n 15 venv/bin/python3 research/131_sensex_thursday_leg_stop/scripts/analyse_leg_stop.py
```

**Safe to inspect / delete:** everything under `research/131_sensex_thursday_leg_stop/results/`.
**Do NOT touch:** `backtest_data/options_data.db` (read-only, the live recorder writes to it), any
`services/*`, `config.py`, or the running `quantifyd` service. This study makes **no live change**.

---

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `SENSEX_DTE0_WIDE_LEG_STOP_1MIN_SWEEP_STATUS.md` | This file | yes |
| `scripts/run_leg_stop_sweep.py` | Replay: 68 cells x 17 DTE0 sessions, writes per-day detail | yes |
| `scripts/analyse_leg_stop.py` | Summary, fire rates, plateau map, permutation family-wise test, OOS split, worst-day table, r/114 reconciliation | yes |
| `results/leg_stop_detail.csv` | Per cell x day rows (2,088 rows, 254 KB) | **NO - gitignored** (`.gitignore:167 research/**/results/*.csv`); regenerate via section 5 |
| `results/leg_stop_summary.csv` | Per-cell aggregate incl. fire% and median fire time | yes (force-added past the ignore rule) |
| `results/r114_reconciliation.csv` | HOLD + LEG30 under both cost models / both day selections | yes (force-added past the ignore rule) |
| `results/analysis.txt` | Full 513-line gate report (per-arm tables, permutation, OOS, worst-day, interaction) | yes |
| `results/run.log` | Run log | **NO - gitignored** (`.gitignore:168`); regenerate via section 5 |
| `results/RESULTS.md` | Verdict, tables, honest read | yes |

All outputs are small (< 1 MB total). The repo-wide rule `research/**/results/*.csv|*.log` gitignores the
254 KB detail CSV and the run log; the two small evidence CSVs were force-added. Everything is
regenerable in ~90 s with the commands in section 5.

---

## 7. Findings

**VERDICT: NO EDGE — no per-leg stop at any level.** The full write-up, with every table, is
`results/RESULTS.md`; the raw gate output is `results/analysis.txt`. The ten headline findings are
in the live-findings block of section 4 above.

**Gate scorecard (all pre-registered in section 3 before any result was seen):**

| Gate | Result |
|---|---|
| 1. Beat HOLD net of measured costs | **FAIL** — 0 of 33 arms, in both entry sets, both outer families |
| 2. Plateau of neighbouring levels | **FAIL** — monotone toward zero *from below*; best arm is simply the one that fires least |
| 3. Family-wise (sign-flip max\|t\|, 10k draws) | **FAIL** — best family-wise p = 0.769; tight stops clear p99 in the *harmful* direction |
| 4. OOS split | **FAIL** — 0 of 33 arms positive in either half, either set |
| 5. Fire rate reported at every level | done — LEG100 still fires on 41% of expiry sessions |
| 6. Worst-day comparison | done — 2026-06-11 closes **+₹3,502/lot** on HOLD; every stop did worse |

**Recommendation: no live change.** Keep `leg_sl_disabled_dtes=(0,)` on SENSEX_ATM and SENSEX_ATM4.
Nothing is deployable, so no after-15:40 window is needed. If more DTE0 tail protection is wanted,
this study says the lever is the **book stop level / position size** (or a bought far wing), not a
per-leg stop — see `results/RESULTS.md` section 10.

**Suggested dated review:** re-run once the recorder holds **>= 40** DTE0 sessions (~2027-01) to see
whether a genuine disaster day appears in-sample and whether the inertness result in section 7 of
RESULTS.md survives it.
