# Monday Window Rehab — Any ≤120-min Straddle Window × Stop That Beats 1:3 R:R@p95? — STATUS: DONE

**VERDICT: NO EDGE — 0 of 3,014 cells clear the gate cascade; the only family-wise-significant
Monday cells are reliable LOSERS (lunchtime + ₹500 stop). Monday stays dark. See `results/RESULTS.md`.**

Study: `research/124_monday_window_rehab/` · Started 2026-08-23 · Host: VPS (canonical) · DBs READ-ONLY

---

## 2. The Ask

**What Arun asked (2026-08-23, verbatim):**
> "now that we are not doing timeB live on Monday, can u do your research to understand if we
> can do it on any timeslot (max 120 mins) with a stop loss which has a better risk reward
> ratio i.e. below 1:3 and a better / workable probability of success. Try all permutation
> combinations and optimizations possible. You can start with the options data we have, beyond
> that we have price action for more than 2 years, which u can use to study the calmness time
> zone or so."

**What we're actually testing:** Monday TimeB NIFTY 13:00–14:00 SL20 DTE1 was dropped from
live on 2026-08-22 after r/120 + r/121 + r/122 concordantly condemned that specific cell
(R:R@p95 1:11.8, modelled P(loss) 52%, median ≈ +₹1.2k/day at 8L). This study asks the
DIFFERENT question: **does ANY Monday intraday ATM-short-straddle window of ≤120 minutes,
under ANY stop (combined-premium % OR rupee-per-lot), clear R:R@p95 better (smaller) than
1:3 with a workable P(win), net of full costs, after multiple-testing discipline, with
long-sample (multi-year) tail agreement?** If yes → a RECOMMENDATION for Arun's sign-off
(never a config edit by this study). If no → Monday stays dark, with the definitive table.

NIFTY Monday = DTE1 in the current Tuesday-expiry era (primary). SENSEX Monday = DTE3
(secondary — it also sits out Mondays, cheap to include).

**Stage-0 frame (from r/122's atlas, before any new compute):** the atlas already contains
159 Monday rows per venue (30-min-start grid × 60/90/120/HOLD × SL20/SL25/NOSTOP). Best
existing NIFTY-DTE1 cells by rr_p95: G_13:50→15:20 (med +4,160@10L, win 70.6%, 1:1.1 — but
modelled P(loss) 45%, worst −18,490@10L, and its SL arms are decorative/identical) and
G_09:20_90 (med +4,650@10L, win 64.7%, 1:2.0, plm 40.5%, worst −15,240). SENSEX-DTE3 best:
G_09:50_H (1:2.2 but a 5.5-h hold — exceeds the 120-min cap) and G_13:20_120 (med +3,580@10L,
win 76.5%, 1:2.5, plm 13.6%, worst −9,710). r/122 explicitly declined to recommend these
(16–17-day medians in a ~220-comparison family, COMB-overlap concerns). The dropped cell
DEP_1300_1400 SL20 sits at med +1,240@10L, win 70.6%, 1:11.8, plm 52.1%. This study's job is
to adjudicate those hints on a finer grid with both stop families and pre-registered gates.

---

## 3. The Base — exact mechanics

**Construction:** sell 1 ATM straddle (strike = spot rounded to venue step) at window start,
buy back at window end or on stop, whichever first. Front weekly expiry from the chain itself.
Sizing for reported rupees: **8 lots** (NIFTY qty 520, SENSEX qty 160) per the ask;
reconciliation vs r/122 done @10 lots (its convention). NIFTY lot 65, SENSEX lot 20
(`option_chain.lot_size` is WRONG — known trap).

**Costs (r/122's exact model, for comparability):** slippage 0.5 pt (NIFTY) / 1.0 pt (SENSEX)
per leg-side × 4 leg-sides + ₹30/leg-side brokerage × 4 → **₹250/lot RT NIFTY, ₹200/lot RT
SENSEX**, charged on every day including stop days.

**Data:**
- Stage A: `backtest_data/options_data.db :: option_chain` — REAL 1-minute chain, 86 recorded
  days/venue 2026-04-20→08-21, of which 18 Mondays (17 expected after guards). Guards carried
  from r/120–123: frozen-chain holiday (<50 distinct spot prints/day; known 2026-05-01,
  05-28, 06-26), partial session (last snap <15:15), thin day (<200 chain minutes).
- Stage B: `market_data.db :: market_data_unified` — SENSEX 1-min 2021→ (~1,354 d), NIFTY50
  **5-min** 2015→2026-07-17 (~2,754 d; NIFTY has NO 1-min series). r/121 licence: 5-min ==
  1-min EXACTLY for max-excursion-in-a-fixed-window (proved 0/4,068 rows differ); excursions
  only, no path-dependent fills, on the long sample.
- **DTE-era labels are mandatory:** NIFTY weekly expiry Thu (2019-02→2025-08) → Tue (2025-09→);
  SENSEX Fri (2024) → Tue (2025-01→08) → Thu (2025-09→). A historical Monday is NOT DTE1 for
  NIFTY before 2025-09. Monday-calmness questions use actual Mondays labelled by era;
  DTE1-behaviour questions use dte_trd==1 days regardless of weekday. Per-era n stated
  everywhere.

**Success criterion (pre-registered, all gates required for a RECOMMENDATION):**
- **G1** median net @8L > 0 AND observed P(win) ≥ 60% (deployed cells run 70–93%).
- **G2** R:R@p95 better than 1:3 — bridged p95 adverse ₹ (credit_med rung, SL-capped) /
  median net win < 3. R:R and bridge definitions copied from r/122 exactly.
- **G3** modelled P(loss) on the long DTE-matched sample ≤ 40% (i.e. modelled P(win) ≥ 60%),
  via r/122's Theil-Sen breakeven-move + stop-trip method.
- **G4** plateau: ≥3 of the cell's window-neighbours (start ±15 min / duration ±15 min, same
  arm) also pass G1+G2, AND the adjacent stop levels do not flip the sign. No isolated peaks.
- **G5** family-wise: cell |t| clears the 95th pct of max-|t| under 2,000 day-level sign-flip
  draws (Westfall–Young style, preserving cross-cell correlation), OR the finding is demoted
  to a "shape" claim as in r/120 — a shape is NOT a recommendation.
- **G6** label-shuffle null (r/121 discipline): the best Monday cell must beat the 95th pct of
  "best cell" over 2,000 draws of 17 random same-venue non-holiday days from the same recorded
  sample (the grid is run on ALL 86 days precisely to make this null exact).
- **G7** era consistency: current-era (Tue-expiry) Monday excursion tail must not blow up the
  bridge — R:R recomputed on the WORSE of full-sample vs current-era-only p95 must still pass.
- **G8** beats the null alternative: expected ₹/Monday and tail vs putting the same margin as
  extra lots on the validated TUE_NIFTY_DTE0 / FRI_NIFTY_DTE2 cells — stated explicitly with
  numbers.

---

## 4. Plan — axes × cell count

**Stage A grid (per venue):**
- Entry starts: 09:16, then 09:30→14:00 in 15-min steps = **20 starts**.
- Durations: 30/45/60/75/90/105/120 min; exit capped at 15:20; window kept if ≥30 min;
  duplicate (start,end) pairs deduped = **137 unique windows** per venue.
- Stop arms (**11**): combined-premium SL **10/15/20/25/30/40/NOSTOP** + rupee-per-lot stops
  **₹500/₹1000/₹1500/₹2500** (fire when (comb−credit)×lot ≥ X).
- Venues: NIFTY + SENSEX. **Family size: 137×11 = 1,507 cells per venue** (~3,014 total) on
  n≈17 Mondays — hence gates G4–G6.
- Run on ALL ~86 recorded days/venue (Mondays = analysis; other days feed G6's exact null and
  the weekday comparison). ~50k output rows.

**Stage B:** same 137 windows × every weekday, both venues, era-DTE labels; per day×window max
|excursion| bp + |terminal| bp. Outputs the Monday calmness clock (per era), P(excursion >
stop-equivalent bp) per window×stop via the b-slope bridge, and the G3/G7 inputs.

**Stage C:** build_atlas-style merge + the G4/G5/G6 discipline + RESULTS.md.

**Reconciliation gate (before anything else is interpreted):** my harness restricted to
DEP_1300_1400 NIFTY dte1 SL20 and DEP_1000_1200 NIFTY dte2 SL20 must reproduce r/122's
stage_a_alldays.csv rows **to the rupee** (same guards, same costs). If not — stop and fix.

---

## 5. Status — live log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-23 ~22:00 | Study opened; STATUS-MD written pre-compute | folder 124 confirmed free; repo clean at 484e651 |
| 2026-08-23 ~22:00 | Stage-0 atlas frame extracted (laptop copy of r/122 atlas.csv) | best existing Monday cells noted in §2; no new compute yet |
| 2026-08-23 ~22:05 | Stage A launched (VPS, niced) | 137 windows × 11 arms = 1,507 cells/venue, all 86 recorded days |
| 2026-08-23 ~22:10 | Stage B relaunch (first launch lost the cd in an `&&`-chain — venv not found) | fixed with explicit cd; both stages then ran clean |
| 2026-08-23 ~22:15 | Stage A DONE (82 days kept/venue; 17 Mondays; 246,840 rows) · Stage B DONE (2,754 NIFTY + 1,354 SENSEX days; 562,534 rows) | fast — pure-python replay is light at 1-min |
| 2026-08-23 ~22:25 | build_monday_atlas.py run: **reconciliation vs r/122 PASSED 82/82 days to the rupee** on all 3 checked cells | gate to interpret results cleared |
| 2026-08-23 ~22:25 | Gate cascade: NIFTY 137→113→36→**0**, SENSEX 43→43→10→**0** | G5: only significant cells are NEGATIVE (lunch R500). G6: Monday best p=0.329/0.969 vs shuffled-best null |
| 2026-08-23 ~22:40 | Forensics: 2026-07-13 broke the excursion bridge (−₹22k@8L on 57.7bp — IV pop, ~10× the b-slope); SLP25 worst WORSE than NOSTOP (4th repro of r/114/116/121) | RESULTS.md written; verdict NO EDGE |

## 6. Crash Recovery

- Stage A runner: `scripts/stage_a_monday.py` → `results/stage_a_monday.csv` + `results/stage_a.log`.
  Check progress: `tail -5 research/124_monday_window_rehab/results/stage_a.log`;
  `wc -l research/124_monday_window_rehab/results/stage_a_monday.csv`.
  Re-run from scratch (idempotent, overwrites): `cd /home/arun/quantifyd && nohup nice -n 15 venv/bin/python3 research/124_monday_window_rehab/scripts/stage_a_monday.py > /tmp/r124_a.log 2>&1 &`
- Stage B runner: `scripts/stage_b_monday_clock.py` → `results/stage_b_window_days.csv` + `results/stage_b.log`. Same pattern (`/tmp/r124_b.log`).
- Analysis: `scripts/build_monday_atlas.py` (pure aggregation, re-runnable any time after A+B) →
  `results/monday_atlas.csv`, `results/gates_report.txt`, `results/percentiles_long.csv`.
- Alive check: `pgrep -af 'stage_[ab]_monday'`. Nothing here touches any live system; all DBs
  opened `mode=ro`. Heavy per-day CSVs are gitignored; regeneration = the two runner commands above.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `MONDAY_WINDOW_REHAB_1MIN_SWEEP_STATUS.md` | this file | yes |
| `scripts/stage_a_monday.py` | 1-min chain sweep, all days, 137 windows × 11 arms | yes |
| `scripts/stage_b_monday_clock.py` | long-sample excursion clock, era-labelled | yes |
| `scripts/build_monday_atlas.py` | atlas + gates G1–G8 + nulls | yes |
| `results/stage_a_monday.csv` | per day×cell×arm rows (~50k) | NO — gitignored |
| `results/stage_b_window_days.csv` | per day×window excursions (~550k) | NO — gitignored |
| `results/monday_atlas.csv` | one row per Monday cell with all gate columns | yes |
| `results/gates_report.txt` | G5/G6 null distributions + pass/fail | yes |
| `results/RESULTS.md` | final verdict | yes |

## 8. Findings

(live — filled as they emerge)
