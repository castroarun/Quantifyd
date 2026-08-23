# research/123 — The Opening Scalp: double the morning entries, book half at +20..+65 min?

## VERDICT: **NO EDGE — CONCLUDED.** The premise itself is weekday-folklore (true-ish on Monday, false on Wednesday and Friday), not one of the 400 scalp cells clears even t = 2 against a ~400-comparison family, doubling the actual 09:16 books is *negative* at every horizon under their own stops, the best cell the sweep can find is literally "run more size on Tuesday TimeB" — which is the null alternative the success criterion already required us to beat — and the doubled margin is fundable on exactly one weekday (Monday, the book's weakest morning). CPR/gap filters: 0 winners in 33 pre-registered rules; the only rule that "beat 97% of random skips" was the day-of-month-parity **placebo**. Change nothing.

---

## 1. The question, and the decomposition that makes it testable

> **Arun:** "I see that we always go into 4–5k of profits within the opening 30min–1hr. How
> about entering double the qty and booking half at +20/25/30/35/40/45/50/55/60/65 minutes —
> any combination which is a sweet spot? Also a simple straddle with combined-SL varied;
> also whether CPR width (today/yesterday/weekly) or gap up/down filtering works better."

**Decomposition (stated up front, it decides the whole study):** "2× qty, book half at T"
= the existing 1× position (completely unchanged — same entry, same stops, same exit)
**plus an independent 1× ATM-straddle scalp of duration T** at the same entry minute.
So the only question is whether that scalp **earns its own full round trip**
(0.5/1.0 pt per leg-side + ₹30/leg-side/lot → **₹250/lot NIFTY, ₹200/lot SENSEX**),
per venue, per DTE, at some T — and nothing about the existing books changes either way.

Scope (the ops-session table, binding): morning entries only — NIFTY 09:16 suite
(ATM per-leg-30%, ATM2 ₹2,500/lot rupee stop, ATM4 per-leg-30%, COMB CSL 25/30/30/20 by
DTE), NIFTY TimeB Tue 09:30 SL25 and Fri 10:00 SL20, SENSEX 09:16 suite (ATM/ATM4
per-leg-30% with leg-SL off on DTE0 per r/114, ATM2 ±0.4% move-stop, CSL30F Wed), TimeB
SENSEX Wed 10:30 SL20. Afternoon cells excluded.

**Harness validation (before anything was believed):** the three deployed morning TimeB
windows replayed by this study's code reproduce the r/122 atlas **exactly to the rupee** —
TUE n=16 med +9,525 mean +7,552 win 81%; FRI n=15 med +3,120 mean +3,871 win 93%; WED n=17
med +3,370 mean +1,402 win 71% (all @10 lots).

## 2. STAGE 0 — the premise, verified first

For every recorded day (real 1-min chain, 2026-04-27 → 2026-08-21, holiday/partial guards
from r/120–122), the **gross open P&L of the aggregate morning book** at wall clock
09:36→10:21, at live sizes (NAS sleeves 3 lots, COMB Thu 5, TimeB 8, CSL30F-Wed 3), with
each system's own stop applied. "We always go ₹4–5k up" is **materially false as stated**:

| Weekday | best median in the hour | worst median | % of days ≥ ₹4k (range over the hour) | p10 (typical) |
|---|---|---|---|---|
| Mon | +₹7,740 (10:06) | +₹3,708 | **47–71%** | −₹300…−3,400 |
| Tue | +₹10,243 (10:16) | −₹1,334 (09:36) | 19–69% | **−₹15,256 → −₹3,096** |
| Wed | +₹378 (10:16) | **−₹2,592** | **12–24%** | −₹9,900…−14,400 |
| Thu | +₹11,019 (10:16) | +₹1,223 | 38–62% | −₹7,700 → **−₹32,618** |
| Fri | −₹2,808 (09:41) | **−₹5,800 (10:01)** | **0–33%** | −₹6,600…−12,600 |

- The feeling is real on **Monday** and on **late-morning Tue/Thu** (~60–70% of days) — and
  those are the days the impression was formed on. It is **flatly false on Wednesday and
  Friday**, where the aggregate morning book's *median* is negative for most of the hour.
- Per book, the only sleeves that individually look like "₹4–5k up" are **TimeB Tue 8L**
  (median +₹4,276 → +₹6,840 across 09:56→10:16) and **COMB Thu 5L** (+₹3,445 → +₹4,958).
  The per-leg-30% sleeves (ATM/ATM4, 3L) have median open P&L ≈ **zero** through the whole
  opening hour on their live days; Wed's CSL30F median is negative.
- Tue at 09:36 has p25 = −₹10,946: the expiry-morning book is often deep under water in the
  first 20 minutes before the decay arrives — the exact opposite of a scalp-friendly open.

The premise being weekday-specific does not by itself kill the idea (the scalp could still
pay on the good mornings), so the sweep was run in full.

## 3. The (T × system) sweep — no sweet spot exists

400 cells: 5 venue-entries × 10 horizons × 8 defence arms (bare / CSL15/20/25/30 /
per-leg-30% / ₹2,500-rupee / 0.4%-move), every recorded day, net of the scalp's own round
trip. **Zero cells reach t ≥ 2 (let alone the ~3.9 a 400-family Bonferroni asks); 67/400
even have a positive mean.** Doubling the systems as they are actually configured, on
their live DTEs:

| Doubled system (arm at 2×) | best T | mean net ₹/lot (t) | worst T (t) | verdict |
|---|---|---|---|---|
| NIFTY ATM/ATM4, per-leg-30% (n=48) | T=65: **−180** (−1.3) | T=20: −337 (**−3.8**) | negative at EVERY T — the per-leg stop whipsaws the scalp |
| NIFTY ATM2, ₹2,500 rupee stop (n=48) | T=60: +194 (+1.6) | T=20: −245 (−2.4) | best of the 09:16 family, still ~₹580/day at 3 lots, n.s. |
| NIFTY COMB DTE0 SL25 (n=16) | T=60: +277 (+0.8) | T=20: −377 (−1.5) | n.s. |
| NIFTY COMB DTE1/2 SL30 (n=32) | T=60: +66 (+0.6) | T=20: −183 (−2.0) | n.s. |
| NIFTY COMB Thu DTE3 SL20 (n=16) | T=45: +180 (+0.8) | T=20: −34 | n.s. |
| TimeB Tue 09:30 SL25 (n=16) | T=65: **+515 (+2.07)** | T=20: −44 | the "best cell" — see below |
| TimeB Fri 10:00 SL20 (n=15) | T=60: +86 (+1.2) | T=20: −190 (−2.2) | n.s. |
| SENSEX ATM/ATM4 Wed per-leg-30% (n=17) | all T negative | T=25: −282 (−2.3) | negative at EVERY T |
| SENSEX ATM/ATM4 Thu no-leg-SL (n=17) | T=60: +12 (+0.03) | T=35: −271 | ≈ zero |
| SENSEX ATM2 move-stop (n=34) | all T negative | T=20: −397 (**−2.9**) | negative at EVERY T |
| SENSEX CSL30F Wed (n=17) | all T negative | T=20: −581 (**−3.3**) | the worst doubling in the study |
| TimeB SENSEX Wed 10:30 SL20 (n=17) | T=65: +202 (+1.9) | T=30: −42 | n.s. |

Three structural facts explain the table:

1. **T ≤ 30 minutes is a cost machine.** At every entry, the first 20–30 minutes have not
   yet accrued enough decay to pay ₹250/₹200 + the opening chop; the short-T cells are the
   most *significantly negative* cells in the study (t −2.9 to −4.9). The very horizons the
   question hoped for ("book half at +20/25/30") are the worst ones.
2. **The scalp inherits the defence's pathology at double density.** The per-leg-30% arm is
   negative at all T on both venues: an ATM leg rising 30% intra-morning is common, and the
   scalp keeps paying that whipsaw without the full-day recovery the 1× book gets. (r/114/
   116/121's "stops convert decay into booked losses", reproduced on a 65-minute clock.)
3. **The only cell with any lean, TimeB Tue T=65 (+515/lot, t 2.07, n=16, worst −2,704), is
   not a scalp discovery at all** — it is 72% of the deployed Tuesday window (09:30–10:35 of
   09:30–11:00) at extra size. Its own full window earns **+755/lot** on the same margin
   (r/122: mean +7,552 @10L, R:R@p95 1:1.5). The sweep's best find is *strictly dominated
   by simply adding a lot to the existing cell* — i.e. by the null alternative the success
   criterion said any recommendation must beat. And t 2.07 against a 400-comparison family
   on 16 days is noise by this project's own standard (r/120's Westfall–Young precedent).

Generic-straddle answer (the "clean" ask): with combined-SL 15/20/25/30/none the CSL rarely
fires inside 65 minutes at all (see §4), so the arms collapse onto the bare scalp — which is
itself t ≤ +1.66 pooled, and negative before T=45. There is no (T, SL) sweet spot either.

## 4. Stage B — the tail the scalp would carry (long price sample)

Excursion inside entry→entry+T from SENSEX 1-min 2021→ (1,354 days) and NIFTY50 5-min
2015→ (2,754 days; r/121 max-excursion licence — NIFTY entry is the first 5-min bar ≥
start, a ~1-bar imprecision), DTE-matched to each cell's live days, bridged to rupees via
the 2026 premium-rise slope and the r/122 credit ladder:

| Cell (live DTEs) | n_px | exc p90/p95/p99 bp | bridged p95 / p99 ₹/lot | deployed-stop trip freq |
|---|---|---|---|---|
| NIFTY 09:16 T=60 | 1,065 | 65 / 81 / 141 | 1,512 / 2,440 | CSL20 0.6%, CSL25 0.4% |
| NIFTY 09:30 T=65 (Tue) | 358 | 60 / 72 / 102 | 1,055 / 1,394 | CSL25 0.0% |
| SENSEX 09:16 T=60 | 246 | 63 / 69 / 98 | 1,243 / 1,667 | ~0% |
| SENSEX 10:30 T=65 (Wed) | 124 | 50 / 58 / 105 | 1,099 / 1,826 | ~0% |

- The tail is *survivable* — that is not the problem; the expectancy is. But the bridge is
  a **floor**: the recorded sample already contains scalp days worse than its bridged p99
  (e.g. −₹4,764/lot at NIFTY 09:16 T=35 bare, −₹2,765 ATM2-arm, −₹5,127 SENSEX CSL30 T=65)
  because opening-hour spikes come with IV pops the linear bridge does not charge (r/122 §4
  caveat, reproduced). Doubled morning size on a 2021-04-style day would wear ~2× those
  numbers per lot across every sleeve simultaneously.
- Combined-SLs 15–30% almost never trip inside 65 minutes on non-catastrophic days (the
  morning credit is large relative to a <1-hour move), so a "tight CSL" is not a defence
  for this scalp — it is decoration, exactly as r/121 found for the full windows.

## 5. Filters — CPR width (today / yesterday / weekly), gap up/down

Same discipline as r/121: fit on the long sample, confirm on options days, beat an exact
random-skip null of equal frequency, monotone terciles, pre-registered list, placebos in
the same pipeline. Confirmation cells pre-registered as the three least-bad scalps (Tue
TimeB 2×, Wed S-TimeB 2×, ATM2 2× T60).

- **Long-sample fit:** every CPR/gap feature predicts the raw opening move only weakly
  (best Spearman vs excursion: cpr_today +0.31–0.36 on NIFTY, +0.11–0.14 on SENSEX; gap
  direction ≈ 0). r/121 already showed the option market prices this regime — and these
  rhos are *half* the size of the ones that collapsed there.
- **Options-day confirmation: 0 winners out of 33 non-placebo rules** (~1.7 expected by
  chance). Best real rule reached the 94.4th percentile of its null (skip-gap-up on Tue
  TimeB) and fails the ≥95 gate; most CPR terciles are non-monotone.
- **The placebo won again:** `placebo_dom_odd` — skip odd calendar days — "beat 97.0% of
  random skips and retained 109% of P&L" on Tue TimeB. A day-of-month parity bit with zero
  information outperformed every real CPR/gap rule on the same cell. That single line is
  the correct weight to put on any 16-day filter table.
- r/67's daily-vs-weekly CPR sign-flip caution was moot: neither sign cleared the null.

**Filters: NO.** Neither CPR width (any lookback) nor gap direction/size improves the scalp.

## 6. Margin and the null alternative

Doubling every morning entry concurrently (NIFTY ₹1.65L/lot, SENSEX ₹2.04L/lot, capital
₹44.7L; assumed lots: NAS sleeves 3, COMB-Thu 5, TimeB 8, CSL30F-Wed 3):

| Weekday | base peak | doubled | fundable? |
|---|---|---|---|
| Mon | 19.8L | 39.6L | **yes — and Monday is the only day** |
| Tue | 33.0L | 66.0L | no |
| Wed | 40.8L | 81.6L | no |
| Thu | 26.6L | 53.2L | no |
| Fri | 33.0L | 66.0L | no |

The one fundable doubling (Monday NIFTY 09:16 suite) is a *negative-expectancy* scalp under
its own stops (§3, rows 1–4 at DTE1). And the standing null alternative — put the same
extra margin into **more size on TUE TimeB**, the book's best-priced risk (mean +₹755/lot
per Tuesday, R:R@p95 1:1.5, r/122) — beats every cell in this study including the sweep's
own maximum (+515/lot, which is the same trade held 25 minutes shorter). The recommendation
of r/121/122 (the honest size lever is *cutting Monday*, not adding morning size) stands.

## 7. Sins accounting

| Sin | Control |
|---|---|
| Look-ahead | Strike from the entry-minute spot; stops evaluated minute-forward; features (CPR/gap) computed from prior-day/-week bars only; expiry/DTE from the chain per day |
| Survivorship / selection | Every recorded day used; r/120–122 holiday guard (2026-05-01/05-28/06-26 rejected by data rule); premise checked on ALL days before any sweep |
| Overfitting / multiple testing | Whole 400-cell surface reported; ~400-comparison family named; best t 2.07 explicitly discounted; filters pre-registered with exact random-skip null + placebos — and the placebo won |
| Cost neglect | The scalp pays its own full RT (₹250/₹200 per lot) on every cell; Stage-0 marks labelled gross (screen P&L) |
| Regime dependence | Tails from 2015→/2021→ DTE-matched samples, not the 2026 quarter; one-benign-quarter caveat on all observed win rates |
| Correlation / single factor | The scalp is the SAME short-gamma family as the 1× book at the same minute — doubling is leverage, not diversification, and is treated as such throughout |
| Capacity / margin | §6: doubled peak vs ₹44.7L per weekday; only Monday fundable |
| Bridge honesty | Bridged tails labelled floors; in-sample counter-examples (observed worsts > bridged p99) shown |

## 8. Caveats

1. n = 15–48 days per live cell, one benign quarter, for every rupee number; the long
   sample exists precisely because of that.
2. Stage-0 marks use recorded LTPs and no exit costs — they estimate the screen P&L, not a
   bookable profit; booking it costs the §3 round trip, which is the whole study.
3. The scalp's per-leg/trail mechanics are simplified (no ST-trail or MAXV roll inside
   65 min; single-entry, 1-min LTP, fixed slippage; no dwell). These simplifications
   *flatter* the scalp — the real fills would be worse.
4. Lots per sleeve (3/5/8) are the assumed live sizes; per-lot numbers are primary and are
   size-invariant.
5. NIFTY long-sample entries carry a ~1-bar (5-min) start imprecision (licence in §4);
   SENSEX is exact to the minute.
6. DTE-era labels ignore intra-week holiday shifts in the long sample (±1 label noise);
   the 2026 chain derives expiry per day and has no such error.

## 9. Files

| File | Purpose | Committed? |
|---|---|---|
| `scripts/stage_a_scalp.py` | chain replay: scalp grid + Stage-0 marks + r/122 recon cells | yes |
| `scripts/stage_b_scalp.py` | long-sample excursions per entry×T + CPR/gap features | yes |
| `scripts/analyze_all.py` | premise tables, sweep, bridge, filters+null+placebos, margin | yes |
| `results/stage0_premise.csv` | aggregate + per-book open-P&L distributions | yes |
| `results/sweetspot.csv` | all 400 cells, ALL-DTE + live-DTE scopes | yes |
| `results/tail_bridge.csv` | exc percentiles + bridged ₹ + stop-trip freq per cell | yes |
| `results/filters_report.txt` | long-sample fit, all 33 rules + placebos vs null | yes |
| `results/margin_null.txt` | margin arithmetic + null-alternative statement | yes |
| `results/stage_a_scalp.csv` (4.6 MB) | per-day per-cell scalp replays | no (gitignored) |
| `results/stage_b_scalp_days.csv` (8.2 MB) | per-day long-sample windows + features | no (gitignored) |
| `results/RESULTS.md` | this report | yes |

**Reproducibility stamp.** Data snapshot 2026-08-23 (`options_data.db` 2026-04-20→08-21,
`market_data.db`, both opened `mode=ro`). All runs `nice -n 10` on the VPS. Costs ₹250/lot
NIFTY, ₹200/lot SENSEX round trip. Deployed cells cross-checked against
`backtest_data/csl_paper_config.json` (refrozen 2026-08-20). No live config, service,
engine, order path, or frontend was touched. Harness reconciled against r/122 to the rupee
before publication.
