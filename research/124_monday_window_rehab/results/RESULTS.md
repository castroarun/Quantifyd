# research/124 — Monday Window Rehab — RESULTS

**Verdict: NO EDGE — CONCLUDED. No Monday window ≤120 min × stop combination clears the
pre-registered gates. Monday stays dark. The only family-wise-significant Monday cells are
reliable LOSERS (lunchtime windows with a tight ₹500 stop). Recommendation: change nothing.**

Study: does ANY Monday ATM-short-straddle window (≤120 min, exit ≤15:20) under ANY stop
(combined-premium % 10–40/none OR rupee ₹500–2500/lot) clear R:R@p95 better than 1:3 with
workable P(win), net of costs, after multiple-testing discipline, with long-sample tail
agreement? Grid: 137 windows × 11 arms = **1,507 cells/venue** (NIFTY DTE1 primary,
SENSEX DTE3 secondary), n = **17 clean Mondays** of real 1-min chain (2026-04-27→08-17;
04-20 partial, holidays guarded), long sample NIFTY 5-min 2015→ (547–557 Mondays/window,
r/121 excursion licence) + SENSEX 1-min 2021→ (280), DTE-era-labelled (r/118 history).
Costs: r/122's exact model (₹250/lot RT NIFTY, ₹200 SENSEX). Sizing: 8 lots (520/160).

**Reconciliation gate PASSED before interpretation:** harness reproduces r/122's
stage_a_alldays rows **82/82 days to the rupee** on all three checked cells (NIFTY
13:00–14:00 SL20, NIFTY 10:00–12:00 SL20, SENSEX 10:30–12:00 SL20).

## 1. The gate cascade (the whole study in one table)

| Gate | NIFTY (1,507 cells) | SENSEX (1,507 cells) |
|---|---|---|
| G1 med>0 & win≥60% AND G2 bridged R:R@p95 < 1:3 | 137 | 43 |
| + G3 modelled P(loss) ≤40% (long sample) | 113 | 43 |
| + G4 plateau (≥3 window-neighbours pass; no isolated peaks) | 36 | 10 |
| + G5 Westfall–Young family-wise (2,000 sign-flip draws) & G6 label-shuffle null | **0** | **0** |

- **G5:** family max\|t\| observed 4.69 (NIFTY) / 4.23 (SENSEX) vs null-95 of max\|t\| 4.52 /
  4.20 — exactly ONE cell per venue clears, and both are **negative**: NIFTY 12:00–12:30
  R500 (t −4.69, median −₹2,024@8L, win 5.9%) and SENSEX 12:15–12:45 R500 (t −4.23, median
  −₹1,888, win 11.8%). The only statistically certain Monday facts are ways to LOSE: a
  lunchtime window collects ~zero decay (r/120's inversion) while a ₹500/lot stop + the
  round trip whipsaw-charges you. **No positive cell reaches the family bar** (best positive
  t = 3.83–3.43 vs bar 4.52).
- **G6 (the decisive null, r/121/123 discipline):** best Monday cell metric (max median@8L
  over cells with win≥60%) = **+₹5,880** vs best-cell over 2,000 random 17-day draws from
  the same 82-day/grid: null median +₹5,408, null-95 +₹7,280 → **empirical p = 0.329**.
  SENSEX: +₹3,392 vs null-95 +₹7,744, **p = 0.969** — Monday's best is *worse* than the
  typical data-mined best. Monday's "winners" are indistinguishable from grid-mining noise.

## 2. Top-5 cells (best of 3,014, all FAILED) + reference rows — @8 lots, net

| Window (venue) | Stop | Median | P(win) | R:R@p95 bridged / empirical@worst | Plateau | Long-tail agreement |
|---|---|---|---|---|---|---|
| 09:16–11:16 (NIFTY) | R1000 | +5,880 | 82.4% | 1:1.0 / **1:1.8** (worst −10,688, SL fired) | Y 3/3 | **NO — bridge broken** (see §3) |
| 09:16–11:16 (NIFTY) | NOSTOP | +5,880 | 82.4% | 1:1.0 / **1:3.7** (worst −22,016) | Y 3/3 | NO — same |
| 13:15–15:15 (NIFTY) | NOSTOP | +4,552 | 82.4% | 1:1.34 / 1:1.02 (worst −4,656) | **N 1/5 — isolated peak** | weak (be_term fallback; plm 0.3% not credible) |
| 09:16–11:01 (NIFTY) | R1000 | +4,632 | 64.7% | 1:1.21 / 1:2.3 | Y 3/4 | NO — same bridge |
| 13:15–15:15 (SENSEX) | any (stop never fires) | +3,392 | 70.6% | 1:1.49 / 1:1.7 (worst −5,728) | Y 3/5 | plm 2.3%, BUT G6 p=0.97 |
| **dropped live cell** 13:00–14:00 (NIFTY) | SLP20 | **+992** | 70.6% | **1:12.19** / plm 52.1% | — | reproduces r/122's 1:11.8@10L |

Every one of the top cells fails G5 AND G6. SLP25 on the morning window has a WORSE worst
(−25,632) than NOSTOP (−22,016) — the **4th independent reproduction** of r/114/116/121's
"tightening the stop makes the worst day worse" (fire-then-revert / overshoot; pooled stop
overshoot p95 ≈ ₹872–1,246/lot beyond the theoretical cap).

## 3. The bridge is broken for morning windows — the "1:1.0" R:R rows are not real

The bridged R:R@p95 (r/122 method, kept for comparability) says the 09:16–11:16 cell risks
only ~₹5,900@p95 against a +₹5,880 median. Its own 17-day sample contains **2026-07-13:
−₹22,016 on a 57.7bp excursion** — the premium rose ~21% of credit on a ~0.6% move, ~10×
what the excursion→premium slope (b=0.00036/bp) predicts. Morning straddle losses are
**IV-pop-driven, not excursion-driven**; an excursion bridge is a hard FLOOR there, and the
floor is already breached inside n=17. Empirical R:R at the observed worst (which at n=17
is only ≈p94!) is 1:3.7 unstopped — right where the 1:3 bar sits, before any true tail.
This also retro-explains why r/122's atlas showed Monday morning/late "dominators": floors,
not tails.

## 4. The calmness clock — the "calm Monday zone" does not exist

- **Monday is NOT a calm day.** NIFTY 09:16–11:16 excursion p50 across 547–557 days/weekday:
  Mon 38.3bp / Tue 37.2 / Wed 35.8 / Thu 36.8 / Fri 37.8 — Monday is the *widest* morning.
  13:00–14:00: all weekdays 23–24bp p50, Monday nothing special. SENSEX similar (Monday
  morning p50 43.8bp = widest of the week).
- The calm zone that DOES exist is intra-day (lunch ~12:00–13:30, everywhere, every weekday)
  and it earns nothing — r/120's decay/danger inversion, reproduced: the G5-significant
  losers sit exactly there.
- Current-era Mondays (NIFTY n=39 since 2025-09, SENSEX n=50) *look* calmer (morning p95
  79.7bp vs 101.5 full-sample) — that is one benign regime, not a structure; G7 forced the
  worse (full-sample Monday) percentile into every R:R, and the cells still passed G2 —
  the failures are statistical (G5/G6), not tail-arithmetic. The gates are independent kills.

## 5. G8 — the null alternative, stated

If believed at face value, the morning cell (+₹5,880/Mon) would out-earn adding its ~₹24L
margin as +2 lots on TUE_NIFTY_DTE0 (+₹1,905/Tue at r/122 medians) + FRI (+₹624/Fri). But
it cannot be believed: p=0.33 vs data-mined noise, a within-sample −₹22k day (vs TUE's
1:1.5 priced risk), and a bridge its own sample breaks. The Tue/Fri cells passed three
independent studies; this cell cannot pass one. Not comparable claims.

## 6. Caveats

n=17 Mondays, one benign 4-month regime, ~3,014-cell family (hence G5/G6 were always going
to be the real judges); long sample is excursion-only (r/121 licence) and cannot see IV
pops (§3 shows they dominate morning risk — the true tail is WORSE than every number here);
NIFTY long sample is 5-min (licenced for max-excursion equality); DTE-era labels ignore
holiday-shifted weeks (±1 day label noise); SENSEX pre-2024 Mondays carry no weekly-expiry
label. All of these caveats point the same direction: against deployment.

## 7. Recommendation

**Monday stays dark. No live change. No paper book.** The paper twin CSL_TIMEB_NIFTY_MON
already covers the only question left (does the old cell's live-sample luck continue). If
anything is ever revisited on Mondays it should be (a) only after ~40+ recorded Mondays
exist, and (b) the 09:16–11:16 R1000 construction is the one to re-examine (it was the
family's best t at 3.43 and its rupee stop genuinely capped the one blowout day) — but
today it is statistically indistinguishable from noise, and this study is its kill-sheet.

*Artifacts: `results/monday_atlas.csv` (3,014 rows, all gate columns), `results/gates_report.txt`
(nulls + cascade), `results/calmness_clock.csv`, `results/percentiles_long.csv`. Heavy per-day
CSVs gitignored; regenerate via the two runner commands in the STATUS-MD §6.*
