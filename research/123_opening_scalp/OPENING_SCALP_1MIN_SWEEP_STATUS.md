# Opening-Hour Double-Entry — Is Booking Half Early a Real Edge, and Where Is the Sweet Spot?

STATUS: **DONE** (2026-08-23; verdict NO EDGE - CONCLUDED)

## 2. The Ask

**What Arun asked (2026-08-23):** "I see that we always go into 4-5k of profits within the
opening 30min-1hr. How about entering double the qty of what we have now and booking half in
this opening 20/25/30/35/40/45/50/55/60/65 minutes - any combination which is a sweet spot?
You can also consider a simple straddle with comb SL which you can vary and test. Also see if
combining CPR width (today/yesterday/weekly), gap ups/downs filtering works better. Test on
our systems - either one or combinations."

**The decomposition that makes this testable.** "2x qty, book half at T" = the current 1x
position PLUS an independent 1x opening scalp of duration T. The deployed books are
unchanged by construction; the question is purely:

> Does a T-minute ATM-straddle scalp from the morning entry carry positive expectancy net of
> its own round trip - per venue, per DTE - and at which T? And do CPR/gap filters improve it?

## 3. Stage 0 - VERIFY THE PREMISE FIRST

"We always go 4-5k up in the opening hour" is an impression formed in a benign week. Before
any sweep: for every recorded day, the distribution of open P&L at entry+20..+65 min for the
deployed constructions. Report the median AND the p10/p25 - if "always" is actually "70% of
days", the story changes. If the premise is materially false, say so and stop there.

## 4. The Base

- **Constructions:** (a) each deployed morning entry as it exists (9:16 suite entry, COMB
  09:16, TimeB cells with morning starts); (b) the clean generic: 09:16 (and each TimeB
  start) ATM straddle with combined-SL swept 15/20/25/30/none.
- **Scalp exits:** T = +20/25/30/35/40/45/50/55/60/65 minutes from entry.
- **Data:** Stage A `options_data.db :: option_chain` 1-min 2026-04-20 -> now, READ-ONLY,
  both venues, per DTE (~85 days/venue; the scalp trades every day, so n is the full set,
  not 16-day cells). Frozen-chain holidays rejected (05-01, 05-28, 06-26).
  Stage B `market_data.db`: the opening window is the WILDEST part of the session (r/120
  clock) - price the scalp's tail from the long sample (SENSEX 1-min 2021->, NIFTY 5-min
  2015-> under the r/121 max-excursion licence): excursion percentiles inside entry->entry+T,
  gap-day behaviour, and P(the scalp's stop/backstop is hit) per T.
- **Costs:** the scalp pays its OWN full round trip (0.5/1.0 pt per leg-side + Rs30/leg-side
  per lot). This is the hurdle that kills most short-duration constructions - charge it
  honestly.
- **Margin reality:** 2x entry qty doubles peak morning margin across every venue
  simultaneously. Report peak concurrent requirement for any recommended combination
  (NIFTY ~1.65L/lot, SENSEX ~2.04L/lot, capital ~44.7L) - a sweet spot we cannot fund on
  Thursday morning is not a recommendation.

## 5. Filters (CPR width today/yesterday/weekly, gap up/down)

r/121 just tested day-level filters for the TimeB windows and found NOTHING (10 winners of
540 rules vs ~27 expected by chance; a Gaussian placebo beat 97% of random skips). This is a
different target variable (opening-scalp P&L, n~85/venue rather than 16), so it is testable -
but under the same discipline: fit on the long sample, confirm on options days, every filter
must beat a random-skip control of equal frequency, monotone response curves only,
pre-registered list, family-wise haircut, and the r/67 sign-flip caution (daily vs weekly CPR
width point OPPOSITE ways). Expect nothing; report honestly if so.

## 6. Success criterion

A (T, construction, venue/DTE-set) combination is a recommendation only if: net-positive
after its own costs with a t-stat clearing the family haircut; robust per-DTE and per-month;
its tail (long-sample p95 inside the scalp window, gap days included) priced and stated; the
margin fundable; and the marginal value survives the fact that the 1x base position already
earns the same decay - i.e. it must beat simply RUNNING THE EXISTING BOOKS AT THE SAME TOTAL
EXTRA MARGIN (the null alternative: 2x on the best existing cell instead).

## 7-8. Standard: read-only, niced, scripts/ + results/ + RESULTS.md with bold verdict, STATUS
log updated, INDEX row 123, commit only the study folder + INDEX. Findings by the agent.

## Plan (locked before launch)

- **Stage A grid:** entries {NIFTY 09:16/09:30/10:00, SENSEX 09:16/10:30} x T {20..65 step 5}
  x arms {NOSTOP, CSL15/20/25/30, PERLEG30, RUP2500, MOVE04} = 5 x 10 x 8 = 400 cells/day,
  ~85 days/venue. Stage 0 = deployed-book marks at wall clock 09:36..10:21 on live-DTE days.
  RECON cells reproduce the three morning TimeB windows vs the r/122 atlas.
- **Stage B:** same entries x T on SENSEX 1-min 2021-> and NIFTY50 5-min 2015->, DTE-era
  labels, + per-day features gap_bp / cpr_t_bp / cpr_y_bp / cpr_w_bp.
- **Filters (pre-registered):** gap (signed, |gap|, direction), CPR today/yday/weekly;
  tercile skip rules; long-sample fit first, options-day confirmation second; every rule vs
  a 2,000-draw random-skip null of equal frequency; placebos: Gaussian noise + day-of-month
  parity; monotone tercile response required; family haircut stated.
- **Success gate:** per STATUS s6 (net + haircut + per-DTE/month robustness + p95 tail +
  fundable margin + beats 2x-on-Tue-TimeB null).

## Status log

| Date/time | Event | Notes |
|---|---|---|
|  IST | Stage A + Stage B launched | 400 cells/day x ~85 days/venue; long sample 2015->/2021-> |
| 2026-08-23 15:45 IST | Stage A done (83+83 days kept) | results/stage_a_scalp.csv |
| 2026-08-23 15:47 IST | Stage 0 + sweep aggregated | Premise weekday-dependent (Wed/Fri FALSE); 0/400 scalp cells reach t2; best t=2.07 = Tue-TimeB-at-more-size (the null itself) |
| 2026-08-23 15:50 IST | Filters + bridge + margin done | 0/33 rules beat the null; the day-of-month PLACEBO "won" at 97%; doubling fundable ONLY on Monday |
| 2026-08-23 16:05 IST | RESULTS.md written, STATUS -> DONE | Verdict: NO EDGE - CONCLUDED |

## Crash recovery (how to resume without Claude)

Everything is already DONE; to regenerate from scratch, in order (all READ-ONLY on DBs):

```
cd /home/arun/quantifyd/research/123_opening_scalp
nice -n 10 python3 scripts/stage_a_scalp.py     # ~2 min -> results/stage_a_scalp.csv (+ stage_a.log)
nice -n 10 python3 scripts/stage_b_scalp.py     # ~3 min -> results/stage_b_scalp_days.csv
nice -n 10 python3 scripts/analyze_all.py       # ~2 min -> stage0_premise/sweetspot/tail_bridge/filters_report/margin_null
```

Safe to inspect: everything under results/. Do NOT touch: backtest_data/*.db (read-only).
Heavy CSVs (stage_a_scalp.csv 4.6MB, stage_b_scalp_days.csv 8.2MB) are gitignored via the
folder .gitignore; all committable outputs are small.

## Findings (final)

1. STAGE 0 - the premise is weekday folklore: "always Rs4-5k up" holds ~47-71% of Mondays
   and late-morning Tue/Thu; on Wednesday the aggregate morning book MEDIAN is negative
   most of the opening hour (12-24% of days reach Rs4k) and on Friday it is negative
   throughout (0-33%). Per book, only TimeB-Tue 8L and COMB-Thu 5L ever look like "Rs4-5k
   up"; the per-leg-30% sleeves have median open P&L ~zero all morning.
2. THE SWEEP - 400 cells (5 entries x 10 T x 8 arms): ZERO cells reach t>=2 vs a
   ~400-comparison family. Doubling the live 09:16 books under their own stops is NEGATIVE
   at every T (per-leg-30% t to -3.8; SENSEX ATM2 move-stop t to -2.9; CSL30F Wed t -3.3).
   T<=30 min - the horizons the question hoped for - are the most significantly negative.
3. Best cell = TimeB Tue T=65 (+515/lot, t 2.07, n 16) which is just 72% of the deployed
   Tuesday window at extra size - strictly dominated by the null alternative (+755/lot on
   the full window, r/122). No sweet spot exists.
4. FILTERS - CPR today/yday/weekly + gap: 0/33 pre-registered rules beat the 2,000-draw
   random-skip null at 95%; the day-of-month-parity PLACEBO beat 97%. NO.
5. MARGIN - doubling every morning entry is fundable only on MONDAY (39.6L of 44.7L), the
   weakest morning; Tue/Wed/Thu/Fri doubled peaks 53-82L all exceed capital.
6. Tail (long sample, bridged): scalp p95 ~Rs1.1-1.5k/lot, p99 ~Rs1.4-2.4k/lot - floors,
   observed worsts already exceed them (to -Rs5.1k/lot); survivable but irrelevant since
   expectancy is absent.

Full report: results/RESULTS.md (verdict NO EDGE - CONCLUDED).
