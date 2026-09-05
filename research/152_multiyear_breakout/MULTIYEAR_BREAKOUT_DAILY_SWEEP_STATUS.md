# Multi-Year Breakout (bananapatterns.com screen) — Does clearing a years-old ceiling pay, and is it just Open Alpha in disguise?

**STATUS: DONE — 2026-09-05.** Verdict: **the screen as published = KILL (Open Alpha in
disguise, 76-93% signal overlap) · the "multi-year" quality itself = NO EDGE (an old ceiling
is subtractive) · the distinctive residual = SIGNAL, real and standalone-viable, but NOT
ADOPTED (fails the pre-registered correlation leg of the complement bar; r/147's gold sleeve
dominates it on risk-adjusted terms at ~zero correlation).**
Full verdict: `results/RESULTS.md`. A post-hoc addendum also found that over
2010-2026 an MYB+OA pair scores above the deployed TN+OA pair on both axes - flagged,
caveated (2008 is excluded and untestable here) and registered for a pre-registered
partner study rather than acted on.
Study: `research/152_multiyear_breakout/` · Started 2026-09-05 · Host: VPS 94.136.185.54 ·
Engine: extension of `research/142_bananapatterns_replication/scripts/bluesky_replay.py`

---

## 1. The Ask

**What Arun asked (via the main session):** test bananapatterns.com's **"Multi-Year Breakout"**
screen — a stock clearing a price level that has capped it for years. Reuse the r/142 harness
that already decoded this site's engine trade-exactly for its "Blue Sky" screen. The site's
exact dials and its published headline numbers for *this* screen were not legible in the
screenshots, so treat their settings as a **sweep axis**, not a spec to match, and leave a
replication gate that can be bolted on if the dials arrive mid-run.

**What we are actually testing.** Three questions, in priority order:

1. **Is a multi-year-high breakout a tradeable signal in Indian cash equities at all**, net of
   25-60 bps/side and after Indian tax (20% STCG / 12.5% LTCG, FY loss-netting)?
2. **THE DESIGN QUESTION — is it just Open Alpha (OA) in disguise?** OA (live, ₹10L book) buys
   a close above the prior **all-time-high** close. A multi-year high that is *also* an
   all-time high is literally an OA signal. So the study splits the signal three ways:
   **(a) inclusive** (ATH allowed), **(b) exclusive** (the multi-year high is NOT an all-time
   high — the distinctive case), **(c) ATH-only** (control, approximately OA). We measure
   **signal-date and holding overlap with the OA signal set**, not just return correlation.
3. **Does it earn a place in the deployed book** — i.e. does it beat the TN+OA 50-50 baseline
   (27.2% CAGR / −16.4% DD / 1.65 Calmar, after tax)?

**Pre-registered kill rule (written before any run):** if variant (a) inclusive shows
>= 60% signal overlap with OA **and** variant (b) exclusive fails the standalone bar
(§2 gates), the family is a **KILL as a duplicate of a deployed system** — the r/145
full-universe-TN precedent.

---

## 2. The Base — exact mechanics

### Universe and eligibility (identical to the validated r/142 mechanics)
- All NSE daily symbols in `backtest_data/market_data.db` (`timeframe='day'`) with >= 260 rows.
- **Liquidity floor:** 20-day median traded value (close x volume) at **t-1** >= **₹5 cr**.
- **ETFs excluded** (`BEES|ETF|LIQUID|GILT|SENSEX|NIF*50` regex, r/142's `ETF_RE`).
- **RS >= 70**: IBD-weighted percentile of `2*r63 + r126 + r189 + r252`, ranked across eligible
  names, **shifted 1 day** (causal).
- **NEW, specific to this screen — minimum history:** a symbol is only eligible for an
  N-year-breakout signal once it has **>= N x 252 prior daily rows**. Without this a
  "5-year high" can be printed by a stock with 8 months of data.

### The signal
For lookback `N` years (`W = N x 252` trading days):

```
PIV[t]  = max(P[t-1 .. t-W])                 # P = close (arm A) or high (arm B)
setup   = prev_close < PIV  and  prev_close >= (1 - maxdist) * PIV    # maxdist default 20%
age     = PIV was set at least X months ago, computed exactly as
          max(P[t-1-X .. t-W]) >= PIV        # i.e. no new W-window high in the last X days
trigger = setup and RS>=70 and liquid and (close > PIV)
```

### The ATH-overlap variants (`ATHp[t] = max(close[..t-1])`, all history)

| Variant | Condition added | Meaning |
|---|---|---|
| **incl** | none | any multi-year high, ATH ones included |
| **excl** | `PIV < ATHp` **and** `close <= ATHp` | clears a years-old ceiling while still **below** its all-time high — the distinctive case |
| **athonly** | `PIV >= 0.999 * ATHp` | the multi-year high *is* the all-time high — control, approximately OA |

### Entry mechanic
Buy-stop **at the pivot**, filled `max(PIV, open)` (the r/142-validated realistic fill).
Alternative arm: fill at the **signal-day close** (their "Breakout close" dial). r/142 showed
this single choice moved the result x536 vs x14.4, so both are run wherever it can flip a verdict.

### Exits (tested jointly with entry, never in isolation)
- **Hard stop on CLOSE**: -7% / -8% / -10% from fill.
- **Trail on CLOSE**: below SMA-15 / SMA-30 / SMA-50 / SMA-150 ("30-week"), not on entry day.
- **Take profit**: +25% (their third dial), tested alone and combined with a trail.

### Book / sizing (two sizing families)
- **Fixed-% of NAV (our house mechanic):** slots x size%: 3x30% / 5x20% / 8x18.75% / 10x10% /
  16x6.25%; hard cap 30% of equity; cash-constrained.
- **Risk-based (the site's mechanic):** position value = `(risk% x equity) / stop%`, **capped at
  30% of equity**; risk in {1, 1.5, 2}%. (Their panel: 2% risk with a 7% stop implies about
  ₹2.86L on ₹10L.)
- Capital ₹10,00,000. Costs 25 / 40 / 60 bps per side. Idle cash 5% p.a.
- **Tax: 20% STCG / 12.5% LTCG on net realised gains, netted within the Indian FY** (settled
  1 April) — an extension of the r/142 engine, which netted per calendar year.
- **Weak-market gate** (their "SKIP WEAK MKTS"): NIFTYBEES < SMA-200 blocks new entries,
  computed NaN-robustly on the dropna'd series then re-aligned (the r/142 phantom-row scar).

### Windows (two-window discipline — both must pass)

| Window | Range | Why |
|---|---|---|
| **W1 "their window"** | 2020-01-01 → 2025-12-31 | the site's period; a bull sample |
| **W2 "long"** | 2010-01-01 → 2026-09-04 | longest window the data supports for N <= 5 |
| **W2b (N=10 only)** | 2015-01-01 → 2026-09-04 | only 488 names had 10y of history by 2015 |

**Data reality (measured 2026-09-05, printed before design was frozen):** 2,671 daily symbols,
2000-01-03 → 2026-09-04, 6.68M rows. The universe is only 2 symbols in 2000-2002, 33-37 in
2003-04, and **527 from 2005**. Names with >= N years of prior history: N=5 → 527 by 2011,
756 by 2013; N=10 → 2 by 2013, **488 by 2015**, 1,065 by 2025. **A multi-year-high screen is
therefore history-starved before 2010 (N<=5) / 2015 (N=10)** — this is the binding constraint
on the window, and the reason W2 starts in 2010 rather than 2006 like OA's study.

### Data-integrity handling (this screen is uniquely exposed)
Multi-year highs are computed from long lookbacks, so a **price-scale break inside the lookback
window manufactures or suppresses breakouts**.

- **Phantom holiday rows: CLEAN.** Scanned every trading day 2010→2026 for the signature
  (row count < 50% of the local 21-day median **and** > 85% zero-volume): only **2 days**
  (2014-04-24, 2014-10-15); the 2026-01-15 purge is intact.
- **Split-scale defect: PRESENT, 176 events.** Scanning 2,143 symbols with >= 500 rows for
  1-day close ratios near clean split factors (2, 2.5, 3, 4, 5, 10 +/-12%) found 176 candidate
  unadjusted corporate-action steps (e.g. `GVPTECH` x5.48 on 2022-09-22, `POCL` x5.54 on
  2023-03-06, and a 10-name cluster on 2026-05-18 that looks like a re-fetch boundary).
  **Mitigation: scale-break blackout.** For each detected event at date `d` on symbol `s`,
  `s` is made **ineligible for entries from `d - N years` to `d + 20 days`** — the whole span
  over which its lookback window straddles the break. Conservative: it can only remove
  signals, never create them. The sweep reports signal counts **with and without** the
  blackout so its cost is visible.

### Success criterion / gates — PRE-REGISTERED before running
- **G1 ranking metric:** after-tax net **CAGR at 25 bps, 10-seed median, on W2**, subject to
  hard gates: (i) after-tax CAGR > NIFTYBEES buy-and-hold in **both** W1 and W2;
  (ii) net per-trade expectancy > 0; (iii) >= 40 trades in W2.
- **Standalone adoption bar (G4):** 30-seed median after-tax CAGR >= 20% **and** Calmar >= 0.8
  on W2, worst seed CAGR >= 15%, and survival of the outlier-deletion test (drop top-10 trades).
- **Complement adoption bar (G4, per the playbook):** vs the TN+OA 50-50 baseline —
  **+0.10 Calmar or -2pp max drawdown at >= equal CAGR, after tax**, robust across OA seeds and
  TN offsets, **correlation < 0.4 to both legs**, and beats the **cash-null** at the same weight.
- **Falsification:** if no cell clears G1 in both windows, the family is **NO EDGE** and the
  study stops at G1.

---

## 3. Plan — grid and cell counts

**Phase A — signal inventory + OA overlap (cheap, no book).** For each of the 72 signal
matrices below, count signals/yr and measure exact (symbol, date) overlap with the OA
signal set. The kill rule can fire here.

**Phase B — G1 sweep (72 cells x 10 seeds).** Fixed default book to isolate the *signal*:
16 slots @ 6.25%, -8% close stop, SMA-50 trail, gate OFF, realistic fill, 25 bps, after-tax,
cash 5%. Run on **W2** and **W1**.

| Axis | Values | n |
|---|---|---|
| Lookback N (years) | 2 / 3 / 5 / 10 | 4 |
| Level series | highest **close** / highest **high** | 2 |
| ATH overlap | incl / excl / athonly | 3 |
| Pivot age (base must have held) | 0 / 6 / 12 months | 3 |
| | **Phase B total** | **72** |

**Phase C — G2 mechanics on survivors only** (only families that clear G1 in both windows):

| Axis | Values | n |
|---|---|---|
| Stop | 7 / 8 / 10 % | 3 |
| Exit | trail-15 / trail-30 / trail-50 / trail-150 / TP+25% / TP+25% with trail-50 | 6 |
| Sizing | fixed 3x30 / 5x20 / 8x18.75 / 10x10 / 16x6.25 · risk-based 1 / 1.5 / 2 % | 8 |
| Gate | OFF / ON (NIFTYBEES 200-SMA) | 2 |
| Fill | pivot buy-stop / signal close | 2 |
| Base quality | maxdist 12/20/30% · tightness none/0.35/0.25 | 9 |

Run as **OFAT + plateau neighbourhood**, not a full cross (a full cross is 5,184 cells x
10 seeds — unaffordable and a multiple-testing disaster). Cell count disclosed in RESULTS.

**Phase D — robustness and adoption (survivors only):** 30-seed ensemble, cost ladder
25/40/60, outlier-deletion (drop top-10 trades; cap winners at +50%/+100%), per-window rows
(2020 crash, 2018 grind, 2022H1 grind), capacity vs held names' traded value.

**Phase E — portfolio fit:** daily and monthly correlation to OA and TN; 3-sleeve blend weight
sweep 10-33% across OA seeds x TN offsets (reusing `research/146/scripts/blend3.py`);
cash-null at the same weight; OA **trade-level** overlap.

**Phase F — deliverables:** `results/myb_equity_seeds.csv` (30 seeds, daily, adopted spec,
after-tax, cash 5%) plus `results/myb_adopted_spec.json` for study r/154; YoY house-format
table; curve vs NIFTY 50 / Midcap 150 / Smallcap 250 with drawdown panel; RESULTS.md;
published study entry; INDEX.md; TODO.md; dated ops review.

**Replication gate (bolt-on, if the dials arrive):** encode their settings verbatim, reproduce
their CAGR / drawdown / trade count, report the match honestly, and only then optimize.

---

## 4. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-05 14:15 | Data reality measured | 2,671 symbols 2000→2026-09-04; universe starts 2005; N=10 needs 2015+ |
| 2026-09-05 14:20 | Integrity scan | phantom rows CLEAN (2 days, 2014); 176 split-scale events → blackout mitigation designed |
| 2026-09-05 14:30 | STATUS doc written (sections 1-4), gates pre-registered | nothing run yet |
| 2026-09-05 14:22 | Frame cache built | 2,321 symbols x 6,636 days, close/high/low/open/volume |
| 2026-09-05 14:27 | **Engine equivalence SELFTEST PASSED** | `simulate_ext` bit-identical to `bluesky_replay.simulate` with extras off (final ₹2,259,279 both, 304 trades both) |
| 2026-09-05 14:29 | **BUG FOUND AND FIXED before any result was used** | `rolling(W, min_periods=W)` on the union-index wide frame demanded zero missing rows in W trading days; N=10 collapsed to 1 symbol / 32 signals. Fixed to `min_periods=1` + a separate per-symbol `nrows >= N*252` history mask. Phase A re-run from scratch. |
| 2026-09-05 14:32 | Phase A done (72 matrices x 2 windows) | 131 split-scale events; blackout costs only 0.1-2.4% of signals |
| 2026-09-05 14:40 | Phase B done (72 cells x 2 windows x 10 seeds = 1,440 sims) | 51/72 cells clear the G1 gate; see Findings |
| 2026-09-05 14:36 | Phase C launched, waiting on `/tmp/qf_sweep.lock` (r/151 holds it) | 7 families x 24 arms x 2 windows x 10 seeds = 3,360 sims |
| 2026-09-05 14:56 | Phase C done (322 cells) | 17 cells clear the standalone bar; trail-15 transforms every family |
| 2026-09-05 15:02 | Plateau addendum done (trail 10/20/25 added, 364 cells total) | monotone gradient, not a lone peak — and the SAME gradient on the OA control |
| 2026-09-05 15:05 | Phase D done — 30 seeds, cost ladder, outlier tests, OA holding overlap | 23.45% [21.74..25.37] / -25.3% / Calmar 0.93; holding overlap with OA only 3.8-4.4% |
| 2026-09-05 15:07 | **BLEND BUG FOUND AND FIXED** | the first blend pass compared a 2006-start TN+OA baseline against 2010-start candidate blends (2008 in the baseline only). `myb_blend2.py` recomputes EVERY row on the common window and reports PAIRED deltas. Baseline corrected 27.16 -> 26.16% CAGR, DD -16.10% |
| 2026-09-05 15:09 | MYB-vs-GOLD head-to-head + exploratory 4-sleeve probe | gold wins the complement contest at every weight; 4-sleeve probe flagged NOT pre-registered |
| 2026-09-05 15:12 | RESULTS.md written, verdict recorded | study CLOSED |
| 2026-09-05 15:30 | Addendum: literal 50-50 pair checks | **MYB+OA 50-50 = 28.71% / -14.5% / Calmar 1.98 vs the deployed TN+OA 26.16% / -16.1% / 1.56 (month-end NAV, 2010-2026).** NOT a recommendation - the window excludes 2008, which is TN gate's entire case and is untestable for this screen. Added to the 2026-11-30 ops review |

---

## 5. Crash recovery — how the human resumes without Claude

All work is on the **VPS**: `/home/arun/quantifyd/research/152_multiyear_breakout/`.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd

# 1. What finished?
tail -40 /tmp/myb_phaseB.log            # or myb_phaseA/C/D/E.log
wc -l research/152_multiyear_breakout/results/*.csv

# 2. Is anything still alive?
ps -ef | grep -E 'myb_' | grep -v grep
ls -l /tmp/qf_sweep.lock                 # shared lock: only one heavy run at a time

# 3. Resume (every runner is resume-safe: it skips cells already present in its CSV)
flock -w 7200 /tmp/qf_sweep.lock \
  setsid nohup venv/bin/python -u research/152_multiyear_breakout/scripts/myb_sweep.py \
  --phase B > /tmp/myb_phaseB.log 2>&1 < /dev/null &
```

- **Safe to inspect:** everything under `research/152_multiyear_breakout/`.
- **Do NOT touch:** anything under `services/`, `backtest_data/*_state.json`, the crontab, or
  `research/142/scripts/bluesky_replay.py` (imported read-only by this study).
- The frame cache `results/frames_cache.npz` can be deleted safely; it only costs a few minutes
  to rebuild.

---

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `MULTIYEAR_BREAKOUT_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/myb_replay.py` | signal builder + extended simulator (risk sizing, TP, FY tax) | yes |
| `scripts/myb_sweep.py` | phase runner (A/B/C/D), resume-safe | yes |
| `scripts/myb_blend.py` | correlation + 3-sleeve blend + cash-null | yes |
| `scripts/myb_report.py` | YoY house table, curves, tearsheet | yes |
| `results/phaseA_signals.csv` | signal inventory + OA overlap per matrix | yes |
| `results/phaseB_g1.csv` | 72-cell G1 sweep | yes |
| `results/phaseC_g2.csv` | mechanics sweep | yes |
| `results/myb_equity_seeds.csv` | 30-seed daily equity (adopted spec) — for r/154 | yes |
| `results/myb_adopted_spec.json` | adopted spec — for r/154 | yes |
| `results/frames_cache.npz` | wide OHLCV cache | NO — gitignored |
| `results/RESULTS.md` | verdict | yes |

---

## 7. Findings

### Phase A — signal inventory and the Open-Alpha overlap (the study's central question)

Measured over W2 (2010-01-01 → 2026-09-04), signal-date (symbol, day) overlap against the
**Open Alpha** signal set (close above the prior all-time-high close, same RS / liquidity /
setup conditions):

| Family (close-basis level, no age filter) | signals | signals/yr | % of MYB signals that are ALSO OA signals | % of ALL OA signals captured |
|---|---|---|---|---|
| N=2 inclusive | 19,963 | 1,198 | **75.6%** | 91.6% |
| N=3 inclusive | 17,716 | 1,063 | **80.2%** | 86.3% |
| N=5 inclusive | 14,929 | 896 | **86.7%** | 78.6% |
| N=10 inclusive | 9,943 | 852 | **93.3%** | 62.1% |
| N=5 exclusive | 1,993 | 120 | 0.0% (by construction) | 0.0% |
| N=5 ATH-only | 12,846 | 771 | 100% (by construction) | 78.0% |

**The inclusive Multi-Year Breakout is Open Alpha with a slightly wider net** — 76-93% of its
signals are literally OA signals, and it captures 62-92% of everything OA fires on. The
pre-registered kill condition's first leg (>= 60% overlap) is met decisively. Everything now
rests on the **exclusive** variant.

(The `high`-basis arms show only 31-41% overlap, but that is a *definition artifact*: OA's
ATH is measured on closes, so a high-basis N-year level is a different number for the same
economic trade. Holding-level overlap is measured in Phase E.)

### Phase B — G1 sweep, 72 cells x 2 windows x 10 seeds

Default book to isolate the signal: 16 slots @ 6.25%, -8% close stop, SMA-50 close trail,
no market gate, buy-stop-at-pivot fill, 25 bps/side, **after tax**, idle cash 5%.
Benchmark NIFTYBEES buy-and-hold: **W2 10.42% CAGR, W1 14.74%**.

**51 of 72 cells clear the pre-registered G1 gate** (after-tax CAGR above NIFTYBEES in BOTH
windows, positive mean per trade, >= 40 trades). So this is not a NO-EDGE family — but the
gate is a low bar and the structure of the results is unkind to the idea:

**1. The dose-response on "how old must the ceiling be" runs the WRONG way (medians across cells, W2):**

| Pivot age required | CAGR | MaxDD | Calmar | trades/yr (typical) |
|---|---|---|---|---|
| 0 months (any N-year high) | **22.8%** | −34.8% | 0.67 | 70-80 |
| 6 months | 12.9% | −19.1% | 0.68 | 20-40 |
| 12 months | 11.1% | −18.1% | 0.61 | 6-32 |

Requiring the resistance to have actually *held for years* — the thing that makes this screen
"multi-year" rather than "a high" — roughly **halves** the return. Drawdown falls in step, so
Calmar is flat: the age filter is **de-levering, not alpha**. It is a cash-null in disguise.

**2. The distinctive (exclusive) variant is the weakest of the three.** Paired within the same
N and level, at age 0:

| Family | inclusive | ATH-only | **exclusive** |
|---|---|---|---|
| N=2 close | 22.0% / −35.1% | 23.3% / −32.9% | 20.9% / −39.3% |
| N=3 close | 24.5% / −33.0% | 23.7% / −31.4% | 22.9% / −38.1% |
| N=5 close | 23.6% / −32.2% | 23.8% / −30.8% | 22.8% / −32.6% |
| N=10 close | 25.6% / −34.7% | 23.3% / −34.9% | 20.5% / −31.8% |

Clearing a years-old ceiling **while still below the all-time high** pays 1-5pp less and, in
the shorter lookbacks, draws down 6pp deeper. Best exclusive cells on the default book:
`N5_close_excl_age0` 22.79% / −32.6% / Calmar **0.70**, `N10_high_excl_age0` 22.72% / −28.6% /
Calmar **0.79** (but on the 2015+ window and only 24 trades/yr).
**Both miss the pre-registered standalone bar (Calmar >= 0.80 on 30 seeds).** Phase C gives the
exclusive families a fair mechanics pass before any verdict is written.

**3. Window confound to flag:** N=10 cells run on W2b (2015 →) because only 488 names had ten
years of history by 2015. Their apparently strong W2 numbers are partly a friendlier window —
on the common W1 (2020-25) window N=3 / N=5 lead N=10 (43.1% / 42.9% vs 39.7%).

**4. Level basis:** highest-**close** beats highest-**high** on return (median 17.8% vs 11.6%
W2 CAGR) because a high-basis level is strictly higher and rarer; high-basis wins on drawdown.
No clean Calmar separation (0.64 vs 0.68).

### Phase C — G2 mechanics (364 cells, OFAT + plateau)

Trail length is the dominant mechanic, and its gradient is a property of the **whole breakout
family**, not of multi-year highs (W2 Calmar, after tax):

| trail SMA | 10 | 15 | 20 | 25 | 30 | 50 | 150 |
|---|---|---|---|---|---|---|---|
| `N3_close_excl_age0` (the distinctive family) | 1.07 | **0.93** | 0.77 | 0.79 | 0.69 | 0.60 | 0.22 |
| `N2_close_athonly_age0` (the Open-Alpha control) | 1.31 | 0.92 | 0.86 | 0.82 | 0.78 | 0.71 | 0.37 |

17 of 364 cells clear the pre-registered standalone bar. Trail-15 adopted over the
better-scoring trail-10 (edge of the tested range; 140 trades/yr).
Sizing: concentration (3x30% / risk-2%) buys CAGR and costs far more drawdown — Calmar falls
in every family. Gate ON, fill-at-close, tightness filters and take-profits all lose.

### Phase D/E — adoption and portfolio fit

30 seeds: **23.45% CAGR [21.74 .. 25.37] / -25.3% DD (worst seed -28.2%) / Calmar 0.93**;
1,929 trades (115.9/yr), win 45.1%, +3.78%/trade net after tax, max losing streak 16.
Cost ladder 25/40/60 bps -> 23.45 / 20.39 / 16.90% CAGR. Survives dropping its top-10 trades
(mean/trade 3.78% -> 3.16%).

**Open-Alpha overlap at the position level: 3.8-4.4% of hold-days** (5 seeds) — genuinely
different holdings. **Return correlation to OA: 0.426 daily / 0.535 monthly** — same factor.

Blend vs the TN+OA 50-50 baseline (all rows on the same 2010-2026 window, paired on 30 paths):
+0.174 Calmar at 10% weight (30/30 paths) at -0.13pp CAGR, beating the cash-null by +0.086
Calmar and +2.00pp CAGR. **Four of five pre-registered complement conditions pass; the
correlation condition (< 0.4 to both legs) fails.**

Head-to-head on the common 2015+ window: **gold 10% = +0.282 paired Calmar at corr ~0;
MYB 10% = +0.240 at corr 0.37-0.56.** Gold wins the complement contest; MYB wins CAGR
(+1.7 to +3.5pp at every weight). Exploratory (NOT pre-registered): 80% TN+OA / 10% gold /
10% MYB = 28.81% / -11.54% / Calmar 2.43 (+0.628 paired, 30/30) — a hypothesis for a
pre-registered follow-on study, not a finding.
