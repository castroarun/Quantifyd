# Volume-Surge + Prev-Day-Range Break, Gated by Narrow Weekly CPR — System Spec

STATUS: **SPEC LOCKED — RUNNER BUILT & SMOKE-VALIDATED — AWAITING VPS LAUNCH APPROVAL**
(spec locked from 10 examples 2026-05-15; smoke run 163/163 signals, 0 spec
violations; full sweep NOT yet launched — needs user go-ahead for VPS run.
More examples still welcome — append to log; signal_lib re-validates them.)

---

## The Ask

**What you asked (paraphrased from chat, 2026-05-15):**
> "Im looking at surge of volumes + breakout from previous day ranges with
> current week's CPR narrow, in aiding to this will be the daily chart being
> in an uptrend if not for a breakout... im sharing more examples so that we
> formulate a system."

**What we're actually formulating:**
A discretionary pattern → systematic strategy. An intraday long entry fires
when, on the same day, ALL of the following align:

1. **Volume surge** — current candle/day volume materially above a recent baseline.
2. **Previous-day-range breakout** — price trades/closes above the prior
   day's high (possibly prior *N* days' combined range).
3. **Narrow weekly CPR** — the *current week's* Central Pivot Range is tight
   (a compressed/coiled week), signalling an impending expansion.
4. **Daily-trend context** — the daily chart is in an uptrend; OR the daily
   bar is itself breaking out of a multi-day base (range expansion).

The thesis: a coiled week (narrow weekly CPR) + a volume-backed break of the
prior day's range, with the higher timeframe already trending up, precedes an
expansion move worth capturing.

### Design principle (LOCKED, from ex#5 VOLTAS)

**Strict AND-gate. All four conditions mandatory.** A prev-day-range break
WITHOUT both (narrow weekly CPR) AND (volume surge) is a **NO-TRADE**, even
if it would have been a winner. The system optimises **precision/selectivity,
not recall** — deliberately missing winning breakouts is acceptable and
expected. This makes it a low-frequency, high-conviction system. The backtest
metric must therefore reward selectivity (per-trade edge + WR), NOT trade count.

---

## The Base — LOCKED SIGNAL DEFINITION (distilled from all 10 examples)

Per stock, per day **D**, per timeframe **TF ∈ {5,10,15,30,60}min** (10/15/30/60
resampled from stored 5-min). A trade triggers when ALL gates pass:

1. **Direction selector (daily, as of D-1):** compute daily trend.
   uptrend ⇒ LONG-only that day; downtrend ⇒ SHORT-only. *(trend def = axis)*
2. **Week context — narrow weekly CPR:** for the week containing D, weekly
   CPR from prior week H/L/C; `width% = (TC−BC)/Pivot×100`. Week eligible
   only if `width% ≤ θ_cpr`. *(θ_cpr = axis; ex#7: week-level, not day 1)*
3. **Trigger candle:** the **opening TF candle of ANY day D** inside that
   narrow-CPR week (ex#7 — not restricted to the week's first day).
4. **Range break — must ESCAPE established range (ex#10):**
   - LONG: trigger close > max(PrevDayHigh, **PrevWeekHigh**)
   - SHORT: trigger close < min(PrevDayLow, **PrevWeekLow**)
   - If price stays inside the prior **week's** range → NOT a breakout.
5. **Clean directional candle (ex#8, mandatory):** body=|close−open|,
   rng=high−low. LONG: green, close in top zone of rng, `body/rng ≥ b_min`.
   SHORT: red, close in bottom zone. Reject doji/pin/opposite-colour even
   if the close breaches the range. *(b_min, close-zone = axis)*
6. **Volume surge (ex#5/6, mandatory):** trigger-candle volume ≥ `k ×`
   baseline (baseline = trailing avg same-slot vol). *(k = axis)*
7. **[Optional] Clear room ahead (ex#6/9 hypothesis):** no opposing prior
   S/R within `R_atr × ATR` in trade direction. *(on/off axis)*

**Exit (ex#9 ⇒ mandatory; user: same-day OR carry):** defined stop
(structural = other side of trigger candle, or ATR) + exit policy from the
VOLBO 13-policy grid × {same-day-close, carry up to M days}.

**Ranking metric:** per-trade R-multiple; cells ranked by Sharpe-style
(meanR/stdR) with research/34 robustness gates (n≥15, WR, payoff,
consistency across ≥3 variants). **Never judged on individual trades (ex#9).**

## The Base — open-param → AXIS mapping (was "not yet locked")

Each axis below is an **open parameter** the examples must help pin down.
Do NOT backtest until these are resolved.

| Component | Open question | Candidate values |
|---|---|---|
| Volume surge | vs what baseline? what multiple? | curr vol ≥ k × SMA(vol, N); k∈{1.5,2,3}, N∈{10,20} day; or intraday-candle vs same-time avg |
| Trigger timeframe | which intraday TF detects the break? | first candle vs any-candle; 5/15/30/60-min (BIOCON & ZYDUS shown on **30-min**) |
| Prev-day range | prior 1 day high, or prior N days? | PDH; or N-day range top, N∈{1,2,3,5} |
| Break confirmation | touch vs close above? | close of trigger candle > PDH; or any trade > PDH |
| Narrow weekly CPR | "narrow" defined how? | weekly CPR width / price ≤ threshold; or CPR width percentile vs trailing weeks; or narrower than prior week |
| Daily uptrend | how defined? | close > SMA(50/200) daily; or higher-highs; OR daily breakout substitutes |
| Direction | ~~long-only?~~ **RESOLVED (ex#4 KALYANKJIL)** | **SYMMETRIC + trend-aligned: LONG when daily uptrend & breaking PDH up; SHORT when daily downtrend & breaking PDL down. Daily-trend filter sets the side.** |
| Exit | not yet specified | reuse VOLBO's 13 exit policies (T_NO, ATR_SL, Chandelier, R-targets, step-trail) |
| Universe | which stocks? | likely Nifty 500 / F&O; ZYDUSLIFE & BIOCON are NOT in VOLBO's 79-stock intraday set |
| Period | data-dependent | intraday history is the binding constraint (see Data note) |

**Data note (binding):** Intraday (5/15/30-min) history in `market_data.db`
exists for only ~79 stocks (Cohort A 2018+, Cohort B 2024-03-18+). ZYDUSLIFE
and BIOCON are in neither cohort — no intraday data yet. Any backtest of this
system on these names requires a VPS-side intraday backfill first (VPS is
canonical for all Kite downloads).

---

## Data Readiness (audited 2026-05-15 vs local snapshot)

`scripts/data_audit.py` output. **F&O universe = 81 stocks** (FNO_LOT_SIZES).

| Timeframe | F&O coverage | Range (local snapshot) | Verdict |
|---|---|---|---|
| 5-min | **79/81** | 2018-01-01 → **2026-03-13** (10 since 2018; 69 since ~2024-03-18) | ✅ workhorse; 10/15/30/60m **derive by resampling** (research/30b precedent) |
| 30-min (stored) | 48/81 | Sep–Nov 2025 only | ❌ unusable directly — resample from 5-min |
| 60-min (stored) | 54/81 | 2018 → 2025-11-19 | partial — prefer resampled-from-5m for consistency |
| day | **80/81** | 2000 → 2025-12-31 | ✅ weekly-CPR + daily-trend filters OK for full set |

- Backtestable universe = the **79 stocks with 5-min**. Missing: TATAMOTORS, ZOMATO (ticker changes).
- **CAVEAT 1 — RESOLVED (VPS verified 2026-05-15):** VPS canonical `market_data.db` has 5-min for **79/81 F&O stocks, ZERO stale** — 15 fresh through 2026-05-15, 64 lagging only by days→6wks (all ≥ Apr 2026), 2 ABSENT (TATAMOTORS, ZOMATO — renamed tickers). VPS 5-min spans 2018→2026-05-15 (380 symbols total); daily 1623 symbols →2026-05-07. The lag on 64 names is irrelevant for a multi-year sweep (only the last few weeks missing). **CONCLUSION: data is sufficient for the underlying backtest. Sweep runs on VPS (binding rule).**
- **CAVEAT 2:** `backtest_data/options_data.db` EXISTS (1.46 GB; option_chain 5.6M rows) — contradicts "no options data" assumption — BUT `option_ohlc` only 3,268 rows → a per-trade historical-options backtest is NOT feasible. Decision stands: **backtest the underlying signal** (R-multiple favourability), options/debit-spread is a live-execution overlay applied later (same approach as research/29/30/34).

## Spec-Readiness Verdict (2026-05-15)

**Status: clear enough to design the sweep — undefined params become grid axes.**

- LOCKED: symmetric trend-aligned direction · opening-30m trigger any day in a narrow-CPR week · clean directional candle · volume surge mandatory · strict AND-gate · probabilistic edge ⇒ mandatory stop + R-target, judged on expectancy.
- OPEN → swept as axes (NOT blockers): ① weekly-CPR-width threshold ② volume-surge multiple+baseline ③ "clear room ahead / no opposing S/R" residual filter (on/off) ④ quantitative clean-candle proxy (body%/close-position) ⑤ exit: same-day vs carry-forward × VOLBO's 13 exit policies ⑥ timeframe ∈ {5,10,15,30,60}m (all resampled from 5-min).

## Examples Log (the evidence base — grows as user shares)

| # | Symbol | Date | TF shown | Volume surge | PDR break | Weekly CPR | Daily trend | Notes |
|--:|---|---|---|---|---|---|---|---|
| 1 | ZYDUSLIFE | 2026-05-14 | 30-min | Yes — tall vol bar on 14-May | 1st 30m candle closed above prior days' ranges | (not yet measured) | Uptrend + multi-month range breakout (~960→1011, +13.74% leg) | Daily breakout was the stronger signal; RSI 61.8 |
| 2 | BIOCON | 2026-05-06 | 30-min | Yes — volume surge 6-May | Prev-day range broke 6-May | (not yet measured) | Daily uptrend, +26.08% on the leg, ran ~360→430 | User-flagged as a clean instance |
| 3 | LAURUSLABS | 2026-05-04/05 | 30-min | Yes — large vol bar on the 04-05 May breakout candle | Strong green opening candle closed above prev day range; gapped ~1100→1160 | User says narrow CPR that week | **Already in daily uptrend** (~780→1319, +48% leg); RSI 30m jumped to ~69 | Move continued "a couple of days at least". **NOTE: LAURUSLABS IS in VOLBO Cohort B — research/34 already validates it long-side (10min, l_vm2.0_gapoff_rsi40_60, Sharpe 0.77, LONG bias). First example with a pre-existing backtested edge + intraday data available.** |
| 4 | KALYANKJIL | 2026-05-11 | 30-min | Yes — tall vol bar on the 11-May breakdown candle | **SHORT mirror:** 11-May opening candle closed *below* prior days' ranges | User says narrow CPR that week | **Daily trend DOWN** (~-42% leg; ran ~413→353 on shown leg, -21%) | Move continued down. **First SHORT example — resolves the "direction" open param: system is TREND-ALIGNED & SYMMETRIC, not long-only. Still a winner (not a failure/counter-example). Not in VOLBO cohorts — no intraday data.** |
| 5 | VOLTAS | 2026-04-13 | 30-min | **NO — no volume support** | 1st 30m candle closed above prev day ranges (break present) | **WIDE CPR (fails the filter)** | Uptrend-ish | **DELIBERATE SKIP. Would have won (~1315→1500+) but NO-TRADE because CPR wide + no volume.** Defines the AND-gate: PDR break alone is NOT enough. Selectivity over recall — accept missed winners. VOLTAS IS in VOLBO Cohort B (intraday data available). |
| 6 | VOLTAS | 2026-04-27 | 30-min | **NO — user confirms "volume was not great"** | Breakout candle above prior range — **then FAILED**, reversed ~1500s→~1234 | (also extended, into prior swing-high ~1511) | Up, but **extended**: price had already run 1234→1500+ | **FIRST TRUE FAILURE — and the volume-surge gate would have CORRECTLY REJECTED it (weak volume).** This is the key validation: the volume filter excludes a real failure. Combined w/ ex#5 (weak vol → skipped a winner), volume surge is an evidenced discriminator. VOLTAS in Cohort B → intraday data available. |
| 7 | TMPV | 2026-04-08 (week of) | 30-min | Yes — strong vol on the breakout day's opening candle | Week's FIRST candle had NO breakout & NO volume; a LATER day's opening 30m candle had the "superior breakout" w/ vol + narrow CPR | Narrow (week-level) | Up | **TIMING REFINEMENT: the breakout trigger is NOT restricted to the first candle of the WEEK. Narrow-CPR is a WEEK-level context filter; the trigger = the opening 30m candle of ANY DAY within that week. Resolves the "trigger timeframe / first-candle" open param.** TMPV in Cohort B → intraday data available. |
| 8 | TMPV | 2026-03-30 | 30-min | n/a | 1st candle closed below prev-day range (would-be SHORT trigger) | — | — | **CANDLE-QUALITY FILTER (new mandatory gate): NO-TRADE. Closed below range but the candle itself is BULLISH (green/recovery), not a clean bearish bar.** Trigger candle must be a clean directional candle IN the trade direction: short → decisive red bar (close near low, real body); long → decisive green bar. Reject dojis / pins / opposite-colour closes even if the close breaches the range. |
| 9 | TMPV | 2026-01-19 | 30-min | **YES — strong volume (tall vol bar)** | **YES — clean breakdown, 1st candle broke below prev-day range** | **YES — narrow CPR** | **YES — daily DOWNtrend (trend-aligned short)** | **★ FULL CONFLUENCE — AND STILL FAILED.** Broke down ~360→345 then reversed up to ~356 (short did not follow through). **The single most important example: proves the 4+candle-quality confluence is NECESSARY BUT NOT SUFFICIENT. The system is a probabilistic edge, not a deterministic rule.** TMPV in Cohort B → intraday data available to dissect. |
| 10 | TMPV | 2026-05-15 | 30-min | Big volume on 1st candle | **NO — not a qualifying breakout: PRIOR WEEK'S RANGE STILL INTACT** (big candle stayed inside the established range) | — | — | **RANGE-DEFINITION REFINEMENT (no-trade): a large volume 1st-candle is NOT a breakout if price remains within the prior WEEK's range. The break must ESCAPE the established range/consolidation, not merely exceed one prior day while the weekly range holds.** Distinguishes "big candle inside range" from "genuine structural break". |

(More rows added as you share examples. For each I'll record: the exact
trigger candle, the volume multiple vs baseline, the PDH level broken, the
weekly CPR width, and the daily-trend state — so the thresholds emerge from
the data rather than being guessed.)

---

## Plan — variant grid (cell count)

| Axis | Values | n |
|---|---|---:|
| Timeframe | 5, 10, 15, 30, 60 min | 5 |
| Daily-trend def | close>SMA50 ; close>SMA200 ; 20d-HH/LL | 3 |
| θ_cpr (narrow weekly CPR width%) | ≤0.25, ≤0.5, ≤0.75, ≤1.0 | 4 |
| Volume mult k | 1.5, 2.0, 3.0 | 3 |
| Clean-candle strictness | loose, strict | 2 |
| Clear-room filter | off, on | 2 |
| Carry | same-day-close, carry≤M days | 2 |
| Exit policy | VOLBO 13-policy grid (parallel per signal) | 13 |

Direction (long/short) is set by the daily-trend selector, both evaluated.
Signal cells (excl. exit, which runs in parallel per signal):
5×3×4×3×2×2×2 = **1,440 signal cells × 79 stocks**, ×13 exits at scoring.
Runs on VPS (binding); resumable; incremental CSV; resample 5m→TF in-runner.

## Status (event log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-05-15 ~16:42 IST | Spec doc created, research/40 scaffolded | 2 examples captured (ZYDUSLIFE, BIOCON) |
| 2026-05-15 ~16:45 IST | Example #3 added: LAURUSLABS 04-05 May | First example that overlaps a VOLBO-validated name w/ intraday data |
| 2026-05-15 ~16:48 IST | Example #4 added: KALYANKJIL 11-May SHORT | Direction param RESOLVED → symmetric, trend-aligned |
| 2026-05-15 ~16:50 IST | Ex#5 VOLTAS 13-Apr (deliberate skip) | Locked strict AND-gate / precision-over-recall principle |
| 2026-05-15 ~16:51 IST | Ex#6 VOLTAS 27-Apr — FIRST TRUE FAILURE | Weak volume; volume-surge gate correctly rejects it → filter validated |

### ★ Core conclusion after ex#9 (LOCKED principle)

**The confluence (volume surge + clean PDR-break candle + narrow weekly CPR +
daily-trend alignment) is a NECESSARY-BUT-NOT-SUFFICIENT, PROBABILISTIC edge.**
Ex#9 (TMPV 19-Jan) had every box ticked and still failed. Implications that
are now binding on the eventual design:

1. **Never judge the system by individual chart outcomes.** It WILL produce
   full-confluence losers. Evaluate only on expectancy / Sharpe / payoff
   across many signals (consistent with research/34 VOLBO: best cells were
   ~60–90% WR, never 100%).
2. **A defined stop + R-target is mandatory, not optional.** The edge is
   realised through risk management on a population of trades, not by the
   entry filter being "right". Reuse VOLBO's 13 exit-policy grid.
3. The 4 conditions are the **selectivity gate** (raise WR/payoff vs random
   breakouts); they are not a guarantee filter.

### Residual failure-mode HYPOTHESIS (open — needs more examples to confirm)

Recurring across the two failures (ex#6 VOLTAS 27-Apr, ex#9 TMPV 19-Jan):
**the move broke INTO a nearby opposing structural level** — VOLTAS broke up
into prior swing-high resistance (~1511); TMPV 19-Jan broke down straight
into a prior support shelf (~338–345) and bounced. Winners (ex#1–4) broke
into relatively open space. → Candidate 5th condition:
**"breakout must have clear room — no opposing S/R level within ~1×ATR / ~1
prior-range ahead in the trade direction."** NOT locked; flag examples that
confirm or break this.

### Evidence scorecard so far (n=9)

| Condition | Evidence it discriminates | Status |
|---|---|---|
| **Volume surge** | ex#5 weak-vol skip (missed winner, acceptable); ex#6 weak-vol → FAILURE correctly excluded | **STRONG — mandatory** |
| **Clean directional trigger candle** | ex#8 TMPV 30-Mar: range broken but bullish candle → NO-TRADE | **mandatory (new gate, ex#8)** |
| Prev-day-range break | present in all; necessary, NOT sufficient (ex#9 had it & failed) | necessary, not sufficient |
| Daily-trend align | sets side (ex#1-3 long/up, ex#4/9 short/down) | side-selector |
| Trigger = opening 30m of ANY day in the narrow-CPR week | ex#7 TMPV: week's 1st candle nil; later day's open was the trigger | timing rule (ex#7) |
| **Narrow weekly CPR** | claimed in winners, "wide" in ex#5 skip; ex#9 had it & still failed → NOT sufficient alone; still NEVER objectively measured | **UNPROVEN — weakest-defined gate; measure on LAURUSLABS/VOLTAS/TMPV (all Cohort B, have intraday data)** |
| Clear room ahead (no opposing S/R) | HYPOTHESIS from ex#6 & ex#9 failures | **OPEN — needs confirming examples** |
| | Awaiting more examples from user | Spec stays locked-OUT of backtest until params resolved |

---

## What I need from the examples to lock the spec

As you share more, I'll be extracting these so the system is data-derived:

1. **The "narrow weekly CPR" threshold** — the single most undefined piece.
   Need several instances to see what CPR width (abs or %-of-price, or vs
   prior weeks) consistently preceded the move.
2. **Volume-surge multiple** — what k actually separated these from noise.
3. **Trigger timeframe** — is it always the *first* candle, or any
   intraday break? (VOLBO tested first-candle only; these look 30-min.)
4. **Prev-day-range definition** — 1 day vs N days.
5. **Counter-examples** — instances where the pattern was present but it
   FAILED. Without failures the spec will overfit to winners.

## Files

| File | Purpose | Committable? |
|---|---|---|
| `VOLSURGE_PDR_BREAK_WEEKLY_CPR_INTRADAY_SWEEP_STATUS.md` | This spec/status doc | yes |
| `examples/` | Annotated chart screenshots + per-example notes | yes (small) |
| `scripts/signal_lib.py` | Pure signal functions (weekly CPR, daily trend, resample, clean candle, range escape, volume surge, clear room) | yes |
| `scripts/run_volsurge_sweep.py` | Resumable sweep runner (1,440 cells × 79 stocks, 13 exits/signal) | yes |
| `results/volsurge_signals.csv` | Per-signal × 13-exit rows (large — gitignored) | NO |
| `results/volsurge_ranking.csv` | Per-cell aggregate (Sharpe/expectancy/WR) | yes if small |
| `results/volsurge_leaders.csv` | Per-stock best cell + robustness counts | yes |
| `results/RESULTS.md` | Final findings (top configs, asymmetry, TF/exit) | yes |

<!-- RUNNER_STATE -->
**Runner state: SMOKE-VALIDATED on laptop snapshot** — 10 stocks, 30min-only
tiny grid, 163 signals / 195 ranked cells / 8 leaders, ~4.6 min, clean
(updated 2026-05-15). Full 79-stock sweep NOT yet launched (runs on VPS).

---

## Crash Recovery — resume the full sweep WITHOUT Claude

The runner is **resumable** and writes the signals CSV incrementally
(appended after every cell). If it dies mid-run, just relaunch the same
command — it rebuilds a `(symbol, timeframe, variant, direction, date)`
skip-set from `results/volsurge_signals.csv` and only computes cells not
already logged.

### Launch the full sweep on the VPS (canonical host — binding rule)

```bash
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && nohup python3 \
  research/40_volsurge_pdr_break_weekly_cpr/scripts/run_volsurge_sweep.py \
  > /tmp/volsurge.log 2>&1 &'
```

(First `git push` from laptop, then on VPS
`cd /home/arun/quantifyd && git reset --hard origin/main` to pull the new
`research/40` scripts. The DB path is relative — `Path(__file__).parents[3]
/ backtest_data / market_data.db` — so it reads the VPS canonical
`market_data.db` unchanged.)

### Monitor progress

```bash
ssh arun@94.136.185.54 'tail -f /tmp/volsurge.log'
# how many signal rows so far:
ssh arun@94.136.185.54 'wc -l /home/arun/quantifyd/research/40_volsurge_pdr_break_weekly_cpr/results/volsurge_signals.csv'
```

The `## Files` block at the top of THIS doc carries a `<!-- RUNNER_STATE -->`
line the runner refreshes every 5 stocks (state / stocks-done / signals).

### Resume after a crash

```bash
# Identical command — it skips everything already in volsurge_signals.csv:
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && nohup python3 \
  research/40_volsurge_pdr_break_weekly_cpr/scripts/run_volsurge_sweep.py \
  > /tmp/volsurge.log 2>&1 &'
```

### Aggregate-only (signal-gen finished but ranking/RESULTS.md crashed)

```bash
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && python3 \
  research/40_volsurge_pdr_break_weekly_cpr/scripts/run_volsurge_sweep.py \
  --aggregate-only'
```

### Files safe / unsafe to touch mid-run

- **Safe to inspect:** `results/volsurge_signals.csv` (read-only — `wc -l`,
  `tail`), `/tmp/volsurge.log`, this STATUS doc.
- **Do NOT touch / delete:** `results/volsurge_signals.csv` while the runner
  is alive (it appends to it). Deleting it forces a full restart.
- `results/volsurge_ranking.csv`, `volsurge_leaders.csv`, `RESULTS.md` are
  regenerated wholesale by aggregation — safe to delete to force a re-aggregate.

### Validate on the laptop snapshot without a VPS run

```bash
python research/40_volsurge_pdr_break_weekly_cpr/scripts/run_volsurge_sweep.py --smoke
```

Runs the 10 snapshot stocks (RELIANCE…HINDUNILVR) on a 1-cell grid
(30min / sma50 / θ_cpr 1.0 / k 2.0 / loose / clearroom-off / sameday).
Done in ~5 min; produces non-empty signals + ranking + leaders + RESULTS.md.

