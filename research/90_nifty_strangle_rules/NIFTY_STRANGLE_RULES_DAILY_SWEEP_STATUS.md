# NIFTY Rules-Based Short Strangle — G1 Daily-Granularity Exit-Rule Sweep (2019→2026)

**STATUS: DONE** · Ran 2026-07-24 14:37–14:38 IST on VPS (12s after cache build) · Verdict: **SIGNAL — G1 PASS → G2** (see `results/RESULTS.md`)

## 1. The Ask

**What Arun asked:** "Automate strangles rules-based so that emotions are left out…
CPR link study may help form criteria for entry/exit/adjustment/react. Don't be
biased; if a new backtest is called for, we must do it. Aim: not 100% profitable —
consistent, minimize losses, control emotions." (2026-07-24, after the W30 mentor
review measured his manual management drag at ≈₹6k/week vs doing nothing.)

**What we're actually testing:** On REAL NSE NIFTY option EOD data (2019→2026),
does a fully mechanical short-strangle cycle — fixed entry day, fixed %OTM strikes,
pre-committed exits (profit-take / per-leg premium stop / giveback / time), zero
adjustments — produce a net-of-cost expectancy and max-loss tail that (a) beats
hold-to-expiry, and (b) is at least as good as Arun's manual management style,
with per-year stability post-2022? Secondary: do VIX and weekly-CPR-width entry
gates (research/67) improve it — tested BOTH signs via per-tercile reporting, not
cherry-picked.

**Success metric:** net expectancy (pts/cycle) with t-stat + per-year positivity +
max-loss distribution; ranked within-family by monotonicity, not peak-picking.

**Falsification plan (decided before launch):** ABANDON the automation-as-strategy
idea if NO exit-rule family shows net-positive post-2022 expectancy with an
acceptable tail (worst cycle bounded vs credit), or if results are driven by
pre-2021 only (r/89 regime). Fallback deliverable stays: mechanical guardrail
overlay for the manual book.

## 2. Economic hypothesis

Short index premium harvests the variance risk premium; counterparty = hedgers +
retail long-option lottery buyers. House prior (r/89): this edge DECAYED ≈0
post-2022 retail boom — so the aim is NOT alpha; it is capturing whatever VRP
remains with a bounded-loss, zero-intervention process that removes the measured
manual-management drag (W30: −₹6k/wk + tail risk + 97% margin). Decay risk:
further options-market efficiency; controlled by per-year tables + post-2022 focus.

## 3. The Base — locked mechanics

- **Data:** `nse_options_bhav` (VPS market_data.db), symbol=NIFTY, 2018-12→2026-07-21.
  Entry/exit priced at daily CLOSE (settle fallback). Liquidity filter (BINDING,
  r/89): entry legs require contracts ≥ 50 that day; skipped cycles counted.
- **Spot proxy:** put-call parity at nearest expiry (strike minimizing |CE−PE|;
  spot = K + CE − PE) — self-contained, no index-series dependency.
- **Cycles:** MONTHLY arm: enter first trading day after prior monthly expiry,
  target = that month's monthly expiry (15–40 cal DTE). WEEKLY arm: enter first
  trading day after each expiry, target = next expiry (3–10 cal DTE).
- **Strikes:** PE at spot×(1−p), CE at spot×(1+p), snapped to nearest liquid
  strike. p ∈ {2.0, 2.5, 3.0}% monthly; {0.8, 1.2, 1.6}% weekly. Both legs must
  quote ≥ 1.5 pts else cycle skipped (counted).
- **Exits (first-hit on daily close; NO adjustments — A0 arm only at G1):**
  - Profit-take: profit ≥ X% of credit, X ∈ {40, 50, 60, none}
  - Per-leg premium stop: leg ≥ Y× its own credit, Y ∈ {1.5, 2.0, 2.5, none}
  - Giveback: once peak profit ≥ 25% credit, exit if profit ≤ 50% of peak ({on, off})
  - Time exit: cal DTE ≤ 2 (monthly) / ≤ 1 (weekly); else expiry settle
- **Costs:** net = gross − [0.5% × (entry+exit premium) + 0.15 pts flat]
  (brokerage+STT+txn+slip approximation); ×2 cost sensitivity in aggregation.
- **Recorded per cycle:** entry VIX, weekly-CPR-width tercile (prior week, from
  Kite NIFTY daily OHLC), max adverse premium ratio, peak profit, exit reason.
  Gates evaluated post-hoc in aggregation (unbiased — no grid inflation).
- **Sizing frame:** results in points; ₹ at 10 lots (650 qty; 1 pt = ₹650).
- **Known G1 limitations (stated):** close-based triggers understate intraday stop
  slippage (G2 adds high-based pessimistic fills); no PANIC intraday flatten
  (EOD data); survivorship n/a (index); look-ahead controlled (all features from
  ≤ entry/trigger day close, trade next value = same close — entry at close of
  signal day is the actual rule, not a forecast).

## 4. Plan — grid

| Axis | Values | Count |
|---|---|---|
| Arm | monthly, weekly | 2 |
| %OTM p | 3 per arm | 3 |
| Profit-take | 40/50/60/none | 4 |
| Premium stop | 1.5/2.0/2.5×/none | 4 |
| Giveback | on/off | 2 |
| **Exit configs** | | **96/arm → 192 total** |
| Cycles | ~91 monthly + ~390 weekly | ~481 |

≈35k cycle-config rows. EOD arithmetic → expected runtime minutes, not hours.
Gates (VIX bands, CPR terciles) applied in aggregation on recorded entry features.

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-24 14:3x | STATUS-MD written; runner authored | pre-launch per convention |
| 2026-07-24 14:36 | Files deployed to VPS (scp) | first nohup attempt died with ssh disconnect |
| 2026-07-24 14:37 | Sweep ran to completion (first launch had in fact succeeded; relaunch reproduced) | 44,928 rows, 27 skipped cells, 12s |
| 2026-07-24 14:4x | Risk table + aggregation reviewed; RESULTS.md written | G1 PASS |

## 6. Crash Recovery (human-runnable, no Claude needed)

- Runner: `/home/arun/quantifyd/research/90_nifty_strangle_rules/scripts/run_g1_daily_sweep.py`
- Launch: `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && nohup venv/bin/python research/90_nifty_strangle_rules/scripts/run_g1_daily_sweep.py > research/90_nifty_strangle_rules/results/run_g1.log 2>&1 &'`
- Progress: `tail -30 /home/arun/quantifyd/research/90_nifty_strangle_rules/results/run_g1.log`
- Output rows: `wc -l /home/arun/quantifyd/research/90_nifty_strangle_rules/results/g1_cycles.csv`
- Alive check: `pgrep -af run_g1_daily_sweep`
- Re-run is idempotent: cached Kite daily fetches in `results/cache_*.csv`; cycles CSV
  rewritten whole (fast). Safe to just relaunch.
- Do NOT touch: `backtest_data/market_data.db` (read-only here).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/run_g1_daily_sweep.py` | G1 runner (sim + aggregate) | yes |
| `NIFTY_STRANGLE_RULES_DAILY_SWEEP_STATUS.md` | this file | yes |
| `DESIGN.md` | G0 design + priors | yes |
| `results/g1_cycles.csv` | per cycle-config rows (~35k) | yes (few MB) |
| `results/g1_ranking.csv` | per-config aggregates + per-year | yes |
| `results/cache_nifty_daily.csv`, `results/cache_vix_daily.csv` | Kite daily OHLC cache | yes |
| `results/run_g1.log` | run log | no |
| `results/RESULTS.md` | verdict (post-run) | yes |

## 8. Findings

Full verdict in `results/RESULTS.md`. One-paragraph summary: **G1 PASS as SIGNAL.**
Monthly strangle + per-leg premium stop (2.0–2.5×) = net t≈2.0–2.4, tail cut 6×
(worst −1,878 → −298 pts), giveback rule harmful, PT 50–60% neutral-but-margin-
friendly. Weekly arm: real mean (t 2.5) but gap-dominated tail that close-based
stops cannot fix → wings/intraday-stop territory (G2). Gates: VIX≥16 helps monthly,
hurts weekly; narrow weekly CPR was GOOD (opposite of r/67 hypothesis — regime
confound suspected, do not gate yet). Caveats: close-fill optimism, r/89 conflict,
multiple testing — all assigned to G2 (pessimistic fills, iron-condor arm, per-year
tables, r/89 reconciliation, chain-recorder intraday validation incl. W30 replay).
