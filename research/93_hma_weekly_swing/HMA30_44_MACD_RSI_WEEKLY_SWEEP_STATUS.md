# HMA 30/44 + MACD(21,39,9) + RSI(9/3/21W) Weekly Retracement-Reversal Swing — Full Daily Universe

STATUS: DONE — **SIGNAL (not investable as tested)**: real +3.2%/trade edge over
year-matched random control (t 7.15, all 27 cells, both halves), but the 20-slot book
loses to NIFTYBEES B&H (6.7% vs 12.75% CAGR) and the index-cash variant is worse
(MaxDD −63.8%). Full verdict: `results/RESULTS.md`.

Research folder: `research/93_hma_weekly_swing/` (93 = next free VPS number; VPS `research/`
went to 92 — do NOT confuse with the laptop-memory research/90/91 topics).
VPS-canonical (`94.136.185.54`, DB `backtest_data/market_data.db`). Laptop is dev-only
(NO local DB, NO git on this machine) — scripts are shipped to VPS via paramiko
(`scripts/deploy_run.py`, password read from the `vps_ssh_paramiko.md` memory file) and run there.

---

## 1. The Ask

**What Arun asked:** "Please test this strategy" — Nitin Hulaji's swing-trading method from
the Market Aur Main Ep.5 video (Vijay Thakkar channel). Summary given: weekly charts; two
Hull MAs (30 & 44); MACD with custom 21,39,9 (histogram must print ≥8 bars below zero before
the reversal); RSI(9) with a 3-period MA and a 21-period WMA (signal = the 3-MA crossing the
21-WMA); enter when price sits between the 30 and 44 HMA with all three aligned; SL below the
previous swing low; target = the previous swing high where the downtrend began; plus a daily
management overlay (any holding up +3% in a day → sell 10% of it, redeploy).

**What we are actually testing:** On WEEKLY bars built from the full daily universe
(~1,600 NSE names, 2000→2026), does the triple-aligned retracement-reversal entry
(fresh close above HMA30 while still inside the 30/44-HMA zone + MACD(21,39,9) histogram
turning up after a ≥8-bar below-zero run + RSI(9) 3-SMA crossing above its 21-WMA), entered
at next week's open with SL at the last confirmed swing low and target at the prior swing
high, produce a positive **net** per-trade expectancy that **beats a random-entry /
drift control** (mandatory since research/87/88 — long-only daily/weekly screens are
drift-and-survivorship traps)? Long-only, as taught. The daily +3%→sell-10% overlay is a
portfolio-construction detail — deferred to G4 and only if the entry edge is real.

## 1a. Economic hypothesis (G0)

Under-reaction / capitulation reversal: after a multi-month retracement inside a larger
structure, the last sellers exhaust (long MACD-negative run = persistent downside momentum
that is now fading), and the first genuine demand (RSI short-MA crossing its long WMA,
price reclaiming the fast HMA) marks the turn; counterparty = capitulating holders selling
the base and momentum shorts pressing a move that is over. Decay risk: MA-pullback reversal
is among the most widely taught retail setups; on weekly bars turnover is low so costs are
small, but the same slowness means the "edge" is easily just market drift harvested by a
long-only screen on a survivor universe. **That is exactly what killed research/87/88/91 —
so the verdict metric is setup-vs-control, not raw expectancy.**

**Falsification (decided now):** if pooled gross expectancy ≤ 0, OR net (25 bps) ≤ 0, OR the
setup does NOT beat the date/regime-matched random-entry control by a meaningful margin
(setup − control ≤ 0), OR the edge is one-year / few-name concentrated at G3 → **NO EDGE, stop.**

## 2. The Base — locked mechanics (all causal, weekly bars)

Weekly bars = daily bars resampled to W-FRI (open=first, high=max, low=min, close=last).
Signal evaluated on the completed weekly bar `i`; **entry at next week's open** (bar `i+1`).

**Indicators (on weekly closes):**
- `HMA30`, `HMA44` — Hull MA: `WMA(2·WMA(n/2) − WMA(n), √n)` with rounded sub-periods.
- MACD custom: `EMA21 − EMA39`, signal `EMA9` of the MACD line, `hist` = MACD − signal.
- `RSI9` (Wilder), `rsi_ma3` = SMA(3) of RSI9, `rsi_wma21` = WMA(21) of RSI9.

**Entry conditions at bar `i` (ALL must hold):**
1. **HMA zone**: `close[i] > HMA30[i]`, the close-above-HMA30 cross happened within the last
   `recent_win=4` weeks, and `close[i] ≤ max(HMA30[i], HMA44[i])` (still inside the 30/44
   zone — not already extended; the taught "positioned between the 30 and 44 HMA").
2. **MACD**: `hist[i] > hist[i−1]` (turning up) AND a run of ≥ `neg_bars=8` consecutive
   below-zero histogram bars ended within the last `recent_win` weeks (hist may still be <0).
3. **RSI**: `rsi_ma3[i] > rsi_wma21[i]` AND the upward cross happened within the last
   `recent_win` weeks.

**Stop:** most recent CONFIRMED fractal swing low (low[j] strictly lowest vs 2 bars each
side, confirmed only once bar j+2 has closed, j+2 ≤ i), with a 0.1% buffer; fallback =
min(low) of last 10 weeks if no valid pivot below entry. Stop must be below entry else skip.
**Target:** max(high) over the last `target_lb=26` weeks up to `i` (the swing high the
retracement fell from). Target must be above entry else skip (not a retracement).
**Management:** weekly bars; stop-first if both hit in the same week (conservative); stop
fills at `min(open, stop)` (gap-through modeled), target fills at `max(open, target)`;
time-stop `max_hold=52` weeks → exit at close. No pyramiding; one position per symbol.

**Costs:** delivery round-trip `cost_bps=25` (STT 0.1%×2 + slippage) as the headline net;
sensitivity 0/10/25/50 bps. Per-trade return also in **R** (risk = entry−stop).

**Control (the verdict benchmark):** per symbol, 3× the setup-trade count of random entries
drawn from the same calendar years as that symbol's setup trades, identical stop/target/
time-stop mechanics (pivot stop, 26w-high target). Setup must beat this to mean anything.

**Universe:** all `timeframe='day'` symbols; exclude index/ETF-like names; require ≥200
weekly bars, median close ≥ ₹20, median daily turnover ≥ ₹1 cr. **Survivorship caveat:**
universe = today's names (stated openly; modern sub-period reported at G3 if it survives G1).

## 3. Plan (stage-gated)

| Stage | What | Kill criterion |
|---|---|---|
| **G1 probe** | Base params, full universe, gross+net, setup vs random control | gross ≤0 OR setup ≤ control → NO EDGE |
| **G2 sweep** | neg_bars {5,8,12} × recent_win {2,4,8} × target_lb {13,26,52} × zone/no-zone × exit variants (HMA30-cross trail) | net ≤0 or non-monotonic |
| **G3** | Per-year, OOS split (≤2018 / 2019+), param sensitivity, super-winner guard, cost-sens | one-year/one-name/overfit |
| **G4** | Portfolio construction incl. the +3%-day → sell-10% overlay, DD/Calmar | DD > budget / Calmar poor |

Base params: `hma=(30,44), macd=(21,39,9), rsi=(9,3,21), neg_bars=8, recent_win=4,
target_lb=26, pivot=2/2, max_hold=52w, cost=25bps`.

## 4. Falsification / gates — see 1a. Primary G1 gate: pooled **gross > 0** AND
**setup net(25bps) > 0** AND **setup − random-control > 0** with pooled t ≳ 3.

---

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-26 22:15 IST | Folder + STATUS written (sections 1–4 locked) | before any code ran |
| 2026-07-26 22:18 IST | Engine smoke-tested locally on synthetic weekly bars | trade mechanics OK (stop-out = R −1.03 incl. cost) |
| 2026-07-26 22:20 IST | Scripts shipped to VPS; **G1 probe launched** (PID 1160524) | log `/tmp/hma93_g1.log`, incremental `results/g1_probe.csv` |
| 2026-07-26 22:31 IST | **G1 DONE — GATE PASSED** (1 min run; 629 names tested, 560 short-history / 322 illiquid / 117 penny skipped) | setup n=4,537: gross +4.87%, net(25bps) +4.62%, R 0.353, win 37.8%, PF 1.66, t 11.25, hold 12.6 wks. Control n=13,376: gross +1.91%, net +1.66%. Setup beats drift by ~+3.0%/trade |
| 2026-07-26 22:38 IST | G3 robustness + G2 27-cell sweep launched sequentially on VPS | logs `/tmp/hma93_g3.log`, `/tmp/hma93_g2.log` |
| 2026-07-26 22:45 IST | **G3 DONE — edge survives** (diff +3.17%/tr, Welch t 7.15; both sub-periods +; super-winner OK) | 2020 monster caveat: ex-2020 diff ≈ +1.7%/tr |
| 2026-07-26 22:49 IST | **G2 DONE — all 27 cells beat control** (+2.6 to +4.3%/tr, flat grid) | `results/g2_sweep_cells.csv` |
| 2026-07-26 22:55 IST | G4 v1 INVALID — same-week-exit trades became zombie positions clogging slots (only 169 taken) | fixed: same-week exits realize at entry |
| 2026-07-26 23:02 IST | **G4 v2 DONE — book UNDERPERFORMS index**: 6.70% CAGR / Sharpe 0.47 / MaxDD −48.9% vs NIFTYBEES 12.75% / 0.73 / −58.0% | 1,610 taken, 2,927 slot-skipped, avg 12.9/20 positions |
| 2026-07-26 23:05 IST | G4b launched: idle cash tracks NIFTYBEES (index + signal-tilt test) | log `/tmp/hma93_g4b.log` |
| 2026-07-26 23:10 IST | **G4b DONE — worse**: 8.93% CAGR, MaxDD −63.8% (index crashes hit cash sleeve; stock sleeve still subtracts) | study CONCLUDED |
| 2026-07-26 23:15 IST | RESULTS.md written; STATUS → DONE; verdict **SIGNAL (not investable as tested)** | close-out: INDEX/TODO/memory updated, docs mirrored to VPS |

## 6. Crash Recovery (resume without Claude)

- **Engine + runner:** `research/93_hma_weekly_swing/scripts/{hma_weekly_engine.py,run_g1_probe.py}`
  (identical copies shipped to VPS `~/quantifyd/research/93_hma_weekly_swing/scripts/`).
- **Deploy+launch from laptop:** `python research/93_hma_weekly_swing/scripts/deploy_run.py`
  (uploads scripts, then `nohup venv/bin/python3 .../run_g1_probe.py > /tmp/hma93_g1.log 2>&1 &` on VPS).
- **Monitor:** `ssh arun@94.136.185.54 'tail -20 /tmp/hma93_g1.log'`; incremental output
  `research/93_hma_weekly_swing/results/g1_probe.csv` on VPS.
- **Resume:** the runner skips symbols already present in `g1_probe.csv` (done-set); safe to re-run.
- **Do NOT touch:** `backtest_data/market_data.db` (canonical, read-only here).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/hma_weekly_engine.py` | Weekly HMA/MACD/RSI engine + control sim (causal) | yes |
| `scripts/run_g1_probe.py` | G1 full-universe probe runner | yes |
| `scripts/deploy_run.py` | Paramiko ship-and-run helper | yes |
| this STATUS file | Live status + recovery | yes |
| `results/g1_probe.csv` | Per-symbol setup + control summary | yes (small) |
| `results/g1_trades.csv` | Pooled setup trade list | yes if <5 MB |
| `results/RESULTS.md` | Final verdict | yes |

## 8. Findings

### G1 (base cell, full universe) — GATE PASSED

Setup n=4,537: gross +4.87%/trade, net(25bps) +4.62%, R 0.353, win 37.8%, PF 1.66,
t 11.25, avg hold 12.6 wks. Year-matched random control n=13,352: gross +1.91%, net +1.45%.
**Setup beats drift by +3.17%/trade, Welch t = 7.15.**

### G3 robustness — SURVIVES

- **Sub-periods**: ≤2018 diff +2.22% (t 3.50); ≥2019 diff +4.11% (t 6.65) — both positive.
- **Super-winner guard**: ex-top3 (ASHOKLEY, BEML, CANBK) mean 4.41% vs 4.62% — no dent.
- **Cost sensitivity**: 0/10/25/50 bps → 4.87/4.77/4.62/4.37% — costs irrelevant at 12.6-wk holds.
- **Exit mix**: 2,749 SL (61%), 1,501 target (33%), 202 time, 85 open-at-end.
- **Warts**: 2020 is a monster (+49.7%/trade, diff +30.7; ex-2020 edge ≈ +1.7%/trade).
  7 of 26 years have negative diff. Median trade −6.1% (positive-skew: most trades lose,
  winners are big) — psychologically hard. Survivorship: deep-retracement entries interact
  with a survivor universe more than random entries; control shares the bias but imperfectly.

### G2 sweep (27 cells: neg_bars 5/8/12 × recent_win 2/4/8 × target_lb 13/26/52) — ROBUST

**All 27 cells beat their matched controls, diff +2.6% to +4.3%/trade — flat across the
grid, no lone peak.** Diff rises mildly with tighter recent_win (2) and longer target_lb (52).
neg_bars mainly gates trade count, not edge quality — the MACD run-length is NOT the driver;
the core alignment (HMA-zone reclaim + momentum turn + RSI cross) is. Full table:
`results/g2_sweep_cells.csv`.

### G4 portfolio — FAILS investability

20-slot 5%-NAV book 2005→2026: **6.70% CAGR / Sharpe 0.47 / MaxDD −48.9%** vs NIFTYBEES
B&H 12.75% / 0.73 / −58.0%. Idle-cash-in-index variant: 8.93% CAGR but MaxDD **−63.8%**.
Mechanics: post-crash entry clustering (up to 52/week vs 20 slots) turns away 65% of
candidates in the best vintages; ~65% avg exposure; equal-weight undersizes tail winners.
(G4 v1 was invalid — same-week-exit zombie-position bug — fixed before these numbers.)

### FINAL VERDICT (phase 1): **SIGNAL (not investable as tested)** — see `results/RESULTS.md`
for caveats (survivorship-retracement interaction, 2020 dominance) and next levers
(regime gate, contention ranking, trailing exit).

### Phase-1b (2026-07-27, Arun's ASIANPAINT chart review): target-rule variants

Arun spotted the ASIANPAINT near-miss (target 3,582.90, high 3,568.00 — Rs15 short →
52wk time-stop +1.3% instead of ~+26%). Tested exact-touch vs 1%-buffer fill vs fractal
target across the universe (`run_g5_target_variants.py` → `results/g5_target_variants.csv`):
**base wins** — buffer99 net 4.39% (−0.23 vs base: the 1% haircut on the 33% of trades
that DO touch outweighs rescued near-misses); fractal net 2.69% (win 53% but winners
truncated). The taught rule (exact touch at full prior swing high) survives its own edge case.

---

## Phase 2 (2026-07-27): can it be made INVESTABLE? (Arun's ask)

Attack the four diagnosed G4 failures with principled levers, two stages:

1. **Per-trade exit variants** (`run_g6_trail_pertrade.py`): base target-26w vs
   trail-HMA44 (weekly close < HMA44) vs trail-Donchian-10w (close < prior 10wk low) —
   r/71 lesson: trailing ≫ target. Initial SL stays active; 25 bps; trade lists saved.
2. **Portfolio replays** (`run_g7_portfolio_opt.py`, all from 2005 for benchmark
   comparability): axes = regime gate (NIFTYBEES weekly close > 40wk SMA blocks NEW
   entries), slot priority (alphabetical vs reward:risk desc), slots (20×5% vs 40×2.5%),
   exit mode (target vs best trail). ~8 named cells, NOT a grid — multiple-testing noted.

Success gate: beat NIFTYBEES B&H on Calmar AND CAGR with MaxDD ≤ ~35%, holding per-year.

### Phase-2 results (2026-07-27 ~11:45 IST) — improved, still NOT investable

- **Per-trade (G6):** Donchian-10w trail replaces target → **net +11.11%/tr, PF 2.72, t 13.9**
  (vs target +4.62%/1.66). HMA44 trail too tight (+3.48%). r/71 confirmed: never a target.
- **Book (G7, 14 cells):** best = **Donchian trail, 40×2.5%, ungated: 15.04% CAGR /
  Sharpe 0.87 / MaxDD −51.2% / Calmar 0.29** vs NIFTYBEES 12.75% / 0.73 / −58.0% / 0.22.
  Beats index on all headline metrics — but fails the pre-set MaxDD ≤35% bar; excess lumpy
  (−28pp 2018, −16pp 2019, −24pp 2025); best-of-14 multiple-testing haircut applies.
- **Structural discovery: the regime gate HURTS every cell** (retracement-reversal alpha
  fires BELOW the 40w SMA — 2009/2020/2023 vintages). Retracement systems are anti-gate.
  R:R contention ranking never helps. 40 slots > 20 for the trail book.
- **Verdict unchanged: SIGNAL (not investable)** — Calmar 0.29 ≪ existing books (RS120 ~1.7).
  Reusable insights recorded: Donchian trail on weekly swing entries; anti-gate behavior.
