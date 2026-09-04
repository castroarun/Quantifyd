# Complementary Third Sleeve for TN+OA — Mean-Reversion / Pullback Candidates, Judged on BLEND VALUE

STATUS: RUNNING

## 1. The Ask (verbatim)

> "curate a complementary with least correlation system for our TN or OA or 50-50 combo of
> both. I'm thinking of mean reversion or some corrected stocks showing up with a bullish
> reversal pattern or some uppish stock after correcting and hit 50 sma/ema, shows up a green
> candle, entry above this green candle with SL below the low of this green and prev red.
> These are just random examples... assess and find out a meaningful and optimized system.
> The system may not be best on its own but complements the TN and OA systems and offers
> better numbers together." — Arun

**What we're actually testing:** long-only NSE cash-equity EOD systems that buy short-term
weakness inside strength (mean reversion / pullback reversal), scored NOT on standalone
performance but on whether a 3-sleeve portfolio (TN + OA + candidate) beats the deployed
TN+OA 50-50 after-tax baseline. A mediocre standalone that lifts the blend is a WIN; a great
standalone that duplicates existing beta is a KILL.

## 2. The Base — baseline & legs

- **Baseline (from r/144-145):** TN+OA 50-50 monthly-rebalanced, both legs after-tax, cash
  5%: **27.2% CAGR / −16.4% DD / Calmar 1.65** (10-OA-seed medians; 30-seed r/144 run:
  27.4/−16.4/1.68).
- **TN leg:** deployed spec (r/144 engine, NIFTYBEES-SMA100 weekly cash gate, top-8, D15),
  after-tax, deterministic — robustness via rebalance-day offsets {0,4,8}.
- **OA leg:** adopted spec (trail-15 SMA, −8% stop, 16 slots @6.25%, no gate, 25bps,
  cash 5%, calendar-year netted tax) via research/142 `bluesky_replay` — 10 seeds
  (r/145 showed 10-seed pair numbers reproduce the 30-seed run within 0.3pp).
- **Candidate sleeve mechanics (common frame):** PIT top-500-by-traded-value universe
  (ETFs/index series excluded; Kelter/KC6's live universe is Nifty 500 — this is its honest
  PIT proxy), 10 slots × 10% of sleeve equity, deterministic signal-strength ranking on slot
  contention (no seeds), 25bps/side cost (50bps RT) with 75bps RT sensitivity, cash 5%,
  after-tax with FY-netted 20% STCG / 12.5% LTCG (r/144 model). Daily marked. Windows:
  WA 2012→now primary, W1 2016-06→2019-12, W2 2020→now, W0 2006→now.

## 3. Pre-registered metric + kill thresholds (BINDING, before any run)

**Adoption rule — a candidate is ADOPTABLE only if ALL hold:**

1. **Blend lift:** best 3-sleeve weight (w3 ∈ {10,15,20,25,33}%, TN=OA=(100−w3)/2 each,
   monthly rebalanced, all legs after-tax) beats baseline by **+0.10 Calmar** (with blend
   CAGR ≥ 25.2%, i.e. ≤2pp give-up) **OR −2pp blend DD at ≥ equal CAGR (≥27.2%)**.
2. **Low correlation:** candidate daily-return corr **< 0.4 vs BOTH** TN and OA legs.
3. **Robustness:** rule 1 holds on the 10-OA-seed MEDIAN, is not catastrophic on the worst
   seed, and survives TN offsets {0,4,8}.
4. **Crash convergence:** blend worst-crash-window DD (2008, 2015-16, 2018, Feb-Apr 2020,
   2022H1) not worse than baseline's same-window DD by >2pp. Correlation averages hide crash
   convergence — mean reversion buys weakness exactly when momentum books are falling; this
   is measured explicitly, per window.

**G1 kill thresholds (cheap, per family):** net expectancy/trade ≤ 0 at 50bps RT → KILL;
daily corr vs TN > 0.6 → KILL (duplicate beta); tradeability gate violations (win rate, avg
win/loss, max losing streak reported IN every table — a low-WR/long-streak system is flagged
even if it ranks). Plateau > peak on every knob.

**Prior to confront (r/84, stated up front):** dip-buy + average-down = NO EDGE — its
win-rate was a take-profit/no-stop illusion, averaging is a tail bomb, and it lost to
NIFTYBEES B&H. Consequences here: every candidate uses **hard stops, no averaging**, and is
ranked on **net expectancy per trade**, never on win rate. Also relevant: r/129 (4th kill of
regime-gated credit spreads — "buy weakness" needs the uptrend filter to even have a chance),
and KC6's own 20-year validation (2,482 trades, 65% win, PF 1.70) as the one mean-reversion
system we already own.

**Falsification:** if nothing clears rule 1-4, the verdict is "TN+OA stands alone — no third
sleeve", and that is a valid, welcome outcome.

## 4. Plan — candidate families & grid

| Family | Rules (all: close>SMA200 uptrend filter, hard SL, no averaging) | Variants |
|---|---|---|
| F1 KC6 (owned, parked — FIRST) | entry close < EMA6 − 1.3×ATR6 (TR-EWM); exit standing limit at KC mid (fill if high ≥ mid), SL 5%, TP 15%, 15d max hold | base · SL 7% · no-TP · crash-filter on/off · mult 1.5 |
| F2 Arun's pullback-reversal | uptrend (close>SMA200, SMA50 rising over 10d); low touches/undercuts SMA50 (or EMA50) within 3d; green candle after ≥1 red; BUY-STOP above green high next day; SL below min(green low, prior red low); exits: 2R target / SMA20-break / 10d time | SMA vs EMA50 · target 1.5R/2R/3R · time 10/15d |
| F3 Connors oversold-in-uptrend | RSI(2) < 10 (var 5/15; RSI(3)<15), enter at close; exit close>SMA5 or RSI2>65 or 7d time; SL 7% (var 5%) | ~6 |
| F4 N-day-low washout | close at 7d (var 10d) low, enter close; exit close > prior high or 7d time; SL 7% | ~4 |

≈ 30 G1 cells × {gross, after-tax} ≈ 60 runs. Then blend stage: survivors × w3 grid ×
10 OA seeds × TN offsets {0,4,8}. Total compute ~30-45 min VPS.

**Seven sins:** look-ahead — signals on close t, entries at close t (KC6/Connors/washout,
matching live KC6's 15:20 entry) or next-day buy-stop (F2); all indicators causal.
**Survivorship — THE key risk for dip-buying:** Kite lists current instruments; names that
crashed and DIED are absent, so mean-reversion backtests are structurally flattered. Guards:
hard stops cap per-trade loss, WA 2012+ primary, stated on every table; treat all standalone
CAGRs as optimistic-end. Overfitting — coarse grids, plateau > peak, blend metric
pre-registered above, baseline default-wins. Costs — 50bps RT base (≥ coordinator floor),
75bps sensitivity. Regime — W1/W2 + per-crash-window table. Correlation — the entire point;
measured daily + monthly + per-crash. Capacity — top-500 universe, 10% sleeve slots at
proposed w3 ≤ 33% of a ₹20L-class book: not binding; stated.

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-03 23:1x | Mission received; KC6 defs read from services/kc6_scanner.py (read-only); STATUS written (sections 1-4) BEFORE any run | — |
| 2026-09-03 23:2x | `scripts/sleeve_engine.py` (OHLC panel + F1-F4 + G1 runner) built + launched | /tmp/tn146_g1.log |
| 2026-09-03 23:4x | **G1 DONE (44 runs) — three of four families KILLED at the gate.** | F2 pullback-reversal (Arun's sketch, 6 variants): expectancy −0.69..−0.91%/trade net, standalone CAGR −11..−19%, DD −85..−97% — the buy-stop + tight candle SL churns with no edge; KILLED plainly. F3 Connors RSI2/3 (6 variants): WR ~60% but expectancy −0.13..−0.16%/trade — the exact r/84 win-rate illusion; KILLED. F4 washout (4): expectancy −0.25..−0.30; KILLED. **F1 KC6 survives**: kc_crash (crash-filter on) is the ONLY positive-expectancy cell (+0.104%/trade net, 4.6% CAGR, DD −16.0, corr-to-TN 0.112, 1,229 trades); kc_m15 (1.5×ATR) = low-exposure quality sleeve (5.5%/−7.2%, Sharpe 1.27, corr 0.061, only 413 trades); kc_base ≈ breakeven (−0.016%/trade). Note: standalone KC6 here (net, slot-constrained, PIT top-500, after-tax) is far below the parked system's advertised gross PF 1.70 — the sleeves are near-cash. |
| 2026-09-03 23:5x | Blend stage launched: kc_crash / kc_m15 / kc_base / **cashnull** (explicit de-levering null — a KC6 sleeve must beat plain cash-third to count; r/134 lesson) | /tmp/tn146_blend.log |

## 6. Crash recovery

- VPS `/home/arun/quantifyd/research/146_complementary_third_sleeve/`.
- `ps aux | grep -E 'sleeve_engine|blend3'`; logs `/tmp/tn146_*.log`.
- Incremental CSVs: `results/g1_candidates.csv` (one row per variant × tax, label-keyed,
  reruns skip done), `results/blend3.csv`, `results/crash_windows.csv`; sleeve NAVs
  `results/nav_*.csv`; OA seed NAVs cached `results/oa_navs.csv` (regenerate via blend3.py).
- Resume: `cd /home/arun/quantifyd && setsid nohup venv/bin/python -u
  research/146_complementary_third_sleeve/scripts/<sleeve_engine|blend3>.py <phase>
  > /tmp/tn146_<phase>.log 2>&1 &`
- Nothing deployed is touched (kc6 services READ only, never imported into the run path).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| this STATUS md | live status | yes |
| `scripts/sleeve_engine.py` | OHLC panel + candidate families + G1 runner | yes |
| `scripts/blend3.py` | 3-sleeve blend stage (OA seeds, TN offsets, crash windows) | yes |
| `results/*.csv` | incremental results | yes (small) |
| `results/RESULTS.md` | final verdict | yes |

## 8. Findings

(populated per phase)
