# Research Index

All research conducted for Quantifyd, organized chronologically by execution order.

**Total: 216 files across 12 research phases + utilities**

---

## Folder Guide

| # | Folder | Period | What | Key Results |
|---|--------|--------|------|-------------|
| 01 | `01_covered_call_baseline/` | Dec 2025 | Initial covered call strategy optimizer & data simulator | Baseline strategy, superseded by MQ |
| 02 | `02_mq_portfolio_optimization/` | Jan-Feb 2026 | MQ (Momentum+Quality) portfolio parameter sweeps — portfolio size, stops, allocations | PS10 = 48.66% CAGR, PS30 = 32.19% CAGR |
| 03 | `03_breakout_v3_analysis/` | Feb 12-15 | Breakout V3 consolidation pattern analysis, filter optimization, trade verification | 65% WR, PF 1.70, Pine scripts for TradingView |
| 04 | `04_technical_indicator_sweep/` | Feb 7-16 | Technical indicators (EMA, RSI, SuperTrend, MACD, ADX) applied to MQ entry | All indicators HURT MQ — baseline wins |
| 05 | `05_crash_filter_confluence/` | Feb 14 | Crash detection filters, signal confluence, TTM Squeeze analysis | Crash filter useful, TTM inconclusive |
| 06 | `06_exit_risk_management/` | Feb 14-15 | Stop loss sweeps, trailing SL, exit rules, options hedging, futures, liquid funds | ATH drawdown exit dominates, stops >8% identical |
| 07 | `07_strategy_exploration/` | Feb 16-17 | Broad multi-strategy sweeps — momentum, mean reversion, hybrid, price action, EMA crossover, long/short | Per-stock analysis, 20Y exploration results |
| 08 | `08_ipo_strategy/` | Feb 17-18 | IPO (new listing) momentum strategy research across 5 sweep phases | 65% WR sweep, practical parameter sets |
| 09 | `09_mq_advanced_variants/` | Feb 18-20 | MQ concentration, KC6 models, MQ+technical hybrid, longterm (v1-v4), COVID analysis, model portfolio | Concentration = #1 CAGR lever, COVID recovery analysis |
| 10 | `10_combined_mq_v3_overlay/` | Feb 20 | Combined MQ + Breakout V3 overlay, 15-year backtest, rebalance optimization | Equity curves, MQ ATH trailing Pine script |
| 11 | `11_cpr_intraday_strategy/` | Mar 16 | CPR (Central Pivot Range) intraday strategy — baseline, BB+KC, RSI+Stoch, SuperTrend, regime filters | 79-stock full & OOS results, MQ correlation |
| 12 | `12_combined_swing_strategy/` | Mar 17 | Combined swing system (EMA trend + RSI mean reversion + NR7 breakout) on 60-min data, 113 config sweep | Best: E25/60 ADX20, 8.19% CAGR — underperforms index |

---

## Folder Structure Convention

Each research folder follows this layout:

```
NN_research_name/
├── scripts/          # Python scripts that ran the research
├── results/          # CSV outputs, JSON data, sample trades
├── reports/          # HTML dashboards & visual reports (if any)
├── verification/     # Trade verification logs, charts (if any)
├── pine_scripts/     # TradingView Pine scripts (if any)
├── logs/             # Computation logs (if any)
└── *.md              # Research summary / findings documentation
```

## Utilities

`_utilities/` — Helper scripts (code generators, data backfill tool, logo exploration)

---

## Best Results Summary

| Strategy | Best Config | CAGR | MaxDD | Sharpe | Calmar |
|----------|------------|------|-------|--------|--------|
| MQ Concentrated (PS10) | PS10_SEC70_POS30_TOP30_BIM | **48.66%** | 26.35% | 1.30 | 1.85 |
| MQ Balanced (PS30) | PS30_HSL50_ATH20_EQ95 | 32.19% | 27.0% | 1.05 | 1.19 |
| MQ + SuperTrend | STREND_atr7_m3.0 | 27.79% | 16.65% | — | 1.67 |
| Breakout V3 (KC6) | KC6 baseline | — | — | — | PF 1.70 |
| Combined Swing | E25_60_ADX20_R14_30_P20_10 | 8.19% | 8.0% | 1.40 | 1.03 |

**Key insight:** MQ portfolio strategy dominates. Technical indicators and swing trading on 60-min data cannot compete with fully-invested momentum+quality stock selection.

## 49 — volbreak_pdh_30min (2026-06-01)
**Vol > own 50-day MA + break prev-day high, 30-min intraday LONG**, all-exit
sweep (EOD/HARD_SL/2R/Chandelier/Supertrend), RSI axis, 8-name G1 smoke,
2018–2026, 6bps. **VERDICT: NO EDGE (net of cost)** — gross faint (best HARD_SL
+0.081R), every exit net-negative @6bps (best −0.029R, PF 0.95), no per-year
persistence, RSI no help. Cost (~0.08–0.11R) eats the edge; intraday < swing
(cf research/44). Shelved before the 30k-cell sweep. See research/49.../RESULTS.md.


## 50 - NAS 8-System 28-Day Live/Paper Review
**Real recorded fills of all 8 NAS systems, 2026-04-20 to 06-02 (25 days, 180 trades).** Factsheet: per-system equity (raw + per-lot), per-day x system heatmap, drawdowns, stats. **VERDICT: SIGNAL/AUDIT ONLY (28-day single regime, not validation).** Surfaced a data bug: the OTM trade recorder books exit=Rs0 -> Squeeze OTM (+115k) & 916 OTM (+204k) show fake 100% win / Rs0 DD, inflating the combined headline to Rs592k (~319k phantom). Trustworthy ATM half: 916 ATM2 +137k / ATM +68k / ATM4 +65k positive; Squeeze ATM/ATM2 ~flat-negative; 9:16 family beat squeeze-entry family. Phase 2 re-prices actual legs vs recorded option_chain for true P&L. See research/50_nas_28day_live_review/results/RESULTS.md.

## 51 - NAS Systems: true replay on recorded NIFTY weekly chain
**All 8 NAS systems replayed (rules re-run) on 28 days of recorded option_chain, lots=2 normalized.** Combined **net Rs-54,123 (NEGATIVE)**; only Squeeze ATM +Rs1.7k, 916 ATM2 cascade -23.9k worst, 9:16 family < squeeze-entry, day-win 32-44%. **VERDICT: NO EDGE this regime** (agrees with research/50 actuals ~flat). Fixed OTM roll-churn (1252->157 legs). 28d single-regime + squeeze-entry/1-min approximations = signal not validation. See research/51_nas_chain_replay/results/.

## 52 - NAS optimization (disciplined: 28d signal + years validation)
Found an evidence-graded playbook, NOT a dredged winner. (1) Edge at 1 DTE. (2) **Tight opening-range -> range day, robust over 6yr/1565d, positive EVERY year 2020-26 (corr 0.52)** - sell on tight open, skip wide. (3) Stop = **+/-0.4% underlying-move** (converges on 28d-real +2yr-stress; beats 1.3x premium whipsaw and no-stop whose 2yr worst-day = -58.8k). (4) Diversify across 916 vs Squeeze families (uncorrelated), drop OTM losers. Recommended config: 1-DTE ATM straddle, tight-open only, cross-family, +/-0.4% move stop. See research/52_nas_optimization/results/RESULTS_COMBINED.md.

## 53 - Regime angles: gap days & CPR width (years + 28d options)
Gap (1564d): big gaps esp DOWN = bigger moves/fewer range days (bad for short-vol). **CPR width is backwards vs textbook (wide->more move) AND its range-rate skill decayed to ~noise by 2024-26** - not a reliable filter. 28d real straddle: CPR narrow +13k/wide -17k (consistent w/ move-magnitude); gap-down +6.3k BEST but that CONTRADICTS the years -> small-sample luck, trust years. 1-DTE positive across all gap types. Verdict: gap/CPR are minor noisy overlays; 1-DTE + tight-open are the robust edges. See research/53_regime_angles/results/.
  - UPDATE (positional, user request): multi-day hold looked strong (daily-Supertrend
    net +0.701R/PF1.54) but the PLACEBO/BENCHMARK kill shows it is pure BETA not alpha
    — SIGNAL ~= BREAK_ONLY ~= random-day BASELINE for every exit. Volume filter adds
    nothing (slightly hurts). FINAL: NO EDGE / CONCLUDED on both intraday & positional.

## 55 — mtf_compression_breakout (2026-06-04)
**MTF compression breakout** (daily uptrend + narrow CPR + multi-day coil + prev-day/
20d-high break + volume), LONG, positional. 4 tests: largecap-5m, smallcap-5m(2024-26),
DAILY full-universe (1099 names, 2018-26, n=7501). **VERDICT: NO ALPHA (beta)** —
SIGNAL does NOT beat hold-the-uptrend baseline on any trailing exit (daily Supertrend
SIGNAL +0.33R vs BASE +0.93R); volume spike consistently HURTS; only +0.04R on tight
R-targets. Examples (TDPOWERSYS/DATAPATTNS/KMEW) = survivorship. Compression has
defensive value ONLY in 2022 bear -> use as regime filter, not entry. CONCLUDED.
See research/55.../results/RESULTS.md.


## 63 — gtaa_etf_rotation (2026-06-14)
**GTAA multi-asset ETF rotation** — validate Upstox "Strategy 1" (top-1 of Nifty/Gold/
Nasdaq-100 by ROC12, monthly; claimed Calmar 0.93) & beat it. Could NOT reproduce 0.93
(Kite serves these ETFs only from 2015). top-1 is WEAK 2016-26: Calmar 0.30 raw / 0.44
gated, -34/-25% DD. **WINNER = drop the selection, hold all 3 EQUAL-WEIGHT monthly:
Calmar 1.73, CAGR 19.5%, MaxDD -11.3%, turnover ~0, cost-insensitive.** Corr Nifty/Gold
-0.08, Nifty/Nasdaq +0.25, Gold/Nasdaq +0.04 -> diversification, not rotation, is the
edge. Same ~1.7 Calmar tier as research/41 & /62 at far lower DD/turnover/tax/complexity.
**VERDICT: STRATEGY (candidate)** for a simple low-DD core; period-dependence caveat
(MON100 +24.6% INR carries the absolute return). Published /app/backtest/gtaa-etf-rotation.
See research/63_gtaa_etf_rotation/results/RESULTS.md.


## 64 - factor_index_rotation (2026-06-14)
**Nifty factor-index rotation/diversification** (follow-on to research/63). Does
"diversify > select" transfer from asset classes to the Nifty single factors
(Momentum/Quality/Value/LowVol/Alpha)? **NO** - factors are mostly the same bet
(mean cross-corr 0.65; 0.79-0.91 vs Nifty). Equal-weight factors tops Calmar 0.76
(best pure-factor = Mom+LowVol, 17.4%/-22.9% DD); rotation 0.67. **The win is COMBINING:
Momentum factor + Gold + Nasdaq, INVERSE-VOL, monthly = Calmar 1.77, CAGR 22.1%, DD -12.5%,
cost-insensitive** - marginally beats research/63 Nifty book 1.75 by upgrading the equity
sleeve + taming Nasdaq vol. AllFactors+Gold+Nasdaq DILUTES 1.18 -> concentrate equity into
one factor. **VERDICT: STRATEGY (candidate)** - incremental upgrade to research/63; factor
selection/diversification alone = SIGNAL. Published /app/backtest/factor-index-rotation.
See research/64_factor_index_rotation/results/RESULTS.md.

## research/67 — Weekly vs Daily CPR (movement, direction, V2 fly gate) — SIGNAL/context, 2026-06-16
Validated the classic CPR rule on NIFTY weekly with the CORRECT metric (net move / containment, NOT
high-low range): narrow CPR -> trends, wide -> sideways/contained (recent close-inside-band 0->32%).
Stable 1st-30min directional tell (close above weekly CPR -> 69% week holds bullish; below -> 58% bearish;
full+recent identical). KEY: daily-vs-weekly SIGN FLIP — DAILY narrow CPR -> CALM (vol-persistence),
WEEKLY narrow -> trend (exhaustion). Weekly CPR too regime-fragile to hard-gate the fly (wide=calmer
recently but breaches MOST in 2020). combo_skip narrow-CPR skip UNCONFIRMED standalone (recent premium gap
only ~9%, breach-free window; research/61 1.03->2.00 Calmar is stacked w/ inside-week) -> owed isolated
re-test; keep conservatively. See research/67_weekly_cpr/results/RESULTS.md.

## research/74 — Intraday ATM premium decay-then-rise (CONFIRMED, pass 1, 2026-07-07)
Near-expiry theta collapse (expiry ATM straddle −40% by ~15:15; far-DTE flat) + a repeatable **IV spike at 15:15** (+1–1.5 vol, the 3:15 rise). Vega not directional; DTE is the master variable. Read-through: expiry 14:00→15:15 melt IS the short-straddle edge; 15:15 square-off sits at the premium trough. Pass 2 owed: exit-minute sweep + CPR/day-type. See research/74_intraday_premium_decay/results/RESULTS.md.

## research/73 — Weekly SuperTrend (10,3) trend-following, all cap bands (SIGNAL, 2026-07-07)
YouTube system (Vijay Khant): buy weekly ST(10,3) green, exit blind on red, size 5-7%, book 40/40/20, +5 hacks.
Tested mechanical core on Nifty50/200/Midcap150/Smallcap250, net 0.30% RT + STCG/LTCG, 2010-26. VERDICT SIGNAL:
real trend signal, beats B&H on every band at <= index DD; **Nifty200 best risk-adj + capacity: 17.5% CAGR /
-31.7% DD / Calmar 0.55 / Sharpe 1.03, beats NIFTYBEES 11/17 yrs, mult=3 robust ridge, OOS test Calmar 1.03
(> in-sample).** Two of the guests own rules HURT: 40/40/20 booking is a return-killer (caps the fat right
tail = 73-82% of return; 17.5->8.8%), regime gate redundant (ST self-exits; 17.5->11%). NOT better than the
regime-gated midcap book (Cal ~1.7) - weekly ST too slow to cut fast crashes (DD ~ market). Survivorship-
flattered (small-cap tail = todays multibaggers). A simple passive-plus, not new alpha. Published
/app/backtest/weekly-supertrend-nifty200. See research/73_weekly_supertrend_investing/results/RESULTS.md.

### ⚠️ research/73 CORRECTION (2026-07-07): NO INVESTABLE TIMING EDGE.
The "+6.9pp beats passive" was a BENCHMARK ARTIFACT (survivorship-selected today-Nifty200 book vs the Nifty50
index). Fair test (`fair_bench.py` / `all_bands_fair.py`) vs same-basket equal-weight B&H, ALL bands 2010-26:
ST timing LOSES on every band -- Nifty50 -6.6, Nifty200 -3.5, Midcap150 -6.4, Smallcap250 -2.8, Nifty500 -4.4
pp/yr, at equal-or-worse Calmar (basket wins except Smallcap where ST only helps by cutting the -54% DD).
Basket beats Nifty50 by +8..+11pp on every band = the whole headline. Per-trade entry edge real (G1) but
swamped at book level (SIGNAL != STRATEGY, cf research/49). Verdict downgraded NO TIMING EDGE. App study
corrected /app/backtest/weekly-supertrend-nifty200.

### research/73 PHASE 2 (2026-07-08): ST as a MARKET-LEVEL CRASH OVERLAY — this is where it works.
`crash_overlay.py` (daily res, causal). Hold the basket always; a DAILY ST(7,3) on the index flattens the
whole book in downtrends → pre-tax Calmar 0.56→**1.28** (Nifty200 DD −39%→−15%) for ~2pp CAGR; consistent
all bands (Small 0.40→0.86, N500 0.50→1.17), robust across fast family (dST 7/10/20 + 50DMA); **200-DMA
HURTS** (0.45). TAX is the real cost: liquidating the cash book ~2.5 sw/yr → net Calmar 1.01 → **build as a
NIFTY-futures/puts hedge (no sale = no tax)**. BONUS lever: dST(7,3) beats the 200/100-DMA gate → candidate
upgrade for the LIVE momentum book (research/62). Verdict: SuperTrend’s real edge = market crash overlay, NOT
per-name entry/exit. App study updated.

### research/73 PHASE 3 (2026-07-08): time the actual INDEX ETF — cleanest tradeable version.
`etf_st.py`/`etf_st2.py`. Trend-timing NIFTYBEES itself (hold green/cash red): net-tax ~1.5pp CAGR give-up
(10.6→9.0%) but DD MORE THAN HALVED (−36→−14%), Calmar 0.29→0.53, Sharpe 0.75→1.11. ST(7,3) marginally best
(least switches→least tax) but 50/100-DMA tied — any fast-medium trend filter; 200-DMA HURTS (halves CAGR).
Robust NIFTYBEES/JUNIORBEES/BANKBEES; GOLDBEES no. Well-known Faber-style timing, clean+scalable, not novel.
GATE CROSS-CHECK: swapping the LIVE momentum book (research/62) 100-DMA gate for a daily-ST gate is WORSE
(net Cal 1.71 vs 1.33) — KEEP the 100-DMA. (62i_st_gate.py.)

## 81 — Swing edge discovery (multi-family, 5-min + daily) — ACTIVE 2026-07-15→
Systematic pre-registered search for automatable ≤4-day swing systems, net-of-cost
(brief: docs/Trading-sytem-research-prompt-fable.md; crash doc EDGE_DISCOVERY_81_STUDY_STATE.md).
~160 cells so far. DEAD on daily 2-4d exits: Donchian, squeeze, EOD-carry, 5d-RS, PDH/PWH,
MA crossovers. SIGNAL: deep-z short fade (t1.5); CPR-open long (t2.4). **LEAD: morning-strength
continuation LONG (gap≥0.25%+ORB / O=L break / CPR-open), ≤4-session hold — index t3.0 +
stocks t2.9-3.2, G3 PASS, G4 Sharpe 1.0 / DD −17%; breadth + OOS pending.** Verdict: **SIGNAL (decaying) — not investable as tested; OOS consumed 2026-07-16.**

## 82 — Medium-swing 5-15d, long+short futures — CONCLUDED 2026-07-17
User-mandated extension of research/81 to the untested hold band. **SHORTS: NO EDGE, final —
all 24 cells across 4 families negative (t −3…−10); combined with r/81, directional short swing
is dead across 1-15 sessions.** LONGS: real cyclical breakout edge at 10-15d (N20-break t=4.0,
PWH-break t=3.7; Val pooled + but all 2020-21) — converges on the r/71 family already LIVE as
the breakout paper book; per-trade 200DMA gating does NOT work (lags tops). No build; OOS unconsumed.

## 83 — Faithful Turtle (Dennis) on F&O equities — CONCLUDED 2026-07-17
**Shorts: FINAL closure at ALL horizons** (turtle multi-week shorts worst yet: t −11, S2 0% yrs
positive; with r/81+r/82 the equity short-swing question is closed permanently). Longs: turtle
mechanics beat live-book rules in-sample (dual-system+2N, EQUAL-notional Calmar 0.45 vs 0.37;
N-sizing LOSES — 3rd sizing-scheme failure) BUT whole family ~flat 2018-23 ex-2020/21 → RECORDED
ONLY, live paper book soak decides. OOS unconsumed.
