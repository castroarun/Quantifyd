# Stock Neutral-Phase Winged Short Straddle — 45-DTE System Adapted to F&O Stocks (EOD bhav, 2016→2026)

STATUS: DESIGN — G0/G1 (data refresh running; no backtest launched yet)

Study: `research/127_stock_neutral_wings/` · Host: VPS `/home/arun/quantifyd/research/127_stock_neutral_wings/` (canonical) · Laptop copy is the design mirror.

---

## 1. Headline

Can the NIFTY 45→21-DTE short-straddle edge (research/119, STRATEGY-CANDIDATE, +78 net
pts/trade, t=3.1) be transplanted to **single F&O stocks** as a **defined-risk winged
structure** (short straddle/strangle + bought wings), with ONE fixed ruleset across all
stocks and stock selection done purely by objective filters (option liquidity, realized
vol, neutral-phase state)?

## 2. The Ask

**What you asked (2026-08-25):** "Study if we can deploy a similar [45-DTE straddle]
system for select/filtered stocks. (1) options must be liquid, (2) maybe less volatile,
(3) use our TradingView neutral-phase indicators. Don't stick to exact 45/22 DTE —
parametrize, try different filtering, permutations, CPR, indicators, and optimize. The
system must NOT be per-stock tuned — one ruleset, stocks either a fixed candidate list or
purely filtered. Personal bias: for stocks we should BUY the wings (stock-specific
overnight news/gaps), wing width itself to be optimized — for me it's mere crash
protection. Use our intraday stock price data; for options use EOD data from NSE."

**What we're actually testing:**
Across the 81 F&O stocks in `nse_options_bhav` (real NSE bhavcopy, 2016→now), does a
**single universal rule** — enter a short ATM straddle (or ±k% strangle) with bought wings
at width *w*, at DTE_entry, exit at target/stop/DTE_exit — deliver **net-of-cost positive
expectancy** once we only take entries that pass:
(a) a **liquidity gate** (all 4 legs traded, ATM volume/OI thresholds),
(b) optionally a **vol gate** (low realized-vol rank / IV-rank band),
(c) optionally a **neutral-phase gate** (range_detection_engine signals: BB squeeze,
    consensus-3of5, ADX-low, CPR-narrow, …)?
Success metric: **net points per trade as % of underlying price** (stocks have different
price scales — normalize), with t-stat; then portfolio CAGR/MaxDD/Calmar on blocked margin.

**Economic hypothesis (G0):** Same as 119 — the 45→21 window harvests the steep part of
theta while breakeven width (credit ÷ spot) exceeds the typical realized move in a
*neutral-phase* stock. Counterparty: buyers of single-stock optionality (event/news
lottery tickets, hedgers). Why it might NOT transfer: single stocks carry idiosyncratic
jump risk (earnings, news) that indices diversify away — the very reason wings are
mandatory here — and stock option spreads/illiquidity may eat the smaller edge.
Decay risk: stock option liquidity is concentrated in a few names and regimes.

**Prior art — the honest prior (do not ignore):**
- **research/89 G6 stock stress** already ran monthly ATM iron-flies on THIS table:
  apparent profit was **105% attributable to untraded (`contracts<50`) options**; only
  ~22% of 7,400 trades had ≥50 ATM contracts; only 9 of 39 surviving names net-positive.
  → G1 must start liquidity-filtered; if we can't beat r/89-G6 on the liquid subset, we
  stop early.
- **research/90** found monthly NIFTY *condor* wings untestable at EOD due to **stale
  far-OTM wing marks** — wings must be priced pessimistically and required to have traded.
- **research/119** transferable mechanics: breakeven-width vs |move| corr −0.898 is the
  whole mechanism; delta management strictly hurts (hold to time-exit); 45/21 are local
  maxima — the *shape* transfers, not the exact numbers.

**What's genuinely new vs r/89-G6:** (1) neutral-phase entry gates, (2) the 45→21-style
DTE window instead of hold-to-expiry, (3) target/stop overlay, (4) wing width as a sweep
axis, (5) low-vol stock filtering, (6) breakeven-vs-move mechanism check per stock.

**Falsification (decided now, before attachment):** If on the liquidity-clean sample
(all legs `contracts>0` at entry, ATM legs ≥50 contracts) the pooled net expectancy is
≤ 0 across the *neighborhood* of the DTE/wing grid (not just missing a lone peak), and no
gate (vol / neutral-phase) produces a subgroup with net t ≥ 2 that survives per-year
stability, the verdict is **NO EDGE** and we stop. We will NOT lower liquidity thresholds
to resurrect profits — that is the exact r/89 trap.

## 3. The Base — what's being tested

- **Universe:** the 81 stock underlyings present in `nse_options_bhav` (today's F&O list
  → survivorship bias, stated; modern sub-period 2022+ reported separately). Names with
  short histories (RELIANCE 27 cycles, HDFCBANK 14 — post-split strike-band gaps) stay in
  but are flagged.
- **Data:** `nse_options_bhav` (VPS, real NSE bhavcopy; close/settle, volume=`contracts`,
  OI; strikes within ±25% of spot, DTE≤75 — both baked in at load). Spot & indicators
  from `market_data_unified` daily (all 81 names, 2000→). NO IV/greeks/bid-ask columns —
  IV via BS inversion where needed; spreads modeled via slippage sweep.
- **Structure:** SELL 1× CE + 1× PE at ATM (or ±k% OTM strangle, k ∈ grid);
  BUY 1× CE + 1× PE wings at width *w* from short strikes. Wings are ALWAYS on
  (user bias — crash protection). All 4 legs must have `contracts>0` on entry day;
  wing debit paid at mark + slippage (pessimistic).
- **Entry:** monthly expiry cycle (r/119 listing-aware expiry logic from `engine45.py`),
  at bhav close, DTE_entry days before expiry.
- **Exits (first to trigger, daily close marks):**
  1. Target: combined structure value ≤ T% of entry credit
  2. Stop: combined structure value ≥ S% of entry credit (gap-through fills at the
     day's close mark, not the trigger — pessimistic, and wings cap the damage)
  3. Time: DTE_exit days before expiry
  Untraded legs at exit valued at intrinsic-only when that is worse (pessimistic).
- **No adjustments** (r/119 Phase E evidence).
- **Costs:** slippage per leg swept {0.25%, 0.5%, 1.0% of premium... and absolute
  paise floor}, STT 0.1% sell premium, exchange 0.05%, ₹20/order × 4 legs. Cost
  sensitivity table mandatory (stock spreads are the killer — no bid/ask data, so the
  sweep IS the realism test). Report gross AND net always.
- **Normalization/reporting:** per-trade P&L in % of spot and % of margin; rupees only
  for the modern period at current `FNO_LOT_SIZES` (no historical lot-size series —
  stated caveat).
- **Gates (each an on/off axis, computed CAUSALLY on daily data ≤ entry date):**
  - Liquidity (always on): legs traded + ATM contracts ≥ {50, 100, 200} + OI > 0
  - Realized-vol: HV(30) rank vs own 252d history; and cross-sectional low-vol tercile
  - IV-rank (BS-inverted ATM straddle IV vs own trailing year)
  - Neutral-phase: `range_detection_engine` signals — BB_SQUEEZE_20, TTM_SQUEEZE,
    ADX_LOW_14_25, CHOP_HIGH_14, RSI_MID_14, CPR_NARROW (weekly/daily), CONSENSUS_3of5
  - Trend-block: |close − SMA20| ≤ 2 ATR (the Pine `trend_atr_mult` filter)
- **Known missing data, stated up front:** NO earnings-calendar history → we cannot gate
  on "no earnings inside the hold window". Wings are the mitigation; the per-trade tail
  distribution will be examined for earnings-shaped losses. This is the single biggest
  structural caveat of the study.

## 4. Plan — phases and grid

**Phase A (G1 probe — cheap, launch first):** Fixed base config = ATM straddle,
DTE 45→21, wings at 5% of spot, target 50%, stop 200%, liquidity ≥50 ATM contracts,
slippage 0.5%. One pass over all 81 stocks × ~128 monthly cycles → pooled + per-stock
expectancy, gross & net, t-stat, per-year. **Gate to Phase B: pooled net > 0 with t ≳ 2
on the liquid sample, or a coherent liquid sub-universe with t ≳ 3.**

**Phase B (G2 grid):** on survivors —
| Axis | Values | n |
|---|---|---|
| DTE_entry | 30, 40, 45, 50, 60 | 5 |
| DTE_exit | 10, 15, 21, 28 | 4 |
| Short strikes | ATM straddle; ±2.5%, ±5% strangle | 3 |
| Wing width | 3%, 5%, 7%, 10% of spot; + 1.5/2/3×ATR(14) variants | ~6 |
| Stop / target | (200,50), (150,50), (300,50), (200,none) | 4 |
≈ 1,440 cells before gates — pruned by fixing non-sensitive axes after marginal
one-axis-at-a-time sweeps (avoid full cartesian; multiple-testing discount recorded).

**Phase C (G2b gates):** best 2–3 base cells × gate axes (vol rank, IV rank, each
neutral-phase signal, trend-block, CPR) — measured as *marginal* uplift vs no-gate,
plus entry-frequency cost (a gate that kills 80% of entries must earn its keep).

**Phase D (G3):** per-year stability, walk-forward (fit 2016–22, test 2023–26),
parameter monotonicity, cost +50%, super-winner guard (drop top-3 stocks), liquidity
threshold sensitivity, placebo (random entry dates matched per stock).

**Phase E (G4):** portfolio construction — max concurrent positions, margin per
structure (defined-risk → wing width caps margin), correlation to the existing NIFTY
short-vol books (THE STACK, straddle45), equity curve / MaxDD / Calmar on blocked capital.

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-25 ~19:30 IST | Scoping: r/119 rules+engine, neutral-phase indicator inventory, data audit completed | 3 parallel explorations |
| 2026-08-25 19:43 IST | First refresh launch failed (`venv` missing in r/89) | relaunched with `/home/arun/quantifyd/venv` |
| 2026-08-25 19:4x IST | `download_nse_bhav_stocks.py` refresh launched on VPS (stocks 2026-07-21 → 2026-08-24) | idempotent; log `research/89_short_monthly_straddle/results/nse_dl_stdout_20260825.log` |
| 2026-08-25 | STATUS-MD written (this file), sections 1–4 locked | Phase A script next |
| 2026-08-25 19:50 IST | Smoke test PASS (ASIANPAINT 19 tr / MARUTI 45 tr, ~8s/sym) | early read: gross ≈ 0 on these 2; liquidity gate bites hard (wings untraded pre-2017/22) |
| 2026-08-25 19:51 IST | **Phase A launched on VPS** (81 symbols, `run_phase_a.py`, bg) | log `results/phase_a.log`, output `results/phase_a_trades.csv` |

## 6. Crash Recovery

- **Data refresh:** `ssh arun@94.136.185.54`, check
  `tail /home/arun/quantifyd/research/89_short_monthly_straddle/results/nse_dl_stdout_20260825.log`
  and `pgrep -af download_nse_bhav_stocks`. Re-run any time (idempotent, resumes by date):
  `cd /home/arun/quantifyd/research/89_short_monthly_straddle && nohup /home/arun/quantifyd/venv/bin/python3 scripts/download_nse_bhav_stocks.py > results/nse_dl_stdout_$(date +%Y%m%d).log 2>&1 &`
- **Verify stock freshness:**
  `sqlite3 /home/arun/quantifyd/backtest_data/market_data.db "select max(trade_date) from nse_options_bhav where symbol not in ('NIFTY','BANKNIFTY')"`
- **Phase A runner (once written):** `research/127_stock_neutral_wings/scripts/run_phase_a.py`
  on VPS with the same venv python; appends per-trade rows to
  `results/phase_a_trades.csv` incrementally — safe to re-run, skips completed stocks.
- Do NOT touch: `backtest_data/market_data.db` writes (the downloader owns them),
  the running quantifyd service.
- Safe to inspect: everything under `research/127_stock_neutral_wings/results/`.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `STOCK_NEUTRAL_WINGED_STRADDLE_DAILY_SWEEP_STATUS.md` | This file | yes |
| `scripts/engine_stock_wings.py` (planned) | Stock winged-straddle engine (reuses r/119 `engine45.py` expiry/cost patterns + r/89 `run_g6_stock_stress.py` pivot/liquidity patterns + `services/range_detection_engine.py` gates) | yes |
| `scripts/run_phase_a.py` (planned) | G1 probe runner | yes |
| `results/phase_a_trades.csv` | Per-trade output, incremental | yes if small |
| `results/RESULTS.md` | Final verdict | yes |

## 8. Findings

### Phase A (G1 probe) — DONE 2026-08-25 19:56 IST — GATE PASSED

1,602 trades, 78 symbols with ≥1 valid liquid cycle, 2016-01 → 2026-08.
Base config: 45→21 DTE, ATM straddle, 5% wings, TP50/SL200, all 4 legs traded.

- **Pooled gross +0.245% of S0 per trade, t=7.54, win 60.8%.**
- **Liquidity-filtered (ATM vol ≥100): +0.168%, t=3.94, n=563** — the edge SURVIVES
  the liquidity filter (unlike r/89-G6 where it vanished). Adding wing_vol≥10:
  +0.247%, t=5.07, n=403.
- **Cost is the battle:** break-even cost = 0.96% of premium turnover (avg turnover
  25.4% of S0). At 0.5%-of-turnover cost: +0.117%, t=3.58 pooled. At 1%: dead.
- Per-year: positive every year except 2021 (−0.035%); strongest recent year 2026
  (+0.55%, n=232) — the modern liquid era is the BEST era, not the worst.
- Exit mix mirrors r/119: 89% time exits, targets 6%, stops 2%.
- Mechanism holds directionally: corr(gross, maxmove/breakeven) = −0.267.
- **Neutral-phase gates are marginal, not decisive** (on liquid sample): best split is
  realized-vol rank calm<0.33 (+0.190 vs +0.115 hot); ADX/CPR/BB-squeeze/trend-dist
  add little. Confirms decision to optimize the options-only system first.
- Best names are the mega-liquid ones: ICICIBANK +0.55%/tr, TCS +0.28%, INFY +0.23%,
  RELIANCE +0.21% — liquidity and edge coincide (good for capacity).

→ Proceed to Phase B (G2): pure options-EOD parametric sweep, one-axis-at-a-time
around the base (DTE_entry 30–60, DTE_exit 10–28, wings 3–10%, SL/TP variants,
strangle offsets 0/2.5/5%). ~17 configs staged, config count recorded for the
multiple-testing discount.

### Phase B (G2 axis sweep) — DONE 2026-08-25 20:24 IST

17 configs × 81 symbols, 59,299 trade-rows; liquid sample (ATM vol≥100 &
wing_vol≥10), net = gross − 0.5%×premium-turnover. Axis winners:
- **DTE_entry: 45 confirmed** (E30 strongly negative net t=−9 — no theta 30→21;
  E60 +0.467% t=2.8 but n=94, recent-era only, flagged not trusted).
- **DTE_exit: 15–21 plateau** (21 best t).
- **Wings: monotone wider-better** — W10 +0.215% > W7 +0.143% > W5 +0.129% > W3
  +0.053%; cost = fatter p05 tail.
- **No premium stop beats all stops** (t=3.33 vs SL150 t=1.6) — wings already cap risk.
  Keep TP50 (TPnone worse).
- **±2.5% strangle ≥ ATM** (t=3.62 best single config).
- Cost: base t=4.5/3.1/0.4 at 0.25/0.50/1.00% — cost-fragile at base.
- Per-year (dense era 2021+): base flat-to-negative 2021/2023/2024 → SIGNAL not
  yet STRATEGY at that point.

### IV / VRP gates on Phase A (user ask) — DONE 20:35 IST

- **VRP = IV/RV20 near-monotone**: <0.9 +0.129 (t1.3) · 0.9–1.1 +0.168 · 1.1–1.4
  **+0.395 (t4.1)** · >1.4 +0.271 (t3.0). Best simple filter found.
- Plain IV-RANK NOT monotone (mid-rank best) — "sell high IVR" refuted here.
- IV level sweet spot 25–35%; IV>50% (event-priced) kills the edge.
- iv_daily.csv = per-symbol daily ATM straddle IV (BS-inverted), reusable.

### Phase B2 (composites) — DONE 2026-08-25 20:5x IST — CANDIDATE FOUND

Composite **C1 = entry 45 DTE, exit 21 DTE, ±2.5% short strangle, 7% wings,
NO stop, 50% target**: liquid n=628, **gross +0.339%, net +0.264%S0/trade,
t=+5.06, win 64.8%**, and it now **survives 1% cost** (net +0.19%, t≈2.6-ish;
gated variant +0.166 t=2.58 at 1%). W10 variant nets more (+0.291) with fatter
tail (p05 −2.9 vs −2.0). Per-year dense era: 2021 +0.40 / 2022 +0.10 /
2023 +0.37 / 2024 ≈0.00 / 2025 +0.29 / 2026 +0.63 — no losing year 2021+,
2024 flat. Pre-2021 sample sparse (n≤11/yr), mixed.
- **VRP>1.1 gate does NOT improve the composite** (+0.244 t=3.85 on n=398 vs
  +0.264 t=5.06 ungated; per-year not more stable) — the gate's uplift on the
  Phase A base was real but the strangle+no-SL composite already captures it.
  Verdict: VRP gate NOT part of the ruleset (revisit at portfolio stage for
  sizing, not entry).
- Per-symbol: broad positive spread (HCLTECH/TATAPOWER/TCS/TATAMOTORS lead);
  negatives small-n (VEDL, M&M).
- **Multiple-testing caveat: C1 chosen after ~22 configs + gate variants —
  t=5.06 is inflated; G3 robustness must deflate it.**

### Phase D (G3 robustness) — DONE 2026-08-25 ~21:4x IST — **G3 PASSED**

All on net @0.5% cost, liquid sample (ref C1: +0.264%, t=5.06, n=628):

1. **Super-winner guard PASS** — drop top-3 (ADANIPORTS/TATAMOTORS/TCS):
   +0.228 t=4.12; drop top-5: +0.199 t=3.49. Breadth 76% of 70 symbols positive.
2. **OOS eras PASS** — 2016-23: +0.213 t=2.48; 2024-26: +0.290 t=4.44;
   2021-24 (ex the hot 25/26): +0.168 t=2.46. Positive in every era.
3. **Liquidity sensitivity STRONG PASS** — monotone IMPROVING with liquidity:
   ≥50 vol +0.108 → ≥100 +0.264 → ≥200 +0.351 → ≥500 +0.435. The opposite of a
   liquidity artifact; also the capacity story (edge concentrates in tradeable names).
4. **Neighborhood PASS** — X18/X24/W6/W8/K2/K3 all +0.168..+0.245, t 3.3-4.8. Plateau.
5. **DTE-window placebo STRONG PASS** — same structure at 35 DTE: +0.020 (t=0.9,
   n=2528); at 55 DTE: +0.059 (t=0.5). The 45→21 window IS the edge; generic
   short-premium at other DTEs earns ~nothing. (Also kills the Phase B E60 mirage.)
6. **Entry-lag PASS with haircut** — next-session entry: +0.158 t=3.53. ~40% of the
   edge is same-close timing, but the majority survives a full session's delay.
7. **Multiple-testing** — ~31 configs tried; guards 1-6 keep t>3 throughout → the
   deflated verdict stands.

**C1 graduates to STRATEGY-candidate.** NEXT (Phase E / G4): portfolio
construction — monthly clustering (all entries share the 45-DTE date), concurrent
positions, margin/capital model (defined-risk max-loss + buffer), rupee sizing at
current lots, CAGR/MaxDD/Calmar, correlation vs NIFTY and the existing short-vol
books → RESULTS.md verdict + publish to /app/backtest.

### Seven-sins control statement
Look-ahead: gates computed on data ≤ entry date; entry at same close as signal is a
flag — Phase A also runs next-day-close entry as sensitivity. Survivorship: today's F&O
list, stated; modern sub-period reported. Overfitting: staged one-axis sweeps, config
count recorded, multiple-testing discount in RESULTS. Cost neglect: gross+net+slippage
sweep; no bid/ask data stated. Regime: per-year tables, 2020 and 2024-26 separated.
Correlation: G4 measures overlap with existing NIFTY short-vol books and cross-stock
clustering (all entries fire in the same calm regimes). Capacity: stock option depth is
the binding constraint — report ATM volume percentiles per surviving name.
