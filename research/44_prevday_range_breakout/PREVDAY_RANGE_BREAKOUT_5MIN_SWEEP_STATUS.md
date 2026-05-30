# Prev-Day Range-Compression → Next-Day Breakout Trend Trade — 5-min trigger, Stocks + NIFTY50

**STATUS: DONE (P1–P4)** — best SIGNAL of the three systems, but not a standalone
strategy. Raw = breakeven (P1); high-beta + HTF filters → long-only PF 1.24 (P2);
causal trailing-beta confirms it, monotonic in beta (P3); **20-DMA market-regime
gate → signal PF 1.31, portfolio MaxDD −55%→−37% (P4)**. Ceiling: still a
single-factor long-beta book (~+5.6% CAGR / −37% DD / Sharpe 0.4, 2024–26) →
deploy only as a hedged/diversified sleeve, not all-in. Verdicts: `results/RESULTS.md`,
`RESULTS_P2.md`, `RESULTS_P3.md`, `RESULTS_P4.md`.

Previous day compresses into a narrow range; the next day, when 5-min price
*closes* beyond that range, we trend-trade the breakout direction and ride it
with a stop/target (fixed or trailing), intraday or as a multi-day swing.

---

## 1. The Ask

**What you asked (paraphrased):** prev day moves in a short range; next day if
price closes above/below that range (5-min TF), trend-trade that direction with
trailing or fixed SL+target. Optional conditions: only narrow-to-normal CPR
(ignore wide), higher-timeframe trend confirmation, only high-beta stocks. SL
candidates: prev-day low (long) / high (short), or 50% of current day's range, or
other combos — find the optimal. "Be flexible, approach from your own angle, test
comprehensively."

**What we're actually testing:** Across the liquid 5-min universe (≈378 names) +
NIFTY50 (separate cohort), 2018→2026: define prev-day *compression*, trigger on
the first 5-min close beyond the prev-day range, and sweep the trade-management
space (SL definition × target × hold horizon × direction) to find the configs
with a real, cost-surviving edge. Then (Phase 2) gate the winner on CPR-width,
higher-timeframe trend, and high-beta selection.

**My angle (flexibility applied):**
- Compression isn't one thing — I test NR7/NR4 (Crabel), a 20-day range
  percentile, inside-day, and *no-compression* (plain prev-day-range breakout) as
  the baseline, so we can see how much the "narrow range" precondition actually
  adds.
- "Trend trade" implies it may run past one day, so **hold horizon is an axis**:
  intraday (square-off 15:25) vs **swing** (carry up to 10 trading days, managed
  on daily bars with overnight-gap handling).
- SL is an axis with four defensible definitions (your two + ATR + breakout-bar).
- Everything is measured in **R-multiples**, gross vs net (6 bps round-trip), with
  per-direction win%/PF — so long and short are judged separately from the start.

---

## 2. The Base — exact mechanics

Daily bars define the setup; 5-min (session 09:15–15:25) provides the trigger and
intraday management. Daily ATR = Wilder-14 on daily.

### Setup (prev trading day `i-1`)
- **Box** = `[PDL, PDH]` = prev-day low/high. `prevrange = PDH − PDL`.
- **Compression definitions (axis A):**
  - `NONE` — any prev day (baseline: plain prev-day-range breakout)
  - `NR7` — prev-day range is the narrowest of the last 7 days
  - `NR4` — narrowest of the last 4
  - `PCT25` — prev-day range ≤ 25th percentile of the last 20 daily ranges
  - `INSIDE` — prev day is an inside day (`PDH<H[i-2]` and `PDL>L[i-2]`)

### Trigger (day `i`, 5-min)
- **Long** when the first 5-min bar (≥ 09:20) *closes* > PDH.
- **Short** when the first 5-min bar *closes* < PDL.
- First side to break wins; one trade per stock per day. Entry = breakout bar close.

### Stop-loss (axis C) — R = |entry − SL|
- `BOX` — opposite box edge (long SL=PDL, short SL=PDH) — your prev-day-level idea
- `HALF` — entry ∓ 0.5 × prevrange — your "50% of range" idea
- `ATR` — entry ∓ 1.5 × daily ATR
- `BAR` — breakout bar's opposite extreme (tight)

### Target / exit (axis D)
- `1R`, `2R`, `3R` — fixed R multiples
- `MM` — measured move = prevrange projected from entry
- `TRAIL` — ratchet trail at 1R behind the run extreme (no fixed target)

### Hold horizon (axis E)
- `INTRA` — square off at 15:25
- `SWING` — if open at EOD, carry and manage on subsequent **daily** bars up to
  `MAXDAYS=10`; overnight gaps handled at next-day open (gap through stop/target
  fills at open).

### Costs
6 bps round-trip of notional, deducted in R. Gross and net both reported.

### Success criterion
Rank by net expectancy (R) per **direction**; gate on net>0, **PF>1.2**, and a
meaningful trade count. Report win%, expectancy, PF, total-R, avg hold for long
and short separately.

---

## 3. Plan — Phase 1 grid

| Axis | Values | n |
|---|---|---|
| Compression | NONE, NR7, NR4, PCT25, INSIDE | 5 |
| Stop-loss | BOX, HALF, ATR, BAR | 4 |
| Target | 1R, 2R, 3R, MM, TRAIL | 5 |
| Hold | INTRA, SWING | 2 |
| Direction | long + short (both reported) | — |
| Cohort | stocks, NIFTY50 | 2 |

**= 5×4×5×2 = 200 variants per cohort.** One breakout/day is detected once;
all 200 SL×target×hold×compression combos are simulated/routed on it in a single
pass (compression is a cheap per-day gate). Crash-safe incremental per-symbol
tallies; resumable.

## 3b. Phase 3 — robustness (DONE)

**FINAL:** Signal edge **survives causal trailing-beta** (vs synthetic market
index; NIFTY50 daily only starts 2023) and is **monotonic in beta** (β≥1.4 →
PF ~1.29) — real alpha, not look-ahead. BUT: (1) broad-universe 5-min mostly
starts ~2024 → robust window thin & bullish; loses in pullback years (2021 PF
0.31, 2026 0.74); (2) **naive portfolio is uninvestable — 55–87% MaxDD** because
it's one correlated long-beta bet. Not deployable as-is; needs a **market-regime
filter** + de-correlation (Phase 4). Verdict: `results/RESULTS_P3.md`.

---


Validate the P2 long-only winner (NR7 / high-beta / HTF / BOX-stop / 2-3R / swing):
1. **Causal trailing-window beta** (rolling 252d, computed only from prior data)
   replaces the full-period beta tag → removes the look-ahead. Confirm PF survives.
2. **β-cutoff sensitivity** (≥1.0 / 1.2 / 1.4) from one trade log.
3. **Per-year / walk-forward** PF stability (is it one lucky regime?).
4. **Portfolio sim** — per-trade R → equity curve with fixed-fractional risk +
   concurrency cap → CAGR, MaxDD, Sharpe.

Scripts: `run_nrb_p3.py` (emits `results/trades_long.csv` with causal beta),
`analyze_p3.py` (per-year + β-sweep + portfolio). Output: `results/RESULTS_P3.md`.

## 4. Phase 2 (after P1 winner) — confluence filters
- **CPR width:** keep only narrow-to-normal CPR days, drop the widest quartile.
- **HTF trend confirmation:** daily SMA50/200 (and/or 60-min trend) agrees with
  the breakout direction.
- **High-beta universe:** restrict to high-beta names (beta vs NIFTY50 over a
  trailing window; top quartile / β>~1.2).

---

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-05-29 | Folder + STATUS doc created | sections 1–4 before launch |
| 2026-05-29 | Engine + smoke (4 stocks) | looked strong (PF 1.11–1.15) — but large-cap selection bias |
| 2026-05-29 | Phase 1 full run DONE | 374 stocks + NIFTY50, 124s |
| 2026-05-29 | Phase 2 (filters) launched | CPR-width, HTF-trend, high-beta gating |
| 2026-05-29 | Phase 2 DONE | 379 syms, 81s. Filters work; long-only PF 1.24 |

**Phase 2 FINAL:** high-beta is the dominant filter (PF 1.003→1.06 alone), HTF
trend stacks (→1.11–1.14 combined). **System is LONG-biased** (mirror of H-pattern).
Best **long-only NR7|BOX|2R + htf+highbeta: PF 1.25, 41% wr, +0.156R, 1,711 trades**;
NR7|BOX|3R+all long PF 1.243. PCT25 = more trades (2.9k) at PF ~1.15. CPR-width
mostly redundant. Caveat: full-period beta = mild look-ahead (Phase-3 fix).

**Phase 1 read (full universe):** raw broad-universe edge is **breakeven** after
costs — best `PCT25\|BOX\|2R\|SWING` net +0.002R PF 1.003; rest breakeven-to-neg.
Compression (PCT25/NR7) + SWING + BOX/HALF SL + 2-3R are the *least-bad* (top of
the list) — directionally right but not enough alone. NIFTY50 PF ~1.2 but n=379,
one instrument → not robust. **Filters needed → Phase 2.** The 4-stock smoke was
misleadingly strong (clean large-caps); broad universe is the honest test.

*(appended live)*

## 6. Crash Recovery
- Progress: `tail -f /tmp/nrb.log`; symbols done =
  `cut -d, -f2 results/symbol_variant_stats.csv | sort -u | wc -l`.
- Resume: re-run `python research/44_prevday_range_breakout/scripts/run_nrb.py`
  (skips symbols already in the stats CSV).
- Aggregate-only: `... run_nrb.py --aggregate-only`.
- Don't edit `results/symbol_variant_stats.csv` while running.

## 7. Files
| File | Purpose | Committable |
|---|---|---|
| `scripts/nrb.py` | Breakout detection + trade simulator (intraday+swing) | yes |
| `scripts/run_nrb.py` | Driver, incremental tallies, aggregation | yes |
| `results/symbol_variant_stats.csv` | Per-symbol×variant tallies (resume key) | yes |
| `results/ranking_stocks.csv`, `ranking_nifty50.csv` | Per-variant rankings | yes |
| `results/RESULTS.md` | Final verdict | yes |

## 8. Findings
*(populated as results land)*
