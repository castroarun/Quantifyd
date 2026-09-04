# Options-Structure Overlay on the High-WR Killed Signals — real NSE bhavcopy, defined-risk

STATUS: DONE — **VERDICT: NO EDGE — 0 of 9 cells passes the bar. Connors/pullback structures significantly NEGATIVE net (t −2.7..−5.3); KC6 cells n≈80, t≈0.2-1.0. The structures achieve the promised 57-73% WR but expectancy stays negative — payoff shape was never the problem, the signal's left tail is; 2026 gives back everything (5th kill of the premium-on-timing-signal family).** See `results/RESULTS.md`.

## 1. The Ask (verbatim)

> "Connors RSI-2/3 ~60% WR, likewise KC6, -ve expectancy ok — how ab deploy options strategy
> like skewed iron condor, flies, bull/bear call/put spreads etc? same thing with pullback
> to 50 sma." — Arun

**The mechanism under test:** these signals have real ~60% directional hit rates (r/146/149)
but negative CASH expectancy because avg loss > avg win. Defined-risk premium structures
(bull put spread wins on flat-or-up; skewed IC on range) restructure the payoff to fit that
distribution. Signal fires → open a defined-risk structure in the front monthly on the F&O
stock → hold to expiry (defined risk, priced only at entry).

## 2. The Base

- **Signals (from the r/146/149 engines, identical definitions):** (a) Connors RSI(2)<10 &
  close>SMA200; (b) KC6: close < EMA6−1.3×ATR6 & close>SMA200; (c) pullback-to-50SMA candle
  pattern (r/146 base) — signal DAY only, cash exits irrelevant here. Max one structure per
  symbol per week per signal. Universe: the 82 F&O stock underlyings in `nse_options_bhav`.
- **Data window (checked, BINDING disclosure):** stock-options bhav is dense only from
  **2024-01 (9.56M rows 2024→2026-09)**; pre-2024 coverage is scraps (RELIANCE 2019: 140
  rows). **Window = 2024-01→2026-09, ~2.7 years — short; a bullish-regime sample. Stated on
  every table.**
- **Structures (small pre-set menu, NOT a grid — 3 per signal):**
  S1 bull put spread ~0.97/0.90 of spot; S2 tighter BPS ~ATM/0.95; S3 skewed iron condor
  (put side 0.97/0.90 + call side 1.07/1.12). Strikes = nearest actually-TRADED strikes
  (contracts>0 on entry day — the r/89 binding volume filter); skip the signal if any leg
  is untraded. Entry at bhav close prices; expiry = front monthly with 20-45 DTE.
- **Exit: HOLD TO EXPIRY.** P&L = net credit − intrinsic payout at the underlying's expiry
  close (from market_data). Deliberate: it avoids pricing illiquid marks mid-life entirely —
  only ENTRY prices need to be real, and they are (traded bhav closes). No early-exit
  variants in v1 (declared limitation).
- **Costs/slippage:** gross first; then credit haircut of 5% and 10% of gross premium
  (stock options monthlies are wide) + ₹40/structure fixed. Metric: net P&L / max-risk
  (return on risk) per structure.

## 3. Priors + pre-registered kill bar

- **r/129 killed MA-regime credit spreads at G1 (4th kill of that family) — but that was
  INDEX-level regime timing; this is STOCK-level signal-triggered entry. Genuinely
  different construction; evidence decides.**
- r/89 binding: options backtests must filter to really-traded strikes — done at entry.
- **Kill bar (BINDING):** a structure family survives only with positive mean return-on-risk
  at the 10%-of-premium haircut, t ≥ 2, positive in ≥2 of the 3 calendar years, and a
  capacity note (median traded contracts of the used strikes). Anything else: NO EDGE.
- Blend test vs TN+OA: only for a survivor.

## 4. Plan

3 signals × 3 structures = 9 cells (+2 haircut tiers each). One pass: build signal days from
the cash engine, bulk-load bhav 2024+ per symbol (indexed), price entries, settle at expiry,
write `results/overlay_trades.csv` + `results/overlay_summary.csv`.

Seven sins: look-ahead — signals at close t, entry at same-day bhav closes (EOD infra
consistent); survivorship — F&O underlyings are today's list (mild, 2024+ window);
multiple-testing — 9 pre-declared cells, no grid; costs — traded-price entries + haircut
stress; regime — per-year split + the 2.7y bullish-window disclaimer (a BPS overlay sells
puts in a bull tape — flagged); capacity — contracts note; correlation — computed for any
survivor only.

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-04 02:2x | Bhav coverage checked (dense 2024+ only), STATUS written, engine building | — |
| 2026-09-04 02:5x | Overlay run: 7,913 structures priced (traded strikes only), settled at expiry. All 9 cells fail the bar; per-year rows show 2024-25 bull-tape credits erased by 2026 (−5..−47% RoR). Two overlay.py instances ran concurrently from launch retries (read-only; identical outputs; harmless). RESULTS.md written, committed. | NO EDGE |

## 6. Crash recovery

- VPS `/home/arun/quantifyd/research/150_signal_options_overlay/`; log `/tmp/tn150.log`.
- Rerun: `cd /home/arun/quantifyd && venv/bin/python research/150_signal_options_overlay/scripts/overlay.py`
  (idempotent single pass, ~minutes; trades CSV rewritten whole).
- Reads market_data.db (incl. nse_options_bhav) READ-ONLY.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| this STATUS md | live status | yes |
| `scripts/overlay.py` | signal→structure engine | yes |
| `results/overlay_trades.csv`, `results/overlay_summary.csv`, `results/RESULTS.md` | outputs | yes |

## 8. Findings

(see RESULTS.md)
