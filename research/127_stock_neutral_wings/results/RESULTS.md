# research/127 — Stock Neutral Winged Short Strangle (45→21 DTE) — RESULTS

**VERDICT: STRATEGY (candidate) — G3 PASSED, G4 CONDITIONAL PASS (margin realism
unverified). Ready for a paper book after the real-margin check.**

One universal ruleset (no per-stock tuning), F&O stocks, real NSE bhavcopy EOD
options (2016 → 2026-08), all-legs-traded liquidity discipline:

> **C1: at 45 DTE sell the ±2.5% strangle on the monthly expiry, buy wings 7% of
> spot away, no stop-loss, take-profit at 50% of net credit, time-exit at 21 DTE.
> Trade only entries with ATM volume ≥100 contracts and wing volume ≥10.**

## Headline numbers

Per trade (liquid sample, n=628, 80 stocks): **gross +0.339% of spot, net +0.264%
at 0.5%-of-premium costs, t=+5.06, win 64.8%**, p05 −1.96%. Survives 1% cost
(+0.166%, t=2.58). 89% of trades reach the time exit (same shape as r/119 NIFTY).

Portfolio (10 slots, ₹1Cr, margin modeled = 1.25×max-loss+2% ≈ 6.7% of notional,
idle capital at 5% — liquid-ETF assumption per Arun 2026-08-25):

| Margin assumption | Era | CAGR | MaxDD | Calmar | Sharpe |
|---|---|---|---|---|---|
| base (6.7%) | 2016–26 | 28.2% | −21.2% | 1.33 | 0.78 |
| base (6.7%) | 2021–26 (dense) | **38.5%** | −21.2% | 1.81 | 1.00 |
| ×1.5 (10%) | 2021–26 | 26.3% | −14.1% | 1.86 | 0.93 |
| ×2.0 (13.4%) | 2021–26 | 20.2% | −10.4% | 1.94 | 0.87 |

**Correlation with NIFTY monthly returns: −0.09.** In the 11 months NIFTY fell
>3%, the strategy averaged **+1.65%/month** — genuine diversification for a book
full of NIFTY short-vol.

Per-year (base margin): 2021 +56.1 / 2022 +16.0 / 2023 +46.1 / 2024 +18.2 /
2025 +39.5 / 2026 +49.2 (9mo). 2016–20 slightly negative on 1–2 positions/cycle
(universe barely tradeable then — capital mostly idle).

## Robustness (G3) — all passed

- **Super-winner guard:** drop top-3 contributors (ADANIPORTS/TATAMOTORS/TCS)
  → +0.228%, t=4.12; drop top-5 → +0.199%, t=3.49. Breadth: 76% of 70 symbols
  positive.
- **Era splits:** 2016–23 +0.213 (t=2.48); 2024–26 +0.290 (t=4.44); 2021–24
  excluding the strong 2025/26: +0.168 (t=2.46).
- **Liquidity monotonicity (the anti-r/89 test):** edge RISES with liquidity:
  vol≥50 +0.108 → ≥100 +0.264 → ≥200 +0.351 → ≥500 +0.435. The opposite of a
  stale-quote artifact.
- **Parameter plateau:** X18/X24/W6/W8/K2/K3 all +0.17..+0.25, t 3.3–4.8.
- **DTE-window placebo:** identical structure at 35 DTE +0.020 (t=0.9, n=2528),
  at 55 DTE +0.059 (t=0.5). **The 45→21 window IS the edge.**
- **Entry-lag:** entering the NEXT session keeps +0.158 (t=3.53) — no
  close-timing look-ahead driving the result.
- **Multiple testing:** ~31 configs tried across B/B2/D; guards keep t>3
  throughout, so the deflated verdict stands.

## Key insights

1. The r/119 NIFTY 45→21 theta-window mechanism transfers to stocks — but ONLY
   inside that window (30-DTE entry is net-negative t=−9; 35/55 placebos ≈ 0).
2. Wings should be cheap crash insurance (7–10% away), exactly Arun's prior; and
   with wings on, **no premium stop beats every stop** (also matches r/119).
3. VRP (IV/RV20) is a real monotone signal on the crude base config but adds
   NOTHING to the optimized composite → left out of the ruleset. Plain IV-rank
   is not monotone (refuted). Price-action calm gates (ADX/BB/CPR/trend-dist)
   are marginal → left out. **The edge is structural, not timing.**
4. Deliverable data asset: `results/iv_daily.csv` — per-stock daily ATM-IV
   series (BS-inverted, 2016→now, ~80 names), reusable for any options study.

## Honest caveats

- **Margin is modeled, not exchange-measured** (1.25×max-loss+2%). Real SPAN+
  exposure for stock condors may be higher; the ×1.5/×2 stress rows bound it and
  stay investable, but G4 fully passes only after a real basket-margin check
  (r/119 Phase-F recorder pattern). CAGR scales ~inversely with margin.
- **Costs are a slippage proxy** (%-of-premium sweep; no bid/ask data exists for
  stock options EOD). 0.5% assumed; break-even ~1.9% on the composite. Stock
  option spreads on non-top names can be worse — start any live test on the
  most-liquid tier only.
- **No earnings calendar in the data** — earnings gaps inside the hold are in
  the marks (wings cap them), but we cannot yet test "skip earnings cycles" —
  likely a free improvement once an earnings-dates source exists.
- Survivorship: today's F&O list applied to the past; mitigated by the modern
  sub-period being the strongest. Pre-2021 numbers are thin (1–2 trades/cycle).
- Bhav closes are settle-ish marks; untraded wing marks valued 0 at exit
  (pessimistic for us). Entry at same-day close; LAG1 sensitivity covers timing.
- Sample: 87 monthly cycles, 628 trades — decent but one regime (no 2008-style
  event; 2020-03 in sample only thinly).

## Next levers

1. **Real margin check** → then G4 full pass (place one paper basket via Kite
   margin API; or forward-record like r/119).
2. **Paper book** (`services/` pattern, top-liquidity tier, 5–10 slots) — soak
   vs backtest tolerance.
3. Earnings-date source → test earnings-skip and earnings-aware sizing.
4. Tearsheet + publish study to `/app/backtest` (registry entry pending).
5. Portfolio overlay ideas parked: VRP-based sizing (not entry), NIFTY-crash
   correlation exploit (strategy is +EV in NIFTY down months).

## Reproducibility

VPS `/home/arun/quantifyd/research/127_stock_neutral_wings/`; data snapshot
`nse_options_bhav` stocks to 2026-07-21 (refresh to 08-24 ran 2026-08-25 evening);
scripts `run_phase_a/b/b2/d/e.py`, `compute_iv_features.py`, analyzers;
costs 0.5%-of-premium base; R=6.5%. All trade tables in `results/*.csv`.
