# B1 Z-Score Mean Reversion (2–4d) — F&O Universe Daily, IS Coarse Screen

STATUS: DONE — WEAK SIGNAL (see Verdict)

Experiment **EXP-B1** of research/81 (`EDGE_DISCOVERY_81_STUDY_STATE.md`).
Family B: mean reversion. Grid locked BEFORE first run.

---

## 1. The Ask

Does fading multi-day overextension (z-score of close vs 20d mean) with a
2–4 session hold show a gross+net edge on the F&O universe in-sample
(2005–2017)? G1 coarse probe for family B.

## 2. Economic hypothesis (G0)

Short-horizon reversal = liquidity-provision premium: panic sellers /
chasing buyers demand immediacy; the fader supplies it and earns the
rebound. Counterparty: momentum/late trend-followers, margin-call flow.
Supported in-house: KC6 (dip-buy at KC lower band above SMA200) is live with
20-yr PF 1.70 — same family, band trigger instead of z-score. Known decay:
reversal has weakened since ~2021 (research/46); costs kill tight-stop
variants. Failure looks like: gross edge < cost, or edge only in crash years.

## 3. The Base — locked mechanics

- **Data/universe/period/engine/costs:** identical to EXP-A1 (daily F&O ~81
  names, canonical loader, IS 2005-01-01→2017-12-31, FUTURES_PROXY 3 bps,
  CA guard, engine @ 32-assertion suite).
- **Signal (long):** z[t] = (close − SMA20)/SD20 ≤ −z_thr AND close > SMA200
  (fade dips in uptrends only — KC6 prior).
  **Short:** z[t] ≥ +z_thr AND close < SMA200. Sides independent.
- **Stop:** FIXED k = 2.5×ATR14 (wide; reversion + tight stops is known-bad —
  research/46 + KC6 design. Held fixed to bound the grid).
- **Target axis:** none vs SMA20[t] (mean-revert-to-mean, KC6-style mid).
- **Time-stop:** 2 or 4 sessions.

## 4. Plan — pre-registered grid (LOCKED)

| Axis | Values |
|---|---|
| z_thr | 2.0, 2.5 |
| Direction | long (dip in uptrend), short (rip in downtrend) |
| Target | none, SMA20 |
| Time-stop sessions | 2, 4 |

**16 cells** × ~81 symbols. Ledger +16 (study total 40).

**G1 pass gate (same as A1, decided now):** pooled NET expectancy > 0 with
t ≥ 3, AND ≥55% symbols net-positive, AND stability across z_thr (no lone
spike). **Falsification:** all cells gross-negative → family B daily variant
= NO EDGE, move on. No grid extension without a new experiment ID.

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-15 ~20:05 IST | Pre-registered | runner adapted from A1 |

## 6. Crash Recovery

Runner `scripts/run_b1_zscore_daily.py`; resumable via done-set on
`results/b1_cells.csv`; trades in `results/b1_trades.csv`; log
`/tmp/b1_zscore.log` (VPS).

## 7. Findings

(after run)

## VERDICT (2026-07-15 ~20:15 IST): WEAK SIGNAL — G1 gate NOT passed, structure worth one pre-registered follow-up

Pre-registered gate (net t>=3) not met: best cell z2.5_S_sma20_ts2 = +31.7 bps
net, t=1.50, 56% symbols positive, n=474. BUT the structure is coherent, not
random: (1) SMA20 target dominates no-target in EVERY comparison; (2) net edge
rises MONOTONICALLY with z depth on the short side (z2.0->z2.5: +13.9->+31.7
bps ts2; +17.4->+35.4 ts4); (3) short-rips-in-downtrends > long-dips-in-uptrends
(longs mostly negative at ts2). Gross positive for 8/16 cells.

Honest read: family B has a real but underpowered gross edge at the tested
depths; n is small because z>=2.5-in-downtrend is rare. Follow-up EXP-B2
(deeper z, short focus) is justified by the monotone dose-response and will be
pre-registered separately (+ledger). Regime concentration (2008/2011 bear
years?) must be checked before any enthusiasm.

STATUS: DONE
