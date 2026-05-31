# Regime Risk-Off Options Overlay — Midcap Momentum (research/41 core)

**STATUS: DONE** (directional, modeled IV). **Cash is near-optimal (Calmar 2.33);
NO options/ETF overlay reliably beats it.** Gate fires when market is weak, so
staying long it (CC_ETF, midcaps+puts) underperforms; CC_ETF dominated by cash;
SHORT worst. **Call-spread re-entry FAILS the all-years test** — 40% win over 47
risk-off months, nets ~0, helps 7 yrs/hurts 6, Calmar 2.09<2.33. Keep PLAIN CASH.
Verdict + charts (curve vs Nifty50, per-year bars): `results/RESULTS.md`.

When the research/41 midcap-momentum system goes risk-off (NIFTYBEES < its
100-day SMA at month-end), instead of going to cash, deploy an options / ETF
structure sized to the deployed capital, and reverse it when the regime flips
back risk-on. Compare every risk-off action over 2014–2026.

## ⚠️ Data caveat (read first)
There is **no historical Nifty options / IV data** (option_chain = 16 recent days;
no India VIX). So **option premiums are MODELED via Black-Scholes** with
IV = NIFTYBEES 63-day realized vol × a stated markup (1.0 / 1.15 / 1.30). No
skew/term-structure/liquidity/option-tax. This is a **directional sensitivity
study, not a decision-grade result** — same caveat research/41 flagged when it
"dropped the options hedge (no data)". Conclusions are read as *directional*:
does an options/ETF risk-off action beat cash and fix the short-futures whipsaw?

## 1. The Ask
"Instead of going to cash on the SMA breach, deploy options (naked put / debit
spreads) in line with deployed capital and manage them if the breach reverses up.
Also: instead of cash, move to Nifty ETF + covered calls; reverse on regime
change. Try all structures and all sizings, compare."

## 2. Base system (unchanged from research/41 SMOOTHEST core)
Mid-cap RS-120 momentum, q0.5 win-rate, ATH≤10%, per-stock-100SMA exit + 12%
trail, monthly rotation, 15 names, top-22 retention buffer. Regime gate:
NIFTYBEES vs 100-SMA at month-end. 0.4% round-trip cost; cash 6.5% p.a.

## 3. Risk-off actions tested
**Overlay (HOLD midcaps + hedge), notional sz ∈ {1.0×, 1.3× beta-adj}:**
- `short` — short sz× Nifty (research/41 baseline; the double-edged one)
- `longput` — long Nifty put (ATM / 5%-OTM), defined risk
- `putspread` — Nifty put debit spread (buy ATM/5%OTM, sell 10%OTM), cheaper

**Replacement (LIQUIDATE midcaps → park capital), reversed on risk-on:**
- `cash` — 6.5% (research/41 baseline)
- `cc_etf` — Nifty ETF + covered call (sell 3%-OTM call) ← your new idea
- `shortput` — cash + sell 5%-OTM Nifty put (premium income)
- `callspread` — cash + Nifty call debit spread (defined-risk bullish re-entry)

IV markup sweep {1.0, 1.15, 1.30} on every option mode. Diagnostic years:
**2020** (Nifty crashed *after* risk-off → hedge should win) and **2025** (Nifty
rebounded *after* risk-off → short backfired −23pp; does defined-risk fix it?).

## 4. Metrics
CAGR, MaxDD, Sharpe, Calmar, per-year — vs the cash & short baselines.

## 5. Files
| File | Purpose |
|---|---|
| `scripts/run_overlay.py` | Engine (reuses research/41 `02_rs_sweep.py`) |
| `results/overlay_compare.csv` | All actions × sizings × IV: CAGR/DD/Calmar/2020/2025 |
| `results/RESULTS.md` | Directional verdict |

## 6. Status
| Date/time | Event | Notes |
|---|---|---|
| 2026-05-29 | Folder + STATUS, engine being written | extends research/41 phase-21 sketch |
