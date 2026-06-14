# Nifty Factor-Index Rotation / Diversification — does research/63's "diversify > select" transfer to factors?

STATUS: G1 PROBE DONE — thesis reframed; awaiting greenlight for G2 sweep

---

## 1. The Ask

**What you asked:** "can we also work on this switching/balancing/diversification over the
nifty factor indices as well?" (follow-on to research/63 GTAA ETF rotation).

**What we're testing:** Apply the research/63 toolkit (momentum rotation vs equal-weight vs
risk-parity, monthly, net-of-cost) to the **Nifty single-factor indices** — Momentum,
Quality, Value, Low-Vol, Alpha — and ask whether the "diversification beats selection"
result transfers, or whether factor *selection* (factor-momentum rotation) is the lever.

**Success metric:** net Calmar vs (a) Nifty 50, (b) research/63's equal-weight asset book.

## 2. Economic hypothesis

Single factors are all **long Indian-equity beta with overlapping constituents**, so they
should be far more correlated to each other than Nifty/Gold/Nasdaq were. If so,
equal-weighting factors ≈ owning the market (small diversification benefit) and the real
edge must come from **factor SELECTION** (factor momentum — the leading factor tends to
keep leading) and/or **adding the one genuinely low-corr factor (Low-Vol)** + the
research/63 asset diversifiers (gold/Nasdaq).

## 3. The Base (G1 probe — DONE)

- Data: NSE INDEX series (price-return), 5 factors + Nifty50, **2010-01 → 2026-06 (192
  months)** — far longer than the factor ETFs (mostly 2022+).
- Probe = coverage + per-factor return + **cross-correlation matrix** (the kill-cheap test).

## 4. G1 FINDINGS (2026-06-14)

**Coverage:** all 6 index series 2010→2026, clean. (ETFs available too but recent; use
indices for history, swap to ETF NAV for the live/tradable version later.)

**Per-factor (price-return, 2010-26):** Momentum 17.0% · Alpha 18.9% · LowVol 12.9% ·
Value 12.1% · Quality 11.5% · **Nifty50 10.4%** (vol: LowVol 25%! Alpha 22%, Mom 17.5%).

**Cross-correlation (monthly returns) — the decisive result:**
- **Mean off-diagonal = 0.65** (research/63 asset trio was ~0.0–0.25) → factors are MUCH
  more correlated. Momentum/Quality/Value/Alpha cluster at **0.71–0.90**.
- **Low-Vol is the lone diversifier: 0.42–0.47** to every other factor AND only **0.47**
  to Nifty50. The others are **0.79–0.91** to Nifty (mostly beta).
- Equal-weight 5 factors: **CAGR 15.0%, DD −22.7%, Calmar 0.66** vs Nifty Calmar 0.35.

**Read:** research/63's "diversification beats selection" does **NOT** transfer cleanly —
factors are too correlated; equal-weight gives Calmar 0.66, not 1.73. The levers are:
1. **Factor SELECTION / factor-momentum rotation** (dispersion is real: 11.5%→18.9%).
2. **Low-Vol as the in-set diversifier** + combining with research/63's gold/Nasdaq.

## 5. Status / event log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-06-14 ~12:30 | research/64 framed; G1 probe run | coverage + corr done; thesis reframed |
| — | G2 sweep | AWAITING GREENLIGHT (direction below) |

## 6. Crash Recovery

- Host VPS `arun@94.136.185.54:/home/arun/quantifyd`, `venv/bin/python`.
- Re-run probe: `venv/bin/python research/64_factor_index_rotation/scripts/g1_probe.py`
  (writes results/factor_monthly_closes.csv + factor_corr.csv).
- Index tokens hard-coded in g1_probe.py FACTORS dict (Momentum 290057, Quality 272393,
  Value 267529, LowVol 272137, Alpha 265993, Nifty50 256265).
- G2 reuses research/63 `gtaa_engine.py` (top-N / equal-weight / cash / cost) — feed it the
  factor monthly panel instead of the ETF panel.

## 7. Proposed G2 (pending approval)

Sweep on the factor panel (2010-26, net cost), reporting per-year + cost-sensitivity:
- (a) **Factor-momentum rotation** top-1/2/3 by ROC(3/6/12) + MA trend gate (does picking
  the leading factor beat equal-weight? — the research/63 question, inverted).
- (b) **Equal-weight & inverse-vol (risk-parity)** factor baskets (risk-parity matters here:
  LowVol index vol is 25% — equal-weight over/under-weights risk badly).
- (c) **Combined book:** best factor sleeve + Low-Vol + research/63 gold/Nasdaq — test whether
  the cross-asset diversifiers lift factor-Calmar toward the 1.7 tier.
- Benchmark: Nifty50 + research/63 equal-weight asset book.

## 8. Caveats (G1)

- Indices are **price-return, not total-return** → CAGR understates ~1.5%/yr dividends;
  correlation/relative conclusions unaffected. Live version must use factor-ETF NAV (TR-ish)
  and real cost; ETF history is short (recheck capacity/tracking).
