# Live Top-15 — `q0.5_dd__v__REG` (mid_120d_N15 core) + Fundamentals Overlay

**Config:** Phase-03 winner — RS(120d vs NIFTYBEES) on supplied MQ100,
quality screen (≥50% trailing-12m blocks positive), SMA200 regime gate.
Backtested edge: **35.3% CAGR / −24.6% MaxDD / Sharpe 1.53 / Calmar 1.44**.

**As-of:** 2026-02-16 (laptop frozen snapshot — **re-run `scripts/05_live_top15.py`
on the VPS for a true today-dated list**).
**Regime:** NIFTYBEES 290.76 ≥ SMA200 285.17 → **RISK-ON, book deploys.**
Quality screen: 68/91 supplied names with history passed.

## The 15 (RS-ranked) + current fundamentals (web: screener.in, ~Mar-2026 FY)

| # | Symbol | RS | PosFrac | ROE | D/E | PAT YoY | ROCE | Fundamental read |
|---|---|---|---|---|---|---|---|---|
| 1 | NATIONALUM | 1.61 | 0.58 | 29% | ~0 | +10% | 39% | **Strong** — debt-free, high ROCE |
| 2 | MUTHOOTFIN | 1.59 | 0.83 | 31% | 3.7* | +150% | 16% | **Strong** — *NBFC leverage normal |
| 3 | GMDCLTD | 1.49 | 0.50 | 8.8% | ~0 | erratic | 11% | **Weak** — low ROE, lumpy profit ⚠ |
| 4 | MCX | 1.43 | 0.58 | 43% | ~0 | +1879%† | 58% | **Strong** — †low base; ROCE elite |
| 5 | ANANDRATHI | 1.42 | 0.50 | 45% | 0.08 | +33% | 58% | **Strong** — best-in-list ROE/ROCE |
| 6 | HBLENGINE | 1.40 | 0.50 | 20% | 0.04 | +179% | 27% | **Strong** — clean balance sheet |
| 7 | INDIANB | 1.37 | 0.83 | 16% | bank | +12% | n/a | **Solid** — PSU bank, steady |
| 8 | NEULANDLAB | 1.37 | 0.67 | 22% | 0.16 | +40% | 27% | **Strong** |
| 9 | CUB | 1.34 | 0.75 | 13% | bank | +30% | n/a | **Solid** — bank, improving |
| 10 | FORCEMOT | 1.31 | 0.67 | 29% | 0.00 | +201% | 35% | **Strong** — debt-free, surging |
| 11 | INDUSTOWER | 1.30 | 0.75 | 20% | 0.57 | +18% | 19% | **Solid** |
| 12 | CUMMINSIND | 1.29 | 0.67 | 29% | ~0 | +15% | 38% | **Strong** |
| 13 | LTF | 1.28 | 0.67 | 12% | 4.4* | +18% | 8.5% | **Mixed** — low ROE for an NBFC ⚠ |
| 14 | ECLERX | 1.21 | 0.58 | 36% | 0.18 | +28% | 42% | **Strong** |
| 15 | FEDERALBNK | 1.20 | 0.50 | 11% | bank | +2% | n/a | **Mixed** — flat profit, low ROE ⚠ |

\* NBFC (Muthoot, LTF): high D/E is structural, not distress.
Banks (INDIANB, CUB, FEDERALBNK): D/E not meaningful — judged on ROE/PAT.

## Cross-check verdict

The price-momentum picks **largely coincide with fundamentally strong
businesses**: 11/15 Strong (high ROE + low/no debt + real profit growth),
2 Solid banks, 2 Mixed, 1 Weak. That convergence is reassuring — the RS
edge isn't selecting junk.

**Human-overlay caution flags (the screen can't see these):**
- **GMDCLTD** — ROE only 8.8%, erratic/declining profit. Momentum-driven
  (likely a commodity/lignite up-cycle); the weakest fundamental in the
  list. Size down or skip on a discretionary overlay.
- **FEDERALBNK** — net profit essentially flat YoY (+2%), ROE 11%. Riding
  price strength more than earnings.
- **LTF** — 12% ROE is low for a leveraged NBFC; thinner quality cushion.
- **MCX +1879%** and **FORCEMOT +201%** PAT growth are real but
  low-base / cyclical — treat the growth number with that context.

## How to action (honest)

This is a **systematic monthly-rotation** signal, not a buy-and-hold tip
sheet. To realize the backtested 35% CAGR you must rotate monthly per the
rule (top-22 buffer, regime gate) and survive a ~−25% drawdown. STCG
(~15-20%) is **not** netted in the backtest CAGR — post-tax is materially
lower. Re-run on the VPS for the current-dated list before any live use.
Phase-04 (walk-forward OOS + post-tax) is still pending and should gate
any real-capital decision.
