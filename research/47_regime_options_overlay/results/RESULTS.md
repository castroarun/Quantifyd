# Regime Risk-Off Options/ETF Overlay — Results (directional, modeled IV)

**Verdict (UPDATED after all-years isolation): cash is near-optimal and NO options/
ETF risk-off action reliably beats it. The regime gate's value is being FLAT; any
directional overlay just adds variance for ~zero mean. Short futures (bets down) =
worst, loses on whipsaws. Covered-call ETF (long a weak market) = dominated by cash.
Call-spread re-entry (bets up) — which an earlier 2020/2025-only view flattered —
across all 47 risk-off months wins only 40% of the time, nets +0.5% total (~0/month),
and slightly LOWERS Calmar (2.09 vs cash 2.33). It helped 7 years, hurt 6 — a coin
flip. Bottom line: keep plain cash; options don't add reliable value here.**

## All-years isolation of the call-spread (the decisive test)

47 risk-off months, 2014–2026, marginal P&L of the call spread vs plain cash:

| | |
|---|---|
| Risk-off months it won | 19/47 = **40%** (market rose in only 47% of them) |
| Total marginal vs cash | **+0.5%** (~**+0.01%/month** — noise) |
| Years helped / hurt | **7 / 6** — coin flip (best +4.5pp 2017; worst −7pp 2015, −4.5pp 2024) |
| Full period | CASH 35.2% CAGR / Calmar 2.33 vs CALLSPREAD 35.1% / **2.09** |

A call spread during risk-off is a directional bet that the breach reverses UP; that
happens ~half the time and often not enough to clear the premium → 40% win rate, zero
net. Charts: `curve_cash_vs_callspread_vs_nifty.png` (CASH & CALLSPREAD curves are
visually identical, ~50x vs Nifty ~4x), `callspread_minus_cash_peryear.png`.

⚠️ **Modeled IV (realized-vol × markup), no historical options data → DIRECTIONAL,
not decision-grade.** IV-markup sensitivity (×1.0/1.15/1.30) barely moves the
rankings, so the *ordering* is robust even if absolute numbers aren't.

Base: research/41 SMOOTHEST midcap-RS core; regime = NIFTYBEES vs 100-SMA; 2014–2026.

## Comparison (modeled IV ×1.15)

| Risk-off action | CAGR | MaxDD | Calmar | Sharpe | 2020 crash | 2025 whipsaw |
|---|--:|--:|--:|--:|--:|--:|
| **CASH (baseline)** | 35.2% | **−15.1%** | **2.33** | 1.77 | +86% | −1.3% |
| SHORT 1× (baseline) | 31.6% | −26.7% | 1.18 | 1.41 | +95% | **−20.6%** |
| SHORT 1.3× | 30.9% | −26.7% | 1.15 | 1.38 | +97% | −21.8% |
| Long put ATM (hold midcaps) | 33.4% | −28.0% | 1.19 | 1.42 | +100% | −16.4% |
| Put spread (hold midcaps) | 32.5% | −27.9% | 1.16 | 1.37 | +84% | −16.4% |
| **Call-spread re-entry** | **35.1%** | −16.8% | **2.09** | 1.74 | +89% | **+1.7%** |
| Short-put income | 34.4% | −18.1% | 1.90 | 1.67 | +69% | −1.4% |
| **CC_ETF (covered-call ETF)** | 34.4% | −22.9% | 1.50 | 1.57 | +68% | −1.2% |

## The core insight

**The regime gate is, by construction, a "the market is weak now" signal.** So the
expected return of being long the market during risk-off is *negative* — that's the
whole reason to de-risk. Every action that retains market exposure during risk-off
therefore starts from behind:
- **CC_ETF** holds Nifty (capped) → eats Nifty's risk-off downside; the call premium
  doesn't compensate → −22.9% DD, dominated by cash. No strike tweak fixes it; it's
  structural.
- **Long put / put spread (hold midcaps)** → −28% DD because you ride falling midcaps;
  the put offsets only part. Worse than cash.
- **Short futures** → profits in a true crash (2020) but backfires catastrophically on
  the far-more-common whipsaw (2025 −21%). Worst Calmar.

The only thing that helps is a position that is **cheap, defined-risk, and only pays
off on the upside** — funded from idle cash:
- **Call-spread re-entry** = cash + a small ATM→5%-OTM call debit spread. Downside is
  just the premium (already negligible vs cash's 6.5%); upside catches the first leg
  of a rebound. Result: matches cash on CAGR (35.1 vs 35.2) and Calmar (2.09 vs 2.33),
  but turns the 2025 whipsaw **positive (+1.7%)** — it does exactly what you wanted
  ("manage if the breach reverses up"), with defined risk instead of a short or an ETF.
- **Short-put income** is a smaller, similar-spirited win (collect premium on parked
  cash) but with assignment risk in a continued fall — it cut 2020 to +69%.

## Answers to your specific ideas

- **"Move to Nifty ETF + covered calls instead of cash"** → tested; **dominated by
  cash** (lower CAGR, ~8pp deeper drawdown). Being long a flagged-weak market loses,
  covered or not. Not recommended.
- **"Naked put / debit spreads in line with capital"** → the *bearish* ones (long put,
  put spread) help a real crash but cost CAGR and deepen DD vs cash. The *bullish*
  defined-risk one (**call-spread re-entry**) is the keeper — it's the productive use
  of idle cash and the only action that beats cash on the whipsaw year.

## Recommendation (updated)
**Keep plain cash as the risk-off action.** The all-years isolation shows no options/
ETF overlay reliably beats it: short and covered-call-ETF clearly lose; the call-spread
re-entry is a coin flip that nets ~zero and slightly lowers Calmar. Drop the
covered-call-ETF, the short, and the call-spread. The regime gate already does its job
by going flat; adding a directional options bet just adds variance.
(If real Nifty options/IV data later confirms a *persistent* cheap-rebound edge, a tiny
call-spread could be revisited — but on this evidence it does not earn its place.)
