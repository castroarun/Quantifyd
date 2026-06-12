# MA-Overlay Exit Variants (always-on / +gate / bear-only) — RESULTS

**STATUS: DONE** · research/41 · Phase 33 · daily-marked engine · VPS canonical

## Ask

User: "don't exit all stocks, add the MA-overlay exit check on each stock." Three
mechanisms tested (L=100/150/200), month-end refill (NOT weekly → avoids Phase-28 churn):
V1 always-on no-gate · V2 always-on + gate (keep-top8) · V3 bear-only hold-15 (=Ph32 B).

## Results (gross / post-tax@20% / MaxDD / Sharpe / Calmar)

| Config | CAGR | Post-tax | MaxDD | Sharpe | Calmar | MA-exits |
|---|---|---|---|---|---|---|
| REF keep-top8 | 33.6% | 28.3% | −20.2% | 1.71 | 1.66 | — |
| REF all-cash base | 34.2% | 28.4% | −22.2% | 1.82 | 1.54 | — |
| REF Ph32 A-L100 | 34.2% | 28.9% | −20.1% | 1.76 | 1.70 | 60 |
| V1 always-on NO-gate L100 | 34.1% | 26.1% | −28.2% | 1.53 | 1.21 | 331 |
| V1 always-on NO-gate L150 | 34.8% | 27.2% | −33.9% | 1.50 | 1.02 | 153 |
| V1 always-on NO-gate L200 | 35.4% | 27.4% | −35.3% | 1.50 | 1.00 | 77 |
| **V2 always-on +gate L100** | 32.9% | 27.8% | **−18.9%** | 1.73 | **1.75** | 173 |
| V2 always-on +gate L150 | 33.6% | 28.4% | −20.0% | 1.73 | 1.68 | 61 |
| V2 always-on +gate L200 | 33.9% | 28.7% | −21.2% | 1.73 | 1.60 | 32 |
| V3 bear-only hold15 L100 | 34.2% | 28.8% | −22.4% | 1.69 | 1.52 | 127 |
| V3 bear-only hold15 L150 | 34.2% | 28.8% | −23.8% | 1.66 | 1.44 | 72 |
| V3 bear-only hold15 L200 | 33.9% | 28.6% | −24.1% | 1.63 | 1.41 | 46 |

## Verdict

- **V1 (always-on, NO gate) KILLED — again the gate is irreplaceable.** DD −28 to −35%
  (laggier L = deeper). Month-end refill DID fix most of Phase-28's tax bleed (post-tax
  26.1 vs Phase-28's 18.1 at L100) — but the structural drawdown problem (no collective
  circuit-breaker) remains. Higher L lifts CAGR (35.4 at L200) but rides the trend down too.
- **V2 (always-on + gate) is the best DRAWDOWN-CONTROLLER in the whole study.** V2-L100:
  **MaxDD −18.9% (shallowest ever) and Calmar 1.75 (highest ever)** — beats keep-top8 1.66,
  Ph32 A-L100 1.70, all-cash+weekly 1.72. BUT post-tax 27.8% — ~1pp below A-L100/keep-top8
  because the always-on (bull-too) overlay churns more (173 exits → tax). V2-L150 is the
  balance: DD −20.0, post-tax 28.4, Calmar 1.68 (61 exits).
- **V3 (bear-only, hold 15) reconfirmed worse** — DD −22 to −24, Calmar 1.41–1.52
  (exactly reproduces Phase-32 B). Skipping the keep-top8 trim holds laggards too long.

## The lesson

Running the per-stock MA overlay ALL the time (V2, even in bull) trims weakening names
*before* a bear is declared → shallowest drawdown (−18.9%) and best Calmar (1.75). The
cost is ~1pp post-tax CAGR from bull-market churn-tax. So it's a **DD-minimizer that
trades a little return** — right for a very risk-averse client. The bear-only overlay
(Ph32 A) keeps more return (28.9% post-tax) at slightly higher DD (−20.1).

## Updated finalist landscape (by use-case)

| Pick | Post-tax | MaxDD | Calmar | When |
|---|---|---|---|---|
| V2 always-on+gate L100 | 27.8% | −18.9% | 1.75 | lowest-drawdown mandate |
| all-cash+weekly | 29.0% | −20.7% | 1.72 | max return, accepts 1-shot |
| Ph32 A keeptop8+bearMA L100 | 28.9% | −20.1% | 1.70 | best balance, gradual |
| keep-top8 | 28.3% | −20.2% | 1.66 | simple gradual |

## Files
- `scripts/33_ma_overlay_variants.py`, `results/phase33_ma_overlay.csv`
