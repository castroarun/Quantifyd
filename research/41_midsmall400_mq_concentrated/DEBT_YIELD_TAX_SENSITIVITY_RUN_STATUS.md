# Debt-Yield Tax Sensitivity — does taxing the 6.5% cash yield change the finalist?

**STATUS: DONE** · research/41 · Phase 31 · daily-marked engine · VPS canonical

## Ask

The 6.5% cash yield was added GROSS; debt funds are slab-taxed (post-Apr-2023). Hypothesis:
all-cash parks 100% in debt during risk-off (vs keep-top8's 53% equity), so taxing it
should hit all-cash harder and NARROW the gap. Test = net the daily cash factor to
6.5%×(1−slab); equity STCG stays 20%; slabs 0/20/30%.

## Results — net CAGR (%)

| Variant | Gross | Debt 0% | Debt 20% | Debt 30% | MaxDD | Calmar |
|---|---|---|---|---|---|---|
| all-cash+weekly | 35.5 | 29.0 | 28.6 | 28.4 | −20.7 | 1.72 |
| keep-top8 | 33.6 | 28.3 | 27.9 | 27.7 | −20.2 | 1.66 |
| all-cash base | 34.2 | 28.4 | 27.9 | 27.6 | −22.2 | 1.54 |

**Finalist gap (all-cash+weekly − keep-top8): +0.7pp at EVERY slab (0/20/30%).**

## Verdict — debt tax is small and does NOT change the ranking

1. **Small effect:** debt@30% costs only −0.6pp (finalists) to −0.8pp (base) of net CAGR
   vs untaxed. The book is fully in equity most of 2014-2026; the cash sleeve is a
   minority, so taxing it barely moves the needle.
2. **Hypothesis WRONG (instructively):** the gap did NOT narrow — it's a flat +0.7pp.
   The all-cash+weekly vs keep-top8 difference is EQUITY-driven (fast re-entry capturing
   the recovery), not a cash-sleeve artifact. Both finalists minimize time-in-cash (one
   by keeping 8 in equity, the other by re-entering fast), so debt tax hits them equally.
   Only the base (which sits in cash longest) loses marginally more (−0.8 vs −0.6pp).
3. **Decision unchanged:** all-cash+weekly stays marginally ahead on CAGR AND Calmar
   (1.72 vs 1.66); keep-top8 stays marginally shallower DD + never-full-dump. The choice
   remains a MANDATE question (max return vs no full liquidation), not a tax question.

## Files
- `scripts/31_debt_tax_sensitivity.py`, `results/phase31_debt_tax.csv`
