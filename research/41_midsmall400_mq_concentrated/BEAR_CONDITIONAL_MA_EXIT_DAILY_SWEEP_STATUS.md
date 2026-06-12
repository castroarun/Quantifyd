# Regime-Bear-Conditional Per-Stock MA Exit — two versions

**STATUS: DONE** · research/41 · Phase 32 · daily-marked engine · VPS canonical

## Ask

User: during a regime-bear, (A) keep-top8 PLUS exit any kept stock breaching its own
100/150/200-MA, or (B) sell nothing wholesale, only exit stocks breaching their own MA.
Key difference vs Phase 28: the per-stock MA exit fires ONLY during bear (not always),
so it doesn't churn/tax-bleed in bull markets.

## Results (gross / post-tax@20% / MaxDD / Calmar)

| Config | CAGR | Post-tax | MaxDD | Sharpe | Calmar | MA-exits |
|---|---|---|---|---|---|---|
| REF keep-top8 (locked) | 33.6% | 28.3% | −20.2% | 1.71 | 1.66 | — |
| REF all-cash base | 34.2% | 28.4% | −22.2% | 1.82 | 1.54 | — |
| **A keeptop8+bearMA L100** | 34.2% | **28.9%** | **−20.1%** | 1.76 | **1.70** | 60 |
| A keeptop8+bearMA L150 | 33.8% | 28.5% | −20.0% | 1.73 | 1.69 | 20 |
| A keeptop8+bearMA L200 | 33.7% | 28.4% | −20.2% | 1.72 | 1.67 | 10 |
| B nosell+bearMA L100 | 34.2% | 28.8% | −22.4% | 1.69 | 1.52 | 127 |
| B nosell+bearMA L150 | 33.2% | 28.0% | −23.8% | 1.62 | 1.39 | 63 |
| B nosell+bearMA L200 | 33.2% | 27.9% | −24.1% | 1.61 | 1.38 | 31 |

## Verdict — Version A WINS, Version B loses

- **Version A (keep-top8 + bear-time MA prune) IMPROVES keep-top8 at every L**, best at
  **L100: Calmar 1.66→1.70, post-tax 28.3→28.9% (+0.6pp), MaxDD −20.2→−20.1%.** And it's
  cheap/low-churn — only 60 MA exits over 12y (fires only in bears, so no Phase-28 tax
  bleed). Mechanism: after keeping the 8 strongest, you also cut any that keep falling
  through their own 100-MA — pruning the survivors that roll over. It's GRADUAL (keep 8,
  then prune name-by-name; never a 1-shot dump) yet nearly matches all-cash+weekly
  (Cal 1.72 / net 29.0). Strong new client-gentle candidate — strictly dominates plain
  keep-top8.
- **Version B (no wholesale sell, MA only) LOSES** — DD −22.4 to −24.1%, Calmar 1.52→1.38,
  worse than keep-top8 AND the 1-shot base. Skipping the up-front keep-top8 trim means you
  hold weak names too long into the bear; MA breaches alone are too slow a circuit-breaker.
- **L100 > L150 > L200** in both — longer MA = laggier = less protective.

## Recommendation

Promote **A keeptop8+bearMA L100** as the refined client-gentle variant (beats plain
keep-top8 on return AND drawdown, stays gradual). Do NOT use Version B. Next: per-year /
robustness re-read on A-L100 before locking (same treatment as Phase 30).

## Files
- `scripts/32_bear_conditional_ma_exit.py`, `results/phase32_bear_ma.csv`
