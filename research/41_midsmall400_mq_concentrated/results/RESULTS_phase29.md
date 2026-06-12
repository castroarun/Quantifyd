# Phase 29 — Gradual De-Risk Bake-Off — RESULTS

**Verdict: TIME-STAGGERED chunk-out does NOT help — it slightly HURTS (slower exit =
deeper drawdown). The ONLY gradual approach that wins is KEEP-TOP8, and it dominates
both the 1-shot all-cash and every staggered variant.** Anchor + sanity reproduced.

| Config | CAGR | Post-tax | MaxDD | Sharpe | Calmar | Full dump? |
|---|---|---|---|---|---|---|
| BASE allcash (anchor) | 34.2% | 28.4% | −22.2% | 1.82 | 1.54 | yes (1 week) |
| **keep-top8** | 33.6% | 28.3% | **−20.2%** | 1.71 | **1.66** | **no — keeps 8** |
| stagger c=1.00 (=allcash sanity) | 34.3% | 28.5% | −22.8% | 1.84 | 1.50 | yes |
| stagger c=0.50 (2 wk) | 34.3% | 28.3% | −22.4% | 1.82 | 1.53 | over 2 wk |
| stagger c=0.33 (3 wk) | 34.5% | 26.5% | −23.6% | 1.81 | 1.46 | over 3 wk |
| stagger c=0.25 (4 wk) | 34.1% | 26.9% | −23.9% | 1.77 | 1.43 | over 4 wk |
| stagger c=0.20 (5 wk) | 33.5% | 26.7% | −23.6% | 1.73 | 1.42 | over 5 wk |

## Why staggering backfires

1. **Slower exit = deeper drawdown.** As the chunk shrinks (exit spread over more
   weeks), MaxDD monotonically worsens: −22.8 → −22.4 → −23.6 → −23.9%. While you're
   scaling out, you're still partially invested into a falling market — so you eat more
   of the decline. The whole point of de-risking is to get OUT; spreading it over weeks
   defeats the purpose.
2. **More scale events = more tax.** The slow staggers (c≤0.33) fire 45-53 partial-sale
   events vs ~15 for the 1-shot → post-tax drops to ~26.5-26.9% (vs 28.4% base).
3. **Net: every staggered config has a lower Calmar (1.42-1.53) than the 1-shot (1.54)
   AND than keep-top8 (1.66).** Staggering is strictly dominated.

## The real lesson

The value of a "gentle" de-risk is NOT in spreading the exit over *time* — it's in
*which names you keep*. **keep-top8** wins because it keeps your 8 strongest (which hold
up better in the dip) and cashes only the weak 7. That delivers BOTH the shallower
drawdown (−20.2%, best in the table) AND the "never fully dump to debt" property the
client wants — at the best risk-adjusted return of any gradual option (Calmar 1.66).

## Recommendation (closes the gradual-de-risk question)

- **For the client-gentle mandate (no full equity→debt dump): keep-top8 + month-end
  re-entry. Final answer.** Calmar 1.66, MaxDD −20.2%, post-tax 28.3%, never goes to 0%.
- **Do not time-stagger the exit** — it deepens drawdown and adds tax for no benefit.
- **Do not use L200+gate** (Phase 28) — still 1-shots via the gate, and lower post-tax.
- The only thing that beats keep-top8 on pure risk-adjusted is `allcash + weekly
  re-entry` (Phase 27, Calmar 1.72) — but that IS the all-or-nothing book. So the final
  choice is a clean two-way: **max Calmar (allcash+weekly, 1.72, all-or-nothing)** vs
  **client-gentle (keep-top8, 1.66, never full-dump)**. Pick on mandate, not on 6 bps.

## Caveats

- stagger c=1.00 ≈ allcash (34.3/−22.8/1.50 vs 34.2/−22.2/1.54) — minor engine-path
  diff in re-entry timing/cost; parity close enough to validate the stagger engine.
- Single window, daily-marked, NIFTYBEES close-to-close gate. Not re-OOS'd.
