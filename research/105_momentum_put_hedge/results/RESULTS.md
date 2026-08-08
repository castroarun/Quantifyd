# RESULTS — Put-Hedge Overlay vs the Cash-Exit Gate (research/105)

**VERDICT: SIGNAL (tenor-dependent), NOT yet a validated STRATEGY.** Replacing the weekly gate's
liquidate-to-cash with a **WEEKLY** NIFTY put hedge is materially better on after-tax risk-adjusted
return over 2019-2026 — but the equivalent **MONTHLY** hedge fails in every window tested, and weekly
options do not exist before 2019, so the winner has never faced a grinding bear.

## Headline (net of costs; "net" = after 20% STCG on gains realized <365d)

Full cycle 2011-2026 (monthly options — the only tenor testable this far back):
| Book | CAGR | net CAGR | MaxDD | net Calmar |
|---|---|---|---|---|
| A0 cash-exit (current live) | 32.1% | 29.4% | -16.6% | **0.96** |
| A1 hold naked (no gate exit) | 34.0% | 30.7% | -38.6% | 0.57 |
| Best monthly hedge (ITM2 r1.0) | 34.4% | 31.3% | -35.2% | 0.63 |

Same window 2019-2026 (weekly options available):
| Book | net CAGR | MaxDD | net Calmar |
|---|---|---|---|
| A0 cash-exit | 31.3% | -15.4% | 1.32 |
| A1 hold naked | 38.9% | -23.2% | 1.11 |
| Monthly hedge ATM r1.0 / r2.0 | 38.5 / 37.8% | -20.5 / -18.3% | 1.20 / 1.25 |
| **Weekly hedge ATM r2.0** | **40.3%** | -17.0% | **1.66** |
| **Weekly bear-put-spread ATM w10 r2.0** | **40.5%** | -17.0% | **1.68** |

## Reads

1. **Tenor is the whole story, not regime.** Run in the SAME 2019-2026 window, monthly hedging still
   trails the cash-exit baseline (1.20-1.25 vs 1.32) while weekly beats it (1.66). The gate flips often
   (69 hedge episodes / 38 rolls over 15y) and many risk-off spells are short — a monthly put pays for
   ~30 days of protection that is frequently abandoned in days; a weekly stops the bleed on gate reversal.
2. **The index hedge cannot cover the book's excess decline.** In 2015-16 the naked book fell -38.6%
   while NIFTY fell ~20%: a NIFTY-notional put only neutralises the index component, leaving the
   beta + idiosyncratic excess unhedged. Raising the ratio to 2.0-2.5x DOES cut MaxDD (-36% -> -32%)
   but the premium bill rises faster than the protection -> net Calmar gets WORSE (0.63 -> 0.55).
3. **The drawdown IS mostly inside the hedge window** (diagnostic): of the worst -38.6% episode, -31.1%
   accrued while the gate was risk-OFF and only -10.8% before it flipped. So gate lag is NOT the problem
   — hedge sizing/cost is.
4. **Moneyness:** ITM2 > ATM > OTM2 > OTM5 on the monthly arm (deep OTM puts are nearly free but useless);
   ATM is best weekly. Bear put spreads ~= long puts (slightly better weekly, cheaper premium).
5. **Hybrid (partial de-risk + hedge) is dominated** in both windows — it gives up return without a
   proportional drawdown benefit.
6. **The Donchian stop does not prevent big drawdowns; the macro GATE does.** The naked arm (-38.6%) has
   the identical per-stock 15-day stop as the gated arm (-16.6%). The stop caps per-position damage but
   the book keeps re-entering a falling market at each monthly rebalance; only the cash gate stops that.
7. **STCG:** raw cumulative STCG is NOT comparable across arms (bigger terminal book -> bigger absolute
   tax). The decision metric is net-of-STCG CAGR, reported above.

## Robustness gap (why this is SIGNAL, not STRATEGY)

- The winning arm (weekly puts) is testable **only 2019-2026** — one V-shaped crash (COVID) and a strong
  bull. It has **never been tested through a grinding bear** (2015-16), which is precisely where the
  monthly hedge failed worst. Do not deploy on this evidence alone.
- Fractional-lot sizing is modelled; at a Rs20L book one NIFTY lot ~ Rs18L notional, so real ratios are
  coarsely quantised (~1 lot). A ratio-2.0 arm needs ~2 lots ~ Rs36L notional against a Rs20L book.
- EOD marks throughout (matches how the book trades). Option prices filtered to OI>0.

## Next

- **Ext 3 (queued):** size each position ~1 F&O lot and hedge with **single-stock puts** — removes the
  index-vs-book beta gap that defeated the index hedge. Data exists from 2016; stock options are
  monthly-only (the tenor that fails here) and less liquid — test with a heavier slippage assumption.
- **Ext 2 (queued):** Nifty-500 universe with an explicit min-ADV liquidity screen + participation cap +
  cost sensitivity (0.15/0.30/0.50% per leg).
