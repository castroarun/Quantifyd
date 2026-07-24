# Research 75 Phase 2 — Universe × Momentum-angle sweep

**VERDICT: the edge generalises and IMPROVES. Best risk-adjusted = MIDCAP + 6-month
relative-strength (Calmar 1.25–1.26, ~37% CAGR, −29% DD). Highest raw CAGR = mid+small COMBO +
12m return (43.5% net, but −42% DD and its cost/capacity are understated).** The per-stock
EMA-stack still hurts everywhere; longer momentum lookbacks beat short ones; the cash-yield
assumption is worth ~1–2.6% of CAGR.

All cells: 2006–2026, daily-marked, net 0.3% RT, gate ON, N=15, EMA-stack OFF unless noted.

## Ranked by Calmar (risk-adjusted — the honest ranking)

| Config | Net CAGR | MaxDD | Calmar | Sharpe | Turn | Mult |
|---|---|---|---|---|---|---|
| **mid · rs126 (6m RS)** | 37.2 | **−29.6** | **1.26** | 1.63 | 0.43 | 656× |
| **mid · rs120 (user's ask)** | 36.3 | **−29.2** | **1.25** | 1.60 | 0.45 | 576× |
| mid · rsblend (6m+12m) | 38.8 | −32.8 | 1.18 | 1.74 | 0.36 | 831× |
| mid · ret252 (12m) | 38.8 | −33.8 | 1.15 | 1.64 | 0.42 | 837× |
| mid · rs252 (12m RS) | 38.4 | −33.8 | 1.14 | 1.64 | 0.36 | 788× |
| n250 · rsblend | 36.2 | −32.8 | 1.11 | 1.57 | 0.35 | 565× |
| combo · rsblend | 41.5 | −38.1 | 1.09 | 1.79 | 0.38 | 1239× |
| n250 · ret252 *(phase-1 A3)* | 34.7 | −32.2 | 1.08 | 1.55 | 0.36 | 449× |
| combo · ret252 | **43.5** | −42.2 | 1.03 | 1.78 | 0.46 | **1649×** |
| combo · rs252 | 43.0 | −42.2 | 1.02 | 1.77 | 0.39 | 1529× |
| small · rsblend | 33.2 | −43.7 | 0.76 | 1.62 | 0.34 | 358× |
| small · ret252 | 33.2 | −46.2 | 0.72 | 1.66 | 0.32 | 356× |
| n250 · rs120 | 33.0 | −39.2 | 0.84 | 1.43 | 0.44 | 347× |
| n250 · rs55 (short) | 25.2 | −46.5 | 0.54 | 1.14 | 0.60 | 101× |

(smallcap/combo short-RS cells omitted — all worse; rs55 everywhere is the worst, highest turnover.)

## Findings

1. **Universe: midcap is the risk-adjusted sweet spot.** Midcap (rank 100–250) gives Calmar
   1.15–1.26 vs large-mid 1.08–1.11. Smallcap (250–500) has high CAGR but brutal DD (−43 to
   −59%, Calmar 0.44–0.76). **Combo (mid+small) has the highest raw CAGR (43.5%) but −42% DD.**
2. **Momentum period: longer is better; short RS is bad.** rs252 ≈ rsblend ≈ ret252 (best);
   rs126/rs120 slightly lower CAGR but *better DD* in midcap; **rs55 is the worst everywhere**
   (whipsaw, turnover 0.60). The user's **rs120 is specifically best-Calmar in midcap**
   (1.25, −29.2% DD) even though mid-pack in large-mid — the 120-day RS instinct was right
   *for midcaps*.
3. **EMA-stack still harmful:** n250 ret252 stackON 31.9% vs stackOFF 34.7%; rs120 30.4% vs
   33.0%. Confirms phase 1 across angles.
4. **Cash-yield sensitivity (base n250 ret252):** 6.5% → 34.7% CAGR · 4% → 33.7% · 0% → 32.1%.
   So the modeled liquid-fund yield is worth **~1% CAGR at a realistic 4%, ~2.6% vs 0% cash.**
   Not negligible; the headline numbers carry a modest cash tailwind.

## HONEST CAVEATS (why the biggest numbers are softer than they look)

- **Cost is understated for smallcap/combo.** Everything ran at 0.3% round-trip. Realistic
  smallcap RT is **0.5–0.7%+** with slippage/impact. The combo 43.5% and all smallcap cells
  are **optimistic** — discount them. Midcap at 0.3% is more defensible (midcaps reasonably
  liquid) but still slightly generous. **A cost-stress (0.5–0.7%) on the smallcap-heavy cells
  is owed before believing 43.5%.**
- **Capacity.** A concentrated top-15 smallcap/combo book has real capacity limits; fine for
  retail size, not for a fund. Large-mid/midcap scale better.
- **Survivorship (early years).** PIT is survivorship-free, but smallcap coverage 2006–2010 is
  thin/noisy — smallcap early-period numbers are the least trustworthy.
- **Redundancy.** Midcap momentum here is close cousin to research/41 midcap-RS120 and the live
  momentum-paper book. This sweep says "midcap + 6m RS is the best knob setting," not "new alpha."

## Next levers

- **Cost-stress** combo/smallcap winners at 0.5/0.7% RT to get their true net (likely knocks
  combo from 43.5% toward mid-30s, i.e. no better than midcap once costs are honest).
- If pursuing: **midcap + rs126/rsblend, gate ON, N15, no EMA-stack** is the defensible pick
  (~37–39% CAGR, −29 to −33% DD, Calmar 1.2–1.26). Tearsheet + paper it alongside momentum-paper
  only if it clears a correlation check vs the existing book.
