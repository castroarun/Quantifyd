# RESULTS — Momentum-250 Book: Leverage × Concentration for Return

**VERDICT: STRATEGY upgrade.** Leverage applied on top of the index-gated Nifty-250 momentum book raises return materially AND survives (0 margin calls across 2006–2026, incl. 2008 & 2020) — because the NIFTYBEES>EMA100 cash gate liquidates before drawdowns turn ruinous. It is a proportional trade (Calmar drifts down), not a free lunch. Net of 0.3% cost, daily-marked. B&H NIFTYBEES = 11.7% CAGR / −60% DD / Calmar 0.20.

## Efficient frontier (radj_z momentum, N8, index gate ON, EMA-stack OFF)

| Leverage | CAGR | Max DD | Sharpe | Calmar | Margin calls |
|---|---|---|---|---|---|
| 1.0 (base) | 33.8% | −29.9% | 1.49 | 1.13 | 0 |
| 1.3 | 41.4% | −38.9% | 1.40 | 1.06 | 0 |
| 1.6 | 48.7% | −47.8% | 1.35 | 1.02 | 0 |
| 2.0 | 57.8% | −59.6% | 1.29 | 0.97 | 0 |

Plain 12m momentum (ret252), N8: L1.0 37.5%/−34.5%/1.09 → L2.0 64.4%/−67.1%/0.96 — higher raw return, worse DD.
Concentration: N8 is the Calmar sweet spot at every leverage (beats N5 over-concentration and N12).

## Reads

1. Return scales strongly with leverage; drawdown ~proportionally; Calmar drifts DOWN — no free lunch, but no collapse (Sharpe stays 1.3+).
2. The index gate is what makes leverage survivable — 0 margin calls in 20y. Ungated, this would wipe out.
3. radj_z = best risk-adjusted; ret252 = highest raw return. N8 = best concentration.

## Recommended

**radj_z N8 at 1.3–1.6× → 41–49% CAGR, −39 to −48% DD, Calmar ~1.0** (+7–15% CAGR over unlevered 34%). Above 1.6× is greed (Calmar<1, DD>−55%).

## Caveats

- Multiples (8k–28k×) are 20y compounding fantasies — trust CAGR/DD/Calmar only.
- The −48% DD at 1.6× is real and daily-marked — brutal to hold.
- Margin-call model = 25% maintenance on DAILY marks; a gapping crash (2020) could force liquidation this understates → real leverage carries gap risk beyond the model.
- Financing assumed 8% p.a.; borrow-rate sensitivity owed on the chosen config.
- Same momentum edge magnified, not new alpha. Universe is a survivorship-free top-250 proxy, not the exact index.

## Vol-targeting — TESTED, does NOT help (counterintuitive but instructive)

Dynamic leverage from the book's own trailing realized vol (target constant vol, weekly re-lever) UNDERPERFORMS static leverage at every matched average leverage:

| Config | avg lev | CAGR | MaxDD | Calmar |
|---|---|---|---|---|
| static L1.3 | 1.30 | 41.4% | −38.9% | 1.06 |
| vt25_lb60 | 1.14 | 33.2% | −39.6% | 0.84 |
| static L1.6 | 1.60 | 48.7% | −47.8% | 1.02 |
| vt35_lb20 | 1.60 | 43.2% | −52.3% | 0.83 |

All 16 vol-target configs land BELOW the static frontier (Calmar 0.76–0.90 vs static ~1.05). Why:
1. The index-EMA gate ALREADY handles the downside (goes to cash in crashes) — vol-targeting duplicates it with lag + cost.
2. Momentum's biggest up-years are HIGH-VOL rallies (2014 +108%, 2021 +101%) — vol-targeting de-levers straight into the melt-ups, amputating the fat right tail that IS the edge.

So the managed-futures vol-target trick BACKFIRES on a gated long-momentum book. Static leverage wins.

## FINAL

Static leverage on gated radj_z N8 is the return upgrade: **1.3–1.6× → 41–49% CAGR, −39 to −48% DD, Calmar ~1.0, 0 margin calls / 20y.** Vol-targeting doesn't help; concentration beyond N8 doesn't; the index gate makes it survivable. Owed if productionised: borrow-rate sensitivity + gap-risk stress + futures-vs-MTF implementation.
