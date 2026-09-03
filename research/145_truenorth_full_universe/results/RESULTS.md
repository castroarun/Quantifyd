# Research 145 — True North Rules on the Open Alpha Universe — RESULTS

**VERDICT: CONCLUDED — FULL-UNIVERSE TN REJECTED; research/62's universe rejection is
REVALIDATED, with one honest nuance.** Running the exact deployed True North mechanics on the
Open Alpha universe (all NSE names with 20d-median traded value ≥ ₹5cr, no mcap floor) buys
+2.2pp median after-tax CAGR at the price of ~10.5pp deeper median drawdown (−35.6% vs
−25.1%), a collapsed 2016-19 window (7.9% vs 13.6%), a worse offset floor, a worse blend with
Open Alpha (Calmar 1.47 vs 1.65 — the pre-registered blend-value test FAILS), and a ~₹7.5cr
capacity ceiling. It fails every pre-registered gate. **The nuance vs r/62:** at today's ₹20L
book the binding objection is risk-adjusted QUALITY, not capacity (participation ≈0.5% —
executable); r/62's capacity wall bites from ~₹1cr AUM up, exactly as its impact model said.
The deployed Nifty-200 universe stands.

> Engine: r/144 deployed-faithful engine, universe swapped only (`scripts/tn_universe.py`).
> Universes: U-200 control (PIT top-200 by traded value) · U-500 (top-500) · U-OA (TV≥₹5cr
> at t-1, `bluesky_replay.TV_FLOOR`/`ETF_RE`, no mcap floor). All net-of-cost figures; tax =
> FY-netted 20% STCG / 12.5% LTCG. WA = 2012→2026-09 primary. Snapshot 2026-09-03.

## The prior confronted (H0: r/62 was right)

r/62 `p2_universe_bands.csv`: top-500 gross Calmar 1.44 vs top-200's 2.21, and at ₹1cr AUM it
collapses (11.7%/−66.3%, participation to 1081% of daily value; small band 0.1%/−82.3% at
₹1cr). Capture analysis: the small-cap mega-runners (SIGIND +5996%) had capture_ratio ≈0 even
when held — monthly momentum ranking catches them too late. **H0 is CONFIRMED here on
risk-adjusted grounds at every level tested.**

## (1) The size gradient — 12-rebalance-offset bands, net-of-tax, 2012→now

| Universe | CAGR med [min..max] | DD med / worst | Calmar med | W1 med | W2 med |
|---|---|---|---|---|---|
| **U-200 (deployed)** | 20.7% [14.9..25.1] | **−25.1% / −28.3%** | **0.88** | **13.6%** | 27.3% |
| U-500 | 19.3% [13.2..28.0] | −34.9% / −43.7% | 0.61 | 14.2% | 23.2% |
| U-OA (full) | 22.9% [12.8..28.5] | −35.6% / −41.7% | 0.66 | 7.9% | 38.3% |

Gross-of-tax medians: 24.8 / 23.2 / 27.5 with the same DD ordering. Findings:

- **U-500 is strictly dominated** — LESS CAGR than U-200 and 10pp more drawdown. The
  101-500 liquidity band adds noise names without adding captured alpha (r/62's capture
  finding reproduced from a second angle).
- **U-OA's extra CAGR is regime-concentrated**: W2 (2020+) 38.3% vs 27.3 — but W1 (2016-19)
  collapses to 7.9% vs 13.6. Per-year: 2021 +141%, 2020 +121%, 2017 +80% against −11.6%
  (2018), −7.7% (2015), −6.9% (2022). It is a higher-beta small-cap momentum book that lives
  and dies with small-cap breadth cycles.
- Offset floor: U-OA min 12.8% < U-200's 14.9% — wider path dependence, worse worst case.

## (2) Cost tiers (offset 0, net-of-tax)

| RT cost | U-200 (CAGR/DD) | U-500 | U-OA |
|---|---|---|---|
| 0.3% | 20.9 / −23.7 | 26.9 / −28.8 | 27.5 / −31.8 |
| 0.5% | 19.6 / −25.5 | 25.6 / −30.4 | 26.1 / −33.4 |
| 0.75% | 18.1 / −27.8 | 23.9 / −32.4 | 24.4 / −35.2 |

Flat-cost sensitivity is graceful for all three (~1.1pp per +25bps) — **flat cost is NOT
where the wider universes die** (r/62's collapse came from participation-scaled impact, i.e.
size). The kill here is drawdown/regime/Calmar, and capacity at size (next).

**Path-dependence warning baked into the numbers above:** these offset-0 rows run 4-8pp
HOTTER than their own 12-offset medians (U-500 26.9 vs median 19.3; U-OA 27.5 vs 22.9),
while U-200 sits on its median (20.9 vs 20.7). Wider universes are far more
rebalance-day-dependent — single-path backtests of them overstate; trust the band table in
section (1).

## (3) Capacity — the explicit note (held-name 20d-median traded value; slot = NAV/8)

| Universe | Held TV p50 | Held p10 typical | p10 worst-5% | Max book @10% participation (typ / worst) |
|---|---|---|---|---|
| U-200 | ₹49.2cr | ₹21.4cr | ₹2.3cr | **₹17.1cr / ₹1.9cr** |
| U-500 | ₹12.5cr | ₹4.5cr | ₹0.2cr | ₹3.6cr / **₹0.2cr** |
| U-OA | ₹21.7cr | ₹9.4cr | ₹4.9cr | ₹7.5cr / ₹3.9cr |

At ₹20L the U-OA book is executable (~0.5% participation). But the strategy family's stated
ambition (momentum books at ₹1cr+, Aurum productization) hits the U-OA ceiling ~2.3x sooner
than U-200, and U-500 is capacity-broken outright. **Also: the ₹5cr floor is nominal across
20 years** — the U-OA universe had only 112 names in 2006, 139 in 2012, 954 in 2026. The
"20-year backtest" therefore mixes a narrow large-cap universe (pre-2013) with today's
small-cap-inclusive one; its early-year numbers are not evidence about today's shape, and its
best years (2020-21) coincide exactly with the breadth explosion.

## (4) Does full-universe TN add anything to the TN+OA pair? NO.

50-50 monthly-rebalanced with the adopted OA spec, both legs after-tax, 10 OA seeds:

| Pair | Blend CAGR med [range] | Blend DD med / worst | Calmar | corr to OA (d/m) |
|---|---|---|---|---|
| **OA + TN U-200 (current)** | 27.2% [26.7..28.1] | **−16.4% / −16.7%** | **1.65** | 0.41 / 0.55 |
| OA + TN U-OA | 30.4% [29.9..31.3] | −20.6% / −21.9% | 1.47 | 0.41 / 0.51 |

- Pre-registered blend-value test (pair's after-tax Calmar must improve): **1.47 < 1.65 →
  FAIL.** The U-OA leg re-imports the small-cap beta OA already carries; the pair's DD
  worsens 4.2pp for the extra CAGR — that is sizing-up risk, not diversification.
- TN U-200 vs TN U-OA correlation: 0.75 daily / 0.68 monthly — largely the same book.
- Holding-name overlap of TN U-OA with concurrent OA positions: median 20% of TN's 8 names
  (p90 50%, 3 OA seeds). The overlap is moderate in NAMES but the factor overlap shows up
  where it matters — in the blend's drawdown.
- Want more return from the pair? The honest lever is weighting OA up (OA solo after-tax
  ~34.9%/−26.4%), not blurring TN into a slower OA.

## Recommendations

1. **REJECT the universe extension** (U-OA and U-500) for the live book and for the blend.
   The deployed Nifty-200 True North stands unchanged — third consecutive confirmation
   (r/62 → r/144 → r/145).
2. Nothing here changes the r/144 conclusions; the TN+OA pair remains the standout
   portfolio construction with the U-200 TN leg.
3. If small-cap momentum exposure is ever wanted at today's ₹20L scale, OA already IS that
   exposure (daily-triggered, stop-managed, seed-diversified). A monthly-ranked top-8 on the
   same names is a strictly worse harvester of it (blend Calmar says so).

## What was NOT tested, and why

- r/62-style participation-scaled slippage (their model, their result — cited as the prior;
  our flat tiers + explicit capacity table cover the same question at today's size).
- Re-tuning TN knobs (n, Donchian, gate) per-universe — the ask was "TN as is"; per-universe
  re-optimization would be a new study with a multiple-testing bill.
- Inflation-adjusted TV floor — would change the early universe materially; noted as the
  right construction if this is ever revisited.
- Overlap vs more than 3 OA seeds / OA slot-level capital overlap.

## Seven sins

Look-ahead: same causal engine as r/144; TV floor uses t-1 (OA convention). Survivorship:
**material in the small-cap tail** — Kite lists current instruments (~528 syms in 2006 vs
2,321 now); U-OA's early years and its 2007/2021 spikes are survivor-flattered; stated on the
tables. Overfitting: zero new tuned parameters (mechanics frozen; universe is the only axis;
metric pre-registered). Costs: 0.3/0.5/0.75% tiers + capacity table. Regime: W1/W2 split is
the story and is reported. Correlation: measured vs OA (0.41) and vs U-200 TN (0.75).
Capacity: the explicit table above.

## Reproducibility

VPS `/home/arun/quantifyd/research/145_truenorth_full_universe/`: `scripts/tn_universe.py`
(phases sweep/capacity/blend; reuses r/144 `tn_sweep.py` engine + r/142 `bluesky_replay`),
`results/universe_sweep.csv`, `capacity.csv`, `blend_universe.csv`, `peryear.csv`,
`nav_*.csv`. Data: market_data.db snapshot 2026-09-03. ~87 sweep runs + analytics, ~12 min.
