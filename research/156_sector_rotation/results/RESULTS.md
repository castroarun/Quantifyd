# Research 156 — Sector trend detection: rotation across sectors, and sector-gated stock picking

**VERDICT: NO EDGE for sector rotation (branch A). NO ADDED VALUE for the sector filter (branch B).
Neither branch is adoptable, standalone or as a complement.**

Two independent things were tested, and both failed for different, cleanly separable reasons.

**Branch A — allocating across sectors.** Of **1,440 rotation configurations** on the nine real NSE
sector indices (8 trend signals × 5 sector counts × 4 weightings × 3 rebalance clocks × 3 gates,
each on 4 rebalance-day offsets = 5,760 runs), **zero** clear the pre-registered bar of 20% CAGR
after tax with Calmar ≥ 1.0. The best-CAGR configuration returns **16.3% CAGR at −36.9% drawdown
(Calmar 0.44)**; the best-Calmar configuration returns **13.5% at −21.5% (Calmar 0.63)** — and its
drawdown protection comes from a NIFTY500-above-200-SMA cash gate, not from the sector choice.
For comparison, over the identical window, **equal-weighting all nine sectors returns 14.0%
(Calmar 0.32), NIFTY 500 buy-and-hold 13.2% (0.35), and the plain Midcap 150 index 18.0%
(0.41)**. Doing nothing but holding the Midcap 150 beats every sector-rotation cell we built, on
both return and risk-adjusted return.

The signal is not *absent* — it is *worthless*. Against a 500-draw random-sector null that rotates
at random on the same clock and pays the same costs and taxes, the best momentum-ranked
configuration sits at the **94th–100th percentile** of the random distribution at every sector
count. Momentum genuinely ranks sectors better than chance. **It just ranks them worth less than
the diversification the ranking destroys** — exactly the r/63 lesson ("diversification beat
selection") repeating on a new asset class.

**Branch B — using sectors to narrow the stock universe.** This is where the interesting numbers
were: holding the momentum leaders inside the top-ranked industries returns **32.5% CAGR after tax
at −38.3% drawdown (Calmar 0.85)** over 2016–2026, which clears Arun's 20% bar with room. But the
pre-registered decomposition kills it. Running **the identical stock rule over the full universe
with no sector filter at all** returns **32.9% at −39.3% (Calmar 0.84)** — statistically the same
book, and on paired offsets the sector-filtered version wins on CAGR in only **5 of 16** paired
comparisons. All of the return comes from the *stock* momentum ranking (which beats random stocks
inside the same sectors on 91–100% of 100 draws); **none of it comes from the sector layer.**
Restricting to random sectors costs about 8pp of CAGR, and restricting to momentum-ranked sectors
recovers almost exactly that 8pp and stops there. The sector layer is a round trip to nowhere.

**A data finding that matters beyond this study.** The wide 20-industry cross-section built from
stock constituents *looks* strongly predictive (momentum information coefficient t = 3.6 to 5.6,
top-minus-bottom spread 8–14% a year). It is an artefact. Validated against the real sector
indices, **every synthetic basket out-drifts its real index by +4 to +14 percentage points of CAGR
per year**. On the same eight sectors over the same window, the real indices score t = 0.9–1.4
while the synthetic baskets score t = 2.6–2.9. And with **industry labels randomly shuffled**, the
synthetic panel still produces t ≈ 1.0–1.9 with a 95th percentile of 3.0–3.5. Most of the apparent
"sector momentum" is stock-level momentum among survivors, aggregated into equal-weight baskets and
mislabelled as a sector effect. Any future study that builds sector proxies from today's index
membership must run this validation first.

**Complement value: none.** Against the deployed TN 40 / OA 40 / IPO 20 book on each candidate's
own window, every candidate at every weight from 5% to 30% **loses CAGR and is beaten by a plain
cash sleeve at the same weight on Calmar**. Correlations to the incumbents are 0.41–0.54 daily
(0.30–0.62 monthly) — above the pre-registered 0.40 ceiling for both legs, in every case.
Sector-flavoured Indian equity is Indian equity.

**This confirms and extends r/147** rather than contradicting it. r/147 killed sector rotation on a
single cell (top-2 by 126-day momentum, monthly, always-invested). Reproduced here after tax, that
cell returns **8.9% CAGR at −56.0% drawdown, Calmar 0.16** — the worst book in the study. The
sweep says the kill was not an artefact of that one cell: the whole design space is dead.

---

## 1. What was asked, and what was tested

> "we have many sector indexes like nifty real estate, nifty it, nifty auto, bank nifty so and
> so.... can we do a completely fresh study - if there is a way to figure out the trending(s),
> to-be trending sector(s), either ride on them in some prporotaios or further drill down into
> those sector leaders stocks and get some curated portfolio than can make execllent returns
> (above 20% cagr), even up to or better than our current oa and/or tn. ... Our idea is to also see
> if this study can complemenet our portfolio if not for a standalong king maker"

Restated: **(A)** can sector leadership be detected well enough to allocate across sectors, and
**(B)** can it be used to narrow the stock universe into a curated portfolio? Success bar: >20%
CAGR standalone, ideally at or above Open Alpha and True North; failing that, does it complement
the book?

Nothing was inherited from Open Alpha or True North. No ATH-close trigger, no RS ≥ 70, no 16 slots,
no −8% stop, no 15-SMA trail, no 100-SMA weekly gate, no Donchian. Twelve trend/strength families
were written from first principles and the data was allowed to choose. Measurement discipline —
costs, taxes, offset ensembles, paired comparison, null controls, plateau checks — was kept, because
that is how a fresh look avoids fooling itself.

## 2. Data reality — and one hard constraint

| Series | Bars | From | Integrity |
|---|---|---|---|
| NIFTYAUTO, NIFTYIT, NIFTYENERGY, NIFTYFINSRV, NIFTYFMCG, NIFTYMETAL, NIFTYPHARMA, NIFTYPSUBANK, NIFTYREALTY | 2,894–2,895 | **01-Jan-2015** | Clean: 0 phantom holiday rows, ≤1 missing day vs the NIFTY50 calendar, no split-scale steps, ≤4 days with abs(return) > 12% (all real events) |
| BANKNIFTY | 3,887 | 03-Jan-2011 | Clean, but a **subset of NIFTYFINSRV** — excluded from the cross-section to avoid double-counting the same bet |
| NIFTYMEDIA, NIFTYINFRA, NIFTYPVTBANK, NIFTYCONSUMPTION, NIFTYCOMMODITIES | — | — | **Not in the database.** Not used |
| NIFTYMIDCAP150 / NIFTYSMLCAP250 | 3,887 | 03-Jan-2011 | 1,990 rows are O=H=L=C — pre-2015 history is close-only. Close used throughout |

Given r/64 found Kite's Quality / LowVol / Commodities index series **corrupt**, the same checks were
run here before anything else. The nine sector series pass.

**The binding constraint is the sample: nine assets over 11.7 years.** ~140 monthly rebalances, one
crash (2020), the 2018 and 2022H1 grinds, no 2008. Book windows start Jan-2016 after the 260-day
signal warm-up. Every headline number carries that window.

### The synthetic 20-industry panel — built, validated, and rejected as evidence

To widen the cross-section and reach 2008, 20 equal-weight industry baskets were built from the
**500-symbol / 20-industry** map in `nifty200_official.csv` + `niftymidcap150_official.csv` +
`niftysmallcap250_official.csv` (all 500 present in the database; 241 have data from ≤2008).

The mandatory validation against the real sector indices over the 2015–2026 overlap:

| Industry | Real index | Corr daily | Corr monthly | Synthetic CAGR | Real CAGR | **Drift** |
|---|---|---|---|---|---|---|
| Information Technology | NIFTYIT | 0.808 | 0.826 | 21.31% | 9.20% | **+12.11pp** |
| Healthcare | NIFTYPHARMA | 0.819 | 0.813 | 21.77% | 8.07% | **+13.70pp** |
| Automobile & Auto Components | NIFTYAUTO | 0.855 | 0.874 | 23.49% | 11.26% | **+12.23pp** |
| Fast Moving Consumer Goods | NIFTYFMCG | 0.743 | 0.781 | 19.40% | 7.54% | **+11.86pp** |
| Metals & Mining | NIFTYMETAL | 0.932 | 0.936 | 26.47% | 14.85% | **+11.62pp** |
| Realty | NIFTYREALTY | 0.932 | 0.964 | 24.96% | 13.72% | **+11.24pp** |
| Financial Services | NIFTYFINSRV | 0.796 | 0.836 | 15.49% | 11.42% | +4.07pp |
| Oil, Gas & Consumable Fuels | NIFTYENERGY | 0.808 | 0.842 | 18.82% | 13.56% | +5.26pp |

For scale: r/154's gold reconstruction was accepted at **+0.5pp** of annual drift. These baskets are
20–27× worse. The shape tracks (correlation 0.74–0.96); the level does not. The panel is reported
as a diagnostic and **is not used for any headline claim**. The industries with the smallest wedge
(Financial Services, Oil & Gas) are the ones whose membership is most stable, which is exactly what
a survivorship explanation predicts.

## 3. G1 — does sector leadership exist at all?

388 information-coefficient cells: 28 signal specs (absolute momentum at six lookbacks, 12-1
momentum, risk-adjusted momentum, trend-persistence vote, distance-from-high, distance-from-SMA,
**acceleration** — the "to-be trending" candidate, cross-sectional reversal, vol-scaled momentum,
low-vol control, plus constituent **breadth** and **breadth change** on the synthetic panel) ×
forward horizons {1 month, 3 months} × asset sets × windows {full, first half, second half}.

**On the nine real sector indices, nothing is significant.** Best across all 168 SECT9 tests is
|t| = 2.33 (vol-scaled momentum, 3-month horizon, first half only). At the 1-month horizon on the
full window the maximum is **t = 1.46** (63-day momentum), with a top-minus-bottom tercile spread of
8.2% a year that the sample cannot distinguish from noise. A maximum |t| of 2.33 across 168
correlated tests is what pure noise looks like.

Ranking by relative strength versus NIFTY 500 is **rank-identical** to ranking by absolute return —
the benchmark term is common to every sector — so RS entered the design as a *gate*, not as a
separate ranking axis. This is stated because Arun named RS specifically.

**"To-be trending" found nothing.** Acceleration (short-horizon momentum minus its long-horizon
share) scores t = 1.37 at 1 month and 1.79 at 3 months on the real indices — the best of the
early-detection family and still not significant. Breadth change (the 21-day change in the share of
an industry's constituents above their own 50-day average) scores t = 0.36. Nothing anticipates
leadership; the momentum-family signals only confirm it, and even that confirmation is weak.

### The falsification tests on the synthetic panel

| Test | Result | Reading |
|---|---|---|
| **A. Same 8 sectors, same window, real index vs synthetic basket** | 63-day momentum: real t = 1.44 (spread 6.7%/yr) vs synthetic t = 2.74 (15.5%/yr). 126-day: 0.92 / 2.6% vs 2.56 / 14.3%. Risk-adjusted-63: 0.84 / 4.7% vs 2.89 / 12.5% | Same sectors, same dates, same signal — only the construction differs, and the construction supplies 2–6× the spread |
| **B. Each basket's own full-sample drift removed** (deliberately look-ahead; a diagnostic) | t falls 5.22 → 4.31 (126-day), 5.59 → 4.60 (risk-adjusted-126) | Constant per-basket drift is **not** the main channel |
| **C. Industry labels randomly shuffled, 50 draws** | Shuffled t: mean 1.0–1.9, 95th percentile **3.0–3.5**, max 4.25. Real-label t: 3.6–5.6 | Random groupings of the same stocks reproduce most of the "sector" signal. What survives above the shuffle distribution is small |

Conclusion: the synthetic panel's momentum is largely **stock-level cross-sectional momentum among
survivors**, aggregated into equal-weight baskets. It is not a tradeable sector effect.

## 4. G2 — branch A: the rotation sweep

1,440 configurations × 4 rebalance-day offsets on the nine real sector indices, after 25 bps per
side, after Indian FY-netted capital-gains tax, idle cash at 5% p.a. Offset-ensemble medians.

**Zero of 1,440 clear the pre-registered bar** (median CAGR ≥ 20%, worst offset ≥ 18%, Calmar ≥ 1.0).

| Book | CAGR (median) | worst offset | MaxDD | Calmar | Notes |
|---|---|---|---|---|---|
| Best by CAGR — 200-SMA distance, top 5 of 9, fortnightly, ungated | **16.3%** | 15.8% | −36.9% | 0.44 | Holds 5 of 9 sectors, i.e. barely a selection at all |
| Best by Calmar — 63-day momentum, top 3, monthly, **NIFTY500 > 200-SMA cash gate** | 13.5% | 9.6% | −21.5% | 0.63 | 74% average exposure; the gate does the work, not the sector choice |
| **Equal-weight all 9 sectors (the r/63 null)** | 14.0% | 13.8% | −43.3% | 0.32 | |
| r/147 SECROT cell reproduced (top-2, 126-day momentum, monthly) | 8.9% | 8.3% | −56.0% | 0.16 | Worst book in the study |
| NIFTY 500 buy-and-hold | 13.2% | — | −38.3% | 0.35 | |
| **Midcap 150 buy-and-hold** | **18.0%** | — | −44.2% | **0.41** | **Beats every rotation cell on CAGR** |
| Smallcap 250 buy-and-hold | 15.8% | — | −60.8% | 0.26 | |
| Cash at 5% | 4.9% | — | 0.0% | — | |

**Random-sector null, 500 draws per sector count, same monthly clock, same costs and taxes:**

| N held | Random median CAGR | Random 95th pct | Best momentum config | Percentile of the best |
|---|---|---|---|---|
| 1 | 4.04% | 13.07% | 12.79% | 94.4th |
| 2 | 6.01% | 11.11% | 12.51% | 98.0th |
| 3 | 7.13% | 11.69% | 14.08% | 99.4th |
| 4 | 8.15% | 11.11% | 14.84% | 100.0th |
| 5 | 9.01% | 11.40% | 14.97% | 100.0th |

This is the study's most precise finding. **Momentum ranks sectors better than chance — and the
ranking is not worth what the concentration costs.** Rotating at random destroys 5–10pp against
equal-weight; rotating on momentum recovers most of it and finishes level with, or behind, simply
holding everything.

**Cost and tax sensitivity** (offset medians):

| Book | 25 bps, taxed | 40 bps, taxed | 60 bps, taxed | 25 bps, untaxed |
|---|---|---|---|---|
| Best CAGR | 16.3% / −37.0% / 0.44 | 15.4% / 0.40 | 14.2% / 0.36 | 19.0% / 0.51 |
| Best Calmar | 13.3% / −24.0% / 0.56 | 11.9% / 0.47 | 10.1% / 0.37 | 16.5% / 0.88 |
| r/147 SECROT | 9.2% / −56.1% / 0.16 | 7.9% / 0.14 | 6.3% / 0.11 | 11.2% / 0.20 |

Tax costs 2.7–3.2pp of CAGR: these are short-holding-period books, taxed at 20% throughout. Nothing
in the family is close enough to the bar for a cost assumption to rescue it.

## 5. G3 — branch B: sector as a universe filter, and the decomposition that kills it

768 sector-gated stock books (2 sector-ranking sources × 4 sector counts × 3 slot counts × 4
within-sector stock signals × 2 clocks × 4 offsets) on the 500-name universe with a ₹5 crore
20-day-median traded-value floor, 2016–2026.

The best book — top 5 industries by 126-day momentum, then the 10 best stocks inside them by
risk-adjusted 126-day momentum, quarterly — returns **32.5% CAGR after tax at −38.3% drawdown,
Calmar 0.85**. That clears the 20% bar. Then the controls:

| Control (paired on the same offsets) | Result | What it proves |
|---|---|---|
| **CTRL_NOSECT** — identical stock rule, **full universe, no sector filter** | 32.9% / −39.3% / 0.84. The sector-filtered book wins on CAGR in **5 of 16** paired comparisons | **The sector layer adds nothing.** Same return, same drawdown, same Calmar |
| **CTRL_RNDSTK** — random stocks inside the same top-K industries, 100 draws | Median 18–22% CAGR; the book beats **91–100%** of draws | The **stock** ranking is doing all the work |
| **CTRL_RNDSEC** — same stock rule inside K **random** industries, 100 draws | Median 19–24% CAGR; the book beats **92–100%** of draws | Picking the right sectors beats picking wrong ones — but both lose to not restricting at all |

Read together: restricting the universe to random sectors costs roughly 8pp of CAGR; restricting it
to momentum-ranked sectors recovers that 8pp and no more. The 32% is a **stock-momentum** book on
the Nifty 500 — the same factor True North and Open Alpha already harvest — with a sector filter
bolted on that neither helps nor hurts.

And it is not a good book on its own terms. It has no stop, no trail and no market gate, and it
shows: −38% drawdown against Open Alpha's −25% and True North's −24% on the same window; **−28.0%
in the 2022H1 grind** against the deployed blend's −12.9%; Calmar 0.85 against Open Alpha's 1.77.
What is missing is exactly the risk machinery the deployed books already carry.

The **real8** sector-ranking source (ranking on the actual NSE sector indices instead of the
synthetic baskets) is materially worse — median 14.4–17.8% CAGR, Calmar 0.31–0.35, against
20.8–23.0% and 0.48–0.56 for the synthetic source. When the sector signal is taken from real,
tradeable prices, the whole branch degrades. That is the §3 finding showing up again downstream.

## 6. The house-format year-by-year table

Each cell is the calendar-year return with the intra-year maximum drawdown beneath it in
parentheses, **measured from the running peak of the full curve**, never from the year's first bar
(the r/154 convention). After tax, net of 25 bps per side, offset-ensemble medians. Common window
**31-Mar-2016 → 28-Aug-2026**. Benchmarks are excluded from the best-of picks.

| Year | Sector rot. (best Calmar) | Sector rot. (best CAGR) | r/147 SECROT | Equal-wt 9 sectors | Sector-gated stocks | Same stocks, NO filter | True North (LIVE) | Open Alpha (LIVE) | Deployed TN40/OA40/IPO20 | NIFTY 500 | Midcap 150 | BEST CAGR | LEAST DD | BEST OVERALL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016 | +15.8 (−9.1) | +12.1 (−12.4) | +21.4 (−13.1) | +11.5 (−11.6) | +11.5 (−18.5) | +20.7 (−21.1) | +32.2 (−6.8) | +27.7 (−8.0) | **+34.0 (−6.3)** | +8.2 (−12.0) | +13.2 (−14.5) | Deployed | Deployed | Deployed |
| 2017 | +30.8 (−6.4) | +38.0 (−9.4) | +23.9 (−10.4) | +34.5 (−7.7) | +58.3 (−13.8) | +99.1 (−14.7) | +39.2 (−8.7) | **+104.4 (−18.1)** | +68.2 (−8.0) | +35.9 (−8.4) | +54.3 (−9.9) | Open Alpha | Sector rot. (Calmar) | Open Alpha |
| 2018 | −10.1 (−18.5) | **−4.6 (−14.9)** | −7.4 (−19.0) | −7.2 (−17.2) | −10.9 (−24.6) | −7.1 (−24.6) | −5.6 (−23.1) | −18.5 (−19.3) | −9.3 (−17.4) | −3.4 (−15.8) | −13.3 (−24.0) | Sector rot. (CAGR) | Sector rot. (CAGR) | Sector rot. (CAGR) |
| 2019 | +0.4 (−19.1) | +6.2 (−13.3) | −5.4 (−26.1) | +1.9 (−20.5) | +13.7 (−22.5) | **+36.5 (−22.8)** | −0.7 (−21.2) | +4.7 (−24.5) | +4.5 (−16.4) | +7.7 (−12.7) | −0.3 (−25.6) | No-filter stocks | Sector rot. (CAGR) | No-filter stocks |
| 2020 | +26.7 (−20.9) | +25.6 (−36.9) | −9.2 (−56.0) | +14.7 (−43.3) | +54.8 (−30.3) | +45.9 (−32.1) | +57.4 (−23.5) | **+117.4 (−14.9)** | +87.9 (−11.3) | +16.7 (−38.3) | +24.4 (−44.2) | Open Alpha | Deployed | Open Alpha |
| 2021 | +32.8 (−14.8) | +29.2 (−14.5) | +23.5 (−28.2) | +35.5 (−11.6) | **+230.6 (−9.6)** | +114.0 (−15.7) | +59.9 (−13.6) | +151.3 (−9.5) | +88.6 (−6.4) | +30.2 (−10.0) | +46.8 (−10.5) | Sector-gated stocks | Deployed | Sector-gated stocks |
| 2022 | +3.5 (−21.5) | +11.4 (−21.9) | +3.3 (−29.6) | +8.4 (−19.3) | −0.7 (−38.3) | −1.0 (−39.3) | **+12.0 (−17.8)** | +4.4 (−20.3) | +10.7 (−12.9) | +3.0 (−18.5) | +3.0 (−21.6) | True North | Deployed | Deployed |
| 2023 | +25.5 (−20.9) | +36.7 (−12.8) | +18.6 (−24.7) | +32.4 (−12.3) | **+107.1 (−26.9)** | +90.1 (−26.8) | +41.0 (−11.0) | +44.6 (−19.9) | +48.9 (−13.7) | +25.8 (−11.2) | +43.7 (−10.4) | Sector-gated stocks | True North | Sector-gated stocks |
| 2024 | +23.6 (−9.3) | +18.0 (−12.3) | +27.2 (−11.7) | +16.4 (−11.2) | −7.4 (−28.7) | +14.2 (−22.0) | +24.6 (−17.3) | **+63.9 (−11.3)** | +50.8 (−10.9) | +15.2 (−10.9) | +23.8 (−11.0) | Open Alpha | Sector rot. (Calmar) | Open Alpha |
| 2025 | −3.4 (−19.2) | −1.8 (−26.8) | +6.0 (−17.4) | +5.0 (−21.7) | +10.1 (−33.2) | −8.1 (−31.6) | +5.1 (−18.0) | **+11.4 (−25.1)** | +7.5 (−14.8) | +6.7 (−18.8) | +5.4 (−21.1) | Open Alpha | Deployed | Deployed |
| 2026 (to Aug) | +4.4 (−9.3) | +6.0 (−19.9) | +0.1 (−15.5) | +0.8 (−16.0) | −2.4 (−20.7) | +2.5 (−24.1) | +5.9 (−10.9) | **+37.1 (−21.7)** | +30.4 (−9.0) | −1.4 (−16.2) | +5.5 (−14.0) | Open Alpha | Deployed | Deployed |
| **FULL PERIOD** CAGR / MaxDD / Calmar | 13.5% / −21.5% / 0.63 | 16.2% / −36.9% / 0.44 | 8.9% / −56.0% / 0.16 | 14.0% / −43.3% / 0.32 | 32.5% / −38.3% / 0.85 | 32.9% / −39.3% / 0.84 | 24.1% / −23.5% / 1.03 | 44.4% / −25.1% / 1.77 | **36.9% / −17.4% / 2.12** | 13.2% / −38.3% / 0.35 | 18.0% / −44.2% / 0.41 | — | — | — |

**Read the last row with the window in mind.** Open Alpha at 44.4% and True North at 24.1% are
their **2016–2026** figures, not their long-run numbers (34.90% / 19.91% over their full histories).
2016–2026 flattered every equity book, sector rotation included. The comparison is fair because
every column shares the window; the levels are not the systems' expectations.

Sector rotation's only best-of win in eleven years is **2018**, where the best-CAGR variant lost
least. That is one grind year, and equal-weight and NIFTY 500 both did about as well.

## 7. Portfolio fit — and the complement question

**Correlation to the deployed sleeves** (median paths, daily / monthly, 2016–2026):

| Candidate | vs True North | vs Open Alpha | vs IPO base |
|---|---|---|---|
| Sector rotation (best CAGR) | 0.435 / 0.330 | 0.462 / 0.550 | 0.288 / 0.306 |
| Sector rotation (best Calmar) | 0.495 / 0.382 | 0.428 / 0.471 | 0.284 / 0.299 |
| Equal-weight all 9 sectors | 0.372 / 0.326 | 0.436 / 0.559 | 0.252 / 0.295 |
| Sector-gated stock book | 0.413 / 0.443 | 0.539 / 0.619 | 0.329 / 0.235 |
| r/147 SECROT | 0.414 / 0.298 | 0.433 / 0.519 | 0.252 / 0.247 |

Every candidate breaches the pre-registered **correlation < 0.40 to both legs**. For reference,
Open Alpha to True North is 0.421 (r/154) — these candidates are as correlated to the incumbents as
the incumbents are to each other. They are the same factor: long Indian equity beta.

**Blend value**, TN 40 / OA 40 / IPO 20 baseline, 12 paired paths (TN offset *i*, OA seed *i*, IPO
seed *i*, candidate offset *i* mod 4), **every block computed on the candidate's own window** — the
baseline and the cash null are re-run on the identical index, because mixing windows is the error
r/152 was caught on.

| Candidate | Baseline (same window) | at 10% weight | Cash null at 10% | Verdict |
|---|---|---|---|---|
| Sector rotation (best CAGR) | 35.45% / −18.45% / **1.99** | 33.67% / −17.60% / 1.99 | 32.23% / −16.33% / **2.04** | loses CAGR, no Calmar gain, **cash wins** |
| Sector rotation (best Calmar) | 35.54% / −18.45% / 1.99 | 33.15% / −18.28% / 1.87 | 32.30% / −16.33% / **2.05** | worse on every axis |
| Equal-weight 9 sectors | 35.54% / −18.45% / 1.99 | 33.48% / −17.78% / 1.92 | 32.30% / −16.33% / **2.05** | **cash wins** |
| Sector-gated stock book | 36.15% / −18.45% / 2.03 | 35.99% / −18.01% / 2.04 | 32.84% / −16.33% / **2.08** | +0.01 Calmar; **cash wins** |
| Same stocks, no sector filter | 36.15% / −18.45% / 2.03 | 36.09% / −17.55% / 2.07 | 32.84% / −16.33% / **2.08** | +0.04 Calmar; **cash still wins** |
| r/147 SECROT | 35.54% / −18.45% / 1.99 | 32.91% / −18.14% / 1.85 | 32.30% / −16.33% / **2.05** | worse on every axis |

The pre-registered complement bar was **+0.10 Calmar or −2pp drawdown at ≥ equal CAGR, beating the
cash null, correlation < 0.40**. The best candidate manages **+0.04 Calmar at −0.06pp of CAGR**,
loses to plain cash at the same weight, and fails the correlation ceiling. Nothing passes. Sweeping
5% to 30% changes nothing: the Calmar gain never exceeds +0.05 and CAGR falls monotonically.

## 8. Stress windows (offset medians; drawdown from the full curve's running peak)

| Book | 2018 grind | 2020 crash | 2022H1 grind | H1 total return | H2 total return |
|---|---|---|---|---|---|
| Sector rotation (best CAGR) | −4.7% (−15.3 DD) | −15.6% (−37.0) | −12.3% (−21.9) | +123.5% | +123.8% |
| Sector rotation (best Calmar) | −10.2% (−20.9) | −7.6% (−21.6) | −11.4% (−22.7) | +114.8% | +73.3% |
| Equal-weight 9 sectors | −6.7% (−17.2) | −22.8% (−43.3) | −10.7% (−19.3) | +97.8% | +105.9% |
| Sector-gated stock book | −10.3% (−24.6) | −6.2% (−30.7) | **−28.0% (−39.4)** | +368.4% | +308.6% |
| Same stocks, no sector filter | −7.7% (−25.6) | −12.5% (−32.1) | **−27.3% (−40.3)** | +627.1% | +144.9% |
| r/147 SECROT | −7.3% (−19.2) | −34.8% (−56.1) | −18.1% (−32.5) | +46.4% | +64.3% |

Both halves of the window agree in every case, so nothing here is a single-regime artefact — the
family is consistently mediocre rather than conditionally good. The best-Calmar rotation book's
gentle 2020 (−7.6%) is the NIFTY500 200-SMA cash gate firing, not a sector call; the same gate is
already inside True North.

## 9. Multiple testing, and what would change this verdict

**~20,300 evaluated cells**: 388 information-coefficient cells, ~864 falsification cells (same-8
head-to-head, drift-stripped, 50 shuffled-label draws), 11,520 rotation runs plus 2,536 null runs,
768 branch-B books plus 3,204 paired control runs, ~88 finalist and cost-ladder re-runs, and ~936
blend cells. With that much searching, a single winner would need heavy discounting. **There is no
winner to discount** — the sweep's ceiling is below the passive benchmarks, which is the cleanest
form a negative result can take.

The pre-registered falsification condition — *"if the best surviving configuration's advantage over
the equal-weight null is smaller than the spread across rebalance-day offsets, the finding is
noise"* — is met with room: the best rotation book beats equal-weight by 2.3pp of CAGR while
individual offsets of the same configuration span 15.8% to 17.2%.

**What would change this:**

1. **Point-in-time sector index constituents.** Every synthetic-panel result here is
   survivorship-contaminated, and the real indices are only cap-weighted top-of-sector. A genuine
   point-in-time constituent history would let the branch-B question be re-asked honestly.
2. **A longer real sector history.** NSE publishes these indices back to 2005; the database starts
   2015. Back-filling would add 2008 and roughly double the sample. That is a data-acquisition task,
   not a modelling one — and it is the single highest-value follow-up if this line is ever revisited.
3. **Non-price sector inputs** — earnings revisions, sector flows, capacity/utilisation data. None
   are in our data and none are cheaply obtainable; that is why they were not tested.

## 10. Guarding the seven deadly sins

| Sin | How it was controlled |
|---|---|
| **Look-ahead** | Every signal uses data up to the close of the rebalance day and trades at the next close. Rolling statistics computed per series then re-aligned, never on a union-index frame. The one deliberate look-ahead (drift-stripping, §3 test B) is labelled a diagnostic |
| **Survivorship** | Measured, not assumed: the +4 to +14pp drift wedge (§2) is the measurement, and the synthetic panel is barred from headline claims because of it. The real sector indices are free of it; the branch-B stock universe is not, and is flagged |
| **Overfitting / data snooping** | ~20,300 cells disclosed; ranking metric and adoption bar pre-registered in the STATUS document before the first run; offset ensembles reported as median with worst case; plateau, not peak |
| **Cost neglect** | 25 bps per side throughout, ladder to 40 and 60, plus gross-of-tax rows. Turnover reported |
| **Regime dependence** | Two halves plus 2018, 2020 and 2022H1 windows, all agreeing |
| **Correlation / single factor** | The whole §7 finding: these books are 0.41–0.54 correlated to the incumbents because they are the same equity beta |
| **Capacity / tradeability** | See below — the branch-A books are not straightforwardly tradeable at all |

**Tradeability, stated plainly.** Nine of the nine sector indices are **not directly investable**.
Liquid futures exist only on BANKNIFTY (a subset of one of them, and excluded here); sector ETFs
exist for a few (BANKBEES, PSUBNKBEES) and are thin, and several sectors have no tradeable wrapper
at all. A branch-A book would in practice be implemented as constituent baskets — which is branch B,
which the decomposition already killed. Even a passing branch-A result would have needed a separate
implementation study before it could be believed.

## 11. What was NOT tested, and why

- **Intraday or weekly sector timing** — the intraday line is closed (r/109, r/110: 58 constructions,
  none clears the ~10 bps cost floor).
- **Long-short sector pairs** — no shorting infrastructure, and index trend long-short was killed in
  r/147 (whipsawed by the 2020 V-recovery, −35.6% in the crash window while short).
- **Options overlays on sector indices** — no liquid sector options in India beyond BANKNIFTY, and
  the structure-on-a-weak-signal family has five independent kills (r/129, r/150).
- **Macro or fundamental sector inputs** — not in our data (§9).
- **Sector rotation as a gate on the existing books** (e.g. only take Open Alpha signals in leading
  sectors) — a legitimate and cheap follow-up, but the branch-B decomposition already shows the
  sector filter is return-neutral on a momentum stock book, which is the strongest prior against it.

## 12. Nothing was deployed

No live or paper engine, crontab, sizing, gate or spec was touched. Research only, on the VPS, with
read-only access to `market_data.db`. No reconstructed series was written to any database.
