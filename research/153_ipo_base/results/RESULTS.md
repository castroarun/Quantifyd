# research/153 — IPO Base Breakout (bananapatterns.com "IPO Base" screen)

## VERDICT: **STRATEGY (candidate)** — as a *third sleeve*, not as a standalone book

Date: 2026-09-05. Host: VPS. Engine: `scripts/ipo_replay.py`, extending research/142's
trade-exactly-decoded bananapatterns engine (`bluesky_replay.py`).
Cells tested and disclosed: **256 (G1a signal geometry) + 384 (G1b exits × book) + ~40
control/mechanic runs = 680**, each a 10- or 30-seed ensemble on two windows.

> **Note on scope:** Arun did not supply the IPO-Base panel's dials or its published headline
> numbers, so **no replication gate was run**. Everything below is *our* encoding of the idea,
> swept. The engine is built so a replication gate can be bolted on when the dials arrive
> (`build_trigger()` + `simulate_ipo()` take every dial the site exposes).

---

## 1. The adopted spec

**IPO-Base MID** — `results/ipo_adopted_spec.json`

| Dial | Value |
|---|---|
| Universe | all NSE dailies with a **vetted listing date** (§2), ETFs excluded |
| Recency of listing | listed **≤ 6 months** ago, and ≥ 25 trading bars of history |
| Base | last **25 trading days**; **pivot = highest close** in that window; **depth ≤ 30%** (pivot to lowest low) |
| Not already extended | `close[t−1] < pivot` |
| Liquidity | 20-day median traded value at t−1 **≥ ₹5 cr** |
| Relative strength | **OFF** — see §3, the 252-day RS score does not exist for these names |
| Trigger | `close[t] > pivot` |
| Fill | **buy-stop AT the pivot**, filled `max(pivot, open[t])` |
| Hard stop | close ≤ buy × 0.92 (**−8%, on the close**) |
| Trail | close < **SMA-20** (not on the entry day) |
| Take profit | **+25%** on the close |
| Book | ₹10,00,000, **8 slots @ 18.75%** of NAV, cash-constrained, no leverage |
| Market gate | **OFF** (the NIFTYBEES-200DMA gate loses on 30/30 seeds) |
| Costs / tax / cash | 25 bps per side · 20% STCG / 12.5% LTCG with **FY (1-April) netting and loss carry-forward** · idle cash 5% p.a. |

**Sizing note (their "RISK/TRADE" dial):** the site sizes as
`position = (risk% × capital) / stop distance`, capped at 30% of capital. With a **fixed-%
stop this is algebraically identical to fixed-fraction sizing** — `size% = risk% / stop%`.
1.5% risk ÷ 8% stop = 18.75%, which is exactly the Blue-Sky sizing r/142 decoded. The two
sizing families can only diverge when the stop distance varies per trade (a structure stop).
So "risk per trade" on this site is a *sizing* dial wearing a risk-management label.

---

## 2. The data problem was the hard part — and it is solved

We have **no listing-date table**. The naive proxy (first row per symbol in
`market_data_unified`) is only **70% accurate** against 48 known NSE IPOs. Three distinct
defects, each measured and each fixed (`scripts/ipo_listing_table.py`,
`results/listing_dates.csv`):

1. **Bulk data-onboarding waves masquerade as IPOs — the defect that would have wrecked the
   study.** 451 symbols' series begin on 2005-01-03, 95 on 2015-01-01, 45 on 2026-08-17,
   41 on 2026-04-20, and 15 on 2025-05-26 — the last of which includes **ABB**, listed in the
   1990s. Untreated, ABB is a "2025 IPO" and its next breakout is an "IPO base breakout".
   *Fix:* reject any symbol whose start day is shared by **≥ 8** symbols. Real onboarding
   waves carry 12–451 symbols; genuine multi-IPO days carry 2–6.
2. **Pre-listing junk rows.** DELHIVERY carries 8 rows at ₹5–11 from 2016 (150–500 shares,
   weeks apart) before its real 2022-05-24 listing at ₹536 — a different instrument on the
   same ticker, and a **93× price jump** that would sit inside a base window. Also FUSION
   (97×), LATENTVIEW, SBICARD, STARHEALTH, MAZDOCK, COHANCE, GOYALALUM.
   *Fix:* strip leading rows to the last of {close jump >3× or <1/3 within the first 250
   rows} / {date gap > 30 days} / {volume < 5,000 shares in the first 60 rows}, then **mask
   those rows out of the price panel entirely**.
3. **A real listing has a fingerprint.** Known-IPO day-1 volume is a median **15×** the next
   20 days' median; onboardings are ~1×. *Fix:* accept if day-1 volume ratio ≥ 1.5 **or**
   day-1 high-low range ≥ 8%.

**Validation of the vetted table: recall 48/48 known NSE IPOs (100%); the listing date is
exact within ±3 days for 47/48 (98%); 0/12 known long-listed onboardings wrongly accepted.**
Result: **1,293 accepted listings, 2006–2026**, of which **786** ever become young-and-liquid
and form the tradeable universe.

**Other pre-flight checks:** the phantom-holiday-row purge is **intact** (no >90%-zero-volume
day since 2024-01-01); 42 split-scale suspects were found across the DB and all of the ones
that touch this screen turned out to be the pre-listing-junk class above, now masked.

**Survivorship — measured, and smaller than feared.** The DB *retains* dead series rather
than purging them: 9.0% of all symbols have a series ending >90 days early, and a sample of
those post-2010 ended a median **−42.9% from their peak** (median total return since listing
−13.2%). Inside the traded cohort, **41 of 334 names later go stale; they account for 5.1% of
trades and returned a mean +2.70% versus +5.52% for the rest** — a real, small drag that the
backtest *does* pay. (The 2025 cohort's 48.6% "ends early" rate is a **feed-freshness
artefact**, not delisting: those series stop in identical batches on 2026-02-17 / 05-07 /
05-15, i.e. symbols dropped from the nightly refresh list.) **The residual, unmeasurable bias
is names that IPO'd and died without ever being onboarded to Kite at all.**

---

## 3. How can the site apply RS ≥ 70 to a six-month-old listing? It cannot.

The IBD-style RS score needs **252 trading days**. Tested four policies:

| RS policy | Signals (age ≤ 12m) | Verdict |
|---|---|---|
| **strict** — RS ≥ 70 required | **0** | Mathematically impossible; only becomes non-empty in the 12–24-month band |
| **relaxed** — apply where computable, pass where not | identical to OFF below 12m | Not a filter at all for this screen |
| **short** — 3-month-return percentile ≥ 70 | 1,010 (vs 2,322) | Costs ~8pp of CAGR (19.8% → 11.9% median across matched cells), mostly by starving the book |
| **off** | 2,322 | Adopted |

**Answer to the question:** the site's RS filter is inert on a genuine IPO base — the screen
is a *price-structure* screen, and any RS-style substitute we could build made it worse.
This is now a documented finding, not an assumption.

---

## 4. G1 — the signal geometry sweep (256 cells, 10 seeds, two windows)

**207 of 256 cells** clear the gate (per-trade expectancy net of 25 bps/side positive in
**both** windows, ≥ 4 trades/yr). The surface is a **broad plateau, not a peak**:

| Axis | Median W2 after-tax CAGR across the axis |
|---|---|
| Age band | 3m 20.1% · 6m 20.0% · 12m 19.1% · 24m 22.2% |
| Base length L | 15d 20.9% · 25d 22.4% · 40d 19.6% · 60d 14.6% |
| Max depth | 20% 18.8% · 30% 20.9% · 40% 19.6% · 60% 20.1% |

Per-trade net expectancy runs **+5% to +16%** depending on how tight the screen is —
tighter screens trade less and earn more per trade, which is the expected shape.

## 5. G1b — exits × book (384 cells). Two of the site's own dials are wrong.

**383 of 384 cells** clear the expectancy gate. Monotone, plateau-shaped results:

| Dial | Result (median W2 Calmar across all cells) |
|---|---|
| **Trail** | **SMA-20 0.99** · SMA-30 0.80 · SMA-50 0.74 · **SMA-150 ("Trail 30-week") 0.49** |
| **Take profit** | **+25% ON 0.93** · OFF 0.69 — helps in *every* geometry, every trail, every slot count |
| **Hard stop** | 7% 0.78 · 8% 0.75 · 10% 0.75 — **inert**; the trail binds first |
| **Slots** | 5 → 0.74 · 8 → 0.75 · 10 → 0.75 · 16 → 0.81 (CAGR falls monotonically 26.6 → 17.4) |

**The site's "Trail 30-week" dial is the single worst exit we tested**, and "Take +25%" is
the best. If their published IPO-Base numbers were generated on the 30-week trail, they
should be materially worse than ours.

## 6. The age band is a clean dose-response — and it decides what you own

30-seed ensembles, 2006 → Sep-2026, after tax, 25 bps, 5% idle cash:

| Sleeve | Age band | CAGR (median [min..max]) | MaxDD (worst seed) | Calmar | Trades/yr | % invested | Edge vs null |
|---|---|---|---|---|---|---|---|
| **NARROW** | ≤ 3 months | 24.10 [22.97..24.87] | −13.80 (−17.68) | 1.74 | 19.4 | **19.6%** | **+2.08pp/trade** |
| **MID — adopted** | ≤ 6 months | **31.03 [28.82..33.44]** | **−20.88 (−23.23)** | **1.50** | 32.6 | 32.7% | +0.96pp/trade |
| WIDE | ≤ 24 months, 15d base | 35.99 [32.40..42.29] | −33.34 (−40.49) | 1.08 | 61.7 | 59.5% | +0.97pp/trade |

**Read this honestly:** as the age band widens the screen stops being an "IPO base" and
becomes a general young-stock breakout — and its profile converges on **Open Alpha's**
(33.8% / −27.3%). The narrow band has the strongest genuine IPO-base effect (biggest edge
over the null) but is 80% in cash. The mid band is the best compromise and is what the blend
test picked.

---

## 7. Robustness — every control, run

| Control | Result |
|---|---|
| **Cost ladder** (25/40/60 bps per side) | 31.03% → 28.77% → 25.76% CAGR. ~1.8pp of CAGR per +15 bps/side. Survives 60 bps |
| **Market gate ON** (NIFTYBEES < SMA-200) | −5.23pp CAGR, **loses on 0/30 paired seeds** → rejected (same as Open Alpha) |
| **Fill mechanic** | **THE critical dependency.** Pivot buy-stop 31.03% vs signal-day close 16.98% — **−14.08pp, close-fill loses on 0/30 seeds** |
| **Random-entry, date-matched null** | Real **+1.75pp CAGR** and **+0.96pp per trade** over buying a random young+liquid name on the same days; **real wins 29/30 paired seeds** |
| **Cohort drift null** | Equal-weight hold of every young+liquid name: 17.8% CAGR at **−82.6% DD** (gross). The cohort drifts up; the screen's contribution is the risk control |
| **Outlier dependence** | Top-10 trades = **11%** of the summed trade return; excluding each seed's 10 best trades still leaves **+4.89% per trade (+4.39% net of a 50 bps round trip)**. Not a lottery-ticket book. (Winner-capping at +50/+100% is moot — the +25% take-profit already caps the right tail) |
| **Seed dispersion** | 30 seeds, CAGR 28.82–33.44, **worst seed 28.82%**, worst-seed DD −23.23% |
| **Two windows** | W2 2006–2026: 31.03% / −20.88%. W1 2020–2025 (the site's window): **44.57% / −20.78% / Calmar 2.18**. Both pass; the recent window is far more generous |
| **Plateau** | Neighbours agree on every axis; 383/384 exit cells positive. This is not a lone peak |
| **Multiple testing** | 680 cells disclosed. Discount the best cell accordingly — but the *median* cell, not just the best, clears the gate |

### Per-year (median of 30 seeds; **the flat years are the story**)

| Year | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Return % | +52.2 | +97.3 | +4.2 | +3.1 | +33.2 | +13.3 | **+2.5** | **+5.1** | **+5.0** | +13.9 | +48.7 |

| Year | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 YTD |
|---|---|---|---|---|---|---|---|---|---|---|
| Return % | +60.0 | +0.0 | +14.3 | +88.1 | +40.7 | +15.0 | +71.3 | +77.8 | **−1.4** | +66.5 |

**2013 and 2014 returned exactly the idle-cash yield: the book took no trades at all.** The
Indian IPO pipeline shut between roughly 2012 and 2014 (8–17 accepted listings a year versus
80–182 in 2021–2025). Any operator running this must be prepared for **multi-year stretches
of doing nothing**, and for the fact that a large part of the record comes from the
2020–2026 IPO boom.

---

## 8. Capacity — comfortable to about ₹10 cr, binding by ₹50 cr

Held names' 20-day median traded value: p10 ₹6.4 cr, **median ₹17.0 cr**, p90 ₹84.0 cr.
For the sleeve held at **10% of a portfolio**, 18.75% of sleeve NAV per position:

| Portfolio | Sleeve | Position | Median % of daily traded value | p90 | p99 |
|---|---|---|---|---|---|
| ₹1 cr | ₹10 L | ₹1.88 L | 0.11% | 0.29% | 0.37% |
| ₹5 cr | ₹50 L | ₹9.38 L | 0.55% | 1.45% | 1.86% |
| ₹10 cr | ₹1 cr | ₹18.75 L | 1.10% | 2.91% | 3.73% |
| ₹50 cr | ₹5 cr | ₹93.75 L | 5.52% | **14.54%** | 18.63% |

Rule of thumb: above ~10% of a name's daily traded value a breakout-day entry moves the
price against you. **Capacity is a non-issue at Arun's current size and starts to bind
around a ₹50 cr portfolio.** (The capacity figure printed in `g3_*.log` scaled by the
*compounding* NAV and is misleading; the table above, from `scripts/ipo_blend2.py`, is the
correct one.)

---

## 9. Portfolio fit — the actual adoption test

Correlations (medians over seed pairs, 2006 → 2026):

| Pair | Daily | Monthly |
|---|---|---|
| IPO ↔ Open Alpha | **0.16** | **0.25** |
| IPO ↔ True North | **0.18** | **0.22** |
| Open Alpha ↔ True North | 0.42 | 0.54 |

Both well under the pre-registered 0.4 ceiling, and **notably lower than the correlation
between the two legs already deployed.**

### 3-sleeve blend, monthly rebalanced, after tax (medians)

| Weighting | CAGR | MaxDD | Calmar | 2018 DD | 2020 DD | 2022H1 DD |
|---|---|---|---|---|---|---|
| **TN+OA 50-50 (deployed baseline)** | **27.14** | **−16.42** | **1.65** | −11.26 | −1.98 | −9.07 |
| + IPO 10% (45/45/10) | 27.72 | −14.44 | 1.92 | −10.61 | −1.75 | −8.86 |
| **+ IPO 20% (40/40/20)** | **28.27** | **−12.79** | **2.21** | −9.82 | −2.14 | −8.75 |
| + IPO 33% | 28.91 | −13.05 | 2.21 | −9.30 | −2.67 | −8.60 |
| **cash-null at 10%** | 24.91 | −14.49 | 1.72 | −9.94 | −1.70 | −8.08 |
| **cash-null at 20%** | 22.67 | −12.64 | 1.79 | −8.65 | −1.43 | −7.07 |

**Pre-registered adoption bar: +0.10 Calmar *or* −2pp drawdown at ≥ equal CAGR, after tax,
robust across seeds/offsets, correlation < ~0.4 to both legs, and beats the cash-null.**

At 20% weight the IPO sleeve delivers **+1.13pp CAGR, −3.63pp drawdown and +0.56 Calmar**
over the deployed pair, at correlations of 0.16/0.18, and beats plain cash at the same
weight by **+5.60pp of CAGR for +0.15pp of drawdown**. **Every condition is met with room.**
(The narrow ≤3-month sleeve also improves Calmar substantially but costs a hair of CAGR
(27.08 vs 27.14 at 10%), so it fails the "≥ equal CAGR" leg. That is why **mid** is adopted
and not narrow — the bar, not the outcome, made the choice.)

### Against the incumbent third-sleeve candidate (gold, r/147), same 2015+ window

| Blend | CAGR | MaxDD | Calmar |
|---|---|---|---|
| TN+OA baseline | 29.63 | −16.10 | 1.84 |
| + GOLDBEES 10% (r/147) | 28.36 | −13.37 | 2.12 |
| + GOLDBEES 20% | 27.03 | −10.54 | 2.56 |
| + **IPO 10%** | 30.84 | −14.24 | 2.17 |
| + **IPO 20%** | **32.01** | −12.76 | 2.51 |
| **4-sleeve 40/40/10 gold/10 IPO** | 29.05 | **−11.55** | 2.52 |

**Gold buys its Calmar by lowering return; the IPO sleeve buys its Calmar while *raising*
return.** They are not substitutes — the 4-sleeve combination reaches Calmar 2.52 at
−11.55% drawdown, and is the most interesting construction in this study. It has *not* been
put through a full weight sweep and should be the subject of its own study before adoption.

---

## 10. Caveats — read before acting

1. **The entire edge lives in the entry price.** Pivot buy-stop 31.0% CAGR vs signal-day
   close 17.0% (0/30 seeds prefer the close). Live, this requires a working **buy-stop /
   GTT at the pivot** on every candidate every day. If fills degrade toward the close, half
   the edge evaporates. This is the same lesson r/142 learned (×536 vs ×14.4).
2. **No replication gate was run** — the site's dials and claimed numbers were not available.
   Two dials the site exposes ("Trail 30-week", "Breakout close") are the *worst* settings we
   tested, so their published figures cannot be assumed comparable to these.
3. **Multi-year dead zones.** 2013 and 2014 earned only the cash yield. The strategy is a
   function of the IPO pipeline, which is a policy/market-cycle variable, not a price series.
4. **Regime concentration.** 2020–2026 supplies a large share of both the trades and the
   return; the site's own window (2020–2025) shows 44.6% CAGR against 31.0% for the full
   period. Expect the forward number to be nearer the full-period figure, or below it.
5. **Survivorship** is measured and small inside the DB (5.1% of trades, −2.8pp mean return
   drag) but the residual — IPOs that died before ever being onboarded to Kite — is
   unmeasurable and biases *upward*.
6. **Listing-date proxy** is validated on a 60-name test set (48 IPOs + 12 onboardings), not
   on all 1,293 accepted listings. A systematic error in the untested remainder is possible.
7. **Multiple testing:** 680 cells. The plateau is broad and the median cell passes, but the
   headline cell's numbers should still be discounted.
8. **Point-in-time universe:** the ₹5 cr liquidity floor is applied causally (t−1), but the
   symbol *universe* is today's Kite coverage.
9. **Nothing was deployed.** No live engine, crontab, or spec was touched.

---

## 11. Deliverables

| File | Contents |
|---|---|
| `results/ipo_equity_seeds.csv` | **Daily equity, 30 seeds, adopted (MID) spec, after tax, 5% idle cash** — consumed by r/154 |
| `results/ipo_adopted_spec.json` | Machine-readable adopted spec |
| `results/ipo_equity_seeds_narrow.csv` / `_wide.csv` | The other two age bands, same format |
| `results/listing_dates.csv` | The vetted listing table (1,293 accepted) |
| `results/g1a_sweep.csv` / `g1b_sweep.csv` | 256 + 384 cells, one row each |
| `results/g3_controls_*.csv`, `g3_peryear_*.csv`, `g3_trades_*.csv` | Controls, per-year, per-trade detail |
| `results/g4_blend.csv`, `g4_yoy_returns.csv`, `g4_yoy_intradd.csv` | Blend sweep and YoY house table |
| `results/ipo_base_research153.png` | Growth of ₹100 (log) vs NIFTY 50 / Midcap 150 / Smallcap 250 + drawdown panel |

## 12. Recommended next steps

1. **Arun's adoption call** on IPO-mid at 10–20% of the book. If yes: G5 paper soak with a
   pre-registered fill criterion (**modeled vs actual fill within 0.5% of the pivot**, miss
   rate < 15%) and a dated review.
2. **Send the IPO-Base panel's dials and claimed numbers** when available — the replication
   gate is a one-command run on this engine.
3. **A dedicated 4-sleeve weight study** (TN / OA / gold / IPO). The 40/40/10/10 point
   estimate is the best risk-adjusted construction this project has produced; it deserves a
   proper sweep rather than a single cell.
