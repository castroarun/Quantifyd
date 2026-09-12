# research/163 — IPO Base re-optimised on a PLACEABLE entry

## VERDICT: **STRATEGY candidate** — but the *adopted* spec is **NO EDGE**

Two verdicts, because the question had two halves.

**1. The r/153 adopted spec, measured honestly, is NO EDGE.** On the entry the live book
actually places, IPO Base as adopted returns 14.90% after tax (not 31.0%) and — the finding
that matters — it **loses to a date-matched random-entry null**: 14.90% real vs 15.11% null,
winning only 14 of 30 paired seeds (−0.15pp median). Gated, it is worse: 8 of 30, −0.44pp.
Picking random young, liquid names on the same days does the same job. r/153 ran this null
too and got +5.8pp in favour of the signal, but it ran it on the close-fill arm; on the
placeable entry the edge is gone.

**2. Re-fitted, it becomes a STRATEGY candidate.** One parameter was wrong —
**the trail** — and fixing it restores a real edge: **+4.91pp over the null on 30 of 30
paired seeds**. With a long-MA index gate the book goes to **21.80% CAGR after tax,
−26.6% drawdown, Calmar 0.819**, against the incumbent's 14.90% / −38.6% / 0.386.

**Plain answer to the question asked: yes, IPO Base gets better on the honest entry once
re-fitted — by +6.9pp of CAGR and +12.0pp of drawdown, paired-unanimous across 30 seeds.
But it never recovers the published 31.0%, and about a quarter of the improvement is simply
the index gate sitting the book out of bear markets.**

Nothing here is deployed. `services/ipo_paper.py` was not touched.

---

## 1. What was wrong, and what was right

`research/153`'s engine decided on bar *i*'s CLOSE (`trig = setup & (C > piv)`) and filled at
bar *i*'s OPEN (`fill = max(piv, O[i])`) — the 8th deadly sin (playbook §4, §5A).
`services/ipo_paper.py` is correct: it triggers on tonight's close and carries a buy-stop at
the pivot into the next morning. Reproduced exactly, same spec / seeds / window / engine,
changing only the entry:

| entry mechanic | CAGR | worst seed | MaxDD | Calmar | invested | WA 06-15 | WB 16-26 |
|---|---|---|---|---|---|---|---|
| `ref_lookahead` — r/153's, **NOT placeable** | 31.32% | 30.16 | −21.2% | 1.476 | 32.8% | 20.98 | 41.78 |
| `close_fill` — buy at the signal close | 17.39% | 15.34 | −36.0% | 0.483 | 33.1% | 14.66 | 19.77 |
| **`nextday_pivot` — the live book's mechanic** | **14.90%** | 13.11 | −38.6% | 0.386 | 31.8% | 14.14 | 15.04 |
| `nextday_candle` — stop above the signal candle | 13.95% | 12.31 | −32.5% | 0.429 | 27.6% | 13.85 | 13.94 |
| `resting_stop` — every crossing, incl. failures | 16.08% | 13.91 | −47.6% | 0.338 | 40.9% | 12.86 | 18.05 |

All after tax (20% STCG / 12.5% LTCG, Indian FY loss netting), 25 bps/side, 5% on idle cash,
30 seeds, W2 = 2006-01-01 → 2026-09-04 (20.7 y). Confirms r/159's 31.0% → 15.0%.

**Every placeable mechanic lands in 14–17%. The look-ahead arm is the only one above 20%** —
16 points of the published figure was the certainty that the breakout would hold to the bell.

### What the re-fit found was ALREADY right
Only the exits were mis-fitted. Measured on the honest entry:

- **Base geometry was correct.** 128 cells of age × base length × depth × RS policy put the
  incumbent `age ≤ 6m / L=25 / depth ≤ 30% / RS off` at the **top of the grid**. Neighbours:
  L=15 18.47%, L=40 13.35%, age 3m 16.96%, age 12m 12.14%, depth 20% 16.61%, depth 40%
  16.77%. RS `short70` is a disaster (median 6.76% vs 11.54% across all geometries),
  confirming r/153's decision to switch RS off.
- **Book sizing was correct.** 18 cells: 8 slots @ 18.75% beats every alternative
  (5×20% 19.74%, 10×10% 15.06%, 16×6.25% 12.58%). Concentration is the lever, as everywhere
  else in this project.
- **Corporate actions are not contaminating it.** Zero trades worse than −40%, zero one-day
  close collapses on an exit bar. The −8/−10% stop fires long before a split can book a fake
  −90%.
- **Fund contamination is immaterial here**, which is a surprise: IPO Base is structurally
  the most exposed book (ETFs are always newly listed) and 146 of the 1,353 vetted listings
  ARE funds, of which r/142's ticker regex catches only 54. Rebuilt with the long-name
  exclusion list (`backtest_data/etf_exclusions.json`): 1,548 vs 1,556 signals, 14.90% vs
  15.00% CAGR. The ₹5 cr liquidity floor and the base-depth test keep them out.
  **r/153's published figures are NOT affected by the r/158 fund defect.**

---

## 2. The exit surface — 160 cells, and the trail inverted

W2 after-tax CAGR, 30-seed median, live entry, target = +25% (incumbent value):

| stop \ trail | 10 | 15 | **20** | 30 | 40* | **50** | 75 | 100 | 150 |
|---|---|---|---|---|---|---|---|---|---|
| 6% | 9.96 | 11.31 | 13.94 | 14.97 | — | 16.91 | 12.88 | 14.01 | 13.02 |
| 8% | 9.33 | 11.75 | **14.90** | 17.68 | — | 20.78 | 16.83 | 17.21 | 15.38 |
| 10% | 9.49 | 12.02 | 15.71 | 17.46 | 19.53* | **22.01** | 18.65 | 17.46 | 14.54 |
| 15% | 9.09 | 11.45 | 15.61 | 17.69 | 19.20* | **22.40** | 18.02 | 15.83 | 14.07 |
| none | 9.05 | 11.06 | 15.24 | 16.67 | — | 21.13 | 14.87 | 14.64 | 13.24 |

Bold = the incumbent cell and the best cell. *trail 40 measured in stage 8 (gated), shown for
the shape only. Full grid also covers targets of +50%, +100% and none (160 cells total).

- **Trail: the incumbent 20 is far too short.** Same direction as Open Alpha's inversion,
  but a hump rather than a monotone ramp — it peaks at 50 and rolls over by 100–150.
- **Stop: 10% is the plateau centre.** Monotone 6% < 8% < 10% ≈ 15% > none. The incumbent 8%
  is one notch tight.
- **Target interacts with the trail, and the incumbent got the worst pairing of the four.**
  At trail 20, no-target beats +25% by 2.5pp; at trail 50, +25% beats no-target by 2.3pp.

---

## 3. The null control — the piece that decides it

Date-matched random entry: the same days, the same number of entries per day, drawn at random
from the same eligible young + liquid universe, **the same fill convention and the same gate
on both arms**, 30 paired seeds. (r/153 ran this only on its close-fill arm.)

| spec | real CAGR | real %/trade | null CAGR | null %/trade | paired edge | real wins |
|---|---|---|---|---|---|---|
| incumbent, ungated | 14.90% | 2.81% | 15.11% | 2.84% | **−0.15pp** | **14 / 30** |
| incumbent, next-open fill both arms | 14.12% | 2.67% | 14.53% | 2.13% | **−0.49pp** | **9 / 30** |
| incumbent + SMA-150 gate | 13.53% | 2.93% | 13.79% | 2.97% | **−0.44pp** | **8 / 30** |
| **refit trail 50 / sl 15% + SMA-150** | 22.36% | 7.08% | 17.52% | 5.50% | **+4.91pp** | **30 / 30** |
| refit trail 30 / sl 10% + SMA-150 | 17.26% | 4.52% | 15.77% | 4.08% | +1.26pp | 30 / 30 |
| cash only at 5% (cash null) | 5.04% | — | — | — | — | — |
| young+liquid cohort, equal weight, GROSS | 17.46% | — | — | — | — | DD −82.6% |

**The edge is a function of the trail, and it is a smooth curve, not a spike.** Real minus
null, in pp of CAGR, gated, 30 paired seeds at each cell:

| trail | 10 | 15 | 20 | 30 | 40 | **50** | 60 | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| stop 10% | +0.04 | −0.71 | −0.10 | +1.26 | +3.08 | **+4.78** | +2.05 | +0.43 | +0.31 |
| stop 15% | −0.29 | −0.88 | −0.06 | +1.54 | +2.80 | **+4.91** | +3.28 | +2.05 | +1.21 |
| real wins /30 (stop 15%) | 11 | 5 | 13 | 30 | 30 | 30 | 30 | 30 | 27 |
| per-trade edge, pp (stop 15%) | 0.01 | −0.08 | 0.06 | 0.47 | 0.84 | **1.57** | 0.99 | 0.76 | 0.31 |

Two readings, and the data chooses the first:
- the curve rises and falls smoothly, reproduces at two independent stop values, and is
  **unanimous (30/30) across the whole trail 30–75 band** → this is a plateau maximum;
- **at trail ≤ 20 — the incumbent's entire region — the edge is zero or negative.** The
  selection rule "a recently listed stock closed above its 25-bar base high" adds nothing
  over drawing a random young liquid name, *unless you then hold it with a slow exit.*

That is the honest characterisation of what IPO Base is: not a signal with an entry edge,
but an entry that is only worth taking if the exit lets the winners run for ~2 months.

---

## 4. The market gate — free insurance, and it is not in r/153

r/153 runs no gate and the look-ahead surface gave it no reason to. On the honest entry,
11 gate cells at the refit geometry, **paired on seed**:

| gate | CAGR | worst seed | MaxDD | Calmar | paired CAGR Δ | A wins | paired DD Δ | A shallower |
|---|---|---|---|---|---|---|---|---|
| none | 22.40% | 20.21 | −46.9% | 0.477 | — | — | — | — |
| **NIFTYBEES < SMA-150** | **22.36%** | **21.50** | **−29.3%** | **0.762** | **−0.03pp** | 14/30 | **+17.80pp** | **30/30** |
| NIFTYBEES < SMA-200 | 21.18% | 19.74 | −31.0% | 0.682 | −1.38pp | 6/30 | +15.66pp | 30/30 |
| NIFTYBEES 126d mom < 0 | 20.02% | 18.48 | −33.5% | 0.600 | — | — | — | — |
| NIFTYBEES < SMA-100 | 19.48% | 17.74 | −42.0% | 0.464 | −3.15pp | 0/30 | +5.11pp | 27/30 |
| index drawdown > 5% | 14.78% | 13.36 | −34.6% | 0.434 | — | — | — | — |
| index 63d mom < 0 | 14.77% | 12.73 | −43.7% | 0.338 | — | — | — | — |

SMA-150 is **insurance with no premium**: a coin flip on return (14/30), 17.8 points of
drawdown removed on **every** path. SMA-200 agrees; SMA-100 fails; so the region is bounded
at both ends, which is what a plateau looks like. This also beats the alternative way of
buying drawdown — a +100% target instead of the gate loses 3.83pp of CAGR on 30/30 AND ends
up deeper.

---

## 5. The finalists — full adoption arithmetic

All after tax, 25 bps/side, 5% idle cash, 30 seeds, live `nextday_pivot` entry,
W2 2006-01-01 → 2026-09-04, ₹10,00,000 book.

| | incumbent r/153 | **A — recommended** | B — best CAGR |
|---|---|---|---|
| trail / stop / target / gate | SMA-20 / 8% / +25% / none | **SMA-50 / 10% / +25% / NIFTYBEES<SMA-150** | SMA-50 / 15% / +25% / SMA-150 |
| CAGR (median) | 14.90% | **21.80%** | 22.36% |
| CAGR band [worst..best seed] | [13.11 .. 16.73] | **[20.83 .. 23.19]** | [21.50 .. 24.31] |
| **CAGR with cash yield set to ZERO** | **11.08%** | **17.97%** | 18.36% |
| cash sweep's share of the headline | **25.6%** | **17.6%** | 17.9% |
| **invested fraction (mean of days)** | **31.8%** | **36.3%** | 37.0% |
| MaxDD (median) | −38.6% | **−26.6%** | −29.3% |
| **MaxDD (WORST seed)** | **−46.4%** | **−32.9%** | **−40.1%** |
| Calmar (median) | 0.386 | **0.819** | 0.762 |
| Calmar on the worst seed | 0.282 | **0.634** | 0.536 |
| WA 2006–2015 CAGR / DD | 14.14% / −12.8% | 16.12% / −15.9% | 15.89% / −16.2% |
| WB 2016–2026 CAGR / DD | 15.04% / −38.0% | **27.20% / −26.6%** | 29.23% / −28.1% |
| net expectancy per trade (after 50 bps) | +2.31% | **+6.24%** | +6.58% |
| WA / WB net expectancy | +4.90% / +1.66% | +8.54% / +5.54% | +8.41% / +6.04% |
| win rate | 39.9% | **49.0%** | 49.8% |
| avg win / avg loss | +15.2% / −5.5% | +21.5% / −7.4% | +21.9% / −7.6% |
| max losing streak | 18 | **11** | 14 |
| trades / yr · median hold | 33.2 · 18 d | 18.9 · 37 d | 18.5 · 38 d |
| **cost ladder 25 / 40 / 60 bps** | 14.90 / 12.94 / 10.47 | **21.80 / 20.81 / 19.02** | 22.36 / 21.04 / 19.67 |
| top-10 trades' share of summed return | 20% | **15%** | 14% |
| mean/trade excluding each seed's 10 best | 2.30% | **5.89%** | 6.24% |
| mean/trade with winners capped at +50% | 2.81% | 6.74% | 7.08% |
| weekly corr: OA Base Age / TN / index | 0.245 / 0.211 / 0.165 | 0.282 / 0.256 / 0.182 | 0.276 / 0.259 / 0.185 |
| distinct names traded over 20.7 y | 331 | 254 | 253 |

**A is the recommendation over B** despite 0.56pp less CAGR: its worst-seed drawdown is
−32.9% against B's −40.1%, its seed band is the tightest in the study (2.4pp wide), its
losing streak is 11 not 14, and it is the most cost-tolerant cell measured (−2.8pp from 25 to
60 bps, because 18.9 trades a year held 37 days is a low-turnover book).

**Not outlier-dependent.** Deleting each seed's ten best trades of twenty years takes the
per-trade mean from 6.24% to 5.89%. Capping winners at +50% changes nothing at all, because
the +25% target already caps them.

### Per-year, after tax, median seed — return with intra-year drawdown beneath
Drawdowns measured from the running peak of the **full** curve, never from the window's own
first bar (the r/154 convention error).

| year | incumbent | **A (recommended)** | B |
|---|---|---|---|
| 2006 | +54.3 (−10.9) | +71.0 (−10.1) | +71.0 (−10.1) |
| 2007 | +59.8 (−11.7) | +51.5 (−15.9) | +51.1 (−16.1) |
| 2008 | **+0.4 (−12.8)** | −10.3 (−15.8) | −10.3 (−15.8) |
| 2009 | +1.4 (−6.0) | +3.8 (−12.7) | +3.8 (−12.7) |
| 2010 | +12.3 (−10.2) | +21.0 (−9.8) | +21.0 (−9.8) |
| 2011 | **+13.5 (−5.5)** | −2.6 (−10.3) | −3.3 (−10.9) |
| 2012 | +0.6 (−4.4) | +11.7 (−10.1) | +11.7 (−10.8) |
| 2013 | +5.1 (−2.8) | +4.6 (−0.4) | +4.8 (−0.3) |
| 2014 | +5.0 (0.0) | +5.0 (0.0) | +5.0 (0.0) |
| 2015 | +5.4 (−9.5) | +27.5 (−9.6) | +26.1 (−10.7) |
| 2016 | +53.1 (−12.7) | +75.6 (−10.6) | +75.9 (−10.6) |
| 2017 | +32.1 (−9.9) | +72.3 (−10.2) | +66.6 (−10.2) |
| 2018 | −6.2 (−20.7) | −9.2 (−19.3) | −5.4 (−19.8) |
| 2019 | +9.3 (−13.5) | +5.2 (−15.8) | +3.0 (−14.2) |
| 2020 | +68.9 (−9.8) | +74.6 (−13.2) | +74.3 (−13.2) |
| 2021 | +2.0 (−19.2) | +52.4 (−14.5) | +51.6 (−14.7) |
| 2022 | −8.8 (−32.3) | +0.8 (−21.2) | +1.9 (−20.6) |
| 2023 | +34.7 (−36.8) | +46.4 (−11.0) | +68.2 (−9.7) |
| 2024 | +1.9 (−19.3) | +13.5 (−22.6) | +16.4 (−14.9) |
| 2025 | −27.5 (−35.7) | **+1.8 (−17.9)** | −5.8 (−23.6) |
| 2026 YTD | +48.5 (−41.8) | +1.3 (−24.0) | +5.2 (−27.9) |
| **full period** | **14.90% / −38.6% / 0.386** | **21.80% / −26.6% / 0.819** | **22.36% / −29.3% / 0.762** |

### Per window
| window | incumbent | A | B |
|---|---|---|---|
| 2008 crash | **+0.4% (dd −12.8%)** | −10.3% (dd −15.8%) | −10.3% (dd −15.8%) |
| 2020 crash (H1) | +28.6% (dd −9.8%) | +27.3% (dd −13.2%) | +27.1% (dd −13.2%) |
| 2018 grind | −6.2% (dd −20.7%) | −9.2% (dd −19.3%) | −5.4% (dd −19.8%) |
| 2022 H1 grind | −14.6% (dd −32.3%) | −14.5% (dd −20.4%) | −12.3% (dd −19.7%) |
| 2025 drawdown | −27.5% (dd −35.7%) | **+1.8% (dd −17.9%)** | −5.8% (dd −23.6%) |

**2008 is the honest black mark: the refit is 10.7pp WORSE than the incumbent there.** The
fast SMA-20 trail that costs 7pp a year in normal times is exactly what sidestepped 2008. The
gate recovers part of it (ungated the refit loses 18.3% in 2008; gated, 10.3%) but not all.
If the pair needs a 2008 cushion, this is not it.

---

## 6. Caveats — what would make this wrong

- **Multiple testing.** ~350 cells were scored (5 mechanics × 2 universes, 160 exits, 128
  geometries, 18 book, 11 gates, 18 null-axis, 8 nulls, 10 adoption). The 30-seed band is
  ±1.2–2.4pp, so the 22.40% peak should be discounted toward its neighbourhood: the trail
  30–75 × stop 8–15% region averages ~19%, not 22%. **A's 21.80% should be read as
  "19–22%", not as a point estimate.**
- **The gate and the trail were both chosen after seeing this data.** The only out-of-sample
  evidence is the two-window split, which both finalists pass strongly (WA +8.5%/trade,
  WB +5.5%/trade for A). There is no held-out period.
- **Capacity is the binding constraint and it is tight.** At ₹10 L the p90 position is 1.56%
  of the name's 20-day median traded value — fine. Scaled to ₹1 cr that is ~90% of a day's
  volume, and to ₹10 cr it is absurd. **This book does not scale past roughly ₹20–25 L.**
  It is a small, permanently small sleeve.
- **The worst-seed drawdown is −32.9%, not −26.6%.** A reader who sees only the median is
  being misled about the unlucky path. Worst-seed Calmar is 0.634.
- **Survivorship and the rename defect.** The universe keeps delisted series (they get
  traded and stopped out), but names never onboarded to Kite are unmeasurable. Separately,
  handover §5.2's rename defect (`LOTUSDEV` → `LOTUSDEV-BE`) hits **this book hardest** —
  it trades exactly the young, thin names NSE moves to trade-for-trade. Ten of eleven stale
  young names are missing from the instrument dump. **Still not fixed**; it silently freezes
  prices and will corrupt live signals, not the backtest.
- **The live paper book over-fills.** Read-only observation in `services/ipo_paper.py`'s
  paper branch: it books `fill = max(pivot, open)` **without checking that the day's high
  reached the pivot**. A buy-stop that never triggered is recorded as filled. Measured
  impact is small (1.5% of signals never reach the level) but it makes the paper book's fills
  slightly optimistic versus this study. Reported, not touched.
- **Not tested** — see §8.

---

## 7. If this were adopted (it is NOT being adopted here)

Spec A, stated in full, for whoever does take it forward:

```
universe   NSE equities with a vetted listing date (research/153/results/listing_dates.csv),
           funds excluded by long name (backtest_data/etf_exclusions.json),
           all rows before the listing date masked
age        listed <= 6 months ago AND >= 25 bars of history
liquidity  20-day median traded value at t-1 >= Rs 5 cr
base       last 25 bars; pivot = highest CLOSE, shifted 1; depth (pivot -> lowest low)
           <= 30%; close[t-1] < pivot (not already extended)
RS         OFF
gate       NEW: no new entries while NIFTYBEES < its 150-day SMA (existing holds continue)
trigger    close[t] > pivot
fill       next day, buy-stop AT the pivot, filled max(pivot, open) -- UNCHANGED, already
           correct in services/ipo_paper.py
exits      stop close <= fill x 0.90   (CHANGED from 0.92)
           target close >= fill x 1.25 (unchanged)
           trail close < SMA-50        (CHANGED from SMA-20)
book       8 slots @ 18.75% of equity, 25 bps/side (unchanged)
```

Three dials change: **trail 20 → 50, stop 8% → 10%, and a new NIFTYBEES SMA-150 entry gate.**
Everything else in r/153 survives re-fitting on the honest entry.

---

## 8. What was NOT tested

1. **The blend.** No 3-sleeve weight sweep against True North + Open Alpha Base Age. Only
   pairwise weekly correlations were computed (A: 0.282 to OA, 0.256 to TN, 0.182 to the
   index — still the best diversifier of the set, though the refit raises correlation from
   the incumbent's 0.245/0.211). **Whether the refit is worth more to the book than the
   incumbent is therefore unanswered**, and it is the question that decides adoption.
   Handover §7 item 7 owns it.
2. **VIX gates.** INDIA VIX starts 2015, so they need their own window and their own
   baseline. Stage 4 tested only price-based index gates.
3. **Risk-based sizing** (`risk_pct`) and the structure stop (`stop_mode='struct'`) — both
   exist in the engine, neither was swept. Equal-weight sizing has failed three times in this
   project, which is why it was deprioritised, not because it was ruled out here.
4. **Pivot on highs rather than closes** (`pivot_mode='high'`), and the base-tightness /
   ATR% dial (`tight_max`). Left at r/153's values.
5. **`close_fill` re-optimisation.** Buying at the signal close was the best placeable
   mechanic at the *incumbent* parameters (17.39% vs 14.90%) and was never re-fitted. If a
   live process can act at 15:10–15:20, that arm might beat the next-day stop — a
   potentially ~2pp finding left on the table.
6. **Walk-forward / held-out period.** Only the WA/WB split.
7. **The rename/refresh defect** was confirmed as a live risk but not fixed (out of scope:
   it touches `scripts/refresh_daily_universe.py`).

---

## 9. Files

| File | What |
|---|---|
| `scripts/ipo_honest.py` | engine fork: 5 entry mechanics, name-based fund mask, stages 0–4 |
| `scripts/ipo_final.py` | stage 5 shortlist adoption arithmetic |
| `scripts/ipo_gate.py` | stage 6 paired gate comparison + per-year |
| `scripts/ipo_null_gated.py` | stage 7 null control on the gated candidate |
| `scripts/ipo_null_axis.py` | stage 8 null edge along the trail axis — the key table |
| `scripts/ipo_adopt.py` | stage 9 full adoption battery on the finalists |
| `results/stage0_mechanics.csv` | 10 cells: 5 mechanics × clean / r/153 universe |
| `results/diagnostics.json` | split exposure, cash-sweep attribution, invested fraction |
| `results/stage1_exits.csv` | 160-cell exit surface |
| `results/stage2a_geometry.csv` | 128-cell base geometry |
| `results/stage2b_book.csv` | 18-cell slots / sizing |
| `results/stage3_nulls.csv`, `results/stage7_null_gated.csv` | null controls |
| `results/cohort_null.json` | young+liquid cohort drift null |
| `results/stage4_gates.csv` | 11-cell gate bake-off |
| `results/stage6_paired.csv` | 30-seed paired deltas |
| `results/stage8_null_axis.csv` | null edge × trail × stop |
| `results/stage9_adoption.csv`, `stage9_peryear.json`, `stage9_curves.csv`, `stage9_correlations.json` | the finalists |

Reproduce: `venv/bin/python research/167_ipo_base_honest_reopt/scripts/ipo_honest.py all`
then `ipo_final.py`, `ipo_gate.py`, `ipo_null_gated.py`, `ipo_null_axis.py`, `ipo_adopt.py`.
All resume-safe; ~35 minutes total on the VPS.
