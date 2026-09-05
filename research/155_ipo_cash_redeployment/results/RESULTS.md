# research/155 — Redeploying the IPO Sleeve's Idle Cash into Open Alpha / True North

**Data snapshot:** `backtest_data/market_data.db`, VPS, max date 04-Sep-2026 · **Run date:** 05-Sep-2026
**Host:** VPS 94.136.185.54 · **Scripts:** `scripts/{ipo_park, run_sweep, replicate, report, ladder, chart, summ}.py`
All figures **after Indian tax** (20% STCG / 12.5% LTCG, FY-netted with loss carry-forward),
**net of 25 bps per side** unless a ladder row says otherwise, idle cash **5% p.a.**, blends
**monthly rebalanced**, and every A-vs-B figure is the **distribution of paired differences
across the same 30 paths** (30 Open-Alpha seeds × 30 IPO seeds × 12 True-North rebalance-day
offsets cycled), never an unpaired median.
**Cells disclosed: 114**, each a 30-path ensemble → **3,420 position-level sleeve simulations.**

---

## VERDICT: **CONCLUDED — the idle cash stays in cash.**
## Continuous redeployment is **NO EDGE** (worse on every limb). The forward-visibility gate is a real but **immaterial SIGNAL** that fails the pre-registered bar and dies on the cost ladder.

Six things, in the order that should change what Arun does:

1. **Arun's premise is confirmed and it is structural.** A name listed one or two days ago
   cannot be a candidate: the adopted spec needs **25 trading bars of history** and a
   **25-trading-day base window**, so a new listing is ineligible for roughly five weeks.
   The set of names that can possibly trigger over the next 25 sessions is therefore fully
   determined by listings that have **already happened** — forward visibility with **no
   look-ahead bias**. The mechanism he proposed is buildable exactly as he described it.

2. **It was built, and it works mechanically — it just does not matter.** The gated arm never
   missed a single IPO entry on any of the 30 paths (`missed = 0`), needed only ~30 pull-backs
   in twenty years, and left the sleeve's 674 trades completely intact. It delivers
   **+0.10 pp of blend CAGR, winning on 30 of 30 paired paths** — genuinely consistent, and
   genuinely one tenth of one percentage point. Median paired **Calmar change +0.006, winning
   21 of 30**, against a pre-registered bar of **+0.10 Calmar on ≥26 of 30**. **REJECT.**

3. **The arithmetic says there was never room for it to matter.** The IPO sleeve is 20% of the
   blend and sits 67.3% in cash, so the idle cash is **13.5% of the portfolio**. The candidate
   pool is empty on **19.0% of days** — and *identically so for horizons of 25, 50 and 100
   trading days*, because the droughts are multi-month, not multi-week. Time-averaged, the
   gate can therefore only touch **2.7% of the portfolio** (measured: 13.4% of sleeve NAV).
   A 2.7% tilt cannot move a Calmar.

4. **Redeploying continuously does move the needle — the wrong way.** Parking all idle cash in
   Open Alpha lifts blend CAGR by **+1.54 pp** (30/30 paths, with settlement waived) and
   worsens blend drawdown by **−3.85 pp**, for a Calmar change of **−0.375, losing on 30 of
   30**. The sleeve's daily correlation to Open Alpha goes from **0.21 to 0.90**. The sleeve
   stops being the uncorrelated thing that earned it a place in the book — **criterion 6
   breached on every continuous arm, frictionless or not.**

5. **The friction is real and large, but the friction is not what kills it.** Frictionless,
   continuous parking still fails: **Calmar −0.223, losing 28 of 30 paths.** Friction merely
   turns a bad idea into a much worse one: it costs **5.26 pp of blend CAGR**, converting
   +3.30 pp into −1.95 pp. On the gated arm friction costs **0.28 pp of blend CAGR — 73% of
   the gross benefit.**

6. **The honest alternative is a static weight, not a mechanism.** A plain **TN 35 / OA 35 /
   IPO 30** static blend returns **29.39% at −13.64%** versus the gated machinery's
   **29.02% at −13.66%** — *more* return, *equal* drawdown, and **zero** new operating
   complexity, settlement risk or pull-back tax. If Arun wants the idle cash working harder,
   the lever is the **sleeve's weight**, which r/154 already swept properly. The mechanism is
   dominated by doing nothing differently.

**What the idle cash actually is:** not waste, but the sleeve's **drawdown brake**. Across the
whole Phase-2 mechanic sweep the relationship is monotone — every configuration that parks
*more* of the sleeve's cash earns *more* CAGR and gives back *more than that* in drawdown.
Converting the cash into equity converts the IPO sleeve into a second Open Alpha. That is
precisely what r/153 and r/154 said the book does **not** need.

---

## 1. Phase R — the replication gate **PASSED, bit-for-bit**

Before anything new was tested, the rebuilt engine was required to reproduce research/153.

| Check | Result |
|---|---|
| Arm A (no parking) vs `research/153/results/ipo_equity_seeds.csv` | **max absolute NAV difference 0.0** across all 30 seeds × 5,128 days |
| r/153's published 40/40/20 blend (10 OA seeds × 3 TN offsets × 10 IPO seeds, unpaired medians) | **28.27% / −12.79% / Calmar 2.21 — reproduced exactly** (also 27.14 / −16.42 / 1.65 at w=0 and 27.72 / −14.44 / 1.92 at w=10%) |

**A convention correction confirmed on this study's own numbers.** The 40/40/20 blend's 2008
drawdown is **−1.66%** measured inside the 2008 calendar slice and **−12.23%** measured from
the running peak of the full curve. Every window figure in this study uses the latter (the
r/154 rule).

### The paired baseline this study must beat

30 paired paths, monthly rebalanced, after tax, 25 bps:

| Book | CAGR | MaxDD | Calmar | DD 2008–09 | DD 2012–14 |
|---|---|---|---|---|---|
| TN + OA 50-50 (the deployed pair) | 27.85 | −17.18 | 1.67 | −16.63 | −8.15 |
| **40/40/20 + IPO, idle → cash (INCUMBENT)** | **28.92** | **−13.59** | **2.181** | −12.55 | −4.36 |

---

## 2. What was tested

**The sleeve is unchanged.** Nothing in the IPO-Base MID signal, sizing or exit logic was
touched. The only new machinery is what happens to cash the sleeve is not using: idle cash
above a reserve is parked in an external NAV (Open Alpha / True North / 50-50 / NIFTYBEES) and
pulled back when a candidate triggers. This required **position-level** simulation of the
sleeve with an external cash sink/source — NAV-level blending cannot answer the question,
because redeployment changes the sleeve's own cash path, position sizes and trade set.

**Every friction is charged** (Arun: *"Pull-back friction must be modelled, not waived"*):

| Friction | Modelled as |
|---|---|
| Transaction cost | 25 bps per side on **both** the redemption and the re-parking (ladder 25 / 40 / 60) |
| Tax on realised gain | Two treatments — see the note below |
| Settlement | **T+1 primary**: cash from a sale on day *t* arrives on *t+1*, so the entry that forced the sale is **missed**. A T+0 arm isolates the settlement artefact |
| Which lot is sold | pro-rata (average cost) · **LIFO** (most-recent, least unrealised gain) · FIFO |
| The smarter mechanic | a liquidity **reserve** of *k* slot-sizes held in settled cash, plus a slower re-park **cadence** (daily / weekly / monthly), so entries are funded from the sleeve's own natural exits and uninvested cash first |

**Note on tax, stated plainly.** `oa_navs30.csv` and `tn_navs12.csv` are **already after-tax**
NAV series (r/154's `build_sleeves.py` passes `stcg=0.20, ltcg=0.125`), so taxing the NAV-lot
gain a second time **double-counts**. Both are reported: `tax=full` (Arun's literal
instruction, a strict **upper bound** on friction — and the arm the adoption decision is made
on) and `tax=txn` (transaction cost only, the economically correct **lower bound**). The gap
between them is 0.11 pp of blend CAGR on the gated arm and 0.31 pp on the continuous arm.

### Grid and cell count

| Phase | What | Cells |
|---|---|---|
| R | replication of the r/153 sleeve and blend | 1 |
| 1 | parking-asset bounds (OA / TN / 50-50 / NIFTYBEES) + frictionless twins | 8 |
| 2 | mechanics: settlement {T+0,T+1} × reserve {0,1,2} × cadence {daily,weekly,monthly} × lot policy {pro-rata,LIFO,FIFO} | 54 |
| 3 | forward-visibility gate: horizon {25,50,100} × asset {OA,TN,50-50} | 9 |
| 3b | the gate without the throttling reserve, + settlement/tax/frictionless twins | 21 |
| 4 | cost and tax ladders on the incumbent and the continuous arm | 15 |
| 5 | cost and tax ladder on the gated arm as specified | 7 |
| **Total** | disclosed for multiple-testing | **114** |

---

## 3. Phase 1 — the naive bound. Continuous parking is a decisive failure

T+1 settlement, no cash reserve, daily re-park, pro-rata lots, `tax=full`, 25 bps. Medians over
the same 30 paired paths:

| Arm | Parked asset | Sleeve CAGR | Sleeve DD | Sleeve trades | % invested in IPOs | Blend CAGR | Blend DD | Blend Calmar | Δ Calmar | Calmar wins | corr(daily) to OA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A incumbent** | cash | 31.38 | −20.88 | **674** | **32.7%** | **28.92** | **−13.59** | **2.181** | — | — | **0.21** |
| B | Open Alpha | 22.45 | −37.22 | 183 | 8.6% | 26.86 | −19.55 | 1.378 | −0.773 | **0/30** | **0.90** |
| C | True North | 15.01 | −32.92 | 196 | 9.2% | 25.35 | −18.67 | 1.409 | −0.746 | 0/30 | 0.40 |
| D | 50/50 OA+TN | 19.11 | −32.73 | 187 | 8.7% | 25.98 | −19.18 | 1.401 | −0.752 | 0/30 | 0.76 |
| N (null) | NIFTYBEES | 7.80 | −62.23 | 191 | 9.2% | 23.93 | −25.29 | 0.952 | −1.206 | 0/30 | 0.32 |

**Why it collapses, and it is not the tax.** With no cash reserve *every* IPO entry needs a
pull-back — **1,206 pull-back events per path**, and under T+1 **every single one missed its
trade** (`missed = pull-backs` exactly). The sleeve's trade count falls from 674 to 183 and its
invested share from 32.7% to 8.6%. The sleeve stops being an IPO sleeve and becomes a
badly-executed Open Alpha tracker.

---

## 4. Phase 2 — 54 mechanics. None of them rescues it

| Best cell in each family | Blend CAGR | Blend DD | Blend Calmar | Δ CAGR (paired) | Δ Calmar (paired) | Calmar wins |
|---|---|---|---|---|---|---|
| **Incumbent** | 28.92 | −13.59 | **2.181** | — | — | — |
| T+1, reserve 2, monthly, LIFO (**best realistic**) | 28.95 | −15.75 | 1.843 | −0.07 | **−0.358** | 0/30 |
| T+1, reserve 0, monthly, LIFO | 29.16 | −17.24 | 1.691 | +0.15 | −0.486 | 0/30 |
| T+1, reserve 0, daily, pro-rata (naive) | 26.86 | −19.55 | 1.378 | −1.95 | −0.773 | 0/30 |
| **T+0**, reserve 1, weekly, pro-rata (**settlement waived**) | 30.44 | −17.03 | 1.771 | **+1.54** | −0.375 | 0/30 |

Two structural readings:

- **The mechanic axis is monotone in one thing only — how much cash you convert.** Larger
  reserve, slower cadence → less parked → closer to the incumbent. The best possible mechanic
  is the limit of parking nothing, which *is* the incumbent.
- **Lot policy is inert.** Pro-rata, LIFO and FIFO differ by ≤0.01 Calmar in every family. The
  question "which position gets sold" — which looked like it might matter — does not.
  LIFO is marginally best (it realises the least unrealised gain), and marginally is the word.

---

## 5. Phase 3 / 3b — the forward-visibility gate (Arun's actual proposal)

Park **only** while no name can possibly become an eligible candidate for the next *N* trading
days. The pool is computed causally: liquidity as of *t*, plus bars-since-listing and calendar
age, both deterministic once a listing has already happened.

**The gate is empty on 19.0% of days — identically for N = 25, 50 and 100.** Droughts in the
Indian IPO pipeline last months, not weeks, so the look-ahead horizon is irrelevant. Use 25.

| Arm (asset, reserve, cadence) | Parked, % of sleeve | Pull-backs / 20 yrs | Entries missed | Blend CAGR | Blend DD | Δ CAGR | CAGR wins | Δ Calmar | Calmar wins | corr OA / TN |
|---|---|---|---|---|---|---|---|---|---|---|
| **Incumbent** | 0% | 0 | 0 | 28.918 | −13.591 | — | — | — | — | 0.21 / 0.22 |
| **GATED — OA, reserve 0, monthly** | 13.4% | 31 | **0** | 29.025 | −13.662 | **+0.105** | **30/30** | **+0.006** | 21/30 | 0.31 / 0.27 |
| GATED — OA, reserve 2, monthly | 8.3% | 30 | 0 | 29.012 | −13.646 | +0.096 | 30/30 | +0.007 | 21/30 | 0.28 / 0.26 |
| GATED — NIFTYBEES (null), reserve 0, daily | 18.5% | 43 | 0 | 29.324 | −13.669 | +0.400 | 30/30 | +0.028 | 21/30 | 0.25 / 0.29 |
| GATED — True North, reserve 0, monthly | 12.9% | 30 | 0 | 28.856 | −13.726 | −0.066 | 8/30 | −0.005 | 6/30 | 0.24 / 0.33 |

**The null wins.** The best gated cell parks into **NIFTYBEES**, not into Open Alpha or True
North. There is no OA- or TN-specific magic here — the effect is simply "hold a little more
equity beta during an IPO drought", and even that clears nothing.

### Where the gate actually fires (per-year, median of 30 paths)

| Year | % of sleeve parked | Pull-backs | Blend return, incumbent → gated |
|---|---|---|---|
| 2006 | 21.3 | 4 | 21.5 → 19.4 |
| 2008 | 2.4 | 2 | 0.8 → 0.6 |
| **2009** | **50.3** | 4 | 49.4 → **53.3** |
| 2010–2011 | 4.9 / 7.3 | 2 / 1 | 22.3 → 21.4 · 2.2 → 1.8 |
| **2012** | **49.4** | 3 | 12.8 → 12.6 |
| **2013** | **46.8** | 4 | 8.6 → 8.6 |
| **2014** | **64.3** | 3 | 52.0 → **59.7** |
| 2015 | 21.5 | 3 | 7.0 → 4.6 |
| 2016–2019 | 0–6.5 | 0–2 | ≈ unchanged |
| **2020–2026** | **0.0** | **0** | **identical** |

The gate does exactly what Arun designed it to do — it fires in 2009 and in the 2012-2014
drought and never once during the 2020-2026 IPO boom. It earns **+3.9 pp in 2009** and
**+7.7 pp in 2014**, gives back **−2.1 pp in 2006** and **−2.4 pp in 2015**, and is silent for
the last seven years. Two good years in twenty-one is not an edge; it is a coin that landed
right twice.

---

## 6. Friction — what the mechanism costs (the headline Arun asked for first)

| Arm | Frictionless Δ blend CAGR | All frictions on (T+1, 25 bps, `tax=full`) | **Friction drag** | Share of the gross benefit eaten |
|---|---|---|---|---|
| **GATED** (OA, reserve 0, monthly) | **+0.386** | **+0.105** | **0.281 pp** | **73%** |
| CONTINUOUS (OA, daily, reserve 0) | +3.303 | −1.954 | **5.257 pp** | 159% — it flips the sign |

Decomposed on the gated arm (all in pp of *blend* CAGR versus the incumbent):

| Removing… | Δ blend CAGR | Cost of that friction |
|---|---|---|
| nothing (frictionless) | +0.386 | — |
| add 25 bps transaction cost | +0.213 | **0.173 pp** — the transaction cost |
| add the (double-counted) NAV-lot tax | +0.105 | **0.108 pp** — the tax layer |
| add T+1 settlement | +0.105 | **0.007 pp** — settlement is nearly free *because the gate never forces a sale into a trade* |

Absolute rupees, per path, on a **₹10,00,000 sleeve** compounding to ≈ **₹26.7 crore** (×267, median seed) over
2006–2026: total pull-back transaction cost **₹4.19 L** and notional pull-back tax **₹7.38 L**
across the twenty years (~30 events) — together **0.43% of the terminal sleeve NAV**, which is
why the drag shows up as 0.28 pp of *compounding* rather than as a large rupee number.

**The settlement finding is worth keeping.** T+1 is catastrophic for continuous parking
(1,206 missed entries) and almost free for the gated design (0 missed entries), because the
gate guarantees, N days ahead, that nothing can trigger. **If this idea is ever revisited, the
gate is the only structure in which the settlement pipe is not a killer.**

---

## 7. Cost ladder — the small advantage does not survive it

Each arm paired against the incumbent **at the same cost** (a 40 bps arm is never scored
against a 25 bps incumbent):

| Arm | bps | Blend CAGR | Blend Calmar | Δ CAGR | Δ Calmar | CAGR wins | Calmar wins |
|---|---|---|---|---|---|---|---|
| Incumbent | 25 | 28.918 | 2.181 | — | — | — | — |
| **GATED (`tax=full`)** | **25** | 29.025 | 2.126 | **+0.105** | +0.006 | 30/30 | 21/30 |
| GATED (`tax=full`) | 40 | 28.477 | 2.064 | +0.005 | −0.001 | 18/30 | 13/30 |
| GATED (`tax=full`) | 60 | 27.808 | 1.956 | **−0.128** | −0.024 | **0/30** | 0/30 |
| GATED (`tax=txn`, lower bound) | 25 | 29.134 | 2.139 | +0.213 | +0.014 | 30/30 | 21/30 |
| GATED (`tax=txn`) | 60 | 27.917 | 1.969 | −0.018 | −0.011 | 9/30 | 5/30 |
| CONTINUOUS best (`tax=full`) | 25 | 28.945 | 1.843 | −0.066 | −0.358 | 12/30 | 0/30 |
| CONTINUOUS best (`tax=full`) | 60 | 27.991 | 1.723 | −0.033 | −0.347 | 13/30 | 0/30 |

**Criterion 4 fails.** The gated arm's entire advantage is gone by 40 bps and negative by 60.

---

## 8. The static-tilt null — a plain weight beats the machinery

A grid of 49 **static** weight vectors (TN / OA / IPO-with-cash), same 30 paired paths, same
tax and cost. (Calmar here = median CAGR ÷ |median MaxDD|, one estimator across the whole
grid.)

| Static vector | CAGR | MaxDD | Calmar |
|---|---|---|---|
| TN 45 / OA 30 / IPO 25 | 28.10 | −12.54 | **2.241** |
| TN 40 / OA 30 / IPO 30 | 28.62 | −12.87 | 2.224 |
| **TN 35 / OA 35 / IPO 30** | **29.39** | **−13.64** | 2.155 |
| TN 40 / OA 40 / IPO 20 (the incumbent) | 28.92 | −13.59 | 2.128 |
| TN 35 / OA 40 / IPO 25 | 29.54 | −13.84 | 2.134 |
| — for comparison, the **GATED** dynamic arm | 29.02 | −13.66 | — |

**TN 35 / OA 35 / IPO 30, a static number in a spreadsheet, returns more than the entire
gated mechanism at the same drawdown.** Redeployment is dominated by re-weighting. Continuous
parking is dominated by **16 of the 49** static vectors on *both* CAGR and drawdown
simultaneously.

*(This 49-cell grid is a null control, not a weight recommendation. r/154's frontier — which
also holds gold and enumerates 1,767 vectors on 360 paths — remains the reference for weights.)*

---

## 9. The pre-registered bar, scored

Fixed in the STATUS doc **before the first run**, and not relaxed afterwards.

| # | Criterion | GATED (OA) | GATED (NIFTYBEES null) | CONTINUOUS best T+1 | CONTINUOUS T+0 |
|---|---|---|---|---|---|
| 1 | +0.10 Calmar **or** −2 pp MaxDD at ≥ equal CAGR | **FAIL** (+0.006) | **FAIL** (+0.028) | FAIL (−0.358) | FAIL (−0.375) |
| 2 | wins on ≥ 26 of 30 paired paths | **FAIL** (21/30) | **FAIL** (21/30) | FAIL (0/30) | FAIL (0/30) |
| 3 | drought-window MaxDD not worse by > 1.5 pp | PASS | PASS | **FAIL** | **FAIL** |
| 4 | survives the 25/40/60 bps ladder | **FAIL** | not run | FAIL | not run |
| 5 | not dominated by a static weight vector | **FAIL** (1 of 49 dominates) | **FAIL** | **FAIL** (16 of 49) | **FAIL** (11 of 49) |
| 6 | correlation < 0.40 to **both** legs | PASS (0.31 / 0.27) | PASS (0.25 / 0.29) | **FAIL** (0.63 / 0.38) | **FAIL** (0.60 / 0.39) |
| | **Verdict** | **REJECT** | **REJECT** | **REJECT** | **REJECT** |

---

## 10. Caveats — read before acting

1. **The parked leg is modelled at NAV level, not position level.** Parking in "Open Alpha"
   means holding units of OA's after-tax NAV series, not simulating extra OA positions. A
   forced pull-back therefore liquidates a *slice* of OA rather than named positions. This is
   the right approximation for the question, but it means the forced-exit tax is modelled on
   wrapper lots (`tax=full`, which double-counts, and `tax=txn`, which ignores the
   early-realisation timing penalty). **The truth sits between the two arms**, and both are
   reported; the adoption call was made on the conservative one.
2. **T+1 was applied only to the parked leg**, not to the sleeve's own equity trades, so the
   comparison against r/153's incumbent stays exactly paired. Modelling T+1 on the sleeve's
   own trades too would penalise both arms roughly equally.
3. **The forward-visibility gate assumes the exchange trading calendar is known ahead**
   (it is published) **and that a name liquid today stays liquid** over the horizon. Neither
   is a price look-ahead, but both are assumptions.
4. **Everything r/153's caveats say still applies**, inherited unchanged: the entire IPO-Base
   edge lives in the **pivot buy-stop fill** (31.0% vs 17.0% CAGR on a close fill); no
   replication gate was run against the source site's own dials; survivorship inside the DB is
   small but the residual (IPOs that died before ever being onboarded to Kite) is unmeasurable
   and biases upward; `market_data.db` is **not retroactively split-adjusted**, mitigated by
   r/153's masking of 42 suspects; and the 2025 cohort's "ends early" rate is a feed-freshness
   artefact.
5. **NIFTYBEES is used as a price series** — dividends are not reinvested, so the index null
   is slightly understated. It still won the gated bake-off, which only strengthens the
   "this is just beta" reading.
6. **114 cells disclosed.** The best cell's numbers should be discounted accordingly — but the
   conclusion here does not rest on a best cell: **every** cell in the sweep failed, and the
   surface is monotone rather than noisy.
7. **Nothing was deployed.** No live engine, no crontab, no spec, no service restart. This
   study is read-only against the database.

---

## 11. Deliverables

| File | Contents |
|---|---|
| `results/paths.csv` | one row per (cell, path) — 114 cells × 30 paths, every metric and diagnostic |
| `results/adoption.csv` / `report.txt` | the scored adoption bar, per-year diagnostics, YoY house table, static-tilt null |
| `results/cost_ladder.csv` | the correctly-paired cost/tax ladder |
| `results/peryear.csv` | per-year blend return, intra-year DD (full-curve peak), % redeployed, pull-backs + cost |
| `results/yoy_returns.csv`, `yoy_intradd.csv`, `yoy_summary.csv` | YoY house-format table data |
| `results/static_tilt_null.csv` | the 49 static weight vectors |
| `results/nav_*.csv` | daily sleeve NAVs, 30 paths, for the headline arms |
| `results/r_baseline_paths.csv` | the paired incumbent baseline |
| `results/ipo_cash_redeployment_research155.png` | factsheet — growth of ₹100 (log), drawdown, what the mechanism can move, and the paired verdict |

## 12. Recommended next steps

1. **Do nothing to the IPO sleeve's cash policy.** Idle cash at 5% stays. It is the sleeve's
   drawdown brake, not dead weight.
2. **If the goal is "make the idle cash work", the lever is the sleeve's weight**, and r/154's
   frontier is the place that question is already answered properly (OA 40 / TN 25 / IPO 20 /
   GOLD 15). This study adds a null-control data point: a static IPO weight of 25–30% beats
   every dynamic redeployment arm tested.
3. **Keep the gate design on file.** It is the one structure in which T+1 settlement costs
   nothing (0 missed entries in 20 years), so if the IPO sleeve is ever run at a much larger
   weight — where 13.5% of the portfolio in idle cash becomes 25%+ — the arithmetic could
   change. **Registered as a dated review for 31-Mar-2027** in the Ops & Review Centre, with
   the pass criterion: revisit only if the sleeve's weight exceeds 30% **or** the pipeline has
   been in drought for more than 12 consecutive months.
4. **This study does not change any system's verdict.** IPO-Base MID remains a r/153 STRATEGY
   candidate at 20% weight, unpapered. No roster report is triggered (§9.1).
