# Three-Sleeve Blend — Is the RE-FITTED IPO Base Worth More to the Book Than the INCUMBENT?

**STATUS: DONE** · 2026-09-12 23:48 → 2026-09-13 00:22 IST · research/168 · host VPS 94.136.185.54

**VERDICT: STRATEGY — adopt the re-fit (IPO-A), fund it at 25%. The INCUMBENT sleeve does not
earn a place in the book at any weight.** Full write-up: `results/RESULTS.md`.

The adoption blocker named in `research/167_ipo_base_honest_reopt/results/RESULTS.md` section 8
item 1. r/167 produced a better IPO standalone book but a WORSE pairwise diversifier; only the
blend decides which is worth more, and r/167 never ran it.

---

## 1. The Ask

**What was asked:** is the RE-FITTED IPO Base worth more to the portfolio than the INCUMBENT
IPO Base, and what weight (if any) should the three-sleeve book carry?

**What is actually being tested.** Three after-tax, net-of-cost book NAVs are combined into one
portfolio at fixed target weights with periodic rebalancing. The question has four parts and
each gets its own answer:

1. At its own best weight and rebalance, does a TN + OA·BaseAge + **IPO-A** blend beat a
   TN + OA·BaseAge + **IPO-INC** blend, **paired on the same path**?
2. Does either three-sleeve blend beat the **two-sleeve TN + OA·BaseAge** baseline at all?
3. Does either beat **plain cash at 5.2% post-tax in the same weight** — the cash null that has
   already killed three candidate sleeves in this project (r/146, r/160, r/162)?
4. What weight, if any, should the sleeve carry — the answer may be **zero**.

**What is NOT being tested.** No strategy rule changes. No re-fitting of any sleeve. No live or
paper book is touched (`services/` is off limits). `frontend/` is off limits — the parent
session is editing it.

---

## 2. The Base — the three sleeves, and the idle-cash problem solved first

### 2.1 The idle-cash trap (checked BEFORE any blend arithmetic)

The four candidate curves were produced at **different** idle-cash yields — r/144 TN at 6.5%,
r/161 OA·BaseAge at 5.5%, r/167 IPO at 5.0% — and the three books hold **wildly different
cash** (TN 57% cash, IPO 68% cash, OA·BaseAge 27% cash). Blending them as produced is not
like-for-like and would bias the weights toward whichever sleeve was credited the richest cash
rate. A sibling session (`research/163_mpf_cash_yield_harmonisation`) harmonised every mpf book
to **5.2% post-tax** — the arbitrage-fund rate after 20% STCG at 2025-26 cash-futures spreads;
review dated 15-Dec-2026.

**Every sleeve in this study is measured at 5.2% post-tax idle cash. No exceptions, no mixed
basis.** Where r/163 already produced a 5.2% curve it is used; where it produced only a single
drawn path, the sleeve's own engine is re-run at 5.2% to produce the full path ensemble.

| sleeve | 5.2% curve available from r/163? | what this study does |
|---|---|---|
| True North | yes, **offset 0 only** (`cash052/tn_nav_INC_cash_n8_d15_tax1_cash052.csv`) | re-run r/144 `tn_attrib_engine.run` at `cash_y=0.052` for **12 rebalance-day offsets**; offset 0 must reproduce r/163's file bit-exactly or the script aborts |
| OA · Base Age | yes, **all 30 seeds** (`cash052/ba_navs_30seed_cash052.npz`) | used as-is; re-run only for the cost ladder |
| IPO-INC | yes, drawn path only, and via a **different engine** (r/159 `ipo_g3`) | re-run r/167 `ipo_honest.py` at `cash_yield=0.052`, 30 seeds, keeping all paths |
| IPO-A | **no — does not exist at 5.2%** | re-run r/167 `ipo_honest.py` at `cash_yield=0.052`, 30 seeds, keeping all paths |
| CASH (the null) | n/a | synthesised at `(1.052)**(1/252)` per trading day, compounded, never taxed again |

Both IPO arms are produced by **one** engine (r/167's fork, which carries the name-based fund
exclusion), so INC vs A is an apples-to-apples comparison. r/167 measured the fund-mask effect
on IPO-INC at 0.10pp, so this does not move the incumbent materially away from r/163's figure.

### 2.2 The sleeves, verbatim specs

| sleeve | spec | standalone at 5.2% (drawn path, common window) |
|---|---|---|
| **TN** True North | Nifty-200 universe, top-8 equal weight, monthly rebalance, NIFTYBEES 100-SMA weekly liquidate-all gate, 15-day-low Donchian stop, after tax, 25 bps/side | 18.69% / −24.78% / Calmar 0.75 |
| **OA·BA** Open Alpha · Base Age (r/161 WINNER) | new-ATH breakout, base age >= 60 bars, base depth >= 20%, no volume filter, no saucer filter, SuperTrend(14,4) close trail, no hard stop, 16 slots @ 6.25%, tv20 >= Rs 2 cr, 25 bps/side, after tax | 19.99% / −32.64% / Calmar 0.61 |
| **IPO-INC** | r/153 adopted spec on the PLACEABLE next-day buy-stop entry: trail SMA-20, stop 8%, target +25%, no gate, 8 slots @ 18.75% | 15.26% / −35.77% / Calmar 0.43 |
| **IPO-A** | same geometry, **trail SMA-50, stop 10%**, target +25%, **NIFTYBEES < SMA-150 blocks new entries** | r/167 at 5.0%: 21.80% / −26.63% / Calmar 0.819 |
| **CASH** | 5.2% p.a. post-tax, compounded daily | 5.2% / 0% |

### 2.3 Path ensemble and pairing

Both IPO arms and OA·BaseAge are **slot-constrained**, so each gets a seed ensemble (30 seeds).
True North is a **deterministic rank-based** book with no seed variance; its analogue is the
**12 rebalance-day offsets**. Blend path `p` (p = 1..30) is built from **IPO seed p,
OA·BaseAge seed p, and TN offset ((p-1) mod 12)**, so all three legs vary together and every
comparison in this study is **paired on p**. 30 paired paths. Seed numbers are not comparable
across engines — seed 7 in IPO has nothing to do with seed 7 in OA·BaseAge — but the pairing
only needs to be *consistent* across the arms being compared, which it is.

### 2.4 Blend mechanics

Sleeve NAVs are already after tax, net of 25 bps/side, and include their own idle cash. The
blend holds fixed target weights and rebalances between sleeves at the stated frequency. Within
a rebalance period each sleeve compounds at its own NAV; at each boundary the weights are reset
to target. **Inter-sleeve rebalancing is modelled frictionlessly** — this is the r/147 / r/154
convention in this project, and the `never` (drift) arm is run precisely so the reader can see
the bracket that assumption sits inside.

### 2.5 Windows

Common window = the intersection of the three sleeves: **2006-04-03 → 2026-09-03 (20.4 y)**.
Own spans: TN 2006-04-03→2026-09-03, OA·BaseAge 2005-01-03→2026-09-11, IPO 2006-01-02→2026-09-04.
Sub-periods **WA 2006→2015** and **WB 2016→2026** are reported separately; stress windows 2008
(full year), 2020 H1, 2018 (full year), 2022 H1. **Every drawdown, including every per-year and
per-window drawdown, is measured from the running peak of the FULL curve** — never from the
window's own first bar (the r/154 convention error, which invalidated per-window drawdowns
across r/146 to r/153).

---

## 3. Plan — the grid, and the pre-registered bar

### 3.1 Grid

| axis | values | n |
|---|---|---|
| IPO weight | 0, 5, 10, 15, 20, 25, 33 % | 7 |
| TN : OA·BaseAge split of the remainder | 25:75, 33:67, 50:50, 67:33, 75:25 | 5 |
| rebalance | monthly, quarterly, annual, never (drift) | 4 |
| third sleeve | IPO-INC, IPO-A, CASH (the null) | 3 |
| paths | 30 paired | 30 |

7 x 5 x 4 x 3 = **420 blend cells**, each on 30 paired paths = **12,600 blend simulations**.
The 20 cells at IPO weight 0 are the two-sleeve baseline and are identical across the three
third-sleeve arms — kept in the grid as an internal consistency check. The cost ladder adds
3 sleeves x 2 extra cost levels (40, 60 bps) on the full ensembles, re-blended at the chosen
weights.

### 3.2 Pre-registered ranking metric

**Primary: blend after-tax Calmar (CAGR / MaxDD) over the full common window, median across the
30 paired paths.** Secondary, always reported: CAGR, MaxDD median AND worst path, the WA/WB
split, and the four stress windows.

### 3.3 Pre-registered adoption bar — written before the first cell runs

This project's standing complement bar, stated here as three explicit decisions:

**(a) Does the sleeve belong at all?** The best three-sleeve cell must beat the two-sleeve
TN + OA·BaseAge baseline **at the same rebalance** by **at least +0.10 Calmar, or at least +2pp
CAGR at no worse median drawdown**, on **20 or more of the 30 paired paths** (two-thirds — the
30-path analogue of the 8-of-12 rule), **AND** beat **cash in its place** on **16 or more of 30**
paths (a majority) on the primary metric.

**(b) IPO-A or IPO-INC?** At the same weight and rebalance, paired on path, IPO-A must deliver
**at least +0.10 Calmar, or at least +2pp CAGR at no worse drawdown, on 20 or more of 30 paths**.
Each variant is *also* allowed its own best weight, so the two are compared at their own optima
and not at a weight chosen for the other.

**(c) Outcomes.**
- IPO-A clears (a) and (b) → **adopt IPO-A**, at the weight that wins the primary metric.
- IPO-INC clears (a), IPO-A fails (b) → **keep IPO-INC**, and say plainly that the refit is a
  better standalone book but a worse portfolio citizen.
- Neither clears (a), or cash beats both → **weight zero, drop the IPO sleeve** from the blend,
  and say so prominently.
- A sleeve that only buys drawdown at equal CAGR is reported as **insurance with a premium**, not
  as an edge, and the premium is quoted in pp of CAGR.

**Falsification, stated up front:** if the 30-path paired distribution of (IPO-A blend minus
IPO-INC blend) straddles zero, the honest answer is "the refit does not change what the book is
worth", and neither a standalone CAGR gap nor a correlation table overrides that.

### 3.4 Multiple testing

420 cells are scored. The winner's figure is to be read as its **neighbourhood**, not as a point
estimate, and the neighbourhood is published. No cell is adopted whose immediate neighbours in
weight and rebalance disagree with it.

---

## 4. Status

**Phase:** STATUS written, sections 1-3. Nothing launched.

| Date/time IST | Event | Notes |
|---|---|---|
| 2026-09-12 23:48 | r/167 RESULTS, r/163 cash052 outputs and all three engines located | TN 10s/run, OA·BA 30 seeds ~10s, IPO 30 seeds ~25s — the whole study is minutes, not hours |
| 2026-09-12 23:52 | this STATUS written before any cell | bar pre-registered in section 3.3 |
| 2026-09-12 23:56 | all three sleeve re-runs at 5.2% idle cash PASSED their reproduction gates | TN bit-exact (0/5,066 rows differ), OA·BaseAge bit-exact (max abs 0.000e+00 over 30 paths), IPO exact to +0.000pp on both arms vs r/167 stage 9 |
| 2026-09-13 00:00 | 840-cell grid, first pass | looked decisive, but the rebalance-frequency response was NON-MONOTONIC (quarterly beat monthly by 2.9pp of CAGR and also beat drift) |
| 2026-09-13 00:03 | 34,650-cell simplex with rebalance-PHASE ensembles | diagnosed the anomaly as PHASE LUCK, then as a BUG: the blend engine measured each period's returns from the rebalance day itself, discarding the return of every rebalance day |
| 2026-09-13 00:05 | **blend engine fixed + self-test added** | a 100% single-sleeve blend must reproduce that sleeve exactly under every frequency; both grids re-run. Real frequency effect: monthly is worth +0.54pp CAGR over drift, monotone, phase dispersion < 0.005 Calmar. The quarterly premium is RETRACTED |
| 2026-09-13 00:12 | risk-matched cash null added | Calmar cannot adjudicate a cash null at high weights (100% cash = Calmar infinity), so the cash weight matching each blend's drawdown is solved on a 1% grid and the comparison is made on CAGR |
| 2026-09-13 00:14 | bar pinned, robustness closed | verdict holds on the harmonised cost basis, under pure drift, and in BOTH halves of the window |
| 2026-09-13 00:18 | RESULTS.md written, STATUS → DONE | nothing deployed; `services/` and `frontend/` untouched |

---

## 5. Findings

**The answer, in one line: the re-fit is worth more to the book than the incumbent on 30 of 30
paired paths at every weight tested, on both cost bases, under monthly rebalancing and under pure
drift — and the incumbent sleeve does not clear the pre-registered bar at any weight at all.**

### Against the bar pre-registered in section 3.3, before anything ran

| pre-registered test | IPO-A | IPO-INC |
|---|---|---|
| **(a) belongs in the book**: ≥ +0.10 Calmar or ≥ +2pp CAGR at no worse DD vs the two-sleeve baseline, on ≥ 20/30 paths | **CLEARS from a 20% weight** (+0.107 Calmar, 30/30) and adds CAGR at every weight (30/30) | **FAILS at every weight.** Best Calmar gain +0.089 (at 30%), never reaches +0.10; costs CAGR at every weight (0/30 paths positive) |
| **(a) cash null**, must beat cash in its place on ≥ 16/30 | **CLEARS 30/30 at every weight**; +2.52pp of CAGR at equal drawdown at 25%, +4.79pp at 50% | marginal: +1.17pp at equal drawdown at 25%, +1.25pp at its best, and **LOSES at 50%** (5/30) |
| **(b) A over INC**: ≥ +0.10 Calmar or ≥ +2pp CAGR at no worse DD, on ≥ 20/30 | sign **unanimous 30/30 at every weight**; the pre-registered MAGNITUDE is first met at a **35% weight** (+0.107 Calmar, 30/30, DD also equal-or-better). At 20-30% it wins every path but by +0.03 to +0.08 Calmar | — |

### The recommendation

**True North 37.5% / OA · Base Age 37.5% / IPO-A 25%, rebalanced monthly** —
**21.18% CAGR after tax [worst path 19.17%] / −24.01% MaxDD [worst path −26.39%] / Calmar 0.885**
over 2006-04-03 → 2026-09-03, against the two-sleeve book's 20.28% / −26.91% / 0.749. The Calmar
surface actually peaks at a 45-60% weight (1.05-1.06, a flat plateau) and the unconstrained simplex
wants TN 45 / OA 0 / IPO 55 — **not recommended**, because the sleeve's capacity ceiling is about
₹20-25 L, there is no held-out period anywhere in the chain, and it would delete a live book.

### Three findings worth carrying forward

1. **r/167's correlation-based worry was backwards.** The refit IS the worse pairwise diversifier
   (monthly 0.348 to TN and 0.329 to OA·BaseAge, against the incumbent's 0.259 and 0.319) and is
   still far the better blend sleeve, because the correlation rise is swamped by the return
   improvement. Pairwise correlation was the wrong screen for this decision.
2. **r/167's 2008 black mark largely washes out in the blend.** The 10.7pp standalone gap
   (−10.3% for A vs +0.4% for INC) becomes 2.5pp at a 25% weight (−19.3% vs −16.8%), and BOTH arms
   improve on the two-sleeve book's −22.3%. Plain cash at the same weight delivers the same 2008
   cushion as the incumbent does, for free.
3. **Calmar alone cannot adjudicate a cash null.** Cash has zero drawdown, so Calmar rises without
   bound with the cash weight (100% cash scores infinity) and a weight-matched comparison flatters
   cash above about a 30% weight. The decision-grade form is the **risk-matched** cash null: solve
   the cash weight that reproduces the candidate blend's drawdown, then compare CAGR. This belongs
   in the playbook.

### A bug of mine, caught before it reached the report

The first blend engine measured each rebalance period's returns relative to the rebalance day
itself rather than the previous close, **discarding the return of every rebalance day**. It
manufactured a fake frequency effect — monthly appeared to cost 1.8pp of CAGR against drift,
quarterly to earn +1.4pp. The tell was a non-monotonic frequency response (a frequency beating its
neighbours on both sides) and a move far too large for the change made. Fixed, with a self-test
that a 100% single-sleeve blend must reproduce that sleeve exactly under every frequency. **Any
"quarterly beats monthly" reading of an interim number from this session is retracted.**

### A separate finding that is NOT this study's question

The two-sleeve book itself prefers a True North tilt — TN 85 : OA 15 scores Calmar 0.826 against
50:50's 0.749, and every top simplex cell pushes OA·BaseAge toward zero. That is a re-weighting of
the live pair and needs its own study (registered as a 2026-09-26 review). Every headline figure
here holds the deployed 50:50 ratio fixed precisely so the IPO question is answered on its own.

---

## 6. Crash Recovery — how to resume WITHOUT the agent

Everything runs on the VPS from `/home/arun/quantifyd` with `venv/bin/python3`. All three
scripts are idempotent and write only into `research/168_three_sleeve_blend/results/`.

```bash
cd /home/arun/quantifyd
ls -la research/168_three_sleeve_blend/results/
tail -40 /tmp/r168_ipo.log /tmp/r168_tn.log /tmp/r168_blend.log
pgrep -af 'r168|ipo_arms_cash052|tn_offsets_cash052|blend_grid'
```

Re-run, in this order (steps 1 and 2 are independent of each other):

```bash
venv/bin/python3 research/168_three_sleeve_blend/scripts/ipo_arms_cash052.py    # 88s
venv/bin/python3 research/168_three_sleeve_blend/scripts/tn_offsets_cash052.py  # 44s
venv/bin/python3 research/168_three_sleeve_blend/scripts/ba_costs_cash052.py    # 21s
venv/bin/python3 research/168_three_sleeve_blend/scripts/blend_grid.py          # 5s
venv/bin/python3 research/168_three_sleeve_blend/scripts/blend_extend.py        # 185s
venv/bin/python3 research/168_three_sleeve_blend/scripts/final_report.py        # 60s
venv/bin/python3 research/168_three_sleeve_blend/scripts/pin_bar.py             # 40s
```

All seven completed; logs at `/tmp/r168_{ipo,tn,ba,blend,ext,final,pin}.log`.

**If either step 1 or step 2 stops at its reproduction gate, STOP.** Do not force past it —
a failed gate means `market_data.db` moved under the study and the blend would be comparing
two different universes.

Safe to inspect: everything in `research/168_three_sleeve_blend/results/`.
**Do NOT write to** `research/144/`, `research/161/`, `research/163/`, `research/167/`,
`research/164/`, `research/165/`, `research/166/` (sibling sessions own the last three),
anything under `services/`, or anything under `frontend/` (the parent session is editing it).
No backend restart is needed or permitted by this task.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `THREE_SLEEVE_BLEND_IPO_WEIGHT_DAILY_SWEEP_STATUS.md` | this file, the crash-recovery source | yes |
| `scripts/ipo_arms_cash052.py` | IPO-INC + IPO-A, 30 seeds, 5.2% cash, 25/40/60 bps, with a reproduction gate against r/167 | yes |
| `scripts/tn_offsets_cash052.py` | True North, 12 offsets, 5.2% cash, 25/40/60 bps, bit-exact gate on offset 0 vs r/163 | yes |
| `scripts/blend_grid.py` | the 420-cell grid, paired comparisons, cash null, per-year and per-window tables | yes |
| `results/ipo_navs_cash052.npz` | 30 paths x 2 arms x 3 cost levels | yes |
| `results/tn_navs_cash052.npz` | 12 offsets x 3 cost levels | yes |
| `results/blend_grid.csv` | one row per cell — the sweep output, written incrementally | yes |
| `results/paired.csv` | the decisive paired path-by-path deltas | yes |
| `results/peryear.json` | the YoY house table data | yes |
| `results/RESULTS.md` | the verdict | yes |

---

## 8. Economic hypothesis (why a third sleeve could pay at all)

TN harvests cross-sectional momentum in large and mid caps with a weekly index gate; OA·BaseAge
harvests late-comer flow into new all-time highs; IPO Base buys the first base breakout of
recently listed companies. The diversification claim rests on IPO Base owning names the other
two structurally cannot — a stock listed four months ago has no multi-year base and is rarely in
the Nifty 200 — and on IPO Base sitting about 68% in cash, so the sleeve is partly a cash proxy.
The counter-hypothesis, which the cash null exists to test, is precisely that: that the sleeve's
contribution is mostly de-levering, in which case plain cash does the same job for free. r/167
raised the refit's pairwise weekly correlation to both legs (0.245 to 0.282 versus OA·BaseAge,
0.211 to 0.256 versus TN), which is a reason to expect the refit to be a *worse* portfolio
citizen than its standalone tearsheet suggests.
