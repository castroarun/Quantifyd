# Idle Cash At 5.2% Post-Tax — The Arbitrage-Fund Rate Across Every Momentum Portfolio Book

**STATUS: DONE** · 2026-09-12, 19:35 → 23:5x IST · research/163 · host: VPS 94.136.185.54

Follows `MPF_CASH_YIELD_HARMONISATION_DAILY_RUN_STATUS.md` (same folder, 12-Sep-2026 18:33→19:05),
which moved True North from 6.5% and Open Alpha · Base Age from 5.5% onto a common **5.0%**.
This run moves **every** book on `/app/mpf-report` from 5.0% to **5.2%** and writes the
instrument assumption onto the page.

---

## 1. The Ask

**What Arun asked (12-Sep-2026 evening):**

> the Momentum Portfolio report `/app/mpf-report` must credit idle cash at **5.2% a year
> post-tax** on EVERY book (was 5.0%). The 5.2% is the arbitrage-fund rate after 20% short-term
> tax at 2025-26 cash-futures spreads; the rule is that idle cash is kept in the best post-tax
> cash instrument. Re-run every curve the page uses at 5.2%, regenerate the page, write the
> assumption onto it, register the review, log it. Nothing else about any system changes.

**What is actually being tested / measured.** Not a strategy question at all: it is a
**re-measurement** of five existing books under one changed input, the annual rate credited to
the cash sleeve. No entry, exit, stop, slot count, universe or gate moves. The question each
re-run must answer is only:

> holding the rule set fixed and the engine fixed, what does each book's after-tax curve become
> when idle cash earns 5.2% a year instead of 5.0%, accrued daily and never taxed again?

and the cross-check is arithmetic rather than statistical: the CAGR each book gains must be
about **(1 − invested fraction) × 0.2 pp**, because that is what the extra 20 bps is worth on
the share of the book that is in cash.

**Why 5.2%, and why it is written on the page.** Arun's standing rule is that idle cash sits in
the **best post-tax cash instrument**, not in a savings account. At 2025-26 cash-futures spreads
an **arbitrage fund** yields roughly **6.5% pre-tax**; arbitrage funds carry **equity taxation**
(20% STCG on units churned inside a year, 12.5% LTCG beyond a year, ~0.25% exit load inside a
month), so ~6.5% pre-tax lands at about **5.2% post-tax**. The alternative — a liquid ETF —
is taxed at slab: `LIQUIDCASE` / `LIQUIDADD` / `LIQUIDBETF` in `market_data.db` realised
**5.4–5.5% pre-tax in 2025** and about **5.0% annualised in 2026**, which at a 30% slab is only
~3.5% post-tax. Operating rule that follows: **bulk in the arbitrage fund, a liquid-ETF buffer
for money needed at the next open**, because arbitrage redemptions settle T+1. 5.2% is a **flat
assumption**, not a measured realised yield, and it is **reviewed 15-Dec-2026**.

**Scope guard.** Nothing under `services/`, no live or paper book, no crontab, no backend
restart, no writes to any live DB. Every re-run is one of the studies' OWN engines called with
one different keyword argument, writing into research/163's own folder. research/144, 159, 160,
161 and 164 files are read only.

---

## 2. The Base — exactly what is re-run, and where each 5.0% number comes from

`/app/mpf-report` is built by `research/_utilities/mpf_report_build.py`. It consumes **five
book curves**, the index, and one derived blend:

| # | Book on the page | Curve the page reads today | Engine | Yield knob |
|---|---|---|---|---|
| 1 | **True North** | `research/163/results/tn_nav_INC_cash_n8_d15_tax1_cash05.csv` | `research/144 .../tn_attrib_engine.run` | `cash_y` |
| 2 | **Open Alpha · Base Age** | `research/163/results/ba_nav_winner_cash05.csv` (+ the 30-seed npz) | `research/161 .../bt_core.simulate` | `cfg['idle_yield']` |
| 3 | **IPO Base** (honest next-day entry) | `research/159 .../ipo_honest_curve.csv` | `research/153 .../ipo_replay.simulate_ipo` via `ipo_g3.run` | `cash_yield` |
| 4 | **Open Alpha · ATH + VIX** | summary row only, from `research/159 .../all_systems_summary.json` | `research/158 .../oa_entry_mechanics.simulate` via `build_curves.oa_curve` | `cash_yield` |
| 5 | **Quality Summit** | `research/160 .../F_Bb7_equity.csv`, 12 offsets | `research/160 .../qg_engine.run_cell` | `Cell.cash_yield` |
| — | NIFTYBEES | price series | none | **no cash — does not move** |
| — | TN + Base Age 50-50 | computed by the generator from (1) and (2) | — | inherits |

Books 1 and 2 were already re-run last night from 6.5% / 5.5% to 5.0%. Books 3, 4 and 5 were
**not touched** last night because their studies already used 5.0%. All five move now.

### 2.1 The post-tax property is preserved, in every engine

In each engine the cash accrual is credited to the cash balance **each bar** and is **never**
passed through the capital-gains settlement, which touches realised equity P&L only. This was
verified line by line in last night's STATUS §2.1 for engines 1, 2, 4 and 5; engine 3
(`ipo_replay.simulate_ipo`) does the same — `y_day = 1 + cash_yield/252`, applied as
`cash *= y_day` inside the daily loop, while `fy_st` / `fy_lt` accumulate only from position
exits. So "5.2% post-tax" means the same thing on every row of the page.

Note one convention difference that is **left alone deliberately**: engines 1, 2 and 5 compound
(`(1+y)**(1/252)`), engine 3 and 4 accrue simple (`y/252`). That is each study's own arithmetic;
changing it would be a second, silent change. At these rates the two differ by under 1 bp a year.

### 2.2 The exact cells

| Book | Cell, verbatim |
|---|---|
| True North | `E.run(ctx, tax=True, offset=0, cash_y=Y, series='NIFTYBEES', cons='sma100', n=8, exit=('donch',15))`, panel truncated to **2026-09-03** (research/144's own last bar). Single path — not an ensemble. |
| Base Age | research/161's pre-registered WINNER: base age ≥ 60 bars, depth ≥ 20%, no volume filter, no saucer filter, SuperTrend(14,4) close trail, no hard stop, 16 slots at 6.25%, tv20 ≥ ₹2 cr, 25 bps a side, after tax, next-open fills both sides, **30 seeds**, calendar cut at `curves161.npz`'s last date. Drawn path = `argsort(cagr)[n//2]`. |
| IPO Base | `ipo_g3.run(ctx, SPEC, seeds 1..30, W2=('2006-01-01','2026-09-04'), trig/piv/lo shifted to the HONEST next-day entry)` exactly as `research/159 .../ipo_curve.py` does it, `cash_yield=Y` passed through `**over`. Drawn path = median-CAGR seed. |
| Open Alpha · ATH + VIX | `build_curves.oa_curve(w, gated=True, win=('2016-01-01','2026-08-31'))` — trail-75 SMA, 8% stop, 16 slots at 6.25%, 25 bps, fill at the breakout close, INDIA VIX above its own 252-day 70th percentile blocks new entries, 30 seeds, `cash_yield=Y`. Its published row (19.23 / −34.15 / 0.56) is measured on `compare_all.py`'s common window, so that alignment is reproduced too. |
| Quality Summit | research/160 finalist `F_Bb7` = `base(label='F_Bb7', mask=masks_study/b7_g10_qual_mc.npz, mask_missing='fail', exits='none', slots=15)` → rebalance/monthly/rs/buffer 1.5/near/k 0.90/tv 2.0/next_open/25 bps/tax on/**12 offsets**, 2018-08-01 → 2026-09-10, on the frozen `panel_2000.npz`. Drawn path = the offset whose CAGR is nearest the median (the generator picks it, not this script). |

### 2.3 Success criteria, pre-registered before running

1. **Harness proof first, every book.** Each script must first reproduce its own **5.0%** curve
   — bit-exactly where the inputs are frozen (Base Age, Quality Summit, and True North against
   last night's cash05 file), or exact-over-history-with-a-tail-only-divergence where
   `market_data.db` has been refreshed since the study ran. Only then is the yield changed.
   A script that fails its gate exits non-zero rather than writing a curve it cannot prove.
2. **NIFTYBEES must be bit-identical** in both assembled curve files.
3. **Every other column must move by a small POSITIVE amount**, and per book that move must
   agree with `(1 − invested fraction) × 0.2 pp` within the book's own path noise.
4. **Nothing reorders.** If the ranking of books by CAGR or by Calmar changes on either window,
   that is reported loudly rather than absorbed.

---

## 3. Plan

| Step | Script (all new, in `scripts/`) | Output (all in `results/cash052/`) |
|---|---|---|
| 1 | `tn_cash052.py` | `tn_nav_..._cash052.csv` + the 5.0% proof copy |
| 2 | `ba_cash052.py` | `ba_nav_winner_cash052.csv`, 30-seed npz, `ba_seed_stats_052.csv`, `ba_invested_052.csv` |
| 3 | `ipo_cash052.py` | `ipo_honest_curve_cash052.csv` + seed stats |
| 4 | `oa_vix_cash052.py` | `oa_gated_cash052.csv`, `athvix_summary_cash052.json` |
| 5 | `qs_cash052.py` | `F_Bb7_equity_cash052.csv` (12 offsets) |
| 6 | `build_inputs_052.py` | `full_period_after_tax_cash052.csv`, `all_systems_after_tax_cash052.csv`, `curve_swap_report_052.json` |
| 7 | `check_cash052.py` | the NIFTYBEES-frozen + per-book consistency report |
| 8 | `research/_utilities/mpf_report_build.py` | defaults re-pointed, every 5%/5.0% string → 5.2%, arbitrage-fund note added; JSON + 11 PNGs |
| 9 | `frontend/src/data/mpf_report.ts` | own-window annotations and cash wording made true |
| 10 | ops_center REVIEWS + `docs/LABS_AND_JOBS_REFERENCE.md` + `TODO.md` | the 15-Dec-2026 review |

**Grid.** There is no sweep. 1 + 30 + 30 + 30 + 12 = 103 simulations at 5.2%, each paired with
the same run at 5.0% for the proof = **206 simulations**.

**One open decision, resolved in §5:** `research/159 .../after_tax_tables.csv` (the entry-mechanic
surface, the null control and the two gate bake-offs, 70 rows × 30 seeds) is also a 5.0% output.
It is re-run at 5.2% **only if it is cheap**; otherwise its captions are labelled "5.0% idle
cash" so the page never carries a mixed basis silently.

---

## 4. Status

**Phase:** DONE. All five books re-run, both curve files assembled, consistency check PASS,
page regenerated, frontend built, review registered.

| Date/time IST | Event | Notes |
|---|---|---|
| 2026-09-12 19:27 | folder read, all five harnesses located | r/163 scripts, r/159 build_curves / ipo_curve / full_period / aftertax_all / compare_all, r/160 qg_engine + make_grid |
| 2026-09-12 19:35 | this STATUS written, sections 1-4 | before anything ran |
| 2026-09-12 19:36 | **True North DONE (25 s)** | gate: reproduced the 5.0% file over **5,066 of 5,066 rows**, max rel 2.5e-16. 5.0% → 5.2% = **+0.130 pp** CAGR against +0.114 predicted at 43% invested |
| 2026-09-12 19:40 | Base Age first attempt STOPPED at its own gate | all 30 paths bit-identical in the npz but the CSV differed by 1.2e-16 — a decimal round trip, not an engine difference. Gate relaxed to "npz exactly 0.0 AND csv < 1e-12" and re-run |
| 2026-09-12 19:43 | **IPO Base DONE (53 s)** | gate: reproduced `ipo_honest_curve.csv` over **5,128 of 5,128 rows**, and the published 15.00% median. Paired **+0.156 pp** against +0.136 predicted at 31.8% invested |
| 2026-09-12 19:45 | OA · ATH + VIX first attempt STOPPED at its own gate | the published column is ffilled onto a **union** index (2,651 rows) while the book has 2,642 trading days, so the equality test was wrong. Changed to "my index is a SUBSET of the published one, compared on my index" |
| 2026-09-12 23:24 | session resumed after an API rate limit | disk state re-verified; TN, BA, IPO already on disk |
| 2026-09-12 23:3x | **Base Age re-run DONE (10 s)** | gate EXACT, 30/30 paths. Paired **+0.060 pp** against +0.054 predicted at 72.9% invested. **Drawn seed frozen at 29** (this run's own median would have been 16) |
| 2026-09-12 23:4x | **OA · ATH + VIX DONE (63 s)** | **two** gates: the raw curve reproduced over 2,642 of 2,642 rows, AND re-measuring it on `compare_all.py`'s aligned index returned the published **19.23 / −34.15 / 0.56 exactly**. Drawn seed frozen at 30 |
| 2026-09-12 23:4x | **Quality Summit DONE (10 s)** | gate BIT-EXACT against `F_Bb7_equity.csv` on the frozen panel (max rel 1.9e-16). Paired **+0.013 pp** against +0.018 predicted at 91.2% invested. Drawn offset stays `offset3` |
| 2026-09-12 23:3x | `aftertax_all_052.py` launched detached | the entry-surface / null / gate tables at 5.2%; ~3.3 min a row × 70 rows ≈ 4 h, so the page was NOT gated on it — see §8.4 |
| 2026-09-12 23:5x | `build_inputs_052.py` DONE | both curve files written; NIFTYBEES asserted unmoved inside the script |
| 2026-09-12 23:5x | `check_cash052.py` **PASS** | NIFTYBEES bit-identical in both files; all five books' paired yield effect consistent; **nothing reorders** on either window |
| 2026-09-12 23:5x | generator + `mpf_report.ts` + `MpfReport.tsx` edited, page regenerated, 11 PNGs | |
| 2026-09-12 23:5x | `npm run build` **BUILD_OK**; bundle and served JSON verified to carry "5.2%" and "ARBITRAGE FUND" | no backend restart — none needed |
| 2026-09-12 23:5x | review registered in `ops_center.py`, renders at the TOP of 68 reviews; mirrored in `docs/LABS_AND_JOBS_REFERENCE.md` | |

---

## 6. Crash Recovery

Everything runs on the VPS from `/home/arun/quantifyd`, python `venv/bin/python3`.

```bash
cd /home/arun/quantifyd
ls -la research/163_mpf_cash_yield_harmonisation/results/cash052/
tail -40 /tmp/tn052.log /tmp/ba052.log /tmp/ipo052.log /tmp/oavix052.log /tmp/qs052.log
pgrep -af 'cash052|mpf_report_build'
```

Re-run any step from scratch — each is idempotent and writes only into `results/cash052/`:

```bash
venv/bin/python3 research/163_mpf_cash_yield_harmonisation/scripts/tn_cash052.py
venv/bin/python3 research/163_mpf_cash_yield_harmonisation/scripts/ba_cash052.py
venv/bin/python3 research/163_mpf_cash_yield_harmonisation/scripts/ipo_cash052.py
venv/bin/python3 research/163_mpf_cash_yield_harmonisation/scripts/oa_vix_cash052.py
venv/bin/python3 research/163_mpf_cash_yield_harmonisation/scripts/qs_cash052.py
venv/bin/python3 research/163_mpf_cash_yield_harmonisation/scripts/build_inputs_052.py
venv/bin/python3 research/163_mpf_cash_yield_harmonisation/scripts/check_cash052.py
venv/bin/python3 research/_utilities/mpf_report_build.py
export PATH=$HOME/.nvm/versions/node/v20.20.2/bin:$PATH; cd frontend && npm run build
```

**If a script stops at its reproduction gate, STOP.** Do not force past it.

To rebuild the page at the previous **5.0%** basis:

```bash
venv/bin/python3 research/_utilities/mpf_report_build.py \
  --curves-dir research/163_mpf_cash_yield_harmonisation/results \
  --full-period-csv full_period_after_tax_cash05.csv \
  --roster-csv all_systems_after_tax_cash05.csv
```
(the generator's hard-coded ATH+VIX row and its prose would still read 5.2% — the 5.0% rebuild
is a curve-level fallback, not a full revert.)

**Do NOT touch** anything under `research/144/results`, `research/159/results`,
`research/160/results`, `research/161/results`, `research/164/`. Nothing in this run writes
there. `results/panel163.pkl` is a rebuildable ~50 MB cache.

**No backend restart is needed or permitted by this task.**

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `MPF_CASH_YIELD_5P2_DAILY_RUN_STATUS.md` | this file | yes |
| `scripts/tn_cash052.py` … `scripts/qs_cash052.py` | the five re-runs, each with its own proof gate | yes |
| `scripts/build_inputs_052.py` | assembles the two curve files | yes |
| `scripts/check_cash052.py` | NIFTYBEES frozen + per-book (1−inv)×0.2 pp consistency | yes |
| `results/cash052/*.csv`, `*.json` | the re-run curves and summaries | yes |
| `results/cash052/ba_navs_30seed_cash052.npz` | all 30 Base Age paths | yes |
| `results/cash052/RESULTS_CASH052.md` | the verdict | yes |
| `results/panel163.pkl` | rebuildable price panel, ~50 MB | **NO — gitignored** |

---

## 8. Findings

### 8.1 Before / after — every headline row, both windows, 5.0% → 5.2%

Taken from the report JSON itself: `results/cash052/mpf_report_before_052.json` (the page as it
stood at 5.0%) against the regenerated `static/app/mpf_report.json`.

**20.4-year window, 2006-04-03 → 2026-09-03, after tax.**

| Row | CAGR 5.0 → 5.2 | Δ | MaxDD 5.0 → 5.2 | Calmar 5.0 → 5.2 |
|---|---|---|---|---|
| **True North** | 18.56% → **18.69%** | **+0.13** | −24.95% → −24.78% | 0.74 → **0.75** |
| **Open Alpha · Base Age** | 19.93% → **19.99%** | **+0.06** | −32.73% → −32.64% | 0.61 → 0.61 |
| **IPO Base** | 15.10% → **15.26%** | **+0.16** | −35.86% → −35.77% | 0.42 → **0.43** |
| **TN + Base Age 50-50** | 19.80% → **19.89%** | **+0.09** | −25.68% → −25.60% | 0.77 → **0.78** |
| NIFTYBEES | 10.58% → 10.58% | **0.00** | −59.71% → −59.71% | 0.18 → 0.18 |

**2018 window, 2018-08-01 → 2026-09-04, after tax.**

| Row | CAGR 5.0 → 5.2 | Δ | MaxDD 5.0 → 5.2 | Calmar 5.0 → 5.2 |
|---|---|---|---|---|
| **True North** | 19.80% → **19.93%** | **+0.13** | −19.14% → −19.04% | 1.03 → **1.05** |
| **Open Alpha · Base Age** | 25.54% → **25.57%** | **+0.03** | −26.61% → −26.60% | 0.96 → 0.96 |
| **Quality Summit** | 20.90% → **20.63%** | **−0.27** | −38.90% → −38.90% | 0.54 → 0.53 |
| **IPO Base** | 12.97% → **13.08%** | **+0.11** | −35.86% → −35.77% | 0.36 → **0.37** |
| **TN + Base Age 50-50** | 23.40% → **23.48%** | **+0.08** | −20.46% → −20.45% | 1.14 → **1.15** |
| NIFTYBEES | 10.92% → 10.92% | **0.00** | −36.34% → −36.34% | 0.30 → 0.30 |

**Open Alpha · ATH + VIX** (the one summary row, 2016-01-01 → 2026-09-04):
19.23% / −34.15% / 0.56 → **20.44% / −32.92% / 0.62**, with a 30-seed CAGR band of
**15.60 – 22.57%, median 18.81%**. See §8.3 — that +1.20 is re-draw, not the cash rate.

**Nothing reorders on either window**, and the Quality Summit offset ensemble still spans
17.04 – 23.42% (median 21.26%) against 17.45 – 23.50% (21.05%) at 5.0%.

### 8.2 The consistency check, per book

`scripts/check_cash052.py` — **PASS**. NIFTYBEES is bit-identical in both curve files
(max absolute difference exactly 0.0). Every cash-holding book's **paired** yield effect is
tested against `(1 − invested) × 0.2 pp`, path index by path index, judged against the standard
error of the paired median:

| Book | Invested | Predicted | Paired median | Spread | SE | Verdict |
|---|---|---|---|---|---|---|
| True North | 43.0% | +0.114 pp | **+0.130 pp** | single path | — | CONSISTENT |
| Open Alpha · Base Age | 72.9% | +0.054 pp | **+0.060 pp** | −1.38 … +0.81 (30 seeds) | 0.088 | CONSISTENT |
| IPO Base | 31.8% | +0.136 pp | **+0.156 pp** | +0.15 … +0.16 (30 seeds) | 0.000 | CONSISTENT |
| Open Alpha · ATH + VIX | 79.0% | +0.042 pp | **−0.153 pp** | −2.22 … +2.20 (30 seeds) | 0.206 | CONSISTENT (inside 1 SE) |
| Quality Summit | 91.2% | +0.018 pp | **+0.013 pp** | −0.41 … +0.63 (12 offsets) | 0.106 | CONSISTENT |

The ordering is exactly what the arithmetic demands: the *least* invested book gains the most
(IPO Base at 32% invested, +0.16) and the *most* invested gains least (Quality Summit at 91%,
+0.01). NIFTYBEES, fully invested, gains nothing.

### 8.3 The one real finding of this run: single-path re-draw swamps the cash effect

**A 20 bps change in the cash rate changes the cash balance, which changes INTEGER SHARE
COUNTS, which changes whether a given buy is affordable, which re-draws every later selection
in that path.** That re-draw is worth up to ±2 points of CAGR on a single path — an order of
magnitude more than the 0.02–0.16 points the cash rate itself is worth.

| Book | Cash rate worth | Single-path move seen | Ratio |
|---|---|---|---|
| Open Alpha · ATH + VIX | +0.04 pp | **+1.20 pp** (drawn seed 30) | 30× |
| Quality Summit | +0.02 pp | **−0.27 pp** (offset3) | 14×, and the wrong sign |
| Open Alpha · Base Age | +0.05 pp | +0.06 pp (drawn seed 29) | 1× — this seed happened to be stable |
| IPO Base | +0.14 pp | +0.16 pp | 1× — this engine has no contested-slot draw |
| True North | +0.11 pp | +0.13 pp | 1× — single deterministic path |

**Two things were done about it, and both are stated on the page.**

1. **The drawn seed is FROZEN** at the one the 5.0% page drew — seed 29 for Base Age, seed 30
   for ATH + VIX — so the curve files differ by the cash rate and by nothing else wherever that
   is achievable. Left to its own rule the 5.2% run would have drawn seed 16 and seed 18, and
   the published tables would have moved by path noise. The house convention is preserved in
   the sense that the published path *was* the median-CAGR path of its ensemble; the
   generator's basis string now says "held at the seed the 5.0% page drew" rather than claiming
   it is this run's median. Quality Summit's drawn offset is picked by the generator from the
   twelve and landed on `offset3` at both rates anyway.
2. **The consistency test is PAIRED across the whole ensemble**, never a difference of two
   medians, and the ATH + VIX row now publishes its 30-seed band plus a note telling the reader
   to read the band rather than the point.

This is why the Quality Summit row *falls* 0.27 points on a page whose cash rate went **up**.
It is not an error and it is not hidden: it is offset3 re-drawing, and the ensemble it comes
from moved +0.013, as predicted.

### 8.4 What is NOT at 5.2% — stated, not hidden

The entry-mechanic surface, the null control and the two gate bake-offs in the "correction"
section are their own 30-seed re-runs of the Open Alpha engine (70 rows), not slices of the
curve files. `scripts/aftertax_all_052.py` — research/159's `aftertax_all.py` with `CASH_Y`
moved to 0.052 and its output redirected into research/163 — was launched, but it runs at about
3.3 minutes a row, roughly **4 hours** for the table. The page was not held for it.

The generator therefore **picks the 5.2% file up automatically once it is complete** and, until
then, uses research/159's 5.0% tables **and labels the section with the rate it actually used**,
in `NOTES['aftertax_incomplete']`. Those rows are 30-seed medians of a book about 79% invested,
so the cash rate is worth ~0.04 pp to every row alike and the ORDERING — the only thing that
section exists for — is unaffected. To finish it:

```bash
cd /home/arun/quantifyd && setsid nohup nice -n 10 venv/bin/python3 \
  research/163_mpf_cash_yield_harmonisation/scripts/aftertax_all_052.py \
  > /tmp/at052.log 2>&1 < /dev/null &
# it resumes from whatever is already in the CSV; then regenerate:
venv/bin/python3 research/_utilities/mpf_report_build.py
export PATH=$HOME/.nvm/versions/node/v20.20.2/bin:$PATH; cd frontend && npm run build
```

### 8.5 Runtime

| Step | Wall clock | Of which |
|---|---|---|
| True North | **25 s** | 24 s panel build |
| Open Alpha · Base Age | **10 s** | warm `panel163.pkl` cache |
| IPO Base | **53 s** | 46 s `ipo_replay.Ctx()` panel |
| Open Alpha · ATH + VIX | **63 s** | 44 s `em.load_frames` |
| Quality Summit | **10 s** | frozen `panel_2000.npz` |
| `build_inputs_052.py` + `check_cash052.py` | ~10 s | |
| generator (JSON + 11 PNGs) | ~90 s | |
| `npm run build` | ~90 s | |
| **the five re-runs together** | **~3 minutes** | 206 simulations |

### 8.6 The review that was registered

`research/111_sensex_manual_mgmt/scripts/ops_center.py` REVIEWS, **top of the list**, dated
**2026-12-15**, status **PENDING**: *"Momentum Portfolio - idle cash instrument: pick the
arbitrage fund, add the liquid-ETF buffer, measure the realised post-tax yield."* It renders at
`/app/straddles#ops-center` (68 reviews, this one first) and is mirrored in
`docs/LABS_AND_JOBS_REFERENCE.md`.

Its task **(0)** is **owed by Arun and is OPERATIONAL, not a model change**: move True North's
idle cash out of its current liquid instrument into an **arbitrage fund**, keeping a liquid-ETF
buffer sized for the gate's re-entry. True North's 100-SMA weekly gate **liquidates the whole
book** and then re-buys 8 names, so the buffer must cover a full re-entry within **T+1** of a
redemption, or the redemption must be placed the day the gate signals. The fund chosen and the
date go in the Capital Desk / True North dashboard note. **No executor is touched.**

PASS = the report's cash line reads a **measured** number with its source named. If the measured
rate differs from 5.2% by more than **0.5 points**, re-run the curves with these scripts.
