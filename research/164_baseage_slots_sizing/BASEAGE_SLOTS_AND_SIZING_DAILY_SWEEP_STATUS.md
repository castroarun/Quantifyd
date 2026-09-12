# Open Alpha · Base Age — How Many Slots, At What Size, And Who Wins A Contested Slot?

**STATUS: DONE — 12-Sep-2026 20:1x IST. Verdict CONCLUDED: the inherited 16 slots at 6.25% survives; no spec change before the 26-Sep paper-book call.**

research # 164 · daily · NSE cash · started 12-Sep-2026 evening IST
Folder `research/164_baseage_slots_sizing/` · host VPS `94.136.185.54` · `/home/arun/quantifyd`

---

## 1. Headline

**OA — Open Alpha · Base Age.** The book's SIXTEEN slots at 6.25% of NAV were never
tested. They were INHERITED from the old Open Alpha book, whose 680-cell sweep
(research/142) was scored entirely against a **same-bar look-ahead entry**. research/158
and research/159 then showed those parameter surfaces **invert** once the entry is made
placeable: every placeable entry improves as the trail lengthens, while the look-ahead
entry degrades. So 16 × 6.25% is a setting fitted on a surface we now know was the wrong
one — and research/161, which swept age, depth, volume, saucer shape, exits and hard
stops, has **no slot column at all** in `results/sweep161.csv`. That is the gap this
study closes, three weeks before the 26-Sep-2026 paper-book call.

Nothing here is deployed or papered without Arun's say-so.

---

## 2. The Ask

**What Arun asked (12-Sep-2026):** the sixteen-slot, 6.25%-per-slot book was never tested
for Base Age — run it properly. How many slots, at what size, does a cash buffer help now
that idle cash pays 5%, and who should win a contested slot?

**What we are actually testing.** Holding research/161's adopted entry and exit spec
COMPLETELY FIXED, does any (slot count, position size, contested-slot rule) combination
beat the inherited 16 × 6.25% × random-draw on after-tax Calmar over 2005-01 → 2026-09,
paired across the same 30 seeds, in BOTH the fit and the holdout window, on a plateau,
surviving a 40 bps cost rung and the tradeability gate? And, separately and honestly:
**how often does slot contention actually bind at all?** If more qualifying signals than
free slots is a rare event, the slot count cannot matter much, and that is itself the
answer to the question.

Four questions to be answered in Arun's house Q&A format in `results/RESULTS.md`:
how many slots; at what size per slot; does a cash buffer earn its place at 5%; does
anything beat a random draw for a contested slot; should the adopted spec change before
26-Sep.

---

## 3. The Base — what is being tested

### 3.1 The book, unchanged from research/161's adopted spec

| Element | Setting |
|---|---|
| Universe | NSE dailies, funds/ETFs/index proxies excluded, `SILLYMONKS` dropped (duplicate series) |
| Liquidity | 20-day median traded value ≥ ₹2 cr at the trigger (causal, shifted back one bar) |
| Entry signal | first CLOSE above the prior all-time-high close, where that prior high is **≥ 60 trading bars old** AND the stock fell **≥ 20% below it** in between |
| Volume filter | **none** (research/161: volume confirmation HURTS this book) |
| Saucer/shape filter | **none** |
| Re-arm | 60 bars per symbol, applied AFTER the filters |
| Exit | **SuperTrend(14,4) close trail**, no hard stop, no time stop |
| Fills | signal on the CLOSE, filled at the **NEXT day's OPEN**, on BOTH legs |
| Costs | 25 bps a side (ladder 40 / 60 on the shortlist) |
| Capital | ₹10,00,000 |
| Tax | after-tax: 20% STCG, 12.5% LTCG beyond 365 days, Indian FY loss netting with carry-forward |
| **Idle cash** | **5.0% post-tax, credited DAILY** — NOT research/161's 5.5%; Arun standardised the whole Momentum Portfolio on 5% on 12-Sep-2026 |
| Window | 2005-01-03 → 2026-09-11 (the calendar is cut at research/161's own last date so the terminal liquidation and last partial-FY tax settlement land where they did) |
| Seeds | 30 (`numpy.random.default_rng(1..30)`), medians reported with the range AND the worst seed |

**Cash-accrual audit (asked for explicitly).** In `bt_core.simulate` the carry is a single
line at the end of each day, `cash *= (1 + daily_yield)` with
`daily_yield = (1+iy)**(1/252) - 1`. It is therefore (a) credited **daily**, on the cash
balance only, never on the market value of open positions; (b) compounded; and (c)
**never routed through the tax settlement** — the FY block only ever nets realised trade
P&L (`fy_st`, `fy_lt`), so the yield is treated as an already-post-tax rate by assumption.
That is the correct reading of "5.0% post-tax". Confirmed by inspection, unchanged in
`sim164.py`.

### 3.2 The three things this study is allowed to vary — and nothing else

1. `slots` — how many concurrent positions the book may hold.
2. `slot_pct` — the fraction of NAV a new position is sized at, measured on the NAV as of
   that morning, before that day's fills.
3. `select` — which candidate wins when more qualifying signals fire on one day than there
   are free slots.

`sim164.py` is `bt_core.simulate` copied verbatim with exactly these three parameters made
live plus two pieces of bookkeeping that change no rule: the daily invested fraction, and
the contention accounting. The `select='random'` branch is byte-identical to research/161
— the rng is drawn only on binding days, in the same order, with the same call — so the
incumbent path must reproduce research/161 exactly. That is the harness proof in §5.

### 3.3 Contested-slot rules (axis D)

All four ranked rules use attributes known on the **trigger close**, before the fill:

| Rule | Key | Winner |
|---|---|---|
| `random` | — | seeded draw. **The incumbent AND the null control.** |
| `rs` | `rs252` = 12-month price return to the trigger close | strongest (IBD-style relative strength) |
| `ext` | `ext_pct` = 100·(entry open / prior ATH − 1) | **smallest** — the least extended above the prior high |
| `tv` | `tv20_cr` | largest 20-day median traded value |
| `age` | `x_bars` | longest base (oldest prior ATH) |

A name with under 12 months of usable history ranks LAST on `rs` rather than being
dropped — the book still owns the event, only its priority changes.

This axis is genuinely open: research/160 found relative-strength ranking was the whole
engine for a different near-high book (+7.7pp), while research/158 found RS selection no
better than random for the old Open Alpha. A ranked rule that cannot beat the random draw
is noise.

### 3.4 Success criterion

**Ranking metric:** after-tax **Calmar over the full window**, subject to after-tax CAGR
≥ the 16-slot baseline's, paired across the SAME 30 seeds.

---

## 4. Plan — the grid, the windows, and the adoption bar

### 4.1 Windows (pre-registered, opened once)

| Window | Dates | Role |
|---|---|---|
| FULL | 2005-01-03 → 2026-09-11 | ranking metric, paired test |
| **W1 (FIT)** | 2005-01-03 → 2015-12-31 | selection decisions are made here |
| **W2 (HOLDOUT)** | 2016-01-01 → 2026-09-11 | **opened ONCE at the end** |

A cell whose **W2 CAGR falls more than 4pp below its W1 CAGR is declared NOT ROBUST**,
exactly as in research/162, which caught an overfit on this rule last night. Window
drawdowns are measured from the running peak of the **FULL** curve, never the window's
first bar (the research/154 retraction).

### 4.2 The axes

| Axis | What it isolates | Grid |
|---|---|---|
| **A** Concentration, fully invested | `slot_pct = 1/slots`, so a full book is ~100% invested | slots ∈ {6, 8, 10, 12, 16, 20, 24, 30} — 8 cells |
| **B** Fixed size, deliberate cash buffer | `slot_pct = 0.0625` fixed; max invested = 50 / 62.5 / 75 / 87.5 / 100% | slots ∈ {8, 10, 12, 14, 16} — 5 cells (16 ≡ A16) |
| **C** Size independent of slot count | separates "how many names" from "how big each bet" | slots ∈ {10, 16, 20} × `slot_pct` ∈ {0.04, 0.05, 0.0625, 0.08, 0.10}, dropping anything above 100% invested — 10 cells |
| **D** Who wins a contested slot | coupled to A (fully invested at 1/slots); `random` at the same slot count is the null | rule ∈ {rs, ext, tv, age} × slots ∈ {8, 16} — 8 cells |

**Never above 100% invested — no leverage anywhere.**

Folding the duplicates (A16 ≡ B16 ≡ C16@0.0625 ≡ the incumbent; A10 ≡ C10@0.10;
A20 ≡ C20@0.05; B10 ≡ C10@0.0625) leaves **26 distinct simulated cells**:
8 (A) + 4 (B) + 6 (C) + 8 (D). Budget was ≤ 90. The cost-ladder and window re-runs on the
shortlist are re-scorings of already-selected cells, not additional selection cells, and
are disclosed separately.

**Every cell also records** `days_signal`, `days_bind` (days where qualifying signals
exceeded free slots, including days the book was full), `days_full`, and `turned_away`
(signals refused for want of a slot).

### 4.3 Adoption bar — strict, because this is a real spec change to a candidate book

To change the slot count or the size, a cell must, paired across the SAME 30 seeds:

1. beat 16 × 6.25% by **≥ +0.10 Calmar OR ≥ +2pp CAGR at no worse drawdown**;
2. win on **≥ 20 of 30 seeds**;
3. do so in **BOTH windows** (and pass the ≤ 4pp W1→W2 degradation rule);
4. sit on a **plateau** — both neighbouring slot counts within ±3pp of CAGR;
5. survive the **40 bps** cost rung;
6. pass the **tradeability gate** — max losing streak, trades per year, and capacity
   (median position ₹ against the held names' own 20-day median traded value; **flag any
   cell whose median position exceeds 1%** of that).

Anything short of all six is reported as an observation, not an adoption.

### 4.4 Scan then confirm

Scan at 10 seeds; confirm everything shortlisted at 30. Both counts disclosed. Shortlist =
top 3 by the pre-registered metric, plus the incumbent.

---

## 5. Status (live log)

**Phase:** COMPLETE. All four axes run at 30 seeds, plus the cost ladder, the zero-yield attribution and the outlier deletion. Verdict written.

| Date/time IST | Event | Notes |
|---|---|---|
| 2026-09-12 18:47 | Environment checked | VPS load 1.42, another agent building the frontend for research/163 — capped at 2 workers, `nice -n 10`, nothing outside `research/164_*` |
| 2026-09-12 18:49 | Inputs copied | `ath_events.csv` + `curves161.npz` from research/161; `panel163.pkl` → `panel164.pkl` (own copy; research/161 kept no panel on disk) |
| 2026-09-12 18:5x | Sections 1-4 frozen | this document, before any sweep cell |
| 2026-09-12 18:53 | Event list frozen | 3,619 adopted-spec events over 1,880 signal days — the SAME count research/163 found independently. Max simultaneous signals on any one day = **13**, median 1, so no day ever carries 16 |
| 2026-09-12 18:53 | **HARNESS PROOF PASSED** | at 5.5% idle cash: 21.26% CAGR / worst seed 19.87% / −34.80% DD / Calmar 0.618 — research/161's published winner to the last digit. At 5.0%: 20.94% / 19.81% / −35.50% / 0.601, invested 72.9% — matches research/163's independent re-run exactly. Cash accrual audited: daily, cash-only, compounded, never taxed |
| 2026-09-12 18:57 | Grid extended 26 → 32 cells | six extra slot counts (7, 9, 11, 13, 14, 18) added to execute the pre-registered PLATEAU test. A tightening of the same axis, not a new one — disclosed in RESULTS.md and in the published caveats |
| 2026-09-12 18:58 | Engine diagnostic added | the first run showed contention behaving oddly across axes, so the engine now also counts entries refused for want of CASH (slot free, no money). This turned out to be the study's biggest finding |
| 2026-09-12 18:59 | Full grid done | 32 cells × 30 seeds in 75s (2 workers, nice 10) |
| 2026-09-12 19:00 | Zero-idle-yield re-run done | 32 cells × 10 seeds — isolates each cell's cash-sleeve contribution |
| 2026-09-12 19:02 | Cost ladder done | 32 cells × 30 seeds at 40 bps and again at 60 bps |
| 2026-09-12 19:05 | Outlier test done | research/161's `outlier_all` product-of-returns reads 1e20× and is not a book multiple; replaced with (a) the ten best trades' share of total RUPEE profit and (b) a fair re-run with those ten EVENTS deleted from the event list across all 30 seeds |
| 2026-09-12 19:1x | RESULTS.md, PUBLISH_NOTE.md written | frontend deliberately NOT touched — another agent held it |
| 2026-09-12 20:1x | **DONE** | verdict CONCLUDED; INDEX, TODO and the Ops & Review Center updated; committed |

---

## 6. Crash Recovery — how to resume without Claude

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/164_baseage_slots_sizing

# what finished?
tail -5 results/cells_scan.csv ; wc -l results/cells_*.csv results/seedstats_*.csv
ls -la results/navs_*/

# still running?
pgrep -af run164.py

# resume — already-finished cells are skipped automatically (the runner reads
# results/cells_<stage>.csv on start)
cd scripts
nohup nice -n 10 /home/arun/quantifyd/venv/bin/python3 run164.py --stage=scan --seeds=10 \
      --workers=2 >> ../results/scan.log 2>&1 &
nohup nice -n 10 /home/arun/quantifyd/venv/bin/python3 run164.py --stage=full --seeds=30 \
      --workers=2 >> ../results/full.log 2>&1 &

# rebuild the event list from scratch if events164.csv is lost (needs panel164.pkl):
/home/arun/quantifyd/venv/bin/python3 build164.py
# and if panel164.pkl is lost, re-copy it:
cp ../../163_mpf_cash_yield_harmonisation/results/panel163.pkl ../results/panel164.pkl
```

**Do NOT touch:** anything under `frontend/`, `static/app/`, `services/`,
`research/161_*`, `research/162_*`, `research/163_*`, `research/_utilities/`, or any live
DB. Do NOT restart `quantifyd`. **Safe to inspect:** everything under
`research/164_baseage_slots_sizing/`.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `BASEAGE_SLOTS_AND_SIZING_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/sim164.py` | the engine — research/161 `bt_core.simulate` + slots / slot_pct / select | yes |
| `scripts/build164.py` | freezes the adopted-spec event list with the four ranking attributes | yes |
| `scripts/run164.py` | the cell runner (incremental CSV, resumable, 2 fork workers) | yes |
| `scripts/finalize164.py` | paired tests, windows, cost ladder, YoY, capacity | yes |
| `results/events164.csv` | the frozen event list (small) | yes |
| `results/cells_*.csv` | one summary row per cell | yes |
| `results/seedstats_*.csv` | one row per (cell, seed) — the paired-test input | yes |
| `results/RESULTS.md` | verdict + Q&A | yes |
| `results/PUBLISH_NOTE.md` | the `BacktestStudy` entry to publish once the frontend is free | yes |
| `results/panel164.pkl` | 138 MB price panel | **NO — gitignored** |
| `results/ath_events.csv` | 8 MB raw event dump copied from research/161 | **NO — gitignored** |
| `results/navs_*/` | per-cell 30-seed NAV arrays | **NO — gitignored** |
| `results/*.log` | run logs | **NO — gitignored** |

---

## 8. Findings

**Verdict: CONCLUDED — the inherited 16 slots at 6.25% survives. No spec change before
26-Sep-2026.** Full write-up and the house YoY table in `results/RESULTS.md`.

1. **The harness is the research/161 book.** Reproduced its published winner exactly at 5.5%
   idle cash (21.26 / −34.80 / 0.618, worst seed 19.87) and research/163's independent 5.0%
   re-run exactly (20.94 / −35.50 / 0.601, worst 19.81, 72.9% invested). 3,619 events, the
   same count research/163 found.
2. **16 slots is not the optimum, but nothing clears the bar.** Calmar has a genuine broad
   hump at 9-12 slots (0.629 / 0.669 / 0.670 / 0.633 vs the incumbent 0.601), worth +1.1 to
   +1.4pp of after-tax CAGR, winning 25-27 of 30 paired seeds and 29-30 of 30 in the fit
   window, on a plateau, surviving 40 and 60 bps. It is below the pre-registered +0.10 Calmar
   / +2pp CAGR bar. 0 of 32 cells clear it.
3. **The bar's blind spots argue the other way.** Concentrating 16 → 10 slots raises the share
   of total profit from the ten best trades from 35.7% to 53.4% and roughly doubles the
   capacity footprint (median position 0.43% → 0.77% of the held name's own 20-day traded
   value; 33% → 46% of trades above 1%; ten times larger again on a ₹1 crore book).
4. **Size per slot is a leverage dial, not hidden alpha.** With the slot count fixed, shrinking
   the position only de-levers along one line. Not one of the 11 eligible cells comes from
   axis B or axis C.
5. **A deliberate cash buffer does not earn its place at 5%.** The best buffered cell (8 slots
   at 6.25%, 45% invested) makes 15.46% / −23.69% / Calmar 0.647, while 11 slots FULLY
   invested gives a better ratio (0.670) AND 6.6pp more return. If Arun ever wants a
   low-drawdown Base Age variant, 10 slots at 4% (12.82% / −17.12% / 0.754) is the cleanest
   point on that frontier — but it is a different product, to be adopted as one.
6. **One contested-slot rule beats the random null consistently: take the most liquid
   candidate.** +2.60pp at 8 slots (30/30 seeds on CAGR and Calmar), +0.78pp at 16 (29/30 on
   both), in both windows. Relative strength wins at 8 slots and LOSES at 16 (5/30) — which
   settles research/160 vs research/158 in favour of research/158 for this book. Least-extended
   is flat; longest-base-age raises CAGR but deepens drawdown and wins Calmar on only 9 of 30.
   The liquidity rule does not clear the bar (+0.064 Calmar) and is 1 of 8 cells on its axis,
   but it is free, deterministic — a live book cannot draw a seed, and the spec gives the
   operator no written tie-break today — and it helps capacity. Worth its own test.
7. **THE BIGGEST FINDING: cash, not slots, is the binding constraint.** At 16 slots, 3,619
   qualifying events produce 688 entries, 977 refused for want of a slot, and **1,955 refused
   for want of cash**. That holds at every slot count. The book never trims a winner, so a few
   bloated positions can absorb 95% of NAV while slots sit nominally free. Contention binds on
   472 of 1,880 signal days at 16 slots, but the book is completely full on only 41 of them.

**Recommended next steps, in order.**

1. **Position drift is the untested first-order knob.** Trim a bloated winner back toward its
   target weight, or size the next entry to available cash rather than skipping it entirely.
   Registered as a dated review for 2026-10-10 in the Ops & Review Center.
2. **Give the liquidity tie-break its own study** — it is the only free change on the table,
   and it removes path randomness a live book cannot reproduce anyway.
3. **Do not re-open the slot count** until (1) is answered. The slot curve measured here is
   the curve of a book that cannot fill its own slots.
