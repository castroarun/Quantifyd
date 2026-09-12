# IPO Base — Re-optimisation on the PLACEABLE Entry (next-day buy-stop at the pivot)

**STATUS: DONE — all ten stages complete, verdict filed**
Research number **167** (renumbered from 163 on 12-Sep-2026: a sibling session had already claimed 163 for the cash-yield harmonisation work, and 164-166 are taken too). Owner: the quant-researcher session dispatched 12-Sep-2026, which completed every stage and then hit its rate limit before filing the index row and the dated review — both now filed by the parent session.

**Verdict: the ADOPTED r/153 spec is NO EDGE when measured on an entry an order can place** (14.90% after tax, and it loses to its own date-matched random-entry null on 14 of 30 paired seeds). **Re-fitted it is a STRATEGY candidate** — trail SMA-50, stop 10%, target +25%, plus a NIFTYBEES<SMA-150 entry gate — 21.80% after tax, −26.6% median drawdown, Calmar 0.819, beating the null on 30 of 30. **Nothing deployed.** The adoption call is blocked on the 3-sleeve blend test against True North and OA Base Age, which was never run. Full findings: `results/RESULTS.md`.
Sibling sessions own research/159 (OA honest re-opt), 160, 161, 162 — **do not edit those**.

---

## 1. The Ask

**What Arun asked (via the dispatching session):** re-optimise IPO Base on an entry that can
actually be placed. Every IPO Base parameter was chosen against a look-ahead entry; Open
Alpha's surface did not merely shift when the entry was fixed, it **inverted**. Are IPO
Base's adopted settings pointing the wrong way too?

**What is actually being tested.** `research/153`'s engine (`ipo_replay.build_trigger` +
`simulate_ipo`) decides on day *i*'s CLOSE (`trig = setup & (C > piv)`) and fills at day
*i*'s OPEN (`fill = max(piv, O[i])`). That needs the close known at the open — the 8th
deadly sin (playbook §4, §5A). `services/ipo_paper.py` is **correct**: it triggers on
tonight's close and carries a buy-stop at the pivot into the **next** morning. So the live
book is honest and the study is not. r/158/r/159 measured the gap on the adopted spec:

| | CAGR (after tax) | MaxDD | Calmar |
|---|---|---|---|
| r/153 published (look-ahead entry) | 31.0% | −20.9% | 1.50 |
| same spec, placeable next-day entry | **15.0%** | −37.6% | 0.40 |

The question this study answers: **does IPO Base get better on the honest entry once its
parameters are re-fitted, and by how much?**

Not re-derived here (already established, do not redo): the defect itself, the replication
gate, the 98.5% signal-survival measurement.

---

## 2. The Base — what is being tested

### 2.1 Universe
- NSE equities with a **vetted listing date** (`research/153_ipo_base/results/listing_dates.csv`,
  1,353 accepted), all rows before the listing date masked.
- 20-day median traded value at *t−1* ≥ ₹5 cr.
- **FUNDS EXCLUDED BY LONG NAME**, `backtest_data/etf_exclusions.json` (346 symbols),
  unioned with r/142's old ticker regex. **This is a change from r/153**, which used the
  ticker regex only. 146 of the 1,353 accepted listings are funds and the old regex catches
  only 54 of them — IPO Base is the book most exposed to the r/158 fund-contamination defect,
  because ETFs are *always* newly listed. Both universes are measured (Stage 0) so the size
  of the contamination is reported, not assumed.

### 2.2 Signal (setup, evaluated on bar *i*)
- age: listed ≤ `max_age_m` months ago, and ≥ `min_bars` bars of history
- base: pivot = highest CLOSE of the last `L` bars, shifted 1; depth (pivot → lowest low of
  the same window) ≤ `max_depth`
- not already extended: `close[i−1] < pivot`
- RS policy: `off` (incumbent) / `short70` / `short80` / `relaxed`
- liquidity + fund mask as above

### 2.3 Entry mechanics — ALL placeable variants measured (playbook §5A, binding)

| id | Trigger | Fill | Placeable |
|---|---|---|---|
| `ref_lookahead` | `close[i] > piv[i]` | `max(piv[i], open[i])` | **NO — reference arm only, r/153's** |
| `close_fill` | `close[i] > piv[i]` | `close[i]` | yes (live process at ~15:10) |
| **`nextday_pivot`** | `close[i−1] > piv[i−1]` and `high[i] ≥ piv[i−1]` | `max(piv[i−1], open[i])` | **yes — THIS IS THE LIVE BOOK'S MECHANIC, the primary arm** |
| `nextday_candle` | `close[i−1] > piv[i−1]` and `high[i] > high[i−1]` | `max(high[i−1], open[i])` | yes |
| `resting_stop` | `high[i] ≥ piv[i]`, every crossing incl. failures | `max(piv[i], open[i])` | yes (survives a dead process) |

`nextday_pivot` is the transformation verified in
`research/159_oa_honest_reoptimization/scripts/ipo_curve.py`:
`TRIG[i] ← TRIG[i−1] AND high[i] ≥ PIV[i−1]`, `PIV[i] ← PIV[i−1]`. Reused verbatim.

### 2.4 Exits (all decided and filled on the CLOSE — already honest in r/153's engine)
priority: hard stop (`close ≤ fill×(1−stop)`) → target (`close ≥ fill×(1+target)`) →
trail (`close < SMA(trail)`, entry bar exempt).

### 2.5 Book
8 slots at 18.75% of equity (incumbent), 25 bps per side, no market gate, idle cash at 5%
p.a., **after tax throughout**: 20% STCG / 12.5% LTCG with Indian FY (1-April) loss netting
and carry-forward. NOTE the incumbent is cash-constrained, not slot-constrained:
8 × 18.75% = 150% of NAV, so cash binds at ~5 positions and the book sits only ~33%
invested. **Invested fraction and the cash-sweep contribution are reported on every table.**

### 2.6 Windows
- **W2 = 2006-01-01 → 2026-09-04** (headline, 20.7 years)
- **WA = 2006-01-01 → 2015-12-31** and **WB = 2016-01-01 → 2026-09-04** (two-window split,
  both must pass)
- W1 = 2020-01-01 → 2025-12-31 (r/153's / the source's window, for continuity only)

### 2.7 Pre-registered ranking metric and adoption threshold
**Rank by: median after-tax CAGR on W2 across 30 seeds**, on the `nextday_pivot` mechanic,
clean (fund-free) universe.

A cell is only *eligible* if all of:
1. per-trade expectancy net of 50 bps round trip > 0 in **both** WA and WB;
2. worst of the 30 seeds > 0% CAGR on W2;
3. its parameter neighbours agree — a cell whose neighbours on the same axis disagree by
   more than the seed band is declared noise, not a winner (plateau, not peak).

**"IPO Base gets better" is declared only if** the best plateau beats the honest incumbent
(15.0% W2 after tax) by **≥ +3.0pp median CAGR** with Calmar ≥ 0.40, **and** beats the
date-matched random-entry null by **≥ +3.0pp** after tax on the same fill convention.
Anything less is reported as "re-fitting does not rescue it".

---

## 3. Plan — staged, with cell counts

| Stage | What | Axes | Cells |
|---|---|---|---|
| **0** | Entry-mechanic bake-off + diagnostics at the incumbent spec | 5 mechanics × 2 universes (clean / r/153 contaminated) | **10** |
| **0b** | Split/corporate-action exposure diagnostic (r/153's engine has no guard; the live book treats a −40% day as a data event) | count trades with a one-day close move < −40%, and a guarded arm | **2** |
| **1** | **Exit economics** — where OA's inversion lived | trail {10,15,20,30,50,75} × target {+25%, +50%, none} × stop {6%, 8%, 10%, none} | **72** |
| **2a** | Base geometry on the Stage-1 plateau | age {3,6,12,24} × L {15,25,40,60} × depth {0.20,0.30,0.40,0.60} × RS {off, short70} | **256** |
| **2b** | Slots / sizing on the Stage-2 top geometry | (slots,size) ∈ {(5,.20),(8,.125),(8,.1875),(10,.10),(12,.0833),(16,.0625)} × top-3 | **18** |
| **3** | **Null controls**, same fill convention on both arms | name-selection null (open-fill both arms), structure null (full honest convention), cohort-drift null | **~8** |
| **4** | Gate bake-off — only if 1–3 leave something worth gating | NIFTYBEES<SMA200, index drawdown bands, momentum sign; INDIA VIX starts 2015 so VIX gates get their own window and their own baseline | **~12** |
| | **Total planned** | | **~378** |

All cells: 30 seeds, 3 windows (W2/WA/WB), after tax. Incremental resume-safe CSV, one row
per completed cell. **378 cells is disclosed for multiple-testing discount: at 30 seeds the
seed band is ±2–3pp, so a single cell beating the incumbent by less than the band is noise.**

Stage 2 is run only on the region Stage 1 identifies — no giant grid.

---

## 4. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-12 18:33 | Folder created, STATUS written | nothing launched |

### Live findings
_(none yet)_

---

## 5. Crash recovery — how to resume WITHOUT Claude

Everything is on the VPS at `/home/arun/quantifyd`, run with `venv/bin/python`.

**1. What finished?** Each stage appends one row per completed cell:
```
cd /home/arun/quantifyd/research/167_ipo_base_honest_reopt/results
wc -l stage0_mechanics.csv stage1_exits.csv stage2a_geometry.csv stage2b_book.csv stage3_nulls.csv stage4_gates.csv
tail -5 /tmp/r163_*.log
```

**2. Is it still alive?**
```
pgrep -af 'ipo_honest.py'
ls -l --time-style=full-iso /tmp/r163_*.log     # mtime should advance every few seconds
```

**3. Resume.** Every stage is resume-safe — it skips labels already present in its CSV.
Re-launch the same command; it picks up where it stopped:
```
cd /home/arun/quantifyd
setsid nohup venv/bin/python -u research/167_ipo_base_honest_reopt/scripts/ipo_honest.py all \
    > /tmp/r163_all.log 2>&1 < /dev/null &
```
Individual stages: replace `all` with `stage0`, `stage1`, `stage2a`, `stage2b`, `stage3`,
`stage4`, `report`.

**4. Aggregation only** (all cells done, report crashed):
```
venv/bin/python -u research/167_ipo_base_honest_reopt/scripts/ipo_honest.py report
```

**5. Do NOT touch:** `services/ipo_paper.py`, `backtest_data/ipo_paper_state.json`, the
crontab, `research/153/**`, `research/159/**`, `research/160-162/**`,
`frontend/src/data/backtests.ts`. This study is read-only against all of them. It imports
`research/153_ipo_base/scripts/ipo_replay.py` and `ipo_g3.py` but writes nothing there.

**6. Safe to inspect / delete and regenerate:** anything under
`research/167_ipo_base_honest_reopt/results/`.

---

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `IPO_BASE_HONEST_ENTRY_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/ipo_honest.py` | engine fork + all stages (library + CLI) | yes |
| `results/stage0_mechanics.csv` | entry-mechanic bake-off, both universes | yes |
| `results/stage1_exits.csv` | 72-cell exit surface | yes |
| `results/stage2a_geometry.csv` | 256-cell base geometry | yes |
| `results/stage2b_book.csv` | slots/sizing | yes |
| `results/stage3_nulls.csv` | null controls | yes |
| `results/stage4_gates.csv` | gate bake-off | yes |
| `results/diagnostics.json` | fund contamination, split events, invested fraction | yes |
| `results/RESULTS.md` | final verdict | yes |

---

## 7. Findings

_(written as they emerge)_
