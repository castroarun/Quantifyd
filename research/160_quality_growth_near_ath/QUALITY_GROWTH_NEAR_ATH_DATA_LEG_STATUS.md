# Quality-Growth Near All-Time-High — DATA LEG (point-in-time fundamentals panel) — STATUS: DONE

The fundamentals half of research/160. This leg builds the **point-in-time fundamental
panel and the eligibility masks** that the ENGINE leg's price-side backtest consumes.
It runs **no strategy backtest**. Its other job is to say honestly how much of the
universe Screener can even see, and whether the mechanical screen reproduces the names
Arun actually bought.

Sister documents: the ENGINE leg's STATUS in this same folder; the doctrine in
`research/QUANT_RESEARCH_PLAYBOOK.md` §5A and `.claude/agents/quant-researcher.md`.

---

## 1. The Ask

**What Arun asked (verbatim, his screener.in query):**

```
Sales growth 3Years > 20 AND Profit growth 3Years > 20 AND
Average return on equity 3Years > 15 AND Return on capital employed > 15 AND
Debt to equity <= 0.2 AND Current price >= 0.9 * High price all time AND
Market Capitalization > 1000
```

plus three manual steps he performs by hand afterwards: (a) a liquidity check,
(b) OPM% steady or rising across the years, (c) buy near the all-time high.

**What this leg is actually testing.** Can that screen be evaluated *as it would have been
knowable on the day*, across the whole NSE cash universe, back to 2015 — and what does the
data refuse to support?

Three questions, in order:

1. **Panel.** For every (month, symbol), what were the seven fundamental quantities the
   screen needs, computed only from fiscal years and quarters that had actually been filed
   by that date?
2. **Coverage and survivorship.** How many universe names does Screener not carry at all,
   how many of those are dead names, and how much *liquid* dead capital is therefore
   invisible to the fundamental leg? That number is the survivorship exposure of this study
   and it must be stated, not hinted at.
3. **Replication of Arun's own process.** Taking his real Zerodha holdings history, would
   the mechanical screen have picked those names on the day they first appear? A screen that
   cannot reproduce his discretionary picks is testing something other than his process, and
   the study must say which.

The price-side conditions — near-ATH, liquidity, entry mechanic, exits, sizing — belong to
the ENGINE leg and are **out of scope here**, except for one read-only use in task 7.

---

## 2. The Base — what is being computed, exactly

### 2.1 Point-in-time rule (the whole reason this leg exists)

- Indian fiscal years end **31-March**. Audited annuals are filed within about four months.
  A fiscal year ending `31-Mar-YYYY` is therefore treated as **usable from `01-Aug-YYYY`**
  (`lag_months = 4`, a script parameter so a 3-month variant can be rebuilt without a re-fetch).
- A **quarter** ending on date `Q` is treated as usable from `Q + 60 days`.
- At each monthly decision date, only years/quarters clearing that test are read. Nothing else.

### 2.2 Honest residuals, recorded not smoothed

- Screener serves figures **as they stand today**, not as first reported. Annual restatements
  are usually small, but this is not a true as-reported vintage and every filter built on it
  carries that much look-ahead.
- **Companies delisted since are absent from Screener.** The price DB deliberately keeps dead
  names; Screener does not. The fundamental leg therefore has a survivorship edge the price
  leg does not. Task 6 quantifies it in rupees of traded value rather than leaving it as a
  sentence.
- **market_data.db is not retroactively split-adjusted.** Pre-split rows keep the old price
  scale. That poisons `mcap_pit` (price × shares) and any ATH screen. Affected names are
  detected and reported, not silently used.

### 2.3 Fields, per (decision date, symbol)

| Field | Definition |
|---|---|
| `sales_g3` | 3-year CAGR of Sales, % — **undefined if the base year is ≤ 0** (a loss-to-profit swing is not a growth rate) |
| `profit_g3` | 3-year CAGR of Net Profit, % — same base rule |
| `roe_avg3` | mean of the last 3 usable FY ROE, ROE = net_profit / (equity_capital + reserves) |
| `roce_latest` | Screener's own `ROCE %` row — **never computed**: the page lumps liabilities and never splits current liabilities, so EBIT/(assets − CL) off it would be invention |
| `is_lender` | ROCE absent on every year while ROE is present → bank/NBFC. Judged on ROE alone; ROCE recorded n/a, **not failed** (failing them is a sector bet dressed as a quality filter) |
| `de_latest` | borrowings / (equity_capital + reserves) |
| `opm_latest`, `opm_slope3`, `opm_range3`, `opm_min3` | Screener's `OPM %` row: latest; OLS slope over the last 3 usable FY (pp/yr); max−min (pp); min (pp) |
| `opm_q_slope8`, `opm_q_std8` | same over the last 8 usable **quarters** |
| `neg3` | any negative Sales or Net Profit in the last 3 usable FY |
| `n_fy_usable` | count of filed fiscal years at that date — drives `has_data` |
| `shares_pit` | equity_capital / face_value, in crore shares — **point-in-time share count**, sanity-checked against today's mcap/price on ~20 large names |
| `mcap_pit` | `shares_pit × close(decision date)` in ₹cr, closes from market_data.db |

### 2.4 Missing-data policy — reported both ways, always

Any mask that needs data some names lack ships alongside `has_data`. The engine runs every
arm treating missing as **ineligible** and as **eligible**; the gap between the two is the
coverage bias. One number hides it. (Playbook §5A, binding.)

---

## 3. Plan

| # | Step | Script | Output |
|---|---|---|---|
| 1 | This STATUS, sections 1–4, before anything runs | — | this file |
| 2 | Universe: all day symbols − funds − stubs | `scripts/build_universe.py` | `results/universe.csv` |
| 3 | Screener fetch, resumable, 1 worker, ~2.5 s pacing, consolidated-first | `scripts/screener_fetch_full.py` | `results/screener_cache/<SYM>.json` |
| 4 | Point-in-time monthly panel, 2015-01 → 2026-09 | `scripts/build_pit_panel.py` | `results/features_pit_monthly.csv.gz` + README |
| 5 | Eligibility masks + index | `scripts/build_masks.py` | `results/masks/*.npz`, `results/masks/INDEX.csv` |
| 6 | Coverage & survivorship audit | `scripts/coverage_audit.py` | `results/coverage_audit.md` |
| 7 | Arun's holdings vs the screen (READ-ONLY) | `scripts/holdings_check.py` | `results/holdings_check.md` + `.csv` |
| 8 | Close out: STATUS → DONE, commit on VPS, sync laptop | — | — |

**Universe rule (step 2).** Every `timeframe='day'` symbol in `market_data.db`, minus
`backtest_data/etf_exclusions.json` (346 funds, built from the broker's long instrument
NAME — a ticker regex rots, see r/158 `etf_filter.py`) **and** the legacy ticker regex
`(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)` as a belt-and-braces second pass, minus symbols
with < 250 daily rows or no rows after 2015-01-01. `screener_ticker` strips the
`-BE/-SM/-BZ/-BL/-IT/-ST` series suffix; the mapping is kept in the CSV.

**Fetch order (step 3).** Descending `max_tv20`, so if the run is cut short the names that
could actually be traded are the ones already cached.

**Mask grid (step 5).** `has_data`; `arun_strict`; five leave-one-out variants; threshold
neighbours growth {15,20,25,30} × mcap {500,1000,2500}, de {0.2,0.5,1.0}, roe/roce {12,15,20};
four OPM variants stacked on strict; `growth_only`; `quality_only`. Every mask is False
wherever `has_data` is False.

**Pre-registered honesty checks.** (a) strict pass-count per month must land in the tens-to-
few-hundreds; a four-digit count means a bug, not an edge. (b) `shares_pit` must reconcile
with today's mcap/price within ~10% on large names or the face-value path is wrong.
(c) The earliest month at which ≥ 80% of active liquid names have `n_fy_usable ≥ 4` sets the
**honest study start**; anything earlier is a coverage artifact.

---

## 4. Contract — what the ENGINE and STUDY agents consume

### 4.1 Masks — `results/masks/<name>.npz`

Identical to the r/158 contract, so existing loader code works unchanged:

```python
z = np.load(path)
z['dates']   # <U10  ISO 'YYYY-MM-01', monthly, ascending
z['cols']    # symbol strings — market_data.db symbols, suffix INTACT (e.g. 'MODISONLTD-BE')
z['mask']    # bool, shape (len(dates), len(cols))
```

Decision dates are the **1st of each month**; the engine forward-fills a row until the next
date. `results/masks/INDEX.csv` lists every mask with its one-line definition and its
pass-rate by year.

**`has_data.npz` must be loaded with every other mask.** Run each arm twice: missing→ineligible
(`mask`) and missing→eligible (`mask | ~has_data`). Report both.

### 4.2 Panel — `results/features_pit_monthly.csv.gz`

Long format, one row per (date, symbol) that has any usable data:
`date, symbol, screener_ticker, n_fy_usable, fy_latest, sales_g3, profit_g3, roe_avg3,
roce_latest, is_lender, de_latest, opm_latest, opm_slope3, opm_range3, opm_min3,
opm_q_slope8, opm_q_std8, n_q_usable, neg3, face_value, shares_pit, close_pit, mcap_pit`.

**CSV.gz, not parquet.** The VPS venv is the live-trading venv and has neither `pyarrow` nor
`fastparquet`; installing into it to write a research file is not a trade worth making. Read
with `pd.read_csv(path, parse_dates=['date'])` — pandas decompresses by extension.

### 4.3 Universe — `results/universe.csv`

`symbol, screener_ticker, first_date, last_date, n_rows, max_tv20, active`.
`max_tv20` = max over history of the 20-day median traded value, ₹cr.
`active` = `last_date >= 2026-08-01`.

### 4.4 Not in this leg

Near-ATH distance, liquidity gating at decision time, entry/exit mechanics, slots, sizing,
costs, taxes. Those are the ENGINE leg's, and this leg deliberately does not pre-empt them.

---

## 5. Status log

| Time (IST) | Event | Notes |
|---|---|---|
| 2026-09-11 14:45 | Leg opened; doctrine read (quant-researcher agent, playbook §4–5A, r/158 STATUS + 3 scripts) | |
| 2026-09-11 14:55 | Sections 1–4 written before any run | this file |
| 2026-09-11 15:05 | `build_universe.py` run | 2,905 day symbols → **2,158 universe** (346 funds, 401 stubs removed); 2,050 active, 108 inactive; 44 `-BE` twin pairs share a Screener ticker |
| 2026-09-11 15:20 | Parser validated on a live RELIANCE page | 12 FY + 13 quarters + face value 10.0; parses by `data-date-key`, so the March-QUARTER column can no longer be read as a fiscal year (a latent defect in r/158's caption parser) |
| 2026-09-11 15:25 | **Screener fetch LAUNCHED detached** (pid on VPS, `nice -n 15`) | 2,116 distinct tickers, 3.3 s/name, ETA ~115 min → ~17:20 IST |
| 2026-09-11 15:35 | All four downstream scripts written and smoke-tested end-to-end on the partial cache | panel → masks → coverage → holdings all run clean; nothing waits on the fetch except the numbers |
| 2026-09-11 15:40 | **Early finding: `has_data` cannot start before Aug-2018** | Screener's free depth is ~12 FY, so the earliest year is FY2015 and four filed years do not exist until FY2018 is filed (Aug-2018). The honest study start is a data fact, not a choice |
| 2026-09-11 15:45 | **Early finding: the screen is a non-financials screen, and D/E is what does it** | Screener's bank/NBFC balance sheets carry no `Borrowings` row (debt sits in Other Liabilities), so `de_latest` is NaN for every lender and D/E ≤ 0.2 removes the entire sector. The lender ROCE carve-out is therefore cosmetic |
| 2026-09-11 15:50 | Share-count reconciliation passed the pre-registered check | `equity_capital / face_value × price` vs Screener's own market cap: median error 0.1%, 95% of names within 10% |
| 2026-09-11 16:05 | Fetch 22% (475/2,116), 468 full / 3 thin / 4 missing | ETA ~94 min. Miss rate is low because the queue is liquidity-ordered — the illiquid tail will miss far more |
| 2026-09-11 16:15 | **Engine leg reported `arun_strict` passing 0.081% of name-months. Cause found: it had loaded the SMOKE-TEST masks**, built mid-fetch from 96 of 2,116 tickers on the full 2,158-symbol column axis | Every un-fetched symbol is False in `has_data` and in every mask, so the rate read ~20× low. `masks/PRELIMINARY_DO_NOT_USE.txt` written; `build_masks.py` now emits `masks/PROVENANCE.json` with `fetch_complete`, prints the last-24-month pass counts against `has_data` (not the column count), and deletes the flag only on a complete build |
| 2026-09-11 16:25 | **Scale check passed at BOTH ends of the size range** (`today_reconcile.py`, 643 tickers) | mcap: median \|error\| 0.34%, 95.6% within 10%, 71.7% within 1%; smallest-20 names agree to ±3%. ROCE vs Screener's published row: median 0.2 pp, 100% within 5 pp — the right row is being read. No lakh/crore slip |
| 2026-09-11 16:25 | **Pass rate is ~4.8%, not 0.08%** | On today's figures, 31 of 643 screenable names clear all five criteria — i.e. tens of names, which is what Arun sees on the live site. Scaled to the full universe that projects to roughly 60–100 |
| 2026-09-11 17:07 | **Screener fetch DONE** — 2,116 / 2,116 in 137 min | 2,075 full, 14 thin, 27 missing |
| 2026-09-11 18:25 | Panel rebuilt on the complete cache | 256,828 rows x 2,131 symbols x 141 months, 3.7 MB gz |
| 2026-09-11 18:25 | Masks rebuilt; `PROVENANCE.json` reads `fetch_complete: true`; `PRELIMINARY_DO_NOT_USE.txt` deleted | 31 masks. **`arun_strict` passes 46-107 names/month over the last 24 months, not 0.3** |
| 2026-09-11 18:27 | Coverage audit — **the inherited survivorship caveat is refuted** | Screener keeps delisted pages: 2,131 of 2,158 covered; exactly one stopped-and-liquid name is missing, and it turns out to be a demerger rather than a death |
| 2026-09-11 18:27 | Today-reconciliation | PIT at 2026-09-01: **46 names**. Today's figures: **46**. Overlap 45 |
| 2026-09-11 18:29 | Holdings check | the mechanical screen reproduces **8 of the 69** names he actually holds |
| 2026-09-11 18:40 | STATUS to DONE, committed on the VPS, laptop synced | |

---

## 6. Crash Recovery

Everything here runs on the VPS at `/home/arun/quantifyd/research/160_quality_growth_near_ath`.
Nothing in this leg writes to `market_data.db`, `holdings_snapshots.db`, any live state file,
or anything under `services/`. There is no restart and no crontab change to undo.

**Is the fetch still alive?**

```bash
ssh arun@94.136.185.54 'pgrep -af screener_fetch_full'
ssh arun@94.136.185.54 'tail -3 /home/arun/quantifyd/research/160_quality_growth_near_ath/results/screener_fetch.log'
ssh arun@94.136.185.54 'ls /home/arun/quantifyd/research/160_quality_growth_near_ath/results/screener_cache | wc -l'
```

The cache count is the true progress: one JSON per ticker, written the moment it is parsed.
2,116 files means done.

**Resume it** — the script skips whatever is already cached, so re-launching is always safe:

```bash
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && setsid nohup nice -n 15 venv/bin/python3   research/160_quality_growth_near_ath/scripts/screener_fetch_full.py   >> research/160_quality_growth_near_ath/results/screener_fetch.log 2>&1 < /dev/null &'
```

**Then, in order** (each is a single command, each overwrites its own outputs only):

```bash
cd /home/arun/quantifyd
venv/bin/python3 research/160_quality_growth_near_ath/scripts/build_pit_panel.py
venv/bin/python3 research/160_quality_growth_near_ath/scripts/build_masks.py
venv/bin/python3 research/160_quality_growth_near_ath/scripts/coverage_audit.py
venv/bin/python3 research/160_quality_growth_near_ath/scripts/holdings_check.py
```

`build_universe.py` only needs re-running if `market_data.db` has changed.

**Safe to inspect, never to edit:** `backtest_data/market_data.db`,
`backtest_data/holdings_snapshots.db`, `backtest_data/etf_exclusions.json`, anything under
`research/158_oa_arming_width/`.

---

## 7. Files

All paths relative to `research/160_quality_growth_near_ath/`. The ENGINE leg writes into the
same `results/` directory; only the rows below belong to this leg.

| File | Purpose | Committed? |
|---|---|---|
| `QUALITY_GROWTH_NEAR_ATH_DATA_LEG_STATUS.md` | this file | yes |
| `.gitignore` | keeps the cache and the heavy panels out of the repo | yes |
| `scripts/build_universe.py` | universe from market_data.db, minus funds and stubs | yes |
| `scripts/screener_fetch_full.py` | the fetch: annual + quarterly + top-ratios, resumable | yes |
| `scripts/build_pit_panel.py` | the point-in-time panel (`--lag-months` rebuilds a variant) | yes |
| `scripts/build_masks.py` | 31 eligibility masks + INDEX + PROVENANCE | yes |
| `scripts/coverage_audit.py` | coverage, survivorship, FY depth, honest start | yes |
| `scripts/holdings_check.py` | his real book vs the screen (read-only on the holdings DB) | yes |
| `scripts/today_reconcile.py` | scale check + PIT-vs-today reconciliation | yes |
| `results/universe.csv` | 2,158 symbols: ticker map, dates, peak tv20, active flag (117 KB) | **no** — repo `.gitignore:167` excludes `research/**/results/*.csv`; rebuild in ~60 s with `build_universe.py` |
| `results/features_pit_monthly.csv.gz` | **the panel** — 256,828 rows (3.7 MB) | yes |
| `results/features_pit_monthly_README.md` | the panel's column contract and its four limits | yes |
| `results/masks/*.npz` (31) + `INDEX.csv` + `PROVENANCE.json` | **the masks** | yes |
| `results/coverage_audit.md` | the survivorship numbers | yes (`coverage_by_month.csv` beside it is CSV-ignored) |
| `results/holdings_check.md` | the replication gate on his own book | yes (the `.csv` is CSV-ignored) |
| `results/today_reconcile.md` | reconciliation against the live site | yes (the `.csv` is CSV-ignored) |
| `results/screener_fetch.log` | the fetch's own progress log | **no** — `research/**/results/*.log` is ignored repo-wide; the key lines are in §5 above |
| `results/screener_cache/*.json` (2,116 files, 17 MB) | raw parsed Screener pages | **NO — gitignored**; rebuild with `screener_fetch_full.py`, ~2¼ h |
| `results/panel_*.npz` (791 MB, ENGINE leg) | price panel | **NO — gitignored** |

Every CSV and log under `results/` is excluded by the repo's own policy (`.gitignore` lines
167-168) and every one of them is regenerated in under a minute by a committed script; the
panel ships as `.csv.gz`, which that rule does not match. Nothing this leg commits exceeds 4 MB. Nothing it wrote touches `market_data.db`,
`holdings_snapshots.db`, any live state file, `services/`, the crontab, or the running service.

---

## 8. Findings

### 8.1 The masks are not starved — the earlier 0.081% was a smoke-test artifact

The engine leg loaded `masks/arun_strict.npz` while the fetch was still running, when 96 of
2,116 tickers were cached against a full 2,158-symbol column axis: every un-fetched symbol is
False in `has_data` and in every mask, so the rate read about twenty times too low. On the
complete build:

**Names passing `arun_strict` per month, last 24 months.** The denominator is `has_data` — the
names that can be screened at all — never the 2,158 columns.

| month | has_data | strict | no_mcap | no_growth | no_de | no_roe |
|---|---:|---:|---:|---:|---:|---:|
| 2024-10 | 2102 | 104 | 126 | 263 | 191 | 129 |
| 2024-11 | 2102 | 107 | 127 | 267 | 196 | 130 |
| 2024-12 | 2102 | 105 | 127 | 265 | 200 | 127 |
| 2025-01 | 2102 | 106 | 127 | 267 | 203 | 129 |
| 2025-02 | 2102 | 103 | 127 | 266 | 193 | 124 |
| 2025-03 | 2102 | 100 | 127 | 262 | 189 | 121 |
| 2025-04 | 2102 | 101 | 127 | 264 | 191 | 121 |
| 2025-05 | 2102 | 102 | 126 | 270 | 192 | 123 |
| 2025-06 | 2102 | 102 | 126 | 272 | 193 | 124 |
| 2025-07 | 2102 | 105 | 126 | 275 | 198 | 127 |
| 2025-08 | 2116 | **74** | 83 | 279 | 146 | 93 |
| 2025-09 | 2116 | 75 | 83 | 278 | 151 | 94 |
| 2025-10 | 2116 | 74 | 83 | 276 | 149 | 92 |
| 2025-11 | 2116 | 73 | 82 | 275 | 148 | 92 |
| 2025-12 | 2116 | 72 | 82 | 273 | 145 | 91 |
| 2026-01 | 2116 | 72 | 82 | 273 | 145 | 89 |
| 2026-02 | 2116 | 71 | 82 | 268 | 142 | 87 |
| 2026-03 | 2116 | 70 | 82 | 268 | 138 | 87 |
| 2026-04 | 2116 | 67 | 82 | 260 | 134 | 83 |
| 2026-05 | 2117 | 68 | 80 | 269 | 138 | 85 |
| 2026-06 | 2117 | 69 | 80 | 269 | 136 | 86 |
| 2026-07 | 2117 | 69 | 80 | 274 | 137 | 86 |
| 2026-08 | 2126 | **48** | 52 | 242 | 103 | 66 |
| 2026-09 | 2126 | 46 | 52 | 239 | 101 | 64 |

**Mean names passing per month, by year:** 2018 · 5 | 2019 · 13 | 2020 · 12 | 2021 · 12 |
2022 · 21 | 2023 · 37 | 2024 · 70 | 2025 · 91 | 2026 · 64. `has_data` over the same years runs
670 → 1,486 → 1,575 → 1,667 → 1,784 → 1,922 → 2,045 → 2,108 → 2,119.

Two things to read in that table:

- **The step-downs at 2025-08 and 2026-08 are the filing lag working, not a break.** Every
  1-August a new fiscal year becomes usable and the three-year growth window rolls forward,
  dropping names whose newest year is weaker than the one it replaced. A point-in-time screen
  is supposed to do that; a today-figures screen cannot.
- **Growth is the binding criterion and the market-cap floor is nearly inert.** At 2026-09,
  dropping the mcap floor moves 46 → 52; dropping growth moves 46 → 239. `no_roce` is 46, the
  same as strict, so — exactly as r/158 found on its 638 names — **ROCE never rejects a name
  that ROE has not already rejected.**

**The near-ATH condition is deliberately absent from every count above.** It belongs to the
ENGINE leg and it cuts these lists further; nothing here is a candidate count for the study.

### 8.2 Scale and reconciliation — the panel agrees with the live site

- **Market cap is on the right scale at both ends of the size range.** `equity_capital (₹cr) /
  face_value (₹)` is a share count in crores; times a rupee price it is ₹cr. Against Screener's
  own market cap on 2,083 names: median |error| **0.72%**, 95.6% within 10%. The twenty largest
  agree to ±0.3% apart from LICI (−50%) and M&M (−10%), both second-share-class cases, and the
  twenty smallest agree to ±3%. A lakh/crore slip would show as ~100× on every name; it does not.
- **ROCE matches Screener's published row to a median 0.2 pp**, 100% within 5 pp — the right row
  is being read. ROE differs by a median 2.6 pp for the stated reason: this panel averages three
  years where Screener publishes the latest.
- **Point-in-time versus today: 46 names both ways, 45 of them the same names.** That is the
  reconciliation against what Arun sees when he runs the query himself.

### 8.3 Coverage and survivorship — the inherited caveat is refuted

| | count |
|---|---:|
| Universe symbols | 2,158 |
| With Screener fundamental history | **2,131 (98.7%)** |
| Screenable (≥ 4 filed years) | 2,126 (98.5%) |
| Stopped series (last bar < 2026-06-01) | 102 — of which **101 still have a Screener page** |
| Stopped AND ever ≥ ₹5cr/day AND no page | **1** |

**Screener keeps the pages of delisted companies**, so the fundamental leg is close to
survivorship-clean — the opposite of what r/158 assumed from a 638-name sample. The single
missing liquid name is TATAMOTORS, whose series stops at a demerger, not at a delisting.

**The real exposure has moved upstream, where this leg cannot measure it.** `market_data.db`
carries only 102 stopped series in 2,158 (4.7%) across eleven years — fewer than the NSE
actually delisted or suspended in that time. A company that never entered the price database is
invisible to both legs *and* to this audit. That is the survivorship caveat the study must
carry; the Screener one is not it.

**FY depth:** 1,375 symbols carry 12 filed years, 25 carry 13, 99 carry 11, 27 carry none.
Screener serves about twelve years to a signed-out reader, so FY2015 is the earliest year for
most names — a hard floor on the window.

### 8.4 The honest study start is 2018-08-01

Share of names trading that month which had ever turned over ≥ ₹5cr/day and had four filed
years: 6% at 2018-01, 7% at 2018-07, **87% at 2019-01**. The step is at **2018-08-01**, when
FY2018 becomes usable under the 4-month lag; the first month ≥ 50% and the first month ≥ 80%
are the same date. Before it, coverage is not random — it is whichever companies happen to have
longer pages — so an earlier start measures Screener's depth rather than the screen.

### 8.5 The replication gate: the mechanical screen does NOT reproduce his book

69 equities in his real Zerodha history (105 daily snapshots, 2026-04-20 → 2026-09-11); all 69
have fundamental data, so nothing here is a coverage effect.

| cohort | n | passes the full screen | passes screen AND near-ATH |
|---|---:|---:|---:|
| Observed buys (first seen after 2026-04-21 — the honest sample) | 43 | **3 (7%)** | 2 (5%) |
| Pre-existing at 2026-04-20 | 26 | 5 (19%) | 3 (12%) |
| All equities held | 69 | **8 (12%)** | 5 (7%) |

Failure reasons across the 69: **growth 50**, debt/equity 33, ROE 26, ROCE 18, negatives 5,
market cap 1.

Relaxing one dial at a time, same names, same dates:

| variant | passes |
|---|---:|
| the full screen as written | 8 |
| growth bar lowered to 15% | 19 |
| growth bar lowered to 10% | 20 |
| growth on sales only (profit growth dropped) | 10 |
| growth dropped entirely | 27 |
| debt/equity dropped | 14 |
| **growth AND debt/equity both dropped** | **40** |
| **near-ATH alone, no fundamentals** | **42 of 69** |

**What he actually buys is "near its high and profitable" — not "20%+ growth on both lines and
nearly debt-free".** 42 of 69 names sat within 10% of their all-time-high close on the day they
first appear in the book; only 8 clear the written screen. The study must therefore say which
of the two it is testing. A backtest of the screen as written is testing a strategy Arun does
not currently run, and its result cannot be presented as a validation of his process. The
price-side condition is the part of his method the data confirms he follows; the fundamental
thresholds as written are aspiration rather than practice. Both are legitimate to test — but
they are different questions and the write-up must not blur them.

### 8.6 Data defects found

1. **Lenders have no debt/equity at all.** Screener's bank and NBFC balance sheets carry no
   `Borrowings` row — their debt sits inside `Other Liabilities`. `de_latest` is NaN for every
   lender, so **D/E ≤ 0.2 excludes the entire financial sector by construction**, and the lender
   ROCE carve-out in the masks is cosmetic because D/E has already removed those names. Call it
   a non-financials screen rather than implying the filter weighed financials and found them
   wanting. Bank OPM on that page is a layout artifact and can print negative — never read it.
2. **Split scale.** `market_data.db` is not retroactively split-adjusted, so pre-split months
   carry the old, higher price and an inflated `mcap_pit`. 88 of 2,158 symbols show a one-day
   close collapse below 0.55×, and every panel row carries `mcap_scale_suspect` when such an
   event lies in its future. The engine leg's independent scan (`panel_split_events.csv`) picks
   up the same family of events. The mcap *level* is sound; its *history* is the suspect part.
3. **Quarterly OPM is a recent-window feature.** Screener carries ~13 quarters, so
   `opm_q_slope8` exists only from about mid-2023 (60,431 of 256,828 rows) and `opm_rising_q` is
   a recent-window mask whose earlier zeros are absence, not rejection. Use the annual
   `opm_slope3` for the long window.
4. **Restated, not as-reported.** Screener shows figures as they stand today. The filing lag
   controls when a year becomes visible; it cannot undo a later restatement. This is the
   residual look-ahead in the panel and it should be named in the write-up.

### 8.7 What the study agent must carry

1. Start no earlier than **2018-08-01**.
2. Run every arm **both ways** against `has_data` — missing→ineligible and missing→eligible. The
   gap between them is the coverage bias.
3. Compare screened arms with an unscreened arm on the **same sub-universe**.
4. On survivorship, quote the price database's own coverage as the open question, not Screener's.
5. Say which strategy is under test: the written screen (8 of his 69 names) or his practice
   (42 of 69 near-ATH).
6. Market cap is the softest and nearly inert criterion; ROCE is fully inert behind ROE; growth
   is what actually binds.
