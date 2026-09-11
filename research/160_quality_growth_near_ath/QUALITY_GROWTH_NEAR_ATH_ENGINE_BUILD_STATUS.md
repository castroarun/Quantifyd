# Quality-Growth Near All-Time-High — POSITIONAL BACKTEST ENGINE (build leg)

STATUS: **DONE** — engine built, all five self-tests PASS, handed to the study agent.
This doc tracks the ENGINE only.
Owner: ENGINE-LEG agent. Sibling: DATA-LEG (`QUALITY_GROWTH_NEAR_ATH_DATA_LEG_STATUS.md`,
point-in-time fundamentals panel + eligibility masks). A third agent runs the study.

Last updated: 2026-09-11 ~15:25 IST

---

## 1. The Ask

**What was asked (orchestrator, in substance):** build "a validated, honest positional
backtest ENGINE with every entry/exit/book axis pluggable, plus its self-tests and
price-only baseline arms" for research/160 *Quality-growth near all-time-high*. Do NOT run
the fundamentals sweep — that is the study agent's job.

**What is actually being built:** a close-decision / next-open-fill, slot-constrained,
equal-weight positional portfolio simulator over the full NSE daily universe
(`market_data_unified`, 2,905 day symbols, 2000 to 2026-09-10), in which the *candidate
set* is the intersection of

1. a **liquidity floor** (20-day median traded value),
2. a **near-all-time-high price state** (close >= k x causal ATH close, or a fresh ATH
   breakout), and
3. an optional **point-in-time fundamental eligibility mask** supplied by the DATA-LEG as
   an npz of monthly boolean rows,

and in which the entry mechanic, the ranking axis, the exit stack, the index regime gate,
the book size, the costs, the taxes and the idle-cash yield are all independent CLI axes.
The deliverable is the engine + its self-tests + price-only baseline arms whose numbers can
be checked against this project's existing anchors. **No fundamental sweep is run here.**

**Why an engine and not a script:** the study agent needs to run a few hundred cells
resume-safely, each with an ensemble of 12 rebalance offsets or 30 selection seeds, and to
report paired A-vs-B differences on the same path. That only works if one binary owns the
whole state machine.

---

## 2. The Base — what the engine implements

### 2.1 Causality contract (BINDING — playbook 5A)

> A trigger evaluated on bar *t*'s CLOSE may never be filled at a price from before that
> close.

The engine is **close-only for every decision**. There is no intraday level, no
`high >= level` trigger, no same-bar fill anywhere. The `high`/`low` arrays are loaded for
data-defect diagnostics only and are never read by the simulator. The trigger/fill trap
that cost r/142 42 CAGR points is therefore *structurally impossible* in this engine, not
merely guarded against.

Two — and only two — fill conventions, both labelled in every output row:

| `--fill` | Decision | Execution | Default |
|---|---|---|---|
| `next_open` | close of bar *t* | `open[t+1]` | **yes** |
| `same_close` | close of bar *t* | `close[t]` (a ~15:10 decision, live process required) | no |

Exits obey exactly the same contract: decided on a close, filled at the next open (or that
same close). An exit is never filled at the level the price passed through intraday.

State used in a decision at the close of *t*: `close[t]`, `athc[t]` (cummax **including**
*t*), `tv20[t]`, `sma_W[t]`, `donch_low_W[t-1]`, the fundamental mask row in force on *t*,
and the index-gate state at *t* (itself shifted one day inside the gate construction).
Nothing from *t+1* or later enters any decision.

### 2.2 Universe and data hygiene

- Source: `backtest_data/market_data.db`, `market_data_unified`, `timeframe='day'`. The
  VPS copy is canonical. The engine opens the DB **read-only** (`mode=ro`) — it can never
  write to it.
- **Funds excluded** via `backtest_data/etf_exclusions.json` (346 symbols, name-based: 338
  by NAME + 8 ticker-only) plus r/158's ticker regex as the net for delisted funds. A
  ticker blacklist alone is known to rot (r/158: 221 gold/silver/index funds leaked into an
  equity book and manufactured a +32% arm).
- **Phantom holiday rows** (O=H=L=C=prev close, volume 0 — Kite placeholders): detected and
  set to NaN per symbol before any rolling statistic. Count logged in the panel build.
- **Partial candles**: if the panel is built during market hours the last date is dropped
  (automatic when max(date) == today and the IST clock is before 17:45).
- **Retroactive split defect** (MCX, HEG, NAZARA, CUPID, ...): pre-split rows keep the old
  price scale, which fakes an ATH distance. Guard: a single-day close move <= -40% is
  treated as a split; the ATH `cummax` is **restarted** at that bar, so history before the
  split can never set the high the screen compares against. Every detected event is written
  to `results/panel_split_events.csv`. This is a *conservative* guard: it also restarts on
  genuine -40% one-day crashes, which only makes the near-ATH state easier to satisfy for
  those names — the direction is stated, not hidden.
- **Data start is independent of the trading start** (`--base-start`, default 2000-01-01).
  An ATH is a `cummax` over the whole history; deriving the data start from the trading
  start silently converts it into an N-month high (the r/142 trap).
- Rolling statistics are computed on a **forward-filled-within-span** matrix and then
  re-masked to the symbol's live span, so a missing row can never NaN-poison every window
  after it (the failure that silently disabled r/142's SMA-200 gate from Apr-2026). The
  difference vs the strict dropna-and-reindex recipe is that a hole counts as a repeated
  close rather than being skipped; holes are rare and the effect is immaterial. Stated here
  rather than buried.

### 2.3 Candidate state (all causal at the decision close)

| Axis | Values |
|---|---|
| liquidity floor | 20-day **median** traded value >= {1, 2, 5} Rs cr (`--tv-floor`, in Rs cr) |
| near-ATH state | `close >= k * athc` with k in {0.85, 0.90, 0.95} (`--k`) |
| near-ATH variant | `--state near` (default) or `--state new_ath` (`close > athc_prev`, a fresh breakout) |
| fundamental mask | `--mask <npz>` with `--mask-missing {fail,pass}`; omit `--mask` for price-only arms |

Mask contract: `dates` (`<U10` ISO, 1st of month), `cols` (symbols), `mask` (bool
[dates x cols]). A row is valid from its date until the next date, so it is aligned by
label and forward-filled to daily. `fail` = a symbol absent from `cols` (or before the
first row) cannot be bought; `pass` = it is left eligible. **Both are reported**; the gap
is the coverage bias.

### 2.4 Entry mechanics (`--entry`)

| Mode | Decision | Slot logic | Ensemble |
|---|---|---|---|
| `rebalance` | on a cadence day: rank the whole qualifying set, hold top-N | full re-selection with hysteresis | **12 rebalance-day offsets** (0..11 trading days after month start) |
| `first_qualify` | the first day a name's state turns true | fill any free slot | **30 random tie-break seeds** |
| `ath_breakout` | qualifying **and** `close > athc_prev` | fill any free slot | **30 random tie-break seeds** |

- Cadence (`--cadence`): `monthly`, `quarterly`, `semiannual`, `annual`.
- Ranking (`--rank`): `rs` (IBD-style `2*r63 + r126 + r189 + r252`, cross-sectional
  percentile among eligible), `dist_ath` (closest to ATH first), `mcap_asc`, `mcap_desc`,
  `tv_desc`, `random` (seeded).
- `mcap_*` uses r/142's `results/mcap_snapshot.json` shares-constant proxy (2,042 of 2,321
  symbols priced). It is a **current-shares** proxy back-projected on price — a real bias,
  flagged in the row and to be quoted as such.
- Hysteresis (`--buffer`, default 1.5): an existing holding is retained while it still
  qualifies and ranks within `buffer x N`; only then are free slots filled from the top of
  the ranking. `--buffer 1.0` = no hysteresis.

### 2.5 Exit families (`--exits`, comma-separated, composable, evaluated jointly)

`none` - `fund_fail` (mask turns false at the next decision date) - `sma_trail:{20,50,100,200}`
(close < SMA) - `peak_dd:{15,20,25,30}` (% from the highest **close** since entry) -
`donchian_low:{20,50}` (close < prior-W-day min close) - `time:{6,12,24}` (months) -
`hard_stop:{10,15,20}` (% from entry close).

Always on, independent of `--exits`: **`stale_exit_days`** (default 60). A position in a
name that has not printed a close for that many sessions is liquidated at the last known
price. Without it a delisted or suspended name is carried at its last traded price for the
rest of the run, which flatters every no-exit arm — self-test 3d surfaced exactly that.
`--stale-exit-days 0` disables it, knowingly.

Index gate (`--index-gate`): `none` - `nifty200sma` (NIFTY50 close < SMA200) -
`niftybees100sma_weekly` (weekly NIFTYBEES close < 100-week SMA), with
`--gate-action {block_new, liquidate_all}`. Gate series are computed on the dropna'd index
series and shifted one day before use.

### 2.6 Book, costs, taxes, cash

- `--slots N` in {8, 10, 15, 20, 30}, equal weight `1/N` of NAV at entry,
  `--max-position-pct` cap (default 0.30 of NAV at the moment of sizing).
- `--cost-bps` per side, reported at 0 / 25 / 40 / 60.
- Tax: 20% STCG, 12.5% LTCG for holds > 365 days, netted within the **Indian FY**
  (1 Apr - 31 Mar), net losses carried forward. Implementation follows r/158: tax is
  accrued per realized trade at its own rate (a loss contributes a negative accrual), the
  FY pool is settled at the FY boundary, and a negative pool is carried forward rather than
  refunded. The approximation vs the statute (STCL/LTCL set-off ordering) is documented,
  not silently applied.
- Idle cash yields `--cash-yield` 5% p.a., accrued `/252` per trading day.

### 2.7 Metrics written per cell

`label` + every parameter, then: CAGR gross / net / after-tax, MaxDD (daily, measured from
the running peak of the **full** curve — the r/154 convention), Calmar (after-tax), Sharpe,
trades per year, win rate, average win %, average loss %, expectancy per trade net, max
losing streak, turnover (x NAV per year), average % invested, capacity (median position
size / median tv20 of the held names), and per-year returns with intra-year drawdowns.
Ensembles report **median [min .. max] and the worst path**; `paired_diff()` compares A vs
B on the same offset/seed.

### 2.8 Pre-registered ranking metric

Cells are ranked by **after-tax Calmar at 25 bps/side**, with the tradeability gate (win
rate, avg win/loss, expectancy net, max losing streak, trades/yr, capacity) shown in the
same table. A cell is only interesting if its parameter neighbours agree (plateau, not
peak). Registered here before any cell is run.

---

## 3. Plan — build order and the self-tests that gate it

| # | Step | Artifact |
|---|---|---|
| 1 | Panel cache builder: clean OHLCV, phantom-row NaN, split-event ATH restart, tv20, athc | `scripts/qg_panel.py` -> `results/panel_2000.npz`, `results/panel_split_events.csv` |
| 2 | Engine: state, entries, exits, book, costs, tax, metrics, CSV | `scripts/qg_engine.py` |
| 3 | Reporting: house YoY table (MD + HTML), tearsheet wrapper | `scripts/yoy_table.py`, `scripts/tearsheet_wrap.py` |
| 4 | Self-tests 1-5 below | `scripts/selftest.py` -> `results/selftest_*.csv`, `results/SELFTEST.md` |

### Self-tests (all recorded, pass or fail)

1. **Interface smoke** — r/158 `fund_mask_strict.npz`, `rebalance` monthly, N=15, rank=rs,
   k=0.90, tv >= Rs 2cr, exits `none`, 2024-08 to 2026-09. Proves the pipeline runs end to
   end and the per-cell row is complete.
2. **Look-ahead probe** — the same cell with (a) `--fill same_close`, (b) `--fill
   next_open`, and (c) **all price data shifted one day later**. The shift test must NOT
   reproduce the unshifted result; if it does, the engine is reading a bar it should not.
3. **Price-only baselines**, 2010-01 to 2026-09 (data from 2000), tv >= Rs 2cr:
   (a) NIFTYBEES / NIFTY500 / NIFTYMIDCAP150 / NIFTYSMLCAP250 buy-and-hold;
   (b) near-ATH-only (k=0.90, no mask) monthly rebalance N=15 rank=rs, 12 offsets;
   (c) random-selection null from the same liquid + near-ATH universe, 30 seeds;
   (d) equal-weight hold-forever of everything entering the liquid universe (should look
   index-like).
   Anchors: r/75 momentum 31.9% net / -31.6% DD (2006-26); NIFTYBEES 11.5% / -59.7%
   (2006-26); True North ~20.7% after tax / -25.1% (12-offset median). A baseline wildly
   off these is a bug until proven otherwise.
4. **Cost / tax monotonicity** — one cell at 0 / 25 / 40 / 60 bps and pre/after tax must be
   correctly ordered.
5. **Speed** — seconds per cell for `rebalance` monthly and for daily `first_qualify`, so
   the study agent can size its grid. Target < 30 s per single path on ~2,500 symbols x
   16 years.

### Cell-count discipline

This leg runs **no sweep**. The self-test cells above (about 20 single paths + 3 ensembles)
are validation, not discovery, and are excluded from any multiple-testing haircut the study
agent computes.

---

## 4. Status (live log)

**Phase:** DONE. Engine, reporting helpers and self-tests complete; committed on the VPS.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-11 14:45 | Doctrine read | quant-researcher agent, playbook sections 4/5/5A, r/158 `oa_entry_mechanics.py`, `etf_filter.py`, r/154 `yoy_five_systems.py`, `_utilities/tearsheet.py` |
| 2026-09-11 14:48 | Environment probed | 2,905 day symbols, max date 2026-09-10; index series `NIFTY50`/`NIFTY500`/`NIFTYMIDCAP150`/`NIFTYSMLCAP250` all 2011-01-03 to 2026-09-10, `NIFTYBEES` 2005-01-03 to 2026-09-10; `etf_exclusions.json` = 346 syms; the mcap proxy lives in **r/142** results (not r/158, whose code references a path that does not exist); numpy 2.4.4 / pandas 3.0.2 / scipy 1.17.1, no bottleneck; VPS load ~5 on 4 cores, so `nice -n 10` and one worker |
| 2026-09-11 14:52 | STATUS sections 1-4 written before any code | this file |
| 2026-09-11 14:56 | `qg_panel.py` written and run | 6,640 sessions x 2,707 symbols, 344 flagged funds/indices, 151 split restarts, 105 s |
| 2026-09-11 15:05 | `qg_engine.py` runs end to end | r/158 strict mask, 133 trades, complete CSV row, 2.2 s |
| 2026-09-11 15:08 | **Self-test 3a found a PANEL defect** | the phantom-row test (volume 0 AND high == low) was deleting 1,990 REAL sessions each from NIFTYMIDCAP150 / NIFTYSMLCAP250 and truncating both benchmarks to 2019. An index has volume 0 on every row by construction. Fixed: the test now applies only to instruments that carry volume at all |
| 2026-09-11 15:10 | **Self-test 3d found an ENGINE defect** | a hold-forever arm carried delisted names at their last traded price forever. Added `stale_exit_days` (default 60) |
| 2026-09-11 15:12 | Panel rebuilt with both fixes | 92,412 phantom rows dropped (8,957 real index sessions restored), 152 split restarts |
| 2026-09-11 15:16 | Full self-test battery re-run from scratch | 1-5 all PASS -> `results/SELFTEST.md` |
| 2026-09-11 15:17 | `yoy_table.py` + `tearsheet_wrap.py` exercised on the baselines | `results/yoy_baselines.{md,html,csv}`, `results/curves_*.png`, `results/tearsheet_*.png` |

---

## 5. Crash recovery — resuming without Claude

Everything runs on the VPS at `/home/arun/quantifyd/research/160_quality_growth_near_ath`.

**What finished?**

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/160_quality_growth_near_ath
ls -la results/                 # panel_2000.npz present = panel built
tail -40 results/selftest.log   # self-test progress
wc -l results/selftest_cells.csv
```

**Is anything still running?**

```bash
ps -ef | grep -E 'qg_engine|qg_panel|selftest' | grep -v grep
```

**Rebuild the panel** (only if `results/panel_2000.npz` is missing or corrupt, ~6 min):

```bash
cd /home/arun/quantifyd
nice -n 10 venv/bin/python3 research/160_quality_growth_near_ath/scripts/qg_panel.py \
  --base-start 2000-01-01 --out research/160_quality_growth_near_ath/results/panel_2000.npz
```

**Re-run the self-tests** (resume-safe; completed cells are skipped):

```bash
cd /home/arun/quantifyd
nohup nice -n 10 venv/bin/python3 \
  research/160_quality_growth_near_ath/scripts/selftest.py \
  > research/160_quality_growth_near_ath/results/selftest.log 2>&1 &
```

**Do NOT touch:** `backtest_data/market_data.db` (read-only, canonical, live systems read
it), anything under `services/`, the crontab, the `quantifyd` service, and the DATA-LEG's
files (`QUALITY_GROWTH_NEAR_ATH_DATA_LEG_STATUS.md`, `scripts/build_universe.py`,
`screener_fetch_full.py`, `build_pit_panel.py`, `build_masks.py`, `coverage_audit.py`,
`holdings_check.py`, `results/screener_cache/`, `results/features_pit*`, `results/masks/`,
`results/coverage*`, `results/holdings*`).

**Safe to inspect / delete and regenerate:** everything matching `results/panel_*`,
`results/selftest*`, `results/cells_*.csv`, `results/*_equity.csv`, `results/yoy_*`,
`results/tearsheet*`.

---

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `QUALITY_GROWTH_NEAR_ATH_ENGINE_BUILD_STATUS.md` | this file | yes |
| `scripts/qg_panel.py` | panel cache builder (clean OHLCV, splits, phantoms, tv20, athc) | yes |
| `scripts/qg_engine.py` | the engine (CLI + importable) | yes |
| `scripts/yoy_table.py` | house YoY table -> Markdown + HTML | yes |
| `scripts/tearsheet_wrap.py` | wrapper over `research/_utilities/tearsheet.py` | yes |
| `scripts/selftest.py` | self-tests 1-5 | yes |
| `results/panel_2000.npz` | cached price panel (~300 MB) | **NO — gitignored** |
| `results/panel_split_events.csv` | detected split restarts | yes |
| `results/selftest_cells.csv` | per-cell rows for every self-test | yes |
| `results/SELFTEST.md` | self-test verdicts + numbers | yes |
| `results/*_equity.csv` | per-path daily equity curves | small ones yes |
| `results/yoy_baselines.md` / `.html` | house YoY table for the baseline arms | yes |

---

## 7. How the study agent calls this

### 7.0 One-time

The panel cache must exist. If `results/panel_2000.npz` is absent, build it (section 5).
Every engine invocation reads it in about 8 s; derived frames (SMAs, Donchian, RS) are
built once per **process**, so run a whole sweep inside one `--grid` invocation rather than
one process per cell.

### 7.1 Single cell

```bash
cd /home/arun/quantifyd
nice -n 10 venv/bin/python3 research/160_quality_growth_near_ath/scripts/qg_engine.py \
  --panel research/160_quality_growth_near_ath/results/panel_2000.npz \
  --start 2010-01-01 --end 2026-09-10 \
  --entry rebalance --cadence monthly --rank rs --slots 15 --buffer 1.5 \
  --state near --k 0.90 --tv-floor 2 \
  --mask research/160_quality_growth_near_ath/results/masks/<mask>.npz --mask-missing fail \
  --exits sma_trail:50,peak_dd:25 --index-gate none \
  --fill next_open --cost-bps 25 --tax --cash-yield 0.05 \
  --offsets 12 \
  --out research/160_quality_growth_near_ath/results/cells_study.csv \
  --label QG_k90_N15_rs_mo_trail50_dd25_strict_fail
```

- `--offsets 12` runs rebalance-day offsets 0..11; `--seeds 30` runs 30 tie-break seeds
  (use with `--entry first_qualify` / `ath_breakout`). Use one or the other.
- Add `--dump-equity` to also write `results/<label>_equity.csv` (one column per path) and
  `--dump-trades` for `results/<label>_trades.csv`.
- The run is **resume-safe**: a cell whose `label` already exists in `--out` is skipped.
  Delete the row to force a re-run.

**Read the two diagnostic lines the engine prints on every masked cell.** They are the
cheapest guard against reporting a coverage artefact as a result:

```
mask <name>.npz: rows 2015-01-01..2026-09-01 (141 monthly), 2158 cols, 2158 matched
                 the panel, true 0.081%; missing=fail
  !! WARNING: 30% of the window precedes the mask's first row ...
  qualifying name-days in the window: 547486 liquid+state -> 1291 after the mask
                 (0.2% survive), 0.3 candidates per session
  !! NOTE: fewer qualifying names per session (0.3) than slots (15) ...
```

A cell carrying either flag is not measuring the screen — it is measuring the missing
policy, or the idle-cash yield. Check `avg_pct_invested` in the row before quoting its CAGR.

### 7.2 A whole sweep in one process (recommended)

Write a JSON list of cells and pass it:

```bash
nice -n 10 venv/bin/python3 .../qg_engine.py --panel .../panel_2000.npz \
  --grid .../results/grid_phase1.json --out .../results/cells_phase1.csv
```

Each grid element is an object whose keys are the long-option names without the leading
`--` and with dashes as underscores. The full set, with defaults:

```
label 'cell' · start '2010-01-01' · end '2026-09-10'
entry 'rebalance' · cadence 'monthly' · rank 'rs' · slots 15 · buffer 1.5 · retain 'strict'
state 'near' · k 0.90 · tv_floor 2.0 · mask '' · mask_missing 'fail'
exits 'none' · index_gate 'none' · gate_action 'block_new'
fill 'next_open' · cost_bps 25.0 · tax true · cash_yield 0.05 · max_position_pct 0.30
capital 10000000 · reentry 'transition' · stale_exit_days 60
offsets 0 · seeds 0 · arms 'all'
```

Anything omitted takes the CLI default. The panel and the derived frames are built once and
reused for every cell. `arms: 'tax'` runs only the after-tax simulation (one third of the
work) when a sweep does not need the gross/net decomposition — use it for wide G1 grids and
switch back to `'all'` for anything you intend to report.

### 7.3 Programmatic use

```python
import sys; sys.path.insert(0, 'research/160_quality_growth_near_ath/scripts')
from qg_engine import Panel, Derived, Cell, run_cell, paired_diff

panel = Panel.load('research/160_quality_growth_near_ath/results/panel_2000.npz')
der   = Derived(panel)                       # SMAs, Donchian, RS — build once
a = run_cell(panel, der, Cell(label='A', exits='sma_trail:50', offsets=12))
b = run_cell(panel, der, Cell(label='B', exits='sma_trail:20', offsets=12))
print(paired_diff(a, b, 'cagr_net_tax'))     # median delta, wins/total, per-path deltas
```

`run_cell` returns a dict with the CSV row under `row`, the per-path rows under `paths`,
and the per-path equity `pd.Series` objects under `curves`.

### 7.4 Exact CSV column list (`--out`)

```
label, start, end, entry, cadence, rank, slots, buffer, state, k, tv_floor,
mask, mask_missing, exits, index_gate, gate_action, fill, cost_bps, tax,
cash_yield, max_position_pct, n_paths, path_kind,
cagr_gross, cagr_net, cagr_net_tax,
cagr_net_tax_min, cagr_net_tax_max, cagr_net_tax_worstpath,
maxdd, maxdd_worst, calmar, sharpe,
trades_per_yr, win_rate, avg_win_pct, avg_loss_pct, expectancy_net_pct,
max_losing_streak, turnover_x_nav_yr, avg_pct_invested, capacity_ratio,
n_trades, final_x, yearly
```

`yearly` is a JSON object `{"2011": [return_pct, intrayear_dd_pct], ...}`, medians across
paths, drawdown measured from the running peak of the full curve. `*_worstpath` and
`maxdd_worst` are the worst single path, which is the number to plan on.

### 7.5 Reporting

```bash
# House YoY table: one column per system + the three index benchmarks
venv/bin/python3 .../scripts/yoy_table.py \
  --curves NAME1=.../results/A_equity.csv NAME2=.../results/B_equity.csv \
  --out .../results/yoy_study            # writes .md, .html and .csv

# Tearsheet (log equity vs the indices + drawdown panel)
venv/bin/python3 .../scripts/tearsheet_wrap.py \
  --curve .../results/A_equity.csv --name "QG k90 N15" --out-dir .../results
```

---

## 8. Findings (engine leg)

Authoritative numbers: `results/SELFTEST.md`. **All five self-tests PASS.**

**1. Interface smoke.** r/158 `fund_mask_strict.npz` (609/609 mask columns matched the
panel), monthly rebalance N=15 RS-ranked, k=0.90, tv >= Rs 2cr, 2024-08 to 2026-09:
gross 12.64% / net 9.58% / after-tax 8.68% CAGR, MaxDD -29.69%, 133 trades, every column of
the row populated.

**2. Look-ahead probe.** Next-open 8.68%, signal-close 8.49%, and with all price data
shifted one day later 11.34% / MaxDD -25.77%. The shifted arm does not reproduce the
unshifted result, so no bar is being read that should not be. The engine is close-only by
construction — the simulator never touches `high`/`low` — so the r/142 trigger/fill trap
cannot be expressed in it at all.

**3. Price-only baselines, 2010-01 to 2026-09, tv >= Rs 2cr.**

| arm | gross | net | after tax | [min..max] | worst path | MaxDD (worst) | Calmar |
|---|---|---|---|---|---|---|---|
| near-ATH k=0.90, N=15, RS, monthly, **12 offsets** | 30.39 | 26.95 | **23.15** | [19.12 .. 27.61] | 19.12 | -53.22 (-61.43) | 0.42 |
| random-selection NULL, same universe + state, **30 seeds** | 22.34 | 16.80 | **13.81** | [10.19 .. 17.96] | 10.19 | -44.84 (-55.56) | 0.32 |
| equal-weight hold-forever, top-250 by turnover | 12.34 | 12.22 | **12.22** | — | — | -48.48 | 0.25 |

Index buy-and-hold on the same window: NIFTYBEES 10.26% / -36.34% (from 2010-01-04),
NIFTY 50 8.91% / -38.44%, NIFTY 500 10.25% / -38.30%, MIDCAP 150 14.48% / -44.23%,
SMALLCAP 250 12.15% / -60.79% (the four index series start 2011-01-03; the DB has no
earlier index history).

*Read against the anchors.* r/75 momentum is 31.9% NET over 2006-26 with -31.6% DD; this
baseline is 26.95% net with -53% DD over 2010-26. The return is the same order; the
drawdown is far worse — which is exactly what r/75 said to expect, since its index-EMA gate
is "the whole risk story and IRREPLACEABLE" and this baseline deliberately has no gate.
True North is ~20.7% after tax / -25.1% on a 12-offset median; this baseline is 23.15%
after tax on a 12-offset median with more than twice the drawdown. Both land where a
gate-free, no-exit, 15-slot near-ATH momentum book should land. Nothing is wildly off, so
the pipeline is trustworthy enough to sweep on.

*The null is the number that matters.* Random selection from the **same** liquid, near-ATH
universe returns 13.81% after tax — barely above equal-weight hold-forever at 12.22% and
inside the 9-14% index range. **The near-ATH state alone is worth very little; the RS
ranking on top of it is worth about +9.3pp after tax.** That gap, not the headline, is the
bar the fundamental mask has to clear. Both arms carry identical survivorship bias, so the
gap is the part survivorship cannot explain.

**4. Cost / tax monotonicity.** 0 / 25 / 40 / 60 bps give net CAGR 34.84 / 30.58 / 29.43 /
26.58 and after-tax 29.15 / 26.36 / 24.80 / 22.58 — strictly ordered, with
gross >= net >= after-tax in every row. Costs are worth ~8.3 CAGR points between 0 and
60 bps on a book turning over 5.7x NAV per year; tax is worth another ~4.

**5. Speed.** Monthly rebalance **1.2-2.0 s per path** for all three arms (~0.5 s per single
simulation) over 2010-2026 on 2,707 symbols; daily `first_qualify` 2.3 s per path. A
12-offset cell is ~14 s, a 30-seed cell ~41 s — far inside the 30 s-per-path target. The
250-slot hold-forever arm is the slow case at 14.7 s (per-position loop). Panel build 105 s
/ 791 MB, done once. Peak RSS ~500 MB, so two workers fit in the VPS's free memory — but no
more than two, and always under `nice`.

**6. Real DATA-LEG mask, both missing policies** (added once the DATA-LEG's `results/masks/`
appeared). Same cell either side — monthly rebalance N=15 RS-ranked, k=0.90, tv >= Rs 2cr,
no exits, 25 bps, 12 offsets, 2010-2026:

| arm | CAGR after tax | MaxDD (worst) | % invested | trades/yr |
|---|---|---|---|---|
| `arun_strict`, missing = fail | 6.10 | -6.80 (-10.64) | **3.8** | 1.0 |
| `arun_strict`, missing = pass | 11.37 | -53.11 (-58.58) | 66.7 | 27.4 |

The interface works against the real artefact, and the test immediately paid for itself.
The engine now prints a **mask-coverage diagnostic on every masked cell**, and this one
raised two flags:

- **30% of the window precedes the mask's first row (2015-01).** Those years are decided by
  the missing policy alone, with no fundamental evidence behind them. Align `--start` to the
  mask, or label the arm a coverage artefact.
- **`arun_strict` passes 0.081% of name-months — 0.3 qualifying names per session against
  15 slots.** The book is 3.8% invested: its 6.10% "return" is the 5% idle-cash yield and
  its shallow -6.8% drawdown is the drawdown of a cash pile, not of a strategy. A mask that
  tight needs far fewer slots or a looser screen before its numbers mean anything. **Always
  read `avg_pct_invested` next to the CAGR.**

The two missing policies differ by 5.3pp of CAGR and 46pp of drawdown here — the coverage
bias measured rather than assumed. Report both, every time.

### Limitations the study agent must carry

- **The universe is not point-in-time.** Names are in the panel only if the DB kept them;
  survivorship pressure is upward on every arm, benchmarks included. The random-selection
  null is the control that neutralises it for any ranking claim.
- **`stale_exit_days` is a partial delisting control, not a delisting model.** The panel
  carries no delisting reason and no recovery value.
- **The split guard is deliberately conservative.** 152 ATH-cummax restarts (logged in
  `results/panel_split_events.csv`); a genuine one-day -40% crash also restarts the cummax,
  which makes the near-ATH state *easier* to satisfy for that name. Direction stated, not
  hidden.
- **Weights are not rebalanced.** A rebalance day re-selects names; it does not resize the
  survivors back to 1/N. Winners run, and the book drifts from equal weight between entries.
- **`mcap_*` ranking uses a shares-constant proxy** (r/142's `mcap_snapshot.json`, 2,042 of
  2,707 panel symbols priced): current shares back-projected on historical price. Quote it
  as a proxy or not at all.
- **Rolling statistics use forward-fill-within-span**, not strict dropna-and-reindex, so a
  missing session counts as a repeated close inside a window rather than being skipped.
  Holes are rare after the phantom purge; the effect is immaterial, but it is a choice.
- **Tax is an approximation of the statute**: a single netted FY pool with per-trade rates
  and loss carry-forward, not the STCL/LTCL set-off ordering.
- **The hold-forever arm books no closed trades**, so its win-rate / average-win /
  average-loss columns describe marked-open positions. Read only its CAGR and drawdown.
- **`capacity_ratio` is quoted at the default Rs 1cr book.** Multiply by the real book size
  before reading it as a constraint.
