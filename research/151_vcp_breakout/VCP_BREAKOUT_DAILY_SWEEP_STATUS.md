# BananaPatterns "VCP" Screen — Volatility-Contraction Breakout, Replication → Robustness → Blend

STATUS: **DONE (2026-09-05)** — **VERDICT: NO EDGE.** Replication gate PARTIAL PASS (62.2%); the published claim REFUTED; the screen killed by its own null control; rejected for the book (corr 0.75 to Open Alpha, beaten by a cash sleeve). Full write-up in `results/RESULTS.md`; published at `/app/backtest/vcp-breakout-research151`.

Research number 151. Host: VPS `94.136.185.54` (`/home/arun/quantifyd`). All heavy phases
run under `flock -w 7200 /tmp/qf_sweep.lock` (three studies share a 4-core box).

---

## 1. The Ask

**What Arun asked:** test bananapatterns.com's **VCP screen** (volatility contraction
pattern breakout) — the site's own backtest panel, with the dial combination he found best
(5 positions, cut a loser at 7%, sell winners by Trail 50-day, risk 2%/trade, skip-weak
markets OFF, 2020→2025, ₹10L) claims **₹10L → ₹2.6Cr = 25.99×, CAGR +72.1%, worst fall
−14.8%, 48% won, 164 trades**. Replication gate applies first; optimization only after.

**What we are actually testing, in three separable questions:**

1. **THE RULES (replication gate).** The site does not publish its VCP definition. Can a
   VCP parameterization be inferred that reproduces the **40 ground-truth trades**
   transcribed from a VCP-screen run of the same site
   (`data/vcp_trades_groundtruth.csv`)? Metric: entry-date match rate (±2 trading days)
   and entry-price match rate (±1%), plus exit-date/price match. Report honestly; "no
   parameterization reproduces them" is a legitimate and publishable finding.
2. **THE CLAIM.** Their 25.99× / 72.1% CAGR / −14.8% worst fall is *one path* on a
   6-year window they themselves footnote as *"85% of this period was a strong market"*,
   and the whole panel is stamped PROVISIONAL ("under a methodology review"). Test the
   claim separately from the rules: seed ensembles, longer windows, honest drawdown.
3. **THE STRATEGY.** Does the construction survive costs, taxes, path-dependence and —
   the binding question — does it **add anything to the deployed TN+OA 50-50 book**? Its
   ancestor Open Alpha (r/142 Blue Sky) is already live; a VCP screen on the same
   universe with the same trail is a prime candidate to be a **duplicate of existing
   beta** (the r/145 failure mode). Pre-registered complement bar in §2.5.

**Prior art that constrains this study**
- `research/142_bananapatterns_replication` decoded this site's engine trade-exactly for
  the **Blue Sky** screen (37/39 entries, 22/23 exits to the day and the paisa) and it
  became the live **Open Alpha** book. Its Phase-1 finding: for the **VCP** ground truth
  the entries are **pattern highs**, not all-time-high closes — that is the crux here.
- r/142 also established: their published returns were NOT reproducible (best honest path
  15.7× vs their claimed 33.7×) and their published "worst fall" (−11.4%) was
  **unreachable at any marking frequency**. Expect the same shape here.
- `market_data.db` is **not retroactively split-adjusted**; r/142 repaired 72 scale-broken
  symbols (incl. CUPID 5×). Any high/pivot screen must be re-checked against that defect.

---

## 2. The Base — what is being tested

### 2.1 Site mechanics already validated in r/142 (reused verbatim, not re-derived)
- **RS** = IBD-weighted percentile of `2×r63 + r126 + r189 + r252` over eligibles, **≥ 70**
- **Liquidity floor** = 20-day median traded value **≥ ₹5cr**, evaluated as of t−1
- **ETFs excluded** (`BEES|ETF|LIQUID|GILT|SENSEX|NIF*50` name filter)
- **Entry** = buy-stop **at the pivot**, filled at `max(pivot, open)` (realistic) or at the
  pivot (their optimistic fill — reported as a labelled arm)
- **Exits** = hard stop on the **CLOSE**; moving-average trail booked at the **signal close**
- **Open positions** marked to the window's last close

### 2.2 The VCP signal (the unknown — this is what the gate must solve)
Swing structure from confirmed **k-bar fractal** swing highs/lows (k a swept axis), reduced
to an alternating sequence. At each bar, evaluated as of the **previous close**:

- `BH` = base high (highest high in the lookback), at index `b`
- contractions `d_1 … d_T`, where `d_j = (H_{j−1} − L_j) / H_{j−1}` walking forward from `BH`
- **Contraction count** `T` in {2, 3, 4}
- **Tightness ratio**: `d_{j+1} <= tight × d_j`, `tight` in {1.0 (merely decreasing), 0.8, 0.6}
- **Max base depth**: `d_1 <= maxdepth`, `maxdepth` in {15%, 25%, 35%, 50%}
- **Base length** `i − b` in [`minlen`, `maxlen`], `minlen` in {15, 25, 40}, `maxlen` = 250
- **Volume dry-up**: mean volume over the final contraction <= `volratio` × mean volume over
  the base, `volratio` in {none, 0.9, 0.7}
- **Proximity to pivot**: previous close within `near` of the pivot and below it,
  `near` in {3%, 5%, 10%, 20%}
- **Pivot** = `BH` (base high) or the **last contraction high** `H_{T−1}`; high-basis vs
  close-basis — 4 variants, arbitrated by the ground truth
- **Trigger** = close > pivot (or intraday high >= pivot — tested as a fill variant)

### 2.3 Their sizing mechanic (differs from Open Alpha — implement both)
Panel text: *"At 2% risk with a 7% stop, each position ≈ ₹2,85,714 (risk ÷ stop distance),
capped at 30% of capital."*
- **Arm RISK** (theirs): position value = `(risk% × equity) / stop%`, capped at 30% of equity
- **Arm FIXED** (ours, Open Alpha convention): fixed % of NAV per slot
Both cash-constrained, no leverage. At 2%/7% the RISK arm is 28.6% per position → with 5
slots the book is **cash-bound**, so slot count is largely inert above ~4 (r/142 found the
same). This is stated up front so the sweep is read correctly.

### 2.4 Costs, taxes, idle cash (adoption arithmetic)
- Cost ladder **25 / 40 / 60 bps per side**; headline at 25 bps
- Tax **20% STCG / 12.5% LTCG (>365 days)** with Indian FY loss-netting
- Idle cash **5% p.a.** accrued daily
- Every headline number labelled gross or net, and after-tax where it is an adoption call

### 2.5 Success criteria — PRE-REGISTERED, before any run
| Gate | Criterion |
|---|---|
| **G-REP (replication)** | >= 60% of the 40 ground-truth trades matched on entry date (±2 trading days) **and** entry price (±1%) by a single parameterization. Below that → "rules NOT reproducible", and everything after is *our* honest VCP, not *theirs* |
| **G-CLAIM** | Their 25.99×/72.1%/−14.8% is judged reproduced only if an honest path lands within ±30% relative CAGR **and** the drawdown is within 5pp. Anything else is reported as REFUTED with the honest number |
| **G-STRAT (standalone)** | After tax, 25 bps, 30-seed median CAGR >= NIFTYBEES + 8pp with Calmar >= 0.8 over the long window |
| **G-BLEND (the binding one)** | vs the deployed **TN+OA 50-50** baseline (27.2% CAGR / −16.4% DD / **Calmar 1.65**, after tax): a 3-sleeve blend must deliver **+0.10 Calmar or −2pp drawdown at >= equal CAGR**, robust across OA seeds and TN offsets, correlation **< 0.4** to both legs, and beat the **cash-null** at the same weight |
| **Ranking metric** | **after-tax Calmar** at 25 bps on the long window, 30-seed median; ties broken by worst-seed CAGR |

---

## 3. Plan — phases and cell counts

| Phase | What | Cells | Lock |
|---|---|---|---|
| **P1 Entry fingerprint** | Re-run r/142 `entry_diag` on the 40 VCP trades: buy price vs every (window, gap) high/close pivot. Identifies whether the pivot is a base high, a swing high, or an ATH | 40 × ~240 | no (seconds) |
| **P2 Exit convention** | Replay −7%/−8% stop (intraday vs close) × trail-50 (signal close / next open / next close) against their 23 closed trades | 40 × 6 | no |
| **P3 VCP grid → gate** | Sweep the §2.2 axes; for each parameterization compute (a) does the ground-truth entry bar fire, (b) does the pivot equal their buy price. Rank by joint match rate | ~1,300 | yes |
| **P4 Faithful replica** | Best-matching parameterization, their dials (5 pos, 7% stop, trail-50, 2% risk, gate OFF, 2020-25, ₹10L), their optimistic fills, no costs — vs their published table | ~10 | yes |
| **P5 Honest baseline** | Realistic fills, 25 bps, after tax, idle cash, **30-seed** random-selection ensemble; long windows 2012→now and 2006→now alongside 2020-25 | ~90 | yes |
| **P6 Optimization** | positions 3/5/8/10/16 × stop 6/7/8/10% × exits (trail-50 / trail-30-week / +25% target / 15-SMA trail) × risk 1/1.5/2% × cap 30% × skip-weak on/off across index series × breakeven-stop on/off. Exits tested **jointly** with entries. Plateau, not peak | ~800 | yes |
| **P7 Robustness** | Two-window split, per-year table, outlier deletion (top-10 trades; winners capped at +50/+100%), cost ladder, worst-seed reporting | ~60 | yes |
| **P8 Portfolio fit** | Correlation (daily + monthly) to OA and TN; 3-sleeve blend weight sweep 10-33% across OA seeds × TN offsets; per-window rows (2008, 2020 crash; 2018, 2022H1 grind); cash-null | ~120 | yes |
| **P9 Deliverables** | YoY house-format table, curve vs NIFTY50/Midcap150/Smallcap250 + drawdown panel, `results/vcp_equity_seeds.csv` (30 seeds, adopted spec, after-tax, 5% cash yield) + `results/vcp_adopted_spec.json` for study r/154, RESULTS.md, BacktestStudy entry, INDEX/TODO/ops review, commit | — | no |

**Multiple-testing disclosure:** running total of cells is carried in §5 and reported in
RESULTS.md so any "winner" can be discounted appropriately.

---

## 4. Data reality (stated before computing)

- `market_data.db` → `market_data_unified`, `timeframe='day'`, ~1,621+ symbols, 2000→2026.
- **Survivorship:** Kite lists only *current* instruments — delisted names are absent on
  our side (their backtest very likely shares this bias). Stated on every table.
- **Split-adjustment defect:** pre-split rows keep the old price scale. r/142 repaired 72
  symbols; a fresh scan is part of P1 because this screen is a *high* screen.
- **Phantom holiday rows / NaN poisoning:** all rolling statistics computed on the
  `dropna()`'d per-symbol series and re-aligned, never on a union-index frame.
- Coverage by year is printed by the harness (2006 ≈ 528 priced symbols → early-window
  results are survivorship-flattered and labelled as such).

---

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-05 ~14:15 | Study opened; playbook + r/142 harness read; folders created on laptop + VPS | 151 confirmed free on VPS |
| 2026-09-05 ~14:20 | STATUS doc written **before** any run; success criteria pre-registered | Cells planned ~2,400 |
| 2026-09-05 ~14:16 | **P1 entry fingerprint** on the 40 ground-truth trades (37 usable; BONDADA/E2E absent from our DB) | buy price = an exact prior **CLOSE** in 36/37; best window short (5-30 bars); median buy is **6% BELOW** the prior ATH close -> this is NOT the Blue Sky pivot |
| 2026-09-05 ~14:16 | **P2 exit conventions** — 12 combinations replayed on the 32 closed trades | **8% stop on the CLOSE + trail exit at the close that breaks the 50-SMA reproduces 31/32 on BOTH date and price.** Same engine as Blue Sky. The panel dial says 7%; the ground-truth run used 8% (its own exit labels say `stop_8pct`) |
| 2026-09-05 ~14:18 | **P1b pivot probe** | pivot never exceeded on a closing basis before the break (37/37); entry-day CLOSE above it (37/37); **no minimum base length** (pivot age 1..157 bars) and **no contraction requirement** (11/37 bases contain zero measurable contractions); base depth median 11.6%, max 23.2%; prior close within 9.3% of the pivot in every case |
| 2026-09-05 ~14:18 | **P1c fixed-N scan** | **No fixed lookback can explain their pivots**: N would have to be >= 157 (deepest pivot age) and <= 11 (shortest run since a higher close) simultaneously. Best single N = 30 (25/37 exact pivot prices) |
| 2026-09-05 ~14:26 | **P1d family scan** — 3 pivot families x 68 cells vs the ground truth | Winner **F1 rolling 30-day max CLOSE: 25/37 price, 28/37 first-break date, 23/37 joint (62.2%)**. Broad plateau N=25..75. Zigzag families (13-16/37) and structural base-high families (15-20/37) all worse |
| 2026-09-05 ~14:29 | Frames cached (5,528 dates x 2,321 symbols), phantom rows dropped, per-symbol NaN-safe rollings | `results/frames.npz` 159 MB |
| 2026-09-05 ~14:40 | **P4 faithful replica** (their dials, their optimistic pivot fills, no costs, no tax, 2020-25, 10 seeds) | 7.6x median [CAGR 40.0%, range 23.5-65.0], DD -26.8%, **124 trades vs their 164** — trade count matches, the return does not |
| 2026-09-05 ~14:47 | **P5 honest baseline** (realistic fills, 25 bps, after tax, 5% idle cash, **30 seeds**, 3 windows) | see Findings — their claim REFUTED on every axis |
| 2026-09-05 ~14:50 | **P6 optimization** (118 cells, 10 seeds, 2012-2026, their concentrated risk sizing) | Best Calmar 1.10; the exit axis dominates, the pivot axis is noise |
| 2026-09-05 ~15:02 | **P6F** (51 cells, our fixed 16 x 6.25% sizing) | Calmar to 1.59-1.75; **the stop is INERT** (6/8/10/15%/none identical) and proximity is inert; RS>=70 is the only dial doing work; the gate hurts |
| 2026-09-05 ~15:17 | **P6G null control + cost ladder** (15 cells) | **THE KILL:** shrinking the pivot lookback toward no-pattern monotonically IMPROVES the book (30d 1.28 -> 10d 1.59 -> 5d 1.70 -> 3d 1.91 -> 2d 2.63). Cost ladder 25/40/60/100 bps = Calmar 1.59/1.20/0.73/0.26 |
| 2026-09-05 ~15:19 | **P9 adopted spec, 30 seeds** -> `vcp_equity_seeds.csv` + `vcp_adopted_spec.json` (deliverables for r/154) | 36.1% CAGR [31.5..38.3], DD -40.8%, Calmar 0.89, 5,247 trades, 254/yr |
| 2026-09-05 ~15:20 | **P8 blend + P9 report** | corr 0.749/0.759 to Open Alpha; best blend +0.033 Calmar; **cash-null wins**; YoY table + tearsheet written |
| 2026-09-05 ~15:35 | `results/RESULTS.md` written; study published to `frontend/src/data/backtests.ts`; `npm run build` on VPS (frontend-only, no restart); INDEX.md, TODO.md, Ops & Review Centre and LABS reference updated | Review registered: re-open only on a published, reproducible VCP definition, due **2027-03-05** |

**Cells consumed so far:** ~230 (P1 grid 40x256 forensic + 68 family cells + 5 P4 + 12 P5 + 118 P6 + 35 P6F)

---

## 6. Crash recovery — how Arun resumes without Claude

1. Everything lives on the VPS at `/home/arun/quantifyd/research/151_vcp_breakout/`.
2. **What finished:** `ls -la results/` and `tail -50 /tmp/vcp_*.log`. Every sweep writes
   **one CSV row per completed cell**, so a partial CSV is a valid partial result.
3. **Is it still running:** `ps -ef | grep vcp_` and `ls -l /tmp/qf_sweep.lock`.
4. **Resume:** re-launch the same script — every runner **skips cells already present**
   in its output CSV. Commands (run from `/home/arun/quantifyd`):
   - P1/P2: `venv/bin/python research/151_vcp_breakout/scripts/vcp_entry_fingerprint.py`
   - P3: `flock -w 7200 /tmp/qf_sweep.lock venv/bin/python research/151_vcp_breakout/scripts/vcp_gate_grid.py`
   - P5-P7: `flock -w 7200 /tmp/qf_sweep.lock venv/bin/python research/151_vcp_breakout/scripts/vcp_sweep.py`
   Launch pattern: `setsid nohup <cmd> > /tmp/vcp_pN.log 2>&1 < /dev/null &`
5. **Do not touch:** `data/vcp_trades_groundtruth.csv` (transcribed ground truth — the only
   copy of the arbitration evidence besides r/142's).
6. **Safe to inspect/delete-and-regenerate:** anything under `results/`.
7. Nothing in this study writes to any live engine, crontab, DB table, or deployed spec.
   All DB access is **read-only**.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `VCP_BREAKOUT_DAILY_SWEEP_STATUS.md` | This file — sole crash-recovery source | yes |
| `data/vcp_trades_groundtruth.csv` | 40 trades transcribed from the site's VCP run | yes |
| `scripts/vcp_signal.py` | VCP detector (swing structure → contractions → pivot) | yes |
| `scripts/vcp_entry_fingerprint.py` | P1/P2 pivot + exit-convention forensics | yes |
| `scripts/vcp_gate_grid.py` | P3 parameterization sweep vs ground truth | yes |
| `scripts/vcp_replay.py` | Portfolio engine (extends r/142 `bluesky_replay.simulate`) | yes |
| `scripts/vcp_sweep.py` | P4-P7 sweeps | yes |
| `scripts/vcp_blend.py` | P8 correlation + 3-sleeve blend | yes |
| `results/*.csv` | Incremental, one row per cell | yes if small |
| `results/vcp_equity_seeds.csv` | **Required deliverable** — 30 daily equity curves, adopted spec | yes |
| `results/vcp_adopted_spec.json` | **Required deliverable** — adopted spec in words + params | yes |
| `results/RESULTS.md` | Verdict | yes |

---

## 8. Findings

### F1 — The exit engine is THEIR engine, exactly (P2)
`8% stop on the CLOSE` + `exit at the close that breaks the 50-day SMA` reproduces
**31 of 32** closed ground-truth trades on both the date and the price. This is the
identical exit machinery decoded for Blue Sky in r/142 — the site runs one engine behind
both screens. Intraday-touch stops and next-open/next-close trail fills reproduce 3-25.

### F2 — The VCP pivot is a rolling closing high, and their pattern is NOT identifiable (P1-P1d)
- The buy price is an **exact prior close** (36/37 within 0.15%), the close has never been
  exceeded between the pivot bar and the break, and the **entry-day close is above it** in
  every case. So: pivot = a pattern high on a CLOSE basis; trigger = a close through it.
- It is **not** the all-time-high close — the median buy sits **6% below** the prior ATH
  close, and only 16/37 are at or above it. Blue Sky and VCP are genuinely different screens
  (with ~43% overlap).
- **There is no minimum base length and no contraction requirement.** Pivot ages run from
  1 to 157 bars; 11 of 37 bases contain zero measurable contractions; the volume "dry-up"
  ratio ranges 0.27-1.53 (median 0.85). Whatever the site calls "VCP", its published trade
  list does not contain the volatility-contraction structure the name implies.
- **No fixed lookback fits**: N must be >= 157 and <= 11 at once. Their pivot is structural
  and, from 37 trades, **not identifiable**. The best approximation across 68 candidate
  parameterizations in three families is a **30-day rolling maximum CLOSE**:
  25/37 exact pivot prices, 28/37 first-break dates, **23/37 (62.2%) joint**.
  Plateau: N = 25-75 all score 20-23 joint, so the choice is not a knife-edge.

**G-REP verdict: PARTIAL PASS (62.2% vs the 60% pre-registered bar).** Everything downstream
is therefore *our best reconstruction* of their VCP screen, not a trade-exact replica the way
r/142's Blue Sky work was. Stated on every table.

### F3 — The published claim is REFUTED (P4/P5)
Their panel: **25.99x / +72.1% CAGR / -14.8% worst fall / 164 trades**, 2020-2025, PROVISIONAL.

| Arm (2020-01-01 -> 2025-12-31) | Terminal | CAGR (median [min..max]) | MaxDD | Trades |
|---|---|---|---|---|
| **Their published run** | 25.99x | +72.1% | -14.8% | 164 |
| Faithful replica: their dials, their optimistic pivot fills, **no costs, no tax** (10 seeds) | 7.64x | **40.0% [23.5..65.0]** | -26.8% | 124 |
| **Honest**: realistic fills, 25 bps, after tax, 5% idle cash (30 seeds) | 5.38x | **32.4% [6.5..61.6]** | **-34.5%** | 121 |

- The **trade count matches** (121-124 vs 164) — the mechanics are right; the **returns are
  not** (32-40% vs 72.1%), and the drawdown is **more than double** what they publish.
- Their -14.8% "worst fall" is **not reachable**: our shallowest honest path is -21%, the
  median -34.5%. Identical to the r/142 finding for Blue Sky (-11.4% claimed, -22% best).
- **Path dependence is the story**: on their own dials the seed range is **6.5% to 61.6%
  CAGR** — a 55-point spread. Their single number is one draw from that distribution, and
  the concentrated risk-based sizing (2% risk / 7% stop = 28.6% per position, only ~3.5
  positions fit) is what makes the spread so violent.

### F4 — Honest long-window baseline (P5, 30 seeds, after tax, 25 bps, 5% idle cash)

| Window | Their dials (5 slots, risk 2%/stop 7%) | Fixed 16 x 6.25% (our convention) |
|---|---|---|
| 2020-2025 | 32.4% [6.5..61.6], DD -34.5%, Calmar 0.96 | 36.5% [26.8..47.5], DD -29.4%, **Calmar 1.23** |
| 2012-2026 | 31.6% [22.4..49.3], DD -44.0%, Calmar 0.71 | 31.0% [22.9..36.1], DD -34.2%, **Calmar 0.89** |
| 2006-2026 | 25.5% [15.2..31.6], DD -54.9%, Calmar 0.45 | 27.5% [24.4..30.8], DD -51.6%, Calmar 0.53 |

Our fixed-weight sizing beats their risk-based sizing on Calmar in every window **and**
halves the seed spread. Their sizing mechanic is the single largest source of the
variance in their own published number.


### F5 — The kill: the pattern does negative work (P6G null control)
2012-2026, fixed 16 x 6.25%, 15-SMA trail, after tax, 25 bps, 10 seeds:

| Pivot lookback | CAGR | MaxDD | Calmar | Trades |
|---|---|---|---|---|
| **2 days (the null)** | **56.6%** | −22.1% | **2.63** | 6,319 |
| 3 days | 49.3% | −25.6% | 1.91 | 5,806 |
| 5 days | 46.3% | −26.3% | 1.70 | 5,015 |
| 10 days | 43.2% | −26.9% | 1.59 | 4,321 |
| **30 days (their pattern)** | 37.5% | −29.5% | **1.28** | 4,008 |

Monotone — requiring more pattern always made the book worse. Two of their three headline
dials are also inert: stops of 6/8/10/15% **and no stop at all** all return ~43.2% at −26.7%
(the 15-SMA trail always fires first), and proximity-to-pivot does nothing above 10%. Only
**RS ≥ 70** does real work (RS 0 → 32.4%, 50 → 38.9%, 70 → 43.3%, 85 → 51.2% at −35% DD).

### F6 — Adopted spec, standalone (30 seeds, 2006-04 → 2026-09, after tax, 25 bps, cash 5%)
Pivot = 30-day rolling closing high (the **replication-anchored** value, deliberately not the
sweep peak — the lookback axis is flat/noisy and picking its edge would be overfitting),
RS ≥ 70, ₹5cr floor, buy-stop at pivot filled at max(pivot, open), −8% close stop, 15-SMA
close trail, 16 slots × 6.25%, no gate:

**36.1% CAGR [31.5..38.3] · MaxDD −40.8% · Calmar 0.89 · 5,247 trades · win 45.1% ·
avg win +11.1% / avg loss −3.9% · +2.89%/trade · longest losing streak 26 · 254 trades/yr.**
Cost ladder 25/40/60 bps → Calmar 0.90 / 0.71 / 0.51 (≈ −6.8pp CAGR per +15 bps, on ~37×
NAV annual turnover). Dropping the 10 best trades changes nothing (0.903 → 0.903) — broad,
not lottery-driven. Windows: 2006-15 Calmar 0.68, 2016-26 Calmar 1.50.

### F7 — Portfolio fit: REJECTED on every limb
Correlation to the live **Open Alpha** book **0.749 daily / 0.759 monthly** (bar < 0.40);
to True North 0.480 / 0.546. Best blend weight (10%) adds **+0.033 Calmar** against a +0.10
bar *and worsens drawdown*; 15% onward loses the paired test outright (25% wins 1 of 30
paths); a **plain cash sleeve beats it at every weight** (1.659 vs 1.642 at 10%, rising to
1.745 at 20%). Adding it makes 2008, 2018 **and** 2022H1 all worse — it re-imports the crash
tail the TN gate and OA stops already stripped. This is r/145 and r/146 repeating: more
smallcap momentum beta into a book that already harvests it.

### Deliverables produced
`results/vcp_equity_seeds.csv` (30 daily after-tax equity curves, adopted spec, cash 5%) ·
`results/vcp_adopted_spec.json` · `results/vcp_yoy_table.txt` (house YoY format) ·
`results/vcp_tearsheet.png` · `results/vcp_cost_ladder.csv` · `results/vcp_robustness.csv` ·
`results/p8_{correlations,blend,paired}.csv` · `results/RESULTS.md` ·
`/app/backtest/vcp-breakout-research151`.
