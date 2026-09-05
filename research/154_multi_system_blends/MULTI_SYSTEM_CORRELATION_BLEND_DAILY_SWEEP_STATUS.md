# Six-Sleeve Correlation & Blend Matrix — OA · TN · VCP · MYB · IPO · GOLD

**STATUS: DONE** — verdict **STRATEGY (candidate)**, published at `/app/backtest/multi-system-blends-research154`
Started 2026-09-05 15:43 IST · Host: VPS 94.136.185.54 · `research/154_multi_system_blends/`

---

## 1. The Ask

**What Arun asked (verbatim):**
> "also find the correlations between each of these systems including our oa and tn in all
> possible combinations."

**What we are actually testing:**
Given the six equity/asset sleeves this project now has a defensible NAV for — the two LIVE
books (Open Alpha, True North) plus the four research sleeves produced in this batch
(VCP r/151 NO EDGE, MYB r/152 SIGNAL, IPO r/153 STRATEGY candidate, GOLD r/147 candidate) —

1. what is the **pairwise correlation** of every one of the 15 pairs, on daily and monthly
   returns, on an **explicitly stated common window**;
2. for every one of the **57 subsets of size ≥ 2**, what is the blended book's
   CAGR / MaxDD / Calmar, monthly-rebalanced, after tax, **paired across the same
   (OA seed × TN offset) path** and reported as median [min..max];
3. does any subset beat the **deployed TN+OA 50-50 pair** by the pre-registered complement
   bar, and does it beat its own **cash-null** at the same weight;
4. at **position level**, not just correlation — do two sleeves hold the same names on the
   same days?

Rejected sleeves (VCP, MYB) are included deliberately: knowing *why* a rejected sleeve fails
in a blend is part of the answer.

---

## 2. The Base — the six sleeves and their provenance

| Sleeve | What it is | Spec | Source of NAV | Paths |
|---|---|---|---|---|
| **OA** — Open Alpha | LIVE ₹10L real money. Close above prior all-time-high close; RS ≥ 70; TV ≥ ₹5cr | 16 slots @ 6.25%, no market gate, −8% close stop, close < SMA-15 trail, buy-stop at pivot filled max(pivot, open), 25 bps/side | **generated here** — `research/142/scripts/bluesky_replay.py`, 30 random-selection seeds | 30 seeds |
| **TN** — True North | LIVE momentum book. Nifty-200, top-8 equal weight, monthly rebalance | NIFTYBEES 100-SMA weekly liquidate-all gate, 15-day-low Donchian stop | **generated here** — `research/144/scripts/tn_sweep.py` `run(ctx, offset=o, tax=True)` (deterministic) | 12 rebalance-day offsets |
| **VCP** (r/151, **NO EDGE**) | 30-day-high pivot breakout, VCP screen approximation | `results/vcp_adopted_spec.json` | `research/151/results/vcp_equity_seeds.csv` | 30 seeds |
| **MYB** (r/152, **SIGNAL**, not adopted) | 3-year-high close that is NOT an all-time high | `results/myb_adopted_spec.json` | `research/152/results/myb_equity_seeds.csv` | 30 seeds |
| **IPO** (r/153, **STRATEGY candidate**) | IPO-Base MID: listed ≤ 6 months, 25-day base, pivot = highest close, ≤30% depth | `results/ipo_adopted_spec.json` (8 slots @ 18.75%, −8% stop, SMA-20 trail, +25% TP) | `research/153/results/ipo_equity_seeds.csv` | 30 seeds |
| **GOLD** (r/147, candidate) | GOLDBEES buy-and-hold | plain B&H, no tax modelled (B&H genuinely defers) | GOLDBEES daily from `market_data.db` **2015-01+**; gold-INR reconstruction before that, chained and **labelled**. *(Planned to reuse r/147's `gold_inr_ref.csv`; at 15:50 that series was found to be missing **40 of 274 months** and was **rebuilt at daily resolution** — see the status log and `scripts/rebuild_gold.py`. `results/gold_nav.csv` is what was actually used.)* | deterministic |

All equity sleeves are **after Indian tax** (20% STCG / 12.5% LTCG, FY 1-April netting with
loss carry-forward), **net of 25 bps per side**, **idle cash 5% p.a.**

### Path pairing convention (this is what makes the comparison paired)
A **path** = (OA seed *s* ∈ 1..30, TN offset *o* ∈ 1..12) → **360 paths**. Within a path, the
stochastic research sleeves (VCP, MYB, IPO) use the **same seed index *s***. Every A-vs-B
comparison is computed on the same path and the *distribution of differences* is reported —
never unpaired medians (r/144's DD10 lesson).

### Blend mechanics
Monthly rebalance to target weights. Primary NAV convention is **month-end marks**
(`(1 + Σ wᵢ·rᵢ,month).cumprod()`), identical to r/146 / r/151 / r/152 / r/153 so the numbers
are directly comparable to the published baselines. A **daily-marked** panel (monthly
rebalance, daily marks, honest intra-month drawdown) is produced as a robustness check for
every combination that does not need pre-2015 gold.

---

## 3. Windows — stated explicitly, never mixed

The sleeves do **not** share a start date. Mixing a blend that contains 2008 with one that
does not is exactly the error caught and corrected inside r/152. Three separate panels:

| Panel | Window | Sleeves available | Contains |
|---|---|---|---|
| **A — MASTER** | **2010-01-04 → 2026-09-03** | all 6 (gold = reconstruction 2010-2014 + real 2015+) | 2020 crash, 2018 + 2022H1 grinds. **No 2008.** |
| **B — CRASH-HONEST** | **2006-04-03 → 2026-09-03** | 5 (OA, TN, VCP, IPO, GOLD) — **MYB cannot exist** | 2008 GFC, 2020, both grinds |
| **C — REAL-DATA-ONLY** | **2015-01-01 → 2026-09-03** | all 6, **no reconstructed data at all** | 2018, 2020, 2022H1 |

- MYB's history begins 2010-01-04 (its pivot needs 3 years of prior highs).
- GOLDBEES has **no Kite data before 2015** (verified in r/147). Panel A's gold leg uses the
  labelled monthly reconstruction for 2010-2014; every figure that touches it is marked.
- Panel B is the only panel that can answer the 2008 question, and it structurally **cannot
  include MYB**. That is stated, not silently dropped.

---

## 4. Plan — phases and cell counts

| Phase | What | Cells |
|---|---|---|
| **P0** | Build sleeve NAVs: OA × 30 seeds, TN × 12 offsets, GOLD chained series; load VCP/MYB/IPO seed CSVs; coverage print for every series | — |
| **P1** | Pairwise correlation: 15 pairs × {daily, monthly} × 3 panels, seed-median + seed range | 90 |
| **P2** | **All 57 subsets** at equal weight × 3 panels (26 of the 57 are MYB-free so also run on B) | 57 (A) + 26 (B) + 57 (C) = **140** |
| **P3** | Coarse weight sweep for combos clearing the baseline: pairs w ∈ {10,20,25,33,50,67,75,80,90}%; TN+OA-core subsets, satellite weight ∈ {10,20,25,33}% each with TN=OA sharing the rest | ~**600** |
| **P4** | Cash-null at the same weight for every weighted cell; paired delta vs the deployed TN+OA 50-50 baseline + win counts | (same cells, ×2) |
| **P5** | Per-window rows for the finalists: 2008, 2020 (crash) · 2018, 2022H1 (grind) | ~60 |
| **P6** | **Position-level overlap**: signal-date and holding-day overlap for every pair whose trade lists exist (OA, VCP, MYB, IPO; TN via monthly holdings; GOLD n/a) | 10 pairs |
| **P7** | Frontier enumeration (5% grid over OA/TN/IPO/GOLD, 10% grid incl. MYB) against three nulls | **7,293** |
| **P8** | Daily-marked robustness · YoY house-format table (deployed pair, best 3-sleeve, best 4-sleeve, NIFTY 50) · figure · publish | — |

**Total disclosed cells ≈ 890**, each evaluated over up to 360 paired paths.

*(Actual, after the run: **8,172** cells — the plan above under-counted because P7 grew from a
hand-picked shortlist into a full 5%-grid frontier enumeration, 1,767 vectors × 3 panels plus
996 × 2 with MYB. The larger count makes the multiple-testing discount larger, which is why
the plateau requirement below is the binding evidence and not any single cell.)*

### Pre-registered ranking metric and adoption bar (set BEFORE the run)
- **Ranking metric:** Calmar of the monthly-marked blend NAV, **median over the 360 paths**,
  on Panel A.
- **Complement adoption bar** (unchanged from r/146/147/152/153, applied per panel):
  a candidate combination is adopted over the deployed TN+OA 50-50 pair only if, on the same
  panel and **paired on the same path**, it delivers
  **(+0.10 Calmar at ≥ equal CAGR) OR (−2pp MaxDD at ≥ equal CAGR)**, after tax, **AND**
  wins on ≥ 24/30 seeds (80% of paths), **AND** beats the cash-null at the same weight,
  **AND** the added sleeve's correlation to BOTH deployed legs is < ~0.40.
- **Plateau requirement:** a winning weight whose neighbours (±1 grid step) disagree is
  treated as noise, not a finding. Multiple testing is real at ~890 cells: a combination
  must win on a *contiguous* weight range and on ≥ 2 of the 3 panels where it is testable.

### Falsification / kill criterion
If no subset beyond the deployed pair clears the bar on ≥ 2 panels with plateau behaviour,
the verdict is **"the deployed pair plus at most one sleeve"** and we say so plainly.

---

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-05 15:43 | Folder created, STATUS written | sections 1-4 locked before any compute |
| 2026-09-05 15:48 | **P0 done** — OA 30 seeds, TN 12 offsets, GOLD chained | 2 min. TN's 12-offset CAGR spread is **14.9%–25.0%**; r/146 cached only offsets 0/4/8, which misses both tails |
| 2026-09-05 15:50 | **DATA DEFECT** — r/147's cached gold-INR reference is missing **40 of 274 months** | Yahoo's *monthly* GC=F / INR=X candles drop months (43 / 22) and their UTC-offset stamps collide across month boundaries, so `duplicated(keep='last')` deletes the survivor |
| 2026-09-05 15:52 | **Gold rebuilt at DAILY resolution**, zero gaps 2005-01→2026-09 | `rebuild_gold.py`. Validated vs real GOLDBEES: monthly corr **0.878** (old sparse series 0.788), daily corr **0.390** (COMEX-vs-NSE timing → daily gold correlations use real data only), drift −1.00pp/yr |
| 2026-09-05 15:53 | **P1+P2+P3 done** — 90 correlation cells, 143 equal-weight, 646 weighted | 789 cells × 360 paths in 3 s. Baselines recomputed: A 26.63/−15.69/1.685 · B 27.74/−17.01/1.678 · C 30.54/−15.69/1.925 |
| 2026-09-05 15:58 | **Exposure audit** — IPO returns exactly the 5% idle-cash yield in 2008/09/12/13/14 | r/153's own print: **19.6% invested**. A cash null cannot catch this → built the **beta-matched null** (IPO → 19.6% OA + 80.4% cash) |
| 2026-09-05 16:00 | **Window-drawdown bug found and fixed** | Measuring window DD from the window's own first bar hides a peak just before it. 2008's peak is **Dec-2007**. Corrected to measure from the running peak of the full curve → the deployed pair's 2008 DD is **−16.5%**, not −2.4% |
| 2026-09-05 16:02 | **P6 done** — position-level overlap | OA~VCP **87.0%** of OA's signals shared, 48.6% holding-day; OA~IPO **0.0% / 0.0%**; VCP~MYB 90.2% of MYB's. IPO calendar reconstruction validated on 20,244 trades at **100.00%** |
| 2026-09-05 16:06 | **P7 done** — frontier enumeration, 5,301 + 1,992 cells | **197 of 1,767** vectors admitted on all three panels against all three nulls; contiguous plateau |
| 2026-09-05 16:12 | **P8 done** — daily-marked robustness, YoY house table, figure | Daily marking deepens every drawdown and changes no ranking |
| 2026-09-05 16:20 | **Published** — `/app/backtest/multi-system-blends-research154` (HTTP 200), frontend built on VPS, no backend restart | RESULTS.md, INDEX.md, TODO.md, ops review all written; r/152's four-sleeve review marked DONE (delivered by this study) |

---

## 6. Crash recovery — how to resume without Claude

All work is on the **VPS** at `/home/arun/quantifyd/research/154_multi_system_blends/`.

**Check what finished:**
```bash
cd /home/arun/quantifyd/research/154_multi_system_blends
ls -la results/
tail -40 results/p0_build.log       # sleeve NAV construction
tail -40 results/p2_subsets.log     # subset sweep
wc -l results/p2_subsets.csv        # one row per completed cell
```

**Check the run is alive:**
```bash
pgrep -af 'research/154' ; ps -o pid,etime,rss,cmd -p $(pgrep -f 'research/154' | head -1)
```

**Resume (every stage is resume-safe — it skips cells already present in its CSV):**
```bash
cd /home/arun/quantifyd
nice -n 10 venv/bin/python -u research/154_multi_system_blends/scripts/build_sleeves.py \
  >> research/154_multi_system_blends/results/p0_build.log 2>&1
nice -n 10 venv/bin/python -u research/154_multi_system_blends/scripts/blend_matrix.py \
  >> research/154_multi_system_blends/results/p2_subsets.log 2>&1
```

**Do NOT touch:** `results/sleeve_navs.parquet` / `results/oa_navs30.csv` /
`results/tn_navs12.csv` while `build_sleeves.py` is running — they are written atomically at
the end of each sleeve, and a partial read produces silently wrong correlations.

**Safe to inspect at any time:** every `results/*.csv`, `results/*.log`, this STATUS file.

**Nothing in this study touches a live engine, a crontab, or a deployed spec.** It reads
`backtest_data/market_data.db` and prior studies' result CSVs only.

---

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `MULTI_SYSTEM_CORRELATION_BLEND_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/build_sleeves.py` | P0 — builds OA×30, TN×12, GOLD chained; caches all six sleeves | yes |
| `scripts/blend_matrix.py` | P1-P5 — correlations, 57 subsets, weight sweep, cash-null, paired deltas, windows | yes |
| `scripts/overlap.py` | P6 — position-level signal/holding overlap | yes |
| `scripts/report.py` | P7 — YoY house table, charts, publish payload | yes |
| `results/oa_navs30.csv`, `results/tn_navs12.csv`, `results/gold_nav.csv` | sleeve NAVs | yes (small) |
| `results/p1_correlations.csv` | 15 pairs × daily/monthly × 3 panels | yes |
| `results/p2_subsets.csv` | 140 equal-weight subset cells | yes |
| `results/p3_weights.csv` | weight sweep + cash-null + paired deltas | yes |
| `results/p5_windows.csv` | per-window rows for finalists | yes |
| `results/p6_overlap.csv` | position-level overlap | yes |
| `results/RESULTS.md` | the verdict | yes |

---

## 8. Findings

Full detail in `results/RESULTS.md`. The three that matter:

1. **A retraction.** The deployed TN+OA pair's worst drawdown in twenty years is the **2008
   crash at −16.5%** (monthly marks) / −17.15% (daily), not the −2.4% that r/146 and r/151
   reported. Those studies started the 2008 window on 2008-01-01, *after* the December-2007
   peak. The standing structural claim that "the TN gate plus OA's stops have already stripped
   the crash tail, so crash-alpha candidates solve a problem the pair does not have" is
   **withdrawn**. Every per-window drawdown in r/146–153 needs re-auditing for the same
   artefact — registered as a dated obligation.
2. **The book owns one factor.** OA↔VCP is 0.749 daily, and at position level **87.0% of Open
   Alpha's signals are also VCP signals** with 42–49% holding-day overlap. MYB shares **90.2%**
   of its signals with VCP. Only **IPO** (0.211 daily to OA; **0.0% signal and 0.0% holding-day
   overlap — not one shared symbol-day in sixteen years**) and **GOLD** (≈0, negative monthly)
   are genuinely different things.
3. **197 of 1,767 enumerated weight vectors** clear the pre-registered bar on all three panels
   against the pair, a cash null and the beta-matched null, forming one contiguous plateau.
   Recommended under operational constraints: **OA 40 / TN 25 / IPO 20 / GOLD 15 → 28.21% /
   −10.77% / Calmar 2.61** vs the pair's 27.74% / −17.01% / 1.68. Deployable today without an
   unproven sleeve: **OA 60 / TN 15 / GOLD 25 → 28.02% / −13.31% / 2.095**. r/147's 45/45/10 is
   **not** admitted (CAGR shortfall −1.13pp).

**Both registered questions answered.** r/152's MYB+OA-beats-TN+OA reproduces (+0.316 Calmar,
314/360 paths) but is **not actionable** — MYB's three-year pivot makes 2008 unreachable *by
construction*, and every 2006-testable substitute that wins does so on being uncorrelated,
which MYB is not. r/152's 80/10/10 four-sleeve probe is **REFUTED** against a properly
specified gold-only null at the same satellite weight (−0.094 Calmar, wins 91/360).

**Nothing was deployed.** No live engine, crontab, sizing or spec was touched.
