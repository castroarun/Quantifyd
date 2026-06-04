# MTF Compression Breakout — Uptrend + Weekly-CPR + Narrow-CPR/NR + PDR Break (research/55)

STATUS: **DONE — VERDICT: NO ALPHA (beta).** 4 tests (large/small × 5min/daily,
2018–26). Compression+volume breakout does NOT beat holding the uptrend; volume
spike consistently fails. Only crumb: +0.04R on tight R-targets; compression has
*defensive* value in 2022 (use as regime filter, not entry). See `results/RESULTS.md`.

> Follows `research/QUANT_RESEARCH_PLAYBOOK.md`. STATUS sections 1–4 written
> BEFORE any run. Recovery in §6. Cumulative findings posted in chat + §8.

---

## 1. The Ask

**What you asked:** "Intraday/positional breakout based on multi-timeframe.
Daily TF — stock in uptrend. 30-min — price above weekly CPR. 5-min — previous
day tight consolidation range, today narrow CPR + prev-day range breakout, volume
considered. e.g. TDPOWERSYS. Interchange/add layers; you have 300+ stocks of data
for years — optimize and find a system that works; free on SL/target/trailing."

**What we're actually testing:** Does a **confluence long breakout** —
(daily) uptrend AND (30m) price above weekly CPR AND (intraday) prior-day tight
range + narrow daily CPR + break of prior-day(/week) high + volume surge —
produce **net-of-cost positive expectancy that BEATS simply holding the same
uptrending stock** (i.e. real alpha, not bull-market beta), across a liquid
300+-stock universe with years of 5-min data, under optimized exits?

## 1b. Prior art (reuse, don't re-litigate — playbook §1, §11)

- **research/40** (`volsurge_pdr_break_weekly_cpr`) — SAME family; its `signal_lib.py`
  (weekly_cpr, daily_trend, resample_5m, is_clean_candle, range_escape,
  volume_surge, clear_room) is **imported wholesale** here. 40's verdict was
  tiny-n per-stock cherry-picks (SIGNAL not STRATEGY).
- **research/31** cpr_compression_breakout; **research/44** prev-day-range / NR7
  (raw breakeven; edge only with compression+high-beta+trend+SWING).
- **research/49** (mine, today) — the hard lesson: a long-only positional breakout
  looked great (+0.70R) but the **placebo proved it was pure beta**. So this study
  bakes the beta benchmark in from the start.

## 1c. Falsification criterion (decided BEFORE results — the alpha test)

The confluence SIGNAL must beat its own **TREND_BASELINE** (enter the same
uptrending name on an arbitrary day, identical exits) by a **meaningful margin**
on net expectancy AND PF — not just be positive. Concretely: SIGNAL net@6bp ≥
TREND_BASELINE net@6bp **+0.10R** with PF uplift, on n ≥ 300 across ≥5 names, and
each compression/volume layer must add (ablation monotonic). If SIGNAL ≈ BASELINE
→ verdict **NO ALPHA (beta)**, same as research/49. Also report per-year (must not
rely on one bull year) and net of realistic cost.

## 2. Economic hypothesis (G0)

Coiled range (low vol) inside an uptrend → energy builds; a high-volume break of
the prior-day/week high above the weekly pivot = trend-continuation expansion as
range sellers' stops trigger and breakout buyers pile in (under-reaction +
short-covering). Counterparty: range/mean-reversion traders, stops above the box.
Decay risk: HIGH (well-known pattern; competed). Alpha only if the *compression +
volume + location* genuinely select better than the trend alone.

## 3. The Base (mechanics locked for G1)

- **Bars:** 5-min → 15-min trigger frame (session-anchored, reuse 40 `resample_5m`);
  daily for trend/CPR/NR/ATR/exits; weekly CPR from prior week.
- **Daily filter (causal, as of D-1):** uptrend via `daily_trend('sma50')`
  (close>SMA50). Axis later: sma200 / hh20.
- **Location (30m≈intraday):** trigger 15-min close **> weekly CPR top** (price
  above weekly CPR). 
- **Compression (causal, as of D-1):** (a) prior-day **NR7** (D-1 range = min of
  last 7 daily ranges) AND/OR (b) **narrow daily CPR** (today's daily-CPR width%
  from D-1 H/L/C below a threshold). Tested together and ablated.
- **Breakout trigger (intraday):** first 15-min candle whose close escapes
  **max(prev-day high, prev-week high)** (40 `range_escape`), is a **clean
  directional bar** (40 `is_clean_candle`), with a **volume surge** (vol ≥
  k×trailing same-frame baseline). Enter **next 15-min bar open** (causal).
- **Exits (optimize):** positional, managed on daily bars (the example is a
  multi-day breakout). G1 set: daily **Supertrend(10,3)** flip, **Chandelier 3×ATR**,
  **Hard SL 1×ATR**, **R_3R** target, **MAXHOLD_10**; R = daily ATR(14). 30-day cap.
- **Direction:** LONG only. **Cost:** 6 bps round-trip; report gross+net (0/6/12).
- **Universe (G1):** ~12 liquid names incl. the example's character (TDPOWERSYS,
  BHARATFORG, ADANIENT + large-caps). Full run later = turnover-filtered 5-min set.

## 4. Plan — entry-ablation arms (the alpha test) × exit policies

Arms (same exits, same universe-days):
| Arm | Conditions |
|---|---|
| **SIGNAL** | uptrend + above-weekly-CPR + (NR7 & narrowCPR) + range-escape + clean + vol-surge |
| **NO_VOL** | drop the volume-surge gate |
| **NO_COMPRESSION** | drop NR7 + narrow-CPR (uptrend + above-CPR + break only) |
| **TREND_BASELINE** | enter first 15-min bar of EVERY uptrend day (no location/compression/break/vol) — the **beta benchmark** |

Exit policies: 5 (above). Cost sensitivity: 0/6/12 bps. Metric: net@6bp R, PF,
WR, per-year. Decision: ablation must be monotone and SIGNAL must beat BASELINE.

---

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-06-04 ~09:50 | STATUS 1–4 written | research/55 created |
| 2026-06-04 ~09:50 | G1 probe authored | reuses research/40 signal_lib |
| 2026-06-04 ~09:55 | G1 DONE (12 names) | looked good (+0.34 alpha) but small-n |
| 2026-06-04 ~10:05 | G2 DONE (218 largecap) | **NO ALPHA at scale (n1424): SIGNAL≈BASELINE → beta. Large/midcap = dead.** Also: data only densifies 2024+ |
| 2026-06-04 ~10:10 | Pivot to SMALLCAP | user: smallcaps + visible vol spike; refs TDPOWERSYS/DATAPATTNS/KMEW (runners, NOT to special-case = holdings) |
| 2026-06-04 ~10:10 | G3 smallcap 5min RUNNING | 160 names turnover<50cr; DATA ONLY 2024-03+ (no smallcap 5min before) → 2yr study |
| 2026-06-04 ~10:15 | KMEW cue → drop weekly-CPR | weekly-CPR gate NOT mandatory; core = uptrend+multiday-coil+narrow-CPR+vol+breakout |
| 2026-06-04 ~10:20 | G4 DAILY full-universe queued | 1600+ names 2018-26 (multi-regime; incl. example names which lack 5min). The honest home for a multi-MONTH positional runner |

## 6. Crash recovery

- Probe: `venv/bin/python research/55_mtf_compression_breakout/scripts/g1_probe.py`
- Imports `research/40_volsurge_pdr_break_weekly_cpr/scripts/signal_lib.py` (must exist).
- Self-contained DB loaders; reads only `backtest_data/market_data.db`.
- Writes `results/g1_trades.csv` + prints per-arm/per-policy summary. Re-runnable
  (overwrites). No live-trading state touched. VPS-canonical host.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `MTF_COMPRESSION_BREAKOUT_MULTITF_SWEEP_STATUS.md` | this doc | yes |
| `scripts/g1_probe.py` | G1 confluence + ablation + benchmark | yes |
| `results/g1_trades.csv` | per-trade × arm × policy | small → yes |
| `results/RESULTS.md` | verdict | yes |

## 8. Findings

### G1 (12 names, 2018–2026) — PASSES G1, promising

Net@6bp R / PF, key arms (full table in chat + g1_trades.csv):

| Exit | SIGNAL n157 | NO_VOL n214 | NO_COMP n1078 | BASELINE n5847 | alpha delta |
|---|--:|--:|--:|--:|--:|
| HARD_SL | +0.576 / 1.76 | +0.758 / 2.03 | +0.296 / 1.38 | +0.239 / 1.31 | **+0.338** |
| R_3R | +0.276 / 1.40 | +0.401 / 1.61 | +0.183 / 1.26 | +0.125 / 1.17 | **+0.150** |
| CHANDELIER | +0.401 | +0.588 | +0.455 | +0.355 | +0.046 |
| SUPERTREND | +0.817 | +0.968 | +0.825 | +0.691 | +0.126 |
| MAXHOLD_10 | +0.159 | +0.255 | +0.130 | +0.134 | +0.025 |

1. **Compression (NR7 + narrow CPR) adds REAL alpha** vs the trend-baseline — but
   only on **tight stops** (HARD_SL/R_3R), where the coil gives a near logical
   stop → asymmetric payoff. Wide trailing exits wash it into beta.
2. **Volume filter HURTS** — NO_VOL > SIGNAL on every policy → **drop the volume
   gate** (the edge is compression, not volume).
3. **CAVEAT: n=157 too small** (bar was ≥300); 20 cells scanned (multiple testing).
   Needs broad universe + per-year stability before believing it. → G2.

### Decisions carried into G2
- Drop volume gate (or keep only as a loose optional axis).
- Focus exits on TIGHT stops (HARD_SL, R_2R/3R) where the entry edge lives.
- Add per-year table; broaden to ~218 liquid names for n in the thousands.
- TDPOWERSYS has NO 5-min data in the DB (example unusable directly) — pattern
  must generalize across the universe.

### G2 (broad universe) — RUNNING (see results/g2_trades.csv, incremental)
