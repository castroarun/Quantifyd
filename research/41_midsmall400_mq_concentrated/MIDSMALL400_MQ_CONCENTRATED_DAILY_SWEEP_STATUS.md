# MidSmallcap400 Momentum-Quality — Concentrated Top-10 Rotation

**STATUS: COMPLETE — RS sweep + Phase-03 overlays + Phase-04 OOS/post-tax all done & passed. See results/MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md
(+ _DETAILED_REPORT.md).**
Verdict: RS-concentrated on PIT mid/small **robustly beats the ~20% hurdle**
(75/75 raw; 12 robust ex-top3 34–39%). RS-alone pick `mid_120d_N15` = 38.4%
CAGR / −29.8% DD / Calmar 1.29. **Phase-03 final pick `q0.5_dd__v__REG`**
(quality≥0.5 + SMA200 regime, volume OFF) = **35.3% CAGR / −24.6% DD
(index-level) / Sharpe 1.53 / Calmar 1.44** — drawdown objection solved,
edge kept. Run #1 VOID (NIFTY50 benchmark only 2023→2026 → 8/12 yrs in
cash); fixed via NIFTYBEES.
**Phase 06/07 — universe decision LOCKED = MID.** Same pipeline on small
(`q0.5_dd-0.4_REG`: 30.2% post-tax / −28-30% DD) and combo
(`q0.58_dd-0.4_REG`: 28.1% post-tax / −30.6% DD — strictly dominated by
mid). Mid wins risk-adjusted (Calmar 1.44 vs 1.27/1.13) + tradability
(22 F&O stocks vs small 1). YoY: beat Nifty-50 10/13 yrs, 35.3% gross
vs 13.6%. See `_DETAILED_REPORT.md` §9/§10, `SMALLCAP_*`, `COMBO_*`,
`MIDCAP_WINNER_YOY_VS_BENCHMARKS.md`. Nifty100/Mid/Small YoY pending
Kite index pull on VPS. **research/41 fully closed; nothing live.**

## 1. The Ask

**What the user asked (verbatim, condensed):**
> I like the Nifty MidSmallcap400 Momentum Quality 100 index (NSE-maintained,
> ~20% CAGR). Not interested in a fund/ETF. Want further stock selection out
> of it — pick the best/winning stocks, ride the trend, rotate (more or less
> frequently) into other winners when current ones lose momentum. Arrive at a
> top-10 portfolio. Assess and come up with stocks giving best possible
> returns, consistently beating the index's ~20% CAGR. Quality: source from
> the web. Chosen methodology: **Approach A (survivorship-free reconstruction)**.

**What we are actually testing/building:**
A concentrated (~10-stock) momentum-led rotation strategy whose eligible
universe is the *reconstructed* point-in-time Nifty MidSmallcap-400-like pool
at each rebalance — NOT today's 100 MQ constituents (that would be survivorship
bias). Goal: a walk-forward-validated rule that historically beat the index's
~20% CAGR, with an honest drawdown/underperformance profile, plus a live
top-10 today.

## 2. The honest scope (locked)

| Factor | Backtest (historical) | Live pick (today) |
|---|---|---|
| Universe | Reconstructed PIT mid/small pool from price+liquidity (no look-ahead) | User's supplied 100 MQ100 constituents (2026-05-15) |
| Momentum | Exact index method: 6-mo + 12-mo price return, volatility-adjusted | same |
| Quality | **Price-based proxy** (return-stability, low max-DD, trend-persistence) — exact ROE/DE/EPS-var historical PIT fundamentals are NOT freely available | **Current fundamentals** web-sourced overlay (ROE, D/E, profit growth) |
| Costs | STT + ~15-20% STCG + impact cost (material for small caps) modelled | — |

No performance guarantee. Concentration ≈ higher CAGR **and** deeper
drawdown than the 20% index; it will trail in some windows. Deliverable is a
measured, walk-forward edge — not "certain" outperformance.

## 3. Universe data coverage (market_data.db daily)

- Supplied list: 100 names (saved `universe_mq100_2026-05-15.csv`)
- 91/100 have daily history; 9 are ticker renames (360ONE←IIFLWAM,
  UNOMINDA←MINDAIND, NAVA←NAVABHARAT, GVT&D←GET&D, APARINDS, GPIL, JSWDULUX,
  KIRLOSBROS, USHAMART) — remap in Phase A.
- Full DB: 1,623 daily symbols, 2000→2026 — broad enough to reconstruct a
  PIT mid/small universe by liquidity/size proxy.

## 4. Plan (REVISED 2026-05-15 — pure RS study, MQ-engine & index-replication dropped)

User redirect: forget replicating the index / MQ-engine. Apply **Relative
Strength vs Nifty50** on point-in-time **mid** and **small** cap universes
(separately + combined), sweep RS lookbacks, hold concentrated top-10, rotate
monthly. All from price data we have — survivorship-free, no fundamentals, no
index membership.

- **Universe (PIT, monthly):** rank all NSE daily symbols by trailing-6mo
  median traded value (price×vol). **Mid = ranks 101–250**, **Small =
  251–500**, **Combo = 101–500**. Liquidity-band proxy, owned as such.
- **Signal:** RS_i = (P_i[t]/P_i[t-L]) / (NIFTY[t]/NIFTY[t-L]).
- **Lookback grid L:** 55, 120, 126 (6m), 252 (1y), blend(126+252).
- **Selection:** top-10 by RS within universe, equal-weight.
- **Rotation:** monthly; hold while in top-15 buffer, refill to top-10.
- **Costs:** ~0.4% round-trip; STCG reported separately.
- **Window:** 2014→2026, must include 2018-19 / Mar-2020 / 2022 bears.
- **Grid:** 3 universes × 5 lookbacks = 15 configs. Rank by CAGR, Sharpe,
  MaxDD, Calmar, and consistency of beating the ~20% hurdle; per-year table.
- **Bear-state cash:** idle cash earns 6.5% (debt) — explicit, not 0%.

Phase A (liquidity-proxy universe) already built/run: `pit_universe.csv`
exists (10,138 rows, 26 semi-annual dates). The RS sweep will rebuild the
universe monthly with the mid/small band split.

## 4b. Sharpened design (user refinements 2026-05-15)

- **Portfolio size is a swept axis**, not fixed: N ∈ {10,15,20,25,30}.
- **Super-winner false-indication guard (mandatory):** for the top configs,
  re-run the SAME strategy *forbidding it from ever holding its best-1 and
  best-3 lifetime contributors* (`exclude` set). If CAGR collapses <20%
  without those names → FRAGILE (apparent edge = 1-3 multibaggers, not
  breadth). Also report top-1 / top-3 share of total profit. The honest
  pick must beat ~20% **even after losing its best 3 names**.
- Grid now: 3 universes × 5 lookbacks × 5 sizes = **75 configs** +
  robustness re-runs on top-12 (×2 each).

## 5. Status (live log)

| Date/time IST | Event | Notes |
|---|---|---|
| 2026-05-15 ~20:45 | research/41 created, universe saved, coverage checked | 91/100 have data; 9 renames |
| 2026-05-15 ~21:10 | Plan redirected: pure RS study (MQ-engine/index-replication dropped) | user |
| 2026-05-15 ~21:25 | Design sharpened: N-sweep + leave-winners-out robustness | user |
| 2026-05-15 ~21:30 | RS sweep launched (background) | 75 configs + 24 robustness re-runs; window 2014→2026 |
| 2026-05-15 ~22:?? | **BENCHMARK BUG CAUGHT (run #1 invalid)** | Per-year table showed flat 6.5% every year 2015-2022 → strategy was 100% in cash. Root cause: `NIFTY50` daily exists only **2023-03→2026-03** (740 bars), so `rs_scores()` returned `None` for 8 of 12 years. The "RS fails / 0 configs beat 20%" output of run #1 is a **false negative** — RS was never tested 2014-2022. |
| 2026-05-15 ~22:?? | **FIXED**: benchmark → `NIFTYBEES` | Nifty-50 ETF, full daily history **2005-01→2026-02** (5,235 bars). RS is a price ratio so ETF scale cancels — faithful Nifty-50 proxy. Also excluded benchmark from investable PIT universe. |
| 2026-05-15 ~22:?? | Smoke test (1 config) | `combo_55d_N20` → CAGR **31.6%** (beats 20%), but MaxDD **−61.4%**, real bears 2018 −23% / 2019 −21%. Conclusion fully inverts vs run #1. |
| 2026-05-15 ~22:?? | **Corrected full sweep re-launched** (bg `bsk79glt8`) | 75 configs + 12-config super-winner robustness; window 2014→2026, real benchmark. RESULTS.md to be written from THIS run only. |

## 6. Crash Recovery

Resume via this file. Scripts under `scripts/`, outputs under `results/`.
Re-run order: `01_reconstruct_universe.py` → `02_factors.py` →
`03_rotation_backtest.py` → `04_walkforward.py` → `05_live_top10.py`.
All read `backtest_data/market_data.db` (canonical on VPS). Idempotent.

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `universe_mq100_2026-05-15.csv` | Supplied 100 (live-pick pool) | yes |
| `scripts/01_reconstruct_universe.py` | PIT universe per rebalance | yes |
| `scripts/02_factors.py` | momentum + quality-proxy | yes |
| `scripts/03_rotation_backtest.py` | concentrated top-10 rotation | yes |
| `scripts/04_walkforward.py` | OOS validation | yes |
| `scripts/05_live_top15.py` | today's portfolio + quality overlay | yes |
| `results/MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md` | findings + honest verdict | yes |
| `results/MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md` | full methodology + all comparisons | yes |

## 8. Findings

**Run #1 (INVALID — discard):** reported 0/75 configs >20%, all ~6-11% CAGR.
This was a benchmark-data artifact: `NIFTY50` daily only spans 2023-03→2026-03,
so 2014-2022 had no RS signal and the book sat in cash at 6.5% for 8/12 years.
Do **not** cite run #1 numbers anywhere.

**Run #2 (corrected, COMPLETE):** benchmark = `NIFTYBEES` (2005→2026).
- 75/75 configs beat ~20% (raw CAGR 25–41%).
- **12 configs robust** — still 34–39% CAGR after forbidding their 3 best
  lifetime names. Edge is breadth, not 1–3 multibaggers (top-3 share ~8–15%).
- Best risk-adjusted robust pick: **`mid_120d_N15`** — 38.3% CAGR, Sharpe
  1.39, MaxDD −29.8%, Calmar 1.29, ex-top3 33.9%, top-3 share 11.9%.
- 55d lookback = worst DD bucket (−54 to −66%); 6m/blend dominate; mid
  universe best risk-adjusted.
- Honest caveats locked in the RESULTS doc: quality not a fundamental
  screen (price-path proxy); PIT universe is liquidity-proxy; ~−25% DD
  real (post-regime); STCG netted in Phase 04; OOS passed.

Full findings + recommended config + caveats →
**results/MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md**; full methodology +
every comparison → **results/MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md**.

**Phase 05 (live deliverable) — DONE.** `scripts/05_live_top15.py` →
today's top-15 from `q0.5_dd__v__REG` on the supplied MQ100 (as-of
2026-02-16 snapshot; regime RISK-ON). Web-sourced current fundamentals
(ROE/D-E/PAT growth/ROCE, screener.in) overlaid →
`results/LIVE_TOP15_WITH_FUNDAMENTALS.md`. Cross-check: 11/15 fundamentally
Strong, 2 Solid banks, 2 Mixed, 1 Weak (GMDCLTD flagged). Re-run on VPS for
current-dated list.

**Phase 04 (OOS + post-tax) — DONE, PASSED.** `scripts/04_walkforward.py`.
(A) Sub-period: H1 2014-19 32.2% CAGR/-24.6% DD, H2 2020-26 37.3%/-14.7%
— edge in both halves. (B) Walk-forward lookback selection chained 33.1%
CAGR vs static 35.0% (only ever picks 120d/126d) — selection robust, not
overfit. (C) Post-tax: net **28.9% CAGR @20% STCG** / 30.4% @15% — still
~9pp above the 20% hurdle. Verdict: robust, OOS-stable, ~29% post-tax at
index-level drawdown. LTCG not modelled (small, stated). **research/41
COMPLETE** — no open items; real-capital use still a user decision.
