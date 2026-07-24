# Nifty-250 Momentum Top-15 — Faithful Replication of the "Only Momentum Strategy" Video

STATUS: **DONE** (3 phases; published to app + Artifact) — see `results/RESULTS.md`, `RESULTS_P2.md`.

**VERDICT (final):** Faithful base **31.8% net / 29.1% post-tax / −31.6% DD / 292×** (2006–2026).
Video's **27% CAGR replicates & is beaten; the −23% MaxDD does NOT** (−31.6%, honest window incl. 2008).
**NIFTYBEES-100EMA gate = the whole risk story** (off → −51%) and **irreplaceable** (no per-stock
quality/ATH/exit substitutes; best gate-less DD −46%). **EMA-stack inert-to-harmful.** Best
risk-adjusted = **midcap + 6-month RS (Calmar 1.26, −29% DD)**; highest CAGR = mid+small combo
(43.5%, −42% DD; cost-robust but DD-uninvestable). **Same family as Aurum `midcap_smoothest` — corroborates,
not new alpha.** App: `/app/backtest/nifty250-momentum-video-research75`.

Study: `research/75_nifty250_momentum_top15/` · Host: VPS `94.136.185.54` (canonical) ·
Snapshot: `market_data.db` (4.9 GB, VPS, as of 2026-07-08 15:30)

---

## 1. The Ask

**What you asked:** Shared the Quantinuous YouTube video *"The Only Momentum Strategy
You Need for Nifty 250 Stocks (20 Year Proof)"* and chose **"Backtest it faithfully."**

**What we're actually testing:** Reproduce, on our own survivorship-free NSE data,
the exact monthly momentum system the video describes and check whether its headline
claim (**~27% CAGR, −23% MaxDD, ₹5L→₹5cr over ~20 yrs, ~12 trades/yr**) holds up
**net of realistic cost**, and attribute *which* rule (the momentum rank, the per-stock
EMA-stack trend filter, or the index-EMA cash gate) actually drives the result.

**Success metric:** CAGR and Calmar (CAGR/|MaxDD|), reported **gross AND net**, with a
per-year table and cost-sensitivity. Faithful base must be within a believable tolerance
of the video's 27%/−23% before we credit the claim.

## 2. Economic hypothesis (G0)

- **Momentum / under-reaction.** Cross-sectional momentum (buy recent winners) is one of
  the most-documented anomalies; the counterparty is under-reacting/disposition-biased
  holders who sell winners too early. Well-harvested → expect it to be *real but partly
  arbitraged*, and turnover/cost-sensitive.
- **Trend filter (50>100>200 EMA).** Removes names in downtrends → avoids momentum's
  known left tail (buying a falling knife that "still ranks high").
- **Index-EMA cash gate.** A regime overlay: be flat when the market is below its 100-EMA.
  Its value is *being out*, not return — consistent with research/41/62/71 where the
  NIFTY-trend gate was the single most decisive risk control.
- **Decay risk:** momentum crowds; the gate can whipsaw in choppy sideways markets.

## 3. The Base — exact mechanics being tested

| Element | Faithful setting | Notes |
|---|---|---|
| **Universe** | Top-250 NSE names by trailing-6mo median traded value (close×vol), rebuilt monthly, ETFs/index proxies excluded | Survivorship-free PIT proxy for **Nifty LargeMidcap 250** (the real index list isn't reconstructable to 2006). Reuses `rs2.pit_universe` with a new band `n250=(0,250)`. |
| **Selection** | Rank universe by **momentum**, take **top 15**, **equal weight**, 100% invested when risk-on | Video: "ranks the universe by momentum, selects top 15" |
| **Per-stock trend filter** | Eligible only if **EMA50 > EMA100 AND EMA100 > EMA200** on the stock's own daily close (causal) | Video's stacked-uptrend gate |
| **Market regime filter** | If **NIFTYBEES close ≤ its own EMA100** → hold **cash** (liquidate), else invest | Video: "Nifty50 must be above its 100 EMA, else move to cash." NIFTYBEES = full-history Nifty-50 proxy (raw NIFTY50 daily only starts 2023 — playbook scar). |
| **Rebalance** | **Monthly** (month-end), full re-selection; gate checked at rebalance | Video: "start of each month" |
| **Marking** | **Daily** mark-to-market → honest MaxDD (video quotes −23%) | research/62 `run()` style, not monthly-only marking |
| **Costs** | **0.3% round-trip** on turnover (net); also gross; cost-sensitivity 0.1/0.3/0.5% | Large-mid caps liquid; momentum_paper uses ~0.3% |
| **Cash yield** | 6.5% p.a. on idle cash | `rs2.CASH_ANNUAL` |
| **Tax** | Report gross/net-of-cost NAV; 20% STCG shown separately (not baked in) | monthly rotation ⇒ mostly STCG |
| **Period** | Attempt **2006→2026** (≈20 yr) + always show **2014→2026** modern sub-period | Coverage in early years verified in smoke test; caveat if thin |

**Momentum definition is the one genuine ambiguity** (the video doesn't specify it).
Primary = **plain 12-month return** (P_t / P_{t−252} − 1) — the most common retail
"momentum". Robustness variants: **12−1** (skip last 21d) and **risk-adjusted z-blend**
(6m & 12m return ÷ vol, z-scored — the NSE Momentum-30 style used in research/62).

## 4. Plan — variant grid

Daily-marked, net@0.3% unless noted. All share the faithful base except the varied axis.

**A. Faithful base + rule attribution (ablation):**
| # | Config | Momentum | EMA-stack | Index gate |
|---|---|---|---|---|
| A1 | **BASE (faithful)** | ret252 | ON | ON |
| A2 | no index gate | ret252 | ON | OFF |
| A3 | no EMA-stack | ret252 | OFF | ON |
| A4 | pure momentum | ret252 | OFF | OFF |

**B. Momentum-definition sensitivity (full filters):** A1 with `ret252_21`, `radj_z`.

**C. Portfolio-size sensitivity (full filters, ret252):** N = 10, 15, 20.

**D. Cost sensitivity (A1):** rt = 0.1% / 0.3% / 0.5% + break-even note.

**E. Gate-frequency sensitivity:** A1 monthly (faithful) vs weekly gate.

≈ 14 cells. Cheap (each daily-marked run is seconds; indicators precomputed once).

**Falsification:** if the faithful base (A1) net CAGR is materially below ~22% or MaxDD
materially worse than ~−30% on the full period, the video's 27%/−23% does **not**
replicate on survivorship-free data and we say so plainly.

## 5. Status (live log)

**State:** RERUN IN FLIGHT (2006–2026 corrected calendar) after a coverage bug was caught & fixed.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-08 ~15:40 | Study framed; STATUS-MD sections 1–4 written | VPS reachable, DB 4.9 GB, rs2 engine confirmed |
| 2026-07-08 ~15:55 | Runner v1 launched on VPS (venv python) | 1642 syms 2004–2026; NIFTYBEES B&H 11.6% CAGR/−59.7% DD (sane) |
| 2026-07-08 ~16:05 | **BUG CAUGHT** in v1 per-year table | 2006–2013 all = 6.4%/yr = cash; +111% jump 2014. Cause: reused `rs2.month_ends` which hard-codes `rs2.START=2014` → no rebalance before 2014. "20yr" was really 2014–2026 + 8yr cash drag, and **2008 never tested**. |
| 2026-07-08 ~16:08 | Fixed: month-ends computed from study START; rerun launched | Now trades the full 2006–2026 incl. the 2008 crash (the real −23% DD test) |

**v1 (buggy, 2014–2026-effective) headline — SUPERSEDED, kept for the record:**
A1 base 21.5% net / −30.7% DD; B2 risk-adj-mom 21.2% / −22.0% DD (Cal 0.96);
A2 no-gate −51% DD (gate = the DD control); A3 no-EMA-stack ≈ identical to base
(stack adds nothing). Cost-robust (turnover ~0.37). The corrected full-period run
replaces these CAGR/DD numbers.

## 6. Crash Recovery — resume without Claude

1. **Where it runs:** VPS `94.136.185.54`, `/home/arun/quantifyd/research/75_nifty250_momentum_top15/`.
2. **Runner:** `scripts/run_nifty250_momentum.py` (self-contained; imports `rs2` =
   `research/41_midsmall400_mq_concentrated/scripts/02_rs_sweep.py` for data/universe).
3. **Check progress:** `tail -f results/run.log`; ranking rows append to
   `results/ranking.csv` (resumable — already-done configs are skipped on re-run).
4. **Re-run full:** `cd /home/arun/quantifyd && nohup python3 research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py > research/75_nifty250_momentum_top15/results/run.log 2>&1 &`
5. **Safe to inspect:** everything in `results/`. **Do NOT** touch the market DB or any
   `services/*` — this study is read-only w.r.t. live trading.

## 7. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `scripts/run_nifty250_momentum.py` | Faithful runner + ablation grid | yes |
| `NIFTY250_..._STATUS.md` | This file | yes |
| `results/ranking.csv` | Per-config metrics (gross+net, CAGR/DD/Sharpe/Calmar/turnover) | yes |
| `results/peryear.csv` | Per-year returns, key configs | yes |
| `results/nav_base.csv` | Daily NAV of the faithful base (for tearsheet) | yes (small) |
| `results/run.log` | Progress log | no (gitignored) |
| `results/RESULTS.md` | Final honest verdict | yes |

## 8. Findings — FINAL (see `results/RESULTS.md` for the full writeup)

**Faithful base, 2006–2026, net 0.3%: 31.9% CAGR · −31.6% DD (daily) · Calmar 1.01 ·
Sharpe 1.45 · 292× (₹5L→₹14.6cr).** vs video's 27%/−23%/100×: **beats return, misses DD**.

- **Index-EMA cash gate = the whole risk story** — remove it → DD −31.6%→−65.9%
  (2008: −20% gated vs −59% ungated vs Nifty −52%).
- **Video's per-stock 50>100>200 EMA filter is inert-to-harmful** — remove it (A3) → 34.7%
  CAGR / Calmar 1.08, *better* than base. The trend-filter "edge" is cosmetic.
- **−23% DD only on modern 2014+ with risk-adjusted momentum** (B2 modern DD −22.0%). Our
  honest full-period daily DD is −31.6%.
- Cost-robust (turnover 0.38; 30–33.5% net across 0.1–0.5% cost); momentum-def & gate-freq
  don't matter; concentration trades return for DD monotonically.
- **Not new alpha** — same family as the live momentum-paper ₹20L book (research/62) and
  research/41. Validates that book from a 2nd angle; adds nothing structural.

Client tearsheet: `results/tearsheet.png` (+ `.html`). Verdict: **STRATEGY (candidate)**.

**Event-log close-out:**

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-08 ~16:15 | Corrected rerun DONE (11 cells, 504s) | Full 2006–2026 incl. 2008 |
| 2026-07-08 ~16:25 | Tearsheet generated; RESULTS.md written; verdict STRATEGY-candidate | Closed loop |
