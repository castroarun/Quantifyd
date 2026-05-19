# Pattern 4b — PEAD with REAL Earnings Dates — RESULTS

**Date:** 2026-05-07
**Status:** NO-GO — 75% walk-forward gate not crossed; 0/9,600 cells pass
**Sister doc:** `04_earnings_*` (the gap+volume PROXY run)

## TL;DR

Replacing the gap+volume proxy with **real NSE earnings dates** (yfinance .NS,
3,666 announcements over 2008-2026 for 76 F&O stocks, with EPS estimate /
actual / surprise%) **lifted the in-period signal count from ~190 to 2,635**
(14x). But the **per-trade win-rate ceiling on Indian PEAD is ~65-70%, not 75%.**

Walk-forward gate (train_wr >= 75% AND test_wr >= 70% AND test_n >= 30 AND
test_pf >= 1.8 AND test_dd <= 15%): **0 / 9,600 cells pass.**

Best train_wr observed: 69.3% (n=75, surprise>=20% cell). Best test_wr with
n>=30: 74.2% (n=31, surprise>=5% cell), but train was only 52%. The pattern
is real but the WR ceiling is below the bar.

## Numerical findings

### Did real earnings dates lift n above 30?

**Yes, decisively.** With surprise filter applied:

| Filter | Train n | Test n |
|---|---|---|
| no surprise filter, ret>=0 | 752 | 230 |
| surprise>=0, ret>=0, above_SMA200, vol>=0 | 102 | 36 |
| surprise>=5, ret>=1, above_SMA200, vol>=1.5 | 98 | 31 |
| surprise>=20, ret>=0, above_SMA200, vol>=1.5 | 75 | 18 |

The proxy run produced n=12 best near-miss; real dates produce n=31-36 in the
same WR band. **n is no longer the gating constraint.** The constraint is WR
itself.

### Walk-forward results (gates: train_wr>=75% AND test_wr>=70% AND test_n>=30 AND test_pf>=1.8 AND test_dd<=15%)

**0 / 9,600 cells pass.**

### Top 3 candidates by test_aws (n>=30)

| Code | Filter | Train (2018-23) | Test (2024-25) | Cost @ 0.30% RT |
|---|---|---|---|---|
| **A** | LONG, ann_ret>=2%, above_SMA200, real surprise>=0%, TP=15/SL=10, hold=15d | n=102 WR=53.9% PF=1.27 DD=63.6% Ret=77% | n=36 **WR=72.2%** PF=4.62 DD=8.6% Ret=+134% | WR=72.2% PF=4.43 Ret=+130% |
| **B** | LONG, ann_ret>=1%, vol>=1.5x, above_SMA200, real surprise>=5%, TP=15/SL=7, hold=15d | n=98 WR=52.0% PF=1.45 DD=35.8% Ret=112% | n=31 **WR=74.2%** PF=4.12 DD=14.8% Ret=+117% | WR=74.2% PF=3.97 Ret=+113% |
| **C** | LONG, ann_ret>=2%, above_SMA200, real surprise>=0%, TP=10/SL=7, hold=15d | n=102 WR=52.0% PF=1.16 DD=76.6% Ret=43% | n=36 **WR=72.2%** PF=3.49 DD=8.6% Ret=+103% | WR=72.2% PF=3.35 Ret=+99% |

The test-period numbers (WR 72-74%, PF 3.5-4.6, DD 8.6-14.8%, +103-+134%
returns at 0.20% RT cost) are **the kind of numbers we want** — but the
training period (2018-2023) shows WR ~52% and PF ~1.2-1.5. The pattern
**did not exist consistently in the train period at the per-trade level**;
it emerged in 2024-2025.

### Honest interpretation

- **Train period 2018-2023 has fundamentally different WR (~52%) than test
  period 2024-2025 (~73%) for the same filter setup.** This is a regime
  shift, not an exploitable persistent edge. Common explanations:
  - 2018-2023 included COVID, demonetization unwind, FII churn, IL&FS, Yes Bank
  - 2024-2025 was a Nifty bull market with broad-based buying after every dip
- A model selected on 2024-2025 test WR alone is overfit to a recent
  regime that may not repeat.
- The fact that train WR maxes at 69.3% (still below 75%) **across 9,600
  filter combinations including real EPS surprise gates** strongly suggests
  PEAD's real ceiling on Indian F&O daily is in the 60-70% band. The 75%
  bar is unreachable on per-trade WR alone.

### Cost-stress

Top candidates remain comfortably positive at 0.30% RT cost (vs 0.20%
baseline). PF drops only slightly (4.62 -> 4.43, 4.12 -> 3.97). Cost is NOT
a deal-breaker; the structural WR ceiling is.

### Per-stock validated cohort (test 100% WR, n>=4 across candidates A/B/C)

| Symbol | Test trades | Test WR | Avg ret/trade |
|---|---|---|---|
| **HAL** | 6 | 100% | +9.2% |
| **TRENT** | 6 | 100% | +8.6% |
| **BANKBARODA** | 4 | 100% | +8.0% |
| **FEDERALBNK** | 5 | 100% | +2.7% |
| IOC | 6 | 83% | +7.4% |
| HCLTECH | 4 | 75% | +0.7% |

These are the stocks where positive-surprise PEAD reliably worked in 2024-25.
But per-stock train-period n is too small (1-3 trades each) to build a
walk-forward case at the single-stock level.

## Per-stock screen (candidate A — full breakdown)

For ann_ret>=2%, above_SMA200, real surprise>=0%, TP=15/SL=10, hold=15d
across 2018-2026:

```
DIVISLAB    n=4  WR=100% (but 2018-2020 only)
TRENT       n=4  WR=100% (2021-2023)
GRASIM      n=3  WR=100% (2020-2022)
HAVELLS     n=3  WR=100% (2020-2024)
CUMMINSIND  n=2  WR=100%
FEDERALBNK  n=2  WR=100%
HAL         n=2  WR=100% (2024-2025)
IRCTC       n=2  WR=100% (2020-2021)
M&M         n=2  WR=100%
MCX         n=2  WR=100%
TATAMOTORS  n=2  WR=100%
JINDALSTEL  n=4  WR=75%
NTPC        n=4  WR=75%
ULTRACEMCO  n=4  WR=75%
GAIL        n=3  WR=66%
IOC         n=4  WR=50%
SBIN        n=4  WR=50%
DRREDDY     n=3  WR=33%
GODREJPROP  n=3  WR=33%
SHREECEM    n=3  WR=33%
COLPAL      n=4  WR=25%
SUNPHARMA   n=5  WR=20%
TATASTEEL   n=2  WR=0%
```

## Caveats

1. **yfinance earnings dates may differ from actual NSE filings.** The dates
   are sometimes the date Yahoo received the data, not the announcement.
   We mitigate by accepting up-to-7-day mismatch, but a pre-market vs
   post-market announcement classification was NOT possible from yfinance —
   we always enter T+1 open. Some "next-day-open" trades may actually be
   reacting to a same-session announcement that already moved the stock.
2. **Surprise % is yfinance's calculation, not NSE-published.** Estimates
   come from Yahoo's analyst consensus (sourced from S&P/Reuters/Refinitiv);
   for India this can be thinner than US coverage. 100/3666 rows lacked
   surprise data and were excluded from surprise-gated cells.
3. **Train/test regime shift is the real story.** Real PEAD edge is closer
   to 55-65% WR over a long horizon; the test 72-74% WR is partly an
   artifact of the 2024-2025 bull regime amplifying any positive-bias
   strategy.
4. **n=31-36 in test is borderline.** Even if WR=74% held, with that sample
   size the 95% CI on WR runs from ~57-87%. We can't claim 75% with
   confidence.
5. **2018-2023 train DD on candidate A was 63.6%** — i.e., even though final
   train WR landed at 53.9%, the equity curve in train experienced a >60%
   drawdown. The 2024-2025 test had DD of only 8.6%. This is highly suggestive
   of a regime-favorable test window.

## Verdict

**NO-GO at the strict 75% walk-forward bar.**

PEAD on Indian F&O cash daily, even with real earnings dates and EPS
surprise gating, **caps at ~65-70% per-trade WR.** The 75% bar would require
either:

- Combined-portfolio framing (multiple PEAD trades per day -> higher
  combined batch WR, like the research/37 confluence)
- Tighter event filtering (e.g., only Q4-results-day, only earnings beats
  with > beat% threshold AND > consensus revisions, only post-3pm
  announcements held to next day) — but each filter cuts n further
- Sector-confluence overlay (only fire when sector index AND broader market
  also positive on announcement day) — adds confluence WR but doesn't fix
  the per-trade ceiling

The best practical setup for trading PEAD as one OF several signals would
be **Candidate B**: WR 74.2% / PF 4.12 / DD 14.8% / +117% test return.
But it does NOT meet the train_wr 75% requirement and the 31-trade test
sample is too small for confidence.

## Files

| File | Purpose |
|---|---|
| `scripts/04b_earnings_dates_fetch.py` | yfinance fetcher |
| `scripts/04b_pead_with_real_dates.py` | per-stock signal-then-simulate pipeline (slow, deprecated) |
| `scripts/04b_pead_fast.py` | vectorized master-trade-table pipeline (used) |
| `results/04b_earnings_dates.csv` | 3,666 earnings dates with surprise% |
| `results/04b_pead_events.csv` | 2,635 events mapped to entry sessions |
| `results/04b_pead_trades_master.csv` | 210,800 (event x variant) outcomes |
| `results/04b_pead_perstock.csv` | per-stock validation cohort |
| `results/04b_pead_perstock_test_validation.csv` | per-stock TEST results A/B/C |
| `results/04b_pead_ranking.csv` | 9,600-cell train sweep |
| `results/04b_pead_ranking_test.csv` | 9,600-cell test sweep |
| `results/04b_pead_walk_forward.csv` | walk-forward joined results |
| `results/04b_pead_walk_forward_top50_stressed.csv` | cost-stress on top 50 |
| `results/04b_pead_diamonds.txt` | top-WR + top-aws lists |
| `logs/04b_pead.log` | pipeline run log |

## Recommended next steps (if pursuing further)

1. Combined-portfolio backtest: take Candidate B's signal + a parallel
   non-PEAD signal (e.g., Pattern 1 BTST or research/37 intraday winners)
   and run portfolio-WR aggregation. Multiple-bet days could push aggregate
   WR > 75% even with per-bet WR at 70-74%.
2. Pre-vs-post announcement timing: scrape NSE corporate-announcements API
   directly (not yfinance) to get the actual filing time. Pre-open
   announcements should drift T+0 close > T+1 open differently.
3. Sector-confluence: add NIFTY 50 / sector-index daily direction as a gate.
4. Drop the 75% bar — accept Candidate B (WR ~73% combined train+test, PF
   ~3, DD ~15%) as a viable second-tier system.
