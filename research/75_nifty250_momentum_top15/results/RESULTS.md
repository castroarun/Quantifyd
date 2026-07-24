# Research 75 — Nifty-250 Momentum Top-15: Faithful Replication of the "Only Momentum Strategy" Video

**VERDICT: STRATEGY (candidate) — the video REPLICATES and on return EXCEEDS its own claim,
but it is essentially our existing momentum book re-derived, and two of its headline details
are wrong.** On survivorship-free data, 2006–2026, net of 0.3% cost: **31.9% CAGR, −31.6%
daily-marked MaxDD, Calmar 1.01, Sharpe 1.45, 292× (₹5L → ₹14.6cr).** That *beats* the
video's advertised 27% CAGR / ₹5cr (100×) on return — but the **drawdown is deeper than the
advertised −23%** (that −23% only appears on the modern 2014+ sub-period with a risk-adjusted
momentum score). The two things the video sells as the system's edge — the per-stock
**50>100>200 EMA filter** and holding exactly **15** names — are **not** where the edge is:
the EMA filter is mildly *harmful*, and the real engine is plain momentum + the index cash gate.

## Claim vs replication

| Metric | Video claim | This backtest (faithful base, net) | |
|---|---|---|---|
| Period | ~20 yr | 2006–2026 (20.5 yr) | ✓ |
| CAGR | 27% | **31.9%** | ✓ exceeds |
| Growth (₹5L →) | ₹5 cr (100×) | **₹14.6 cr (292×)** | ✓ exceeds |
| Max drawdown | −23% | **−31.6%** (daily-marked, full period) | ✗ deeper |
| Trades/yr | ~12 | 12 rebalances/yr; turnover ~0.38 | ✓ |
| 2008 protection | "moved to cash" | **−20.4% vs Nifty −52.1%** | ✓ (gate lagged, not fully cash) |

The −23% DD reconciles only two ways: **(a) modern sub-period** — 2014→2026 DD is −30.7%
(plain) / **−22.0% (risk-adjusted momentum, B2)**; and/or **(b)** the video used cleaner
index-provided (survivorship-biased) constituents and likely monthly-marked DD. Our −31.6%
is the honest, daily-marked, survivorship-free number and is the one to trust.

## Full grid (2006–2026, daily-marked, net of 0.3% unless noted)

| Config | Gross CAGR | Net CAGR | Net@STCG20 | MaxDD | Sharpe | Calmar | Mult | Turn |
|---|---|---|---|---|---|---|---|---|
| **A1 BASE (faithful)** | 34.4 | **31.9** | 29.2 | **−31.6** | 1.45 | 1.01 | 292× | 0.38 |
| A2 no index gate | 29.7 | 26.5 | 26.5 | **−65.9** | 1.03 | 0.40 | 124× | 0.35 |
| A3 no EMA-stack | 37.0 | **34.7** | 31.5 | −32.2 | 1.55 | **1.08** | 449× | 0.36 |
| A4 pure momentum (both off) | 31.4 | 28.5 | 28.5 | −72.5 | 1.09 | 0.39 | 171× | 0.32 |
| B1 momentum 12−1 | 31.8 | 29.2 | 26.9 | −32.3 | 1.39 | 0.90 | 190× | 0.42 |
| B2 momentum risk-adj z-blend | 33.8 | 31.0 | 29.0 | −30.4 | 1.53 | 1.02 | 256× | 0.42 |
| C1 N=10 | 35.1 | 32.6 | 29.9 | −34.7 | 1.39 | 0.94 | 326× | 0.38 |
| C2 N=20 | 32.6 | 30.2 | 27.8 | −31.0 | 1.44 | 0.97 | 223× | 0.37 |
| D1 cost 0.1% | 34.4 | 33.5 | 30.8 | −31.6 | 1.51 | 1.06 | 376× | 0.38 |
| D2 cost 0.5% | 34.4 | 30.2 | 27.6 | −32.5 | 1.38 | 0.93 | 226× | 0.38 |
| E1 gate weekly | 34.2 | 31.6 | 28.9 | −31.6 | 1.50 | 1.00 | 281× | 0.39 |

Benchmark NIFTYBEES buy-hold: **11.6% CAGR, −59.7% DD, 9.4×.** Strategy beats the index in
**71% of years**; excess CAGR **+20.3%/yr**; beta 0.47.

## Rule attribution — what actually drives it

1. **The index-EMA cash gate is the ENTIRE risk story.** Remove it (A2) and MaxDD explodes
   −31.6% → **−65.9%**, Calmar 1.01 → 0.40. In 2008 the gated book lost −20% while the
   ungated book lost −59%. This matches every prior study (research/41 "gate irreplaceable",
   /62, /71). It is *the* rule that turns raw momentum into an investable book.
2. **The per-stock 50>100>200 EMA filter — the video's flagship "special sauce" — is
   inert-to-harmful.** Removing it (A3) *raises* CAGR 31.9 → 34.7% and Calmar 1.01 → 1.08,
   with essentially the same DD. Cross-sectional momentum already prefers uptrending names;
   the EMA stack just occasionally benches a winner. (Same lesson as research/41 ph28:
   per-stock trailing-MA filters add no net edge on top of the index gate.)
3. **Momentum definition barely matters:** plain 12m (31.9%), risk-adj z-blend (31.0%,
   *best DD −30.4% / modern −22.0%*), 12−1 (29.2%). The video's unspecified momentum is not
   a fragile choice — the conclusion is robust to it. Risk-adjusted is the best risk-adjusted.
4. **Concentration:** N10 32.6%/−34.7%, N15 31.9%/−31.6%, N20 30.2%/−31.0% — monotonic
   trade of return for drawdown, no free lunch; N15 is a sensible middle (as the video picks).
5. **Cost-robust, low-turnover:** net CAGR 33.5% → 31.9% → 30.2% across 0.1/0.3/0.5% cost.
   Break-even cost is far above realistic large-mid-cap cost. Not a turnover-fragile edge.
6. **Gate frequency irrelevant:** weekly gate (E1) ≈ monthly. Monthly (faithful) is fine.

## Honest caveats

- **Universe proxy, not the real index.** We use a survivorship-**free** point-in-time
  top-250-by-traded-value basket as a stand-in for Nifty LargeMidcap 250 (the actual index
  membership isn't reconstructable to 2006). This is *more* honest than the video (no
  survivorship bias) but is not the exact index. Early years are thinner — **~218 names in
  2006**, reaching 250 by ~2010.
- **DD is daily-marked (−31.6%)**; the video's −23% is likely monthly-marked and/or modern
  and/or survivorship-cleaned. Do not quote −23% as ours.
- **Two monster years** (2014 +108%, 2021 +101%) contribute heavily, but participation is
  broad (2006/2012/2017/2020/2023/2024 all >55%), so it is not a 1–2-year mirage.
- **STCG drag is real:** monthly rotation ⇒ mostly short-term gains; net-of-20%-STCG CAGR is
  **~29%**, still excellent, but that is the number an Indian taxable investor actually keeps.
- **Multiple testing:** 11 configs; the faithful base (A1) was pre-specified from the video,
  not selected as the winner, so snooping risk is low. A3-beats-A1 is a robustness finding,
  not an optimized peak.
- **Nothing structurally new.** This is the **same family** as our live **momentum-paper**
  ₹20L book (research/62) and research/41 midcap RS: momentum rank + NIFTY-trend cash gate +
  monthly rebalance. It validates that book from a second angle; it does **not** add a new
  source of alpha.

## Next levers

- If we ever want a "large-mid 250" flavour alongside the running Nifty-200 momentum-paper
  book: **drop the EMA-stack filter, use risk-adjusted momentum, keep the index gate, N≈15**
  (that's A3/B2 territory: ~31–35% CAGR, −30% DD, Calmar ~1.05). But first ask whether a
  second, ~0.8-correlated momentum sleeve earns its slot — likely redundant with momentum-paper.
- The honest client-facing takeaway for the video: **the strategy is real and strong, but its
  advertised −23% DD is optimistic and its trend-filter "edge" is cosmetic — the cash gate is
  the whole risk story.**

## Reproducibility stamp

- Data snapshot: `market_data.db` on VPS `94.136.185.54`, as of 2026-07-08 15:30 (4.9 GB).
- Runner: `research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py` (venv python).
- Cost 0.3% round-trip; cash 6.5% p.a.; benchmark NIFTYBEES. 11 configs, 504s total.
- Bug fixed mid-study: v1 reused `rs2.month_ends` (hard-coded START=2014) → parked 2006–2013 in
  cash and never tested 2008; corrected to compute month-ends from the study start.
