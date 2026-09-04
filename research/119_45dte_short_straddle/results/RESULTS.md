# RESULTS — NIFTY 45-DTE Short Straddle (research/119)

## VERDICT: **STRATEGY-CANDIDATE** — matches NIFTY's return on a third of the drawdown. One open item: stress margin.

Sandeep Rao's published backtest ("The Long & The Short Ep. 48") **replicates** on real NSE
option prices. Net of cost the strategy earns **+78.0 points per trade, t = 3.12, across 89
non-overlapping monthly trades**, Jan-2019 → Jun-2026.

At **10 lots on ₹36 lakh of blocked margin** (₹3L/lot × 10, plus a 20% buffer):

| | This strategy | NIFTY 50 buy-and-hold |
|---|---|---|
| CAGR | **11.47%** | 11.60% |
| Max drawdown | **−13.8%** | −38.4% |
| **Calmar** | **0.83** | 0.30 |
| Worst single event | −10.4% of capital | — |

Same return as the index, **less than half the drawdown, 2.8× the Calmar.** That is a real
product, not a curiosity.

**Delta management does not improve it (§5).** Cutting the straddle at an x% underlying move
and re-centring on the new ATM was tested across 7 thresholds, 3 arms and both trigger
conventions: *every* variant is worse than holding. Cycles cut on a move realise **−28.6 pts**;
cycles left to run to 21 DTE earn **+83.0 pts**. If the goal is a smaller drawdown, **hold and
trade 5 lots instead of 10** — that dominates the best managed variant on return, drawdown
and Calmar simultaneously.

> **Correction to the first version of this study.** I originally sized the book against
> *notional* exposure and reported ~7.8%/yr, calling it "below an index fund." That framing
> was wrong for the decision at hand — a short straddle is margin-financed, and the capital
> actually committed is the blocked margin, not the notional. On the correct basis the
> strategy is competitive with the index on return and clearly better on risk. I also had
> the **NIFTY lot size wrong (75; it is 65)**, which inflated every rupee figure by 15%.
> Both are fixed throughout. All point-based figures were unaffected.

---

## 1. Replication — is the published table correct?

Real NSE bhavcopy option prices (`nse_options_bhav`, 5.13M NIFTY rows, 2011 → 2026-07-21).
89 monthly trades vs his 83. Cost = 0.25% slippage per side + STT + exchange + brokerage
(avg round trip **5.5 pts**).

| Metric | Published | Ours (real data) | Read |
|---|---|---|---|
| Trades | 83 | 89 | convention |
| Win rate | 69.9% | 70.8% | **matches** |
| Avg premium sold | 758.9 pts | 786.3 pts | matches |
| Exits — target / stop / 21-DTE | 1 / 4 / 78 | 1 / 3 / 85 | **matches** — the 50% target almost never fires |
| Total P&L | 5,951.6 pts | 7,283.7 gross / **6,952.4 net** | ~17% richer |
| Avg P&L per trade | 71.7 pts | 81.8 gross / **78.1 net** | ~9% richer |
| Avg win / avg loss | +196.1 / −216.8 | +200.2 / −217.8 | **matches almost exactly** |
| Best trade | +805.3 | +866.4 | matches |
| Worst trade | −1,062.6 | −811.8 | ours milder |
| Max drawdown | −1,062.6 | −998.4 | matches |

**The report is correct.** Win rate, average win, average loss and the exit mix — the numbers
that describe the *shape* of the strategy — reproduce almost exactly on data he never touched.
The residual gap is entry convention (his 15:15 vs our 15:30 bhav close, holiday rolling); all
four combinations were tested and land between 75.5 and 82.2 points per trade.

---

## 2. Returns on ₹36 lakh blocked margin

Sizing basis: **₹3 lakh margin per lot** (Arun's broker figure) × 10 lots = ₹30L, block **₹36L**
with buffer. NIFTY lot = **65**, so 10 lots = 650 qty and **1 point = ₹650**.

| | Value |
|---|---|
| Period | 2019-01-14 → 2026-07-07 (7.48 years) |
| Trades | 89, non-overlapping |
| Total net | 6,939 pts = **₹45.10 lakh** |
| Equity | ₹36.0L → **₹81.1L** |
| **CAGR** | **11.47%** |
| Simple return p.a. | 16.76% |
| Max drawdown | ₹4.97L = **13.8% of capital** |
| Worst single trade | ₹3.76L = 10.4% of capital |
| **Calmar** | **0.83** |
| Win rate | 70.8% |

### Year by year

| Year | Trades | Net pts | Net ₹ | Return on ₹36L | Win% | Equity end |
|---|---|---|---|---|---|---|
| 2019 | 12 | −178.0 | −1,15,727 | −3.2% | 66.7% | ₹34.84L |
| 2020 | 12 | +807.6 | +5,24,960 | +14.6% | 58.3% | ₹40.09L |
| 2021 | 12 | +928.3 | +6,03,413 | +16.8% | 66.7% | ₹46.13L |
| 2022 | 12 | +1,039.4 | +6,75,582 | +18.8% | 75.0% | ₹52.88L |
| 2023 | 12 | −7.1 | −4,603 | −0.1% | 66.7% | ₹52.84L |
| 2024 | 12 | +971.2 | +6,31,287 | +17.5% | 66.7% | ₹59.15L |
| 2025 | 12 | +1,915.5 | +12,45,082 | +34.6% | 83.3% | ₹71.60L |
| 2026 H1 | 5 | +1,462.2 | +9,50,446 | +26.4% | 100% | ₹81.10L |

Returns are on the fixed ₹36L base — the rule trades a **fixed 10 lots** and does not
compound. Six positive years, one flat (2023), one mildly negative (2019). No year worse
than −3.2%. **2020 was positive (+14.6%)**: the 21-DTE exit cleared the February trade before
the worst of the COVID crash, and the March trade sold into panic-priced premium for the best
single result in the sample (+866 pts).

### The same year by year, under each VIX filter

Each variant starts from its own ₹36L. Return is on the fixed ₹36L base; DD is the deepest
intra-year drawdown; equity compounds that variant's own rupee flow.

| Year | n | none Ret | none DD | none Equity | n | >25 Ret | >25 DD | >25 Equity | n | >50 Ret | >50 DD | >50 Equity | n | >75 Ret | >75 DD | >75 Equity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | 12 | −3.2% | −9.6% | ₹34.84L | 9 | **+2.4%** | −4.8% | ₹36.85L | 5 | −3.1% | −6.2% | ₹34.88L | 2 | −0.4% | −1.5% | ₹35.87L |
| 2020 | 12 | +14.6% | −8.4% | ₹40.09L | 10 | +19.7% | −7.8% | ₹43.96L | 8 | +27.6% | −6.2% | ₹44.80L | 4 | +16.6% | −6.2% | ₹41.84L |
| 2021 | 12 | +16.8% | −8.8% | ₹46.13L | 7 | +22.6% | 0.0% | ₹52.10L | 3 | +10.3% | 0.0% | ₹48.50L | **0** | — | — | ₹41.84L |
| 2022 | 12 | +18.8% | −8.2% | ₹52.88L | 10 | +13.5% | −8.2% | ₹56.97L | 8 | +6.9% | −8.3% | ₹51.00L | 4 | +14.0% | 0.0% | ₹46.89L |
| 2023 | 12 | −0.1% | **−13.8%** | ₹52.84L | 3 | **+1.9%** | −3.4% | ₹57.64L | 1 | −3.4% | −3.4% | ₹49.78L | **0** | — | — | ₹46.89L |
| 2024 | 12 | +17.5% | −3.8% | ₹59.15L | 10 | +12.9% | −3.8% | ₹62.29L | 9 | +15.0% | −3.8% | ₹55.18L | 5 | +9.5% | −3.8% | ₹50.31L |
| 2025 | 12 | +34.6% | −10.2% | ₹71.60L | 7 | +14.7% | −10.2% | ₹67.59L | 4 | +5.1% | −8.0% | ₹57.00L | 3 | +3.9% | −8.0% | ₹51.71L |
| 2026 H1 | 5 | +26.4% | 0.0% | ₹81.10L | 5 | +26.4% | 0.0% | ₹77.09L | 4 | +22.1% | 0.0% | ₹64.97L | 3 | +21.6% | 0.0% | ₹59.48L |
| **Whole period** | **89** | **11.47%** CAGR | **13.8%** MaxDD | **₹81.10L** | **61** | **10.72%** | **10.2%** | **₹77.09L** | **42** | **8.22%** | **11.7%** | **₹64.97L** | **21** | **6.95%** | **8.0%** | **₹59.48L** |

| Variant | Trades | Net ₹ | CAGR | Max DD | Calmar | Equity end | Losing years |
|---|---|---|---|---|---|---|---|
| No filter | 89 | +45,10,439 | 11.47% | −13.8% | 0.83 | ₹81.10L | 2 |
| **VIX > 25** | 61 | +41,09,314 | 10.72% | **−10.2%** | **1.05** | ₹77.09L | **0** |
| VIX > 50 | 42 | +28,97,001 | 8.22% | −11.7% | 0.70 | ₹64.97L | 2 |
| VIX > 75 | 21 | +23,48,240 | 6.95% | −8.0% | 0.87 | ₹59.48L | 1 (+2 idle) |
| NIFTY 50 buy-and-hold | — | — | 11.60% | −38.4% | 0.30 | — | 1 |

**Reading down the columns changes what the filter appears to be for.**

- **VIX > 25 has no losing year in eight.** It turns 2019 from −3.2% into +2.4% and 2023 from
  −0.1% into +1.9%, because the months it drops are exactly the thin-premium ones that lose. It
  gives up the fat 2025 (+14.7% vs +34.6%) and finishes ₹4L behind, but with a 10.2% max drawdown
  against 13.8%.
- **VIX > 75 has two completely dead years** — zero trades in 2021 and zero in 2023, with ₹36L
  blocked throughout. That is why 172 points per trade collapses into a 6.95% CAGR. High
  per-trade economics with no opportunities is a statistic, not a strategy.
- **VIX > 50 is the worst of the four** (Calmar 0.70): enough selectivity to halve the trade
  count, not enough to improve the drawdown.
- Every variant's drawdown lands in the same episodes, so the filters trim opportunity rather
  than removing a distinct risk.

### Versus the benchmark

| | CAGR | MaxDD | Calmar |
|---|---|---|---|
| 45-DTE straddle, 10 lots on ₹36L | 11.47% | −13.8% | **0.83** |
| NIFTY 50 (price index, same window) | 11.60% | −38.4% | 0.30 |

NIFTY's figure excludes dividends (~1.2%/yr), so on total return the index is modestly ahead
on raw return — and still far behind on risk.

### Capital-buffer sensitivity

| Margin blocked | CAGR | MaxDD as % of capital | Calmar |
|---|---|---|---|
| ₹36L (₹3L/lot + 20% buffer) | 11.47% | 13.8% | 0.83 |
| ₹54L (1.5× buffer) | 8.46% | 9.2% | 0.92 |
| ₹72L (2× buffer) | 6.72% | 6.9% | 0.97 |

---

## 3. Monitoring frequency — now answered on REAL 1-minute option prices

**What we could and could not get.** There is no intraday option history for 2019–2026, and
this was verified rather than assumed: Kite returns **`invalid token`** for expired contracts
— tested against NIFTY 24000/24050/24100 CE on the *July-2026* expiry, one month old, on both
`60minute` and `day` intervals. Expired-contract intraday data is not obtainable.

What we do have is our own recorder: `option_chain`, **28.3M real 1-minute NIFTY option
quotes** from 2026-04-20. It picks each contract up only ~27 days before expiry, so it cannot
host a 45-DTE *entry* — but it covers the DTE 27→0 window at 1-minute resolution, and the
monitoring question lives entirely inside the holding window.

### 3a. Real intraday travel of the ATM straddle (240 recorded day-contracts)

How far the combined premium ranged intraday versus where it closed — exactly what a
daily-close backtest is blind to:

| DTE band | Day-contracts | Travel above close (mean / p95 / max) | Travel below close (mean / p95 / max) |
|---|---|---|---|
| **≥ 21 — the strategy's band** | 60 | **6.3% / 14.2% / 36.5%** | **4.3% / 20.9% / 27.7%** |
| 3–20 (after our exit) | 153 | 8.8% / 20.9% / 35.3% | 4.4% / 21.1% / 44.2% |
| 0–2 (expiry week, never held) | 27 | 381% / 966% / 7669% | 22.6% / 66.9% / 70.2% |

In the DTE ≥ 21 band, across 60 real sessions: **zero days travelled ≥50% in either
direction**, and exactly one travelled ≥30%. The strategy's triggers sit **+100%** (stop) and
**−50%** (target) away from entry credit. Nothing observed comes close to jumping a trigger
within a single session.

The DTE 0–2 row is the gamma cliff the strategy exists to avoid — and it is spectacular
(premium collapsing to near-zero at the close makes "travel above close" explode). It is also
irrelevant here, because the position is closed at 21 DTE. That contrast is the clearest
possible vindication of the 21-DTE exit rule.

### 3b. The three real 45-DTE trades the recorder overlaps

For each, the real 1-minute combined premium as a multiple of entry credit:

| Expiry | Entry | Strike | Credit | Overlap days | Real intraday range (× credit) | Trigger touched? |
|---|---|---|---|---|---|---|
| 2026-05-26 | 2026-04-10 | 24050 | 1,189.7 | 4 (DTE 27→21) | 0.66 – 0.79 | no |
| 2026-06-30 | 2026-05-15 | 23650 | 1,155.4 | 5 (DTE 27→21) | 0.55 – 0.74 | no |
| 2026-07-28 | 2026-06-12 | 23600 | 951.4 | 5 (DTE 27→21) | 0.82 – 1.08 | no |

Across 14 real sessions the premium never left the **0.55× – 1.08×** band. The 0.50 target and
2.00 stop were never approached, at any minute, on any of them.

**Bhav fidelity check (a bonus).** The real 1-minute last quote versus the bhavcopy close we
used for the 7.5-year study differed by at most **17.8 points on ~900** (≈2%), and typically
under 8. The EOD data underpinning the whole replication is faithful.

### 3c. What the sweep says

| Check frequency | n | Win% | Net pts | Net/trade | t | Max DD | Worst trade | T/S/21-DTE |
|---|---|---|---|---|---|---|---|---|
| Daily close | 89 | 70.8 | 6,952.4 | 78.1 | 3.03 | −998.4 | −811.8 | 1 / 2 / 86 |
| **60-min (his)** | 89 | 70.8 | 6,939.1 | 78.0 | 3.12 | **−765.3** | **−578.7** | 1 / 3 / 85 |
| 30-min | 89 | 70.8 | 6,939.1 | 78.0 | 3.12 | −765.3 | −578.7 | 1 / 3 / 85 |
| 15-min | 89 | 70.8 | 6,939.1 | 78.0 | 3.12 | −765.3 | −578.7 | 1 / 3 / 85 |
| 5-min | 89 | 70.8 | 6,939.1 | 78.0 | 3.12 | −765.3 | −578.7 | 1 / 3 / 85 |

**Use hourly; nothing finer earns its keep.** Daily → hourly changes one trade: P&L flat, but
the worst trade improves 29% (−812 → −579 pts, ₹5.28L → ₹3.76L) and max drawdown 23%. Hourly
→ 30 → 15 → 5-minute changes nothing at all. The real 1-minute evidence in §3a/§3b explains
why: in this DTE band the premium simply does not travel far enough within a session.

*(Rows below daily are computed on a reconstructed path — real 5-min NIFTY spot, forward and
IV backed out of real option closes using the previous session's IV, snapped back to the real
price at each daily close. §3a/§3b are the real-tick check on that reconstruction, and they
agree with it.)*

---

## 4. India VIX percentile filter (rank vs previous 252 sessions)

| VIX rank | n (his) | n (ours) | Win% (his / ours) | Avg premium | Net/trade | t | CAGR on ₹36L | MaxDD | Calmar |
|---|---|---|---|---|---|---|---|---|---|
| No filter | 83 | 89 | 69.9 / 70.8 | 786.3 | 78.0 | 3.12 | **11.47%** | 13.8% | 0.83 |
| **> 25** | 55 | 61 | 74.5 / 72.1 | 857.5 | 103.6 | **3.55** | 10.72% | **10.2%** | **1.05** |
| > 50 | 39 | 42 | 76.9 / 71.4 | 919.8 | 106.1 | 2.77 | 8.22% | 11.7% | 0.70 |
| > 75 | 21 | 21 | 85.7 / 71.4 | 1,052.9 | 172.0 | 2.71 | 6.95% | 8.0% | 0.87 |

**His filter works; his explanation of why does not.** Trade counts line up almost exactly
(21 vs 21 at >75), so we are selecting the same days — but the 85.7% win rate **does not
reproduce** (ours 71.4%). What improves is the **size of the win, not the frequency**: average
premium sold rises 786 → 1,053 points and net per trade more than doubles. You are paid more
for the same hit rate. That is the more durable description.

**On capital, >25 is the best cell** (Calmar 1.05 vs 0.83) and no filter has the highest CAGR.
**>75 is worst of both worlds** on this basis — 21 trades in 7.5 years for a 6.95% CAGR.


### Why the filter works — the mechanism, and what it says about December

The filter above is an empirical association until you can say *what* it is selecting for.
It turns out to be one thing, and it is measurable at entry.

**Movement relative to premium is the whole game.** Across all 89 campaigns:

| Predictor | Correlation with net P&L |
|---|---|
| \|move\| (entry spot → exit spot) | −0.770 |
| **\|move\| ÷ breakeven width** | **−0.898** |
| breakeven width alone | +0.330 |

Breakeven width = entry credit ÷ entry spot, i.e. how far NIFTY can travel before the straddle
loses. It is known the moment you sell. The direct test:

**Every campaign where NIFTY moved ≥ +4%: 15 of 17 were losses, average −209 points.** Big
up-moves are the defining risk, and they are not a seasonal artefact.

**Grouped by expiry month** (one campaign per month; the Dec-expiry campaign is held mid-Nov to
early-Dec):

| Month | n | Avg pts | Avg move | \|move\| | Breakeven | move/BE | Win% |
|---|---|---|---|---|---|---|---|
| Jan | 7 | +7.2 | +0.62% | 2.65% | 3.83% | 0.67 | 57% |
| Feb | 8 | +182.1 | −0.01% | 2.10% | 4.43% | 0.46 | **100%** |
| Mar | 7 | +39.9 | −1.02% | 2.62% | 4.47% | 0.65 | 71% |
| **Apr** | 8 | **+230.7** | +1.74% | 2.09% | **6.55%** | **0.44** | 75% |
| May | 8 | +57.0 | +0.58% | 3.19% | 5.90% | 0.62 | 62% |
| Jun | 8 | +169.8 | +3.24% | **3.84%** | 5.72% | 0.61 | 62% |
| Jul | 8 | +10.3 | +3.29% | 3.43% | 4.67% | 0.70 | 62% |
| Aug | 7 | +12.0 | +0.74% | 3.46% | 4.07% | 0.85 | 57% |
| Sep | 7 | +70.0 | +1.59% | 2.26% | 4.05% | 0.60 | 71% |
| Oct | 7 | +101.5 | +0.13% | 2.17% | 3.78% | 0.56 | 86% |
| Nov | 7 | +71.2 | +1.13% | 2.72% | 4.22% | 0.64 | 71% |
| **Dec** | 7 | **−51.4** | +1.94% | 2.95% | **3.75%** | **0.80** | 71% |

**June is the tell.** It has the *largest* average move of any month (3.84%) and is still the
third-best month, because it is paid 5.72% of breakeven. Move alone does not decide the outcome;
move relative to premium does.

**Why December is the only losing month.** Not because the move is large — 2.95% is middling.
Because it is **paid the least of any month (3.75% breakeven, the thinnest in the study)** while
the market drifts up +1.94%. Thin credit against upward drift is a genuinely bad setup, and
December is where that combination clusters. Its average entry VIX rank is 34, among the lowest.

**The "NIFTY rallies in Nov/Dec" premise does not survive a longer sample.** NIFTY 50 calendar
returns, 2011–2026, independent of this study: **Nov +0.95% (53% win), Dec +0.68% (53%)** — both
middling against an all-month average of +0.91%, and far behind Apr (+2.30%) and Oct (+2.07%).
October is the strongest up-month and is our *fourth-best* straddle month. The seasonal premise is
not what drives the December result; the thin premium is.

**And December's total is one trade.** Dec-2023 lost 811.8 points — the worst in the study — sold
on a **2.95% breakeven** into a **+6.39%** move. Strip it and December averages **+75.3**,
mid-table. Both December losses (2023, 2020) were its two largest up-moves.

### Can breakeven width replace the VIX filter? No — but it explains it

If move/premium is the mechanism, filtering directly on breakeven width should work. It does,
monotonically — but it does not beat filtering on VIX.

| Filter | n | Avg pts | t | CAGR | MaxDD | Calmar | Losing years |
|---|---|---|---|---|---|---|---|
| None | 89 | 78.1 | 3.03 | 11.49% | 18.0% | 0.64 | 1 |
| BE ≥ 3.5% | 67 | 91.1 | 3.26 | 10.45% | 10.2% | 1.03 | **0** |
| BE ≥ 4.0% | 49 | 105.3 | 3.03 | 9.20% | 8.9% | 1.04 | 1 |
| BE ≥ 4.5% | 34 | 109.0 | 2.33 | 7.09% | 8.2% | 0.87 | 2 |
| BE ≥ 5.0% | 26 | 113.4 | 2.13 | 5.87% | 11.0% | 0.53 | 2 |
| BE ≥ 6.0% | 13 | 181.4 | 2.34 | 4.86% | 6.2% | 0.79 | 1 |
| **VIX > 25** | 61 | 104.5 | **3.54** | **10.78%** | 10.2% | **1.06** | **0** |
| BE ≥ 4.0% AND VIX > 25 | 46 | 109.3 | 2.97 | 9.02% | 8.9% | 1.02 | 1 |

Average points per trade rises **monotonically** with the breakeven threshold (78 → 91 → 105 →
109 → 113 → 181), which is the signature of a real effect rather than a lucky cell. But VIX > 25
beats every breakeven cell on t-stat, CAGR and Calmar at the same drawdown, and **combining the
two is worse than either alone**.

They are not the same filter — correlation between breakeven width and VIX rank is **0.570**, and
they skip 22 and 28 trades respectively with only 18 in common (56% overlap). What separates them
is visible in the worst trades:

| Worst trade | Net | BE% | VIX rank | BE ≥ 3.5% | VIX > 25 |
|---|---|---|---|---|---|
| **Dec-2023** | **−811.8** | **2.95%** | 21 | **skipped** | **skipped** |
| Mar-2020 | −466.3 | 3.68% | 23 | taken | **skipped** |
| Aug-2022 | −452.3 | 5.01% | 53 | taken | taken |
| May-2025 | −443.9 | 5.54% | 97 | taken | taken |
| Jul-2023 | −405.9 | 3.11% | 2 | **skipped** | **skipped** |
| Sep-2021 | −314.9 | 3.19% | 13 | **skipped** | **skipped** |

Both filters would have skipped the worst trade in the study, on information available at entry.
Neither catches Aug-2022 or May-2025 — fat-premium entries (5.0%, 5.5%) where the move simply
exceeded even a generous breakeven. That residue is the tail no entry filter removes.

**Conclusion: keep VIX > 25.** The breakeven work does not improve it, but it converts it from a
statistical association into a mechanism — the filter is selecting for *being paid enough for the
move you are about to get* — and it explains the December result far better than the calendar does.

---

## 5. Delta management — hold to an x% move, then exit and re-centre?

**Verdict: NO. Every move-managed variant is worse than doing nothing, and re-centring
is worse than the exit alone. If you want less drawdown, trade fewer lots — do not manage.**

The idea is intuitive: a short straddle is hurt by *movement*, not by time, so cut when the
underlying has moved x% and re-sell at the new ATM. Tested on real bhavcopy prices, with the
campaign (one month's expiry, 45 → 21 DTE) as the unit of account so n stays 89 and the
t-stats are directly comparable to §1.

Three arms: **hold** (baseline), **exit_only** (cut on the move, stay flat to 21 DTE), and
**recentre** (cut and immediately sell the new ATM on the same expiry, cap of 1 / 2 / unlimited).

| Move threshold | Arm | Net/campaign | t | CAGR | MaxDD | Calmar | Cycles/campaign |
|---|---|---|---|---|---|---|---|
| — | **hold (baseline)** | **78.1** | **3.03** | **11.48%** | 18.0% | **0.64** | 1.00 |
| 1.0% | recentre (uncapped) | 3.0 | 0.15 | 0.62% | 32.8% | 0.02 | 5.85 |
| 1.0% | exit_only | 10.1 | 1.29 | 2.02% | 9.0% | 0.23 | 1.00 |
| 1.5% | recentre (uncapped) | 18.4 | 0.88 | 3.53% | 25.1% | 0.14 | 4.06 |
| 1.5% | **exit_only** (best managed) | 28.5 | 2.84 | 5.16% | **9.6%** | 0.54 | 1.00 |
| 2.0% | recentre (uncapped) | 11.0 | 0.47 | 2.20% | 25.0% | 0.09 | 3.17 |
| 2.0% | exit_only | 21.2 | 1.60 | 3.99% | 11.1% | 0.36 | 1.00 |
| 2.5% | recentre (uncapped) | 27.9 | 1.15 | 5.07% | 35.0% | 0.15 | 2.47 |
| 3.0% | recentre (uncapped) | 20.8 | 0.84 | 3.93% | 27.0% | 0.15 | 2.10 |
| 4.0% | recentre (uncapped) | 6.5 | 0.23 | 1.33% | 60.4% | 0.02 | 1.62 |
| 5.0% | recentre (uncapped) | 8.6 | 0.30 | 1.74% | 56.6% | 0.03 | 1.35 |

Not one cell beats 78.1 pts. The best managed variant keeps **36%** of the baseline's return.

### Why — the mechanism, not just the number

Decomposing the 2% re-centre arm by how each cycle ended:

| Cycle ended by | n | Avg net | Win rate |
|---|---|---|---|
| **MOVE cut** | 201 | **−28.6 pts** | 38.3% |
| Ran to 21 DTE | 81 | **+83.0 pts** | 81.5% |

**That is the whole story.** A straddle allowed to run to 21 DTE earns +83 points at an 81%
win rate. The same position, cut when the underlying moves 2%, realises −28.6 points at a 38%
win rate. The move rule systematically converts a winner-in-waiting into a booked loser,
because the edge *is* sitting through the move and collecting the decay.

Cost is the minor part of the damage. At 2% the arm runs 3.17 cycles per campaign at 5.5 pts
a round trip = **17.4 pts/campaign of friction against the baseline's 5.5** — roughly 12 points
of the 67-point shortfall. **The other ~55 points is the mechanism.**

Re-deployment adds nothing on top:

| Cycle position in the campaign | n | Avg net |
|---|---|---|
| 1st (the original straddle) | 89 | +21.2 |
| 2nd | 78 | −10.2 |
| 3rd | 50 | −6.2 |
| 4th and later | 65 | +3.1 |

Every re-centre after the first is a coin flip that pays a round trip to play.

### Two checks that could have rescued it, and didn't

**Trigger timing.** Perhaps the rule only loses because a daily-close check reacts a day late.
Re-run with the trigger on the **real 5-minute NIFTY spot** — the first day the intraday range
breaches x% from the anchor, exit at that day's real close. Trigger and fill both real.

| Move | Arm | Close trigger | Intraday trigger |
|---|---|---|---|
| 1.5% | recentre | 18.4 pts | **4.0** |
| 1.5% | exit_only | 28.5 | **13.4** |
| 2.0% | recentre | 11.0 | **5.3** |
| 2.0% | exit_only | 21.2 | 21.4 |
| 3.0% | recentre | 20.8 | 29.6 |

Reacting *earlier* mostly makes it worse, which is what you would expect if cutting is the
problem. (60 of 89 campaigns have complete 5-min coverage; the rest fall back to the close on
missing days.)

**Direction.** Are the cuts symmetric? No — and not in the direction folklore suggests:

| Move | Arm | Up-move cuts | Avg | Down-move cuts | Avg |
|---|---|---|---|---|---|
| 2.0% | recentre | 118 | **−40.7** | 83 | −11.3 |
| 3.0% | recentre | 65 | **−84.9** | 37 | −65.5 |

**Cutting on the way up costs about 3× more than cutting on the way down.** The likely reason
is vega: NIFTY rallies come with falling IV, so a straddle losing on delta is simultaneously
being helped on vol, and it often repairs itself if left alone. Selling out of that move
banks the delta loss and throws away the vol gain.

### The decision that actually matters

The one honest attraction of managing is drawdown: 1.5% exit_only halves MaxDD from 18.0% to
9.6%. But you can buy the same drawdown far more cheaply by just trading smaller:

| Configuration | CAGR | MaxDD | Calmar |
|---|---|---|---|
| Hold, 10 lots | 11.49% | 18.0% | 0.64 |
| Hold, 7 lots | 8.80% | 12.6% | 0.70 |
| **Hold, 5 lots** | **6.73%** | **9.0%** | **0.75** |
| Hold, 4 lots | 5.59% | 7.2% | 0.78 |
| 1.5% exit_only, 10 lots | 5.16% | 9.6% | 0.54 |

**Hold at 5 lots strictly dominates the best managed variant at 10 lots** — more return
(6.73% vs 5.16%), less drawdown (9.0% vs 9.6%), better Calmar (0.75 vs 0.54). Sizing is a
free lever; management is a lever you pay for twice, in friction and in forfeited decay.

*Reproduce:* `run_phase_e_recentre.py` (grid) and `run_phase_e2_diag.py` (direction +
intraday trigger). The baseline arm inside Phase E reproduces §1 exactly (78.1 pts, t 3.03),
which is the control that validates the campaign machinery.

---

## 5b. Premium-triggered management — ADD a second straddle instead of cutting (Phase G)

**Verdict: the ADD idea is directionally right but NOT proven (t 1.15). The
"equidistant pair" is REFUTED. The one solid result is negative and about the
rule we already run: STOPPING OUT costs ~130–190 points per fired campaign at
t ≈ −2.1 to −2.3.**

Phase E killed *move*-triggered management. This asks the different question: fire
on the **premium ratio** (the live 200%-of-credit stop), and instead of cutting,
**add** a second straddle. An arm that never closes the original cannot suffer
Phase E's mechanism — so it deserved its own test.

Applied to **all** campaigns, not only the known losers (that would be look-ahead),
n = 61 on the live VIX-rank>25 ruleset, real bhavcopy closes, both legs of any new
straddle required to carry real volume (≥100) and OI (≥500).

### Aggregate — every arm, VIX>25 book

| Trigger | Arm | fires | net/camp | t | size-normalised | win % | MaxDD | peak units |
|---|---|---|---|---|---|---|---|---|
| — | **HOLD (baseline)** | 0 | **104.2** | **3.53** | 104.2 | 72.1 | **−564.8** | 1.00 |
| 130% | STOP | 12 | 78.5 | 2.39 | 78.5 | 67.2 | −1224.6 | 1.00 |
| 130% | RECENTRE | 10 | 82.6 | 2.56 | 82.6 | 67.2 | −1186.7 | 1.00 |
| 130% | **ADD_ATM** | 11 | **109.7** | 3.77 | **121.0** | 72.1 | **−500.7** | 1.18 |
| 130% | ADD_MIRROR | 8 | 98.9 | 3.17 | 116.5 | 72.1 | −1054.9 | 1.13 |
| 150% | STOP | 7 | 82.4 | 2.45 | 82.4 | 70.5 | −1049.5 | 1.00 |
| 150% | RECENTRE | 7 | 88.7 | 2.74 | 88.7 | 70.5 | −921.2 | 1.00 |
| 150% | **ADD_ATM** | 6 | **110.4** | 3.82 | 116.4 | **75.4** | **−464.3** | 1.10 |
| 150% | ADD_MIRROR | 3 | 99.2 | 3.25 | 107.9 | 72.1 | −847.2 | 1.05 |
| 175% / 200% | *all arms* | **0** | 104.2 | 3.53 | — | 72.1 | −564.8 | 1.00 |

### The paired test — the number that decides

Aggregates over 61 campaigns are mostly the 50 campaigns where nothing fired. The
honest test is the **paired difference on the fired campaigns only**:

| Trigger | Arm | n fired | mean Δ vs HOLD | t | helped |
|---|---|---|---|---|---|
| 130% | **ADD_ATM** | 11 | **+30.6** | **+1.15** | 6/11 |
| 150% | **ADD_ATM** | 6 | +63.7 | +1.75 | 4/6 |
| 130% | ADD_MIRROR | 8 | −39.7 | −0.54 | 4/8 |
| 150% | ADD_MIRROR | 3 | −101.6 | −1.12 | **0/3** |
| 130% | **STOP** | 12 | **−130.6** | **−2.06** | 3/12 |
| 150% | **STOP** | 7 | **−189.2** | **−2.29** | 1/7 |
| 130% | RECENTRE | 10 | −110.5 | −1.38 | 4/10 |
| 150% | RECENTRE | 7 | −135.0 | −1.72 | 1/7 |

**Only the STOP rows clear significance, and they clear it in the harmful
direction.** Cutting helped in 4 of 19 fired campaigns across both triggers. This
reproduces Phase E's mechanism on a completely different trigger — the seventh time
in this repo that tightening a short-premium book has been shown to be destructive.

### ADD_ATM, campaign by campaign (130%) — why t is only 1.15

| Entry | HOLD | ADD_ATM | Δ |
|---|---|---|---|
| 2025-02-10 | **+64.1** | **−83.8** | **−147.9** |
| 2022-07-11 | −452.7 | −500.7 | −48.0 |
| 2020-11-13 | −292.2 | −316.7 | −24.5 |
| 2019-05-13 | −82.1 | −95.1 | −13.0 |
| 2021-11-15 | +56.0 | +49.2 | −6.9 |
| 2025-04-11 | −444.2 | −399.2 | +45.0 |
| 2019-10-14 | −263.6 | −208.7 | +54.9 |
| 2019-03-11 | −119.2 | −26.2 | +93.0 |
| 2022-09-12 | +122.6 | +243.3 | +120.7 |
| 2023-12-11 | −187.0 | −64.2 | +122.8 |
| 2025-03-10 | −120.6 | +20.1 | +140.7 |

The shape is unmistakable: **adding rescues campaigns that were already deep in
the red, and occasionally converts a winner into a loser** (2025-02-10, +64 → −84).
That is averaging down. research/84 already labelled averaging a **tail bomb** — its
apparent win rate is a no-stop illusion and its risk lives in a tail that a
handful of observations cannot show. Eleven campaigns, none of them a sustained
one-way continuation, cannot price that tail. The +30.6 mean is real in-sample and
unproven out of it.

### Two structural findings

**1. The equidistant pair is the worst of the three add variants.** ADD_MIRROR
loses on both triggers and helped 0 of 3 times at 150%. Placing the second straddle
symmetrically *away* from spot sells cheaper, further options, so it collects less
premium precisely where the decay is needed, and it widens the position's damage
band. ADD_ATM — selling the new straddle **at** the money — beat it on every metric.

**2. The live 200% stop never fires on the book we actually run.** On the VIX-rank>25
ruleset the premium ratio never reaches 1.75, let alone 2.00, on a close basis. The
three campaigns in the full 89 that do reach it carry VIX ranks of 21, 23 and 0 —
**all below the filter's own threshold.** The filter already declines to trade every
campaign the stop was built to catch. The two rules are redundant, and the app's
stop card ("fires 2–3 times in 61 trades") is quoting the count from the *unfiltered*
89-trade book. Fortunate, given the stop is harmful when it does fire — but the card
should say so.

### Recommendation

**No change to the live book.** The stop stays where it is precisely because it is
inert; removing it is a separate decision with its own evidence bar, and the arms
that would replace it do not clear theirs. ADD_ATM at 130% is the only idea here
worth keeping on the shelf — mean +30.6 pts, t 1.15, better drawdown — but it needs
either a materially bigger sample or a mechanism argument that survives a
continuation event, and it needs **double the reserved capital**, which the
2026-08-31 stress test says the book does not currently have.

## 6. Robustness

**Convention** — irrelevant: roll back/close 78.1 · roll back/settle 75.5 · roll forward/close
82.2 · roll forward/settle 78.7 pts per trade.

**Parameters** — 45 and 21 are each a local maximum (data-snooping flag), but the *risk*
gradient is cleanly monotonic, which is the part to trust:

| Entry DTE | Net/tr | t | MaxDD | | Exit DTE | Net/tr | t | MaxDD |
|---|---|---|---|---|---|---|---|---|
| 40 | 63.9 | 2.69 | −709 | | 0 (expiry) | 60.6 | 1.20 | −2,803 |
| **45** | **78.1** | **3.03** | −998 | | 7 | 69.3 | 1.55 | −1,840 |
| 50 | 62.1 | 2.05 | −1,421 | | 14 | 69.1 | 1.86 | −1,215 |
| 60 | 59.8 | 1.33 | −3,144 | | **21** | **78.1** | **3.03** | **−998** |
| | | | | | 28 | 41.6 | 2.19 | −1,057 |

Drawdown worsens steadily the later you enter and the longer you hold into expiry. The design
idea — collect fat premium early, leave before gamma — is real and mechanical. "Exactly 45 and
exactly 21" is not special; every neighbouring parameter is still profitable.

**Cost** — not the binding constraint: net/trade 81.8 (gross) → 78.1 (0.25%) → 74.4 (0.5%) →
66.9 (1.0%) → 52.1 (2.0%). Break-even slippage is far beyond anything realistic on ATM NIFTY
monthlies.

**Concentration** — top-3 trades = 25% of total profit; five worst trades cost −2,347 pts
against +6,939 total.

**Seven deadly sins** — look-ahead controlled (causal IV, trailing-252 VIX rank, entry priced
on its own day's close); survivorship N/A (single instrument, every monthly taken); overfitting
— rules fixed by the video, not fitted, with 45/21 flagged as peaks; cost — gross and net plus
a sweep; regime — per-year table above; correlation — one non-overlapping trade at a time, so
the t-stat is honest; capacity — entry requires both legs actually traded, and ATM NIFTY
monthlies are the deepest options in India.

---

## 7. The open item: stress margin

**This is now the main risk, and it is not in the numbers above.**

₹3 lakh per lot is today's margin, with India VIX at **10.83**. SPAN scales with volatility.
India VIX peaked at **83.61 on 2020-03-24** — nearly 8× today's level — and short-option margin
would have risen by a large multiple with it. A ₹36L block would very likely have been
breached in March 2020, forcing either a top-up or a liquidation **at the worst possible
moment**, which the fixed-capital CAGR above does not model.

The buffer table in §2 is a partial answer (at ₹72L the CAGR is 6.7%), but the honest position
is that **a proper stress-margin test is owed before this is sized live**: reconstruct
per-lot SPAN+exposure across 2019-26 from the actual margin files and re-run the equity curve
with a margin-call rule. Until that is done, the 11.47% CAGR is an upper bound.

**Two further risks:**

- **Gap risk.** The stop is evaluated on candle closes; a gap-open past 200% of credit fills
  wherever the market opens. Three stop events in the sample and no overnight catastrophe.
  Short-straddle losses are left-skewed by construction and 89 trades cannot price that tail.
- **Correlation.** This is another short-vol NIFTY position, alongside THE STACK, the NAS book
  and the straddle paper books — they lose in the same week. It adds concentration, not
  diversification. research/89 separately found the *unconditional* monthly NIFTY straddle
  net-negative over 2015-26; the 45→21 window plus a stop is what rescues it.

---

## 8. Recommendation

1. **Believe the table.** It replicates on independent real data.
2. **Use hourly checks.** Free 29% improvement in the worst trade and 23% in drawdown; below
   60 minutes there is provably nothing to gain — now confirmed on real 1-minute quotes.
3. **Do not delta-manage it.** No move threshold, exit rule or re-centring scheme beats
   simply holding to 21 DTE. To cut risk, cut lots.
4. **Use VIX > 25 if you want the best risk-adjusted version** (Calmar 1.05); use no filter if
   you want maximum CAGR (11.47%). Do **not** use > 75 — worst of both on a capital basis.
5. **Run the stress-margin test before sizing live.** This is the one thing standing between
   STRATEGY-CANDIDATE and STRATEGY.
6. **Paper first (G5)**, sized against margin with a margin-call rule, and measure its
   correlation with the existing short-vol book before adding capital.

**Stage gates: G3 robustness PASSED. G4 portfolio CONDITIONAL PASS** — the return-on-capital
objection that killed the first version does not survive the correct (margin) capital basis;
what remains open is stress margin and short-vol correlation, not the edge.

---

*Reproduce:* `run_phase_a.py` (replication) · `run_phase_bc.py` (timeframe + VIX grid) ·
`run_phase_d_intraday.py` (real 1-minute evidence) · `diag_convention.py` · `diag_touch.py`.
All read-only against `market_data.db` and `options_data.db` on the VPS.
