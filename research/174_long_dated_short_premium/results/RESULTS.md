# research/174 — NIFTY long-dated short premium: 60 / 90 / 180 / 365-DTE straddles, strangles and condors

## VERDICT: **NO EDGE beyond 45 DTE — CONCLUDED.** The live book is already at the sweet spot, the longer tenors decay monotonically to nothing, the 1-year contracts are untradeable, and an eighth independent test says no stop beats no stop.

Nothing is recommended for deployment. Nothing live was touched. One permanent data
repair was made to `nse_options_bhav` (+2,836,719 rows) as a by-product.

---

## 1. What was asked and what came back

> **Arun, 2026-09-15:** *"now that we are live with 45 DTE, can we test for more like 2 months
> away straddles/strangles/condors with different DTE entries and DTE exits and different stop
> losses, same for other liquid ones 3 months, 6 months away, 1 year away etc?"* and *"sls can be
> combined premium, single side, underlying price move, vix, relative vix or combinations"*.

| Question | Answer |
|---|---|
| **2 months away (60 DTE)?** | **No.** t 1.36 against 45 DTE's 2.96, drawdown 3× deeper, loses money in the first half of the sample. |
| **3 months (90 DTE)?** | **No.** Net **negative** (−5.5 pts/trade), win rate 48%. |
| **6 months (180 DTE)?** | **No.** Net −70.4 pts/trade, and only 22 independent trades exist in twelve years. |
| **1 year (365 DTE)?** | **Untradeable.** The ATM call trades **54 contracts a day**; the 2-year contracts trade **four**. Killed on the data, not on the P&L. |
| **Different DTE exits?** | **Not resolvable.** At 140 trades with a per-trade SD of ~225 pts, exit at 9 / 14 / 21 / 27 DTE cannot be told apart. 21 DTE is defensible and unrefuted; it is not demonstrated. |
| **Strangles?** | Lower variance, lower margin, **lower return**. Paired, the 5% strangle loses 26 pts a trade to the straddle and wins on only 39% of them. |
| **Condors / winged strangles?** | **Killed at every tenor.** Index wings cost more than they save — the same result r/128 reached from the other direction. |
| **Stops — premium, single-side, move, VIX, relative VIX?** | **All refuted, paired, at every tenor.** Damage is monotone in how often the stop fires. Single-side is no better than whole-position at matched fire rates. |

---

## 2. The data gate, and a repo-wide defect found and fixed

**`nse_options_bhav` was missing every NIFTY expiry beyond ~75 DTE for 2016-01 → 2024-02,
2026-04/05/06 and 2026-09.** Cause: `MAX_DTE = 75` in
`research/89_short_monthly_straddle/scripts/download_nse_bhav_stocks.py`, which resumed by
trade-date, so every session it touched was marked "done" and the uncapped production
downloader (`download_nse_bhav.py`) skipped it for ever.

Repaired: 2,102 sessions re-downloaded (0 errors), 7,567,118 rows staged,
**+2,836,719 new rows merged**. Every year 2015→2026 now has complete long-dated coverage.
The download was staged to a separate file during market hours and merged after 15:40 IST so
it never took a write lock on the DB the live executors read.

### Tenor liquidity gate (2,887 sessions, 2015→2026, best listed contract near each tenor)

| target tenor | sessions with a fillable near-ATM pair | nearest strike's distance from spot | ATM call contracts/day | ATM call OI | verdict |
|---|---|---|---|---|---|
| 45 | 99% | 0.10% | 1,245 | 87,600 | tradeable |
| 60 | 100% | 0.12% | 566 | 59,750 | tradeable |
| 90 | 97% | 0.27% | 256 | 47,888 | tradeable |
| 120 | 70% | 1.03% | 462 | 210,700 | tradeable when listed |
| 180 | 73% (100% since 2022) | 1.21% | 311 | 175,600 | tradeable when listed |
| 270 | 62% | 1.20% | 125 | 115,962 | marginal |
| **365** | 59% | 1.17% | **54** | 67,150 | **too thin** |
| **545** | 48% | 1.62% | **9** | 18,375 | **dead** |
| **730** | 23% | 2.12% | **4** | 6,188 | **dead** |

Three structural facts the tenor axis has to live with:

1. **You cannot sell an ATM straddle past ~105 DTE.** The strike grid widens from 50 points to
   1,000 and then 1,500, so the nearest listed strike is 1.0–1.2% from spot for everything from
   120 to 365 days. Whatever the long end is, it is not the same trade.
2. **The 120/180 gaps are calendar gaps, not liquidity failures.** Beyond ~75 days NSE lists
   only the quarterly and semi-annual series.
3. **Long tenors mechanically starve the statistics.** A 365-day book has one or two
   non-overlapping trades a year. Twelve years of perfect data yields nine. No amount of care
   makes that conclusive.

---

## 3. The tenor bake-off — a monotone decay, not a peak

Fixed rule, no per-tenor cherry-picking: short ATM straddle entered at tenor T, exited at
21 DTE, no stop, 50% target, 0.75% slippage, liquidity floor 25 contracts on both legs,
2015-01 → 2026-09. Margin is **measured from Kite** (`basket_order_margins`, NRML, standalone,
2026-09-15, spot 23,190), not modelled.

| entry tenor | n | days held | avg credit | **avg net / trade** | **t** | pts per lot-year | entry margin | return on margin | max DD | win% |
|---|---|---|---|---|---|---|---|---|---|---|
| **45 (live)** | 140 | 24.6 | 639.5 | **+56.6** | **2.96** | **838** | ₹2.22L | **24.5%/yr** | −1,243 | **73.6%** |
| 60 | 138 | 41.2 | 745.0 | +46.3 | 1.36 | 410 | ₹2.26L | 11.8% | −3,671 | 66.7% |
| 75 | 139 | 54.0 | 812.3 | +30.5 | 0.78 | 206 | ₹2.33L | 5.7% | −4,795 | 59.0% |
| 90 | 50 | 67.3 | 923.1 | −5.5 | −0.08 | −30 | ₹2.44L | −0.8% | −4,415 | 48.0% |
| 105 | 35 | 81.3 | 1,056.5 | −32.2 | −0.31 | −145 | ₹2.55L | −3.7% | −4,277 | 54.3% |
| 120 | 32 | 93.0 | 1,117.7 | +0.2 | 0.00 | 1 | ₹2.66L | 0.0% | −2,914 | 62.5% |
| 150 | 25 | 118.3 | 1,317.6 | −11.4 | −0.08 | −35 | ₹2.88L | −0.8% | −3,024 | 52.0% |
| 180 | 22 | 149.1 | 1,478.8 | −70.4 | −0.35 | −172 | ₹3.09L | −3.6% | −4,763 | 54.5% |
| 210 | 23 | 169.0 | 1,582.1 | −164.3 | −0.86 | −355 | ₹3.59L | −6.4% | −8,554 | 52.2% |
| 240 | 19 | 187.9 | 1,698.5 | −158.0 | −0.59 | −307 | ₹4.41L | −4.5% | −7,760 | 52.6% |
| 270 | 14 | 231.5 | 1,925.5 | −173.8 | −0.47 | −274 | ₹5.23L | −3.4% | −6,743 | 57.1% |
| 300 | 11 | 259.5 | 1,875.7 | −47.2 | −0.16 | −66 | ₹5.73L | −0.8% | −3,707 | 36.4% |
| 365 | 9 | 307.2 | 1,906.1 | +88.0 | 0.25 | 105 | ₹5.94L | 1.1% | −1,446 | 55.6% |

**This is a monotone dose-response, which is stronger evidence than a peak.** Win rate falls
smoothly 73.6 → 66.7 → 59.0 → 48.0. Net per trade falls 56.6 → 46.3 → 30.5 → −5.5 → −32.2.
Drawdown grows 1,243 → 3,671 → 4,795. Beyond 90 days the book is negative on average.

**Selling more premium earns less.** The 365-day straddle collects three times the credit
(1,906 vs 640 points), nets nothing, and blocks 2.7× the margin for twelve times as long.

Across the full **624-cell** sweep (14 tenors × 6 exits × 4 targets × 2 liquidity floors),
exactly **two cells clear t = 2** — and both of them are the live book: 45→21 (t 2.96) and
45→20 (t 2.80). The third-best is a 10-trade sample.

### Margin, measured not modelled (Kite NRML, standalone, per lot, spot 23,190)

| DTE | 42 | 69 | 105 | 196 | 287 | 469 | 651 | 1015 |
|---|---|---|---|---|---|---|---|---|
| ATM straddle | ₹2.21L | ₹2.29L | ₹2.55L | ₹3.20L | **₹5.69L** | ₹6.28L | ₹6.81L | ₹7.76L |
| 2.5% strangle | ₹1.87L | ₹1.97L | ₹1.98L | ₹2.37L | ₹4.88L | ₹5.34L | ₹6.10L | ₹7.10L |
| 2.5/7 condor | ₹1.48L | ₹1.72L | ₹1.41L | ₹1.61L | ₹3.10L | ₹3.91L | ₹5.51L | ₹6.32L |

---

## 4. Why every management rule fails — the decomposition

The 45→21 DTE hold was chopped into five 5-day pieces, each re-picking the ATM strike at its
own start. Same calendar exposure, re-centred instead of held.

| | gross pts/trade | net pts/trade | t |
|---|---|---|---|
| **held as ONE position, 45→21 DTE** | **+68.5** | **+56.6** | **2.96** |
| 45→40, fresh ATM | +14.2 | +1.8 | 0.33 |
| 40→35, fresh ATM | +15.0 | +3.3 | 0.52 |
| 35→30, fresh ATM | −8.7 | −20.2 | −2.20 |
| 30→25, fresh ATM | +14.7 | +3.3 | 0.47 |
| 25→21, fresh ATM | +10.1 | −1.1 | −0.20 |
| **sum of the five pieces** | **+45.2** | **−12.8** | — |

Re-centring costs **23 points of gross** (a third of the edge) and **four extra round trips**
(~58 points). Together that is the whole +69. Attribution, not an identity — the pieces have
different trade counts.

**There is no theta window to go and collect.** Every individual DTE window across the whole
contract life is statistically indistinguishable from zero (t between −2.2 and +0.9, including
the long ones: 365→300 t 0.66, 300→240 t 0.84, 240→210 t 0.21). The return comes from holding
one strike through the drift. That single fact explains r/119's Phase E (move-triggered
re-centring), Phase G (premium-triggered stop) and all seven stop families below.

---

## 5. Stops — the eighth independent refutation

Every family run **paired against no-stop on identical trades**, all tenors. 45 DTE, target 50%,
0.75% slippage, 141 trades:

| family | Arun's words | best param that actually fires (≥10% of trades) | % fired | paired mean delta | paired t | paired win% |
|---|---|---|---|---|---|---|
| **NONE** | — | — | 0% | **+51.28 avg net, t 2.56** | — | — |
| MOVE | "underlying price move" | 5% | 22.7% | **−37.3** | **−2.84** | 6.4% |
| VIXL | "vix" | 25 | 12.1% | **−26.0** | **−2.68** | 2.1% |
| VIXR | "relative vix" (rank) | 95 | 14.2% | −19.2 | −1.87 | 2.8% |
| SIDE | "single side" | 2.5× leg entry | 19.9% | −19.1 | −1.88 | 7.1% |
| SIDEK | single side on strike cross | 2% | 83.7% | −20.0 | −0.85 | 33.3% |
| PREM | "combined premium" | 1.5× credit | 12.1% | −5.3 | −0.74 | 3.5% |
| VIXD | VIX vs entry-day VIX | 1.25× | 12.1% | −11.6 | −1.40 | 2.1% |

Three things worth stating plainly:

1. **The damage is monotone in fire rate.** Move-stop: 86% fired → −49.7; 61% → −68.5;
   41% → −65.2; 25% → −58.6; 10% → −22.7. VIX-level: 63% → −63.6 down to 5% → −14.3. The only
   settings that do not lose are set so wide they fire on 1–3% of trades, where the paired win
   rate is 1.1% — it helped once, by luck.
2. **The single-side idea is refuted specifically, and it was the most promising one.** The
   hypothesis was that keeping the untouched leg preserves the theta that pays for the loss. At
   matched fire rates single-side (19.9% fired, −19.1) and whole-position premium stop
   (12.1% fired, −5.3 / 23.7% fired, −28.9) are indistinguishable. All the damage is in cutting
   the threatened leg.
3. **Stops are genuine insurance at a genuine premium.** VIX-rank ≥ 60 cuts the worst trade from
   −1,049 to −464 points; the 2% move-stop cuts it to −264. If the goal is a smaller tail rather
   than more return, that is the honest menu — but r/119 already showed the cheaper way to buy
   the same thing is to trade fewer lots.

The same table at 60, 90, 120, 180 and 240 DTE reaches the same conclusion: **not one family at
any tenor has a positive paired delta with t > 1.4.**

---

## 6. Structures — condors and wings are dead on the index

45 DTE, exit 21, target 50%, 0.75% slippage charged **per leg on gross premium turnover** (not
on net credit — that is how four-leg structures get flattered):

| structure | n | win% | avg credit | avg net | t | worst trade | entry margin | return on margin |
|---|---|---|---|---|---|---|---|---|
| 5% strangle | 143 | 82.5 | 182.8 | +41.4 | **3.62** | **−615** | ₹1.57L | **30.6%** |
| 3.5% strangle | 143 | 75.5 | 268.2 | +44.6 | 3.20 | −830 | ₹1.74L | 27.1% |
| **ATM straddle (live)** | 140 | 73.6 | 639.5 | **+56.6** | 2.96 | −1,047 | ₹2.22L | 24.5% |
| 2.5% strangle | 142 | 72.5 | 343.2 | +44.4 | 2.81 | −892 | ₹1.88L | 24.2% |
| 1.5% strangle | 141 | 72.3 | 440.4 | +47.9 | 2.70 | −994 | ₹2.00L | 23.6% |
| winged straddle ±10% | 139 | 69.1 | 577.1 | +21.4 | 1.36 | −992 | ₹1.89L | 10.9% |
| condor 5% body / 7% wing | 139 | 77.0 | 125.7 | +10.2 | 1.18 | −618 | ₹1.30L | 8.7% |
| condor 2.5% / 3% | 139 | 59.7 | 190.2 | **−0.7** | −0.10 | −365 | ₹1.10L | −0.7% |

Condors and winged straddles lose at 45, 60, 90, 120, 180 and 240 DTE, and get worse with tenor
(at 90 DTE, condor 2.5/3 is −42.1 pts at t −2.39). **Index wings are cost without protection** —
r/128 found the same thing testing index wings on stock strangles. This is *not* a contradiction
of r/127: that study's 2.5% body + 7% wings works on **stocks**, whose tails are idiosyncratic.

### The strangle, correctly stated

The 5% strangle's higher t-stat, shallower tail and better return-on-margin are all real, and
**it still loses paired against the straddle in points**:

| vs the live 45-DTE ATM straddle, same 140 entry days | 3.5% strangle | 5% strangle |
|---|---|---|
| median paired delta | −6.8 pts | **−26.4 pts** |
| mean paired delta | −10.1 | −13.6 |
| paired t | −1.32 | −1.27 |
| trades it wins on | 63/140 (45.0%) | **55/140 (39.3%)** |

It is a **lower-return, lower-variance, lower-margin version of the same bet** (correlation
0.87), not a better one. It wins only if the freed ₹0.65L per lot is actually redeployed — a
capital-allocation decision, not a signal. An earlier interim of mine called it a winner on the
unpaired table; **that claim is retracted.**

---

## 7. Robustness on the one thing that does work (the live 45-DTE book)

| | full | first half | second half |
|---|---|---|---|
| no VIX filter | +64.3/trade, t 3.18, 128 trades | +42.1, t 1.80 | +82.6, t 2.63 |
| **VIX rank > 25 (the live rule)** | **+86.5/trade, t 3.64, 87 trades** | +56.6, t 1.85 | +113.2, t 3.18 |
| VIX rank > 50 | +95.5/trade, t 2.97, 58 trades | +77.5, t 1.90 | +112.3, t 2.26 |

Both halves positive in every filter setting. **The VIX-rank entry filter is worth a lot and
behaves monotonically** — 27.8% / 37.3% / 41.5% per year on margin at off / >25 / >50. The live
rule is validated; the >50 variant is better still in-sample on 58 trades and is **not** proposed
on that basis.

### Cost ladder (return on measured margin, no VIX filter, full window)

| slippage per side | 45 ATM straddle | 45 5% strangle | 60 ATM straddle |
|---|---|---|---|
| 0.25% | 30.5%/yr (t 3.49) | 35.5%/yr (t 4.06) | 7.3%/yr (t 0.72) |
| 0.75% | 27.8%/yr (t 3.18) | 34.3%/yr (t 3.93) | 5.1%/yr (t 0.50) |
| 1.50% | 23.7%/yr (t 2.72) | 32.5%/yr (t 3.73) | 1.8%/yr (t 0.17) |

The strangle's cost robustness is its one genuine advantage: it loses **8%** of its return across
that ladder where the straddle loses **22%**, because it sells 189 points of premium instead of
663 and pays slippage on a third of the turnover.

### Tradeability gate (net of cost at 0.75%, points; 1 pt = ₹65/lot)

| spec | n | win% | avg win | avg loss | expectancy | max losing streak | trades/yr | worst |
|---|---|---|---|---|---|---|---|---|
| 45 ATM straddle (live) | 140 | 73.6 | +156.6 | −221.6 | **+56.6** | 3 | 11.7 | −1,047 |
| 45 3.5% strangle | 143 | 75.5 | +110.7 | −159.5 | +44.6 | 2 | 11.9 | −830 |
| 45 5% strangle | 143 | 82.5 | +84.0 | −159.9 | +41.4 | 2 | 11.9 | −615 |
| 60 ATM straddle | 138 | 69.6 | +199.0 | −372.7 | +25.0 | 3 | 11.5 | −2,000 |
| 60 5% strangle | 140 | 72.1 | +114.0 | −257.6 | +10.5 | 3 | 11.7 | −1,326 |

### Per-year net points (0.75% slippage, no VIX filter; intra-year drawdown in brackets, measured off the running peak of the FULL curve per r/154)

| year | 45 ATM (live) | 60 ATM | 45 3.5% strangle | 45 5% strangle | 60 5% strangle |
|---|---|---|---|---|---|
| 2016 | +670.0 (−59) | +22.1 (−449) | +565.0 (−52) | +533.2 (0) | +103.3 (−212) |
| 2017 | +240.9 (−294) | −442.2 (−764) | +217.5 (−147) | +167.8 (−106) | −133.8 (−253) |
| 2018 | +109.4 (−578) | −1,104.3 (−1,683) | +263.7 (−337) | +154.8 (−259) | −446.7 (−680) |
| 2019 | +87.1 (−395) | +658.1 (−1,534) | +127.7 (−237) | +159.8 (−134) | +675.9 (−707) |
| 2020 | +1,035.0 (−363) | −1,850.6 (−2,875) | +1,196.0 (−317) | +1,344.4 (−237) | −778.2 (−1,326) |
| 2021 | +671.5 (−520) | +1,406.9 (−3,254) | +812.3 (−246) | +905.9 (−145) | −379.3 (−1,758) |
| 2022 | +1,224.8 (−464) | +1,197.4 (−1,432) | +1,146.1 (−356) | +988.1 (−268) | +800.3 (−1,233) |
| 2023 | **−241.9 (−1,047)** | −1,674.9 (−3,392) | −242.8 (−830) | −203.9 (−624) | −820.0 (−1,455) |
| 2024 | +1,044.9 (−1,243) | +1,339.2 (−3,080) | +800.9 (−913) | +961.4 (−653) | +802.8 (−1,420) |
| 2025 | +2,135.3 (−167) | +2,361.7 (−1,696) | +964.2 (−315) | +614.9 (−221) | +819.8 (−510) |
| 2026 | +1,249.1 (−644) | +267.3 (−1,428) | +732.6 (−543) | +475.9 (−453) | +120.5 (−798) |
| **best year / worst year** | +2,135 / −242 | +2,362 / −1,851 | +1,196 / −243 | +1,344 / −204 | +820 / −820 |
| **years positive** | **10 / 11** | 7 / 11 | 10 / 11 | 10 / 11 | 7 / 11 |

2023 is the only losing year for the 45-DTE book in eleven. The 60-DTE book loses in four,
including −1,850 in 2020 and −1,675 in 2023.

---

## 8. Honest caveats

- **Daily close only.** Expired-contract intraday option data cannot be obtained from Kite
  ("invalid token"), and the 1-minute recorder starts 2026-04-20 at ~27 DTE. Every exit here is
  a close-based decision on a real traded price. **No intraday claim is made anywhere.** A stop
  that would have fired intraday and reverted by the close is invisible to this study — which
  means the stop families are, if anything, tested in their *most favourable* form.
- **Slippage at the long end is assumed, not measured.** Bhavcopy carries no bid/ask. For a
  strike trading 54 contracts a day, 0.75% of premium is probably generous; the long tenors are
  reported at 0.25/0.75/1.5% and are negative at all three, so the assumption is not load-bearing.
- **Sample size is the binding limit at the long end, and no method fixes it.** 365 DTE yields
  nine non-overlapping trades in twelve years. Those cells are reported for completeness and
  should not be read as evidence either way; the kill rests on liquidity and on the monotone
  decay from 45 outward, not on their point estimates.
- **Multiple testing.** 624 cells in P1, 168 in P2, ~340 in P3. The 45→21 result is not a
  discovery of this study — it is the incumbent, independently reproduced (t 3.12 on r/119's
  2019+ window, t 2.96 on 2015+), and it sits at the top of a monotone ordering rather than on a
  lone peak.
- **Margin is a single-day snapshot** (2026-09-15, spot 23,190). SPAN scans move with volatility;
  the long-tenor requirements in particular will be higher in a stressed market, which makes the
  long end worse, not better.
- **Survivorship / regime.** 2015→2026 contains one genuine volatility crisis (2020) and one bad
  grind (2023). It does not contain 2008.
- **The strangle finding is in-sample and unadopted.** It was selected from 14 structures. Its
  plateau (1.5/2.5/3.5/5.0 all beat the straddle on t-stat) is reassuring, but it loses paired on
  points and is a replacement, not an addition (correlation 0.87).

## 9. Next levers

1. **None on tenor.** This line is closed. If it is reopened, it needs new evidence — a genuine
   change in NSE's long-dated listing and liquidity, which the liquidity probe can re-measure in
   one command.
2. **The strangle-vs-straddle capital question is open and is a portfolio question, not a signal
   question**: does freeing ₹0.65L per lot and accepting 26 fewer points per trade improve the
   whole book? That needs the Capital Desk's view of what the freed margin would do, and it is a
   change to a live book, so it needs its own study and its own deploy.
3. **The stop question should now be treated as closed** (eight independent refutations across
   r/119 E, r/119 G, r/127 B3, r/128, r/129, r/130, r/135 and this study). Any future proposal to
   manage a short-premium position should be required to explain why it is not a ninth.

---

**Reproducibility stamp.** Data snapshot: `backtest_data/market_data.db` on the VPS as of
2026-09-15 after the r/174 merge (`nse_options_bhav` 36,054,242 rows). Margin: Kite
`basket_order_margins`, NRML, `consider_positions=False`, 2026-09-15, NIFTY spot 23,190, lot 65.
Scripts: `research/174_long_dated_short_premium/scripts/`. Cost model: `engine_lt.costs_points_legs`
— slippage on gross per-leg premium turnover, STT 0.10% of sell-side premium, exchange 0.05% of
turnover, ₹20/order per leg per side, GST 18% on brokerage + exchange.
