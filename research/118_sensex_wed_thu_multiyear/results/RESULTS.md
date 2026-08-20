# research/118 — SENSEX Wednesday vs Thursday, characterised day by day and then across the years

## VERDICT: **NO EDGE in the Wednesday rule — the research/114 Wednesday finding does not replicate and should be reverted. The Thursday HOLD rule survives on expectancy but its risk picture was badly wrong.**

Wednesday is not the dangerous day. Over 5.6 years of 1-minute SENSEX price action it is the
**calmest weekday of the week**, and over 2.6 years of **real BSE option prices** it has the
**fewest catastrophic days of any weekday (1 in 125, 0.8%)** — against Friday's 4.0%, Tuesday's
3.8%, Monday's 3.1% and Thursday's 2.7%. The one catastrophic Wednesday in 125 is
**2026-07-08**, the single day research/114 built its rule on.

The fat tail in a SENSEX short straddle is real, but it belongs to **DTE0 — expiry day, which
is Thursday** — not to Wednesday. 8.7% of DTE0 days lost more than 500 points versus 3.3% on
DTE1 and ~1% on DTE2+. The deployed configuration removes the stop on the fat-tailed day and
keeps it on the thinnest-tailed day.

---

## 1. What was tested, and against what

research/114 replayed 12 Wednesdays and 12 Thursdays of the real 1-minute SENSEX option chain
and concluded **Thursday = HOLD** (+2,630/lot, 92% win, worst −127) and **Wednesday = do not
hold** (−1,112/lot, worst −16,502). Both are deployed today in `config.py` as
`leg_sl_disabled_dtes: (0,)` on `SENSEX_ATM_DEFAULTS` / `SENSEX_ATM4_DEFAULTS` — the per-leg
30% stop is switched off on expiry day and, in the comment's own words, *"Wed and other days
keep it."*

Wednesday's whole verdict rested on a single day. This study asks whether that day was
representative, using three datasets of increasing length and decreasing precision.

| Stage | Source | Coverage | What it gives |
|---|---|---|---|
| **A** | `options_data.db :: option_chain`, 1-minute | **55 days**, 2026-06-03 → 08-20 | Rupee truth, minute-by-minute path, all five weekdays |
| **A2** | `market_data.db :: bse_options_bhav`, daily | **618 days**, 2024-01-01 → 2026-07-30 | **Real traded option prices** across three expiry regimes |
| **B** | `market_data.db :: market_data_unified`, 1-minute | **1,354 days**, 2021-01-01 → 2026-08-20 | Underlying behaviour, five and a half years |

Stage A2 was not in the original brief. It exists because `bse_options_bhav` turned out to
hold the BSE daily bhavcopy for SENSEX options back to January 2024 — real option prices for
618 trading days instead of the recorder's 55. It is the load-bearing evidence in this study.

---

## 2. The trap the brief warned about — and it was a real one

**The SENSEX weekly expiry day moved twice inside our own option history.** Derived from the
data (an expiry is "real" when the last day a contract is quoted equals its expiry date):

| Era | Weekly expiry weekday | Consequence |
|---|---|---|
| 2024-01 → 2024-12 | **Friday** | Wednesday = DTE2, Thursday = DTE1 |
| 2025-01 → 2025-08 | **Tuesday** | Wednesday = DTE6, Thursday = DTE5 |
| 2025-09 → 2026-08 | **Thursday** | Wednesday = DTE1, Thursday = DTE0 |

Cross-checked against `option_chain.expiry_date` for 2026, which shows Thursday expiries
throughout (with Wednesday substitutions when the Thursday was a holiday). Every day in this
study is labelled by **DTE**, not by calendar name.

This is not a technicality — it is the experiment. Because the expiry day moved, "Wednesday
is dangerous" and "DTE1 is dangerous" make *different* predictions, and the data can tell
them apart. Both fail.

---

## 3. Stage A — every recorded day characterised (55 days, 1-minute truth)

09:16 ATM straddle held to 15:15, net of Rs200/lot. All five weekdays, as a control.

| weekday | n | DTE | total | mean | median | win% | worst | best | mean credit | mean MAE | mean abs move |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mon | 10 | 3 | 4,115 | 412 | 1,254 | 70 | −5,156 | 2,678 | 886 | 378.1 | 204.8 |
| Tue | 10 | 2 | 2,635 | 264 | 884 | 80 | −4,088 | 2,490 | 706 | 379.3 | 269.0 |
| **Wed** | 12 | 1 | **−13,347** | **−1,112** | 702 | 67 | **−16,502** | 2,653 | 592 | 574.5 | 353.8 |
| **Thu** | 12 | 0 | **32,512** | **2,709** | 2,162 | 92 | −127 | 8,192 | 377 | 397.1 | 185.9 |
| Fri | 11 | 6 | 4,563 | 415 | 1,635 | 73 | −5,139 | 2,214 | 1,071 | 382.4 | 266.2 |

### Every Wednesday, characterised

| day | credit | terminal move | MAE | MAE at | dir | move/credit | MAE/credit | shape | worst mark | HOLD net |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-03 | 657.2 | +252.1 | 612.0 | 12:03 | DOWN | 0.38 | 0.93 | MIXED | −5,160 | −1,257 |
| 2026-06-10 | 721.9 | −202.3 | 453.9 | 13:05 | UP | 0.28 | 0.63 | MIXED | −891 | 2,476 |
| 2026-06-17 | 602.0 | +277.8 | 340.8 | 11:33 | UP | 0.46 | 0.57 | TREND | −1,129 | 1,356 |
| 2026-06-24 | 635.3 | +672.3 | 857.4 | 13:47 | UP | 1.06 | 1.35 | TREND | −7,998 | −3,778 |
| 2026-07-01 | 654.6 | +314.6 | 499.7 | 12:09 | UP | 0.48 | 0.76 | MIXED | −1,780 | 1,214 |
| **2026-07-08** | 568.1 | **−1,225.7** | **1,507.9** | 14:51 | DOWN | **2.16** | **2.65** | TREND | **−20,623** | **−16,502** |
| 2026-07-15 | 600.5 | −66.4 | 405.1 | 11:33 | UP | 0.11 | 0.68 | SPIKE_REVERT | −2,424 | 2,124 |
| 2026-07-22 | 614.4 | −275.0 | 451.8 | 12:40 | DOWN | 0.45 | 0.73 | MIXED | −1,645 | 109 |
| 2026-07-29 | 503.8 | +191.3 | 304.0 | 14:57 | UP | 0.38 | 0.60 | MIXED | −1,194 | 289 |
| 2026-08-05 | 512.0 | −423.3 | 639.4 | 14:36 | DOWN | 0.83 | 1.25 | MIXED | −6,821 | −3,145 |
| 2026-08-12 | 570.4 | −111.7 | 503.7 | 12:14 | DOWN | 0.20 | 0.88 | SPIKE_REVERT | −728 | 2,653 |
| 2026-08-19 | 465.9 | −233.5 | 318.7 | 14:23 | DOWN | 0.50 | 0.68 | TREND | −429 | 1,114 |

**What the losing days have in common.** Only three Wednesdays lost money, and all three share
one feature: the adverse move was **sustained into the afternoon and still running at the
close** (MAE at 13:47, 14:51, 14:36; revert ratios 0.78, 0.81, 0.66 — the index did not come
back). Nine of twelve Wednesdays never exceeded 1.0× credit at any point. 2026-07-08 is not a
slightly worse version of the others — it is 2.65× credit against a distribution whose next
worst is 1.35×, and it is the only day of the twelve where the move exceeded the credit at all
by more than 6%.

**Which winners nearly became losers.** Across all 55 days, 12 winning days were ever
Rs1,500+/lot under water. Three were Wednesdays (07-01 −1,780; 07-15 −2,424; 07-22 −1,645) and
four were Thursdays (06-04 −5,040; 06-11 −8,221; 06-25 −4,476; 07-09 −4,384). **The Thursdays
that "won 92% with a worst of −127" were repeatedly deep under water intraday** — one of them
(06-11) by Rs8,221/lot before finishing +3,562. Thursday's calm summary statistics hide a
violent path.

### The leave-one-out test

| weekday | n | mean | worst day | mean ex-worst | sign flips? |
|---|---|---|---|---|---|
| Mon | 10 | 412 | −5,156 | 1,030 | no |
| Tue | 10 | 264 | −4,088 | 747 | no |
| **Wed** | 12 | **−1,112** | −16,502 | **+287** | **YES** |
| Thu | 12 | 2,709 | −127 | 2,967 | no |
| Fri | 11 | 415 | −5,139 | 970 | no |

Wednesday is the only weekday whose verdict is hostage to a single observation. The
Wed-vs-Thu permutation test gives p = 0.008 with 2026-07-08 in, and Wednesday's mean becomes
**+287/lot** with it out.

---

## 4. Stage A2 — 618 days of real option prices, and the natural experiment

Open-to-close ATM straddle, front weekly expiry, ATM strike chosen from the **causal 09:16
index level** (never the bhavcopy's own end-of-day `underlying` field, which would be
look-ahead). Both legs required real traded volume and open interest. Unit is **points**
because the SENSEX lot size changed from 10 to 20 inside the window.

### By calendar weekday, whole window

| weekday | n | mean pts | median | win% | sd | p05 | p01 | worst | mean credit |
|---|---|---|---|---|---|---|---|---|---|
| Mon | 128 | 70.6 | 110.9 | 74 | 256.5 | −293.3 | −729.0 | −1,339.2 | 1,037.1 |
| Tue | 131 | 110.4 | 126.8 | 76 | 314.6 | −374.1 | −564.4 | −1,062.0 | 849.7 |
| **Wed** | 125 | **82.9** | 94.8 | **77** | **192.4** | **−265.1** | **−385.2** | −999.5 | 912.5 |
| Thu | 110 | 91.8 | 100.4 | 67 | 283.6 | −354.4 | −753.3 | −1,003.0 | 749.6 |
| Fri | 124 | **41.3** | 63.9 | **62** | 292.7 | −388.5 | −1,022.8 | −1,186.2 | 862.3 |

Wednesday has the **lowest standard deviation, the best p05, the best p01 and the highest win
rate of any weekday**. It is not the worst day — **Friday is**. Wed vs Thu permutation test:
**p = 0.776**. There is no difference to find.

### By DTE, whole window

| DTE | n | mean pts | win% | sd | p05 | worst | mean credit |
|---|---|---|---|---|---|---|---|
| **0** | 127 | **94.7** | 66 | **397.6** | **−616.9** | **−1,186.2** | 521.5 |
| 1 | 123 | 84.2 | 72 | 254.1 | −323.8 | −999.5 | 766.9 |
| 2 | 91 | 89.2 | 71 | 287.5 | −321.9 | −1,062.0 | 918.0 |
| 3 | 92 | 86.1 | 77 | 182.2 | −181.9 | −729.0 | 1,046.0 |
| 4 | 76 | 73.5 | 74 | 265.3 | −203.7 | −1,339.2 | 1,098.3 |
| 6 | 72 | 46.7 | 72 | 113.1 | −148.6 | −246.1 | 1,163.2 |

DTE0 has the highest mean **and by far the fattest tail** — sd 397.6 against DTE1's 254.1 and
DTE3's 182.2. DTE0 vs DTE1 permutation test: **p = 0.806** on means; the difference between
them is entirely in the tail, not the average.

### The natural experiment: does the danger follow the calendar or the expiry?

If Wednesday were intrinsically dangerous it would be the worst weekday in all three eras. If
the danger were DTE1's, it would move with the expiry day.

| Era | Mon | Tue | Wed | Thu | Fri | Worst weekday |
|---|---|---|---|---|---|---|
| Friday-expiry (2024) | 79.9 | 71.3 | 60.9 | 109.9 | 38.2 | **Fri** |
| Tuesday-expiry (2025 Jan–Aug) | 83.5 | 119.0 | 81.9 | 40.9 | 77.1 | **Thu** |
| Thursday-expiry (2025 Sep – 2026) | 52.1 | 149.2 | **105.7** | 110.1 | 18.8 | **Fri** |

*(mean points per day)*

**Wednesday is never the worst weekday in any era.** In the current Thursday-expiry regime —
the one the live rule operates in — Wednesday earns +105.7 points/day at an **80% win rate**
over 46 days. The catastrophic Wednesday of 2026-07-08 is *inside* that sample, contributing
−999.5 points; forty-five other Wednesdays absorb it and the bucket is still comfortably
positive.

### Catastrophic-day frequency — the number that matters most

Days losing more than 500 points:

| weekday | days < −500 pts | DTE | days < −500 pts |
|---|---|---|---|
| Fri | 5 / 124 (**4.0%**) | DTE0 | 11 / 127 (**8.7%**) |
| Tue | 5 / 131 (3.8%) | DTE1 | 4 / 123 (3.3%) |
| Mon | 4 / 128 (3.1%) | DTE2 | 1 / 91 (1.1%) |
| Thu | 3 / 110 (2.7%) | DTE3 | 1 / 92 (1.1%) |
| **Wed** | **1 / 125 (0.8%)** | DTE4 | 1 / 76 (1.3%) |
| — | — | DTE5/6 | 0 / 103 (0%) |

And the frequency of the straddle closing beyond 1×/1.5×/2× the credit:

| bucket | n | loss% | >1× credit | >1.5× | >2× |
|---|---|---|---|---|---|
| DTE0 | 127 | 34 | **7.1** | **4.7** | **1.6** |
| DTE1 | 123 | 28 | 0.8 | 0.8 | 0.8 |
| Wed | 125 | 23 | 0.8 | 0.8 | 0.8 |
| Thu | 110 | 33 | 1.8 | 1.8 | 0.9 |
| Fri | 124 | 38 | 4.0 | 3.2 | 0.8 |

**The tail lives on expiry day.** A DTE0 straddle blows through its entire credit nine times as
often as a DTE1 one. This is the gamma trap research/103 identified, now measured on real
option prices rather than modelled ones.

### The twenty worst days, cross-checked

Every large loss was verified against the index move measured independently from the 1-minute
series. The `intrinsic_1515` rows match the index move almost exactly (e.g. 2026-02-19:
terminal 1,350.68 vs index move −1,348.4; 2026-04-02: 1,802.02 vs 1,770.1), which validates
the expiry-day reconstruction described in §7.

| day | wd | DTE | era | credit | terminal | pnl pts | index move | index MAE | MAE at |
|---|---|---|---|---|---|---|---|---|---|
| 2024-08-05 | Mon | 4 | Fri | 330.4 | 1,669.6 | −1,339.3 | −835.2 | 1,318.8 | 11:24 |
| 2024-11-22 | Fri | 0 | Fri | 565.0 | 1,751.2 | −1,186.2 | +1,715.9 | 1,882.9 | 14:59 |
| 2024-01-23 | Tue | 2 | Fri | 751.4 | 1,813.3 | −1,062.0 | −1,592.5 | 1,756.8 | 15:04 |
| 2024-06-07 | Fri | 0 | Fri | 593.2 | 1,616.0 | −1,022.8 | +1,620.2 | 1,799.5 | 15:00 |
| 2026-02-19 | Thu | 0 | Thu | 347.7 | 1,350.7 | −1,003.0 | −1,348.4 | 1,378.0 | 15:14 |
| **2026-07-08** | **Wed** | **1** | Thu | 425.4 | 1,424.8 | **−999.5** | −1,271.8 | 1,553.9 | 14:51 |
| 2024-07-26 | Fri | 0 | Fri | 469.8 | 1,305.9 | −836.2 | +1,259.5 | 1,274.6 | 15:12 |
| 2026-04-02 | Thu | 0 | Thu | 1,048.7 | 1,802.0 | −753.3 | +1,770.1 | 1,771.4 | 15:15 |

Of the eight worst days in 2.6 years, **four are DTE0 and exactly one is a Wednesday.**

---

## 5. Stage B — 1,354 days of underlying behaviour, 2021–2026

Options do not exist for most of this window, so we measure what actually decides a short
straddle's fate: how far the index travels from 09:16, and how far it goes against you first.

| weekday | n | mean \|move\| | median | p75 | p90 | p95 | p99 | max | mean MAE | MAE p95 | MAE max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mon | 280 | 331.8 | 254.9 | 469.8 | 760.0 | 881.0 | 1,201.7 | 1,840.1 | 501.1 | 1,101.5 | 1,948.1 |
| Tue | 279 | 333.9 | 237.5 | 419.2 | 716.8 | 986.9 | 1,297.1 | 3,086.3 | 494.4 | 1,091.8 | 5,117.9 |
| **Wed** | 276 | **305.5** | 257.0 | **389.9** | **635.3** | **769.5** | 1,271.8 | **1,660.2** | **476.4** | **962.6** | 1,773.4 |
| **Thu** | 242 | **360.7** | 273.9 | 476.2 | 727.4 | 929.5 | **1,348.4** | 1,770.1 | **524.8** | **1,113.0** | 1,771.4 |
| Fri | 274 | 351.8 | 265.9 | 535.2 | 754.4 | 953.0 | 1,205.0 | 1,715.9 | 518.8 | 1,074.2 | 1,882.9 |

**Wednesday has the lowest mean move, the lowest p75, p90 and p95, the lowest mean maximum
adverse excursion and the lowest MAE p95 of any weekday. Thursday has the highest mean move
and the highest mean MAE.** The ordering is the exact opposite of the deployed rule's premise.

It holds in both regimes:

| regime | Mon | Tue | Wed | Thu | Fri |
|---|---|---|---|---|---|
| 2021–2022 (high vol) | 343.5 | 333.4 | **306.2** | 357.4 | 309.6 |
| 2023–2026 (modern) | 325.2 | 334.1 | **305.1** | 362.5 | 375.3 |

And year by year:

| year | Mon | Tue | Wed | Thu | Fri | calmest | wildest | Wed rank |
|---|---|---|---|---|---|---|---|---|
| 2021 | 354.1 | 289.6 | 288.5 | 305.1 | 308.7 | **Wed** | Mon | 1/5 |
| 2022 | 332.6 | 380.0 | 323.8 | 407.3 | 310.4 | Fri | Thu | 2/5 |
| 2023 | 230.8 | 182.1 | 286.5 | 299.3 | 242.0 | Tue | Thu | 4/5 |
| 2024 | 360.1 | 409.0 | 362.2 | 431.1 | 479.7 | Mon | Fri | 2/5 |
| 2025 | 326.5 | 362.1 | 244.3 | 345.4 | 378.6 | **Wed** | Fri | 1/5 |
| 2026 | 411.1 | 387.5 | 345.1 | 380.8 | 428.5 | **Wed** | Fri | 1/5 |

Wednesday is the calmest or second-calmest weekday in five of six years and **has never been
the wildest**. Thursday was the wildest in 2022 and 2023; Friday in 2024, 2025 and 2026.

### Credit ladder — the loss frequency under every plausible premium

We cannot price options before 2024, so a loss day is defined as |terminal move| > credit and
the credit is swept across the range actually observed in 2026.

| credit | Mon | Tue | **Wed** | Thu | Fri |
|---|---|---|---|---|---|
| 300 | 41.1 | 41.6 | 44.6 | 47.5 | 46.4 |
| 400 | 30.7 | 28.0 | **23.9** | 33.1 | 36.5 |
| 465 | 25.4 | 20.4 | **17.4** | 25.6 | 29.9 |
| 550 | 16.8 | 18.6 | **13.0** | 17.4 | 24.1 |
| 650 | 12.5 | 13.6 | **9.4** | 12.8 | 16.4 |
| 720 | 11.1 | 10.0 | **6.2** | 10.3 | 12.0 |
| 850 | 6.1 | 7.5 | **4.0** | 5.8 | 7.7 |

*(% of days the move exceeded the credit)*

**Wednesday has the lowest loss frequency at every credit level from 400 points upward**, and
the ordering is monotonic in the credit — a dose-response, not a lone peak. The result is not
an artefact of one credit assumption.

### Regime split and shape

By causal trailing-20-day realised vol (terciles 0.52% / 0.64%), Wednesday's mean move is the
lowest or second-lowest in the NORMAL and STRESSED buckets and its "loss at 550" frequency is
the lowest in NORMAL (16.3%) and STRESSED (16.3%). Day shape is essentially identical across
weekdays (trend 45–49%, spike-and-revert 21–29%, mean revert ratio 0.60–0.65) — **there is no
"Wednesday trends and Thursday reverts" structure.** Whatever separates the days, it is not
the shape of the intraday path.

Permutation tests on mean |terminal move|, 2021–2026: Wed vs Thu p = 0.021, Wed vs Fri
p = 0.052, Wed vs Mon p = 0.260, Wed vs Tue p = 0.264. **Under a multiple-testing correction
(≈0.005 for ten comparisons) none of these is significant.** The honest reading is that
Wednesday is mildly and *consistently* calmer, and is certainly not more dangerous.

---

## 6. Consensus — and where the datasets disagree

| Question | Stage A (55 d, 1-min) | Stage A2 (618 d, real options) | Stage B (1,354 d, index) | Consensus |
|---|---|---|---|---|
| Is Wednesday the worst weekday? | Yes, by a mile | **No — best win rate, lowest sd, fewest disasters** | **No — calmest weekday** | **NO** — Stage A is the outlier |
| Is Wednesday's verdict robust? | **No, flips on one day** | Yes (drop worst day: 82.9 → 91.7) | Yes | **Stage A's Wednesday is n=1** |
| Is Thursday HOLD positive? | Yes, +2,709/lot | Yes, +1,105/lot adjusted | — | **YES, agreed** |
| Is Thursday HOLD low-risk? | Yes, worst −127 | **NO — worst ≈ −Rs21,500/lot, 8.7% of days < −500 pts** | Thu has highest mean move & MAE | **NO — the datasets disagree loudly** |
| Where is the fat tail? | Wednesday | **DTE0 (Thursday)** | Thursday | **DTE0** |

**The one genuine disagreement is Stage A vs everything else on Wednesday, and it resolves
cleanly:** Stage A covers 12 Wednesdays in a single quarter, one of which was a −1,226-point
trend day. Stage A2 contains that same day (−999.5 points) inside a 46-Wednesday sample from
the same expiry regime and still reports +105.7 points/day at an 80% win rate. Twelve
observations were not enough to see the distribution; forty-six are, and 125 across three
regimes are better still.

**The second disagreement is more important operationally.** Stage A says Thursday's worst day
was −Rs127/lot. Over 127 DTE0 days the worst was around **−Rs21,500/lot**, and 8.7% of DTE0
days lost more than 500 points (>Rs10,000/lot at lot 20). research/114's Thursday risk picture
is not merely optimistic, it is off by two orders of magnitude — because a quarter happened to
contain no bad expiry day.

### Restated on the live construction's terms

The bhavcopy proxy sells at the day's first trade (09:15, when premium is richest) and buys
back at the 15:30 close rather than 15:15. Measured on 38 overlapping days it collects
**64 points (Rs1,281/lot) per day of decay the live 09:16→15:15 trade never sees**
(correlation with the 1-minute truth 0.938, sign agreement 95%). Applying that haircut:

| bucket | n | raw mean pts | adjusted pts | adjusted Rs/lot | win% | p05 pts | worst pts |
|---|---|---|---|---|---|---|---|
| Mon | 128 | 70.6 | 6.5 | −70 | 74 | −293.3 | −1,339.2 |
| Tue | 131 | 110.4 | 46.4 | +727 | 76 | −374.1 | −1,062.0 |
| Wed | 125 | 82.9 | 18.9 | **+178** | 77 | −265.1 | −999.5 |
| Thu | 110 | 91.8 | 27.7 | +355 | 67 | −354.4 | −1,003.0 |
| Fri | 124 | 41.3 | −22.7 | **−654** | 62 | −388.5 | −1,186.2 |
| DTE0 | 127 | 94.7 | 30.6 | +413 | 66 | −616.9 | −1,186.2 |
| DTE1 | 123 | 84.2 | 20.2 | +204 | 72 | −323.8 | −999.5 |
| **Wed & DTE1** (today's rule) | 46 | 73.7 | 9.6 | **−7** | 76 | −323.8 | −999.5 |
| **Thu & DTE0** (today's rule) | 41 | 129.3 | 65.2 | **+1,105** | 71 | −489.6 | −1,003.0 |

Held all day, a SENSEX ATM straddle is roughly **break-even to modestly positive on every
weekday except Friday**. Wednesday-at-DTE1 is a coin flip around zero (−Rs7/lot on 46 days) —
which is a very different statement from "Wednesday loses Rs1,112/lot." It is not proof that
holding Wednesday makes money; it is proof that **Wednesday is not distinguishable from the
other non-expiry days**, and that singling it out has no basis.

### research/114 next to research/118

| rule | r114 n | r114 mean | r114 win | r114 worst | r118 n | r118 mean (adj) | r118 win | r118 worst |
|---|---|---|---|---|---|---|---|---|
| Wednesday HOLD | 12 | −Rs1,112 | 67% | −Rs16,502 | 46 | **−Rs7** | 65% | −Rs21,470 |
| Thursday HOLD | 12 | +Rs2,709 | 92% | **−Rs127** | 41 | **+Rs1,105** | 68% | **−Rs21,542** |

---

## 7. Seven deadly sins accounting

| Sin | How it is controlled here |
|---|---|
| **Look-ahead** | ATM strike always chosen from the **causal 09:16 index level**, never the bhavcopy's end-of-day `underlying` field. Regime buckets use a trailing-20-day realised vol computed only from prior days. The expiry-era map is derived from contracts' own last-quote dates, not assumed. |
| **Survivorship** | Not applicable — one index, and the bhavcopy sample is every trading day in the window rather than a screened set. The 1-minute chain window is simply the recorder's coverage, and its non-representativeness is the study's central finding. |
| **Overfitting / multiple testing** | No parameter is fitted. Five weekdays × a handful of metrics; a Bonferroni-style threshold (~0.005) is stated and applied to the interpretation. Leave-one-out is run on every bucket. The credit ladder is swept, not picked. |
| **Cost neglect** | 1.0 pt/leg-side slippage + Rs30/leg-side/lot = Rs200/lot round trip on both legs, applied everywhere. On top of that, the **measured Rs1,281/lot proxy optimism** is deducted from every Stage A2 conclusion — a second and much larger cost correction. |
| **Regime dependence** | Three expiry eras reported separately; 2021–22 vs 2023–26 split; per-year table; causal trailing-vol terciles. The conclusion holds in all of them. |
| **Correlation / single-factor** | One instrument, one construction. Stated as a limit: this study answers "is Wednesday structurally dangerous", not "is the book diversified". |
| **Capacity / shortability** | Not a binding constraint at 2 lots on ATM SENSEX weeklies, the most liquid contracts on BFO; both legs were required to show real traded volume and OI (the binding rule from research/89). Not analysed at size. |

---

## 8. Honest caveats

1. **Stage A2 is open-to-close, not 09:16-to-15:15.** It is corrected by a level shift measured
   on only **38 overlapping days, all in 2026**. Per-weekday bias is noisy (Mon +1,781,
   Thu +1,666, Tue +1,586, Wed +1,249, **Fri +79**). Friday's near-zero measured bias means
   the harsh adjusted Friday number is the least trustworthy figure in this study and should
   not be acted on without its own check.
2. **Stage A2 is hold-to-close only.** It never simulates the per-leg 30% stop, because no
   intraday option data exists before 2026-04. It therefore refutes the *premise* of the
   Wednesday rule; it does not directly price the stop itself.
3. **DTE0 terminal values are reconstructed as intrinsic against the 15:15 index level**,
   because the BSE file overwrites `close`, `settle` and `underlying` with the settlement
   index on expiry rows (see §9). This ignores residual time value, so DTE0 P&L is slightly
   flattered. The worst-day cross-check matched the independently-measured index move to
   within a point or two, so the error is small — but it is one-directional.
4. **`open` is the day's first trade**, not a guaranteed 09:15:00 print. For ATM weeklies this
   is a fraction of a second's difference; for anything less liquid it would not be.
5. **2021–2023 contributes underlying behaviour only.** SENSEX weekly options do not appear in
   our data before 2024-01-01, so no DTE label exists for those years and the weekday tables
   there measure the index, not a straddle.
6. **n per era-weekday cell is 29–52.** Era-level statements are directional; only the pooled
   weekday and DTE tables (n ≈ 110–131) carry real weight.
7. **India VIX was available** (`INDIAVIX` in `market_data_unified`) but not used; regime
   bucketing uses causal trailing realised vol instead. A VIX-conditioned cut is an open lever.
8. Two prior studies flagged SENSEX-Wednesday as a fat-tail danger — **research/104** (p05
   ≈ −Rs17k/lot, n=15) and **research/114** (n=12). Both drew on the same 2026 quarter and,
   substantially, the same handful of days. This study supersedes both on that specific claim.

---

## 9. Data findings worth keeping

- **`market_data.db :: bse_options_bhav`** holds real BSE daily option prices for **SENSEX and
  BANKEX, 2024-01-01 → 2026-08-04** (289,859 SENSEX rows). This was not previously used in any
  study and is the only multi-year real SENSEX option source we have. research/103 was forced
  to model straddles off the index because of this gap; it did not need to be.
- **Expiry-day rows are corrupt for buy-back purposes.** Where `trade_date == expiry_date`, the
  `close`, `settle` and `underlying` columns are all overwritten with the settlement index
  level (every option on 2026-07-30 shows close = 77,928.15). Naively using `close` produces
  losses of about −Rs3,000,000/lot. Open/high/low are fine. **Any future study touching this
  table must settle DTE0 at intrinsic.**
- **Open interest legitimately collapses to 0 on 2024 expiry rows**, so an OI filter applied
  uniformly silently deletes every expiry day — which would have removed exactly the days that
  carry the tail. Apply OI filters to DTE > 0 only.
- The SENSEX index minute series has small gaps (2026-06-26 and 2026-07-09 are absent);
  `option_chain.underlying_spot` is a workable fallback.

---

## 10. Recommendation

### The deployed Wednesday rule should be **reverted**, but not by editing an engine today

`config.py` currently keeps the per-leg 30% stop on Wednesday on the strength of one day. That
justification is void:

- Wednesday is the **calmest weekday** of the SENSEX week over 5.6 years and **never the
  wildest in any year**.
- Wednesday has the **fewest catastrophic days of any weekday** over 2.6 years of real options
  (0.8% vs 2.7–4.0%).
- Wednesday's research/114 verdict **flips sign** when one day is removed, and the same day
  sits inside a 46-Wednesday sample that is comfortably positive.
- research/114's own Wednesday table already showed the retained rule losing money: LEG30 on
  Wednesday was **−Rs412/lot at a 25% win rate**, ranked 6th of 17 variants and negative. It
  was never the best Wednesday choice even on that data — it merely beat HOLD *because of*
  2026-07-08.

**But** this study cannot price the per-leg stop over 2024–2026 (caveat 2), so the correct next
step is a **dedicated G2 study of the per-leg 30% stop on Wednesday using the 1-minute chain**,
now that the recorder has accumulated more Wednesdays, rather than a config change on this
evidence. Until then the live setting is a small ongoing cost, not a risk.

### The deployed Thursday rule should **stand — with its risk assumption corrected**

Holding on DTE0 is positive-expectancy in every independent cut (r114 +Rs2,709/lot on n=12;
r118 Thu&DTE0 +Rs1,105/lot on n=41; DTE0 overall +Rs413/lot on n=127). Keep
`leg_sl_disabled_dtes: (0,)`.

What must change is the **stated risk**. The config comment and any sizing derived from it
carry research/114's "92% win, worst −127". The truth over 127 DTE0 days is **34% losers, 8.7%
of days worse than −500 points, and a worst day near −Rs21,500/lot**. DTE0 is the single
fattest-tailed slot in the entire dataset — it blows through the full credit nine times as
often as DTE1. Removing the stop there is defensible on expectancy, but it is an explicit
decision to accept a gamma tail, and it should be **sized for a −Rs21,500/lot day, not a
−Rs127 one**. This aligns with research/103's conclusion that the real DTE0 lever is sizing.

### If any weekday deserves special treatment, it is Friday

Friday is the worst weekday in the real-option sample (adjusted −Rs654/lot, 62% win, 4.0%
catastrophic days) and the worst in the current Thursday-expiry era. Under Thursday expiry
Friday is DTE6 — the freshly listed weekly, richest credit, least decay. Flagged as a lever,
subject to caveat 1.

### Process lesson, for the playbook

**A weekday rule on SENSEX must be validated against the expiry-era history.** BSE moved the
SENSEX weekly expiry twice inside our own option data (Friday → Tuesday → Thursday). Any study
that says "Wednesday" without saying which DTE that was in each era is comparing three
different instruments. And a live rule set from **n=12 inside a single quarter** is a rule set
from noise — this is the second time (research/104, research/114) the same quarter has
produced the same false SENSEX-Wednesday alarm.

---

## Files

| File | Purpose |
|---|---|
| `scripts/stage_a_characterise_options_days.py` | 1-min chain, per-day characterisation, all weekdays |
| `scripts/stage_a2_bhav_straddle_multiyear.py` | Real BSE bhavcopy straddle replay 2024–2026 + expiry-era derivation |
| `scripts/stage_b_multiyear_price_action.py` | SENSEX 1-min index, one row per day, 2021–2026 |
| `scripts/stage_c_analysis.py` | All aggregation, permutation tests, cross-validation |
| `scripts/list_worst_days.py` | Worst-day audit with independent index cross-check |
| `results/stage_a_day_characterisation.csv` | 55 days, 1-min truth |
| `results/stage_a2_bhav_straddle_daily.csv` | 618 days, real option prices |
| `results/stage_a2_expiry_eras.csv` | Data-derived expiry weekday by month |
| `results/stage_b_daily_price_action.csv` | 1,354 days of index behaviour |
| `results/worst_days_2024_2026.csv` | The 20 worst days, cross-checked |
| `results/stage_c_tables.md` | Every table in full |
