## OA — Base Age · axis A: concentration, fully invested (slot_pct = 1/slots)

Full window 2005-01-03 → 2026-09-11, after tax, 25 bps a side, 5.0% post-tax idle cash, 30 seeds, medians.

| slots | size/slot | CAGR med | [worst..best] | MaxDD | Calmar | invested | tr/yr | max loss streak | entries taken | slot-blocked | cash-blocked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 16.67% | 21.89% | 20.69 .. 25.39 | -42.50% | 0.516 | 74.1% | 12.3 | 11 | 266 | 1785 | 1567 |
| 7 | 14.29% | 20.75% | 19.83 .. 22.41 | -39.98% | 0.519 | 74.5% | 14.5 | 9 | 315 | 1651 | 1656 |
| 8 | 12.50% | 20.37% | 19.08 .. 22.53 | -35.64% | 0.571 | 75.3% | 16.6 | 11 | 361 | 1587 | 1671 |
| 9 | 11.11% | 21.56% | 20.89 .. 22.45 | -34.26% | 0.629 | 75.0% | 18.7 | 11 | 405 | 1533 | 1680 |
| 10 | 10.00% | 22.46% | 20.70 .. 24.11 | -33.57% | 0.669 | 75.0% | 20.7 | 11 | 448 | 1442 | 1733 |
| 11 | 9.09% | 22.03% | 20.45 .. 23.72 | -33.85% | 0.670 | 74.9% | 22.5 | 12 | 487 | 1371 | 1759 |
| 12 | 8.33% | 21.33% | 19.88 .. 22.31 | -33.60% | 0.633 | 74.2% | 24.4 | 13 | 530 | 1225 | 1863 |
| 13 | 7.69% | 20.76% | 19.57 .. 21.80 | -33.29% | 0.616 | 73.8% | 26.3 | 13 | 570 | 1128 | 1920 |
| 14 | 7.14% | 21.21% | 20.59 .. 22.39 | -34.41% | 0.615 | 73.7% | 27.9 | 14 | 606 | 1074 | 1940 |
| **16 | 6.25% | **20.94%** | 19.81 .. 21.74 | -35.50% | **0.601** | 72.9% | 31.7 | 14 | 688 | 977 | 1955 |
| 18 | 5.56% | 20.29% | 19.25 .. 21.00 | -36.54% | 0.556 | 71.6% | 35.3 | 14 | 766 | 845 | 2007 |
| 20 | 5.00% | 20.57% | 19.42 .. 21.52 | -34.86% | 0.585 | 71.0% | 38.3 | 15 | 830 | 781 | 2008 |
| 24 | 4.17% | 19.87% | 19.32 .. 20.31 | -35.12% | 0.566 | 69.0% | 45.1 | 15 | 977 | 617 | 2024 |
| 30 | 3.33% | 17.55% | 17.26 .. 17.89 | -31.08% | 0.565 | 66.3% | 55.1 | 15 | 1195 | 423 | 2000 |

*16 slots in bold is the incumbent. "slot-blocked" = qualifying signals refused because no slot was free; "cash-blocked" = signals whose slot WAS free but the book had no cash to fill it.*

## OA — Base Age · axis B: fixed 6.25% size, deliberate cash buffer

| cell | slots | size/slot | max invested | CAGR | MaxDD | Calmar | avg invested | cash sleeve pp | tr/yr |
|---|---|---|---|---|---|---|---|---|---|
| B_s08_p0625 | 8 | 6.25% | 50% | 15.46% | -23.69% | 0.647 | 45.3% | 2.89 | 18.8 |
| B_s10_p0625 | 10 | 6.25% | 62% | 16.94% | -26.13% | 0.651 | 54.6% | 2.37 | 23.3 |
| B_s12_p0625 | 12 | 6.25% | 75% | 18.52% | -30.57% | 0.601 | 63.3% | 2.44 | 27.3 |
| B_s14_p0625 | 14 | 6.25% | 88% | 20.73% | -33.45% | 0.615 | 71.2% | 1.97 | 30.8 |

## OA — Base Age · axis C: size independent of slot count

| cell | slots | size/slot | max invested | CAGR | MaxDD | Calmar | avg invested | cash sleeve pp | tr/yr |
|---|---|---|---|---|---|---|---|---|---|
| C_s10_p0400 | 10 | 4.00% | 40% | 12.82% | -17.12% | 0.754 | 35.9% | 3.30 | 23.3 |
| C_s10_p0500 | 10 | 5.00% | 50% | 14.68% | -21.25% | 0.691 | 44.3% | 2.89 | 23.3 |
| C_s10_p0800 | 10 | 8.00% | 80% | 20.05% | -32.45% | 0.627 | 68.5% | 1.64 | 23.3 |
| C_s16_p0400 | 16 | 4.00% | 64% | 16.07% | -25.84% | 0.621 | 52.8% | 2.36 | 35.2 |
| C_s16_p0500 | 16 | 5.00% | 80% | 18.68% | -31.01% | 0.589 | 65.0% | 1.69 | 35.2 |
| C_s20_p0400 | 20 | 4.00% | 80% | 17.74% | -29.88% | 0.593 | 62.7% | 2.18 | 42.7 |

## OA — Base Age · axis D: who wins a contested slot (vs the random draw null)

| rule | slots | CAGR | MaxDD | Calmar | vs random CAGR | seeds beaten (CAGR) | seeds beaten (Calmar) |
|---|---|---|---|---|---|---|---|
| random draw (null) | 8 | 20.37% | -35.64% | 0.571 | — | — | — |
| rs | 8 | 21.62% | -35.64% | 0.607 | +1.25 pp | 25/30 | 25/30 |
| ext | 8 | 21.75% | -36.46% | 0.596 | +1.38 pp | 26/30 | 23/30 |
| tv | 8 | 22.97% | -35.64% | 0.645 | +2.60 pp | 30/30 | 30/30 |
| age | 8 | 19.75% | -35.64% | 0.554 | -0.62 pp | 8/30 | 8/30 |
| random draw (null) | 16 | 20.94% | -35.50% | 0.601 | — | — | — |
| rs | 16 | 20.59% | -36.05% | 0.571 | -0.35 pp | 5/30 | 5/30 |
| ext | 16 | 20.86% | -35.99% | 0.579 | -0.08 pp | 9/30 | 9/30 |
| tv | 16 | 21.72% | -32.67% | 0.665 | +0.78 pp | 29/30 | 29/30 |
| age | 16 | 21.75% | -37.59% | 0.579 | +0.81 pp | 30/30 | 9/30 |

*The ranked rules are DETERMINISTIC — they consume no randomness, so each has a single path. "seeds beaten" is that one path against all 30 random-draw paths, which is the honest test of a fixed rule against the draw's luck.*

## OA — Base Age · pre-registered ranking: Calmar subject to CAGR >= the 16-slot baseline, paired on the same 30 seeds

| rank | cell | CAGR | MaxDD | Calmar | ΔCalmar | Calmar seeds won | ΔCAGR | CAGR seeds won | W1 CAGR (seeds won) | W2 CAGR (seeds won) | W1→W2 | robust? | clears the bar? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | **A_s16_eq (incumbent)** | 20.94% | -35.50% | 0.601 | — | — | — | — | 18.94% | 22.90% | +3.96 | — | — |
| 1 | A_s11_eq | 22.03% | -33.85% | 0.670 | +0.058 | 26/30 | +1.13 pp | 27/30 | 19.72% (29/30) | 24.38% (22/30) | +4.66 | YES | **no: ΔCalmar +0.058 < 0.10 and ΔCAGR +1.13 < 2pp** |
| 2 | A_s10_eq | 22.46% | -33.57% | 0.669 | +0.064 | 25/30 | +1.40 pp | 27/30 | 20.20% (29/30) | 24.82% (21/30) | +4.62 | YES | **no: ΔCalmar +0.064 < 0.10 and ΔCAGR +1.40 < 2pp** |
| 3 | D_tv_s16 | 21.72% | -32.67% | 0.665 | +0.064 | 29/30 | +0.78 pp | 29/30 | 19.67% (30/30) | 23.73% (22/30) | +4.06 | YES | **no: ΔCalmar +0.064 < 0.10 and ΔCAGR +0.78 < 2pp** |
| 4 | D_tv_s08 | 22.97% | -35.64% | 0.645 | +0.044 | 25/30 | +2.03 pp | 30/30 | 18.27% (0/30) | 27.94% (30/30) | +9.67 | YES | **no: fails a seed-win or drawdown clause** |
| 5 | A_s12_eq | 21.33% | -33.60% | 0.633 | +0.025 | 24/30 | +0.33 pp | 22/30 | 19.55% (29/30) | 23.09% (15/30) | +3.54 | YES | **no: ΔCalmar +0.025 < 0.10 and ΔCAGR +0.33 < 2pp** |
| 6 | A_s09_eq | 21.56% | -34.26% | 0.629 | +0.028 | 22/30 | +0.53 pp | 24/30 | 20.07% (30/30) | 23.07% (15/30) | +3.00 | YES | **no: ΔCalmar +0.028 < 0.10 and ΔCAGR +0.53 < 2pp** |
| 7 | A_s14_eq | 21.21% | -34.41% | 0.615 | +0.014 | 16/30 | +0.32 pp | 21/30 | 19.90% (30/30) | 22.47% (11/30) | +2.57 | YES | **no: ΔCalmar +0.014 < 0.10 and ΔCAGR +0.32 < 2pp** |
| 8 | D_rs_s08 | 21.62% | -35.64% | 0.607 | +0.006 | 19/30 | +0.68 pp | 28/30 | 18.27% (0/30) | 25.10% (30/30) | +6.83 | YES | **no: ΔCalmar +0.006 < 0.10 and ΔCAGR +0.68 < 2pp** |

## OA — Base Age · cost ladder (after-tax CAGR / Calmar, 30 seeds)

| cell | 25 bps | 40 bps | 60 bps |
|---|---|---|---|
| A_s16_eq | 20.94% / 0.601 | 20.29% / 0.569 | 19.39% / 0.560 |
| A_s11_eq | 22.03% / 0.670 | 21.61% / 0.639 | 20.59% / 0.600 |
| A_s10_eq | 22.46% / 0.669 | 21.55% / 0.645 | 20.76% / 0.617 |
| D_tv_s16 | 21.72% / 0.665 | 21.02% / 0.633 | 20.08% / 0.536 |

## OA — Base Age · tradeability gate and capacity

| cell | win rate | avg win | avg loss | max losing streak | trades/yr | median position ₹ | as % of the name's 20-day traded value | p95 | trades above 1% | multiple all | multiple drop-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_s16_eq | 49.0% | +37.41% | -11.58% | 14 | 31.7 | ₹542,587 | 0.426% | 4.704% | 32.9% |
| A_s11_eq | 49.2% | +38.59% | -11.55% | 12 | 22.5 | ₹888,504 | 0.673% | 8.894% | 42.4% |
| A_s10_eq | 49.0% | +38.93% | -11.55% | 11 | 20.7 | ₹911,877 | 0.772% | 10.703% | 45.9% |
| D_tv_s16 | 49.9% | +37.19% | -11.52% | 14 | 31.5 | ₹602,579 | 0.435% | 5.172% | 34.4% |

*Capacity is measured on a ₹10,00,000 book. The percentages scale linearly with capital: at ₹1 crore every figure here is 10× larger, which is the number that matters for sizing this book up.*

## OA — Base Age · does slot contention actually bind? (30-seed medians, 3,619 qualifying events over 1,880 signal days)

| slots | entries actually taken | refused: no free slot | refused: free slot but no cash | days the slot cap bound | days the book was completely full |
|---|---|---|---|---|---|
| 6 | 266 | 1785 | 1567 | 894 | 152 |
| 7 | 315 | 1651 | 1656 | 826 | 112 |
| 8 | 361 | 1587 | 1671 | 792 | 115 |
| 9 | 405 | 1533 | 1680 | 761 | 111 |
| 10 | 448 | 1442 | 1733 | 715 | 100 |
| 11 | 487 | 1371 | 1759 | 676 | 96 |
| 12 | 530 | 1225 | 1863 | 608 | 52 |
| 13 | 570 | 1128 | 1920 | 559 | 60 |
| 14 | 606 | 1074 | 1940 | 529 | 50 |
| 16 | 688 | 977 | 1955 | 472 | 41 |
| 18 | 766 | 845 | 2007 | 407 | 31 |
| 20 | 830 | 781 | 2008 | 381 | 17 |
| 24 | 977 | 617 | 2024 | 312 | 5 |
| 30 | 1195 | 423 | 2000 | 223 | 1 |

*Read the third column first. At EVERY slot count the commonest reason a qualifying signal is not taken is that the book has no CASH, not that it has no SLOT — because the book never trims a winner, so a handful of bloated positions can absorb 95% of NAV while slots sit nominally free.*

## OA — Base Age · house YoY table: incumbent vs best cell vs NIFTYBEES

Each cell is the calendar-year return with that year's max drawdown beneath it, measured from the running peak of the FULL curve. After tax, net of 25 bps a side, median-seed path (incumbent seed 29, challenger seed 23). Benchmarks are excluded from the best-of picks.

| year | OA · Base Age 16 slots (incumbent) | OA · Base Age A_s11_eq | NIFTYBEES (benchmark) | BEST CAGR | LEAST DD | BEST OVERALL |
|---|---|---|---|---|---|---|
| 2005 | +14.0<br><sub>(-12.3)</sub> | +10.2<br><sub>(-14.5)</sub> | +32.8<br><sub>(-14.0)</sub> | 16 slots | 16 slots | 16 slots |
| 2006 | +37.7<br><sub>(-20.6)</sub> | +49.7<br><sub>(-21.0)</sub> | +41.3<br><sub>(-29.9)</sub> | A_s11_eq | 16 slots | A_s11_eq |
| 2007 | +86.7<br><sub>(-10.4)</sub> | +61.6<br><sub>(-11.4)</sub> | +53.0<br><sub>(-14.9)</sub> | 16 slots | 16 slots | 16 slots |
| 2008 | -30.1<br><sub>(-32.7)</sub> | -30.4<br><sub>(-32.2)</sub> | -52.1<br><sub>(-59.7)</sub> | 16 slots | A_s11_eq | A_s11_eq |
| 2009 | +62.7<br><sub>(-31.9)</sub> | +79.7<br><sub>(-31.3)</sub> | +75.6<br><sub>(-59.1)</sub> | A_s11_eq | A_s11_eq | A_s11_eq |
| 2010 | +15.5<br><sub>(-15.7)</sub> | +15.8<br><sub>(-17.5)</sub> | +18.6<br><sub>(-25.0)</sub> | A_s11_eq | 16 slots | 16 slots |
| 2011 | -10.6<br><sub>(-18.7)</sub> | -10.8<br><sub>(-19.5)</sub> | -24.0<br><sub>(-27.3)</sub> | 16 slots | 16 slots | 16 slots |
| 2012 | +28.2<br><sub>(-19.6)</sub> | +16.1<br><sub>(-20.2)</sub> | +26.5<br><sub>(-26.0)</sub> | 16 slots | 16 slots | 16 slots |
| 2013 | +3.7<br><sub>(-9.3)</sub> | +6.3<br><sub>(-16.7)</sub> | +7.2<br><sub>(-16.0)</sub> | A_s11_eq | 16 slots | 16 slots |
| 2014 | +49.4<br><sub>(-7.1)</sub> | +66.0<br><sub>(-8.7)</sub> | +31.6<br><sub>(-6.2)</sub> | A_s11_eq | 16 slots | A_s11_eq |
| 2015 | -2.3<br><sub>(-22.8)</sub> | -1.5<br><sub>(-24.7)</sub> | -4.3<br><sub>(-15.0)</sub> | A_s11_eq | 16 slots | 16 slots |
| 2016 | +7.2<br><sub>(-28.8)</sub> | +13.2<br><sub>(-28.7)</sub> | +4.0<br><sub>(-21.6)</sub> | A_s11_eq | A_s11_eq | A_s11_eq |
| 2017 | +64.6<br><sub>(-12.6)</sub> | +66.5<br><sub>(-11.0)</sub> | +29.9<br><sub>(-8.5)</sub> | A_s11_eq | A_s11_eq | A_s11_eq |
| 2018 | -26.3<br><sub>(-32.2)</sub> | -26.8<br><sub>(-30.2)</sub> | +4.8<br><sub>(-14.1)</sub> | 16 slots | A_s11_eq | A_s11_eq |
| 2019 | +30.0<br><sub>(-32.6)</sub> | +35.6<br><sub>(-30.0)</sub> | +13.6<br><sub>(-10.5)</sub> | A_s11_eq | A_s11_eq | A_s11_eq |
| 2020 | +48.1<br><sub>(-19.6)</sub> | +59.4<br><sub>(-17.9)</sub> | +15.4<br><sub>(-36.3)</sub> | A_s11_eq | A_s11_eq | A_s11_eq |
| 2021 | +83.7<br><sub>(-10.9)</sub> | +56.7<br><sub>(-13.3)</sub> | +26.0<br><sub>(-9.5)</sub> | 16 slots | 16 slots | 16 slots |
| 2022 | -4.0<br><sub>(-26.6)</sub> | +2.3<br><sub>(-24.2)</sub> | +5.5<br><sub>(-16.1)</sub> | A_s11_eq | A_s11_eq | A_s11_eq |
| 2023 | +53.3<br><sub>(-18.1)</sub> | +60.8<br><sub>(-10.4)</sub> | +21.0<br><sub>(-9.7)</sub> | A_s11_eq | A_s11_eq | A_s11_eq |
| 2024 | +4.1<br><sub>(-24.5)</sub> | +1.2<br><sub>(-19.9)</sub> | +10.4<br><sub>(-10.5)</sub> | 16 slots | A_s11_eq | A_s11_eq |
| 2025 | +2.1<br><sub>(-15.9)</sub> | +6.7<br><sub>(-20.5)</sub> | +11.7<br><sub>(-15.2)</sub> | A_s11_eq | 16 slots | 16 slots |
| 2026 | +28.9<br><sub>(-18.1)</sub> | +31.0<br><sub>(-24.5)</sub> | -9.5<br><sub>(-14.8)</sub> | A_s11_eq | 16 slots | 16 slots |
| **CAGR / MaxDD / Calmar** | **20.95% / -32.73% / 0.64** | **22.07% / -32.22% / 0.69** | **12.30% / -59.71% / 0.21** | | | |

*All three columns span the same window, 2005-01-03 → 2026-09-11.*
