# research/141 — aggregation output

Source: `results/arms_daily.csv` — 2464 rows, 176 venue-days.

## NIFTY

### NIFTY — ALL DAYS  (n=88 days, per LOT)

| arm | total ₹/lot | mean ₹/lot/day | median | win% | worst day | p5 | stop-fire% | avg re-centers | max rc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ONE_AND_DONE** | +18869 | +214 | +969 | 64% | -4541 | -3949 | 28% | 0.00 | 0 |
| RECENTER_1 | +13685 | +156 | +1012 | 66% | -8279 | -7265 | 28% | 0.25 | 1 |
| RECENTER_2 | +7793 | +89 | +1012 | 66% | -11391 | -7038 | 28% | 0.30 | 2 |
| RECENTER_3 | +8493 | +97 | +1012 | 66% | -10945 | -7038 | 28% | 0.31 | 3 |
| RECENTER_5 | +8493 | +97 | +1012 | 66% | -10945 | -7038 | 28% | 0.31 | 3 |
| RECENTER_2_CD15 | +12663 | +144 | +1012 | 66% | -10945 | -6949 | 28% | 0.30 | 2 |
| RECENTER_3_CD15 | +12663 | +144 | +1012 | 66% | -10945 | -6949 | 28% | 0.30 | 2 |
| RECENTER_5_CD15 | +12663 | +144 | +1012 | 66% | -10945 | -6949 | 28% | 0.30 | 2 |
| RECENTER_5_NOGUARD | +8493 | +97 | +1012 | 66% | -10945 | -7038 | 28% | 0.31 | 3 |
| MOVESTOP_ONE | -23078 | -262 | -639 | 35% | -4875 | -3114 | 65% | 0.00 | 0 |
| MOVESTOP_RECENTER | -66057 | -751 | -122 | 47% | -12270 | -5800 | 65% | 1.03 | 4 |
| MOVESTOP_RC1 | -48660 | -553 | -122 | 47% | -5648 | -5205 | 65% | 0.58 | 1 |
| MOVESTOP_RC_CD15 | -64296 | -731 | -122 | 47% | -10404 | -5800 | 65% | 1.02 | 4 |
| NOSTOP_HOLD | +34066 | +387 | +1046 | 67% | -12892 | -5270 | 0% | 0.00 | 0 |

**Paired vs the incumbent `ONE_AND_DONE`** (same days, net of measured cost; family-wise Holm over 13 comparisons):

| arm | Δ mean ₹/lot/day | t | p (raw) | Holm-adj p | beats incumbent? |
|---|---:|---:|---:|---:|---|
| RECENTER_1 | -59 | -0.41 | 0.680 | 1.000 | no |
| RECENTER_2 | -126 | -0.73 | 0.466 | 1.000 | no |
| RECENTER_3 | -118 | -0.70 | 0.485 | 1.000 | no |
| RECENTER_5 | -118 | -0.70 | 0.485 | 1.000 | no |
| RECENTER_2_CD15 | -71 | -0.46 | 0.648 | 1.000 | no |
| RECENTER_3_CD15 | -71 | -0.46 | 0.648 | 1.000 | no |
| RECENTER_5_CD15 | -71 | -0.46 | 0.648 | 1.000 | no |
| RECENTER_5_NOGUARD | -118 | -0.70 | 0.485 | 1.000 | no |
| MOVESTOP_ONE | -477 | -2.09 | 0.036 | 0.362 | no |
| MOVESTOP_RECENTER | -965 | -3.94 | 0.000 | 0.001 | no |
| MOVESTOP_RC1 | -767 | -3.43 | 0.001 | 0.007 | no |
| MOVESTOP_RC_CD15 | -945 | -3.96 | 0.000 | 0.001 | no |
| NOSTOP_HOLD | +173 | +0.84 | 0.401 | 1.000 | not after haircut |

### NIFTY — churn-cost decomposition (what the extra round trips actually cost)

| arm | extra cycles | extra-cycle GROSS ₹/lot | extra-cycle COST ₹/lot | extra-cycle NET ₹/lot | cost as %% of extra gross |
|---|---:|---:|---:|---:|---:|
| RECENTER_1 | 22 | +1150 | 6334 | -5184 | 551% |
| RECENTER_2 | 26 | -2772 | 8304 | -11076 | 300% |
| RECENTER_3 | 27 | -1992 | 8383 | -10375 | 421% |
| RECENTER_5 | 27 | -1992 | 8383 | -10375 | 421% |
| RECENTER_2_CD15 | 26 | +1274 | 7479 | -6205 | 587% |
| RECENTER_3_CD15 | 26 | +1274 | 7479 | -6205 | 587% |
| RECENTER_5_CD15 | 26 | +1274 | 7479 | -6205 | 587% |
| RECENTER_5_NOGUARD | 27 | -1992 | 8383 | -10375 | 421% |
| MOVESTOP_RECENTER | 91 | +4414 | 47392 | -42979 | 1074% |
| MOVESTOP_RC1 | 51 | +2502 | 28084 | -25582 | 1122% |
| MOVESTOP_RC_CD15 | 90 | +6100 | 47318 | -41218 | 776% |

### NIFTY — per trading-DTE (net ₹/lot/day; DTE0 = expiry day)

| DTE | n days | ONE_AND_DONE | RECENTER_1 | RECENTER_2 | RECENTER_3 | RECENTER_5 | RECENTER_3_CD15 | MOVESTOP_ONE | MOVESTOP_RECENTER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | +842 | +1024 | +1024 | +1024 | +1024 | +1038 | +90 | -19 |
| 1 | 19 | -30 | +243 | +243 | +243 | +243 | +243 | +89 | -28 |
| 2 | 16 | +208 | +81 | +81 | +81 | +81 | +81 | +240 | -538 |
| 3+ | 35 | +27 | -305 | -473 | -453 | -453 | -341 | -864 | -1616 |

### NIFTY — OOS split (IS = r/96's own day set ≤ 2026-07-28; OOS = after the deploy decision)

| period | n days | ONE_AND_DONE | RECENTER_1 | RECENTER_2 | RECENTER_3 | RECENTER_5 | RECENTER_3_CD15 | MOVESTOP_ONE | MOVESTOP_RECENTER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS  ≤2026-07-28 | 64 | +306 | +237 | +145 | +156 | +156 | +221 | -249 | -1000 |
| OOS ≥2026-07-29 | 24 | -29 | -62 | -62 | -62 | -62 | -62 | -298 | -85 |

### NIFTY — NEAR-EXPIRY trading DTE<=1  (n=37 days, per LOT)

| arm | total ₹/lot | mean ₹/lot/day | median | win% | worst day | p5 | stop-fire% | avg re-centers | max rc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ONE_AND_DONE** | +14581 | +394 | +1350 | 59% | -4541 | -4276 | 35% | 0.00 | 0 |
| RECENTER_1 | +23050 | +623 | +1491 | 65% | -8279 | -4760 | 35% | 0.30 | 1 |
| RECENTER_2 | +23050 | +623 | +1491 | 65% | -8279 | -4760 | 35% | 0.30 | 1 |
| RECENTER_3 | +23050 | +623 | +1491 | 65% | -8279 | -4760 | 35% | 0.30 | 1 |
| RECENTER_5 | +23050 | +623 | +1491 | 65% | -8279 | -4760 | 35% | 0.30 | 1 |
| RECENTER_2_CD15 | +23294 | +630 | +1491 | 65% | -8279 | -4760 | 35% | 0.30 | 1 |
| RECENTER_3_CD15 | +23294 | +630 | +1491 | 65% | -8279 | -4760 | 35% | 0.30 | 1 |
| RECENTER_5_CD15 | +23294 | +630 | +1491 | 65% | -8279 | -4760 | 35% | 0.30 | 1 |
| RECENTER_5_NOGUARD | +23050 | +623 | +1491 | 65% | -8279 | -4760 | 35% | 0.30 | 1 |
| MOVESTOP_ONE | +3323 | +90 | -231 | 41% | -4240 | -3056 | 65% | 0.00 | 0 |
| MOVESTOP_RECENTER | -885 | -24 | +293 | 51% | -5195 | -4652 | 65% | 0.84 | 3 |
| MOVESTOP_RC1 | -5572 | -151 | +293 | 51% | -5648 | -5205 | 65% | 0.54 | 1 |
| MOVESTOP_RC_CD15 | -953 | -26 | +293 | 51% | -5195 | -4720 | 65% | 0.84 | 3 |
| NOSTOP_HOLD | +14470 | +391 | +1350 | 62% | -12892 | -5270 | 0% | 0.00 | 0 |

**Paired vs the incumbent `ONE_AND_DONE`** (same days, net of measured cost; family-wise Holm over 13 comparisons):

| arm | Δ mean ₹/lot/day | t | p (raw) | Holm-adj p | beats incumbent? |
|---|---:|---:|---:|---:|---|
| RECENTER_1 | +229 | +0.92 | 0.355 | 1.000 | not after haircut |
| RECENTER_2 | +229 | +0.92 | 0.355 | 1.000 | not after haircut |
| RECENTER_3 | +229 | +0.92 | 0.355 | 1.000 | not after haircut |
| RECENTER_5 | +229 | +0.92 | 0.355 | 1.000 | not after haircut |
| RECENTER_2_CD15 | +235 | +0.93 | 0.350 | 1.000 | not after haircut |
| RECENTER_3_CD15 | +235 | +0.93 | 0.350 | 1.000 | not after haircut |
| RECENTER_5_CD15 | +235 | +0.93 | 0.350 | 1.000 | not after haircut |
| RECENTER_5_NOGUARD | +229 | +0.92 | 0.355 | 1.000 | not after haircut |
| MOVESTOP_ONE | -304 | -0.74 | 0.459 | 1.000 | no |
| MOVESTOP_RECENTER | -418 | -1.06 | 0.289 | 1.000 | no |
| MOVESTOP_RC1 | -545 | -1.29 | 0.196 | 1.000 | no |
| MOVESTOP_RC_CD15 | -420 | -1.06 | 0.287 | 1.000 | no |
| NOSTOP_HOLD | -3 | -0.01 | 0.993 | 1.000 | no |

## SENSEX

### SENSEX — ALL DAYS  (n=88 days, per LOT)

| arm | total ₹/lot | mean ₹/lot/day | median | win% | worst day | p5 | stop-fire% | avg re-centers | max rc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ONE_AND_DONE** | -2620 | -30 | +600 | 59% | -4474 | -3562 | 36% | 0.00 | 0 |
| RECENTER_1 | +9864 | +112 | +679 | 65% | -8092 | -6658 | 36% | 0.34 | 1 |
| RECENTER_2 | +4465 | +51 | +679 | 65% | -11489 | -6705 | 36% | 0.43 | 2 |
| RECENTER_3 | +6201 | +70 | +679 | 65% | -10969 | -6705 | 36% | 0.47 | 3 |
| RECENTER_5 | +6201 | +70 | +679 | 65% | -10969 | -6705 | 36% | 0.47 | 3 |
| RECENTER_2_CD15 | +6310 | +72 | +679 | 65% | -11489 | -5951 | 36% | 0.43 | 2 |
| RECENTER_3_CD15 | +5354 | +61 | +679 | 65% | -11489 | -5951 | 36% | 0.44 | 3 |
| RECENTER_5_CD15 | +5354 | +61 | +679 | 65% | -11489 | -5951 | 36% | 0.44 | 3 |
| RECENTER_5_NOGUARD | +6201 | +70 | +679 | 65% | -10969 | -6705 | 36% | 0.47 | 3 |
| MOVESTOP_ONE | -11496 | -131 | -583 | 44% | -3973 | -2569 | 68% | 0.00 | 0 |
| MOVESTOP_RECENTER | -27673 | -314 | +316 | 57% | -8672 | -5081 | 68% | 1.14 | 5 |
| MOVESTOP_RC1 | -27204 | -309 | +22 | 51% | -5942 | -3709 | 68% | 0.61 | 1 |
| MOVESTOP_RC_CD15 | -29380 | -334 | +316 | 57% | -9499 | -5081 | 68% | 1.12 | 4 |
| NOSTOP_HOLD | +43891 | +499 | +892 | 69% | -16347 | -4126 | 0% | 0.00 | 0 |

**Paired vs the incumbent `ONE_AND_DONE`** (same days, net of measured cost; family-wise Holm over 13 comparisons):

| arm | Δ mean ₹/lot/day | t | p (raw) | Holm-adj p | beats incumbent? |
|---|---:|---:|---:|---:|---|
| RECENTER_1 | +142 | +0.75 | 0.454 | 1.000 | not after haircut |
| RECENTER_2 | +81 | +0.37 | 0.712 | 1.000 | not after haircut |
| RECENTER_3 | +100 | +0.47 | 0.637 | 1.000 | not after haircut |
| RECENTER_5 | +100 | +0.47 | 0.637 | 1.000 | not after haircut |
| RECENTER_2_CD15 | +101 | +0.47 | 0.638 | 1.000 | not after haircut |
| RECENTER_3_CD15 | +91 | +0.41 | 0.680 | 1.000 | not after haircut |
| RECENTER_5_CD15 | +91 | +0.41 | 0.680 | 1.000 | not after haircut |
| RECENTER_5_NOGUARD | +100 | +0.47 | 0.637 | 1.000 | not after haircut |
| MOVESTOP_ONE | -101 | -0.44 | 0.658 | 1.000 | no |
| MOVESTOP_RECENTER | -285 | -1.22 | 0.223 | 1.000 | no |
| MOVESTOP_RC1 | -279 | -1.15 | 0.248 | 1.000 | no |
| MOVESTOP_RC_CD15 | -304 | -1.28 | 0.200 | 1.000 | no |
| NOSTOP_HOLD | +529 | +1.94 | 0.053 | 0.688 | not after haircut |

### SENSEX — churn-cost decomposition (what the extra round trips actually cost)

| arm | extra cycles | extra-cycle GROSS ₹/lot | extra-cycle COST ₹/lot | extra-cycle NET ₹/lot | cost as %% of extra gross |
|---|---:|---:|---:|---:|---:|
| RECENTER_1 | 30 | +16070 | 3586 | +12484 | 22% |
| RECENTER_2 | 38 | +11737 | 4652 | +7085 | 40% |
| RECENTER_3 | 41 | +13587 | 4766 | +8821 | 35% |
| RECENTER_5 | 41 | +13587 | 4766 | +8821 | 35% |
| RECENTER_2_CD15 | 38 | +13582 | 4652 | +8930 | 34% |
| RECENTER_3_CD15 | 39 | +12653 | 4679 | +7974 | 37% |
| RECENTER_5_CD15 | 39 | +12653 | 4679 | +7974 | 37% |
| RECENTER_5_NOGUARD | 41 | +13587 | 4766 | +8821 | 35% |
| MOVESTOP_RECENTER | 100 | +3919 | 20096 | -16177 | 513% |
| MOVESTOP_RC1 | 54 | -4082 | 11626 | -15708 | 285% |
| MOVESTOP_RC_CD15 | 99 | +2172 | 20056 | -17884 | 923% |

### SENSEX — per trading-DTE (net ₹/lot/day; DTE0 = expiry day)

| DTE | n days | ONE_AND_DONE | RECENTER_1 | RECENTER_2 | RECENTER_3 | RECENTER_5 | RECENTER_3_CD15 | MOVESTOP_ONE | MOVESTOP_RECENTER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | +74 | +773 | +584 | +531 | +531 | +641 | +138 | +41 |
| 1 | 18 | -406 | -779 | -978 | -829 | -829 | -964 | -1248 | -1553 |
| 2 | 18 | -300 | +183 | +183 | +183 | +183 | +162 | -216 | +17 |
| 3+ | 34 | +258 | +197 | +243 | +243 | +243 | +243 | +364 | -22 |

### SENSEX — OOS split (IS = r/96's own day set ≤ 2026-07-28; OOS = after the deploy decision)

| period | n days | ONE_AND_DONE | RECENTER_1 | RECENTER_2 | RECENTER_3 | RECENTER_5 | RECENTER_3_CD15 | MOVESTOP_ONE | MOVESTOP_RECENTER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS  ≤2026-07-28 | 64 | -93 | +75 | -34 | -7 | -7 | -20 | -217 | -501 |
| OOS ≥2026-07-29 | 24 | +140 | +212 | +277 | +277 | +277 | +277 | +100 | +182 |

### SENSEX — NEAR-EXPIRY trading DTE<=1  (n=36 days, per LOT)

| arm | total ₹/lot | mean ₹/lot/day | median | win% | worst day | p5 | stop-fire% | avg re-centers | max rc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ONE_AND_DONE** | -5977 | -166 | -86 | 50% | -4474 | -3621 | 44% | 0.00 | 0 |
| RECENTER_1 | -117 | -3 | +528 | 61% | -8092 | -6967 | 44% | 0.44 | 1 |
| RECENTER_2 | -7094 | -197 | +528 | 61% | -11489 | -9889 | 44% | 0.64 | 2 |
| RECENTER_3 | -5358 | -149 | +528 | 61% | -10969 | -8306 | 44% | 0.72 | 3 |
| RECENTER_5 | -5358 | -149 | +528 | 61% | -10969 | -8306 | 44% | 0.72 | 3 |
| RECENTER_2_CD15 | -4863 | -135 | +528 | 61% | -11489 | -9697 | 44% | 0.64 | 2 |
| RECENTER_3_CD15 | -5819 | -162 | +528 | 61% | -11489 | -9697 | 44% | 0.67 | 3 |
| RECENTER_5_CD15 | -5819 | -162 | +528 | 61% | -11489 | -9697 | 44% | 0.67 | 3 |
| RECENTER_5_NOGUARD | -5358 | -149 | +528 | 61% | -10969 | -8306 | 44% | 0.72 | 3 |
| MOVESTOP_ONE | -19975 | -555 | -898 | 22% | -3973 | -2037 | 81% | 0.00 | 0 |
| MOVESTOP_RECENTER | -27224 | -756 | -120 | 50% | -8672 | -6471 | 81% | 1.42 | 5 |
| MOVESTOP_RC1 | -29124 | -809 | -753 | 39% | -5942 | -4133 | 81% | 0.72 | 1 |
| MOVESTOP_RC_CD15 | -28674 | -797 | -120 | 50% | -9499 | -6471 | 81% | 1.39 | 4 |
| NOSTOP_HOLD | +34201 | +950 | +1021 | 67% | -16347 | -3620 | 0% | 0.00 | 0 |

**Paired vs the incumbent `ONE_AND_DONE`** (same days, net of measured cost; family-wise Holm over 13 comparisons):

| arm | Δ mean ₹/lot/day | t | p (raw) | Holm-adj p | beats incumbent? |
|---|---:|---:|---:|---:|---|
| RECENTER_1 | +163 | +0.38 | 0.706 | 1.000 | not after haircut |
| RECENTER_2 | -31 | -0.06 | 0.952 | 1.000 | no |
| RECENTER_3 | +17 | +0.03 | 0.972 | 1.000 | not after haircut |
| RECENTER_5 | +17 | +0.03 | 0.972 | 1.000 | not after haircut |
| RECENTER_2_CD15 | +31 | +0.06 | 0.951 | 1.000 | not after haircut |
| RECENTER_3_CD15 | +4 | +0.01 | 0.993 | 1.000 | not after haircut |
| RECENTER_5_CD15 | +4 | +0.01 | 0.993 | 1.000 | not after haircut |
| RECENTER_5_NOGUARD | +17 | +0.03 | 0.972 | 1.000 | not after haircut |
| MOVESTOP_ONE | -389 | -0.89 | 0.373 | 1.000 | no |
| MOVESTOP_RECENTER | -590 | -1.19 | 0.233 | 1.000 | no |
| MOVESTOP_RC1 | -643 | -1.24 | 0.215 | 1.000 | no |
| MOVESTOP_RC_CD15 | -630 | -1.25 | 0.212 | 1.000 | no |
| NOSTOP_HOLD | +1116 | +1.80 | 0.071 | 0.929 | not after haircut |
