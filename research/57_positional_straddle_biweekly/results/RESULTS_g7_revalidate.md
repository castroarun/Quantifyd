# research/57 G1 — management sweep (bi-weekly short straddle, no wings, 27 trades)

Short ATM straddle 2nd-nearest weekly, daily carry, exit on move-stop / profit-target / DTE<=1. Net Rs80/leg. **30d SIGNAL, overlapping entries.**

| management | total | mean/trade | median | win% | worst | std |
|---|---|---|---|---|---|---|
| move 0.4% | +47325 | +1753 | +1328 | 67 | -3627 | 3672 |
| move 0.5% | +50513 | +1871 | +1328 | 67 | -4917 | 4356 |
| move 0.7% | +98623 | +3653 | +1328 | 67 | -4917 | 6990 |
| move 1.0% | +119960 | +4443 | +3090 | 74 | -4917 | 6775 |
| move 1.5% | +177169 | +6562 | +3103 | 74 | -4999 | 8441 |
| move 0.4%+PT40 | +47325 | +1753 | +1328 | 67 | -3627 | 3672 |
| move 0.7%+PT40 | +103177 | +3821 | +1328 | 67 | -4917 | 7150 |
| move 1.5%+PT40 | +208847 | +7735 | +6454 | 74 | -4999 | 8900 |
| no-mgmt | +194518 | +7204 | +8183 | 67 | -12396 | 11775 |

## Read
- Which move-stop cuts the worst-trade tail WITHOUT killing the mean (monotonic > peak)?
- PT raises win% but caps the big theta runs. 30d SIGNAL, overlapping entries. Best -> G2 wings + G3 entry.