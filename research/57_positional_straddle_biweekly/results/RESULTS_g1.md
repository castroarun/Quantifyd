# research/57 G1 — management sweep (bi-weekly short straddle, no wings, 26 trades)

Short ATM straddle 2nd-nearest weekly, daily carry, exit on move-stop / profit-target / DTE<=1. Net Rs80/leg. **30d SIGNAL, overlapping entries.**

| management | total | mean/trade | median | win% | worst | std |
|---|---|---|---|---|---|---|
| no-mgmt | +205468 | +7903 | +8987 | 69 | -12396 | 11324 |
| move 1.0% | +113152 | +4352 | +2875 | 73 | -4917 | 6903 |
| move 1.5% | +170361 | +6552 | +3417 | 73 | -4999 | 8614 |
| move 2.0% | +113555 | +4368 | +2846 | 58 | -18685 | 12450 |
| move 2.5% | +162272 | +6241 | +8987 | 69 | -23238 | 13847 |
| move 3.0% | +162272 | +6241 | +8987 | 69 | -23238 | 13847 |
| PT 40% | +237146 | +9121 | +15042 | 69 | -12396 | 11540 |
| PT 60% | +205468 | +7903 | +8987 | 69 | -12396 | 11324 |
| PT 80% | +205468 | +7903 | +8987 | 69 | -12396 | 11324 |
| move2.0+PT60 | +113555 | +4368 | +2846 | 58 | -18685 | 12450 |

## Read
- Which move-stop cuts the worst-trade tail WITHOUT killing the mean (monotonic > peak)?
- PT raises win% but caps the big theta runs. 30d SIGNAL, overlapping entries. Best -> G2 wings + G3 entry.