# research/57 G5 — crash-stop level + intraday move distribution

## Per-trade MAX intraday move-from-entry (n=26): median 2.12%, biggest 4.32%
| threshold | trades that ever reached it |
|---|---|
| 1.5% | 22 of 26 |
| 2.0% | 14 of 26 |
| 2.5% | 12 of 26 |
| 3.0% | 6 of 26 |

## Sequential book by crash-stop level
| crash stop | closes | final P&L | book max-DD |
|---|---|---|---|
| 1.75% | 8 | +36475 | -6505 |
| 2.00% | 6 | +74694 | -3683 |
| 2.50% | 6 | +74694 | -3683 |
| 3.00% | 6 | +74694 | -3683 |
| no crash (EOD only) | 6 | +74694 | -3683 |

## Read
- If few/no trades reach 2%%, a 2%% crash stop rarely fires on this calm sample (same as 3%%).
- The crash stop is INSURANCE for a fast intraday move the EOD-1.5%% check would miss until 15:20 - its value shows up in a violent month, not this one.
- 30d SIGNAL.