# research/54 Stage 3 — defined-risk structures vs naked straddle (real NIFTY chain)

Short ATM straddle +/- long OTM wings. Enter 09:20, exit 14:45, net ₹80/leg. **28-29 days => SIGNAL.** Wings cap the tail with a KNOWN max loss, no stop needed.

## ALL-DTE (n / total ₹ / ₹-day / worst-day / worst3-avg)
| structure | n | total ₹ | ₹/day | worst ₹ | worst3 ₹ |
|---|---|---|---|---|---|
| naked_nostop | 29 | 6735 | 232 | -20284 | -10872 |
| naked_move0.4 | 29 | 4434 | 153 | -4755 | -3811 |
| fly_300 | 29 | -10443 | -360 | -19697 | -9461 |
| fly_400 | 29 | -8279 | -285 | -20789 | -10066 |
| fly_500 | 29 | -6387 | -220 | -20665 | -10181 |
| fly_400_move0.4 | 29 | -1837 | -63 | -3284 | -2400 |

## 1-DTE ONLY (the edge — Mondays, n~7)
| structure | n | total ₹ | ₹/day | worst ₹ | worst3 ₹ |
|---|---|---|---|---|---|
| naked_nostop | 13 | 9165 | 705 | -20284 | -9971 |
| naked_move0.4 | 13 | 18356 | 1412 | -3260 | -1850 |
| fly_300 | 13 | -4901 | -377 | -19697 | -9461 |
| fly_400 | 13 | -1918 | -148 | -20789 | -10066 |
| fly_500 | 13 | 566 | 44 | -20665 | -10181 |
| fly_400_move0.4 | 13 | 7618 | 586 | -3284 | -2108 |

## Read
- Compare the naked-straddle+move-stop (current best) vs iron-flies on **worst-day / worst3** (tail) AND ₹/day.
- A fly that keeps most of the 1-DTE ₹/day while cutting the worst-day = a better LIVE system (bounded, no whipsaw).
- 28-29d SIGNAL, single regime; wing liquidity/slippage not modelled. Confirm as recorder grows.