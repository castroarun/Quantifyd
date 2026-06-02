# 28-day REAL ATM straddle P&L by gap / CPR (1.3x stop, exit 14:45)

**n per bucket is tiny (28-day window) -> directional only, NOT proof.** Prior H/L/C & gap from the chain's own underlying_spot.

## By gap bucket
| gap | net ₹ | n | avg ₹/day |
|---|---|---|---|
| gap dn | 6,338 | 8 | 792 |
| flat | -8,049 | 12 | -671 |
| gap up | -2,244 | 7 | -321 |

## By CPR width
| CPR | net ₹ | n | avg ₹/day |
|---|---|---|---|
| narrow | 13,217 | 14 | 944 |
| wide | -17,173 | 13 | -1,321 |

## gap x DTE (net ₹)
dte          0       1       4       5       6
gap_b                                         
gap dn  -593.0  8939.0  2303.0 -3020.0 -1291.0
flat   -2332.0   158.0 -4449.0 -3147.0  1721.0
gap up -1980.0  4444.0     0.0  -385.0 -4324.0

- Consistent with the years layer if gap-down days are the worst here too. CPR split here is tiny-n.