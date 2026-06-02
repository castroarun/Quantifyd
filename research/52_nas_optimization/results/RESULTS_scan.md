# Part (a) — one-factor sensitivity (short straddle on recorded NIFTY chain)

Base: enter 09:20, ATM CE+PE, lots=2, per-leg SL 1.3x, SL->naked ST(7,2), time-exit 14:45. Vary ONE axis at a time. **28 days => SIGNAL, read the gradient (monotonic > peak), not the peak.**

## DTE-at-entry (base ATM/1.3/ST)
| variant | net ₹ |
|---|---|
| 0DTE | -4,340 |
| 1DTE | 9,410 |
| 4DTE | -4,317 |
| 5DTE | -9,861 |
| 6DTE | -4,875 |

## Strike offset (OTM strikes from ATM)
| variant | net ₹ |
|---|---|
| ATM | -13,983 |
| 1-OTM | -9,446 |
| 2-OTM | -7,620 |
| 3-OTM | -2,861 |

## SL multiple (x entry premium)
| variant | net ₹ |
|---|---|
| 1.2x | -2,276 |
| 1.3x | -13,983 |
| 1.5x | -18,494 |
| 2.0x | -8,809 |
| none | 7,058 |

## Exit mode
| variant | net ₹ |
|---|---|
| time-1445 | -46,177 |
| EOD-1515 | -46,177 |
| SL->ST(7,2) | -13,983 |

- DTE axis confirms where the edge concentrates. Monotonic responses (e.g. P&L rising as DTE->1) are more trustworthy than isolated peaks. Multiple-testing: few cells/axis, each is one 28-day path.