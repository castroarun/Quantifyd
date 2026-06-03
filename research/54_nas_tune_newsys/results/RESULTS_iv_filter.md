# research/54 Stage 1 — IV-level filter (real NIFTY chain, 23 days, ±0.4% move stop)

Short ATM straddle, enter 09:20, ±0.4%% underlying-move stop, exit 14:45. Net ₹80/leg. **28-29 days => SIGNAL.** IV = entry ATM (avg CE+PE) implied vol from the chain.

## Headline signal
- **corr(entry ATM-IV, day P&L) = 0.41** (all 23 days); 1-DTE-only corr = -0.14 (n=7)
- Positive corr => richer entry vol -> better short-straddle day (the 'sell rich vol' edge).

## P&L by entry-IV tercile (all-DTE; flag DTE confound + small n)
| IV bucket | n | total ₹ | ₹/day |
|---|---|---|---|
| low | 8 | -3242 | -405 |
| mid | 7 | -8315 | -1187 |
| high | 8 | 14092 | 1761 |

## Real-chain books (n / total ₹ / ₹-day / worst-day) — tiny n, SIGNAL
| book | n | total ₹ | ₹/day | worst ₹ |
|---|---|---|---|---|
| all days | 23 | 2534 | 110 | -4755 |
| 1-DTE only | 7 | 15988 | 2284 | -36 |
| high-IV tercile | 8 | 14093 | 1762 | -1895 |
| tight-open only | 11 | -1487 | -135 | -4755 |
| **1-DTE + IV>=med + tight-open** | 3 | 6429 | 2143 | 548 |

- DTE dominates short-vol P&L (finding #1), so the all-DTE tercile mixes DTEs — read corr + the stacked book.
- If high-IV total > low-IV AND corr>0, the IV filter adds to the stack. Confirm as the recorder grows.