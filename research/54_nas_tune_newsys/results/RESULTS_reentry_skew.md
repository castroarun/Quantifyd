# research/54 Stage 5 — intraday re-entry + directional skew (real NIFTY chain)

Base: 1-DTE focus, ~100pt-OTM (2-strike) strangle, 09:20 entry, ±0.4% move stop, exit 14:45, net ₹80/leg/entry. **29d SIGNAL.**

## 0/1-DTE (the edge)
| variant | n | total ₹ | ₹/day | worst ₹ |
|---|---|---|---|---|
| base (no re-entry) | 13 | 20410 | 1570 | -2695 |
| re-entry x1 | 13 | 12015 | 924 | -4272 |
| re-entry x3 | 13 | 15129 | 1164 | -3571 |
| directional skew | 13 | 20410 | 1570 | -2695 |
| skew + re-entry x3 | 13 | 15129 | 1164 | -3571 |

## all-DTE
| variant | n | total ₹ | ₹/day | worst ₹ |
|---|---|---|---|---|
| base (no re-entry) | 29 | 5858 | 202 | -4866 |
| re-entry x1 | 29 | -8503 | -293 | -6072 |
| re-entry x3 | 29 | -19922 | -687 | -11196 |
| directional skew | 29 | 7125 | 246 | -4866 |
| skew + re-entry x3 | 29 | -18654 | -643 | -11196 |

## Read
- Re-entry: if re-entry x1/x3 >> base on net WITHOUT a worse worst-day, re-selling recovers theta after a whipsaw-out.
  If it just adds losses/worse tail, the move-stop should stay one-and-done.
- Skew: leaning strikes with the morning move should help only if the open predicts the rest of the day.
- 29d SIGNAL, ~13 obs 0/1-DTE. Gradient only.