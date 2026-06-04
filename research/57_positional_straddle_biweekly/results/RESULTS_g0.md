# research/57 G0 — baseline bi-weekly short straddle + EOD wings (real NIFTY chain, 26 trades)

Short ATM straddle in 2nd-nearest weekly, carry to DTE<=1, EOD ±500pt wings overnight. 09:20 entry, 15:20 marks. Net Rs80/leg. **30d SIGNAL, overlapping daily entries (correlated).**

## Per-trade P&L
- **straddle-only:** n=26  total=+193170  mean=+7430  median=+8988  win%=65  worst=-12396
- **straddle + EOD wings:** n=26  total=+156362  mean=+6014  median=+8386  win%=73  worst=-14952
- wings total P&L: -36804 (cost of overnight protection across all trades)
- avg days held: 6.5  |  worst single-trade MTM: -23078  |  trades with a gap-down night: 15

## Read
- Does the naked short bi-weekly straddle decay net-positive (theta) before management?
- Do the EOD wings cost more than the gap protection they buy (compare straddle-only vs +wings worst & total)?
- 30d = SIGNAL; daily entries overlap (same expiry cycles) -> treat as directional, not validated.