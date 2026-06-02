# Part (b) — Diversification / uncorrelated subset

Daily P&L of the 8 replayed systems (all-DTE, lots=2). **28-day sample -> correlations and Sharpe are NOISY; directional only, not validation.**

## Per-system (daily P&L stats)

| System | Total ₹ | ₹/day | Sharpe(ann) | MaxDD ₹ |
|---|---|---|---|---|
| 916 ATM4 | 20,986 | 656 | 4.26 | -6,434 |
| 916 ATM2 | 7,240 | 226 | 1.87 | -7,278 |
| Squeeze ATM | 1,736 | 54 | 1.05 | -3,097 |
| Squeeze ATM4 | 274 | 9 | 0.30 | -2,231 |
| Squeeze ATM2 | -443 | -14 | -0.76 | -2,135 |
| 916 OTM | -1,448 | -45 | -0.85 | -6,069 |
| 916 ATM | -13,983 | -437 | -2.73 | -25,471 |
| Squeeze OTM | -3,127 | -98 | -4.01 | -3,272 |

## Greedy max-Sharpe uncorrelated subset
**Selected: ['916 ATM4', 'Squeeze ATM']**
- Subset book: total ₹22,722, Sharpe(ann) 4.36, MaxDD ₹-5,997
- All-8 book: total ₹11,236, Sharpe(ann) 0.90, MaxDD ₹-24,608

## Correlation matrix
| | 916 ATM | 916 ATM2 | 916 ATM4 | 916 OTM | Squeeze ATM | Squeeze ATM2 | Squeeze ATM4 | Squeeze OTM |
|---|---|---|---|---|---|---|---|---|
| 916 ATM | 1.00 | 0.72 | 0.56 | 0.42 | 0.07 | -0.15 | -0.03 | -0.32 |
| 916 ATM2 | 0.72 | 1.00 | 0.39 | 0.36 | -0.05 | -0.27 | -0.08 | -0.24 |
| 916 ATM4 | 0.56 | 0.39 | 1.00 | 0.39 | 0.01 | -0.16 | -0.12 | -0.23 |
| 916 OTM | 0.42 | 0.36 | 0.39 | 1.00 | 0.26 | 0.10 | 0.20 | -0.47 |
| Squeeze ATM | 0.07 | -0.05 | 0.01 | 0.26 | 1.00 | 0.70 | 0.81 | -0.65 |
| Squeeze ATM2 | -0.15 | -0.27 | -0.16 | 0.10 | 0.70 | 1.00 | 0.85 | -0.21 |
| Squeeze ATM4 | -0.03 | -0.08 | -0.12 | 0.20 | 0.81 | 0.85 | 1.00 | -0.52 |
| Squeeze OTM | -0.32 | -0.24 | -0.23 | -0.47 | -0.65 | -0.21 | -0.52 | 1.00 |

- Non-trading days counted as ₹0 (portfolio view). 28d, ~16-28 obs -> noisy.