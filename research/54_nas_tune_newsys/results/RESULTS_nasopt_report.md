# NAS-OPT — backtest performance report (recorded NIFTY chain)

System: 0/1-DTE · ~100pt-OTM strangle · 09:20 · ±0.4% underlying-move stop (one-and-done) · exit 14:45. Net ₹80/leg. 2026-04-20 → 2026-06-02.

## KPIs
| metric | value |
|---|---|
| trades | 13 |
| total | 20409 |
| pertrade | 1569 |
| win | 69.2 |
| avgwin | 2733 |
| avgloss | -1049 |
| worst | -2695 |
| best | 6334 |
| maxdd | -2695 |
| sharpe | 10.35 |

## Calm vs wide opening-range day
| | n | ₹/trade |
|---|---|---|
| calm (OR<median) | 6 | 2132 |
| wide | 7 | 1088 |

Artifacts: `nasopt_trades.csv`, `nasopt_perf.png`. 29-day window = SIGNAL; paper-forward as recorder grows.