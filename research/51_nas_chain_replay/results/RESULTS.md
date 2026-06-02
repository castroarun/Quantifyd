# NAS Systems — REPLAY backtest on recorded NIFTY chain

2026-04-20 → 2026-06-02 · 28 days · lots=2 · **combined net ₹-54,123**

**VERDICT: SIGNAL/AUDIT (28-day single regime, not validation).** Rules replayed on real recorded premiums.

| System | Legs | Days | Net | PerDay | DayWin | MaxDD | Best | Worst |
|---|---|---|---|---|---|---|---|---|
| Squeeze OTM | 35 | 16 | -3249 | -203 | 44 | -3394 | 379 | -1923 |
| Squeeze ATM | 32 | 16 | 1736 | 109 | 44 | -3097 | 3662 | -1174 |
| Squeeze ATM2 | 32 | 16 | -1728 | -108 | 44 | -3097 | 1205 | -1174 |
| Squeeze ATM4 | 34 | 16 | -1011 | -63 | 38 | -3516 | 2139 | -1174 |
| 916 OTM | 157 | 28 | -8205 | -293 | 43 | -11406 | 1526 | -2726 |
| 916 ATM | 56 | 28 | -13983 | -499 | 32 | -25471 | 6646 | -3956 |
| 916 ATM2 | 196 | 28 | -23870 | -852 | 36 | -41318 | 6646 | -12705 |
| 916 ATM4 | 81 | 28 | -3815 | -136 | 43 | -24623 | 7469 | -7125 |

- 9:16 entry exact; squeeze entry reconstructed from per-min spot (approx).
- 1-min cadence (intra-min SL/ST spikes missed); LTP pricing (no slippage) => mildly optimistic.
- Validate against actual trades (research/50) for faithfulness.