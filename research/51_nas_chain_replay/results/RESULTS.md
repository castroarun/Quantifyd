# NAS Systems — REPLAY on recorded NIFTY chain (skip_weekdays respected)

2026-04-20 → 2026-06-02 · 28 traded days · lots=2 · **combined net ₹11,236**

OTM/ATM2/ATM4 skip Wed+Thu (near-expiry only); basic ATM trades all days.

**VERDICT: SIGNAL/AUDIT (28-day single regime, not validation).**

## Per-system

| System | Legs | Days | Net | PerDay | DayWin | MaxDD | Best | Worst |
|---|---|---|---|---|---|---|---|---|
| Squeeze OTM | 23 | 10 | -3127 | -313 | 40 | -3272 | 217 | -1923 |
| Squeeze ATM | 32 | 16 | 1736 | 109 | 44 | -3097 | 3662 | -1174 |
| Squeeze ATM2 | 20 | 10 | -443 | -44 | 40 | -2135 | 815 | -1031 |
| Squeeze ATM4 | 22 | 10 | 274 | 27 | 30 | -2231 | 2139 | -1031 |
| 916 OTM | 126 | 18 | -1448 | -80 | 56 | -6069 | 1526 | -2726 |
| 916 ATM | 56 | 28 | -13983 | -499 | 32 | -25471 | 6646 | -3956 |
| 916 ATM2 | 134 | 18 | 7240 | 402 | 44 | -7278 | 6646 | -3690 |
| 916 ATM4 | 50 | 18 | 20986 | 1166 | 56 | -6434 | 7469 | -4584 |

## Net ₹ by DTE (days-to-expiry at entry) — combined
| DTE | Net ₹ | Legs |
|---|---|---|
| 0 | -2,075 | 209 |
| 1 | 37,272 | 146 |
| 4 | -7,939 | 76 |
| 5 | -9,437 | 16 |
| 6 | -6,584 | 16 |

## Net ₹ by system × DTE
| System | 0DTE | 1DTE | 4DTE | 5DTE | 6DTE |
|---|---|---|---|---|---|
| Squeeze OTM | -2,504 | -596 | -27 | 0 | 0 |
| Squeeze ATM | 4,588 | -720 | -847 | 423 | -1,708 |
| Squeeze ATM2 | 1,123 | -720 | -847 | 0 | 0 |
| Squeeze ATM4 | 1,841 | -720 | -847 | 0 | 0 |
| 916 OTM | -1,760 | 273 | 39 | 0 | 0 |
| 916 ATM | -4,340 | 9,410 | -4,317 | -9,861 | -4,875 |
| 916 ATM2 | -4,778 | 11,533 | 485 | 0 | 0 |
| 916 ATM4 | 3,756 | 18,810 | -1,580 | 0 | 0 |

- 9:16 entry exact; squeeze entry reconstructed (approx); 1-min SL/ST; LTP (no slippage).
- skip_weekdays applied per config (OTM/ATM2/ATM4 skip Wed/Thu). Validate vs actuals (research/50).