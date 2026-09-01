# research/141 — the live re-center arm (recorded books)

### SENSEX ATM2 (live re-center arm, kept by c95f10a)

`sensex_atm2_trading.db`, lot 20, 0.4% move-stop + re-center; matrix live on DTE0/DTE1. Days with a recorded strangle: 25, strangles: 31.

| day | lots | cycle 1 reason | cycle-1 net ₹ | re-centers | re-center net ₹ | re-center ₹/lot | re-center exits |
|---|---:|---|---:|---:|---:|---:|---|
| 2026-08-04 | 2 | MOVE_STOP | -3474 | 1 | +364 | +182 | eod_squareoff |
| 2026-08-12 | 3 | MOVE_STOP | -232 | 1 | +2114 | +705 | eod_squareoff |
| 2026-08-19 | 2 | MOVE_STOP | -1144 | 1 | +1566 | +783 | eod_squareoff |
| 2026-08-24 | 2 | MOVE_STOP | -4012 | 1 | +562 | +281 | eod_squareoff |
| 2026-08-25 | 2 | MOVE_STOP | -3086 | 0 | — | — | (no re-center) |
| 2026-08-26 | 2 | MOVE_STOP | -2522 | 1 | +534 | +267 | eod_squareoff |
| 2026-08-27 | 2 | MOVE_STOP | -2538 | 1 | +2052 | +1026 | eod_squareoff |
| 2026-09-01 | 2 | MOVE_STOP | -3030 | 0 | — | — | (no re-center) |

**Live re-center total: 6 re-centers over 8 stop-days (2 stop-days ended with no re-center).**

- as recorded (book's own ₹160/strangle brokerage): **+7192** total, **+3244 ₹/lot** across all re-centers (mean **+541 ₹/lot** each)
- re-priced with the research/122 measured cost model (charges + slippage by exit type): extra-cycle cost **₹451** total
- of the 6 re-centered straddles, **0 stopped out again** (0%) — the re-center's own risk of repeating the loss

### NIFTY 916-ATM2 (re-center arm, BEFORE research/96)

`nas_916_atm2_trading.db`, lot 65, 0.4% move-stop + re-center, the pre-July behaviour. Days with a recorded strangle: 50, strangles: 127.

| day | lots | cycle 1 reason | cycle-1 net ₹ | re-centers | re-center net ₹ | re-center ₹/lot | re-center exits |
|---|---:|---|---:|---:|---:|---:|---|
| 2026-06-24 | 1 | MOVE_STOP | -488 | 0 | — | — | (no re-center) |
| 2026-06-25 | 10 | MOVE_STOP | -4775 | 2 | -8608 | -861 | MOVE_STOP, eod_squareoff |
| 2026-06-29 | 10 | MOVE_STOP | -6725 | 1 | +7575 | +758 | eod_squareoff |
| 2026-06-30 | 10 | MOVE_STOP | -36300 | 0 | — | — | (no re-center) |
| 2026-07-01 | 10 | MOVE_STOP | -2045 | 1 | +19015 | +1902 | eod_squareoff |
| 2026-07-02 | 10 | MOVE_STOP | +18072 | 1 | +2830 | +283 | eod_squareoff |
| 2026-07-06 | 10 | MOVE_STOP | -8968 | 1 | +15505 | +1550 | eod_squareoff |
| 2026-07-08 | 2 | MOVE_STOP | +416 | 3 | -16912 | -8456 | MOVE_STOP, MOVE_STOP, EOD_SQUAREOFF |
| 2026-07-09 | 2 | MOVE_STOP | +132 | 1 | +347 | +174 | MOVE_STOP |
| 2026-07-13 | 2 | MOVE_STOP | -2910 | 2 | -883 | -441 | MOVE_STOP, eod_squareoff |
| 2026-07-15 | 2 | MOVE_STOP | -2350 | 2 | -346 | -173 | MOVE_STOP, eod_squareoff |
| 2026-07-17 | 2 | MOVE_STOP | -4170 | 1 | +2573 | +1286 | eod_squareoff |
| 2026-07-22 | 2 | MOVE_STOP | +880 | 1 | +1439 | +720 | eod_squareoff |
| 2026-07-23 | 2 | MOVE_STOP | +1738 | 2 | -1848 | -924 | MOVE_STOP, eod_squareoff |
| 2026-07-24 | 2 | MOVE_STOP | -2864 | 1 | +3324 | +1662 | eod_squareoff |

**Live re-center total: 19 re-centers over 15 stop-days (2 stop-days ended with no re-center).**

- as recorded (book's own ₹160/strangle brokerage): **+24012** total, **-2521 ₹/lot** across all re-centers (mean **-133 ₹/lot** each)
- re-priced with the research/122 measured cost model (charges + slippage by exit type): extra-cycle cost **₹23478** total
- of the 19 re-centered straddles, **7 stopped out again** (37%) — the re-center's own risk of repeating the loss

### NIFTY squeeze-ATM2 (re-center arm, BEFORE research/96)

`nas_atm2_trading.db`, lot 65, 0.4% move-stop + re-center, the pre-July behaviour. Days with a recorded strangle: 46, strangles: 103.

| day | lots | cycle 1 reason | cycle-1 net ₹ | re-centers | re-center net ₹ | re-center ₹/lot | re-center exits |
|---|---:|---|---:|---:|---:|---:|---|
| 2026-06-24 | 1 | MOVE_STOP | -914 | 1 | -914 | -914 | MOVE_STOP |
| 2026-06-25 | 10 | MOVE_STOP | -10462 | 1 | +1238 | +124 | eod_squareoff |
| 2026-06-29 | 10 | MOVE_STOP | -22130 | 2 | +14468 | +1447 | eod_squareoff, eod_squareoff |
| 2026-06-30 | 10 | MOVE_STOP | -17255 | 1 | -10950 | -1095 | MOVE_STOP |
| 2026-07-08 | 10 | MOVE_STOP | -4385 | 2 | -51150 | -5115 | MOVE_STOP, MOVE_STOP |
| 2026-07-09 | 10 | MOVE_STOP | -15728 | 4 | -21375 | -2138 | MOVE_STOP, MOVE_STOP, MOVE_STOP, MOVE_STOP |
| 2026-07-13 | 10 | MOVE_STOP | -28500 | 1 | -28500 | -2850 | MOVE_STOP |
| 2026-07-14 | 2 | MOVE_STOP | +620 | 1 | -511 | -256 | eod_squareoff |
| 2026-07-15 | 2 | MOVE_STOP | -1830 | 1 | +1238 | +619 | eod_squareoff |
| 2026-07-17 | 2 | MOVE_STOP | -3508 | 1 | -173 | -86 | eod_squareoff |
| 2026-07-23 | 2 | MOVE_STOP | -3215 | 1 | +1082 | +541 | eod_squareoff |
| 2026-07-24 | 2 | MOVE_STOP | -43 | 1 | +1348 | +674 | eod_squareoff |

**Live re-center total: 17 re-centers over 12 stop-days (0 stop-days ended with no re-center).**

- as recorded (book's own ₹160/strangle brokerage): **-94201** total, **-9050 ₹/lot** across all re-centers (mean **-532 ₹/lot** each)
- re-priced with the research/122 measured cost model (charges + slippage by exit type): extra-cycle cost **₹74527** total
- of the 17 re-centered straddles, **9 stopped out again** (53%) — the re-center's own risk of repeating the loss
