# research/57 G2b — overnight GAP RISK on the recipe (move1.5+PT40, 26 trades)

User Q: a position <1.5%% at EOD carries NAKED overnight; a gap-open breaches the stop at a worse price. Tested.

## Overnight gaps (15:20->09:20, n=29)
- mean|gap| 0.45%, median 0.39%, **worst -1.25%**; nights >0.5%: 11, >1%: 4
- biggest: 2026-04-27 -1.25%, 2026-05-11 -1.06%, 2026-05-25 +1.02%, 2026-05-18 -1.01%, 2026-06-03 -0.90%

## Recipe under each stop model
| stop model | mean | median | win% | worst | std |
|---|---|---|---|---|---|
| 15:20-only (the G2 number) | +7839|+6151|73|-4999|9042 |
| GAP-AWARE (09:20+15:20) | +8072|+6239|81|-4917|8307 |
| GAP-AWARE + EOD wings | +5019|+4259|69|-5982|6283 |

## Read
- If GAP-AWARE worst >> 15:20-only worst, overnight gaps DO breach the stop -> real risk, the 15:20 number understated it.
- If wings RECOVER that gap loss (worst back up), the 'wings redundant' verdict was an artifact -> wings DO earn their keep for the overnight gap the stop can't prevent.
- 30d SIGNAL.