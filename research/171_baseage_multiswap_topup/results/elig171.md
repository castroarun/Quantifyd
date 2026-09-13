# research/171 -- the eligibility histogram

On an evening when at least one qualifying Base Age signal was REFUSED, how many open positions were simultaneously more than 10% below their average buy price? Medians across 30 seeds (7001-7030), whole window 2005-01-03 -> 2026-09-11 (21.7 years). This is a RECORDING inside the engine: it decides nothing and it draws no random number.

| book | refused-signal evenings | >=1 eligible | >=2 eligible | >=3 eligible | most ever eligible | mean eligible per evening |
|---|---:|---:|---:|---:|---:|---:|
| CONTROL measure-only (fires nothing) | 1396 | 418 (30.0%) | 92 (6.6%) | 16 (1.1%) | 4 | 0.38 |
| OA-ROT-1 - k=1, entrant rs252 (staged live) | 1412 | 97 (6.9%) | 11 (0.8%) | 2 (0.1%) | 3 | 0.08 |
| k=2 - sell the two weakest | 1408 | 95 (6.7%) | 9 (0.6%) | 2 (0.1%) | 4 | 0.08 |
| k=all eligible | 1407 | 94 (6.7%) | 9 (0.6%) | 2 (0.1%) | 4 | 0.08 |

Per YEAR (divide by 21.7):

| book | refused-signal evenings / yr | >=2 eligible / yr | swaps actually fired / yr |
|---|---:|---:|---:|
| CONTROL measure-only (fires nothing) | 64.3 | 4.2 | 0.0 |
| OA-ROT-1 - k=1, entrant rs252 (staged live) | 65.1 | 0.5 | 4.5 |
| k=2 - sell the two weakest | 64.9 | 0.4 | 4.5 |
| k=all eligible | 64.8 | 0.4 | 4.5 |
