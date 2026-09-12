## OA — Base Age · outlier dependence: delete the ten best trades

| cell | total book profit | top-10 trades' share of it | CAGR (all events) | CAGR (top-10 events DELETED, 30 seeds) | cost |
|---|---|---|---|---|---|
| BASE_rand | ₹67,304,544 | 40.3% | 20.98% | 19.18% | **-1.80 pp** |
| X_entrs_unre_m010 | ₹93,275,768 | 45.5% | 22.73% | 20.48% | **-2.25 pp** |
| X_entrs_unre_m007 | ₹78,824,976 | 35.8% | 21.82% | 20.43% | **-1.39 pp** |
| A_unre_m010 | ₹91,805,088 | 48.9% | 22.69% | 20.77% | **-1.92 pp** |
| X_trimM_k250 | ₹70,445,096 | 35.8% | 21.13% | 19.39% | **-1.74 pp** |
| C_unre010_k150d_f000 | ₹94,894,664 | 30.0% | 22.76% | 21.33% | **-1.43 pp** |
| B2_trimD_k150 | ₹76,355,104 | 28.8% | 21.59% | 20.50% | **-1.10 pp** |
| C_none000_k150d_f025 | ₹83,382,320 | 29.3% | 22.05% | 20.91% | **-1.14 pp** |

*The ten best trades are identified on the median-seed path, then those ten EVENTS are removed from the event list and the whole book is re-run on all 30 seeds — so the slots they occupied are freed for whatever else qualified that day. That is a fair deletion, not a bookkeeping subtraction. Trims count as realisations here, which is why a trimming cell shows a lower concentration.*
