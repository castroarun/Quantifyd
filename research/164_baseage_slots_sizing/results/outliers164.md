## OA — Base Age · outlier dependence: delete the ten best trades

| cell | total book profit | top-10 trades' share of it | CAGR (all events) | CAGR (top-10 events DELETED, 30 seeds) | cost |
|---|---|---|---|---|---|
| A_s16_eq | ₹67,725,152 | 35.7% | 20.94% | 18.62% | **-2.31 pp** |
| A_s11_eq | ₹82,697,880 | 48.5% | 22.03% | 19.99% | **-2.04 pp** |
| A_s10_eq | ₹88,980,656 | 53.4% | 22.46% | 20.29% | **-2.18 pp** |
| D_tv_s16 | ₹77,999,144 | 35.7% | 21.72% | 19.88% | **-1.84 pp** |

*The ten best trades are identified on the median-seed path, then those ten EVENTS are removed from the event list and the whole book is re-run on all 30 seeds — so the slots they occupied are freed for whatever else qualified that day. That is a fair deletion, not a bookkeeping subtraction.*
