# Stop designs stressed over REAL NIFTY intraday (2024-2026, 453 days)

Model: short ATM straddle (1-DTE structure), BS-priced, **ATM IV calibrated to the real 28-day chain = 16.9%**, driven by REAL NIFTY 5-min paths. Premium side is a model; the MOVE DISTRIBUTION and TAIL are real. This is the right lens for stop design (28 calm days could not show the tail).

| Design | n | Total ₹ | ₹/day | Win% | WORST day ₹ | Worst-5 avg ₹ |
|---|---|---|---|---|---|---|
| prem 1.2x | 453 | -663,291 | -1,464 | 49% | -12,620 | -10,502 |
| prem 1.3x | 453 | -813,330 | -1,795 | 52% | -13,717 | -12,951 |
| prem 1.5x | 453 | -942,216 | -2,080 | 55% | -22,241 | -19,175 |
| no stop | 453 | -937,440 | -2,069 | 57% | -58,771 | -41,707 |
| undl 0.4% | 453 | -413,643 | -913 | 29% | -7,906 | -6,024 |
| undl 0.6% | 453 | -627,525 | -1,385 | 45% | -9,204 | -8,587 |
| undl 0.8% | 453 | -792,230 | -1,749 | 52% | -13,717 | -13,084 |
| undl 1.0% | 453 | -934,464 | -2,063 | 54% | -19,160 | -17,500 |
| maxloss 2k | 453 | -423,137 | -934 | 38% | -7,906 | -6,140 |
| maxloss 3k | 453 | -528,367 | -1,166 | 43% | -7,906 | -7,281 |
| maxloss 5k | 453 | -697,559 | -1,540 | 49% | -12,620 | -10,502 |

## Worst days by design (the tail no-stop hides)
- **prem 1.3x**: 2025-04-08 ₹-13,717, 2026-03-09 ₹-13,353, 2025-01-21 ₹-13,268, 2024-07-23 ₹-12,620, 2024-07-12 ₹-11,797
- **no stop**: 2024-06-05 ₹-58,771, 2025-04-17 ₹-40,040, 2024-06-07 ₹-38,152, 2024-11-22 ₹-38,064, 2026-03-11 ₹-33,510
- **undl 0.6%**: 2024-04-04 ₹-9,204, 2025-05-15 ₹-8,695, 2025-05-21 ₹-8,583, 2024-03-01 ₹-8,331, 2024-10-22 ₹-8,123
- **maxloss 3k**: 2025-04-07 ₹-7,906, 2024-09-12 ₹-7,339, 2026-02-27 ₹-7,239, 2024-10-07 ₹-7,063, 2024-11-08 ₹-6,861

**Read:** no-stop's WORST-day / worst-5 is far uglier than the bounded stops -> the 28-day +profit was survivorship (no crash in window). Underlying-move and max-loss stops trigger on REAL adverse moves (not premium noise) so they avoid the 1.3x whipsaw AND cap the tail. Premium is BS-modelled (caveat), but the comparison/ranking across stops over the real move distribution is the signal.