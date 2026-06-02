# Stop-DESIGN comparison (ATM straddle, recorded NIFTY chain, lots=2)

Consistent mgmt: stop triggers a FULL strangle exit; else exit 14:45. **28 days, NO crash in sample -> worst-day UNDERSTATES the tail.** The structural point: underlying/max-loss stops have a BOUNDED loss by design; no-stop is unbounded.

| Design | All-days net ₹ | All-days worst-day ₹ | 1-DTE-only net ₹ |
|---|---|---|---|
| prem 1.2x | 1,136 | -2,422 | 8,039 |
| prem 1.3x | -3,589 | -3,020 | 13,908 |
| prem 1.5x | -13,794 | -6,140 | 12,244 |
| no stop | 7,058 | -20,284 | 9,423 |
| undl 0.4% | 4,757 | -4,755 | 15,988 |
| undl 0.6% | -15,088 | -7,492 | 7,018 |
| undl 0.8% | -5,923 | -13,036 | 9,248 |
| undl 1.0% | -10,798 | -14,785 | 2,540 |
| maxloss 2k | -24,929 | -4,203 | 7,018 |
| maxloss 3k | -29,167 | -4,755 | 3,879 |
| maxloss 5k | -248 | -6,972 | 5,237 |

- Premium stops (1.2-1.5x) whipsaw on premium noise.
- Underlying-move / max-loss stops trigger on REAL adverse moves, not premium spikes -> avoid whipsaw AND cap loss (structurally bounded). no-stop wins in-sample only because no trend/crash day occurred.
- Read the GRADIENT + the worst-day column together: want decent net AND a bounded worst-day.