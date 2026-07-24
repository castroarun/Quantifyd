# RSI 70/40 Regime — Filter-Overlay Marginal-Add Study (RELIANCE + 8-name basket)

STATUS: DONE (NO EDGE — filters do not rescue; best basket net CAGR 2.74% vs index 10.95%)

## The Ask
**What you asked:** Do the trend/momentum filters materially rescue the single-name
RSI-regime system, or do they just reduce exposure?

**What we're testing:** On the base RSI 70/40 (len 14, 15bps) regime system, add each
filter as a marginal overlay and measure whether any Calmar/return improvement comes from
HIGHER return (skill) or merely LOWER exposure (parking in cash). Targets: RELIANCE single
name + equal-weight basket-average NAV of RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN,
ITC, LT. Benchmark NIFTYBEES. Period 2015-01-01 onward.

## The Base
- Entry: RSI(14) close >= 70 (buy at close if flat)
- Exit: RSI(14) close < 40 (sell at close if long); flat = cash
- Cost: 15 bps round-trip. Net includes cost; gross re-runs cost 0.
- Basket = simple mean of each name's own single-name NAV curve, renormalised to 1 on
  common start date (so a filter's effect shows beyond one lucky name).

## Plan — 13 filter configs x 2 targets = 26 cells
| # | label | overlay |
|---|---|---|
| 1 | base | none |
| 2 | ma200 | enter only if close>SMA200 |
| 3 | ma50 | enter only if close>SMA50 |
| 4 | adx20 | ADX(14)>=20 |
| 5 | adx25 | ADX(14)>=25 |
| 6 | wrsi50 | weekly RSI>=50 (causal, ffill last completed week) |
| 7 | wrsi60 | weekly RSI>=60 |
| 8 | st_regime | Supertrend(10,3) bullish gate + ST-flip exit |
| 9 | donchian20 | add Donchian-20 low trailing exit |
| 10 | ma200_adx20 | combo |
| 11 | ma200_wrsi50 | combo |
| 12 | ma200_donchian20 | combo |
| 13 | ma50_adx20 | combo |

Verdict tag per row: ADDS RETURN (dCalmar>0.05 & dNetCAGR>0.3) / JUST CUTS EXPOSURE/DD
(dCalmar>0.05 & dExposure<-2) / WORSE / neutral.

## Status
| Date/time | Event | Notes |
|---|---|---|
| 2026-07-07 | STATUS written, runner built | 26 cells planned |

## Crash Recovery
- Runner: `research/72_rsi_regime_7040/scripts/run_phaseB_filters.py` (resumable, skips done labels).
- Resume: `cd /home/arun/quantifyd && venv/bin/python research/72_rsi_regime_7040/scripts/run_phaseB_filters.py`
- Output: `research/72_rsi_regime_7040/results/phaseB_filters.csv` (append per row, header once).
- Check progress: `wc -l` the CSV. Log at `/tmp/phaseB.log`.
- Do NOT edit the CSV mid-run.

## Files
| File | Purpose | Committable? |
|---|---|---|
| scripts/run_phaseB_filters.py | runner | yes |
| results/phaseB_filters.csv | per-cell output | yes (small) |
| results/RESULTS_phaseB_filters.md | final findings | yes |
