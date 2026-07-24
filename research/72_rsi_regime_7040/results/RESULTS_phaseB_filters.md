# Phase B — Filter-Overlay Marginal-Add on RSI 70/40 Regime

## VERDICT: **NO EDGE — filters do NOT rescue the RSI-regime system; they mostly cut exposure/DD, and the best-case return bump still leaves the system at ~1/4 of the index.**

Period 2015-01-01 .. 2026-07-07. Base = RSI(14) entry 70 / exit 40, 15 bps RT, long-only, flat=cash.
Benchmark **NIFTYBEES B&H: CAGR 10.95%, MaxDD -36.34%, Calmar 0.30**.
Basket B&H (equal-wt 8 names): CAGR 10.98%, MaxDD -39.79%, Calmar 0.28.

The base system is already broken: RELIANCE net CAGR 4.19%, basket net CAGR 1.76% — a small
fraction of the ~11% index, because it sits in cash ~63% of the time (exposure ~37%) and only
holds during late-momentum bursts. The question was whether a trend/momentum entry gate rescues it.

## Basket — top filters by net Calmar (the honest ranking)
| filter | net CAGR | gross CAGR | MaxDD | Calmar | Sharpe | exp% | dCAGR | dDD | dExp | decomposition |
|---|---|---|---|---|---|---|---|---|---|---|
| ma200_wrsi50 | 2.74 | 3.71 | -10.68 | 0.26 | 0.27 | 33.95 | +0.98 | +3.71 | -3.43 | **adds return** (CAGR up AND DD down, exposure barely cut) |
| ma200 | 2.73 | 3.65 | -10.76 | 0.25 | 0.27 | 34.23 | +0.97 | +3.63 | -3.15 | **adds return** |
| wrsi60 | 2.33 | 3.03 | -9.86 | 0.24 | 0.24 | 28.68 | +0.57 | +4.53 | -8.70 | mostly exposure/DD cut (-8.7pp exp) |
| ma200_adx20 | 2.54 | 3.41 | -11.49 | 0.22 | 0.25 | 32.29 | +0.78 | +2.90 | -5.09 | adds some return, part exposure cut |
| wrsi50 | 2.16 | 2.86 | -10.77 | 0.20 | 0.22 | 35.01 | +0.40 | +3.62 | -2.37 | adds return (small) |
| ma50 | 2.20 | 3.02 | -10.80 | 0.20 | 0.22 | 36.25 | +0.44 | +3.59 | -1.13 | adds return (small) |
| **base** | 1.76 | 2.42 | -14.39 | 0.12 | 0.18 | 37.38 | — | — | — | reference |
| adx25 | 1.84 | 2.42 | -11.47 | 0.16 | 0.19 | 30.14 | +0.08 | +2.92 | -7.24 | just cuts exposure/DD |
| adx20 | 1.64 | 2.19 | -13.52 | 0.12 | 0.16 | 34.56 | -0.12 | +0.87 | -2.82 | neutral |
| st_regime | 1.47 | 1.90 | -13.52 | 0.11 | 0.16 | 30.77 | -0.29 | +0.87 | -6.61 | neutral/worse |
| ma50_adx20 | 2.06 | 2.75 | -11.52 | 0.18 | 0.21 | 34.11 | +0.30 | +2.87 | -3.27 | just cuts exposure/DD |
| ma200_donchian20 | 1.55 | 2.02 | -12.81 | 0.12 | — | 27.60 | -0.21 | +1.58 | -9.78 | neutral (exposure gutted) |
| donchian20 | 0.75 | 1.12 | -14.31 | 0.05 | — | 29.85 | -1.01 | +0.08 | -7.53 | **WORSE** (alt exit whipsaws) |

## RELIANCE single-name (same story, noisier)
Base net CAGR 4.19%, DD -23.3%, Calmar 0.18, exp 36.7%. Best was **st_regime**: net 6.26%,
Calmar 0.26, exp 29.5% (tagged "adds return", +2.07 CAGR) — but this is one lucky name and does
NOT generalise (st_regime is neutral/worse on the basket). ma200_donchian20 net 5.35%/Cal 0.21.
Nothing approaches RELIANCE B&H or the index.

## Return-vs-exposure decomposition (the key analytical point)
- The only filters that genuinely **add return** are the **long-term trend gates (SMA-200, weekly-RSI)**:
  ma200 / ma200_wrsi50 lift basket net CAGR by ~+1.0pp AND cut MaxDD by ~+3.6pp while barely
  touching exposure (-3pp). That is a real, if tiny, quality improvement — the gate keeps you out
  of the worst momentum-fakeout entries.
- **ADX and Supertrend gates and the Donchian alt-exit do NOT add return.** ADX-25 / ma50_adx20 /
  wrsi60 improve Calmar almost entirely by **parking more capital in cash** (exposure -7 to -9pp) —
  the classic "Calmar looks better because I stopped playing" illusion. Donchian-20 as an alt exit
  is outright destructive (net CAGR 0.75%, whipsaw churn on 185 trades).

## Does any filter make the system beat NIFTYBEES (1.5x return + lower DD)?
**No.** Best basket net CAGR after any filter = **2.74%** vs index **10.95%** — the system delivers
~**0.25x** the index, not 1.5x. Best filtered Calmar = **0.26**, still **below** NIFTYBEES 0.30 and
basket B&H 0.28. The filters lower drawdown (good) but only because they hold you in cash most of the
time; the residual invested return is far too weak. There is no configuration, single-name or basket,
that clears the bar.

## Multiple-testing honesty
13 configs x 2 targets = 26 cells tested; combos were chosen post-hoc from singles that helped, so the
best rows are mildly cherry-picked. Even so, the "best" survivor (ma200_wrsi50, Calmar 0.26) fails to
beat the passive index Calmar (0.30) — i.e. the winner loses even before any multiple-testing haircut.
The RELIANCE st_regime "win" is a single-name artifact (contradicted on the basket), a textbook example
of why we required the basket-average target.

## Read & next levers
The RSI 70/40 regime is a **low-exposure, late-momentum** system whose problem is structural: it is out
of the market during the bulk of the compounding and only overlays cannot manufacture return that the
entry logic never captures. Trend gates tidy the risk profile but cannot rescue a ~2-4% CAGR core.
This line of inquiry is **CONCLUDED as NO EDGE**. If revisited, the lever is not a filter — it is the
regime definition itself (lower entry threshold / higher exposure, or pairing with a hold-through-cash
carry), which is a different system, not this one.
