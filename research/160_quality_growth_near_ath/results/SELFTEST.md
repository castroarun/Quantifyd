# research/160 engine — SELF-TEST RESULTS

Panel: `results/panel_2000.npz`. All cells: equal weight, idle cash 5% p.a., decisions on the close, fills at the next open unless stated. After-tax = 20% STCG / 12.5% LTCG, Indian FY netting.

Anchors used to sanity-check the baselines: r/75 momentum 31.9% net CAGR / -31.6% DD (2006-26) | NIFTYBEES 11.5% / -59.7% (2006-26) | True North ~20.7% after tax / -25.1% (12-offset median)


## 1. Interface smoke (r/158 strict mask, 2024-08 to 2026-09)

| label | CAGR gross | net | after-tax | MaxDD | trades | verdict |
|---|---|---|---|---|---|---|
| SMOKE_r158mask_k90_N15 | 12.64 | 9.58 | 8.68 | -29.69 | 133 | **PASS** — every column populated |

## 2. Look-ahead probe

| arm | fill | CAGR after-tax | MaxDD | trades |
|---|---|---|---|---|
| LA_next_open | next open | 8.68 | -29.69 | 133 |
| LA_same_close | signal-day close | 8.49 | -28.24 | 133 |
| LA_shift1 | next open, ALL prices shifted +1 day | 11.34 | -25.77 | 133 |

**Verdict: PASS** — the shifted-data arm does not reproduce the unshifted result, so the engine is not reading a bar it should not.

The engine is close-only by construction: no `high`/`low` array is read by the simulator, so a close trigger can never be filled at an earlier price in the same bar. `same_close` and `next_open` are the only two placeable conventions and both are labelled in every row.


## 3. Price-only baselines (2010-01 to 2026-09, tv >= Rs 2cr)

### 3a. Index buy-and-hold

| series | window | CAGR | MaxDD | Calmar | x |
|---|---|---|---|---|---|
| NIFTYBEES | 2010-01-04..2026-09-10 | 10.26 | -36.34 | 0.28 | 5.10 |
| NIFTY50 | 2011-01-03..2026-09-10 | 8.91 | -38.44 | 0.23 | 3.81 |
| NIFTY500 | 2011-01-03..2026-09-10 | 10.25 | -38.30 | 0.27 | 4.62 |
| NIFTYMIDCAP150 | 2011-01-03..2026-09-10 | 14.48 | -44.23 | 0.33 | 8.34 |
| NIFTYSMLCAP250 | 2011-01-03..2026-09-10 | 12.15 | -60.79 | 0.20 | 6.04 |

### 3b-3d. Engine baselines

| arm | paths | CAGR gross | net | after-tax | [min..max] | worst path | MaxDD (worst) | Calmar | trades/yr | win% | turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|
| near-ATH k=0.90, N=15, RS-ranked, monthly, 12 offsets | 12 | 30.39 | 26.95 | 23.15 | [19.12 .. 27.61] | 19.12 | -53.22 (-61.43) | 0.42 | 82.9 | 45.3 | 5.73 |
| random-selection NULL, same universe+state, 30 seeds | 30 | 22.34 | 16.80 | 13.81 | [10.19 .. 17.96] | 10.19 | -44.84 (-55.56) | 0.32 | 142.7 | 50.9 | 10.36 |
| equal-weight hold-forever, top-250 by turnover | 1 | 12.34 | 12.22 | 12.22 | [12.22 .. 12.22] | 12.22 | -48.48 (-48.48) | 0.25 | 15.3 | 76.1 | 0.00 |

## 4. Cost / tax monotonicity

| cost (bps/side) | CAGR gross | CAGR net | CAGR after-tax |
|---|---|---|---|
| 0 | 34.84 | 34.84 | 29.15 |
| 25 | 34.84 | 30.58 | 26.36 |
| 40 | 34.84 | 29.43 | 24.80 |
| 60 | 34.84 | 26.58 | 22.58 |

**Verdict: PASS** — net CAGR is monotonically decreasing in cost (yes) and gross >= net >= after-tax in every row (yes).


## 5. Speed

| cell | mode | paths | total s | s per path (3 arms: gross/net/after-tax) |
|---|---|---|---|---|
| SMOKE_r158mask_k90_N15 | monthly rebalance | 1 | 1.9 | 1.86 |
| LA_next_open | monthly rebalance | 1 | 0.6 | 0.56 |
| LA_same_close | monthly rebalance | 1 | 0.3 | 0.33 |
| MONO_cost0 | monthly rebalance | 1 | 1.7 | 1.68 |
| MONO_cost25 | monthly rebalance | 1 | 1.6 | 1.56 |
| MONO_cost40 | monthly rebalance | 1 | 1.3 | 1.34 |
| MONO_cost60 | monthly rebalance | 1 | 2.0 | 2.02 |
| BASE_nearATH_k90_N15_rs_mo_12off | monthly rebalance | 12 | 14.2 | 1.18 |
| BASE_randomnull_N15_mo_30seed | monthly rebalance | 30 | 41.3 | 1.38 |
| BASE_ew_holdforever_top250 | monthly rebalance | 1 | 14.7 | 14.72 |
| SPEED_first_qualify_k90_N15 | daily first_qualify | 1 | 2.3 | 2.26 |

## 6. Real DATA-LEG mask, both missing policies

The mask interface exercised against the DATA-LEG's own npz rather than r/158's stand-in. Same cell either side: monthly rebalance, N=15, RS-ranked, k=0.90, tv >= Rs 2cr, no exits, 25 bps, 12 offsets.

| arm | CAGR after-tax | MaxDD (worst) | % invested | trades/yr |
|---|---|---|---|---|
| DATALEG_arun_strict_fail | 6.10 | -6.80 (-10.64) | 3.8 | 1.0 |
| DATALEG_arun_strict_pass | 11.37 | -53.11 (-58.58) | 66.7 | 27.4 |

**PASS as an interface test, and it immediately earned its keep.** The engine now prints a mask-coverage diagnostic on every masked cell, and on this one it raised two flags the study agent must not ignore:

1. **30% of the 2010-2026 window precedes the mask's first row** (2015-01). Those years are decided by the missing policy alone, with no fundamental evidence behind them. Align `--start` to the mask, or label the arm a coverage artefact.
2. **`arun_strict` passes 0.081% of name-months — 0.3 qualifying names per session against 15 slots.** That book physically cannot stay invested: its 6.10% after-tax "return" is close to the 5% idle-cash yield, and its shallow -6.8% drawdown is the drawdown of a cash pile, not of a strategy. Any mask this tight needs either far fewer slots or a looser screen before its numbers mean anything. Always read `avg_pct_invested` next to the CAGR.

The two missing policies differ by 5.3pp of CAGR and 46pp of drawdown here, which is the coverage bias measured rather than assumed. Report both, every time.


## Caveats the study agent must carry

1. The **hold-forever arm books no closed trades**, so its win rate, average win and average loss columns describe marked-open positions, not realised trades. Read only its CAGR and its drawdown.
2. A position in a name that stops printing a close is liquidated at the last known price after `stale_exit_days` (default 60) sessions. With 0 a delisted name would be carried at its last traded price for the rest of the run, which flatters every no-exit arm. This is a partial control, not a delisting model: the panel has no delisting reason or recovery value.
3. Index history in this DB starts **2011-01-03** for NIFTY 50 / NIFTY 500 / MIDCAP 150 / SMALLCAP 250; only NIFTYBEES reaches 2005. Windows differ and are printed with every table.
4. The universe is **not point-in-time**: names that delisted before 2026 are in the panel only if the DB kept them. Survivorship pressure is upward on every arm here, benchmarks included. The random-selection null is the control that matters — it carries the same bias, so the RS ranking premium over it (23.15 vs 13.81 after tax) is the part that survivorship cannot explain.
5. `capacity_ratio` is quoted at the default Rs 1cr book. Multiply by the real book size before reading it as a constraint.


One *path* runs three simulations (gross, net, after-tax). Divide by three for a single simulation. A 12-offset cell therefore costs ~12x the per-path figure; pass `arms=tax` to run only the after-tax arm when a sweep does not need the cost decomposition.
