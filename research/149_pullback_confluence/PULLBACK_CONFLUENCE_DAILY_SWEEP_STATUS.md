# Pullback Family Done Properly — MA Grid × Confluence Filters (second-pass audit of a killed family)

STATUS: DONE — **VERDICT: NO EDGE — 0 of 96 cells positive after-tax; best −0.195%/trade (ema20+CCI). The pullback sketch is closed permanently.** See `results/RESULTS.md`.

## 1. The Ask (verbatim)

> "i said 50 sma as a sample, did u study other sma/emas? additional criteria like rsi, rs,
> stochs, ccrb, etc are analyzed with?" — Arun, on r/146's pullback kill.

Correct: r/146 tested only the 50-SMA/EMA touch with exit variants. This study sweeps the
full sketch space: **MA ∈ {20, 50, 100, 200} × {SMA, EMA} × confluence ∈ {none, RSI(14)<40,
RSI(2)<10, RS-percentile≥70 (OA-style relative-strength leaders — the classically claimed
edge), Stoch(14)<20 turning up, CCI(20)<−100 turning up} × exits {2R/10d, 2R/15d}** =
**96 cells** (× gross/after-tax runs). Same engine, frame and candle mechanics as r/146
(green-after-red at the MA touch, buy-stop above the green high, SL below min(green low,
prior red low), hard stops, no averaging, PIT top-500, 10 slots, 25bps/side, FY-netted tax).

## 2. The prior (this is SECOND-PASS MINING of a killed family — highest overfit risk)

r/146: all six 50-MA pullback variants lost **−0.69..−0.91% per trade net** (CAGR −11..−19%,
DD −85..−97%). r/84 (dip-buy), r/87-88 (structure screens) are adjacent kills. The base rate
going in: dead.

## 3. Pre-registered resurrection bar (BINDING — multiple-testing guard)

The family is resurrected ONLY if there exists a cell with:

1. **positive AFTER-TAX net expectancy per trade** (50bps RT) on the full window, AND
2. **a plateau**: at least 2 of its neighbors (adjacent MA length, or same MA with another
   confluence) ALSO positive after-tax — no lone cells, AND
3. positive in **BOTH** W1 (2016-06→2019-12) and W2 (2020→now), AND
4. tradeability gate reported (WR, avg win/loss, max losing streak) and survivorship caveat
   applied (dip-buying on survivor data is optimistic — a marginal positive is treated as
   zero).

**The number of cells tested (96) is reported so any discovery is discounted properly
(best-of-96 at t≈2 is expected under pure noise).** If the family stays dead with
confluence, Arun's sketch is closed permanently with a clean conscience.
Blend tests vs TN+OA: only for a resurrected survivor.

## 4. Plan / grid

96 cells × {tax0, tax1} ≈ 192 runs on the r/146 engine (`sleeve_engine.py`), extended with:
RSI(14), Stoch(14) %K, CCI(20) (rolling-mean-deviation form), and the OA-style RS percentile
(2·r63 + r126 + r189 + r252, cross-sectional pct rank, causal at close t). Engine edits are
additive/back-compatible (new optional params on the 'pull' family; defaults reproduce r/146).

Seven sins: as r/146 (same engine, STATUS discipline); the specific risk here is
multiple-testing — handled by the pre-registered plateau bar + cell-count disclosure +
survivor-data discount.

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-04 01:5x | STATUS written (bar + grid locked) before any run; engine extension building | — |
| 2026-09-04 02:1x | GRID DONE (192 runs, 160s): 0/96 cells positive after-tax; conf means CCI −0.59 (least bad) / none −0.76 / rs70 −0.74 / rsi14 −0.82 / rsi2 −0.94 / stoch −0.69; MA means ema20 −0.52 (least bad) → sma200 −1.06. Resurrection bar never approached. RESULTS.md written; committed. | 4th independent kill of the dip-buy entry family |

## 6. Crash recovery

- VPS `/home/arun/quantifyd/research/149_pullback_confluence/`; log `/tmp/tn149.log`.
- Incremental `results/pullback_grid.csv` (label-keyed, resume-safe).
- Resume: `cd /home/arun/quantifyd && setsid nohup venv/bin/python -u
  research/149_pullback_confluence/scripts/pull_sweep.py > /tmp/tn149.log 2>&1 &`
- Engine: `research/146_complementary_third_sleeve/scripts/sleeve_engine.py` (shared).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| this STATUS md | live status | yes |
| `scripts/pull_sweep.py` | grid runner | yes |
| `results/pullback_grid.csv`, `results/RESULTS.md` | outputs | yes |

## 8. Findings

(see RESULTS.md)
