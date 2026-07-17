# research/83 — Faithful Turtle (Dennis/Covel) on Indian F&O Equities: RESULTS

**VERDICT: (1) SHORTS — FINAL CLOSURE AT ALL HORIZONS: turtle-style multi-week
trailing shorts are the worst yet (S1_S −220bps t=−11.3, S2_S −372bps t=−11.7,
12% syms positive, S2 0% of years). With r/81 (≤4d) and r/82 (5-15d), the
directional-short-swing question on Indian equities is closed permanently.
(2) LONGS — turtle mechanics WORK and modestly beat the live-book rules
in-sample (Calmar 0.45 vs 0.37), but the ENTIRE equity-trend family is
roughly flat 2018-2023 outside the 2020-21 spike — the decay shadow again.
User decision: RECORDED ONLY; no change to the live breakout paper book,
whose forward soak is the real arbiter. OOS never consumed.**

## EXP-T1 probe (IS 2005-2017, 4 cells, net 3bp)

| Cell | n | net | t | syms+ | yrs+ | hold |
|---|---|---|---|---|---|---|
| S1(20/10) L | 3,397 | +241bps | 7.5 | 82% | 77% | 36d |
| S2(55/20) L | 1,779 | +444bps | 6.4 | 76% | 77% | 58d |
| S1 S | 2,804 | −220bps | −11.3 | 12% | 15% | 29d |
| S2 S | 1,270 | −372bps | −11.7 | 12% | 0% | 40d |

Trailing exits produce the classic trend profile (win ~33%, huge winners) and
keep chop years positive (10/13 yrs) where fixed holds (r/82 M1) failed.

## EXP-T2 book bakeoff (gated NIFTY>200DMA, 78 F&O names, 2005-2023)

| Arm | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| A live-book rules (20/20 trail, no stop, cap 8, EQ) | 11.2% | 0.79 | −30.4% | 0.37 |
| B turtle S1+S2, 2N stop, N-SIZED, cap 12 | 13.6% | 0.76 | −36.5% | 0.37 |
| **C turtle S1+S2, 2N stop, EQUAL-NOTIONAL, cap 12** | **14.4%** | **0.82** | −31.9% | **0.45** |
| NIFTYBEES B&H | 13.5% | 0.73 | −59.7% | 0.23 |

- Dual-system + 2N stops add value over single-channel (C > A on the
  pre-declared criterion).
- **Turtle N-sizing LOSES to equal-notional** — third independent failure of
  stop/vol-keyed sizing in this engagement (r/81 OR-width inversion, G5
  equal-risk inversion, now N-units). Equal-notional is the house default.
- **Dominant caveat:** every arm is ~flat 2018-2023 ex-2020/21. The family's
  returns are front-loaded in 2005-2017. Backtest wins here must be
  discounted for forward decay — the live paper book's soak decides.

## Honest caveats

Arm A is a reconstruction of the live book's rules, not the running
implementation; survivorship-biased universe; futures-proxy costs; IS+Val
only (OOS 2024+ unconsumed); pyramiding untested (phase 2 not earned).

## Reproducibility

Dedicated trailing-exit simulator w/ synthetic sanity check
(`scripts/run_turtle_probe.py`), bakeoff `scripts/run_t2_bakeoff.py`,
data snapshot VPS market_data.db 2026-07-17. Ledger: T1 4 cells + T2 3
constructions.
