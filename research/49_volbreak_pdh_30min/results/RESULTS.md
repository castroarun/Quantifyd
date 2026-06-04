# Volume-MA + Prev-Day-High Breakout, 30-min — Results (G1: intraday + positional)

**VERDICT: NO EDGE — intraday dies to cost; positional is pure BETA, not alpha.**

- **Intraday (Phase 1):** faint gross signal (best +0.081R on a 1-ATR stop) but
  every exit is **net-negative at 6 bps** (best −0.029R / PF 0.95); no persistence.
- **Positional / multi-day hold (Phase 2):** looks great at first — daily-Supertrend
  exit **net +0.701R / PF 1.54**, Chandelier +0.490R / PF 1.53 — and clears the
  falsification bar. **BUT the placebo/benchmark kill (Phase 3) shows it is 100%
  market beta:** entering on *any random day* and holding ~a month with the same
  trailing stop gives the **same** result (BASELINE Supertrend +0.646R / PF 1.49).
  The volume filter slightly **hurts** (SIGNAL +0.701 < BREAK_ONLY +0.733), and the
  prev-day-high break adds **nothing** over a random entry (BREAK_ONLY ≈ BASELINE).
  The +0.70R is just large-cap forward drift in the 2018–2025 bull, captured by
  anything. **No incremental alpha from the volume+breakout entry.**

Consistent with prior art (research/44: raw breakout breakeven; intraday < swing)
and the playbook's cost-units scar (research/43). **Recommendation: CONCLUDED —
shelve both forms.** The entry signal carries no edge over unconditional exposure.

## Setup (G1 cheap probe)

- Universe: 8 liquid names (RELIANCE, TCS, HDFCBANK, ICICIBANK, SBIN, INFY,
  MARUTI, TATASTEEL). Cohort-A from 2018; MARUTI/TATASTEEL from 2024 (5-min start).
- Bars: 30-min resampled 6×5-min anchored 09:15 (12 bars/day).
- Signal (LONG): first 30-min bar/day with `close > prev-day high` AND
  `bar volume > trailing 50-day (600-bar) volume MA, shifted 1`. Enter next bar open.
- R = 30-min ATR(14) at the signal bar (causal). 6 bps round-trip cost.
- n = **3,486 trades** (no-RSI), 2018→2026, 5 exit policies scored per entry.

## Result — per exit policy (no-RSI, LONG)

| Exit | n | WR% | gross R | net@0bp | **net@6bp** | net@12bp | PF@6bp |
|---|--:|--:|--:|--:|--:|--:|--:|
| EOD | 3486 | 45.6 | +0.032 | +0.032 | −0.078 | −0.189 | 0.89 |
| **HARD_SL (1 ATR)** | 3486 | 38.2 | **+0.081** | +0.081 | **−0.029** | −0.139 | **0.95** |
| R_TARGET_2R | 3486 | 39.8 | +0.032 | +0.032 | −0.078 | −0.188 | 0.87 |
| CHANDELIER_3ATR | 3486 | 45.0 | +0.037 | +0.037 | −0.073 | −0.184 | 0.90 |
| SUPERTREND_10_3 | 3486 | 45.6 | +0.032 | +0.032 | −0.078 | −0.189 | 0.89 |

RSI≥55 variant: uniformly equal-or-worse (best HARD_SL net@6bp −0.029, PF 0.95).

**Cost is the killer.** Gross edge is small-positive (+0.03 to +0.08 R); the
6 bps round-trip costs ~0.08–0.11 R per trade (small intraday-ATR R → large
cost-in-R), flipping everything negative. At 0 bps several policies are mildly
positive — i.e. a *gross* signal exists, but it is not investable.

## Per-year net@6bp (no-RSI) — no persistence

Only 2019 is broadly positive; 2020/2021/2023/2025/2026 negative across policies.
Best policy (HARD_SL) is positive in 2018/2019/2020/2022/2024 but negative in
2021/2023/2025/2026 — fragile, regime-dependent, no monotonic stability.

## Seven deadly sins — how controlled

- **Look-ahead:** prev-day high known pre-open; volume-MA shifted 1; entry on
  *next* bar open; ATR causal. ✓
- **Cost neglect:** gross + net + 0/6/12 bps sensitivity reported. ✓ (and decisive)
- **Survivorship:** smoke uses today's-liquid large-caps → upward-biased; the real
  edge across a broad universe would be ≤ this. Flagged.
- **Overfitting:** single threshold tested, no peak-picking; result is negative so
  multiple-testing inflation is moot.
- **Regime:** per-year table shown — fails the weak years.
- **Correlation / capacity:** not reached (died at G2 on cost).

## Honest caveats

- Smoke universe is 8 large-caps (selection bias **favourable**); broad-universe
  result would very likely be **worse** (44's 374-name raw edge was breakeven vs a
  4-name smoke of PF 1.11–1.15).
- "50-day MA" implemented as a 600-bar (≈50×12) trailing mean of 30-min volume; a
  same-time-of-day seasonal MA was not tested (unlikely to change a cost-bound verdict).
- Only LONG tested (per spec). 2R uses pessimistic stop-before-target intrabar.

## Phase 2 — Positional (multi-day hold, daily management)

Same entry, held across days, R = daily ATR(14). n=3,482 (no-RSI), 8 names.

| Exit | WR% | hold | gross R | net@6bp | net@12bp | PF@6bp |
|---|--:|--:|--:|--:|--:|--:|
| SUPERTREND daily(10,3) | 56.4 | 29.8d | +0.730 | +0.701 | +0.672 | 1.54 |
| CHANDELIER 3ATR daily | 47.0 | 17.4d | +0.519 | +0.490 | +0.460 | 1.53 |
| HARD_SL (1 ATR) | 24.9 | 12.3d | +0.377 | +0.347 | +0.318 | 1.45 |
| MAXHOLD_10 | 36.2 | 6.0d | +0.203 | +0.174 | +0.145 | 1.28 |
| R_3R | 31.1 | 7.2d | +0.198 | +0.169 | +0.140 | 1.24 |

Caveats: per-signal expectancy (event study, overlapping) — a SIGNAL not a curve;
Supertrend hold ≈ the 30-day backstop (rarely flips → really "hold ~1 month");
per-year shows 2024 & 2026(partial) negative (regime-dependent long beta).

## Phase 3 — Placebo / benchmark (the kill): NO ALPHA

Identical positional exits across three entry arms (mean net@6bp R / PF):

| Exit | SIGNAL (vol+break) | BREAK_ONLY (no vol) | BASELINE (any day) |
|---|--:|--:|--:|
| HARD_SL | +0.347 / 1.45 | +0.377 / 1.50 | +0.333 / 1.44 |
| CHANDELIER_3ATR | +0.490 / 1.53 | +0.532 / 1.58 | +0.508 / 1.54 |
| SUPERTREND_D | +0.701 / 1.54 | +0.733 / 1.56 | +0.646 / 1.49 |
| MAXHOLD_10 | +0.174 / 1.28 | +0.185 / 1.30 | +0.169 / 1.27 |

**SIGNAL ≈ BASELINE for every policy.** Volume filter adds nothing (slightly hurts);
prev-day-high break adds nothing over a random-day entry. The positional return is
**unconditional large-cap drift** (2018–25 bull), not signal alpha.

## Next levers (low expected value)

1. **CONCLUDED** is the honest call — the entry has no edge over buy-and-hold-ish
   exposure on either timeframe.
2. If a long-hold momentum sleeve is wanted, that's the MQ book's job (32–48% CAGR),
   not this signal — and any such sleeve must be benchmarked vs buy&hold.
3. A genuinely different test (not this signal) would need short-side / market-neutral
   construction so the result isn't just beta.
