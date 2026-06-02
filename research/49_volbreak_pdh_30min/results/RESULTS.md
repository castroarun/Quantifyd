# Volume-MA + Prev-Day-High Breakout, 30-min Intraday LONG — Results (G1 smoke)

**VERDICT: NO EDGE (intraday, net of cost).** There is a faint *gross* signal
(best gross +0.081R on a 1-ATR hard stop), but it **does not survive realistic
cost**: every exit policy is net-negative at 6 bps round-trip, the best is
**−0.029R / PF 0.95**, and there is no per-year persistence. It fails the
pre-registered falsification bar (need net ≥ +0.16R AND PF ≥ 1.24 — research/44).
RSI gating does not help. Consistent with prior art: research/44 found the raw
prev-day-high breakout is breakeven and **intraday is worse than swing**, and the
playbook's cost-units scar (research/43) — tight intraday stops make a flat bps
cost enormous in R-terms. **Recommendation: SHELVE the intraday version.**

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

## Next levers (low expected value, listed for completeness)

1. **Volume dose-response** — require `vol > k×VolMA`, k∈{1.5,2,3}, to see if
   stronger surges concentrate any edge (cheap; but 40 suggests cherry-pick risk).
2. **Swing hold (carry multi-day)** — 44 shows this is the only place this family
   has *any* edge (~PF 1.24 with NR7+high-beta+HTF-trend), so a fresh intraday
   sweep adds little; if pursued, build *on 44's* filtered swing core, not this.
3. Otherwise **CONCLUDED** — do not spend the 30k-cell sweep on the intraday form.
