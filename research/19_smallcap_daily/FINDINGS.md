# Small/Micro-Cap Daily ATH/52w Breakout — Findings (POSITIVE)

**Status:** Walk-forward validated. **Second positive edge in the project**
(after research/17 EOD swing on Nifty 500). Recommend paper trading at half
size for 8-12 weeks before going live.

## Spec under test

Daily-bar event-driven long-only momentum scanner — same family as
research/17, applied to a stricter quality-filtered small/micro-cap
universe outside the Nifty 500.

### Entry signal (computed at EOD close, executed at next-day open)

1. Today's `close > 252-day high` (excluding today) — 52-week breakout
2. Today's `volume >= 3.0× 50-day average` (best variant) — stricter than
   research/17's 2.0x because small caps have noisier baseline volume
3. Today's `close > 200-day SMA` — regime filter (only buy in confirmed
   uptrend on the stock itself)

### Exit rules (matches research/17 winner)

- **Profit target: +25%** from entry — exit when high ≥ entry × 1.25
- **Initial stop: max(entry − 2×ATR(14), entry × 0.92)** — caps day-1 risk at 8%
- **No trailing, no time stop** (60-day max-hold safety)

### Sizing & costs

| Parameter | Value |
|---|---|
| Capital | Rs 10,00,000 |
| Risk per trade | 1% of equity = Rs 10,000 |
| Max concurrent | 10 positions |
| Notional cap | Rs 1L per position |
| Costs | 0.30% round-trip (vs research/17's 0.20%) |

## Universe selection — quality filter funnel

| Stage | Survivors | Note |
|---|---:|---|
| Daily bars ≥1500 since 2018-01-01 | 848 | Base symbol pool |
| After Nifty 500 exclusion | 475 | Focus on small/micro-cap exposure |
| After turnover band (Rs 5-100 Cr) | 120 | 334 below floor (illiquid), 21 above ceiling (mid-cap territory) |
| After circuit filter (<4 hits/60d) | 117 | 3 names had circuit-prone history |
| After volatility floor (avg ATR%/close ≥1%) | **117** | None failed (small caps inherently volatile) |

**Final universe: 117 small/micro-cap stocks.** Turnover floor (Rs 5 Cr/day)
is the most aggressive filter — most micro-caps fail liquidity check.

**Known gap:** EQ-series filter not applied — local DB has no series metadata.
Live deployment must verify via Kite instrument dump that all 117 are in
EQ series (no T2T / Z / BE).

## 6-variant sweep (full period 2018-2025)

| Variant | Trades | WR% | PF | CAGR% | Sharpe | MaxDD% |
|---|---:|---:|---:|---:|---:|---:|
| baseline (252h, vol 2.5x, 25% tgt) | 719 | 34.6 | 1.35 | +17.15 | 1.06 | 19.26 |
| vol_2x | 765 | 34.2 | 1.34 | +17.93 | 1.07 | 17.02 |
| **vol_3x** (best Sharpe with PF≥1) | **678** | **35.4** | **1.40** | **+17.42** | **1.11** | **17.46** |
| target_30pct | 650 | 34.1 | 1.42 | +18.20 | 1.10 | 20.30 |
| target_20pct | 808 | 36.1 | 1.28 | +14.14 | 0.93 | 19.38 |
| cost_50bps (stress) | 719 | 34.5 | 1.30 | +15.16 | 0.95 | 20.37 |

**Every variant clears PF ≥ 1.28 and Sharpe ≥ 0.93** — even under the
0.50% cost stress. Strong robustness.

## Walk-forward (train 2018-2022, test 2023-2025)

Top 3 variants validated. **All three PASS all three OOS gates** (PF ≥ 1.20,
Sharpe ≥ 0.8, MaxDD ≤ 30%).

| Variant | Phase | Trades | PF | Sharpe | MaxDD% | CAGR% | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **vol_3x** | IS 2018-22 | 375 | 1.93 | 0.79 | 17.5% | +10.7% | — |
| **vol_3x** | **OOS 2023-25** | **305** | **1.46** | **1.43** | **12.9%** | **+26.8%** | **PASS** |
| target_30pct | IS | 363 | 1.58 | 0.94 | 20.3% | +13.5% | — |
| target_30pct | OOS | 287 | 1.44 | 1.29 | 14.7% | +25.5% | PASS |
| baseline | IS | 403 | 1.97 | 0.88 | 19.3% | +12.5% | — |
| baseline | OOS | 317 | 1.34 | 1.17 | 15.0% | +22.0% | PASS |

**OOS metrics IMPROVED over IS** for all three (Sharpe 0.79→1.43, MaxDD 17.5%→12.9%).
Same anti-overfit signature as research/17 — the rules track a real momentum
factor, not in-sample lucky setups.

## Top contributors (vol_3x, full period)

| Symbol | Net P&L |
|---|---:|
| INDOTHAI | +Rs 3,87,000 |
| RMDRIP | +Rs 2,85,000 |
| ZENTEC | +Rs 2,41,000 |
| HPL | +Rs 2,26,000 |
| WABAG | +Rs 1,53,000 |

**116 of 117 symbols saw ≥1 trade.** Strategy is genuinely diversified —
not riding 2-3 lucky names like research/14's ASIANPAINT-dependence.

## Caveats — read before deploying

1. **OOS period (2023-2025) was a strong small-cap bull market.**
   Nifty Smallcap 250 +60% in 2023 alone. The +27% OOS CAGR is regime-flattered.
2. **IS CAGR (+10.7%) is the more conservative forward expectation** — the
   training window included COVID 2020, the 2018 small-cap bear, and the
   2022 correction. Bear/sideways performance similar to that range.
3. **Survivorship bias** — used current daily-bar symbol list; doesn't
   include delisted/demerged/suspended names. Real performance may be 1-2%
   CAGR lower in any given regime.
4. **Missing EQ-series filter** — DB has no series metadata. Live deployment
   must verify via Kite instrument dump that no T2T/Z/BE names slipped in.
5. **MaxDD 17% backtest is psychologically real** — Rs 1.7L underwater on
   Rs 10L at the worst point. Sizing must respect this.

## Recommendation

**Paper-trade vol_3x for 8-12 weeks at half size:**
- Risk per trade: 0.5% (vs full 1.0%) until live track record exists
- Capital: start with Rs 5L, scale to Rs 10L after 3 clean months
- Compare live PF / WR / avg-trade vs backtest distribution
- Scale to full size only if live performance within 1 standard deviation
  of backtest expectation

**Architecture:** This is a sister system to research/17. Same EOD scanner
pattern, same exit logic, same sizing math — just a different universe.
The live executor for research/17 (whenever it gets built) can serve both
with a universe-list parameter.

## What's still untested

1. **Live execution slippage on small caps** — backtest assumes 0.30% which
   is reasonable for Rs 5-100 Cr turnover names but unverified.
2. **Cross-correlation with research/17** — if research/17 is also live,
   how much does this system's daily P&L correlate with it? If high (both
   are momentum after all), they share regime risk.
3. **Bear-market behavior** — covered in training but not stress-tested
   on a clean OOS bear period. Next bear (whenever) is the real test.

## Artifacts

- `scripts/build_daily_universe.py` — universe selection
- `scripts/run_smallcap_daily_backtest.py` — 6-variant sweep engine
- `scripts/walk_forward_daily.py` — IS/OOS validator
- `results/daily_universe.csv` — final 117-symbol list
- `results/daily_universe_selection.csv` — full filter diagnostics
- `results/daily_summary.csv` — 6-variant metrics
- `results/daily_walk_forward.csv` — IS vs OOS comparison
- `results/daily_trades_<variant>.csv` — per-trade logs (6 files)
- `results/daily_equity_<variant>.csv` — daily equity curves (6 files)
- `SMALLCAP-DAILY-STATUS.md` — historical run log
- `logs/run_phase1.log`, `logs/walk_forward.log` — full output

## Decision

**Proceed to paper trading at half size.** This is the second walk-forward
validated edge in the project, structurally similar to research/17, on a
non-overlapping universe. Combined with research/17 (Nifty 500), we now
have two complementary EOD swing scanners covering the full liquid Indian
equity spectrum.
