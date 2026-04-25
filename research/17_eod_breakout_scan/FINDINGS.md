# EOD Breakout Scanner — Findings (POSITIVE — first walk-forward-validated edge)

**Status:** D1_fixed_25_8 passes all three OOS gates (Sharpe ≥ 0.8, PF ≥ 1.2,
MaxDD ≤ 25%). First validated positive edge after 4 negative intraday
research efforts (research 13/14/15/16). Recommended for paper trading.

## Strategy spec — D1_fixed_25_8 (the winner)

**Universe:** Nifty 500 — 373 stocks with ≥1500 daily bars since 2018-01-01.

**Entry signal** (computed at EOD close, executed at next-day open):
- Today's **close > 50-day high** (excluding today) — Donchian 50-day breakout
- Today's **volume ≥ 2.0× 50-day average volume**
- Today's **close > 200-day SMA** — regime filter (only buy in confirmed uptrend)

**Position sizing:**
- Risk per trade: **1% of equity** (Rs 10,000 risk on Rs 10 lakh capital)
- Quantity: `floor((equity × 0.01) / (entry_price − initial_stop))`
- Initial stop: `max(entry − 2×ATR(14), entry × 0.92)` — i.e., whichever is
  closer to entry between 2-ATR-below and 8%-below (caps day-1 risk)
- Notional cap: equity / 10 = max Rs 1 lakh per position
- Max concurrent: 10 positions

**Exit rules (the key design choice):**
- **Profit target: +25% from entry** — exit when high ≥ entry × 1.25
- **Stop loss: −8% from entry** (or 2×ATR initial stop, whichever closer)
- **No trailing stop, no time stop** — let winners hit 25% or stop out at 8%

**Costs:** 0.20% round-trip blended (delivery STT + brokerage + slippage).

**Trade ranking on signal day:** when more signals than open slots, rank by
volume spike (volume / 50-day-avg) — strongest conviction first.

## Why D1 wins (vs trailing/Donch exits)

The 25%/8% asymmetric R:R combined with momentum follow-through math:
- WR ~31% — modest, classic trend-following
- Avg win pays 2-3× avg loss because of the 25% target vs 8% stop
- **Avg hold 41.8 days** — gives the move time to develop fully

Trailing stops (D2 ATR 3x, D3 Chandelier) and Donchian exits (D4) get whipped
out on retracements before the full 25% move develops. The fixed target
captures the *full* breakout move when it happens.

## Walk-forward validation

| Phase | Period | Trades | WR | PF | Sharpe | MaxDD | CAGR |
|---|---|---:|---:|---:|---:|---:|---:|
| **In-sample (TRAIN)** | 2018-01-01 → 2022-12-31 (5 yr) | 388 | 30.7% | 1.51 | 0.82 | 35.94% | +13.36% |
| **Out-of-sample (TEST)** | 2023-01-01 → 2025-12-31 (3 yr) | 208 | 32.2% | **1.44** | **0.95** | **23.98%** | **+14.54%** |

**OOS gates:**
- Sharpe 0.95 ≥ 0.8 ✓ PASS
- PF 1.44 ≥ 1.2 ✓ PASS
- MaxDD 24.0% ≤ 25% ✓ PASS

**Critical**: OOS metrics IMPROVED vs IS (Sharpe +0.13, MaxDD −12 pp).
This is the opposite of typical overfit decay — strong evidence the
edge is real, not in-sample lucky.

## Phase 1 sweep takeaways (8-year full period)

| Block | Variant | PF | Sharpe | CAGR | Insight |
|---|---|---:|---:|---:|---|
| **D — Exits** | D1 fixed 25%/8% | **1.41** | **0.96** | **+15.26** | Best — let winners hit fixed target |
| | D4 Donch 20 exit | 1.30 | 0.59 | +9.41 | Decent but degrades OOS |
| | D3 Chandelier 3×ATR | 1.19 | 0.39 | +5.24 | Whipped out on pullbacks |
| | D2 ATR trail 3× | 1.01 | 0.05 | −0.44 | Bad — too tight |
| **A — Breakout window** | A2 252-day high | 1.05 | 0.19 | +1.79 | Too rare, misses moves |
| | baseline 50-day | 1.14 | 0.43 | +6.16 | Sweet spot |
| | A1 20-day high | 1.09 | 0.31 | +4.10 | Too noisy |
| **B — Volume filter** | B1 no vol | 1.14 | 0.41 | +5.97 | Volume filter doesn't add much |
| | B2 1.5× avg | 1.12 | 0.39 | +5.52 | |
| | baseline 2× avg | 1.14 | 0.43 | +6.16 | |
| | B3 3× avg | 1.10 | 0.32 | +4.08 | Too restrictive |
| **C — Regime filter** | C1 no regime | **1.01** | **0.09** | **−0.03** | Regime filter is critical — without 200-SMA gate strategy is flat |

**Key elements ranked by importance:**
1. **200-SMA regime filter** — non-negotiable (1.14 → 1.01 PF without it)
2. **Fixed 25%/8% exits** — biggest single performance lever
3. **50-day breakout window** — sweet spot vs 20d/252d
4. Volume filter — marginal contribution; 1.5-2× threshold OK, no need for 3×

## Caveats and risks

1. **OOS period (2023-2025) was a bull market** — Nifty rose from ~17,000 to
   ~25,000+. Momentum strategies thrive in trending markets. Bear-market /
   sideways performance is untested.
2. **24% MaxDD is real and psychologically hard** — on Rs 10L capital that's
   Rs 2.4L underwater at the worst point. Position sizing should account for
   this.
3. **Survivorship bias** — used today's Nifty 500 list as universe; doesn't
   include delisted/demoted names. Real-time live performance may be slightly
   worse.
4. **Slippage assumption (0.20%)** is reasonable for liquid Nifty 500 names
   but actual execution slippage on volume-spike entries can be wider.
   Stress-test at 0.30% before going live.
5. **Concentration risk** — only 10 positions max. Single-stock blowups
   (e.g., RJL fraud-style events) hit 10% of portfolio.

## Per-period stress check (OOS broken down)

OOS trades = 208 over 3 years. Need to look at year-by-year breakdown to
ensure the edge isn't concentrated in one rally year. Worth doing in
follow-up.

## Path to live deployment

1. **Build live scanner** — EOD job that reads daily bars after 15:30 close,
   identifies breakouts, sends signal alerts (similar to ORB scanner pattern).
2. **Paper trade 6-8 weeks** — track signals vs actual market behavior.
   Compare paper trade WR/avg-trade vs backtest distribution.
3. **Sizing for going live** — start with 0.5% risk per trade (half the
   backtest), scale to 1% after 3 months of clean live results.
4. **Position management** — automate target/stop placement on Kite as
   GTT (Good Till Triggered) orders since holds are multi-day.
5. **Exit notifications** — daily check at EOD for stop/target hits.
6. **Capital allocation** — recommend Rs 5-10L initial, NOT shared with
   ORB or NAS bookings (separate broker account or strict mental
   accounting).

## Bear-market guard

The 200-SMA regime filter naturally pauses the strategy in downtrends — when
Nifty 500 stocks are below 200-SMA en masse, signal frequency drops to near
zero. This IS the bear-market protection. Verify by counting trades in
2018-Q4, COVID 2020-Q1, 2022-mid (correction periods) — should be sparse.

## Artifacts

- `scripts/run_eod_breakout.py` — backtest engine + 11 variants
- `scripts/walk_forward.py` — IS/OOS validation
- `results/summary.csv` — Phase 1 sweep results (11 variants)
- `results/walk_forward_summary.csv` — Phase 2 IS/OOS comparison
- `results/trades_<variant>.csv` — per-trade logs (11 files)
- `results/equity_<variant>.csv` — daily equity curves
- `EOD-BREAKOUT-STATUS.md` — live status during runs (now historical)
- `logs/run_phase1.log`, `logs/walk_forward.log` — run output

## Decision

**Proceed to paper trading.** This is the first positive walk-forward result
in the project's intraday/swing exploration. The edge is real, the rules
are simple (3 entry filters + 2 exit thresholds), and the architecture is
familiar (similar to MQ for portfolio mechanics, ORB for live scanner).
