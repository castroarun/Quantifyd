# research/136 — Universal Swing Rule: G1 RESULTS

**VERDICT: breakout REFUTED. Short-horizon reversion is a SIGNAL — real, highly
significant, and currently too thin to trade on its own.**

Run 2026-08-31 · daily bars 2015-01-01 → 2026-08-28 · 1,101 eligible symbols ·
1,050,447 eligible rows · entry at next open · date-matched random control.

---

## 1. Buying strength LOSES — on every cell

| lookback | horizon | signal | random (same day) | drift | **excess** | t |
|---:|---:|---:|---:|---:|---:|---:|
| 20d high | 2d | +0.088% | +0.197% | +0.153% | **−0.109%** | −7.20 |
| 20d high | 5d | +0.282% | +0.431% | +0.377% | **−0.149%** | −6.62 |
| 20d high | 10d | +0.708% | +0.813% | +0.748% | **−0.105%** | −3.34 |
| 55d high | 2d | +0.062% | +0.167% | +0.153% | **−0.105%** | −5.70 |
| 55d high | 5d | +0.242% | +0.344% | +0.377% | **−0.102%** | −3.64 |
| 55d high | 10d | +0.681% | +0.794% | +0.748% | **−0.112%** | −2.86 |

231,510 signals. 8 of 11 years negative. **Note the raw signal returns are all
positive** — a backtest without the control would have reported a working breakout
system. Against a random liquid stock on the same day it loses, consistently, at
t up to −7.2.

**This is the control earning its keep**, exactly as it did in research/87 and /88.

## 2. Buying weakness WINS — and RSI(2) is the carrier

| signal | horizon | signal | random | **excess** | t | n |
|---|---:|---:|---:|---:|---:|---:|
| 20d low | 2d | +0.143% | +0.091% | +0.052% | 2.91 | 112,701 |
| 20d low | 5d | +0.373% | +0.342% | +0.031% | 1.14 | 112,466 |
| 20d low | 10d | +0.578% | +0.585% | −0.006% | −0.17 | 112,173 |
| 55d low | 2d | +0.096% | +0.041% | +0.055% | 2.05 | 57,851 |
| **RSI(2)<10** | **2d** | +0.172% | +0.080% | **+0.091%** | **9.12** | 311,628 |
| **RSI(2)<10** | **5d** | +0.398% | +0.294% | **+0.104%** | **6.74** | 310,801 |
| **RSI(2)<10** | **10d** | +0.760% | +0.644% | **+0.117%** | **5.35** | 309,738 |

The Donchian-low construction is weak and dies by 10 days. **RSI(2)<10 is strong,
and its excess rises monotonically with horizon** — monotonic, not peaked, which is
the signature of an effect rather than a fit.

## 3. Why this is a SIGNAL and not yet a STRATEGY

The excess is **~9–12 bps per trade**. A realistic delivery round trip — brokerage,
0.1% STT on the sell, exchange and GST, plus slippage on a stock that just fell hard
— is 20–30 bps. **The edge is smaller than the toll.**

That does not make it useless; it makes it not-yet-tradeable *in this form*. Three
routes out, in order of promise:

1. **Hold longer.** The excess grows with horizon (0.091 → 0.104 → 0.117) while cost
   is paid once. Test 15 / 20 / 30 day holds — if the excess keeps climbing, cost
   amortises to irrelevance. Cheapest next test and the most likely to work.
2. **Condition harder.** Deeper oversold, only above a rising 200-DMA, only in
   uptrending sectors, only on liquid large caps. Fewer, better trades — but each
   condition is a fitting opportunity, so hold out data.
3. **Use it as an overlay, not a book.** An entry-timing filter on the momentum book
   already running, rather than a standalone system.

## 4. What is now settled

- **Short-horizon breakout momentum does not exist in this universe.** 2–10 day
  breakout entries are a *negative* edge, before costs. This retires the family and
  explains N500M's vol-BO half without needing any other story.
- **Short-horizon reversion does exist**, is highly significant on a third of a
  million signals, and strengthens with holding period.
- The intraday line stays closed (research/109 + /110).

## 5. Guarding the seven sins

| sin | control |
|---|---|
| look-ahead | entry at next open; every window trailing; no same-bar fills |
| survivorship | acknowledged — the symbol list is today's Nifty-500, so absolute levels are upper bounds. The date-matched control draws from the same tainted pool, so the *excess* is far more trustworthy than the level |
| overfitting | two constructions, three horizons, no parameter search, all cells reported |
| cost neglect | cost sensitivity 0/20/30/50 bps on every cell |
| regime | per-year table |
| single-factor | random-entry AND drift controls |
| capacity | ₹5 crore trailing-20d turnover floor; capacity untested at G1 |

## 6. Next

**G1c — horizon extension on RSI(2)<10: 15 / 20 / 30 day holds.** One script change.
If the excess keeps rising, this becomes a G2 candidate with real headroom over
costs. If it flattens near 12 bps, the standalone book is dead and route 3 (overlay)
is the survivor.
