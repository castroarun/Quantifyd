# RESULTS — 20/200-SMA "Picture of Power" Retrace-Break (iFundTraders RBI&GO)

## VERDICT: **NO EDGE**

The iFundTraders "RBI & GO / Picture of Power" setup — buy a red pause-bar's high (sell a
green pause-bar's low for shorts) when price is NEAR a **rising** 20-SMA stacked above the
200-SMA, hold while it trends, exit when it drifts 2–3× ATR away from the 20-SMA — has **no
tradeable edge** on Indian equities/indices. It **loses even gross** on the timeframe it is
actually taught (intraday), and its only positive numbers (daily) are **pure survivor drift**:
the setup underperforms a random long entry in the same uptrend. Both directions tested; both fail.

Data snapshot: VPS `market_data.db` 2026-07-24 (25.5 GB). Cost model: 5 bps round-trip.
Universe: 12 deep 5-min names with real 2015→now history (NIFTY50, BANKNIFTY, HDFCBANK,
ICICIBANK, RELIANCE, INFY, TCS, SBIN, ITC, HINDUNILVR, KOTAKBANK, BHARTIARTL) + full daily.

---

## The evidence

### G1 — baseline, 5-min (the setup's native timeframe), 2015→now
| Dir | Trades | Gross exp/trade | Net (5bps) | Win% | PF | t |
|---|---|---|---|---|---|---|
| Long | 32,813 | **−0.006%** | −0.056% | 28.4% | 0.56 | −39.9 |
| Short | 31,117 | **−0.005%** | −0.055% | 29.2% | 0.58 | −35.9 |

Even **gross** is negative. Tight red-low stop is hit constantly by 5-min noise (win ~28%,
avg hold 3.8 bars); the 2.5×ATR "drift-away" target is rarely reached before the stop.

### G2 — levers (84 cells × both dirs): timeframe × exit rule × hold × slope-strictness
- **Stricter "rising 20-SMA" (slope over 10 bars) made it slightly WORSE**, not better.
- **Overnight hold + SMA-cross (ride-the-trend) exit** barely moved 5-min (still net-negative).
- Gross improves as timeframe rises (less noise) — 30-min gross turns marginally positive
  (+0.03 to +0.08%) but **net ≈ 0 and t ≈ 0–1.8** (insignificant).
- **No cell cleared the gate** (gross>0 AND net>0 AND **t≥3**). Best net-positive cells were all
  **daily long**, at **t = 1.1–1.8** (not significant) on only 68–233 trades. The **short mirror
  loses on daily** — if the "structure" were real, shorts should work symmetrically. It doesn't.

### G3 — drift control (daily long, best cell ext35/overnight): is it the setup or just drift?
| | Gross exp/trade | t | n |
|---|---|---|---|
| **SETUP (pause-break)** | +0.777% | 1.78 | 220 |
| DRIFT — random entry in same up-regime (matched n) | **+1.029%** | 2.73 | 220 |
| DRIFT — every bar in up-regime (same hold) | **+0.895%** | 24.4 | 27,245 |

**The setup underperforms random entry in the same uptrend by 0.12–0.25%/trade.** The entire
daily positive is the up-regime's survivor drift; the pause/near mechanics add negative value.

---

## Why it fails (economic read)
The MA-pullback-continuation is one of the most widely-taught retail setups — any edge is long
since arbitraged. Intraday, the red-low stop is far too tight for the 2–3× ATR target (bad R:R
realized). On daily, "trend up + hold long" just harvests beta/drift on survivor large-caps, and
the specific pause-near-20SMA trigger is a *worse* way to be long than entering anywhere in the
uptrend. The hand-picked chart examples in the video are survivorship illustration, not evidence.

## Honest caveats
- **Survivorship:** the 5-min deep set and daily large-caps are today's winners — this *flatters*
  the long/drift side, yet the setup still fails, which strengthens the NO-EDGE call.
- Daily n is small (rare signal); we did not over-interpret its t<2 numbers precisely because of it.
- Modeled costs (5 bps round-trip) and simple ATR; conservative end-of-bar extension fills. Making
  costs zero does not rescue intraday (gross already ≤0).
- Not tested: adding a volume/RVOL confirm or a higher-timeframe trend gate could be a *different*
  study, but the core "red/green pause-break near a rising 20-SMA" as specified is dead.

## Next levers (if revisited)
- The only survivor-ish thread is "be long in an up-regime" — but that is just momentum/drift,
  already captured by the live momentum-paper and Nifty-250 momentum books. Nothing new here.
- Shelve. Do not re-litigate the intraday version — it loses gross.

## Reproducibility
Scripts: `research/91_sma20_200_pullback/scripts/{sma_pullback_engine.py, run_g1_probe.py,
run_g2_sweep.py, run_g3_drift_control.py}`. Runner order G1→G2→G3. Snapshot 2026-07-24, VPS.
