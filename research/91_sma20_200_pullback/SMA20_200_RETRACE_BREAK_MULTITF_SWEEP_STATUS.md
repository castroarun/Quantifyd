# 20/200-SMA "Picture of Power" Retrace-Break — Multi-TF Sweep (iFundTraders RBI&GO)

STATUS: DONE — **NO EDGE** (G1 loses gross intraday; daily "profit" = survivor drift, setup < random)

Research folder: `research/91_sma20_200_pullback/`. VPS-canonical (DB 25.5 GB on `94.136.185.54`).
Laptop is dev-only (NO local DB) — all runs execute on VPS via paramiko.

---

## 1. The Ask

**What Arun asked (across several chart clips from an iFundTraders "RBI & GO" video):**
> "20-200 sma assessment ... test these exact scenarios for a trade setup ... the rising 20
> sma, the crossover, more greens and less reds ... find those little red bars after a green,
> now the next candle if it takes out the red high, entry just above the red and SL a step
> below the red ... entry must always be near the 20 sma and not far ... trade holding period,
> when the price drifts away from the 20 sma, u start selling ur positions ... the retracement
> for entry, the going away for getting out ... now this is for shorts too."

**What we are actually testing:** A trend-continuation micro-pullback break around a rising/falling
20-SMA, filtered to be NEAR the 20-SMA, held while the trend runs, and exited when price extends
FAR (ATR-measured) from the 20-SMA. Tested LONG and SHORT separately, across timeframes
(5-min deep names → 15/30-min resample → daily full universe), gross AND net of cost. Success =
positive **net** per-trade expectancy with t≳3 and monotonic-ish behavior across the near/ATR knobs,
holding out-of-sample and per-year.

## 1a. Economic hypothesis (G0)

Trend-continuation / under-reaction: in an established trend (20>200 SMA stacked, 20 rising),
a shallow pullback to the moving average is where trend-followers re-load; the break of the
pause-bar's extreme is a low-risk continuation trigger. Counterparty = late shorts/profit-takers
who faded the pause and get run over as the trend resumes. Decay risk: this is one of the most
widely-taught retail setups (MA pullback) → likely arbitraged/thin edge after cost; the ATR-exit
("sell when extended") is the part most likely to add value by harvesting mean-distance reversion.
**Falsification:** if gross per-trade expectancy ≤ 0 on the deep 5-min names, or net ≤ 0 after
5 bps round-trip, or the edge is one lucky name / one year → NO EDGE, stop.

## 2. The Base — locked mechanics

Context bar = the **signal (pause) bar** `i`. All features causal (≤ bar `i`).

**LONG setup** (Picture of Power):
- Regime: `SMA20[i] > SMA200[i]` (bullish stack) AND `SMA20` rising (`SMA20[i] > SMA20[i-slope_lb]`)
  AND `close[i] > SMA200[i]`.
- Signal bar: **red after green** — `close[i] < open[i]` and `close[i-1] > open[i-1]`.
- NEAR filter: signal bar's low within ±`near_pct` band of `SMA20[i]`
  (`SMA20*(1-near_pct) ≤ low[i] ≤ SMA20*(1+near_pct)`) — i.e. the pullback touched/hugged the line.
- Trigger: within next `entry_win` bars, a bar's `high ≥ high[i]` → **stop-buy at `high[i]`**
  (gap-up fills at `open`). If `low` breaks `low[i]` before trigger → setup cancelled.
- Initial SL: a step below `low[i]`.
- Exit / take-profit (drift away): first bar where `high - SMA20 ≥ atr_mult × ATR14` → exit at
  bar **close** (conservative, end-of-bar decision). SL checked first each bar (gap-through modeled).
- Hold mode: `intraday` (force exit at session close 15:25 for 5-min) or `overnight` (hold across days).

**SHORT setup** (NARROW-TO-WIDE mirror): symmetric — `SMA20<SMA200`, `SMA20` falling, `close<SMA200`;
signal = **green after red**; trigger = break of `low[i]` → stop-sell; SL above `high[i]`;
exit when `SMA20 - low ≥ atr_mult × ATR14`. India shortability caveat applied at portfolio stage.

**Costs:** `cost_bps` round-trip on notional (default 5 bps), reported gross vs net + a
cost-sensitivity (0/5/10/15 bps). Per-trade return also expressed in **R** (risk = |entry−SL|/entry).

## 3. Plan (stage-gated)

| Stage | What | Universe | Kill criterion |
|---|---|---|---|
| **G1 probe** | baseline params, gross, both dirs | 13 deep 5-min names (2015→now) | gross expectancy ≤0 → NO EDGE |
| **G2 sweep** | near% × slope × atr_mult × entry_win × cluster, gross+net | deep 5-min | net ≤0 or non-monotonic |
| **G2b TF** | 15/30-min (resample) + daily full universe | daily 1,642 | doesn't generalize |
| **G3** | OOS/per-year, param sensitivity, super-winner guard, cost-sens | survivors | overfit/one-name/one-year |

**Baseline params (G1):** `slope_lb=3, near_pct=0.005 (0.5%), atr_len=14, atr_mult=2.5,
entry_win=3, hold=intraday(5min), cost=0 (gross)`.

**Sweep grid (G2, provisional):** near_pct∈{0.25,0.5,1.0}% · slope_lb∈{3,5} · atr_mult∈{1.5,2,2.5,3,3.5}
· entry_win∈{1,3} · cluster∈{single,last-red,cluster} · dir∈{long,short} → ~360 cells × universe.

## 4. Falsification / gates — decided up front
- Gross per-trade expectancy must be > 0 on the deep 5-min sample (else NO EDGE, stop at G1).
- Net (5 bps) expectancy > 0 AND t-stat ≳ 3 to pass G2.
- Must not depend on a single name (super-winner guard) or a single year (per-year table) at G3.

---

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-24 ~21:55 IST | Folder + STATUS written, engine built | sections 1–4 locked before launch |
| 2026-07-24 ~22:05 IST | G1 probe run on VPS (12 deep 5-min names, 2015→now) | ~64k trades, both dirs |
| 2026-07-24 ~22:06 IST | **G1 = NO EDGE** — gross expectancy ≤0 both dirs | long −0.006% / short −0.005% gross; net −0.055%; win ~28%; PF 0.56; t≈−38 |
| 2026-07-24 ~22:10 IST | G2 lever sweep launched (tf × exit × hold × slope, both dirs) | testing overnight-hold + SMA-cross exit + higher TFs before firm verdict |

## 6. Crash Recovery (resume without Claude)
- **Engine + runners:** `research/91_sma20_200_pullback/scripts/{sma_pullback_engine.py,run_g1_probe.py}`.
- **Run G1 on VPS:** `ssh arun@94.136.185.54 'cd /home/arun/quantifyd && nohup venv/bin/python3 research/91_sma20_200_pullback/scripts/run_g1_probe.py > /tmp/sma91_g1.log 2>&1 &'`
- **Monitor:** `tail -f /tmp/sma91_g1.log`; incremental output `research/91_sma20_200_pullback/results/g1_probe.csv`.
- **Resume:** runner skips names already in `g1_probe.csv` (done-set). Safe to re-run.
- **Do NOT touch:** `backtest_data/market_data.db` (canonical, being read).

## 7. Files
| File | Purpose | Committable? |
|---|---|---|
| `scripts/sma_pullback_engine.py` | Signal + sim engine (causal) | yes |
| `scripts/run_g1_probe.py` | G1 cheap probe runner | yes |
| this STATUS file | Live status + recovery | yes |
| `results/g1_probe.csv` | Per-name gross expectancy | yes (small) |
| `results/RESULTS.md` | Final verdict | yes (after G-gates) |

## 8. Findings

### G1 (baseline, 5-min deep names, 2015→now) — NO EDGE
| Dir | Trades | Gross exp/trade | Net (5bps) | Win% | PF | t-stat |
|---|---|---|---|---|---|---|
| Long | 32,813 | −0.006% | −0.056% | 28.4% | 0.56 | −39.9 |
| Short | 31,117 | −0.005% | −0.055% | 29.2% | 0.58 | −35.9 |

Even **gross** is negative → not a cost problem, the raw signal has no edge on intraday
5-min. Tight red-low stop hit constantly (win ~28%, avg hold 3.8 bars); the 2.5×ATR "drift
away" target rarely reached before the stop. Rising-20-SMA filter IS enforced
(`SMA20[i]>SMA20[i-slope_lb]`). The video's charts are survivor-selected winners.

### G2 (levers: overnight hold, SMA-cross exit, 5/15/30-min + daily, stricter slope)

No cell cleared the gate (gross>0 AND net>0 AND t≥3). Stricter "rising 20-SMA" = worse.
Overnight/cross exit barely moved intraday. Gross rises with timeframe but net ≈0 (t≈0–1.8).
Only net-positive cells = daily LONG (t 1.1–1.8, n 68–233); daily SHORT mirror loses.

### G3 (drift control — daily long) — the setup UNDERPERFORMS random
| | Gross exp/trade | t | n |
|---|---|---|---|
| SETUP (pause-break) | +0.777% | 1.78 | 220 |
| Random entry, same up-regime | +1.029% | 2.73 | 220 |
| Every bar in up-regime (same hold) | +0.895% | 24.4 | 27,245 |

The daily positive is 100% survivor drift; the pause/near mechanics subtract 0.12–0.25%/trade.
**Final verdict: NO EDGE.** See `results/RESULTS.md`.
