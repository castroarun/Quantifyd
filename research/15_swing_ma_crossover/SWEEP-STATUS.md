# MA-Crossover Sweep — Live Status

**Started:** 2026-04-24, post-close
**Goal:** Find a viable intraday MA-crossover system on 5/10/15/30-min signal
timeframes, with higher-timeframe (60-min / daily) confirmation filters, as a
second intraday system alongside live ORB.

## Plan

### Signal timeframes (4)
5-min · 10-min · 15-min · 30-min

### Filter variants (8) per timeframe
1. `baseline` — raw EMA cross only
2. `vwap` — price on correct side of intraday VWAP
3. `htf` — 60-min EMA(21) slope same direction
4. `rsi50` — RSI(14) > 50 long / < 50 short
5. `cpr` — close vs previous day's pivot
6. `bb` — above/below BB(20,2) middle
7. `confluence` — VWAP + HTF + RSI50 stacked
8. `fast_9_21` — same filters off, just faster EMA pair for whipsaw test

Total configs: 4 TFs × 8 variants = **32 backtests**.

### Universe (15 stocks — non-ORB, liquid F&O)
INDUSINDBK, PNB, BANKBARODA, WIPRO, HCLTECH, MARUTI, HEROMOTOCO, EICHERMOT,
TITAN, ASIANPAINT, JSWSTEEL, HINDALCO, JINDALSTEL, LT, ADANIPORTS

### Period & sizing
2024-03-18 → 2026-03-12 (~500 trading days, 5-min bars)
Rs 3L capital, Rs 2,500 risk/trade, 0.15% round-trip costs.

### EMA periods per TF (intraday scaling)
- 5-min:  20/50 (100m / 250m lookback)
- 10-min: 10/30 (100m / 300m)
- 15-min:  8/21 (120m / 315m)
- 30-min:  5/13 (150m / 390m — spans most of trading day)

---

## Status

### TF 5-min
- **Status:** ✅ DONE (533.9s)
- **Output:** `results/tf_5min/summary.csv` (8 variants written)

### TF 10-min
- **Status:** ✅ DONE (498.5s)
- **Output:** `results/tf_10min/summary.csv`

### TF 15-min
- **Status:** ✅ DONE (488.5s)
- **Output:** `results/tf_15min/summary.csv`

### TF 30-min
- **Status:** ✅ DONE (480.1s)
- **Output:** `results/tf_30min/summary.csv`

### Aggregated
- **`results/summary_all_tfs.csv`** — 32 rows (TF × variant)
- **`FINDINGS.md`** — all cells negative, system DOA. Best PF 0.55 (10m vwap),
  need >1.0. No rescue from per-stock curation either: 0/15 profitable on
  best variant.

---

## Crash recovery — full instructions for the human

**If the Claude session crashes, here's everything you need to continue alone:**

### 1. Check which TFs finished

```bash
for tf in 5min 10min 15min 30min; do
  echo "--- tf_$tf ---"
  tail -3 research/15_swing_ma_crossover/results/tf_$tf/progress.txt 2>/dev/null
done
```

A completed TF will have `DONE in Xs` on the last line. Otherwise it's either
mid-run or killed.

### 2. Check if background processes are still running

The 4 backtest processes were launched via Claude's background bash with IDs:
- `b3n1m8n1j` — TF 5-min
- `b57m7kv5g` — TF 10-min
- `b4wwtebrp` — TF 15-min
- `bsticql6a` — TF 30-min

If Claude crashed, these Python processes may still be running. Check with:
```bash
# From Git Bash / WSL
ps -ef | grep run_intraday_ma_crossover
# Or Windows:
tasklist | findstr python
```

Let them finish if they're alive — each writes `summary.csv` incrementally
per stock, so partial progress is preserved.

### 3. Restart any missing/killed TF

All 4 runs are independent. To resume:
```bash
cd c:/Users/Castro/Documents/Projects/Covered_Calls
python research/15_swing_ma_crossover/scripts/run_intraday_ma_crossover.py --tf 5min
python research/15_swing_ma_crossover/scripts/run_intraday_ma_crossover.py --tf 10min
python research/15_swing_ma_crossover/scripts/run_intraday_ma_crossover.py --tf 15min
python research/15_swing_ma_crossover/scripts/run_intraday_ma_crossover.py --tf 30min
```

Each run overwrites its own `tf_{xxmin}/` folder cleanly, so no stale data concern.

### 4. Aggregate when all 4 are done

Quick one-liner to merge into a single comparison CSV:
```bash
python -c "
import csv, pathlib
out = pathlib.Path('research/15_swing_ma_crossover/results')
rows = [['tf'] + ['variant','trades','wins','losses','win_rate_pct','avg_win','avg_loss','profit_factor','net_pnl','gross_pnl','costs','days_traded','cagr_pct','sharpe','max_dd','max_dd_pct','calmar']]
for tf in ['5min','10min','15min','30min']:
    p = out / f'tf_{tf}' / 'summary.csv'
    if p.exists():
        for r in list(csv.reader(p.open()))[1:]:
            rows.append([tf] + r)
with (out / 'summary_all_tfs.csv').open('w', newline='') as f:
    csv.writer(f).writerows(rows)
print(f'Wrote {len(rows)-1} rows to summary_all_tfs.csv')
"
```

### 5. Rank viable cells

```bash
# After aggregation — sort by Sharpe to find what actually works
python -c "
import csv
rows = list(csv.DictReader(open('research/15_swing_ma_crossover/results/summary_all_tfs.csv')))
viable = sorted([r for r in rows if float(r['profit_factor']) > 1.0], key=lambda r: -float(r['sharpe']))
for r in viable[:10]:
    print(f\"{r['tf']}/{r['variant']:12s} PF={r['profit_factor']} Sharpe={r['sharpe']} Net=Rs{float(r['net_pnl']):+,.0f} MaxDD={r['max_dd_pct']}%\")"
```

### 6. Files / artifacts written

For each TF `xxmin`:
- `results/tf_xxmin/progress.txt` — heartbeat, shows which stock last completed
- `results/tf_xxmin/summary.csv` — 8 rows (one per variant), updated per-stock
- `results/tf_xxmin/trades.csv` — per-trade log (written at end only)
- `results/tf_xxmin/daily_pnl.csv` — daily P&L series (written at end only)
- `logs/run_xxmin.log` — full stdout of the run

Other files that should NOT be modified:
- `scripts/run_intraday_ma_crossover.py` — the backtest engine
- `SWEEP-STATUS.md` — this file

---

## Final aggregation

When all 4 TFs complete:
- `results/summary_all_tfs.csv` — 32 rows, one per (TF × variant)
- Rank by Sharpe, Calmar, PF — identify viable cells
- If any cell clears PF > 1.3 with Sharpe > 1.0: proceed to OOS + paper trade plan
- If none do: document findings, pivot to next idea
