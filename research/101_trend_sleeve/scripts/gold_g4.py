# -*- coding: utf-8 -*-
"""GOLD weekly-trend G4 — proper study. Weekly Donchian on GOLDBEES (gold-futures proxy), net of
0.1%/side slippage, compounded equity. Long/short vs long-only, N-robustness, per-year, walk-forward."""
import sqlite3, math, datetime as dt
from collections import defaultdict
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
SLIP = 0.001
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=20000")
def weekly(frm="2015-06-01"):
    b = c.execute("SELECT date,high,low,close FROM market_data_unified WHERE symbol='GOLDBEES' AND timeframe='day' AND close>0 AND date>=? ORDER BY date", (frm,)).fetchall()
    wk = {}
    for d, h, l, cl in b:
        y, w, _ = dt.date.fromisoformat(d).isocalendar(); k = (y, w)
        if k not in wk: wk[k] = [d, h, l, cl]
        else: wk[k][1] = max(wk[k][1], h); wk[k][2] = min(wk[k][2], l); wk[k][3] = cl
    return [wk[k] for k in sorted(wk)]
def run(b, N, longonly):
    rows = []; pos = 0
    for i in range(N, len(b) - 1):
        hh = max(x[1] for x in b[i-N:i]); ll = min(x[2] for x in b[i-N:i]); cl = b[i][3]
        newpos = pos
        if cl > hh: newpos = 1
        elif cl < ll: newpos = 0 if longonly else -1
        turnover = abs(newpos - pos); pos = newpos
        gret = b[i+1][3] / b[i][3] - 1
        rows.append((b[i+1][0], pos * gret - turnover * SLIP, turnover))
    return rows
def stats(rows):
    eq = 1.0; curve = []; peak = 1.0; mdd = 0; wins = 0; act = 0; trades = 0
    for d, r, tv in rows:
        eq *= (1 + r); curve.append((d, eq)); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
        if r != 0: act += 1; wins += (r > 0)
        trades += (tv > 0)
    n = len(rows); yrs = n / 52
    cagr = eq ** (1 / yrs) - 1
    rr = [r for _, r, _ in rows]; mu = sum(rr) / n; sd = (sum((x - mu) ** 2 for x in rr) / n) ** 0.5
    sharpe = mu / sd * math.sqrt(52) if sd else 0
    return dict(cagr=cagr, sharpe=sharpe, mdd=mdd, calmar=cagr / abs(mdd) if mdd else 0,
                win=wins / act if act else 0, trades_yr=trades / yrs, eq=eq, yrs=yrs, curve=curve, rows=rows)

b = weekly()
print("GOLD weekly Donchian — net of 0.1%/side, compounded (GOLDBEES 2015-2026)\n")
print(f"{'config':>22}  {'CAGR':>6} {'Sharpe':>7} {'maxDD':>7} {'Calmar':>7} {'win':>5} {'trd/yr':>6}")
best = None
for N in [8, 10, 13, 20]:
    for lo in [True, False]:
        s = stats(run(b, N, lo))
        tag = f"Donchian-{N} {'long-only' if lo else 'L/S'}"
        print(f"{tag:>22}  {s['cagr']*100:>5.1f}% {s['sharpe']:>7.2f} {s['mdd']*100:>6.0f}% {s['calmar']:>7.2f} {s['win']*100:>4.0f}% {s['trades_yr']:>6.1f}")
        if best is None or s['sharpe'] > best[0]: best = (s['sharpe'], N, lo, s)
sh, N, lo, s = best
print(f"\nBEST: Donchian-{N} {'long-only' if lo else 'L/S'} — CAGR {s['cagr']*100:.1f}%, Sharpe {s['sharpe']:.2f}, Calmar {s['calmar']:.2f}, maxDD {s['mdd']*100:.0f}%")
yr = defaultdict(float); eqy = {}
for d, r, tv in s['rows']: yr[d[:4]] += r
print("per-year (approx sum of weekly net returns):", " ".join(f"{y}:{yr[y]*100:+.0f}%" for y in sorted(yr)))
# walk-forward
tr = [x for x in b if x[0] < "2021-06-01"]; te = [x for x in b if x[0] >= "2021-01-01"]
bN = max([8, 10, 13, 20], key=lambda n: stats(run(tr, n, lo))['sharpe'])
oos = stats([x for x in run(te, bN, lo) if x[0] >= "2021-06-01"])
print(f"walk-forward: train picks N={bN}; OOS(2021H2-2026) CAGR {oos['cagr']*100:.1f}%, Sharpe {oos['sharpe']:.2f}, maxDD {oos['mdd']*100:.0f}%")
import json
json.dump([[d, round(e, 4)] for d, e in s['curve']], open("/tmp/gold_curve.json", "w"))
print("saved equity curve -> /tmp/gold_curve.json")
c.close()
