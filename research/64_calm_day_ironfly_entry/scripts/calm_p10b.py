"""research/64 P10b — the REAL whipsaw lever: BUFFER vs CONFIRMATION on the 5-min stop.
Variants: exit at first 5-min close beyond {2.0, 2.2, 2.5}% (buffer); or first time {2,3} CONSECUTIVE
5-min closes are beyond 2.0% (confirmation). Metrics: whipsaw% (fired but daily close stayed inside →
false exit), missed% (daily breached but the variant never fired → rode a real breach), exit fill.
Real NIFTY 5-min from DB, 2015-2026."""
import sqlite3
import numpy as np, pandas as pd
db = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")
db.execute("PRAGMA busy_timeout=30000")
m = pd.read_sql("SELECT date,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", db, parse_dates=["date"])
m["day"] = m["date"].dt.normalize()
daily = m.groupby("day").close.last()
days = list(daily.index)
by_day = {d: g["close"].values for d, g in m.groupby("day")}
HH = 5

def first_buffer(mv, thr):
    h = np.where(mv >= thr)[0]
    return (h[0], mv[h[0]]) if len(h) else (None, None)

def first_confirm(mv, thr, k):
    over = mv >= thr
    run = 0
    for idx, o in enumerate(over):
        run = run + 1 if o else 0
        if run >= k:
            return idx, mv[idx]
    return None, None

variants = [("buffer 2.0%", "b", 2.0), ("buffer 2.2%", "b", 2.2), ("buffer 2.5%", "b", 2.5),
            ("confirm 2×2.0%", "c", 2), ("confirm 3×2.0%", "c", 3)]
agg = {v[0]: dict(fire=0, whip=0, miss=0, fills=[]) for v in variants}
tot = 0
for k in range(1, len(days)-HH):
    es = daily.iloc[k-1]; hold = days[k:k+HH]; tot += 1
    cl = np.concatenate([by_day[d] for d in hold]); mv = np.abs(cl-es)/es*100
    dbreach = bool((np.abs(daily.loc[hold]-es)/es >= 0.02).any())
    for name, kind, p in variants:
        idx, fillmv = (first_buffer(mv, p) if kind == "b" else first_confirm(mv, 2.0, p))
        fired = idx is not None
        a = agg[name]
        if fired:
            a["fire"] += 1; a["fills"].append(fillmv)
            if not dbreach: a["whip"] += 1
        else:
            if dbreach: a["miss"] += 1
print(f"NIFTY50 5-min 2015-2026, entries={tot}\n")
print(f"{'variant':16s} {'fire%':>6} {'whipsaw%':>9} {'missed%':>8} {'medFill':>8}")
for name, _, _ in variants:
    a = agg[name]; f = np.array(a["fills"])
    print(f"{name:16s} {a['fire']/tot*100:5.0f}% {a['whip']/tot*100:8.1f}% {a['miss']/tot*100:7.1f}% {np.median(f):7.2f}%")
print("""
whipsaw% = false exit (fired but day closed back inside 2%); missed% = a real daily breach the variant
did NOT stop (rode it, fly to max-loss/wing). Want LOW whipsaw AND LOW missed with a sane fill. Confirmation
filters single-bar spikes (cuts whipsaw, keeps misses ~0); a wide buffer cuts whipsaw but MISSES real breaches.""")
