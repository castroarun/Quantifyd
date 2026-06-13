"""research/64 P10 — STOP GRANULARITY sweep: 5-min vs 15-min vs 30-min vs daily close stop.
Tradeoff: coarser stop -> fewer WHIPSAWS (false exits on reverting spikes) but worse FILL (you exit
further past 2% on real breaches). Real NIFTY 5-min from DB, 2015-2026. Underlying terms."""
import sqlite3
import numpy as np, pandas as pd
db = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")
db.execute("PRAGMA busy_timeout=30000")
m = pd.read_sql("SELECT date,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", db, parse_dates=["date"])
m["day"] = m["date"].dt.normalize()
daily = m.groupby("day").close.last()
days = list(daily.index)
by_day = {d: g["close"].values for d, g in m.groupby("day")}   # 5-min closes per day
HH = 5
def gran_closes(holddays, step):
    """concatenated g-min closes over the hold (every `step`-th 5-min close, per day)."""
    out = []
    for d in holddays:
        c = by_day[d]
        out.extend(c[step-1::step])     # every step-th close within the day
        if (len(c) % step) != 0:
            out.append(c[-1])           # always include the day's last close (daily-ish)
    return np.array(out)

grans = {"5-min": 1, "15-min": 3, "30-min": 6}
print(f"NIFTY50 5-min 2015-2026, {len(days)} sessions\n")
print(f"{'stop':9s} {'exits%':>7} {'whipsaw%':>9} {'medExit':>8} {'meanExit':>9} {'exits>3%':>9}")
res = {}
for name, step in list(grans.items()) + [("daily", None)]:
    nexit = whip = cont = 0; exitmoves = []; tot = 0
    for k in range(1, len(days)-HH):
        es = daily.iloc[k-1]; hold = days[k:k+HH]; tot += 1
        dbreach = bool((np.abs(daily.loc[hold]-es)/es >= 0.02).any())
        if name == "daily":
            # exit on first daily close beyond 2%
            dm = np.abs(daily.loc[hold].values - es)/es
            hit = np.where(dm >= 0.02)[0]
            if len(hit):
                nexit += 1; cont += 1; exitmoves.append(dm[hit[0]]*100)
            continue
        cl = gran_closes(hold, step); mv = np.abs(cl-es)/es
        hit = np.where(mv >= 0.02)[0]
        if len(hit):
            nexit += 1; exitmoves.append(mv[hit[0]]*100)
            if dbreach: cont += 1
            else: whip += 1
    em = np.array(exitmoves)
    res[name] = (nexit, whip, tot, em)
    print(f"{name:9s} {nexit/tot*100:6.0f}% {whip/tot*100:8.1f}% {np.median(em):7.2f}% {em.mean():8.2f}% {(em>3).mean()*100:8.0f}%")
print("\nINTERPRETATION: finer stop = more exits incl. more WHIPSAWS but tighter fills (~2.0-2.1%);")
print("coarser = fewer whipsaws but you ride further past 2% on real breaches; DAILY = no intraday")
print("whipsaw but you exit at the daily close (worst fill on trend/gap days). 15-min is the usual sweet spot.")
