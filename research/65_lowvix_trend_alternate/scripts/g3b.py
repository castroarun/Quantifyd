"""research/65 G3b — INTRADAY VIX exit for the low-VIX long. Use 5-min INDIAVIX + NIFTY: hold the long
while VIX<13; exit the MOMENT a 5-min VIX bar prints >=13 (at that bar's NIFTY), vs the naive next-day
exit. Does it dodge the regime-ending spike (2026 lag)? DB 5-min, 2015-2026."""
import sqlite3, numpy as np, pandas as pd
db = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db"); db.execute("PRAGMA busy_timeout=30000")
nf = pd.read_sql("SELECT date,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", db, parse_dates=["date"]).rename(columns={"close": "nifty"})
vx = pd.read_sql("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='5minute' ORDER BY date", db, parse_dates=["date"]).rename(columns={"close": "vix"})
m = nf.merge(vx, on="date", how="inner"); m["day"] = m.date.dt.normalize()
dayclose = m.groupby("day").nifty.last(); dayvix = m.groupby("day").vix.last()
days = list(dayclose.index); prevc = dayclose.shift(1); prevv = dayvix.shift(1)
bars = {d: g for d, g in m.groupby("day")}

def run(intraday):
    long = False; pnl = {}
    for k in range(1, len(days)):
        d = days[k]; pc = prevc.iloc[k]; pv = prevv.iloc[k]
        if np.isnan(pc) or np.isnan(pv): continue
        if not long and pv < 13:
            long = True
        if long:
            b = bars[d]
            if intraday:
                hit = b[b.vix >= 13]
                if len(hit):
                    px = hit.iloc[0].nifty; pnl[d] = (px-pc)/pc; long = False     # exit intraday at the cross
                else:
                    pnl[d] = (dayclose.loc[d]-pc)/pc
            else:
                pnl[d] = (dayclose.loc[d]-pc)/pc
                if dayvix.loc[d] >= 13: long = False                              # naive: exit next-day
        # else flat -> no pnl that day
    return pd.Series(pnl)

for name, intra in [("naive daily exit", False), ("INTRADAY VIX<13 exit", True)]:
    s = run(intra); yr = s.index.year
    tot = (1+s).prod()-1; shp = s.mean()/s.std()*np.sqrt(252) if s.std() > 0 else 0
    eq = (1+s).cumprod(); dd = (eq/eq.cummax()-1).min()
    py = {y: ((1+g).prod()-1)*100 for y, g in s.groupby(yr)}
    print(f"{name:22s} days={len(s):4d} tot={tot*100:4.0f}% Sharpe={shp:.2f} maxDD={dd*100:.0f}% | " + " ".join(f"{py.get(y,0):+5.1f}" for y in range(2019, 2027)))
print("years:" + " "*45 + " ".join(f"{y:5d}"[2:] for y in range(2019, 2027)) + "  (watch 2026/2019)")
