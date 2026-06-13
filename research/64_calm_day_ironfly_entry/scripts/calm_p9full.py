"""research/64 P9 (full) — daily-close breach vs TRUE intraday 5-min-CLOSE stop, over the FULL period
(2015-2026) using the now-complete NIFTY50 5-min in the DB. No Kite. Definitive whipsaw calibration."""
import sqlite3
import numpy as np, pandas as pd
db = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")
db.execute("PRAGMA busy_timeout=30000")
m = pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", db, parse_dates=["date"])
m["day"] = m["date"].dt.normalize()
daily = m.groupby("day").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
days = list(daily.index)
print(f"NIFTY50 5-min: {len(m):,} bars, {m.date.iloc[0].date()} → {m.date.iloc[-1].date()}; {len(days)} sessions")
# pre-index 5-min by day for speed
by_day = {d: g for d, g in m.groupby("day")}
HH = 5
clean = whip_touch = whip_close = cont = 0; overrun = []
for k in range(1, len(days)-HH):
    es = daily.close.iloc[k-1]; up = es*1.02; dn = es*0.98
    hold = days[k:k+HH]
    seg = pd.concat([by_day[d] for d in hold])
    touched = bool(((seg.high >= up) | (seg.low <= dn)).any())
    closed5 = bool(((seg.close >= up) | (seg.close <= dn)).any())     # TRUE 5-min CLOSE breach
    dbreach = bool((np.abs(daily.close.loc[hold]-es)/es >= 0.02).any())
    if dbreach:
        cont += 1; overrun.append((np.abs(daily.close.loc[hold]-es)/es).max()*100)
    elif touched:
        whip_touch += 1
        if closed5:
            whip_close += 1
    else:
        clean += 1
tot = clean+whip_touch+cont
print(f"\n=== FULL PERIOD ({days[0].date()}→{days[-1].date()}), entries={tot} ===")
print(f"  clean (never touch ±2%)            : {clean}  ({clean/tot*100:.0f}%)")
print(f"  CONTINUED (daily close breached)   : {cont}  ({cont/tot*100:.0f}%)  | over-run beyond 2%: median {np.median(overrun)-2:.2f}pp")
print(f"  WHIPSAW by TOUCH (high/low proxy)  : {whip_touch}  ({whip_touch/tot*100:.1f}%)")
print(f"  WHIPSAW by TRUE 5-min CLOSE        : {whip_close}  ({whip_close/tot*100:.1f}%)")
if whip_touch:
    print(f"  → {whip_close/whip_touch*100:.0f}% of touch-whipsaws were TRUE 5-min-close whipsaws (rest = wicks, no cost)")
print(f"\n  INTERPRETATION: the intraday stop genuinely whipsaws ~{whip_close/tot*100:.0f}% of entries (locks a loss the")
print(f"  daily-close rule would have kept). ~1-min resolution would be a touch higher. Net ₹ still +ve per the")
print(f"  AlgoTest real-premium run (tail-capping > whipsaw); a less-twitchy stop is the lever to cut whipsaws.")
