"""research/64 P9b — calibrate the daily-touch proxy with REAL intraday CLOSES. Pull NIFTY 5-min from
Kite (recent window), and re-measure whipsaws on a true 5-min-CLOSE basis: a 5-min candle CLOSE beyond
±2% = the intraday stop fires. Compare whipsaw rate (5-min-close-breach but daily close reverts) to the
daily high/low TOUCH proxy. Caches the 5-min pull."""
import sys, json, os, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect
ak = os.environ.get("KITE_API_KEY")
try:
    import config; ak = ak or getattr(config, "KITE_API_KEY", None)
except Exception: pass
tj = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
at = tj.get("access_token") if isinstance(tj, dict) else tj
ak = ak or (tj.get("api_key") if isinstance(tj, dict) else None)
kite = KiteConnect(api_key=ak); kite.set_access_token(at)
CACHE = "/tmp/nifty_5min.pkl"
if os.path.exists(CACHE):
    bars = pd.read_pickle(CACHE); print("cached 5-min:", len(bars))
else:
    rows = []; s = dt.date(2023, 7, 1)
    while s < dt.date.today():
        e = min(s + dt.timedelta(days=90), dt.date.today())
        for _t in range(4):
            try:
                rows += kite.historical_data(256265, s, e, "5minute"); break
            except Exception as ex:
                import time; time.sleep(1.2)
        s = e + dt.timedelta(days=1)
    bars = pd.DataFrame(rows)
    bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None)
    bars = bars.drop_duplicates("date").set_index("date").sort_index()
    pd.to_pickle(bars, CACHE); print("pulled 5-min:", len(bars), bars.index[0], "->", bars.index[-1])

bars["day"] = bars.index.normalize()
daily = bars.groupby("day").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
days = list(daily.index)
HH = 5
whip_touch = whip_close = cont = clean = 0
cmoves = []
for k in range(1, len(days) - HH):
    es = daily.close.iloc[k - 1]; up = es * 1.02; dn = es * 0.98
    holddays = days[k:k + HH]
    seg = bars[(bars.day >= holddays[0]) & (bars.day <= holddays[-1])]
    touched = ((seg.high >= up) | (seg.low <= dn)).any()                      # high/low touch (proxy)
    closed_breach_5m = ((seg.close >= up) | (seg.close <= dn)).any()          # a 5-min CLOSE beyond 2%
    dclose_breach = (np.abs(daily.close.loc[holddays] - es) / es >= 0.02).any()
    if dclose_breach:
        cont += 1
        m = (np.abs(daily.close.loc[holddays] - es) / es).max() * 100; cmoves.append(m)
    elif touched:
        whip_touch += 1
        if closed_breach_5m:
            whip_close += 1
    else:
        clean += 1
tot = clean + whip_touch + cont
print(f"\nwindow {days[0].date()}→{days[-1].date()}  entries={tot}")
print(f"  clean (no touch)                : {clean} ({clean/tot*100:.0f}%)")
print(f"  CONTINUED (daily close breached): {cont} ({cont/tot*100:.0f}%)")
print(f"  WHIPSAW by TOUCH (proxy)        : {whip_touch} ({whip_touch/tot*100:.1f}%)")
print(f"  WHIPSAW by 5-min CLOSE (TRUE)   : {whip_close} ({whip_close/tot*100:.1f}%)")
if whip_touch:
    print(f"  -> of touch-whipsaws, only {whip_close/whip_touch*100:.0f}% were TRUE 5-min-close whipsaws (rest were wicks = no cost)")
