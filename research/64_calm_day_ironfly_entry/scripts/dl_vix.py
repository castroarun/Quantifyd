import sqlite3, json, os, datetime as dt, time, sys
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
db = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")
# volume sanity on NIFTY50 5min
nz = db.execute("SELECT COUNT(*) FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' AND volume>0").fetchone()[0]
print("NIFTY50 5min rows with volume>0:", nz, "(0 = index has no volume, as expected)", flush=True)
VIXT = 264969
def grab(interval, start):
    s = start; ins = 0
    while s < dt.date.today():
        e = min(s + dt.timedelta(days=(90 if interval!="day" else 1500)), dt.date.today())
        rows = []
        for _t in range(5):
            try: rows = kite.historical_data(VIXT, s, e, interval); break
            except Exception: time.sleep(1.5)
        data = [('INDIAVIX', interval, r['date'].strftime('%Y-%m-%d %H:%M:%S' if interval!='day' else '%Y-%m-%d'),
                 r['open'], r['high'], r['low'], r['close'], 0) for r in rows]
        if data:
            db.executemany("INSERT OR IGNORE INTO market_data_unified (symbol,timeframe,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)", data)
            db.commit(); ins += len(data)
        s = e + dt.timedelta(days=1); time.sleep(0.35)
    return ins
print("downloading INDIAVIX day ...", flush=True); print("  +", grab("day", dt.date(2015,1,1)), flush=True)
print("downloading INDIAVIX 5minute ...", flush=True); print("  +", grab("5minute", dt.date(2015,2,2)), flush=True)
for tf in ("day","5minute"):
    r = db.execute("SELECT COUNT(*),MIN(date),MAX(date) FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe=?", (tf,)).fetchone()
    print(f"INDIAVIX {tf}: {r}", flush=True)
