"""Backfill NIFTY50 5-minute history into the DB (market_data_unified) for the period BEFORE the
existing 2024-03-01 start, from Kite. VPS-only (canonical host). Non-overlapping range -> no dup risk."""
import sqlite3, json, os, datetime as dt, time
import sys; sys.path.insert(0, "/home/arun/quantifyd")
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
existing_min = db.execute("SELECT MIN(date) FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute'").fetchone()[0]
print("existing NIFTY50 5min min date:", existing_min, flush=True)
END = dt.date(2024, 2, 29)            # stop just before the existing 2024-03-01
s = dt.date(2015, 2, 2); ins = 0
while s <= END:
    e = min(s + dt.timedelta(days=90), END)
    rows = []
    for _t in range(5):
        try:
            rows = kite.historical_data(256265, s, e, "5minute"); break
        except Exception as ex:
            time.sleep(1.5)
    data = [('NIFTY50', '5minute', r['date'].strftime('%Y-%m-%d %H:%M:%S'),
             r['open'], r['high'], r['low'], r['close'], r.get('volume', 0)) for r in rows]
    if data:
        db.executemany("INSERT OR IGNORE INTO market_data_unified (symbol,timeframe,date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?,?)", data)
        db.commit(); ins += len(data)
    print(s, "->", e, "+", len(data), "| total", ins, flush=True)
    s = e + dt.timedelta(days=1); time.sleep(0.4)
r = db.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute'").fetchone()
print("DONE. inserted", ins, "| NIFTY50 5min now:", r, flush=True)
