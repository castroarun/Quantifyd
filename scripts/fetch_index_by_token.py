"""Fetch real index history by Kite instrument token (names resolved from the instruments dump)."""
import sys, time, sqlite3
from datetime import date
sys.path.insert(0, "/home/arun/quantifyd")
from services.kite_service import get_kite_with_refresh

DB = "/home/arun/quantifyd/backtest_data/market_data.db"
TARGETS = [(267273, "NIFTYSMLCAP250", "NIFTY SMLCAP 250"),
           (289545, "NIFTYLARGEMID250", "NIFTY LARGEMID250"),
           (267017, "NIFTYSMLCAP100", "NIFTY SMLCAP 100"),
           (266505, "NIFTYMIDSML400", "NIFTY MIDSML 400")]
k = get_kite_with_refresh()
con = sqlite3.connect(DB)
for tok, sym, name in TARGETS:
    have = con.execute("SELECT COUNT(*) FROM market_data_unified WHERE symbol=? AND timeframe='day'",
                       (sym,)).fetchone()[0]
    if have > 500:
        print(f"{name}: already {have} rows -> skip"); continue
    rows = []
    for y in range(2011, date.today().year + 1):
        try:
            d = k.historical_data(tok, f"{y}-01-01", f"{y}-12-31", "day")
            rows += [(x["date"].date().isoformat(), x["open"], x["high"], x["low"], x["close"]) for x in d]
            time.sleep(0.35)
        except Exception as e:
            print(f"  {name} {y}: {e}")
    rows = sorted(set(rows))
    if not rows:
        print(f"{name}: NO DATA"); continue
    con.executemany(
        "INSERT OR IGNORE INTO market_data_unified(symbol,timeframe,date,open,high,low,close,volume) "
        "VALUES(?,'day',?,?,?,?,?,0)", [(sym, d, o, h, l, c) for d, o, h, l, c in rows])
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM market_data_unified WHERE symbol=? AND timeframe='day'",
                    (sym,)).fetchone()[0]
    print(f"{name} -> {sym}: {n} rows  {rows[0][0]} .. {rows[-1][0]}")
print("done")
