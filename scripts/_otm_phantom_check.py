"""Verify what actual 24750CE and 24400PE premiums were during 12:19-12:21 today."""
import sqlite3
c = sqlite3.connect('/home/arun/quantifyd/backtest_data/options_data.db')
c.row_factory = sqlite3.Row

for sym in ('NIFTY2642124750CE', 'NIFTY2642124400PE'):
    print(f'=== {sym} during 12:19-12:21 ===')
    rs = c.execute("""SELECT snapshot_time, ltp, bid, ask
                      FROM option_chain
                      WHERE tradingsymbol=? AND snapshot_time LIKE '2026-04-21T12:1%'
                      ORDER BY snapshot_time""", (sym,)).fetchall()
    for r in rs:
        d = dict(r)
        print(f"  {d['snapshot_time'][11:19]}  ltp={d['ltp']}  bid={d['bid']}  ask={d['ask']}")
    print()

# Also pull spot
print('=== NIFTY spot during 12:19-12:21 ===')
rs = c.execute("""SELECT snapshot_time, spot_price
                  FROM underlying_spot
                  WHERE symbol='NIFTY' AND snapshot_time LIKE '2026-04-21T12:1%'
                  ORDER BY snapshot_time""").fetchall()
for r in rs:
    d = dict(r)
    print(f"  {d['snapshot_time'][11:19]}  spot={d['spot_price']}")
