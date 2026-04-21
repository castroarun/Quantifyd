"""Dump today's NAS Squeeze OTM positions with timings + actions."""
import sqlite3, sys

db = sys.argv[1] if len(sys.argv) > 1 else '/home/arun/quantifyd/backtest_data/nas_trading.db'
c = sqlite3.connect(db)
c.row_factory = sqlite3.Row
r = c.execute("""SELECT id, leg, tradingsymbol, strike, qty, entry_price, exit_price,
                        entry_time, exit_time, exit_reason, adjustment_count, status
                 FROM nas_positions
                 WHERE DATE(entry_time)=?
                 ORDER BY entry_time""",
              ('2026-04-21',)).fetchall()
print(f'Total positions today: {len(r)}')
print('id   leg symbol                  qty   entry   exit   entry_t    exit_t     reason                   adj')
print('-' * 110)
for row in r:
    et = (row['entry_time'] or '')[11:19]
    xt = (row['exit_time'] or '')[11:19] if row['exit_time'] else '-'
    reason = str(row['exit_reason'] or '')
    print(f'{row["id"]:<4} {row["leg"]:<3} {row["tradingsymbol"]:<22} {row["qty"]:>5} {row["entry_price"]:>6.2f} '
          f'{(row["exit_price"] or 0):>6.2f} {et:<10} {xt:<10} {reason:<24} {row["adjustment_count"] or 0:>3}')

# Also peek at nas_trades (child table for adjustments)
print()
print('=== nas_trades on 2026-04-21 ===')
try:
    r = c.execute("""SELECT * FROM nas_trades WHERE DATE(entry_time)=? OR DATE(exit_time)=?
                     ORDER BY COALESCE(exit_time, entry_time)""",
                  ('2026-04-21', '2026-04-21')).fetchall()
    print(f'Trades rows: {len(r)}')
    for row in list(r)[:5]:
        print(dict(row))
except Exception as e:
    print('trades err:', e)
