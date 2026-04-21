"""Audit today's OTM rolls: are they really 9 distinct entries at 24750CE?"""
import sqlite3
c = sqlite3.connect('/home/arun/quantifyd/backtest_data/nas_trading.db')
c.row_factory = sqlite3.Row
rs = c.execute("""SELECT id, strangle_id, leg, tradingsymbol, strike, qty,
                         entry_price, exit_price, entry_time, exit_time,
                         exit_reason, status, adjustment_count
                  FROM nas_positions
                  WHERE DATE(entry_time) = '2026-04-21'
                  ORDER BY entry_time""").fetchall()
print(f'Total OTM positions today: {len(rs)}')
print()
print(f"{'id':<4} {'sid':<4} {'leg':<3} {'symbol':<22} {'entry':>6} {'exit':>6} "
      f"{'entry_t':<10} {'exit_t':<10} {'reason':<22} {'adj':>3}")
print('-' * 105)
for r in rs:
    d = dict(r)
    et = (d['entry_time'] or '')[11:19]
    xt = (d['exit_time'] or '')[11:19] if d['exit_time'] else '-'
    print(f"{d['id']:<4} {d['strangle_id'] or '-':<4} {d['leg']:<3} {d['tradingsymbol']:<22} "
          f"{(d['entry_price'] or 0):>6.2f} {(d['exit_price'] or 0):>6.2f} "
          f"{et:<10} {xt:<10} {str(d['exit_reason'] or ''):<22} {d['adjustment_count'] or 0:>3}")

# Also show trades (which often logs ROLL events explicitly)
print()
print('=== nas_trades today (roll/adjustment events) ===')
try:
    rs = c.execute("""SELECT * FROM nas_trades
                      WHERE DATE(entry_time)='2026-04-21' OR DATE(exit_time)='2026-04-21'
                      ORDER BY COALESCE(exit_time, entry_time)""").fetchall()
    print(f'trades rows: {len(rs)}')
    for r in rs[:20]:
        d = dict(r)
        print(' ', {k: d.get(k) for k in ['id','strangle_id','exit_reason','exit_time','call_strike','put_strike']})
except Exception as e:
    print('err:', e)

# Signals / action log
print()
print('=== nas_signals today ===')
try:
    rs = c.execute("""SELECT * FROM nas_signals WHERE DATE(timestamp)='2026-04-21' ORDER BY timestamp""").fetchall()
    print(f'signals rows: {len(rs)}')
    for r in rs[:20]:
        d = dict(r)
        print(' ', d.get('timestamp','')[:19], d.get('signal_type',''), (d.get('action_taken','') or '')[:80])
except Exception as e:
    print('err:', e)
