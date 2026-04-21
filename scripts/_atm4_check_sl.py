"""Inspect today's 9:16 ATM V4 positions — why didn't CE SL fire at 1.3x?"""
import sqlite3
c = sqlite3.connect('/home/arun/quantifyd/backtest_data/nas_916_atm4_trading.db')
c.row_factory = sqlite3.Row
rs = c.execute("""SELECT id, leg, tradingsymbol, entry_price, sl_price,
                         exit_price, entry_time, exit_time, exit_reason,
                         status, adjustment_count, strangle_id
                  FROM nas_atm_positions
                  WHERE DATE(entry_time)='2026-04-21'
                  ORDER BY id""").fetchall()
print(f'Rows: {len(rs)}')
print('id  leg  symbol                  entry    sl      exit    reason          adj  status   entry_t    exit_t')
print('-' * 120)
for r in rs:
    d = dict(r)
    et = (d['entry_time'] or '')[11:19]
    xt = (d['exit_time'] or '')[11:19] if d['exit_time'] else '-'
    print(f'{d["id"]:<3} {d["leg"]:<3}  {d["tradingsymbol"]:<22} {(d["entry_price"] or 0):>6.2f}  '
          f'{(d["sl_price"] or 0):>6.2f}  {(d["exit_price"] or 0):>6.2f}  '
          f'{str(d["exit_reason"] or ""):<14}  {d["adjustment_count"] or 0:>3}  {d["status"]:<8} '
          f'{et:<9} {xt}')

# Also look at signals/adjustments for strangle
print()
print('=== signals today ===')
rs = c.execute("""SELECT * FROM nas_atm_signals WHERE DATE(signal_time)='2026-04-21' ORDER BY signal_time""").fetchall()
for r in rs:
    d = dict(r)
    print(f'  {d.get("signal_time","")[:19]}  {d.get("signal_type","")} {d.get("action_taken","")[:80]}')
