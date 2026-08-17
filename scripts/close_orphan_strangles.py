#!/usr/bin/env python3
"""Square off the 5 orphaned ORB-index strangle paper positions.

The book was retired mid-session (2026-08-17): its scheduler jobs were
removed before today's 15:25 EOD squareoff had closed everything, so five
short strangles were left OPEN with no job left to close them. Mark them
closed at today's actual option prices, exit_reason BOOK_RETIRED, so the
record is complete and nothing dangles.

Paper book — no broker action involved.
"""
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, '/home/arun/quantifyd')

from services.kite_service import get_kite_with_refresh  # noqa: E402

DB = '/home/arun/quantifyd/backtest_data/strangle_trading.db'
COST_PER_LEG = 25.0          # paper charge per leg, mirrors the book's model

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT * FROM strangle_positions WHERE status='OPEN'").fetchall()
print(f'orphaned open positions: {len(rows)}')
if not rows:
    raise SystemExit(0)

kite = get_kite_with_refresh()

# Build the NFO tradingsymbols for this expiry (2026-08-18 weekly).
exp = rows[0]['expiry_date']            # 'YYYY-MM-DD'
y, m, d = exp.split('-')
# Kite weekly format: NIFTY + YY + M + DD  (M = 1-9,O,N,D)
mon_code = {'01': '1', '02': '2', '03': '3', '04': '4', '05': '5', '06': '6',
            '07': '7', '08': '8', '09': '9', '10': 'O', '11': 'N', '12': 'D'}[m]
prefix = f'NIFTY{y[2:]}{mon_code}{d}'


def ltp_for(strike, opt):
    sym = f'NFO:{prefix}{int(strike)}{opt}'
    try:
        q = kite.ltp([sym])
        return float(q[sym]['last_price']), sym
    except Exception as e:
        print(f'  ltp failed {sym}: {e}')
        return None, sym


now = datetime.now().isoformat(timespec='seconds')
today = now[:10]
total = 0.0

for r in rows:
    pe_x, pe_sym = ltp_for(r['pe_strike'], 'PE')
    ce_x, ce_sym = ltp_for(r['ce_strike'], 'CE')
    if pe_x is None or ce_x is None:
        print(f"  position {r['id']} ({r['variant_id']}): no price, skipped")
        continue
    qty = r['qty'] or r['lot_size'] or 65
    # short strangle: collected at entry, paid to close
    gross = ((r['pe_entry_price'] - pe_x) + (r['ce_entry_price'] - ce_x)) * qty
    costs = COST_PER_LEG * 2
    net = gross - costs
    total += net
    con.execute(
        "UPDATE strangle_positions SET status='CLOSED', exit_date=?, exit_ts=?, "
        "pe_exit_price=?, ce_exit_price=?, gross_pnl=?, net_pnl=?, "
        "exit_reason='BOOK_RETIRED', updated_at=? WHERE id=?",
        (today, now, pe_x, ce_x, round(gross, 2), round(net, 2), now, r['id']))
    con.execute(
        "INSERT INTO strangle_trades (position_id, variant_id, entry_date, "
        "exit_date, entry_ts, exit_ts, direction, pe_strike, ce_strike, "
        "pe_entry, ce_entry, pe_exit, ce_exit, gross_pnl, net_pnl, costs, "
        "exit_reason, hold_minutes) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (r['id'], r['variant_id'], r['entry_date'], today, r['entry_ts'], now,
         r['direction'], r['pe_strike'], r['ce_strike'],
         r['pe_entry_price'], r['ce_entry_price'], pe_x, ce_x,
         round(gross, 2), round(net, 2), costs, 'BOOK_RETIRED', None))
    print(f"  {r['variant_id']:14s} PE {r['pe_entry_price']:.2f}->{pe_x:.2f}  "
          f"CE {r['ce_entry_price']:.2f}->{ce_x:.2f}  net {net:+.0f}")

con.commit()
left = con.execute(
    "SELECT COUNT(*) FROM strangle_positions WHERE status='OPEN'").fetchone()[0]
print(f'\nclosed at book retirement, net Rs {total:+,.0f}; still open: {left}')
con.close()
