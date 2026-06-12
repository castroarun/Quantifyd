"""Concise live snapshot of the 3 live NAS 9:16 ATM straddles + monitoring health.
Run on VPS with env sourced. Read-only."""
import logging; logging.disable(logging.CRITICAL)
import sqlite3, subprocess
from datetime import datetime
from services.kite_service import get_kite

VARIANTS = [('nas_916_atm', '916-ATM'), ('nas_916_atm2', '916-ATM2'), ('nas_916_atm4', '916-ATM4'),
            ('nas_atm', 'Sq-ATM'), ('nas_atm2', 'Sq-ATM2'), ('nas_atm4', 'Sq-ATM4')]
LOT = 65
k = get_kite()
spot = k.ltp(['NSE:NIFTY 50'])['NSE:NIFTY 50']['last_price']
now = datetime.now().strftime('%H:%M:%S')
print('=== NAS LIVE %s | spot %.1f ===' % (now, spot))

# collect all active legs, batch-quote
active = {}
for db, lbl in VARIANTS:
    c = sqlite3.connect('backtest_data/%s_trading.db' % db)
    rows = c.execute("select strangle_id,leg,strike,instrument_type,tradingsymbol,entry_price,sl_price,status from nas_atm_positions where status='ACTIVE'").fetchall()
    active[lbl] = rows

syms = set()
for lbl, rows in active.items():
    for r in rows:
        syms.add('NFO:' + r[4])
q = k.quote(list(syms)) if syms else {}

def _today():
    from datetime import datetime, timedelta
    return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d')

grand_day = 0.0
open_n = 0
for db, lbl in VARIANTS:
    rows = active[lbl]
    c = sqlite3.connect('backtest_data/%s_trading.db' % db)
    # REALIZED from legs closed today: short P&L = (entry - exit) * qty
    realized = 0.0
    for leg, entry, exitp, qty in c.execute(
            "select leg,entry_price,exit_price,qty from nas_atm_positions where status='CLOSED' and substr(updated_at,1,10)=?",
            (_today(),)):
        if exitp is not None:
            realized += (entry - exitp) * (qty or LOT)
    # UNREALIZED on active legs
    unreal = 0.0
    worst = 999
    legdesc = []
    for sid, leg, strike, itype, tsym, entry, sl, st in rows:
        ltp = q.get('NFO:' + tsym, {}).get('last_price', 0)
        unreal += (entry - ltp) * (LOT)
        room = (sl - ltp) / ltp * 100 if ltp and sl < 900000 else 999
        worst = min(worst, room)
        legdesc.append('%s %.0f n%.1f%s' % (leg, strike, ltp, (' SL%.1f(%.0f%%)' % (sl, room)) if sl < 900000 else ' ST-trail'))
    day = realized + unreal
    grand_day += day
    if rows:
        open_n += 1
        print('  %-9s sid=%s: %s | realized %+.0f unreal %+.0f -> DAY Rs %+.0f%s' % (
            lbl, rows[0][0], '  '.join(legdesc), realized, unreal, day,
            (' | nearest SL %.0f%%' % worst) if worst < 900 else ''))
    else:
        print('  %-9s flat: realized %+.0f -> DAY Rs %+.0f' % (lbl, realized, day))

print('  COMBINED open=%d  DAY P&L Rs %+.0f  (realized+unrealized)' % (open_n, grand_day))

# monitoring health
try:
    out = subprocess.run(['bash', '-c', "journalctl -u quantifyd --since '70 sec ago' 2>/dev/null | grep -c 'nas_916_sl_monitor.*executed successfully'"], capture_output=True, text=True, timeout=10)
    sl_runs = out.stdout.strip()
except Exception:
    sl_runs = '?'
import json, urllib.request
try:
    t = json.load(urllib.request.urlopen('http://127.0.0.1:5000/api/nas/ticker/status', timeout=6))
    tick = 'connected=%s running=%s ltp=%s legs=%s/%s/%s' % (t.get('is_connected'), t.get('is_running'), t.get('last_ltp'), t.get('atm_option_legs_monitored'), t.get('atm2_option_legs_monitored'), t.get('atm4_option_legs_monitored'))
except Exception as e:
    tick = 'ticker err: %s' % e
import os
flags = []
for f in ('nas_manual_freeze.flag', 'nas_kill.flag'):
    if os.path.exists('backtest_data/' + f):
        flags.append(f)
print('  MONITOR: sl_monitor ~%s/70s (expect ~7) | ticker %s | flags:%s' % (sl_runs, tick, flags or 'clear'))
