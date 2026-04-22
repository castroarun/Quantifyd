"""Insert today's 3 orphaned Kite positions into orb_positions + update daily_state.

Background: catchup_missed_breakouts() placed Kite orders for M&M/HAL/APOLLOHOSP
but never called db.add_position(), leaving the positions untracked.
monitor_positions() uses in-memory _positions only, so without this repair
no SL/target monitoring would fire.
"""
from services.orb_db import get_orb_db
from datetime import datetime

db = get_orb_db()
today = '2026-04-22'
now_iso = datetime.now().isoformat()

repairs = [
    dict(sym='M&M',        direction='SHORT', qty=6, entry=3208.00, sl=3249.90, tgt=3145.15, orh=3249.9, orl=3212.0, gap=-1.0224, cpr_tc=3243.92, cpr_bc=3237.15, cpr_w=0.2088, kite='260422150821604'),
    dict(sym='HAL',        direction='LONG',  qty=4, entry=4408.80, sl=4331.00, tgt=4521.50, orh=4391.9, orl=4331.0, gap=-0.1927, cpr_tc=4362.13, cpr_bc=4369.6,  cpr_w=0.171,  kite='260422150821672'),
    dict(sym='APOLLOHOSP', direction='LONG',  qty=2, entry=7768.25, sl=7697.00, tgt=7873.25, orh=7752.0, orl=7697.0, gap=-0.5424, cpr_tc=7734.75, cpr_bc=7716.25, cpr_w=0.2395, kite='260422150821739'),
]
for r in repairs:
    try:
        pid = db.add_position(
            instrument=r['sym'], trade_date=today, direction=r['direction'], qty=r['qty'],
            entry_price=r['entry'], entry_time=now_iso,
            sl_price=r['sl'], target_price=r['tgt'],
            or_high=r['orh'], or_low=r['orl'],
            kite_entry_order_id=r['kite'], status='OPEN',
            gap_pct=r['gap'], cpr_tc=r['cpr_tc'], cpr_bc=r['cpr_bc'],
            cpr_width_pct=r['cpr_w'],
            notes='REPAIRED from orphaned catchup entry'
        )
        sym = r['sym']
        print(f'  added {sym} pos_id={pid}')
        db.update_daily_state(sym, today, trades_taken=1)
    except Exception as e:
        print(f"  ERR {r['sym']}: {e}")

print()
print('=== Verify ===')
for p in db.get_open_positions():
    print(' ', p['instrument'], p['direction'], 'qty=', p['qty'], 'entry=', p['entry_price'], 'sl=', p['sl_price'], 'tgt=', p['target_price'], 'status=', p['status'])
