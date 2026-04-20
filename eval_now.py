import os, sys, logging
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')

from config import ORB_DEFAULTS
from services.orb_live_engine import ORBLiveEngine
from services.orb_db import get_orb_db
from datetime import date

engine = ORBLiveEngine(ORB_DEFAULTS)
engine.initialize_day()
engine.update_or()

print('')
print('=== RUNNING EVALUATE_SIGNALS ===')
engine.evaluate_signals()

db = get_orb_db()
today = date.today().isoformat()

print('')
print('=== INDICATOR VALUES AFTER EVAL ===')
for sym in ORB_DEFAULTS['universe']:
    ds = db.get_or_create_daily_state(sym, today)
    v = ds.get('vwap')
    r = ds.get('rsi_15m')
    print('  %s VWAP=%s RSI=%s wide=%s trades=%s' % (sym, v, r, ds.get('is_wide_cpr_day'), ds.get('trades_taken')))

sigs = db.get_recent_signals(limit=30)
today_sigs = [s for s in sigs if (s.get('signal_time') or '')[:10] == today]
print('')
print('Today signals: %d' % len(today_sigs))
for s in today_sigs:
    print('  %s %s %s | %s' % (s.get('signal_time','')[11:16], s.get('instrument'), s.get('direction'), s.get('action_taken')))

positions = db.get_open_positions()
print('')
print('Open positions: %d' % len(positions))
for p in positions:
    print('  %s %s @%s SL=%s TGT=%s' % (p['instrument'], p['direction'], p['entry_price'], p['sl_price'], p['target_price']))
