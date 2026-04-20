#!/usr/bin/env python3
"""Quick ORB engine test — run directly on VPS."""
import os, sys, logging
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('test')

from config import ORB_DEFAULTS
from services.orb_live_engine import ORBLiveEngine
from services.orb_db import get_orb_db
from datetime import date

engine = ORBLiveEngine(ORB_DEFAULTS)

log.info('=== INIT ===')
engine.initialize_day()

log.info('=== UPDATE OR ===')
engine.update_or()

db = get_orb_db()
today = date.today().isoformat()
log.info('=== OR STATE ===')
for sym in ORB_DEFAULTS['universe']:
    ds = db.get_or_create_daily_state(sym, today)
    oh = ds.get('or_high')
    ol = ds.get('or_low')
    fin = ds.get('or_finalized')
    wide = ds.get('is_wide_cpr_day')
    log.info(f'  {sym:12s} OR={oh}/{ol} fin={fin} wide={wide}')

log.info('=== EVAL SIGNALS ===')
engine.evaluate_signals()

sigs = db.get_recent_signals(limit=30)
today_sigs = [s for s in sigs if (s.get('signal_time') or '')[:10] == today]
log.info(f'Today signals: {len(today_sigs)}')
for s in today_sigs:
    log.info(f'  {s.get("signal_time","")[11:16]} {s.get("instrument"):12s} {s.get("direction"):5s} | {s.get("action_taken")}')

positions = db.get_open_positions()
log.info(f'Open positions: {len(positions)}')
for p in positions:
    log.info(f'  {p["instrument"]} {p["direction"]} @{p["entry_price"]} SL={p["sl_price"]} TGT={p["target_price"]}')
