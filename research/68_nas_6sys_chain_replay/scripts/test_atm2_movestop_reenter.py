"""Isolated test (temp db) of ATM2 v3: 0.4% move-stop that CASCADES (re-centers) to the new
ATM strike, gated so it won't re-enter the same strike.
A) 0.4% move, new ATM != old strike -> MOVE_STOP closes both + re-enters once at new strike.
B) move fires but new ATM == old strike (forced via tiny move_pct) -> close both, NO re-enter."""
import sys, shutil, datetime
sys.path.insert(0, '/home/arun/quantifyd')
shutil.copy('/home/arun/quantifyd/backtest_data/nas_atm2_trading.db', '/tmp/test_atm2_v3.db')
from services.nas_atm2_executor import NasAtm2Executor
from config import NAS_ATM2_DEFAULTS

cfg = dict(NAS_ATM2_DEFAULTS)
ex = NasAtm2Executor(config=cfg)
ex.db.db_path = '/tmp/test_atm2_v3.db'
print('cfg move_stop=%s reenter=%s re_enter_on_sl=%s' % (cfg['move_stop_pct'], cfg.get('move_stop_reenter'), cfg['re_enter_on_sl']))

calls = {'entry': 0, 'strike': None}
def _stub(spot=None, scan_result=None):
    calls['entry'] += 1; calls['strike'] = ex.scanner.get_atm_strike(spot) if spot else None
    return (999, 'stub')
ex.execute_strangle_entry = _stub


def setup(sid, strike):
    now = datetime.datetime.now().isoformat()
    for leg, it in [('CE', 'CE'), ('PE', 'PE')]:
        ex.db.add_position(strangle_id=sid, leg=leg, tradingsymbol='V3%d%s' % (sid, it), exchange='NFO',
                           transaction_type='SELL', instrument_type=it, qty=130, strike=strike,
                           expiry_date='2026-06-26', entry_price=100.0, entry_time=now,
                           entry_spot=float(strike), sl_price=9999.0, signal_type='V3', status='ACTIVE', mode='paper')


def run(sid, spot, move_pct):
    ex.cfg['move_stop_pct'] = move_pct
    ex.scanner.get_live_spot = lambda: spot
    ex.scanner.get_live_option_premium = lambda t: 90.0
    legs = [p for p in ex.db.get_active_positions() if p['strangle_id'] == sid]
    before = calls['entry']
    acts = ex.check_and_handle_sl(positions=legs, live_ltps={p['tradingsymbol']: 90.0 for p in legs})
    left = [p for p in ex.db.get_active_positions() if p['strangle_id'] == sid]
    re = any('re_entry' in a for a in acts)
    return [a['type'] for a in acts], len(left), calls['entry'] - before, re, calls['strike']


ok = True
# A: entry strike 24100, spot 24200 (+0.41%), ATM 24200 != 24100 -> re-center
setup(770001, 24100)
t, left, ent, re, nk = run(770001, 24200.0, 0.004)
pA = ('MOVE_STOP' in t and left == 0 and ent == 1 and re and int(nk) == 24200)
print('A move+new-strike (24100->spot24200->ATM24200): types=%s left=%d reenter=%d new_strike=%s -> %s'
      % (t, left, ent, nk, 'PASS' if pA else 'FAIL')); ok &= pA

# B: entry strike 24100, spot 24125, move_pct 0.001 fires but ATM round(24125/50)*50=24100==old -> no re-enter
setup(770002, 24100)
t, left, ent, re, nk = run(770002, 24125.0, 0.001)
pB = ('MOVE_STOP' in t and left == 0 and ent == 0 and not re)
print('B move+same-strike (24100->spot24125->ATM24100): types=%s left=%d reenter=%d -> %s'
      % (t, left, ent, 'PASS' if pB else 'FAIL')); ok &= pB

print('\n%s' % ('ALL PASS' if ok else '*** FAIL ***'))
sys.exit(0 if ok else 1)
