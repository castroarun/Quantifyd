"""Isolated test (temp db) of the ATM2 strike-change cascade gate.
A) SL hit but ATM strike unchanged -> HOLD (no close, no re-entry).
B) SL hit and ATM strike changed   -> cascade (close BOTH + re-enter once)."""
import sys, shutil, datetime
sys.path.insert(0, '/home/arun/quantifyd')
shutil.copy('/home/arun/quantifyd/backtest_data/nas_atm2_trading.db', '/tmp/test_atm2_sg.db')
from services.nas_atm2_executor import NasAtm2Executor
from config import NAS_ATM2_DEFAULTS

cfg = dict(NAS_ATM2_DEFAULTS)
ex = NasAtm2Executor(config=cfg)
ex.db.db_path = '/tmp/test_atm2_sg.db'
print('cfg re_enter=%s move_stop=%s strike_gate=%s interval=%s'
      % (cfg['re_enter_on_sl'], cfg['move_stop_pct'], cfg.get('cascade_require_strike_change'),
         cfg.get('strike_interval', 50)))

calls = {'entry': 0}
ex.execute_strangle_entry = lambda spot=None, scan_result=None: (calls.__setitem__('entry', calls['entry'] + 1) or (999, 'stub'))


def setup(sid, strike, sl):
    now = datetime.datetime.now().isoformat()
    for leg, it in [('CE', 'CE'), ('PE', 'PE')]:
        ex.db.add_position(strangle_id=sid, leg=leg, tradingsymbol='SG%d%s' % (sid, it), exchange='NFO',
                           transaction_type='SELL', instrument_type=it, qty=130, strike=strike,
                           expiry_date='2026-06-26', entry_price=100.0, entry_time=now,
                           entry_spot=float(strike), sl_price=sl, signal_type='SG', status='ACTIVE', mode='paper')


def run(sid, spot, prem):
    ex.scanner.get_live_spot = lambda: spot
    ex.scanner.get_live_option_premium = lambda t: prem
    legs = [p for p in ex.db.get_active_positions() if p['strangle_id'] == sid]
    before = calls['entry']
    acts = ex.check_and_handle_sl(positions=legs, live_ltps={p['tradingsymbol']: prem for p in legs})
    left = [p for p in ex.db.get_active_positions() if p['strangle_id'] == sid]
    return [a['type'] for a in acts], len(left), calls['entry'] - before


ok = True
setup(880001, 24100, sl=120.0)
t, left, ent = run(880001, 24110.0, 130.0)   # SL hit (130>=120); ATM round(24110/50)*50=24100 == strike -> HOLD
pA = (t == [] and left == 2 and ent == 0)
print('TEST A ATM unchanged (spot24110->ATM24100==24100): actions=%s left=%d reentry=%d -> %s'
      % (t, left, ent, 'PASS' if pA else 'FAIL')); ok &= pA

setup(880002, 24100, sl=120.0)
t, left, ent = run(880002, 24160.0, 130.0)   # SL hit; ATM 24150 != 24100 -> cascade
pB = ('SL_EXIT_BOTH' in t and left == 0 and ent == 1)
print('TEST B ATM changed   (spot24160->ATM24150!=24100): actions=%s left=%d reentry=%d -> %s'
      % (t, left, ent, 'PASS' if pB else 'FAIL')); ok &= pB

print('\n%s' % ('ALL PASS' if ok else '*** FAIL ***'))
sys.exit(0 if ok else 1)
