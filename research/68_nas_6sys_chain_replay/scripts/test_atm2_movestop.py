"""ATM2 move-stop, part 3/3: isolated test against a TEMP db copy (live db untouched).
Covers: (1) >=0.4% move -> MOVE_STOP closes both, no re-entry; (2) <0.4% move -> nothing
fires; (3) per-leg 30% SL backstop still fires (SL_EXIT_BOTH) with no re-entry."""
import sys, shutil, datetime, sqlite3
sys.path.insert(0, '/home/arun/quantifyd')
shutil.copy('/home/arun/quantifyd/backtest_data/nas_atm2_trading.db', '/tmp/test_atm2.db')
from services.nas_atm2_executor import NasAtm2Executor
from config import NAS_ATM2_DEFAULTS

cfg = dict(NAS_ATM2_DEFAULTS)
ex = NasAtm2Executor(config=cfg)
ex.db.db_path = '/tmp/test_atm2.db'   # redirect to temp (this process only)
print('cfg move_stop_pct=%s re_enter_on_sl=%s' % (cfg['move_stop_pct'], cfg['re_enter_on_sl']))


def setup(sid, espot, sl=9999.0):
    now = datetime.datetime.now().isoformat()
    for leg, it in [('CE', 'CE'), ('PE', 'PE')]:
        ex.db.add_position(strangle_id=sid, leg=leg, tradingsymbol='TST%d%s' % (sid, it),
                           exchange='NFO', transaction_type='SELL', instrument_type=it, qty=130,
                           strike=24000, expiry_date='2026-06-23', entry_price=100.0, entry_time=now,
                           entry_spot=espot, sl_price=sl, signal_type='TST', status='ACTIVE', mode='paper')


def run(sid, cur_spot, prem):
    ex.scanner.get_live_spot = lambda: cur_spot
    ex.scanner.get_live_option_premium = lambda t: prem
    legs = [p for p in ex.db.get_active_positions() if p['strangle_id'] == sid]
    ltps = {p['tradingsymbol']: prem for p in legs}
    acts = ex.check_and_handle_sl(positions=legs, live_ltps=ltps)
    left = [p for p in ex.db.get_active_positions() if p['strangle_id'] == sid]
    return [a['type'] for a in acts], acts, len(left)


ok = True
# TEST 1: +0.5% underlying -> MOVE_STOP, both closed, no re-entry
setup(990001, 24000.0)
t, acts, left = run(990001, 24120.0, 90.0)
re1 = any('re_entry' in a for a in acts)
p1 = ('MOVE_STOP' in t) and left == 0 and not re1
print('TEST1 +0.5%%: types=%s active_left=%d re_entry=%s -> %s' % (t, left, re1, 'PASS' if p1 else 'FAIL'))
ok &= p1

# TEST 2: +0.2% underlying, premiums benign -> nothing fires, both stay open
setup(990002, 24000.0)
t, acts, left = run(990002, 24048.0, 90.0)
p2 = ('MOVE_STOP' not in t) and left == 2
print('TEST2 +0.2%%: types=%s active_left=%d -> %s' % (t, left, 'PASS' if p2 else 'FAIL'))
ok &= p2

# TEST 3: tiny move but a leg breaches 30% SL -> per-leg backstop SL_EXIT_BOTH, no re-entry
setup(990003, 24000.0, sl=120.0)
t, acts, left = run(990003, 24010.0, 130.0)   # prem 130 >= sl 120; move 0.04% (no move-stop)
re3 = any('re_entry' in a for a in acts)
p3 = ('MOVE_STOP' not in t) and left == 0 and not re3
print('TEST3 per-leg backstop: types=%s active_left=%d re_entry=%s -> %s' % (t, left, re3, 'PASS' if p3 else 'FAIL'))
ok &= p3

print('\n%s' % ('ALL TESTS PASSED' if ok else '*** SOME TESTS FAILED ***'))
sys.exit(0 if ok else 1)
