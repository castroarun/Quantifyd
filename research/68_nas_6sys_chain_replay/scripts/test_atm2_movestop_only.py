import sys, shutil, datetime
sys.path.insert(0,'/home/arun/quantifyd')
shutil.copy('/home/arun/quantifyd/backtest_data/nas_atm2_trading.db','/tmp/test_ms.db')
from services.nas_atm2_executor import NasAtm2Executor
from config import NAS_ATM2_DEFAULTS
ex=NasAtm2Executor(config=dict(NAS_ATM2_DEFAULTS)); ex.db.db_path='/tmp/test_ms.db'
calls={'e':0}
ex.execute_strangle_entry=lambda spot=None,scan_result=None:(calls.__setitem__('e',calls['e']+1) or (999,'stub'))
def setup(sid,strike):
    now=datetime.datetime.now().isoformat()
    for leg,it in [('CE','CE'),('PE','PE')]:
        ex.db.add_position(strangle_id=sid,leg=leg,tradingsymbol='MS%d%s'%(sid,it),exchange='NFO',transaction_type='SELL',
            instrument_type=it,qty=130,strike=strike,expiry_date='2026-06-26',entry_price=100.0,entry_time=now,
            entry_spot=float(strike),sl_price=130.0,signal_type='MS',status='ACTIVE',mode='paper')  # realistic 30% SL=130
def run(sid,spot,prem):
    ex.scanner.get_live_spot=lambda:spot; ex.scanner.get_live_option_premium=lambda t:prem
    legs=[p for p in ex.db.get_active_positions() if p['strangle_id']==sid]; b=calls['e']
    acts=ex.check_and_handle_sl(positions=legs,live_ltps={p['tradingsymbol']:prem for p in legs})
    left=[p for p in ex.db.get_active_positions() if p['strangle_id']==sid]
    return [a['type'] for a in acts],len(left),calls['e']-b
ok=True
# A: prem 135 (>30% SL 130) but underlying +0.25% (<0.4%) -> HELD (SL must NOT pre-empt move-stop)
setup(660001,24100); t,left,e=run(660001,24160.0,135.0)
pA=(t==[] and left==2 and e==0); print('A SL-no-preempt (prem135>SL130, move0.25%%): types=%s left=%d -> %s'%(t,left,'PASS' if pA else 'FAIL')); ok&=pA
# B: underlying +0.41% -> MOVE_STOP + re-center
setup(660002,24100); t,left,e=run(660002,24200.0,135.0)
pB=('MOVE_STOP' in t and left==0 and e==1); print('B move0.41%%: types=%s left=%d reenter=%d -> %s'%(t,left,e,'PASS' if pB else 'FAIL')); ok&=pB
print('\n%s'%('ALL PASS' if ok else '*** FAIL ***')); sys.exit(0 if ok else 1)
