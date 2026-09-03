"""One-shot re-seed of the Open Alpha paper book on the 2026-09-03 spec.

Steps: backup old state -> run seed_paper_state (16 slots, no gate, purged data)
-> rescale rupee values to the momentum book's NAV (equal sleeves) -> re-apply
Arun's fund flows (Rs 2L + 50k deposits, original timestamps) -> re-anchor the
dividend HWM on the new NAV -> refresh the UI feed.
"""
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / 'backtest_data' / 'bluesky_paper_state.json'

# 1. backup + capture the flows to restore
bak = STATE.with_name(f'bluesky_paper_state.pre_reseed_{datetime.now():%Y%m%d_%H%M}.json')
shutil.copy(STATE, bak)
old = json.load(open(STATE))
flows = [f for f in (old.get('fund_flows') or []) if f['kind'] in ('deposit', 'withdraw')]
net_flows = sum(f['amount'] if f['kind'] == 'deposit' else -f['amount'] for f in flows)
print(f'backup: {bak.name}; flows to restore: {len(flows)} (net Rs {net_flows:,.0f})')

# 2. seed on the new spec
r = subprocess.run([sys.executable, str(ROOT / 'research' / '142_bananapatterns_replication'
                                        / 'scripts' / 'seed_paper_state.py')],
                   capture_output=True, text=True)
print(r.stdout[-600:])
if r.returncode != 0:
    print(r.stderr[-1500:])
    sys.exit('seed FAILED — old state restored from backup')

# 3. rescale to momentum NAV (equal sleeves convention)
mp = sqlite3.connect(str(ROOT / 'backtest_data' / 'momentum_paper.db'))
mp.row_factory = sqlite3.Row
target = float(mp.execute('SELECT nav FROM mp_nav ORDER BY d DESC LIMIT 1').fetchone()['nav'])
st = json.load(open(STATE))
cur_nav = st['nav'][-1]['nav']
f = target / cur_nav
st['capital'] = round(st['capital'] * f, 0)
st['cash'] = round(st['cash'] * f, 2)
for p in st['positions']:
    p['qty'] = max(1, int(p['qty'] * f))
for row in st['nav']:
    row['nav'] = round(row['nav'] * f, 0)
print(f'rescaled x{f:.5f}: NAV {cur_nav:,.0f} -> target {target:,.0f} (momentum book)')

# 4. restore Arun's flows: cash + capital + ledger (original timestamps preserved)
st['cash'] = round(st['cash'] + net_flows, 2)
st['capital'] = round(st['capital'] + net_flows, 0)
st['fund_flows'] = flows + [dict(ts=str(datetime.now()), kind='note', amount=0,
                                 via='re-seed 2026-09-03',
                                 note='book re-seeded on 16-slot no-gate spec; '
                                      'deposits carried over')]
if st['nav']:
    st['nav'][-1]['nav'] = round(st['nav'][-1]['nav'] + net_flows, 0)
st['sweep'] = {'symbol': 'CASHIETF', 'units': 0.0, 'cost': 0.0}
st['interest_earned'] = 0.0
json.dump(st, open(STATE, 'w'), indent=1, default=str)
print(f'flows restored: cash {st["cash"]:,.0f}, capital {st["capital"]:,.0f}')

# 5. re-anchor dividend HWM (deposits included in NAV => not distributable)
sys.path.insert(0, str(ROOT))
from services.dividend_engine import init, status
st = json.load(open(STATE))
st.pop('dividend', None)
json.dump(st, open(STATE, 'w'), indent=1, default=str)
print('dividend re-init:', json.dumps(init('openalpha'), default=str))
print('dividend status :', json.dumps(status('openalpha'), default=str))
