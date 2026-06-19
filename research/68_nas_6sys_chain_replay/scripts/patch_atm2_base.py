"""ATM2 move-stop, part 1/3: persist entry spot at entry.
- Migrate the 6 ATM-family DBs: add column entry_spot REAL (idempotent).
- Base NasAtmExecutor._place_order: accept entry_spot, store it in add_position.
- Base execute_strangle_entry: pass entry_spot=spot to both leg orders.
Additive + backward-compatible (entry_spot defaults None -> NULL for any path that
doesn't pass it). Guarded: count asserts + ast.parse + .bak.
"""
import ast, shutil, sqlite3, os

ROOT = '/home/arun/quantifyd'
DBS = ['nas_atm', 'nas_atm2', 'nas_atm4', 'nas_916_atm', 'nas_916_atm2', 'nas_916_atm4']
for d in DBS:
    p = f'{ROOT}/backtest_data/{d}_trading.db'
    if not os.path.exists(p):
        print('  MISSING', p); continue
    c = sqlite3.connect(p)
    cols = [r[1] for r in c.execute("PRAGMA table_info(nas_atm_positions)")]
    if 'entry_spot' in cols:
        print(f'  {d}: entry_spot already present')
    else:
        c.execute("ALTER TABLE nas_atm_positions ADD COLUMN entry_spot REAL")
        c.commit(); print(f'  {d}: added entry_spot column')
    c.close()

P = f'{ROOT}/services/nas_atm_executor.py'
s = open(P, encoding='utf-8').read()
if 'entry_spot=entry_spot' in s:
    print('BASE ALREADY PATCHED'); raise SystemExit

edits = [
    # 1) _place_order signature
    ("                     signal_type, strangle_id, sl_price=None):",
     "                     signal_type, strangle_id, sl_price=None, entry_spot=None):"),
    # 2) store in add_position
    ("            entry_time=now,\n            sl_price=sl_price,",
     "            entry_time=now,\n            entry_spot=entry_spot,\n            sl_price=sl_price,"),
    # 3) CE leg call
    ("            strangle_id=strangle_id,\n            sl_price=ce_sl,\n        )",
     "            strangle_id=strangle_id,\n            sl_price=ce_sl,\n            entry_spot=spot,\n        )"),
    # 4) PE leg call
    ("            strangle_id=strangle_id,\n            sl_price=pe_sl,\n        )",
     "            strangle_id=strangle_id,\n            sl_price=pe_sl,\n            entry_spot=spot,\n        )"),
]
for i, (old, new) in enumerate(edits, 1):
    assert s.count(old) == 1, 'edit %d count=%d' % (i, s.count(old))
    s = s.replace(old, new, 1)
ast.parse(s)
shutil.copy(P, P + '.bak_atm2move')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED base: _place_order stores entry_spot; execute_strangle_entry passes it')
