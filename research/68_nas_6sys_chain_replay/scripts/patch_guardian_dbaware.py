import ast, shutil
P='/home/arun/quantifyd/scripts/nas_live_guardian.py'
s=open(P,encoding='utf-8').read()
if 'db_active' in s and '_db_ok' in s:
    print('already DB-aware'); raise SystemExit
# 1) load DB-active tsyms right before leg_groups
anchor_lg="        leg_groups = [('atm_option_legs', 'atm_naked_st', False),"
assert s.count(anchor_lg)==1,'leg_groups anchor=%d'%s.count(anchor_lg)
inject=("        # DB-aware (2026-06-23): only flag ticker legs that are ACTUALLY active in a DB,\n"
        "        # so phantom ticker monitors left after a manual close / reconcile don't false-FAIL.\n"
        "        import sqlite3 as _sq, glob as _gl\n"
        "        db_active = set(); _db_ok = False\n"
        "        for _p in _gl.glob('backtest_data/nas_*_trading.db'):\n"
        "            try:\n"
        "                _c = _sq.connect(_p)\n"
        "                for _tbl in ('nas_atm_positions', 'nas_positions'):\n"
        "                    try:\n"
        "                        for (_ts,) in _c.execute(\"SELECT tradingsymbol FROM %s WHERE status='ACTIVE'\" % _tbl):\n"
        "                            db_active.add(_ts)\n"
        "                        _db_ok = True; break\n"
        "                    except Exception:\n"
        "                        continue\n"
        "            except Exception:\n"
        "                continue\n")
s=s.replace(anchor_lg, inject+anchor_lg, 1)
# 2) skip phantom ticker legs in the leg loop
anchor_ts="                tsym = leg.get('tradingsymbol', '?')\n"
assert s.count(anchor_ts)==1,'tsym anchor=%d'%s.count(anchor_ts)
skip=anchor_ts+("                if _db_ok and tsym not in db_active:\n"
                "                    continue  # phantom ticker leg (not DB-active) — already closed, not a real failure\n")
s=s.replace(anchor_ts, skip, 1)
ast.parse(s)
shutil.copy(P, P+'.bak_dbaware'); open(P,'w',encoding='utf-8').write(s)
print('PATCHED guardian: DB-aware (phantom ticker legs no longer false-FAIL)')
