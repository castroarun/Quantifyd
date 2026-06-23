import ast, shutil, json
# 1) gate(): decouple ENTRY (paper-shadow always) from LIVE/PAPER (gated)
P='/home/arun/quantifyd/services/nas_day_matrix.py'
s=open(P,encoding='utf-8').read()
if 'paper_shadow' in s:
    print('gate already paper-shadow-aware')
else:
    old=("    mm = master_mode if master_mode is not None else _master_mode()\n"
         "    mode = 'live' if (mm == 'live' and row.get('live')) else 'paper'\n"
         "    return {'allow': enter, 'mode': mode, 'dte': dte, 'gap_pct': gap_pct,\n"
         "            'reason': '; '.join(reasons) or ('DTE%s off, no gap' % dte)}")
    new=("    mm = master_mode if master_mode is not None else _master_mode()\n"
         "    # live only when gated-on (enter) AND master live AND this is a live row; else paper\n"
         "    mode = 'live' if (mm == 'live' and row.get('live') and enter) else 'paper'\n"
         "    # paper-shadow (user 2026-06-23): these systems ALSO enter in PAPER on EVERY day for\n"
         "    # the daily P&L curve, regardless of the live gating / master mode.\n"
         "    allow = enter or bool(row.get('paper_shadow'))\n"
         "    if not enter and row.get('paper_shadow'):\n"
         "        reasons.append('paper-shadow')\n"
         "    return {'allow': allow, 'mode': mode, 'dte': dte, 'gap_pct': gap_pct,\n"
         "            'reason': '; '.join(reasons) or ('DTE%s off, no gap' % dte)}")
    assert s.count(old)==1,'gate anchor=%d'%s.count(old)
    s=s.replace(old,new,1); ast.parse(s)
    shutil.copy(P,P+'.bak_papershadow'); open(P,'w',encoding='utf-8').write(s)
    print('PATCHED gate(): paper-shadow always-allow + mode gated on live-eligibility')
# 2) matrix: add paper_shadow=true to the 6 ATM rows
MP='/home/arun/quantifyd/backtest_data/nas_day_matrix.json'
m=json.load(open(MP))
ATM6=['nas_atm','nas_atm2','nas_atm4','nas_916_atm','nas_916_atm2','nas_916_atm4']
for kk in ATM6:
    m['systems'][kk]['paper_shadow']=True
shutil.copy(MP,MP+'.bak_papershadow')
json.dump(m,open(MP,'w'),indent=2)
print('matrix: paper_shadow=True on', ATM6)
