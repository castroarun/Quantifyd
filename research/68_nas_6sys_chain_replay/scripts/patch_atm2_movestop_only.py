import ast, shutil
P='/home/arun/quantifyd/services/nas_atm2_executor.py'
s=open(P,encoding='utf-8').read()
if "not self.cfg.get('move_stop_pct'" in s:
    print('already gated'); raise SystemExit
old="            if live_prem >= sl_price:\n"
new=("            # v3 (2026-06-22): when the move-stop is active it is the SOLE exit trigger\n"
     "            # (matches the backtested move-stop). The 30% premium SL (~0.2% underlying)\n"
     "            # would otherwise pre-empt the 0.4% move-stop and the re-center would never fire.\n"
     "            if (not self.cfg.get('move_stop_pct', 0)) and live_prem >= sl_price:\n")
assert s.count(old)==1, 'anchor count=%d'%s.count(old)
s=s.replace(old,new,1); ast.parse(s)
shutil.copy(P,P+'.bak_msonly'); open(P,'w',encoding='utf-8').write(s)
print('PATCHED: per-leg SL disabled while move-stop active (move-stop = sole trigger)')
