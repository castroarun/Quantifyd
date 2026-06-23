import ast, shutil
P='/home/arun/quantifyd/scripts/nas_live_guardian.py'
s=open(P,encoding='utf-8').read()
if "is_movestop" in s:
    print('already v3-aware'); raise SystemExit
old_lg=("        leg_groups = [('atm_option_legs', 'atm_naked_st'),\n"
        "                      ('atm2_option_legs', None),\n"
        "                      ('atm4_option_legs', 'atm4_naked_st'),\n"
        "                      ('option_legs', None)]")
new_lg=("        leg_groups = [('atm_option_legs', 'atm_naked_st', False),\n"
        "                      ('atm2_option_legs', None, True),  # v3: ATM2 on 0.4%% move-stop, per-leg SL disabled\n"
        "                      ('atm4_option_legs', 'atm4_naked_st', False),\n"
        "                      ('option_legs', None, False)]")
assert s.count(old_lg)==1,'leg_groups count=%d'%s.count(old_lg)
s=s.replace(old_lg,new_lg,1)
s=s.replace("        for legs_key, naked_key in leg_groups:",
            "        for legs_key, naked_key, is_movestop in leg_groups:",1)
old_sl=("                elif slp > 0 and cp >= slp:\n"
        "                    gaps.append('%s: premium %.1f >= SL %.1f but STILL OPEN \xe2\x80\x94 stop not firing' % (tsym, cp, slp))".encode().decode())
# robust: match without the em-dash specifics
import re
pat="                elif slp > 0 and cp >= slp:\n                    gaps.append("
idx=s.find(pat)
assert idx!=-1,'SL-flag block not found'
# insert the is_movestop guard: wrap the gaps.append in 'if not is_movestop:'
s=s.replace("                elif slp > 0 and cp >= slp:\n                    gaps.append(",
            "                elif slp > 0 and cp >= slp:\n                    if not is_movestop:  # ATM2 v3 move-stop: per-leg SL intentionally off\n                        gaps.append(",1)
ast.parse(s)
shutil.copy(P,P+'.bak_v3aware'); open(P,'w',encoding='utf-8').write(s)
print('PATCHED guardian: ATM2 move-stop legs no longer false-flagged as "stop not firing"')
