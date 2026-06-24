import ast, shutil
PC='/home/arun/quantifyd/config.py'
s=open(PC,encoding='utf-8').read()
oldc="    'lots_per_leg': 1,              # GO-LIVE 2026-05-01: dropped 5→1 (75 qty)\n"
newc=("    'lots_per_leg': 1,              # GO-LIVE 2026-05-01: dropped 5→1 (75 qty). LIVE real-money size.\n"
      "    'paper_lots_per_leg': 10,       # user 2026-06-24: PAPER book = 10 lots (650 qty) for a meaningful daily P&L curve; LIVE stays at lots_per_leg. Inherited by all 6 ATM via spread.\n")
if 'paper_lots_per_leg' in s:
    print('config already patched')
else:
    assert s.count(oldc)==1,'config anchor=%d'%s.count(oldc)
    s=s.replace(oldc,newc,1); ast.parse(s)
    shutil.copy(PC,PC+'.bak_paperlots'); open(PC,'w',encoding='utf-8').write(s)
    print('PATCHED config: paper_lots_per_leg=10 on NAS_ATM_DEFAULTS')
PE='/home/arun/quantifyd/services/nas_atm_executor.py'
s2=open(PE,encoding='utf-8').read()
olde="        cfg = self.cfg\n        lots = cfg.get('lots_per_leg', 5)\n        qty = lots * LOT_SIZE\n"
newe=("        cfg = self.cfg\n"
      "        # Per-mode sizing (user 2026-06-24): the PAPER book runs paper_lots_per_leg (10 lots) for a\n"
      "        # meaningful daily P&L curve; LIVE real money stays at lots_per_leg (1). _force_mode is set by\n"
      "        # the day-matrix gate in _check_guardrails above; matches _place_order's live/paper decision.\n"
      "        _fm = cfg.get('_force_mode')\n"
      "        _is_paper_size = (_fm == 'paper') if _fm else cfg.get('paper_trading_mode', True)\n"
      "        lots = cfg.get('paper_lots_per_leg', cfg.get('lots_per_leg', 5)) if _is_paper_size else cfg.get('lots_per_leg', 5)\n"
      "        qty = lots * LOT_SIZE\n")
if '_is_paper_size' in s2:
    print('executor already patched')
else:
    assert s2.count(olde)==1,'executor anchor=%d'%s2.count(olde)
    s2=s2.replace(olde,newe,1); ast.parse(s2)
    shutil.copy(PE,PE+'.bak_paperlots'); open(PE,'w',encoding='utf-8').write(s2)
    print('PATCHED executor: per-mode qty')
