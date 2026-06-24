import ast, shutil
# nas_atm_executor: thread bypass_cooldown through _check_guardrails + execute_strangle_entry
P='/home/arun/quantifyd/services/nas_atm_executor.py'
s=open(P,encoding='utf-8').read()
if 'bypass_cooldown' in s:
    print('base already patched'); 
else:
    edits=[
      ("    def _check_guardrails(self, is_entry=True):",
       "    def _check_guardrails(self, is_entry=True, bypass_cooldown=False):"),
      ("            cooldown_min = cfg.get('reentry_cooldown_min', 0)\n            if cooldown_min and today_trades:",
       "            cooldown_min = cfg.get('reentry_cooldown_min', 0)\n            if cooldown_min and today_trades and not bypass_cooldown:"),
      ("    def execute_strangle_entry(self, spot=None, scan_result=None):",
       "    def execute_strangle_entry(self, spot=None, scan_result=None, bypass_cooldown=False):"),
      ("        ok, reason = self._check_guardrails(is_entry=True)",
       "        ok, reason = self._check_guardrails(is_entry=True, bypass_cooldown=bypass_cooldown)"),
    ]
    for old,new in edits:
        assert s.count(old)==1,'edit count=%d for %r'%(s.count(old),old[:40])
        s=s.replace(old,new,1)
    ast.parse(s); shutil.copy(P,P+'.bak_recenter_cd'); open(P,'w',encoding='utf-8').write(s)
    print('PATCHED base: bypass_cooldown threaded through guardrails + entry')
# nas_atm2_executor: the move-stop re-center bypasses the cooldown
P2='/home/arun/quantifyd/services/nas_atm2_executor.py'
s2=open(P2,encoding='utf-8').read()
old2="                                new_sid, _msg = self.execute_strangle_entry(spot=cur_spot)"
new2="                                new_sid, _msg = self.execute_strangle_entry(spot=cur_spot, bypass_cooldown=True)"
if 'bypass_cooldown=True' in s2:
    print('atm2 re-center already bypasses cooldown')
else:
    assert s2.count(old2)==1,'atm2 anchor=%d'%s2.count(old2)
    s2=s2.replace(old2,new2,1); ast.parse(s2)
    shutil.copy(P2,P2+'.bak_recenter_cd'); open(P2,'w',encoding='utf-8').write(s2)
    print('PATCHED atm2: move-stop re-center bypasses cooldown')
