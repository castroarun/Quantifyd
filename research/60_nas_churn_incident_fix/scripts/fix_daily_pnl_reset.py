"""FIX 2026-06-16: daily-loss circuit-breaker used the PERSISTED state['daily_pnl'],
which is only reset by the EOD job. When that reset is skipped (e.g. a freeze, like
2026-06-12), the stale prior-day loss carries over and wrongly blocks entries the next
day -- it self-blocked 916-ATM2 on 2026-06-16 (stale -37638). Fix: compute the daily
P&L from TODAY's closed positions (already in scope as today_trades) -- date-aware and
self-correcting. nas_atm_executor covers all live ATM variants (916_atm/atm2/atm4 +
squeeze atm/atm2/atm4). Guarded."""
import ast, shutil
P = '/home/arun/quantifyd/services/nas_atm_executor.py'
s = open(P, encoding='utf-8').read()
if 'compute from TODAY' in s:
    print('ALREADY PATCHED'); raise SystemExit
OLD = (
    "            # Daily P&L circuit breaker\n"
    "            state = self.db.get_state()\n"
    "            daily_pnl = state.get('daily_pnl', 0) or 0\n"
    "            max_loss = cfg.get('max_daily_loss', 25000)\n"
)
NEW = (
    "            # Daily P&L circuit breaker -- compute from TODAY's closed positions,\n"
    "            # NOT the persisted state['daily_pnl'] (which goes stale if a prior EOD\n"
    "            # reset was skipped, e.g. during a freeze -- a stale -37638 wrongly blocked\n"
    "            # 916-ATM2 on 2026-06-16). today_trades is already in scope; date-aware.\n"
    "            daily_pnl = sum(((p.get('entry_price') or 0) - (p.get('exit_price') or 0)) * (p.get('qty') or 0)\n"
    "                            for p in today_trades if p.get('exit_price') is not None)\n"
    "            max_loss = cfg.get('max_daily_loss', 25000)\n"
)
assert s.count(OLD) == 1, 'OLD count=%d' % s.count(OLD)
s = s.replace(OLD, NEW, 1)
ast.parse(s)
shutil.copy(P, P + '.bak_dailypnl')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED #1: daily-loss now computed from today closed trades (all ATM variants)')
