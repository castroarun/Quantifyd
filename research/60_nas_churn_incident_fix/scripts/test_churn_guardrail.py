"""Offline test for the NAS ATM re-entry guardrail fix (research/60, 2026-06-08 churn).
Tests _check_guardrails in isolation via a stub DB — no live, no Kite, no restart.
Replays the exact churn (5 ST_EXIT cycles) and asserts it is now BLOCKED."""
import sys
sys.path.insert(0, "/home/arun/quantifyd")
from datetime import datetime, timedelta
from services.nas_atm_executor import NasAtmExecutor
from services.nas_atm2_executor import NasAtm2Executor
from services.nas_atm4_executor import NasAtm4Executor

def mins_ago(m):
    return (datetime.now() - timedelta(minutes=m)).strftime('%Y-%m-%d %H:%M:%S')

class StubDB:
    def __init__(self, closed=None, active=None, orders=0, daily_pnl=0):
        self._c = closed or []; self._a = active or []; self._o = orders; self._p = daily_pnl
    def get_today_order_count(self): return self._o
    def get_active_positions(self): return self._a
    def get_today_closed_positions(self): return self._c
    def get_state(self): return {'daily_pnl': self._p}

def make_ex(cooldown=15, max_re=5, db=None):
    ex = NasAtmExecutor.__new__(NasAtmExecutor)   # bypass heavy __init__
    ex.cfg = {'enabled': True, 'skip_weekdays': (), 'max_dte_at_entry': None,
              'max_daily_orders': 40, 'max_strangles': 1, 'max_reentries': max_re,
              'reentry_cooldown_min': cooldown, 'max_daily_loss': 25000,
              'entry_start_time': '00:00', 'entry_end_time': '23:59'}
    ex.db = db or StubDB()
    return ex

def leg(sid, reason, exit_m):
    return {'strangle_id': sid, 'exit_reason': reason, 'exit_time': mins_ago(exit_m)}

P = F = 0
def check(name, got, want, why=''):
    global P, F
    ok = (got == want); P += ok; F += (not ok)
    print(("  PASS " if ok else "  **FAIL ") + name + " -> allowed=%s (want %s) | %s" % (got, want, why))

# 1) THE BUG: 5 ST_EXIT cycles, last exit 2 min ago. Old code allowed; new BLOCKS.
closed = [leg(i, 'ST_EXIT', 2) for i in range(1, 6)]
ok, why = make_ex(db=StubDB(closed=closed))._check_guardrails(True)
check("churn replay: 5 ST_EXIT cycles, exit 2min ago", ok, False, why)
old_counter = len([t for t in closed if t['exit_reason'] == 'SL_HIT'])
print("     [old SL_HIT-only counter would see %d -> would NOT block: bug reproduced]" % old_counter)

# 2) cooldown blocks even a single recent cycle
ok, why = make_ex(db=StubDB(closed=[leg(1, 'ST_EXIT', 3)]))._check_guardrails(True)
check("cooldown: 1 cycle, exit 3min ago (<15)", ok, False, why)

# 3) cap blocks after max_reentries even when cooldown has passed
ok, why = make_ex(db=StubDB(closed=[leg(i, 'ST_EXIT', 30) for i in range(1, 6)]))._check_guardrails(True)
check("cap: 5 cycles, exit 30min ago", ok, False, why)

# 4) legit re-entry: 1 cycle, exit 30 min ago -> ALLOWED
ok, why = make_ex(db=StubDB(closed=[leg(1, 'ST_EXIT', 30)]))._check_guardrails(True)
check("legit: 1 cycle, exit 30min ago", ok, True, why)

# 5) active strangle exists -> blocked (existing guard)
ok, why = make_ex(db=StubDB(active=[{'strangle_id': 9}]))._check_guardrails(True)
check("active strangle exists", ok, False, why)

# 6) cooldown disabled (0) + 1 recent cycle -> allowed (cap not reached)
ok, why = make_ex(cooldown=0, db=StubDB(closed=[leg(1, 'ST_EXIT', 1)]))._check_guardrails(True)
check("cooldown=0, 1 cycle 1min ago", ok, True, why)

# 7) cap counts MIXED exit reasons (SL_HIT + ST_EXIT + TIME)
mix = [leg(1, 'SL_HIT', 30), leg(2, 'ST_EXIT', 30), leg(3, 'SL_HIT', 30), leg(4, 'ST_EXIT', 30), leg(5, 'TIME', 30)]
ok, why = make_ex(db=StubDB(closed=mix))._check_guardrails(True)
check("cap: 5 mixed-reason cycles, 30min ago", ok, False, why)

# 8) exit just OUTSIDE cooldown (16min), 2 cycles -> allowed
ok, why = make_ex(db=StubDB(closed=[leg(1, 'ST_EXIT', 16), leg(2, 'ST_EXIT', 16)]))._check_guardrails(True)
check("exit 16min ago (>15), 2 cycles", ok, True, why)

# 9) two legs of the SAME strangle = ONE cycle (distinct strangle_id), not two
two_legs = [leg(7, 'SL_HIT', 30), leg(7, 'ST_EXIT', 30)]  # same strangle_id 7
ok, why = make_ex(db=StubDB(closed=two_legs))._check_guardrails(True)
check("2 legs same strangle = 1 cycle, 30min ago", ok, True, why)

# 10) subclass inheritance — atm2/atm4 use the same fixed method
inh = (NasAtm2Executor._check_guardrails is NasAtmExecutor._check_guardrails
       and NasAtm4Executor._check_guardrails is NasAtmExecutor._check_guardrails)
check("atm2/atm4 inherit the fixed guardrail", inh, True, "shared method")

# 11/12) base/OTM executor (nas_executor) cooldown — defense-in-depth
from services.nas_executor import NasExecutor
def make_base(cooldown=15, db=None):
    ex = NasExecutor.__new__(NasExecutor)
    ex.cfg = {'enabled': True, 'skip_weekdays': (), 'max_dte_at_entry': None,
              'max_daily_orders': 20, 'max_strangles': 1, 'reentry_cooldown_min': cooldown,
              'max_daily_loss': 15000, 'max_adjustments_total': 4,
              'entry_start_time': '00:00', 'entry_end_time': '23:59'}
    ex.db = db or StubDB()
    return ex
ok, why = make_base(db=StubDB(closed=[leg(1, 'SL_HIT', 3)]))._check_guardrails(True)
check("base/OTM cooldown: exit 3min ago (<15)", ok, False, why)
ok, why = make_base(db=StubDB(closed=[leg(1, 'SL_HIT', 30)]))._check_guardrails(True)
check("base/OTM: exit 30min ago -> allowed", ok, True, why)

print("\n=== %d passed, %d failed ===" % (P, F))
sys.exit(1 if F else 0)
