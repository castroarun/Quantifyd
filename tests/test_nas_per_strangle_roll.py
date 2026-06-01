"""Offline scenario test: per-strangle cross-leg roll independence.

Validates the per-strangle refactor of NasTicker._check_premium_tick:
  - an imbalanced strangle rolls even when BOTH OTM variants are pooled (4 legs)
    -> the old blunt `len != 2` guard would have returned and rolled nothing
  - a balanced strangle does NOT roll
  - two strangles roll INDEPENDENTLY (per-strangle trigger/cooldown state)
  - a single 2-leg strangle still rolls (regression)

No network/kite/orders: _fire_tick_adjustment is replaced with a recorder.
"""
import sys
import time
sys.path.insert(0, ".")
from config import NAS_DEFAULTS
from services.nas_ticker import NasTicker

fires = []


def _leg(tsym, sid, side, entry):
    return {"tradingsymbol": tsym, "entry_premium": entry, "leg": side,
            "instrument_type": side, "position_id": None, "strangle_id": sid}


def _setup(t, ltp17pe=28, ltp34pe=16):
    t._sl_triggered = False
    t._option_tokens = {
        100: _leg("OTM24000CE", 17, "CE", 12),
        101: _leg("OTM23350PE", 17, "PE", 18),
        200: _leg("SQ23900CE", 34, "CE", 21),
        201: _leg("SQ23350PE", 34, "PE", 23),
    }
    t._option_ltps = {100: 12, 101: ltp17pe, 200: 15, 201: ltp34pe}
    t._adj_confirm = {}
    t._adj_triggered_by_sid = {}
    t._adj_next_direction_by_sid = {}
    t._last_adj_ts_by_sid = {}
    fires.clear()

    def rec(info, prem, action, target, sid=None):
        fires.append({"sid": sid, "action": action,
                      "tsym": info["tradingsymbol"], "target": target})
    t._fire_tick_adjustment = rec
    t._fire_emergency_exit = lambda reason: fires.append({"emergency": reason})


def _wait():
    time.sleep(0.3)  # let the daemon fire-thread run


t = NasTicker(config=dict(NAS_DEFAULTS))

# TEST 1: 4 legs pooled (both variants) -> imbalanced strangle 17 STILL rolls.
_setup(t)
t._check_premium_tick(101, 28)   # tick 1 (confirm count -> 1)
t._check_premium_tick(101, 28)   # tick 2 (confirm count -> 2 -> fire)
_wait()
assert len(fires) == 1, "TEST1 expected exactly 1 fire, got %r" % fires
assert fires[0]["sid"] == 17, "TEST1 expected strangle 17, got %r" % fires[0]
assert fires[0]["action"] == "ROLL_OUT", "TEST1 expected ROLL_OUT, got %r" % fires[0]
print("TEST 1 PASS: imbalanced strangle 17 rolls even with 4 legs pooled "
      "(old guard would have paused it)")

# TEST 2: balanced strangle 34 did NOT roll.
assert all(f.get("sid") != 34 for f in fires), "TEST2 strangle 34 rolled: %r" % fires
print("TEST 2 PASS: balanced strangle 34 did not roll")

# TEST 3: strangle 34 rolls INDEPENDENTLY while 17 is mid-roll (per-sid state).
_setup(t, ltp34pe=33)
t._adj_triggered_by_sid[17] = True   # pretend strangle 17 is mid-adjustment
t._check_premium_tick(201, 33)
t._check_premium_tick(201, 33)
_wait()
s34 = [f for f in fires if f.get("sid") == 34]
assert len(s34) == 1, "TEST3 expected strangle 34 to roll independently, got %r" % fires
print("TEST 3 PASS: strangle 34 rolls while 17 is mid-roll (independent per-sid)")

# TEST 4 (regression): a single 2-leg strangle still rolls normally.
t._sl_triggered = False
t._option_tokens = {100: _leg("OTM24000CE", 17, "CE", 12),
                    101: _leg("OTM23350PE", 17, "PE", 18)}
t._option_ltps = {100: 12, 101: 28}
t._adj_confirm = {}
t._adj_triggered_by_sid = {}
t._adj_next_direction_by_sid = {}
t._last_adj_ts_by_sid = {}
fires.clear()
t._check_premium_tick(101, 28)
t._check_premium_tick(101, 28)
_wait()
assert len(fires) == 1 and fires[0]["sid"] == 17, "TEST4 regression: %r" % fires
print("TEST 4 PASS: single 2-leg strangle still rolls normally")

print("\nALL TESTS PASSED")
