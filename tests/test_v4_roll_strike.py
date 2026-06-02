"""Offline test for the V4 roll-strike fix (nas_atm4_executor._find_roll_strike).

GOAL (V4 only): roll the stopped leg to the strike whose premium is CLOSEST to
the surviving leg's premium (premium-balanced adjustment). The OLD version
forced the new strike >= min_otm_distance (100) from spot and searched OUTWARD
(cheaper) only, so it UNDERSHOT — 2026-06-02 09:19 it rolled CE to 23400 @23.6
when the surviving PE 23250 was 43.1 and CE 23350 @36.7 was the right match.

This test embeds the NEW _find_roll_strike logic and replays the exact 09:19
chain with the real prices, asserting it now picks 23350 (not 23400).
"""


# --- the NEW logic (to be dropped into nas_atm4_executor.py after close) ---
def find_roll_strike_new(cfg, scanner_get_prem, build_tsym, spot, inst_type,
                         target_prem, expiry_date):
    """Strike whose live premium is closest to target_prem (the surviving leg).
    Scans OTM strikes from a small floor outward (so it can land NEARER spot
    than the old 100-pt floor allowed), premiums fall monotonically further OTM,
    so we stop once we've passed the target."""
    step = cfg.get('strike_interval', 50)
    min_floor = cfg.get('roll_min_otm', 50)   # stay at least this far OTM (gamma)
    atm = int(round(spot / step)) * step
    best_strike = best_prem = None
    best_diff = float('inf')
    for i in range(15):
        if inst_type == 'CE':
            strike = atm + i * step
            if strike - spot < min_floor:
                continue
        else:
            strike = atm - i * step
            if spot - strike < min_floor:
                continue
        prem = scanner_get_prem(build_tsym(inst_type, strike, expiry_date))
        if prem is None or prem <= 0:
            continue
        diff = abs(prem - target_prem)
        if diff < best_diff:
            best_diff = diff
            best_strike = strike
            best_prem = prem
        elif best_strike is not None and prem < target_prem:
            # going further OTM only lowers premium past the target -> stop
            break
    return best_strike, best_prem


# --- the OLD logic (for the before/after contrast) ---
def find_roll_strike_old(cfg, scanner_get_prem, build_tsym, spot, inst_type,
                         target_prem, expiry_date):
    step = cfg.get('strike_interval', 50)
    min_otm = cfg.get('min_otm_distance', 100)
    if inst_type == 'CE':
        start = int(round((spot + min_otm) / step)) * step
        direction = 1
    else:
        start = int(round((spot - min_otm) / step)) * step
        direction = -1
    best_s = best_p = None
    best_d = float('inf')
    for i in range(20):
        strike = start + i * step * direction
        prem = scanner_get_prem(build_tsym(inst_type, strike, expiry_date))
        if prem is None:
            continue
        d = abs(prem - target_prem)
        if d < best_d:
            best_d = d
            best_s = strike
            best_p = prem
        if d <= 2.0:
            break
        if prem < target_prem * 0.5:
            break
    return best_s, best_p


# --- replay the real 2026-06-02 09:19 prices ---
CE_PRICES = {23300: 56.7, 23350: 36.7, 23400: 23.6, 23450: 15.0, 23500: 10.0}
PE_PRICES = {23250: 43.1, 23200: 30.0, 23150: 20.0, 23100: 13.0}


def build(inst_type, strike, expiry):
    return "NIFTY%s%d%s" % (expiry, strike, inst_type)


def getprem(tsym):
    # parse "NIFTY26602<strike><CE|PE>"
    side = tsym[-2:]
    strike = int(tsym[len("NIFTY26602"):-2])
    return (CE_PRICES if side == "CE" else PE_PRICES).get(strike)


cfg = {'strike_interval': 50, 'min_otm_distance': 100, 'roll_min_otm': 50}
SPOT = 23280.0
TARGET = 42.2  # surviving PE 23250 (price_x logged that morning)

old_s, old_p = find_roll_strike_old(cfg, getprem, build, SPOT, 'CE', TARGET, '26602')
new_s, new_p = find_roll_strike_new(cfg, getprem, build, SPOT, 'CE', TARGET, '26602')

print("OLD logic picked: CE %s @ %s  (diff from 42.2 = %.1f)"
      % (old_s, old_p, abs(old_p - TARGET)))
print("NEW logic picked: CE %s @ %s  (diff from 42.2 = %.1f)"
      % (new_s, new_p, abs(new_p - TARGET)))

assert old_s == 23400, "expected OLD to reproduce the bug (23400), got %s" % old_s
assert new_s == 23350, "NEW should pick 23350 (closest to 42.2), got %s" % new_s
assert abs(new_p - TARGET) < abs(old_p - TARGET), "NEW must be closer to target"
print("TEST 1 PASS: V4 roll now picks CE 23350 (~37, the premium match) not 23400 (~24)")

# PE-side symmetry: stopped PE, surviving CE ~37 -> should pick PE ~37 strike
pe_s, pe_p = find_roll_strike_new(cfg, getprem, build, SPOT, 'PE', 36.0, '26602')
assert pe_s is not None and abs(pe_p - 36.0) <= abs(PE_PRICES[23200] - 36.0), \
    "PE-side pick %s @ %s not closest" % (pe_s, pe_p)
print("TEST 2 PASS: PE-side roll also premium-matches (PE %s @ %s for target 36)"
      % (pe_s, pe_p))

# floor respected: never rolls closer than roll_min_otm
assert new_s - SPOT >= cfg['roll_min_otm'], "floor violated"
print("TEST 3 PASS: roll respects the %d-pt OTM floor (no near-ATM/ITM roll)"
      % cfg['roll_min_otm'])

print("\nALL TESTS PASSED")
