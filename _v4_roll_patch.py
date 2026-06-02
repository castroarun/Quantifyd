"""Staged patch for after close: V4 premium-balanced roll strike.

Replaces nas_atm4_executor._find_roll_strike so it picks the OTM strike whose
live premium is CLOSEST to target_prem (the surviving leg), scanning from a
small OTM floor outward instead of forcing >=100-pt OTM + outward-only (which
undershot). Validated offline in tests/test_v4_roll_strike.py.
"""
import py_compile
import sys

F = "services/nas_atm4_executor.py"
s = open(F, encoding="utf-8").read()

OLD = '''    def _find_roll_strike(self, spot, inst_type, target_prem, expiry_date):
        """Find OTM strike with premium closest to target."""
        strike_step = self.cfg.get('strike_interval', 50)
        min_otm = self.cfg.get('min_otm_distance', 100)

        if inst_type == 'CE':
            start_strike = int(round((spot + min_otm) / strike_step)) * strike_step
            direction = 1
        else:
            start_strike = int(round((spot - min_otm) / strike_step)) * strike_step
            direction = -1

        best_strike = None
        best_prem = None
        best_diff = float('inf')

        for i in range(20):  # Check 20 strikes outward
            strike = start_strike + (i * strike_step * direction)
            tsym = self._build_roll_tradingsymbol(inst_type, strike, expiry_date)
            prem = self.scanner.get_live_option_premium(tsym)
            if prem is None:
                continue
            diff = abs(prem - target_prem)
            if diff < best_diff:
                best_diff = diff
                best_strike = strike
                best_prem = prem
            # Early exit if very close
            if diff <= 2.0:
                break
            # If premium is getting too far from target, stop searching
            if prem < target_prem * 0.5:
                break

        return best_strike, best_prem'''

NEW = '''    def _find_roll_strike(self, spot, inst_type, target_prem, expiry_date):
        """V4 premium-balanced roll: pick the OTM strike whose live premium is
        CLOSEST to target_prem (the surviving leg's premium). Scans OTM strikes
        from a small floor outward, so it can land NEARER spot than the old
        100-pt floor + outward-only search allowed — that forced an undershoot
        (2026-06-02 09:19 rolled CE to 23400 @23.6 when the surviving leg was 43
        and CE 23350 @36.7 was the match). Premiums fall monotonically further
        OTM, so we stop once we have passed the target."""
        strike_step = self.cfg.get('strike_interval', 50)
        # Stay at least this far OTM to avoid a near-ATM/ITM (high-gamma) roll,
        # but small enough to reach the premium-matching strike.
        min_floor = self.cfg.get('roll_min_otm', 50)
        atm = int(round(spot / strike_step)) * strike_step

        best_strike = None
        best_prem = None
        best_diff = float('inf')

        for i in range(15):
            if inst_type == 'CE':
                strike = atm + i * strike_step
                if strike - spot < min_floor:
                    continue
            else:
                strike = atm - i * strike_step
                if spot - strike < min_floor:
                    continue
            tsym = self._build_roll_tradingsymbol(inst_type, strike, expiry_date)
            prem = self.scanner.get_live_option_premium(tsym)
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

        return best_strike, best_prem'''

n = s.count(OLD)
if n != 1:
    print("ABORT: _find_roll_strike matched %d times (expected 1)" % n)
    sys.exit(1)
s = s.replace(OLD, NEW)
open(F, "w", encoding="utf-8").write(s)
py_compile.compile(F, doraise=True)
print("V4 roll-strike patch applied + py_compile OK")
