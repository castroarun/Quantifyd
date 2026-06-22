"""ATM2: revert to CASCADE (re-entry) but add the user's 2026-06-22 strike-change gate —
on a 30% SL hit, only close BOTH + re-enter if price has moved to a DIFFERENT ATM strike
(so the re-entry is on a new strike). If the ATM is unchanged, HOLD the position (no
same-strike exit/re-enter churn) even though the SL is hit; wait for a strike move or EOD.

- config NAS_ATM2_DEFAULTS (inherited by 916_atm2): re_enter_on_sl True, move_stop_pct 0
  (move-stop OFF), add cascade_require_strike_change=True.
- nas_atm2_executor.check_and_handle_sl: strike-change gate at the top of the SL-hit branch.
Guarded: count asserts + ast.parse + .bak."""
import ast, shutil

# ---- config ----
C = '/home/arun/quantifyd/config.py'
s = open(C, encoding='utf-8').read()
old_cfg = ("    're_enter_on_sl': False,   # research/68: move-stop is one-and-done\n"
           "    'exit_both_on_sl': True,\n"
           "    'move_stop_pct': 0.004,    # research/68: +-0.4% underlying move-stop (0=off)\n")
new_cfg = ("    're_enter_on_sl': True,    # user 2026-06-22: cascade re-entry restored\n"
           "    'exit_both_on_sl': True,\n"
           "    'move_stop_pct': 0,        # user 2026-06-22: move-stop OFF (back to cascade)\n"
           "    'cascade_require_strike_change': True,  # only cascade if ATM strike changed (no same-strike churn)\n")
assert s.count(old_cfg) == 1, 'config block count=%d' % s.count(old_cfg)
s = s.replace(old_cfg, new_cfg, 1)
ast.parse(s)
shutil.copy(C, C + '.bak_strikegate')
open(C, 'w', encoding='utf-8').write(s)
print('PATCHED config: ATM2 cascade restored (re_enter=True, move_stop=0, strike-change gate on)')

# ---- handler ----
P = '/home/arun/quantifyd/services/nas_atm2_executor.py'
s = open(P, encoding='utf-8').read()
if 'no same-strike churn' in s:
    print('handler ALREADY has strike gate'); raise SystemExit
anchor = ("            if live_prem >= sl_price:\n"
          "                logger.info(f\"[NAS-ATM2] SL HIT: {tsym} live={live_prem:.2f} >= SL={sl_price:.2f}\")\n")
assert s.count(anchor) == 1, 'SL-hit anchor count=%d' % s.count(anchor)
gate = ("            if live_prem >= sl_price:\n"
        "                # user 2026-06-22: only cascade (close+re-enter) if price has moved to a\n"
        "                # DIFFERENT ATM strike. If the ATM is unchanged the re-entry would be the\n"
        "                # SAME strike (pointless churn) -> HOLD the position instead, even though\n"
        "                # the 30% SL is hit. Wait for a strike move or EOD squareoff.\n"
        "                if self.cfg.get('cascade_require_strike_change', False):\n"
        "                    _cs = self.scanner.get_live_spot()\n"
        "                    _ck = pos.get('strike')\n"
        "                    if _cs and _ck and int(self.scanner.get_atm_strike(_cs)) == int(_ck):\n"
        "                        logger.info(f\"[NAS-ATM2] SL hit on {tsym} but ATM unchanged \"\n"
        "                                    f\"(spot {_cs:.0f} -> ATM {int(self.scanner.get_atm_strike(_cs))} \"\n"
        "                                    f\"== current strike {int(_ck)}); HOLD, no same-strike churn\")\n"
        "                        continue\n"
        "                logger.info(f\"[NAS-ATM2] SL HIT: {tsym} live={live_prem:.2f} >= SL={sl_price:.2f}\")\n")
s = s.replace(anchor, gate, 1)
ast.parse(s)
shutil.copy(P, P + '.bak_strikegate')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED nas_atm2_executor: strike-change gate on SL cascade')
