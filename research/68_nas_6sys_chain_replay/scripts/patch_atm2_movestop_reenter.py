"""ATM2 v3 (user 2026-06-22): keep the 0.4% move-stop as the EXIT trigger, but on that exit
CASCADE — re-enter a fresh ATM strangle at the NEW current strike — only if the new ATM
differs from the strike just closed (the strike closest to current price must not be the one
already taken). The per-leg 30% SL stays as a one-and-done backstop.

- config NAS_ATM2_DEFAULTS (inherited by 916_atm2): move_stop_pct 0->0.004, re_enter_on_sl
  True->False, cascade_require_strike_change True->False, add move_stop_reenter=True.
- nas_atm2_executor: after the move-stop closes both legs, re-enter at get_atm_strike(cur_spot)
  if it != the closed strike.
Guarded: count asserts + ast.parse + .bak."""
import ast, shutil

# ---- config ----
C = '/home/arun/quantifyd/config.py'
s = open(C, encoding='utf-8').read()
old_cfg = ("    're_enter_on_sl': True,    # user 2026-06-22: cascade re-entry restored\n"
           "    'exit_both_on_sl': True,\n"
           "    'move_stop_pct': 0,        # user 2026-06-22: move-stop OFF (back to cascade)\n"
           "    'cascade_require_strike_change': True,  # only cascade if ATM strike changed (no same-strike churn)\n")
new_cfg = ("    're_enter_on_sl': False,   # user 2026-06-22 v3: per-leg 30% SL = one-and-done backstop\n"
           "    'exit_both_on_sl': True,\n"
           "    'move_stop_pct': 0.004,    # user 2026-06-22 v3: 0.4% move-stop = exit trigger\n"
           "    'cascade_require_strike_change': False,\n"
           "    'move_stop_reenter': True, # user 2026-06-22 v3: move-stop re-centers to new ATM (if strike changed)\n")
assert s.count(old_cfg) == 1, 'config block count=%d' % s.count(old_cfg)
s = s.replace(old_cfg, new_cfg, 1)
ast.parse(s)
shutil.copy(C, C + '.bak_msreenter')
open(C, 'w', encoding='utf-8').write(s)
print('PATCHED config: move_stop 0.004 + re-center (re_enter_on_sl off, move_stop_reenter on)')

# ---- handler: re-enter after the move-stop closes both ----
P = '/home/arun/quantifyd/services/nas_atm2_executor.py'
s = open(P, encoding='utf-8').read()
if 'MOVE-STOP re-center' in s:
    print('handler ALREADY has move-stop re-center'); raise SystemExit
anchor = ("                        action['total_pnl'] = round(sum(l['pnl'] for l in action['closed_legs']), 2)\n"
          "                        actions.append(action)\n")
assert s.count(anchor) == 1, 'move-stop tail anchor count=%d' % s.count(anchor)
add = anchor + (
    "                        # user 2026-06-22 v3: cascade the move-stop to the NEW current ATM\n"
    "                        # (re-center), only if it differs from the strike just closed.\n"
    "                        if self.cfg.get('move_stop_reenter', False):\n"
    "                            old_strike = next((l.get('strike') for l in legs if l.get('strike')), None)\n"
    "                            new_atm = self.scanner.get_atm_strike(cur_spot)\n"
    "                            if old_strike is None or int(new_atm) != int(old_strike):\n"
    "                                new_sid, _msg = self.execute_strangle_entry(spot=cur_spot)\n"
    "                                if new_sid:\n"
    "                                    action['re_entry'] = {'strangle_id': new_sid, 'strike': int(new_atm)}\n"
    "                                    logger.info(f\"[NAS-ATM2] MOVE-STOP re-center -> #{new_sid} at new ATM \"\n"
    "                                                f\"{int(new_atm)} (was {old_strike})\")\n"
    "                                else:\n"
    "                                    action['re_entry_blocked'] = _msg\n"
    "                            else:\n"
    "                                logger.info(f\"[NAS-ATM2] MOVE-STOP: new ATM {int(new_atm)} == closed strike \"\n"
    "                                            f\"{old_strike}; no same-strike re-enter\")\n"
)
s = s.replace(anchor, add, 1)
ast.parse(s)
shutil.copy(P, P + '.bak_msreenter')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED nas_atm2_executor: move-stop re-centers to new ATM strike')
