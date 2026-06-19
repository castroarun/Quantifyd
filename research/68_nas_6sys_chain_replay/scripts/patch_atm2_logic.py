"""ATM2 move-stop, part 2/3: config knob + handler logic.
- config NAS_ATM2_DEFAULTS: add move_stop_pct=0.004 (0=off); re_enter_on_sl True->False
  (research/68: move-stop is one-and-done; re-entry hurts).
- nas_atm2_executor.check_and_handle_sl: add a +-move-stop pass (close BOTH legs when the
  underlying moved >= move_stop_pct from entry spot) BEFORE the per-leg premium SL (which
  stays as a backstop). Re-entry gated on re_enter_on_sl. Move-stopped strangles are skipped
  by the per-leg loop (existing `if strangle_id in exited_strangles: continue`).
Guarded: count asserts + ast.parse + .bak.
"""
import ast, shutil

# ---- config ----
C = '/home/arun/quantifyd/config.py'
s = open(C, encoding='utf-8').read()
if 'move_stop_pct' not in s:
    old = ("    're_enter_on_sl': True,\n    'exit_both_on_sl': True,\n")
    new = ("    're_enter_on_sl': False,   # research/68: move-stop is one-and-done\n"
           "    'exit_both_on_sl': True,\n"
           "    'move_stop_pct': 0.004,    # research/68: +-0.4% underlying move-stop (0=off)\n")
    assert s.count(old) == 1, 'config block count=%d' % s.count(old)
    s = s.replace(old, new, 1)
    ast.parse(s)
    shutil.copy(C, C + '.bak_atm2move')
    open(C, 'w', encoding='utf-8').write(s)
    print('PATCHED config: NAS_ATM2_DEFAULTS move_stop_pct=0.004, re_enter_on_sl=False')
else:
    print('config already has move_stop_pct')

# ---- handler ----
P = '/home/arun/quantifyd/services/nas_atm2_executor.py'
s = open(P, encoding='utf-8').read()
if 'MOVE-STOP' in s:
    print('ATM2 handler ALREADY PATCHED'); raise SystemExit

# A) insert the move-stop pass right after `exited_strangles = set()`
anchor = "        # Track which strangles we've already exited this call\n        exited_strangles = set()\n"
assert s.count(anchor) == 1, 'anchor count=%d' % s.count(anchor)
block = anchor + (
    "\n        # --- research/68: +-move-stop (one-and-done). Close BOTH legs when the\n"
    "        # underlying has moved >= move_stop_pct from entry spot. Fires before the\n"
    "        # per-leg premium SL on trend days; the per-leg SL stays as a backstop.\n"
    "        move_pct = self.cfg.get('move_stop_pct', 0) or 0\n"
    "        if move_pct > 0:\n"
    "            cur_spot = self.scanner.get_live_spot()\n"
    "            if cur_spot:\n"
    "                by_strangle = {}\n"
    "                for p in positions:\n"
    "                    if p['status'] == 'ACTIVE':\n"
    "                        by_strangle.setdefault(p.get('strangle_id'), []).append(p)\n"
    "                for sid, legs in by_strangle.items():\n"
    "                    if sid in exited_strangles:\n"
    "                        continue\n"
    "                    espot = next((l.get('entry_spot') for l in legs if l.get('entry_spot')), None)\n"
    "                    if not espot:\n"
    "                        continue\n"
    "                    moved = abs(cur_spot - espot) / espot\n"
    "                    if moved >= move_pct:\n"
    "                        logger.info(f\"[NAS-ATM2] MOVE-STOP: spot {cur_spot:.1f} vs entry {espot:.1f} = \"\n"
    "                                    f\"{moved*100:.2f}% >= {move_pct*100:.2f}% -> close strangle #{sid}\")\n"
    "                        exited_strangles.add(sid)\n"
    "                        action = {'type': 'MOVE_STOP', 'strangle_id': sid, 'entry_spot': espot,\n"
    "                                  'cur_spot': cur_spot, 'move_pct': round(moved*100, 2), 'closed_legs': []}\n"
    "                        for leg_pos in legs:\n"
    "                            leg_tsym = leg_pos.get('tradingsymbol', '')\n"
    "                            leg_live = live_ltps.get(leg_tsym) or self.scanner.get_live_option_premium(leg_tsym) or 0\n"
    "                            self._close_leg(leg_pos, leg_live, 'MOVE_STOP')\n"
    "                            pnl_leg = (leg_pos['entry_price'] - leg_live) * leg_pos['qty']\n"
    "                            st = self.db.get_state()\n"
    "                            self.db.update_state(daily_pnl=round((st.get('daily_pnl', 0) or 0) + pnl_leg, 2))\n"
    "                            action['closed_legs'].append({'tradingsymbol': leg_tsym,\n"
    "                                                          'exit_price': round(leg_live, 2), 'pnl': round(pnl_leg, 2)})\n"
    "                        fresh = self.db.get_positions_by_strangle(sid)\n"
    "                        self._record_trade(sid, fresh, 'MOVE_STOP')\n"
    "                        action['total_pnl'] = round(sum(l['pnl'] for l in action['closed_legs']), 2)\n"
    "                        actions.append(action)\n"
)
s = s.replace(anchor, block, 1)

# B) gate the re-entry on re_enter_on_sl
old_re = "                # Immediate re-entry at current ATM\n                spot = self.scanner.get_live_spot()"
new_re = ("                # Immediate re-entry at current ATM (gated; research/68 one-and-done)\n"
          "                spot = self.scanner.get_live_spot() if self.cfg.get('re_enter_on_sl', True) else None")
assert s.count(old_re) == 1, 're-entry anchor count=%d' % s.count(old_re)
s = s.replace(old_re, new_re, 1)

ast.parse(s)
shutil.copy(P, P + '.bak_atm2move')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED nas_atm2_executor: move-stop pass + re-entry gate')
