"""FIX #5 (2026-06-16, user-flagged): the SQUEEZE_ENTRY signal persists across a long
coil, so after a stop + 15-min cooldown the squeeze variants re-sold into the SAME (now
stale, more-likely-to-break) squeeze. Make it ONE ENTRY PER SQUEEZE EPISODE: block
re-entry while squeeze_count keeps climbing since our last entry; allow again only after
the squeeze resets (count drops) = a fresh episode. In nas_ticker candle-close handler
(atm/atm2/atm4). Guarded."""
import ast, shutil
P = '/home/arun/quantifyd/services/nas_ticker.py'
s = open(P, encoding='utf-8').read()
if '_fresh_episode' in s:
    print('ALREADY PATCHED'); raise SystemExit

edits = [
    # 1) helper after has_squeeze
    ("            has_squeeze = any(s['type'] == 'SQUEEZE_ENTRY' for s in scan.get('signals', []))\n\n            # Shared state update values",
     "            has_squeeze = any(s['type'] == 'SQUEEZE_ENTRY' for s in scan.get('signals', []))\n\n"
     "            # One-entry-per-squeeze-episode gate (fix 2026-06-16): SQUEEZE_ENTRY persists\n"
     "            # across a long coil, so after a stop+cooldown the system re-sold into the SAME\n"
     "            # stale squeeze. Block re-entry while squeeze_count keeps climbing since our last\n"
     "            # entry; allow again only after the squeeze resets (count drops) = a fresh episode.\n"
     "            _sq_count = scan.get('squeeze_count', 0)\n"
     "            if not hasattr(self, '_sq_entry_count'):\n"
     "                self._sq_entry_count = {}\n"
     "            def _fresh_episode(_key):\n"
     "                _last = self._sq_entry_count.get(_key)\n"
     "                return _last is None or _sq_count <= _last\n\n"
     "            # Shared state update values"),
    # 2) ATM gate
    ("                    if has_squeeze:\n                        executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)",
     "                    if has_squeeze and _fresh_episode('atm'):\n                        executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)"),
    # 3) ATM record
    ('                            logger.info(f"[NAS-ATM] Entry: strangle #{sid} at spot {spot:.1f}")\n                            self.subscribe_atm_option_legs(atm_db.get_active_positions())',
     '                            logger.info(f"[NAS-ATM] Entry: strangle #{sid} at spot {spot:.1f}")\n                            self._sq_entry_count[\'atm\'] = _sq_count\n                            self.subscribe_atm_option_legs(atm_db.get_active_positions())'),
    # 4) ATM2 gate
    ("                    if has_squeeze:\n                        executor = NasAtm2Executor(config=NAS_ATM2_DEFAULTS)",
     "                    if has_squeeze and _fresh_episode('atm2'):\n                        executor = NasAtm2Executor(config=NAS_ATM2_DEFAULTS)"),
    # 5) ATM2 record
    ('                            logger.info(f"[NAS-ATM2] Entry: strangle #{sid} at spot {spot:.1f}")\n                            self.subscribe_atm2_option_legs(atm2_db.get_active_positions())',
     '                            logger.info(f"[NAS-ATM2] Entry: strangle #{sid} at spot {spot:.1f}")\n                            self._sq_entry_count[\'atm2\'] = _sq_count\n                            self.subscribe_atm2_option_legs(atm2_db.get_active_positions())'),
    # 6) ATM4 gate
    ("                    if has_squeeze:\n                        executor = NasAtm4Executor(config=NAS_ATM4_DEFAULTS)",
     "                    if has_squeeze and _fresh_episode('atm4'):\n                        executor = NasAtm4Executor(config=NAS_ATM4_DEFAULTS)"),
    # 7) ATM4 record
    ('                            logger.info(f"[NAS-ATM4] Entry: strangle #{sid} at spot {spot:.1f}")\n                            self.subscribe_atm4_option_legs(atm4_db.get_active_positions())',
     '                            logger.info(f"[NAS-ATM4] Entry: strangle #{sid} at spot {spot:.1f}")\n                            self._sq_entry_count[\'atm4\'] = _sq_count\n                            self.subscribe_atm4_option_legs(atm4_db.get_active_positions())'),
]
for i, (old, new) in enumerate(edits, 1):
    assert s.count(old) == 1, 'edit %d count=%d' % (i, s.count(old))
    s = s.replace(old, new, 1)
ast.parse(s)
shutil.copy(P, P + '.bak_oneshot')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED #5: one-entry-per-squeeze-episode (atm/atm2/atm4)')
