"""FIX #2-tick: make the naked-survivor ST exit tick-responsive. Cache st_val on
each candle-close compute; check it on every tick in the candle updater; clear it
on exit so a fresh survivor never uses a stale value. ATM + ATM4. Guarded."""
import ast, shutil
P = '/home/arun/quantifyd/services/nas_ticker.py'
s = open(P, encoding='utf-8').read()
if '_atm_naked_st_val' in s:
    print('ALREADY PATCHED'); raise SystemExit

edits = []

# --- ATM: cache st_val before latest_close ---
edits.append((
    "        latest_close = candles[-1]['close']\n        logger.info(f\"[NAS-ATM] ST check: close=",
    "        self._atm_naked_st_val = st_val\n        latest_close = candles[-1]['close']\n        logger.info(f\"[NAS-ATM] ST check: close=",
))
# --- ATM: clear cache on candle-close exit ---
edits.append((
    "            logger.warning(f\"[NAS-ATM] ST EXIT! Premium {latest_close:.1f} reversed above ST {st_val:.1f}\")\n            threading.Thread(target=self._fire_atm_st_exit, daemon=True).start()",
    "            logger.warning(f\"[NAS-ATM] ST EXIT! Premium {latest_close:.1f} reversed above ST {st_val:.1f}\")\n            self._atm_naked_st_val = None\n            threading.Thread(target=self._fire_atm_st_exit, daemon=True).start()",
))
# --- ATM: tick-level check in the candle updater else-branch ---
edits.append((
    "        else:\n            cur['high'] = max(cur['high'], ltp)\n            cur['low'] = min(cur['low'], ltp)\n            cur['close'] = ltp\n\n    def _check_atm_st_exit(self):",
    "        else:\n            cur['high'] = max(cur['high'], ltp)\n            cur['low'] = min(cur['low'], ltp)\n            cur['close'] = ltp\n            # tick-level ST exit: don't wait for the 5-min candle close. If the live\n            # premium breaches the last-computed SuperTrend, exit now (fix 2026-06-12).\n            _stv = getattr(self, '_atm_naked_st_val', None)\n            if _stv is not None and ltp > _stv:\n                logger.warning(f\"[NAS-ATM] ST TICK-EXIT: live {ltp:.1f} > ST {_stv:.1f}\")\n                self._atm_naked_st_val = None\n                threading.Thread(target=self._fire_atm_st_exit, daemon=True).start()\n\n    def _check_atm_st_exit(self):",
))

# --- ATM4: cache st_val ---
edits.append((
    "        latest_close = candles[-1]['close']\n        logger.info(f\"[NAS-ATM4] ST check: close=",
    "        self._atm4_naked_st_val = st_val\n        latest_close = candles[-1]['close']\n        logger.info(f\"[NAS-ATM4] ST check: close=",
))
# --- ATM4: clear cache on candle-close exit ---
edits.append((
    "            threading.Thread(\n                target=self._fire_atm4_st_exit,\n                daemon=True\n            ).start()",
    "            self._atm4_naked_st_val = None\n            threading.Thread(\n                target=self._fire_atm4_st_exit,\n                daemon=True\n            ).start()",
))
# --- ATM4: tick-level check in the candle updater else-branch ---
edits.append((
    "        else:\n            cur['high'] = max(cur['high'], ltp)\n            cur['low'] = min(cur['low'], ltp)\n            cur['close'] = ltp\n\n    def _check_atm4_st_exit(self):",
    "        else:\n            cur['high'] = max(cur['high'], ltp)\n            cur['low'] = min(cur['low'], ltp)\n            cur['close'] = ltp\n            _stv = getattr(self, '_atm4_naked_st_val', None)\n            if _stv is not None and ltp > _stv:\n                logger.warning(f\"[NAS-ATM4] ST TICK-EXIT: live {ltp:.1f} > ST {_stv:.1f}\")\n                self._atm4_naked_st_val = None\n                threading.Thread(target=self._fire_atm4_st_exit, daemon=True).start()\n\n    def _check_atm4_st_exit(self):",
))

for i, (old, new) in enumerate(edits, 1):
    assert s.count(old) == 1, 'edit %d count=%d' % (i, s.count(old))
    s = s.replace(old, new, 1)
ast.parse(s)
shutil.copy(P, P + '.bak_fix2tick')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED #2-tick: ST exit now tick-responsive (ATM + ATM4)')
