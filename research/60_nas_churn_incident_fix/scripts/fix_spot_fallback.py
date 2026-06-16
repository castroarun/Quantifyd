"""FIX 2026-06-16: the 09:16 one-shot 916 entry missed on ATM/ATM4 because
self.scanner.get_live_spot() returned None for a moment (transient ticker-spot gap) and
the code gave up ('Cannot fetch live spot price'). A one-shot must not miss on a blip.
Fall back to a direct REST quote of the NIFTY index. In nas_atm_executor (covers all
ATM/916 variants). Guarded."""
import ast, shutil
P = '/home/arun/quantifyd/services/nas_atm_executor.py'
s = open(P, encoding='utf-8').read()
if 'spot REST fallback' in s:
    print('ALREADY PATCHED'); raise SystemExit
OLD = (
    "        # Get current spot if not provided\n"
    "        if spot is None:\n"
    "            spot = self.scanner.get_live_spot()\n"
    "            if spot is None:\n"
    "                return None, 'Cannot fetch live spot price'\n"
)
NEW = (
    "        # Get current spot if not provided\n"
    "        if spot is None:\n"
    "            spot = self.scanner.get_live_spot()\n"
    "            if not spot:\n"
    "                # fallback: direct REST quote of the NIFTY index, so the 09:16 one-shot\n"
    "                # doesn't miss on a transient ticker-spot gap (2026-06-16 ATM/ATM4 missed).\n"
    "                try:\n"
    "                    from services.kite_service import get_kite\n"
    "                    spot = get_kite().ltp(['NSE:NIFTY 50'])['NSE:NIFTY 50']['last_price']\n"
    "                    logger.info(f\"[NAS-ATM] spot REST fallback used: {spot}\")\n"
    "                except Exception as _e:\n"
    "                    logger.warning(f\"[NAS-ATM] spot REST fallback failed: {_e}\")\n"
    "                    spot = None\n"
    "            if not spot:\n"
    "                return None, 'Cannot fetch live spot price'\n"
)
assert s.count(OLD) == 1, 'OLD count=%d' % s.count(OLD)
s = s.replace(OLD, NEW, 1)
ast.parse(s)
shutil.copy(P, P + '.bak_spotfb')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED #2: 916/ATM entry falls back to REST NIFTY quote on a ticker-spot gap')
