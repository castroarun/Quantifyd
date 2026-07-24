"""Verify per-variant LIVE/PAPER posture for the next Monday session (DTE=1)."""
from datetime import date
from services import nas_day_matrix as M
import config as C

mon = date(2026, 6, 15)
print("Monday 2026-06-15  trading_dte =", M.trading_dte(mon), " master_mode =", M._master_mode())
try:
    print("matrix system keys:", list(M.load().keys()))
except Exception as e:
    print("matrix load err:", e)
print()

defaults = {
    "nas": "NAS_DEFAULTS", "nas_atm": "NAS_ATM_DEFAULTS", "nas_atm2": "NAS_ATM2_DEFAULTS",
    "nas_atm4": "NAS_ATM4_DEFAULTS", "nas_916_otm": "NAS_916_OTM_DEFAULTS",
    "nas_916_atm": "NAS_916_ATM_DEFAULTS", "nas_916_atm2": "NAS_916_ATM2_DEFAULTS",
    "nas_916_atm4": "NAS_916_ATM4_DEFAULTS",
}
hdr = "%-14s %-7s %-8s %-12s | %-6s %-6s -> %s"
print(hdr % ("system", "enabled", "force_pp", "matrix_key", "gate", "mode", "EFFECTIVE Monday"))
for key, dname in defaults.items():
    cfg = getattr(C, dname, None)
    if cfg is None:
        print("%-14s (no %s in config)" % (key, dname))
        continue
    en = cfg.get("enabled", True)
    fp = cfg.get("force_paper", False)
    mk = cfg.get("matrix_key", key)
    try:
        g = M.gate(mk, today=mon)
        allow = g.get("allow")
        gmode = g.get("mode") or g.get("_force_mode")
    except Exception as e:
        allow = "ERR"
        gmode = str(e)[:30]
    if fp:
        eff = "PAPER (force_paper)"
    elif not allow:
        eff = "no entry Monday"
    elif gmode == "live":
        eff = "*** LIVE (real money) ***"
    else:
        eff = "PAPER"
    print(hdr % (key, en, fp, mk, allow, gmode, eff))
