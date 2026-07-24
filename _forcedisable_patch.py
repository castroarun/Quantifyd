"""2026-06-05: make the squeeze-disable survive boot. _apply_nas_master_mode() force-enables all 8
on the master-mode restore, overriding config 'enabled'. Add a 'force_disabled' marker the squeeze
configs carry, which master-mode honours (never enables). config.py marks the 4 squeeze configs."""
import io, py_compile, sys

# --- 1. config.py: mark the 4 squeeze configs force_disabled ---
cp = "config.py"
s = io.open(cp, encoding="utf-8").read()
anchor = "NAS_916_ATM4_DEFAULTS['enabled'] = True\n"
add = ("NAS_916_ATM4_DEFAULTS['enabled'] = True\n"
       "NAS_DEFAULTS['force_disabled'] = True\n"
       "NAS_ATM_DEFAULTS['force_disabled'] = True\n"
       "NAS_ATM2_DEFAULTS['force_disabled'] = True\n"
       "NAS_ATM4_DEFAULTS['force_disabled'] = True\n")
if "force_disabled" in s:
    print("config already has force_disabled")
elif s.count(anchor) == 1:
    s = s.replace(anchor, add); io.open(cp, "w", encoding="utf-8").write(s); py_compile.compile(cp, doraise=True)
    print("config: 4 squeeze configs marked force_disabled")
else:
    print("ABORT config: anchor count", s.count(anchor)); sys.exit(1)

# --- 2. app.py: _apply_nas_master_mode honours force_disabled ---
ap = "app.py"
a = io.open(ap, encoding="utf-8").read()
old = ("    for cfg in _nas_mode_configs():\n"
       "        if mode == 'off':\n"
       "            cfg['enabled'] = False\n"
       "        elif mode == 'paper':\n"
       "            cfg['enabled'] = True\n"
       "            cfg['paper_trading_mode'] = True\n"
       "        elif mode == 'live':\n"
       "            cfg['enabled'] = True\n"
       "            cfg['paper_trading_mode'] = False")
new = ("    for cfg in _nas_mode_configs():\n"
       "        if cfg.get('force_disabled'):\n"
       "            cfg['enabled'] = False   # permanently disabled (over-margin) — master mode can't enable it\n"
       "            continue\n"
       "        if mode == 'off':\n"
       "            cfg['enabled'] = False\n"
       "        elif mode == 'paper':\n"
       "            cfg['enabled'] = True\n"
       "            cfg['paper_trading_mode'] = True\n"
       "        elif mode == 'live':\n"
       "            cfg['enabled'] = True\n"
       "            cfg['paper_trading_mode'] = False")
if "force_disabled" in a and "cfg.get('force_disabled')" in a:
    print("app.py already honours force_disabled")
elif a.count(old) == 1:
    a = a.replace(old, new); io.open(ap, "w", encoding="utf-8").write(a); py_compile.compile(ap, doraise=True)
    print("app.py: _apply_nas_master_mode honours force_disabled")
else:
    print("ABORT app.py: old block count", a.count(old)); sys.exit(1)

# --- 3. offline-validate: master-mode 'live' must keep squeeze OFF, 916 ON ---
import importlib, config
importlib.reload(config)
import importlib.util
print("=== offline-sim _apply_nas_master_mode('live') ===")
for c in (config.NAS_DEFAULTS, config.NAS_ATM_DEFAULTS, config.NAS_916_OTM_DEFAULTS, config.NAS_916_ATM_DEFAULTS):
    fd = c.get('force_disabled', False)
    enabled = False if fd else True   # what _apply_nas_master_mode('live') will set
    print("  %-26s force_disabled=%s -> enabled=%s" % (c.get('name', '?')[:26], fd, enabled))
print("OK both files patched + py_compile clean")
