"""2026-06-18 (user: "do paper trades for ORB cash and ORB nifty as well"):
turn ON both ORB systems in PAPER mode (no real money).

- ORB Cash  (ORB_DEFAULTS)      : enabled False->True   (paper=True, live=False already)
- ORB Index (STRANGLE_DEFAULTS) : enabled False->True   (engine is paper-ONLY by code)
- All 10 strangle variants      : enabled False->True   (per-variant gate)

No live flags are touched. Strangle has no live order route at all. Guarded:
asserts each edit's match count, ast.parse before write, .bak backup."""
import ast, shutil

P = '/home/arun/quantifyd/config.py'
s = open(P, encoding='utf-8').read()

# 1) ORB Cash — unique by its trailing comment
orb_old = "    'enabled': False,                   # 2026-05-05: re-enabled in PAPER mode (call sites wrapped)"
orb_new = "    'enabled': True,                    # 2026-06-18: ON in PAPER (do paper trades for ORB cash)"
assert s.count(orb_old) == 1, 'ORB_DEFAULTS enabled count=%d' % s.count(orb_old)

# 2) STRANGLE_DEFAULTS — unique by the # Safety block ending in the dict close
str_old = ("    # Safety\n"
           "    'enabled': False,\n"
           "    'paper_trading_mode': True,\n"
           "    'live_trading_enabled': False,\n"
           "}")
str_new = ("    # Safety\n"
           "    'enabled': True,   # 2026-06-18: ON in PAPER (ORB Index/strangle is paper-only by code)\n"
           "    'paper_trading_mode': True,\n"
           "    'live_trading_enabled': False,\n"
           "}")
assert s.count(str_old) == 1, 'STRANGLE_DEFAULTS safety block count=%d' % s.count(str_old)

# 3) the 10 variants — anchor on enabled + the following backtest_wr_pct key
var_old = "        'enabled': False,\n        'backtest_wr_pct'"
var_new = "        'enabled': True,\n        'backtest_wr_pct'"
n = s.count(var_old)
assert n == 10, 'strangle variant enabled count=%d (expected 10)' % n

s = s.replace(orb_old, orb_new, 1)
s = s.replace(str_old, str_new, 1)
s = s.replace(var_old, var_new)  # all 10

ast.parse(s)
shutil.copy(P, P + '.bak_orbpaper')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED config.py: ORB Cash ON (paper), ORB Index ON (paper) + 10 variants enabled')
