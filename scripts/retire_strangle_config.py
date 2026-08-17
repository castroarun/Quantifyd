#!/usr/bin/env python3
"""Disable the ORB-index strangle at its SOURCE (config.py).

The engine reads `enabled` from the in-memory STRANGLE_VARIANTS configs and
pushes that value INTO the DB at boot — so flipping the DB flag alone is
undone on the next restart. This turns the master switch off and every
per-variant flag inside STRANGLE_VARIANTS.
"""
import re
from pathlib import Path

P = Path('/home/arun/quantifyd/config.py')
s = P.read_text()

RETIRED = ("  # RETIRED 2026-08-17: 342 trades, net -Rs 1.97L, all 10 variants "
           "negative, gross negative before costs (paper book, no money lost)")

# 1. master switch for the strangle system
old_master = ("    'enabled': True,   # 2026-06-18: ON in PAPER "
              "(ORB Index/strangle is paper-only by code)")
new_master = "    'enabled': False," + RETIRED
assert old_master in s, 'strangle master switch not found'
s = s.replace(old_master, new_master, 1)

# 2. every per-variant flag inside the STRANGLE_VARIANTS list
start = s.index('STRANGLE_VARIANTS = [')
end = s.index('\n]', start) + 2
block = s[start:end]
block_new, n = re.subn(r"'enabled': True,", "'enabled': False,  # retired 2026-08-17",
                       block)
s = s[:start] + block_new + s[end:]
P.write_text(s)
print(f'master switch OFF; per-variant flags disabled: {n}')
