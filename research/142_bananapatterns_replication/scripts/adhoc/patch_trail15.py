# -*- coding: utf-8 -*-
"""Strategies index + study caveat: trail 20 -> 15 (change-log per binding rule)."""
from pathlib import Path

p = Path('/home/arun/quantifyd/frontend/src/data/strategies.ts')
s = p.read_text(encoding='utf-8')

assert 'SMA20 (trail' in s
s = s.replace('SMA20 (trail — the after-tax winner over SMA50)',
              'SMA15 (trail — after-tax paired winner 03-Sep-2026; previously SMA20)')
if '20-SMA trail' in s:
    s = s.replace('20-SMA trail', '15-SMA trail')

a = "changeLog: [{ date: '3 Sep 2026', text: 'SPEC REVISION after the gate audit:"
b = ("changeLog: [{ date: '3 Sep 2026', text: 'TRAIL 20 -> 15: the pre-declared exit "
     "no-cliff check under the new no-gate/16-slot spec showed trail-15 beats trail-20 "
     "by +1.6-2.0pp AFTER-TAX on 24-26/30 paired seeds with a better worst-seed and "
     "shallower DD (the faster trail earns its churn once gate-filtered entries are gone). "
     "Stop stays -8% (stop axis flat = noise). Book re-seeded (seed 8, 14 open).' }, "
     "{ date: '3 Sep 2026', text: 'SPEC REVISION after the gate audit:")
assert a in s
s = s.replace(a, b)
p.write_text(s, encoding='utf-8')
print('strategies.ts patched')

q = Path('/home/arun/quantifyd/frontend/src/data/backtests.ts')
t = q.read_text(encoding='utf-8')
a2 = 'ADOPTED SPEC NOW: 16 slots @6.25%, NO gate'
assert a2 in t
t = t.replace(a2, 'ADOPTED SPEC NOW: trail-15 SMA (was 20; after-tax paired winner '
                  'under the new regime), -8% stop, 16 slots @6.25%, NO gate')
q.write_text(t, encoding='utf-8')
print('backtests.ts patched')
