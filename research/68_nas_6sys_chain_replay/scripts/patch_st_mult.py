"""research/68 finding -> live: widen the naked-survivor SuperTrend multiplier 2.0 -> 3.0
(period 7, 5-min kept). Phase D showed m3.0 stops whipsawing the profitable survivor out
early: ATM-systems naked contribution ~17k -> ~35k, lower DD, held train+test. All 4 call
sites in nas_ticker are the ATM/ATM4 naked-leg trail (verified). Reversible: 3 -> 2.
Guarded: assert exact count, ast.parse, .bak backup."""
import ast, shutil
P = '/home/arun/quantifyd/services/nas_ticker.py'
s = open(P, encoding='utf-8').read()
OLD = 'period=7, multiplier=2)'
NEW = 'period=7, multiplier=3)'   # research/68 Phase D
n = s.count(OLD)
assert n == 4, 'expected 4 naked-trail ST calls, found %d' % n
s = s.replace(OLD, NEW)
ast.parse(s)
shutil.copy(P, P + '.bak_stmult')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED nas_ticker.py: naked-survivor SuperTrend multiplier 2 -> 3 (4 sites)')
