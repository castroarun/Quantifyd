"""Update the NAS page (frontend/src/pages/Nas.tsx) so the DISPLAYED rules match what now
runs live (research/68): ATM2 = +-0.4% move-stop one-and-done (was cascade+re-entry); naked
SuperTrend trail mult 2 -> 3. Run ON THE VPS (authoritative frontend), then npm build there."""
import ast as _ast  # noqa (unused; keep import style consistent)
P = '/home/arun/quantifyd/frontend/src/pages/Nas.tsx'
s = open(P, encoding='utf-8').read()
orig = s

# 1) naked-trail SuperTrend multiplier 2 -> 3 (rules strings + tooltip)
n_st = s.count('ST(7,2)')
assert n_st == 4, 'expected 4 ST(7,2), found %d' % n_st
s = s.replace('ST(7,2)', 'ST(7,3)')
assert s.count('SuperTrend(7,2)') == 1
s = s.replace('SuperTrend(7,2)', 'SuperTrend(7,3)')

# 2) ATM2 squeeze card
reps = [
    ("    subtitle: 'Cascading ATM, 4-24 premium bounds',",
     "    subtitle: '±0.4% underlying move-stop, one-and-done',"),
    ("      'Entry: ATR squeeze → SELL ATM CE+PE, SL = 1.3x. Any SL closes BOTH legs and re-enters a new ATM strangle. Max 5 re-entries. EOD 15:15.',",
     "      'Entry: ATR squeeze → SELL ATM CE+PE. Exit: ±0.4% underlying move from entry closes BOTH legs (one-and-done, NO re-entry); per-leg 1.3x (30%) SL is a backstop. EOD 15:15.',"),
    ("    configNote: 'ATM 2.0: 5L | 1.3x SL | 5 re-entries',",
     "    configNote: 'ATM 2.0: 5L | ±0.4% move-stop | 30% SL backstop',"),
    # 3) ATM2 916 card
    ("    subtitle: '9:16 entry, cascading ATM 2.0',",
     "    subtitle: '9:16 entry, ±0.4% move-stop (one-and-done)',"),
    ("      'Entry: Auto-enter at 9:16 AM. SELL ATM CE+PE, SL = 1.3x. Any SL closes BOTH legs and re-enters a new ATM strangle. Max 5 re-entries. EOD 15:15.',",
     "      'Entry: Auto-enter at 9:16 AM. SELL ATM CE+PE. Exit: ±0.4% underlying move from entry closes BOTH legs (one-and-done, NO re-entry); per-leg 1.3x (30%) SL is a backstop. EOD 15:15.',"),
    ("    configNote: '916 ATM 2.0: 5L | 1.3x SL | 5 re-entries',",
     "    configNote: '916 ATM 2.0: 5L | ±0.4% move-stop | 30% SL backstop',"),
]
for old, new in reps:
    assert s.count(old) == 1, 'rule not found uniquely: %r (count=%d)' % (old[:50], s.count(old))
    s = s.replace(old, new, 1)

assert s != orig
import shutil
shutil.copy(P, P + '.bak_uirules')
open(P, 'w', encoding='utf-8').write(s)
print('PATCHED Nas.tsx: ATM2 move-stop rules + ST(7,3) trail (display now matches live)')
