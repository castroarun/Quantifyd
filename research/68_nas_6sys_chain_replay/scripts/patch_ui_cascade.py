import shutil
P='/home/arun/quantifyd/frontend/src/pages/Nas.tsx'
s=open(P,encoding='utf-8').read(); orig=s
reps=[
 ("    subtitle: '±0.4% underlying move-stop, one-and-done',",
  "    subtitle: 'Cascading ATM — ±0.4% move-stop, re-center to new CMP',"),
 ("      'Entry: ATR squeeze → SELL ATM CE+PE. Exit: ±0.4% underlying move from entry closes BOTH legs (one-and-done, NO re-entry); per-leg 1.3x (30%) SL is a backstop. EOD 15:15.',",
  "      'Entry: ATR squeeze → SELL ATM CE+PE. Exit: ±0.4% underlying move closes BOTH legs AND re-enters (cascades) at the new ATM with the same ±0.4% stop. Move-stop is the sole trigger (no per-leg SL). Max 5 re-centers/day. EOD 15:15.',"),
 ("    configNote: 'ATM 2.0: 5L | ±0.4% move-stop | 30% SL backstop',",
  "    configNote: 'ATM 2.0: 5L | ±0.4% move-stop + re-center | max 5/day',"),
 ("    subtitle: '9:16 entry, ±0.4% move-stop (one-and-done)',",
  "    subtitle: '9:16 entry, ±0.4% move-stop → re-center (cascade)',"),
 ("      'Entry: Auto-enter at 9:16 AM. SELL ATM CE+PE. Exit: ±0.4% underlying move from entry closes BOTH legs (one-and-done, NO re-entry); per-leg 1.3x (30%) SL is a backstop. EOD 15:15.',",
  "      'Entry: Auto-enter at 9:16 AM. SELL ATM CE+PE. Exit: ±0.4% underlying move closes BOTH legs AND re-enters (cascades) at the new ATM with the same ±0.4% stop. Move-stop is the sole trigger (no per-leg SL). Max 5 re-centers/day. EOD 15:15.',"),
 ("    configNote: '916 ATM 2.0: 5L | ±0.4% move-stop | 30% SL backstop',",
  "    configNote: '916 ATM 2.0: 5L | ±0.4% move-stop + re-center | max 5/day',"),
]
for old,new in reps:
    assert s.count(old)==1,'count=%d for %r'%(s.count(old),old[:45])
    s=s.replace(old,new,1)
assert s!=orig
shutil.copy(P,P+'.bak_cascade_ui'); open(P,'w',encoding='utf-8').write(s)
print('PATCHED Nas.tsx: ATM2 cards now describe the cascade (move-stop + re-center)')
