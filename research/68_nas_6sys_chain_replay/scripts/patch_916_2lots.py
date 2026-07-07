import ast, shutil
P = "/home/arun/quantifyd/config.py"
s = open(P, encoding="utf-8").read()
blocks = [
 ("NAS_916_ATM_DEFAULTS = {\n    **NAS_ATM_DEFAULTS,\n    'entry_start_time': '09:16',\n}",
  "NAS_916_ATM_DEFAULTS = {\n    **NAS_ATM_DEFAULTS,\n    'entry_start_time': '09:16',\n    'lots_per_leg': 2,   # user 2026-07-07: LIVE 2 lots for 9:16 go-live (paper-shadow still paper_lots_per_leg=10)\n}"),
 ("NAS_916_ATM2_DEFAULTS = {\n    **NAS_ATM2_DEFAULTS,\n    'entry_start_time': '09:16',\n}",
  "NAS_916_ATM2_DEFAULTS = {\n    **NAS_ATM2_DEFAULTS,\n    'entry_start_time': '09:16',\n    'lots_per_leg': 2,   # user 2026-07-07: LIVE 2 lots\n}"),
 ("NAS_916_ATM4_DEFAULTS = {\n    **NAS_ATM4_DEFAULTS,\n    'entry_start_time': '09:16',\n}",
  "NAS_916_ATM4_DEFAULTS = {\n    **NAS_ATM4_DEFAULTS,\n    'entry_start_time': '09:16',\n    'lots_per_leg': 2,   # user 2026-07-07: LIVE 2 lots\n}"),
]
for old, new in blocks:
    assert s.count(old) == 1, "anchor count=%d for %r" % (s.count(old), old[:40])
    s = s.replace(old, new, 1)
ast.parse(s)
shutil.copy(P, P + ".bak_916_2lots")
open(P, "w", encoding="utf-8").write(s)
print("PATCHED: 916 ATM/ATM2/ATM4 lots_per_leg=2 (live)")
