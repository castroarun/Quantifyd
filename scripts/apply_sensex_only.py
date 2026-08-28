#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply Arun's 2026-08-27 directive: LIVE = SENSEX suite + TimeB SENSEX, Wed/Thu only.

Deferred to after Friday 2026-08-28's close, because Arun asked that Friday still run on the
plan frozen before 27-Aug. From Monday 31-Aug the restriction stands:

  LIVE   sensex_atm / atm2 / atm4  (2L each, Wed=DTE1, Thu=DTE0) - the ONLY live book.
         TimeB was removed from live on 2026-08-28 (risk/reward), paper continues.
  PAPER  everything else - the NIFTY 9:16 suite keeps entering daily via paper_shadow, and
         each CSL book keeps trading its own cells on paper for the record.
"""
import json

Q = "/home/arun/quantifyd/"
done = []

p = Q + "backtest_data/nas_day_matrix.json"
mx = json.load(open(p))
for k in ("nas_916_atm", "nas_916_atm2", "nas_916_atm4"):
    if mx["systems"].get(k, {}).get("live"):
        mx["systems"][k]["live"] = False
        mx["systems"][k]["paper_shadow"] = True
        done.append("matrix %-14s live -> False" % k)
for k in ("sensex_atm", "sensex_atm2", "sensex_atm4"):
    mx["systems"].setdefault(k, {})["paper_shadow"] = True
json.dump(mx, open(p, "w"), indent=1)

p = Q + "research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py"
KEEP = "__none__"   # 2026-08-28: TimeB is no longer live on any venue
TO_PAPER = ("CSL_TIMEB_NIFTY", "CSL_TIMEB_NIFTY_MON_AM", "CSL_TIMEB2_LIVE",
            "NAS_COMB20", "CSL_TIMEB_NIFTY_THU", "CSL30F_SENSEX_WED",
            "CSL_TIMEB_SENSEX")
out = []
for ln in open(p, encoding="utf-8").readlines():
    hit = next((b for b in TO_PAPER if ('"%s": {' % b) in ln), None)
    if hit and '"mode": "live"' in ln and ('"%s": {' % KEEP) not in ln:
        out.append(ln.replace(', "mode": "live"', '').replace('"mode": "live", ', ''))
        done.append("BOOKS  %-24s live -> paper" % hit)
    else:
        out.append(ln)
open(p, "w", encoding="utf-8").write("".join(out))
print("\n".join(done) if done else "nothing to change (already applied)")
