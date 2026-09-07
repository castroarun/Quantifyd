"""r/156 close the loop: research/INDEX.md row, TODO.md entry, Ops & Review Centre entry."""
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")

# ---------------------------------------------------------------- research/INDEX.md
IDX = ROOT / "research/INDEX.md"
ROW = ("| 156 | [Sector trend detection - rotation across sectors, and sector-gated stock "
       "picking](156_sector_rotation/) - Arun: \"we have many sector indexes ... is there a way to "
       "figure out the trending(s), to-be trending sector(s), either ride on them in some "
       "proportions or further drill down into those sector leaders stocks ... above 20% cagr, "
       "even up to or better than our current oa and/or tn ... do not hv any bias to any existing "
       "systems\". Twelve trend/strength families from first principles (incl. acceleration and "
       "constituent breadth as the \"to-be trending\" candidates) x 1,440 rotation configs x 4 "
       "rebalance-day offsets on the 9 real NSE sector indices, plus a 500-draw random-sector "
       "null, plus a 768-book sector-gated STOCK branch with three paired controls. ~20,300 "
       "cells. | 2016-2026 (real indices start 2015) | **0 of 1,440 configs clear the 20%-CAGR "
       "bar**; best 16.3%/-36.9%/Calmar 0.44 vs equal-weight-all-9 14.0%/0.32, NIFTY500 B&H "
       "13.2%/0.35 and **Midcap150 B&H 18.0%/0.41 which beats every cell**. Momentum ranks "
       "sectors at the 94th-100th pct of a random-sector null yet still loses to holding "
       "everything (the r/63 lesson again). \"To-be trending\" is empty: acceleration t=1.37/1.79, "
       "breadth-change t=0.36. BRANCH B: sector-gated stock book 32.5%/0.85 but the SAME stock "
       "rule with NO sector filter gives 32.9%/0.84 and wins 11 of 16 paired offsets - the sector "
       "layer is a round trip to nowhere. DATA FINDING: synthetic sector proxies from today's "
       "index membership out-drift the real indices by +4 to +14pp CAGR/yr and **shuffled industry "
       "labels reproduce most of the apparent sector momentum**. No complement value: corr "
       "0.41-0.54 to the live legs (bar 0.40), best blend gain +0.04 Calmar, beaten by a cash "
       "sleeve. r/147's SECROT cell reproduces after tax at 8.9%/-56.0%/0.16. Published: "
       "/app/backtest/sector-trend-rotation-research156 | **NO EDGE (rotation) / NO ADDED VALUE "
       "(sector filter)** |")

t = IDX.read_text(encoding="utf-8")
if "156_sector_rotation" not in t:
    t = t.rstrip("\n") + "\n" + ROW + "\n"
    IDX.write_text(t, encoding="utf-8")
    print("INDEX.md row appended")
else:
    print("INDEX.md already has r/156")

# ------------------------------------------------------------------------- TODO.md
TODO = ROOT / "TODO.md"
ENTRY = """## ✅ 2026-09-07 — research/156 sector trend: NO EDGE (rotation) / NO ADDED VALUE (sector filter)

Full verdict: `research/156_sector_rotation/results/RESULTS.md` · study page
`/app/backtest/sector-trend-rotation-research156`.

Arun asked whether the sector indices can be read for trend — ride the leaders in proportions, or
drill into their leader stocks for a curated book above 20% CAGR — and whether it complements
TN+OA. Both branches were built with **no inherited design** from OA or TN and both failed.

- **Branch A (allocate across sectors): 0 of 1,440 configurations** clear the bar. Best 16.3%
  CAGR / −36.9% DD / Calmar 0.44 after tax. Equal-weighting all nine sectors gives 14.0% / 0.32;
  **Midcap 150 buy-and-hold gives 18.0% / 0.41 and beats every cell we built.** Momentum does rank
  sectors better than chance (94th–100th percentile of a 500-draw random-sector null) — and the
  ranking is worth less than the diversification it destroys. That is the r/63 lesson again.
- **"To-be trending" is empty.** Acceleration t = 1.37 (1m) / 1.79 (3m); breadth-change t = 0.36.
  Nothing anticipates leadership.
- **Branch B (sector as a universe filter): the sector layer adds nothing.** The sector-gated stock
  book returns 32.5% / Calmar 0.85 — and the *identical stock rule with no sector filter at all*
  returns 32.9% / 0.84 and wins 11 of 16 paired offsets. All the return is stock momentum, which
  TN and OA already harvest.
- **No complement value.** Correlation 0.41–0.54 to the live legs (ceiling 0.40); best blend gain
  +0.04 Calmar at slightly lower CAGR; **a plain cash sleeve at the same weight beats every
  candidate.**
- **Confirms and extends r/147** — its single SECROT cell reproduces after tax at 8.9% / −56.0% /
  0.16, the worst book in the study.

**A reusable data warning was produced and belongs to future studies, not just this one:** synthetic
sector proxies built from today's index membership out-drift the real sector indices by **+4 to
+14pp of CAGR per year**, and **shuffled industry labels reproduce most of the apparent "sector
momentum". Any future sector work must run the same-universe head-to-head and the shuffled-label
null before believing a wide-panel result.**

### Pending — the only follow-up worth having

**Back-fill the nine real NSE sector indices to their 2005 inception** (Ops review 2027-03-07).
The database starts them at 2015, which caps the sample at 11.7 years with one crash and no 2008.
NSE publishes the history. This is a data-acquisition task, not a modelling one, and it is the only
thing that could reopen this line honestly. It would also improve any future sector-aware work
(regime gates, sector caps on TN/OA) regardless of this verdict.

"""
t = TODO.read_text(encoding="utf-8")
if "research/156" not in t:
    marker = "## ✅ 2026-09-06"
    i = t.index(marker)
    t = t[:i] + ENTRY + "\n" + t[i:]
    TODO.write_text(t, encoding="utf-8")
    print("TODO.md entry inserted")
else:
    print("TODO.md already has r/156")

# --------------------------------------------------------------- ops_center REVIEWS
OPS = ROOT / "research/111_sensex_manual_mgmt/scripts/ops_center.py"
REVIEW = '''    ("Sector indices - back-fill to 2005 inception, then re-ask research/156",
     "2027-03-07", "PENDING",
     "research/156 found NO EDGE in sector rotation and NO ADDED VALUE in sector-gated stock "
     "picking, but the sample is the binding constraint: the 9 real NSE sector indices start "
     "2015-01-01 in market_data.db, giving 11.7 years, one crash (2020) and no 2008. NSE "
     "publishes these series back to ~2005. TASK: back-fill NIFTYAUTO/IT/ENERGY/FINSRV/FMCG/"
     "METAL/PHARMA/PSUBANK/REALTY to inception, then re-run research/156's p1_ic.py and "
     "p2_rotation.py unchanged (both are resume-safe). PASS CRITERION for reopening the line: "
     "any signal family reaching abs(IC t) >= 2.5 on the real indices over the LONGER window "
     "with monotone terciles, AND a rotation config clearing 20% CAGR / Calmar 1.0 after tax. "
     "Otherwise the NO EDGE verdict stands permanently and this review closes. Study: "
     "/app/backtest/sector-trend-rotation-research156."),
'''
t = OPS.read_text(encoding="utf-8")
if "research/156" not in t:
    anchor = "REVIEWS = [\n"
    t = t.replace(anchor, anchor + REVIEW, 1)
    OPS.write_text(t, encoding="utf-8")
    print("ops_center REVIEWS entry added")
else:
    print("ops_center already has r/156")
