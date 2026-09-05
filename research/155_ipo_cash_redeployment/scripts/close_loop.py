"""research/155 — close the loop: research/INDEX.md row, TODO.md entry, Ops & Review Centre
dated review. Idempotent."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
IDX = ROOT / "research" / "INDEX.md"
TODO = ROOT / "TODO.md"
OPS = ROOT / "research" / "111_sensex_manual_mgmt" / "scripts" / "ops_center.py"

ROW = (
    "| 155 | [Redeploying the IPO sleeve's idle cash into OA / TN](155_ipo_cash_redeployment/) "
    '- Arun: "we would have an idea beforehand if there are any listings or potential candidates '
    "that might meet our criteria at all, in which case we can look at deploying the idle cash "
    '... and then a mechanism to pull back money". Position-level sleeve simulation with an '
    "external cash sink/source; arms = idle to cash (incumbent) / OA / TN / 50-50 / NIFTYBEES "
    "(null) / a forward-visibility GATE that parks only while no candidate can exist for N "
    "{25,50,100} trading days; every pull-back friction charged (25/40/60 bps both ways, tax on "
    "the realised gain with FY netting, T+1 settlement so the entry that forced the sale is "
    "MISSED, and pro-rata / LIFO / FIFO lot policy), plus reserve {0,1,2 slots} x cadence "
    "{daily, weekly, monthly} | daily 2006-01 -> 2026-09-04; **114 cells** x 30 PAIRED paths "
    "(30 OA seeds x 30 IPO seeds x 12 TN offsets cycled) = 3,420 position-level sleeve "
    "simulations; after tax, idle cash 5%; replication gate PASSED BIT-FOR-BIT against "
    "r/153 | **Arun's premise is confirmed** (25 bars + 25-day base makes a 2-day-old listing "
    "ineligible for ~5 weeks, so the next 25 sessions are fully visible with NO look-ahead) "
    "**and the mechanism works** (0 missed entries, ~30 pull-backs in 20 years) **but there is "
    "no room for it to matter**: the sleeve is 20% of the blend and 67.3% cash, the pool is "
    "empty on only 19.0% of days (identically for N=25/50/100 - droughts last months), so the "
    "gate touches 2.7% of the portfolio and buys **+0.105pp CAGR (30/30 paths) but only +0.006 "
    "Calmar (21/30)** against a pre-registered +0.10 on >=26/30; it is **gone by 40 bps and "
    "negative by 60**. Redeploying the whole 13.5% continuously DOES move it - **+1.54pp CAGR "
    "for -3.85pp of drawdown, Calmar -0.375 on 30/30**, corr to OA **0.21 -> 0.90**. Friction "
    "drag **0.28pp** (gated, 73% of the gross benefit) and **5.26pp** (continuous, flips +3.30 "
    "to -1.95) - but frictionless continuous parking still loses 28/30. A plain STATIC **TN35 / "
    "OA35 / IPO30** returns 29.39% at -13.64% vs the gated machine's 29.02% at -13.66%. The "
    "idle cash is the sleeve's drawdown brake, not dead weight | **CONCLUDED - the idle cash "
    "stays in cash; no change to r/153's spec** |"
)

s = IDX.read_text(encoding="utf-8")
if "155_ipo_cash_redeployment" not in s:
    IDX.write_text(s.rstrip("\n") + "\n" + ROW + "\n", encoding="utf-8")
    print("INDEX.md row appended")
else:
    print("INDEX.md already has r/155")

# ───────────────────────────────────────────────────────────────────── TODO.md
TODO_BLOCK = """
## Done - 2026-09-05 - research/155 IPO idle-cash redeployment

- **CONCLUDED - the idle cash stays in cash.** Tested Arun's proposal to park the IPO sleeve's
  idle cash in Open Alpha / True North during listing droughts and pull it back when supply
  returns, with every pull-back friction modelled (25/40/60 bps both ways, tax on the realised
  gain with FY netting, T+1 settlement, pro-rata/LIFO/FIFO lot policy). The premise is
  CONFIRMED and the mechanism WORKS (0 missed entries in 20 years), but it can only touch 2.7%
  of the portfolio and buys +0.105pp CAGR / +0.006 Calmar - gone by 40 bps. Continuous
  redeployment costs 0.375 of Calmar and takes the sleeve's correlation to Open Alpha from
  0.21 to 0.90. A plain static TN35/OA35/IPO30 beats the whole mechanism.
  Published: `/app/backtest/ipo-idle-cash-redeployment-research155`.
  Nothing deployed; research/153's spec unchanged.
- **Dated review registered:** 31-Mar-2027 - revisit only if the IPO sleeve's weight exceeds
  30% or the pipeline has been in drought for more than 12 consecutive months.
"""

t = TODO.read_text(encoding="utf-8")
if "research/155" not in t:
    TODO.write_text(t.rstrip("\n") + "\n" + TODO_BLOCK, encoding="utf-8")
    print("TODO.md updated")
else:
    print("TODO.md already mentions research/155")

# ──────────────────────────────────────────────────── Ops & Review Centre (REVIEWS)
NOTE = (
    "research/155 CONCLUDED: leave the IPO sleeve's idle cash in cash (5% p.a.). Arun's "
    "forward-visibility proposal was built and tested properly - 114 cells x 30 paired paths, "
    "every pull-back friction charged (25/40/60 bps both ways, tax on the realised gain with FY "
    "netting, T+1 settlement, pro-rata/LIFO/FIFO lot policy). The premise is CONFIRMED (25 bars "
    "plus a 25-day base make a 2-day-old listing ineligible for about five weeks, so the next 25 "
    "sessions are fully visible with NO look-ahead) and the mechanism WORKS (0 missed entries, "
    "about 30 pull-backs in 20 years) - but it can only touch 2.7% of the portfolio, because the "
    "sleeve is 20% of the blend, is 67.3% cash, and the candidate pool is empty on only 19.0% of "
    "days. It buys +0.105pp blend CAGR (30/30 paths) but only +0.006 Calmar (21/30) against a "
    "pre-registered bar of +0.10 on at least 26 of 30, and the advantage is gone by 40 bps and "
    "negative by 60. Continuous redeployment is worse: -0.375 Calmar on 30/30 and correlation to "
    "Open Alpha 0.21 -> 0.90. REVISIT ONLY IF (a) the IPO sleeve's blend weight exceeds 30%, or "
    "(b) the IPO pipeline has been in drought for more than 12 consecutive months. Then re-run "
    "research/155 phase 3b at the then-current weights; PASS = the gated arm clears +0.10 blend "
    "Calmar on at least 26 of 30 paired paths AND survives the 40 bps rung. Artifacts: "
    "research/155_ipo_cash_redeployment/results/{RESULTS.md, paths.csv, cost_ladder.csv, "
    "static_tilt_null.csv}. Published at "
    "/app/backtest/ipo-idle-cash-redeployment-research155."
)

o = OPS.read_text(encoding="utf-8")
if "research/155" not in o:
    anchor = "REVIEWS = ["
    assert anchor in o, "REVIEWS anchor not found in ops_center.py"
    nl = chr(10)
    q = chr(34)
    entry = (
        anchor + nl
        + "    (" + q + "research/155 - IPO sleeve idle-cash policy re-check" + q + "," + nl
        + "     " + q + "2027-03-31" + q + ", " + q + "SCHEDULED" + q + "," + nl
        + "     " + q + NOTE.replace(q, "'") + q + ")," + nl
    )
    o = o.replace(anchor + nl, entry, 1)
    OPS.write_text(o, encoding="utf-8")
    print("ops_center.py REVIEWS entry added")
else:
    print("ops_center.py already has the r/155 review")
