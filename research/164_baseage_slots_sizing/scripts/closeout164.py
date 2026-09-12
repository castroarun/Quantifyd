# -*- coding: utf-8 -*-
"""research/164 close-out: STATUS doc to DONE, local .gitignore, research/INDEX.md row,
TODO.md entry, and the dated Ops & Review Center entry."""
import io
import re
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
R = ROOT / 'research/164_baseage_slots_sizing'
STATUS = R / 'BASEAGE_SLOTS_AND_SIZING_DAILY_SWEEP_STATUS.md'

# ------------------------------------------------------------------ 1. gitignore
(R / 'results' / '.gitignore').write_text(
    "# heavy / regenerable — see the STATUS doc's crash-recovery section\n"
    "panel164.pkl\nath_events.csv\ncurves161.npz\nnavs_*/\n*.log\n", encoding='utf-8')

# ------------------------------------------------------------------ 2. STATUS doc
s = STATUS.read_text(encoding='utf-8')
s = s.replace('**STATUS: PRE-REGISTERED — sections 1-4 frozen before the first sweep cell ran**',
              '**STATUS: DONE — 12-Sep-2026 20:1x IST. Verdict CONCLUDED: the inherited '
              '16 slots at 6.25% survives; no spec change before the 26-Sep paper-book call.**')

log_add = """| 2026-09-12 18:53 | Event list frozen | 3,619 adopted-spec events over 1,880 signal days — the SAME count research/163 found independently. Max simultaneous signals on any one day = **13**, median 1, so no day ever carries 16 |
| 2026-09-12 18:53 | **HARNESS PROOF PASSED** | at 5.5% idle cash: 21.26% CAGR / worst seed 19.87% / −34.80% DD / Calmar 0.618 — research/161's published winner to the last digit. At 5.0%: 20.94% / 19.81% / −35.50% / 0.601, invested 72.9% — matches research/163's independent re-run exactly. Cash accrual audited: daily, cash-only, compounded, never taxed |
| 2026-09-12 18:57 | Grid extended 26 → 32 cells | six extra slot counts (7, 9, 11, 13, 14, 18) added to execute the pre-registered PLATEAU test. A tightening of the same axis, not a new one — disclosed in RESULTS.md and in the published caveats |
| 2026-09-12 18:58 | Engine diagnostic added | the first run showed contention behaving oddly across axes, so the engine now also counts entries refused for want of CASH (slot free, no money). This turned out to be the study's biggest finding |
| 2026-09-12 18:59 | Full grid done | 32 cells × 30 seeds in 75s (2 workers, nice 10) |
| 2026-09-12 19:00 | Zero-idle-yield re-run done | 32 cells × 10 seeds — isolates each cell's cash-sleeve contribution |
| 2026-09-12 19:02 | Cost ladder done | 32 cells × 30 seeds at 40 bps and again at 60 bps |
| 2026-09-12 19:05 | Outlier test done | research/161's `outlier_all` product-of-returns reads 1e20× and is not a book multiple; replaced with (a) the ten best trades' share of total RUPEE profit and (b) a fair re-run with those ten EVENTS deleted from the event list across all 30 seeds |
| 2026-09-12 19:1x | RESULTS.md, PUBLISH_NOTE.md written | frontend deliberately NOT touched — another agent held it |
| 2026-09-12 20:1x | **DONE** | verdict CONCLUDED; INDEX, TODO and the Ops & Review Center updated; committed |
"""
s = s.replace("| 2026-09-12 18:5x | Sections 1-4 frozen | this document, before any sweep cell |\n",
              "| 2026-09-12 18:5x | Sections 1-4 frozen | this document, before any sweep cell |\n" + log_add)
s = s.replace('**Phase:** pre-registration complete. Harness proof next.',
              '**Phase:** COMPLETE. All four axes run at 30 seeds, plus the cost ladder, the '
              'zero-yield attribution and the outlier deletion. Verdict written.')

findings = """## 8. Findings

**Verdict: CONCLUDED — the inherited 16 slots at 6.25% survives. No spec change before
26-Sep-2026.** Full write-up and the house YoY table in `results/RESULTS.md`.

1. **The harness is the research/161 book.** Reproduced its published winner exactly at 5.5%
   idle cash (21.26 / −34.80 / 0.618, worst seed 19.87) and research/163's independent 5.0%
   re-run exactly (20.94 / −35.50 / 0.601, worst 19.81, 72.9% invested). 3,619 events, the
   same count research/163 found.
2. **16 slots is not the optimum, but nothing clears the bar.** Calmar has a genuine broad
   hump at 9-12 slots (0.629 / 0.669 / 0.670 / 0.633 vs the incumbent 0.601), worth +1.1 to
   +1.4pp of after-tax CAGR, winning 25-27 of 30 paired seeds and 29-30 of 30 in the fit
   window, on a plateau, surviving 40 and 60 bps. It is below the pre-registered +0.10 Calmar
   / +2pp CAGR bar. 0 of 32 cells clear it.
3. **The bar's blind spots argue the other way.** Concentrating 16 → 10 slots raises the share
   of total profit from the ten best trades from 35.7% to 53.4% and roughly doubles the
   capacity footprint (median position 0.43% → 0.77% of the held name's own 20-day traded
   value; 33% → 46% of trades above 1%; ten times larger again on a ₹1 crore book).
4. **Size per slot is a leverage dial, not hidden alpha.** With the slot count fixed, shrinking
   the position only de-levers along one line. Not one of the 11 eligible cells comes from
   axis B or axis C.
5. **A deliberate cash buffer does not earn its place at 5%.** The best buffered cell (8 slots
   at 6.25%, 45% invested) makes 15.46% / −23.69% / Calmar 0.647, while 11 slots FULLY
   invested gives a better ratio (0.670) AND 6.6pp more return. If Arun ever wants a
   low-drawdown Base Age variant, 10 slots at 4% (12.82% / −17.12% / 0.754) is the cleanest
   point on that frontier — but it is a different product, to be adopted as one.
6. **One contested-slot rule beats the random null consistently: take the most liquid
   candidate.** +2.60pp at 8 slots (30/30 seeds on CAGR and Calmar), +0.78pp at 16 (29/30 on
   both), in both windows. Relative strength wins at 8 slots and LOSES at 16 (5/30) — which
   settles research/160 vs research/158 in favour of research/158 for this book. Least-extended
   is flat; longest-base-age raises CAGR but deepens drawdown and wins Calmar on only 9 of 30.
   The liquidity rule does not clear the bar (+0.064 Calmar) and is 1 of 8 cells on its axis,
   but it is free, deterministic — a live book cannot draw a seed, and the spec gives the
   operator no written tie-break today — and it helps capacity. Worth its own test.
7. **THE BIGGEST FINDING: cash, not slots, is the binding constraint.** At 16 slots, 3,619
   qualifying events produce 688 entries, 977 refused for want of a slot, and **1,955 refused
   for want of cash**. That holds at every slot count. The book never trims a winner, so a few
   bloated positions can absorb 95% of NAV while slots sit nominally free. Contention binds on
   472 of 1,880 signal days at 16 slots, but the book is completely full on only 41 of them.

**Recommended next steps, in order.**

1. **Position drift is the untested first-order knob.** Trim a bloated winner back toward its
   target weight, or size the next entry to available cash rather than skipping it entirely.
   Registered as a dated review for 2026-10-10 in the Ops & Review Center.
2. **Give the liquidity tie-break its own study** — it is the only free change on the table,
   and it removes path randomness a live book cannot reproduce anyway.
3. **Do not re-open the slot count** until (1) is answered. The slot curve measured here is
   the curve of a book that cannot fill its own slots.
"""
s = re.sub(r"## 8\. Findings\n\n\*\(to be filled.*?\)\*\n?", findings, s, flags=re.S)
STATUS.write_text(s, encoding='utf-8')
print('STATUS updated ->', 'DONE' in s.split('\n')[2])

# ------------------------------------------------------------------ 3. research/INDEX.md
idx = ROOT / 'research/INDEX.md'
t = idx.read_text(encoding='utf-8').rstrip('\n')
row = ("| 164 | [**OA · Base Age — how many slots, at what size, and who wins a contested slot?**]"
       "(164_baseage_slots_sizing/) - the 16-slot / 6.25%-per-slot book was never tested for Base "
       "Age; it was INHERITED from the old Open Alpha, whose 680-cell sweep (r/142) was scored "
       "against a same-bar look-ahead entry that r/158 and r/159 showed INVERTS once the entry is "
       "placeable, and r/161's sweep has no slot column at all | 3,619 adopted-spec events, "
       "2005-01-03 -> 2026-09-11, 30 seeds, after tax, 25 bps/side, idle cash 5.0% post-tax; 32 "
       "selection cells on four axes (concentration / cash buffer / size-independent-of-count / "
       "contested-slot rule) | **Harness reproduces r/161 exactly** (21.26/-34.80/0.618 at 5.5%, "
       "20.94/-35.50/0.601 at 5.0%, matching r/163). **16 is NOT the optimum but nothing clears "
       "the bar**: Calmar humps at 9-12 slots (0.629/0.669/0.670/0.633 vs 0.601), +1.1 to +1.4pp "
       "CAGR, 25-27/30 paired seeds, 29-30/30 in the fit window, on a plateau, surviving 40/60 bps "
       "- all below the pre-registered +0.10 Calmar / +2pp CAGR bar; **0 of 32 cells clear it**. "
       "The bar's blind spots argue the OTHER way: 16->10 slots lifts the ten-best-trades share of "
       "profit 35.7%->53.4% and doubles the capacity footprint (median position 0.43%->0.77% of the "
       "name's own 20-day traded value, 33%->46% of trades above 1%, x10 again at Rs1cr). **Size per "
       "slot is a leverage dial** - with the count fixed, shrinking the position only de-levers; not "
       "one of the 11 eligible cells comes from axis B or C. **A cash buffer does NOT earn its place "
       "at 5%**: best buffered cell 15.46%/-23.69%/0.647 vs 11 slots fully invested 22.03%/-33.85%/"
       "0.670. **One contested-slot rule beats the random null consistently - take the most liquid "
       "candidate**: +2.60pp at 8 slots (30/30) and +0.78pp at 16 (29/30), both windows; RS wins at "
       "8 and LOSES at 16 (5/30), settling r/160 vs r/158 in favour of r/158 here; base-age ranking "
       "wins CAGR 30/30 but Calmar only 9/30. **BIGGEST FINDING: cash, not slots, is the binding "
       "constraint** - 3,619 events -> 688 entries, 977 refused for no SLOT, **1,955 refused for no "
       "CASH**, because the book never trims a winner. Position drift is the untested first-order "
       "knob | 2026-09-12 | **CONCLUDED - keep 16 slots at 6.25% into the 26-Sep paper-book call; "
       "re-open only after position drift is tested (review 2026-10-10)** |")
idx.write_text(t + '\n' + row + '\n', encoding='utf-8')
print('INDEX appended')

# ------------------------------------------------------------------ 4. TODO.md
todo = ROOT / 'TODO.md'
t = todo.read_text(encoding='utf-8')
entry = """## ✅ 2026-09-12 — research/164: Open Alpha · Base Age slot count and position size finally tested — **16 × 6.25% survives, no spec change**

Arun: the sixteen-slot, 6.25%-per-slot book was never tested for Base Age. It was **inherited**
from the old Open Alpha, whose 680-cell sweep (research/142) was scored entirely against a
**same-bar look-ahead entry** — and research/158 / research/159 showed those surfaces INVERT
once the entry is made placeable. research/161 swept age, depth, volume, saucer, exits and hard
stops but has **no slot column at all**. That gap is now closed.

**Nothing was deployed or papered. No engine under `services/` was touched, no backend restart,
no frontend change** (another agent held `frontend/` for research/163).

**Harness proof first.** Reproduced research/161's published winner to the last digit at 5.5%
idle cash (21.26% / −34.80% / Calmar 0.618, worst seed 19.87%) and research/163's independent
5.0% re-run exactly (20.94% / −35.50% / 0.601, worst 19.81%, 72.9% invested), on an event list
rebuilt from scratch that contains the same 3,619 events.

| Question | Answer |
|---|---|
| How many slots? | About **ten**, not sixteen — Calmar humps at 9-12 (0.629 / 0.669 / 0.670 / 0.633 vs 0.601). Worth +1.1 to +1.4pp CAGR, 25-27 of 30 paired seeds, both windows, on a plateau, surviving 40 and 60 bps. **Below the pre-registered bar** of +0.10 Calmar or +2pp CAGR. 0 of 32 cells clear it. |
| At what size per slot? | **Keep 6.25%, tied to the slot count.** With the count fixed, shrinking the position only de-levers — not one of the 11 eligible cells comes from that axis. |
| Does a cash buffer help at 5%? | **No.** Best buffered cell 15.46% / −23.69% / 0.647; 11 slots fully invested gives a better ratio (0.670) AND 6.6pp more return. |
| Who wins a contested slot? | **The most liquid candidate** — the only rule that beats the random null at both slot counts (+2.60pp at 8, 30/30 seeds; +0.78pp at 16, 29/30), both windows. Relative strength wins at 8 and LOSES at 16, so it is noise here. |
| Change the spec before 26-Sep? | **No.** Nothing cleared the bar, and the two things the bar does not measure — outlier dependence (ten best trades go from 35.7% to 53.4% of profit) and capacity (median position 0.43% → 0.77% of the name's own traded value, ×10 at ₹1 crore) — both argue against concentrating. |

**The finding that matters most is not on any of the four axes.** At every slot count, the
commonest reason a qualifying signal is NOT taken is that the book **has no cash**, not that it
has no free slot: 3,619 events → 688 entries, 977 refused for want of a slot, **1,955 refused
for want of cash**. The book never trims a winner, so a few bloated positions absorb 95% of NAV
while slots sit nominally free. **Position drift is the untested first-order knob; the slot
count is second-order.**

**Pending — dated review 2026-10-10 (registered in the Ops & Review Center):** test position
drift on Base Age — trim a bloated winner toward its target weight, or size the next entry to
available cash instead of skipping it. Do not re-open the slot count until that is answered.

**Pending — not yet published.** `research/164_baseage_slots_sizing/results/PUBLISH_NOTE.md`
holds the exact `BacktestStudy` entry and the two charts to draw; publish once the other
agent's `frontend/` work is committed, then `cd frontend && npm run build` (frontend-only, safe
any hour).

Evidence: `research/164_baseage_slots_sizing/results/RESULTS.md`.

"""
i = t.find('## ✅ 2026-09-12 — research/163')
todo.write_text(t[:i] + entry + t[i:] if i != -1 else t + '\n' + entry, encoding='utf-8')
print('TODO updated')

# ------------------------------------------------------------------ 5. Ops & Review Center
ops = ROOT / 'research/111_sensex_manual_mgmt/scripts/ops_center.py'
o = ops.read_text(encoding='utf-8')
review = '''REVIEWS = [
    ("research/164 - Open Alpha - Base Age: test POSITION DRIFT, the first-order constraint the "
     "slot study exposed",
     "2026-10-10", "PENDING",
     "research/164 (12-Sep-2026) re-fitted the slot count and position size of Open Alpha - Base "
     "Age, which had been INHERITED from the old Open Alpha's look-ahead-scored sweep and never "
     "tested. VERDICT: 16 slots at 6.25% SURVIVES - the Calmar curve does hump at 9-12 slots "
     "(0.629/0.669/0.670/0.633 vs the incumbent 0.601, worth +1.1 to +1.4pp CAGR on 25-27 of 30 "
     "paired seeds and 29-30/30 in the fit window, on a plateau, surviving 40 and 60 bps), but "
     "that is below the pre-registered bar of +0.10 Calmar or +2pp CAGR, and 0 of 32 cells cleared "
     "it. The bar's blind spots argue the other way: concentrating 16->10 slots lifts the ten "
     "best trades' share of total profit from 35.7% to 53.4% and doubles the capacity footprint "
     "(median position 0.43%->0.77% of the held name's own 20-day traded value, 33%->46% of trades "
     "above 1%, ten times larger again on a Rs 1 crore book). WHY THIS REVIEW EXISTS: the study's "
     "biggest finding is that the slot count is the SECOND-order knob. At every slot count the "
     "commonest reason a qualifying signal is not taken is that the book has no CASH, not that it "
     "has no SLOT - 3,619 qualifying events produce 688 entries, 977 refused for want of a slot "
     "and 1,955 refused for want of cash - because the book never trims a winner, so a few bloated "
     "positions absorb 95% of NAV while slots sit nominally free. TO DO BY THIS DATE: test "
     "position drift on the same harness (research/164_baseage_slots_sizing/scripts/sim164.py, "
     "which reproduces research/161 exactly) - trim a bloated winner back toward its target weight, "
     "and/or size the next entry to available cash instead of skipping it. Do NOT re-open the slot "
     "count until that is answered: the slot curve measured in r/164 is the curve of a book that "
     "cannot fill its own slots. Secondary: give the contested-slot liquidity tie-break its own "
     "test - ranking same-day candidates by 20-day traded value beat the random draw at both slot "
     "counts (+2.60pp at 8 slots on 30/30 seeds, +0.78pp at 16 on 29/30) in both windows, is free, "
     "needs no new data and is deterministic, which matters because a LIVE book cannot draw a seed "
     "and the spec gives the operator no written tie-break today. Evidence: "
     "research/164_baseage_slots_sizing/results/RESULTS.md"),
'''
assert o.count('REVIEWS = [') == 1
o = o.replace('REVIEWS = [', review, 1)
ops.write_text(o, encoding='utf-8')
print('ops_center REVIEWS entry added')
