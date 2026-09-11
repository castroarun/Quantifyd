# -*- coding: utf-8 -*-
"""research/162 — close the loop: research index row, TODO entry, labs-reference mirror.

Idempotent: each write is guarded by a marker so re-running changes nothing.
"""
import io
import sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]

INDEX_ROW = (
    '| 162 | [**Quality Summit optimised, and the quality screen tested INSIDE Base Age**]'
    '(162_quality_summit_optimisation/results/RESULTS.md) — Arun: *"go"* — make Quality Summit '
    '(r/160 Family B `b7`: near its own ATH at k=0.90, profitable, mcap >₹1,000cr, ROE and ROCE '
    '>15, growth >10, no debt test, 15 names by RS, monthly, no exit — 21.19% after tax / −37.1% '
    '/ Calmar 0.58) earn more and fall less, and run r/160\'s 10-Oct-2026 overlay review now. '
    'Attacked the three axes r/160 never tried: **ATR-scaled trails** (SuperTrend 7/3, 10/3, 14/4, '
    '20/3 and chandelier 22-day high −2 and −3 ATR14, each ± a fundamental-failure exit), '
    '**ranking axes** (3y profit growth, 3y OPM slope, two z-composites with RS, market cap) and '
    '**inverse-vol sizing**; sector cap DROPPED because no sector field exists anywhere in the '
    'project. Fit window W1 2018-08→2022-06, holdout W2 2022-07→2026-09, and the 4pp fit-to-holdout '
    'gap limit, all pre-registered before the first cell. **PART A: the best cell looked like a '
    'discovery and was not** — keep the screen, widen the band to k=0.85, cut to 10 names, size '
    'inverse-vol: **+6.06pp CAGR and +0.282 Calmar on 12 of 12 offsets in the fit window**, on a '
    'verified 24-cell plateau, surviving the cost ladder — then **−3.48pp on 3/12 and −0.152 Calmar '
    'on 1/12 in the holdout**, W2 running 9.22pp below W1, and its compounding proxy falls **below '
    'one** when its ten best trades are deleted. Both plateau neighbours fail identically, so it is '
    'the construction. Best trail worth +0.06 Calmar at −1.0pp CAGR and partly idle cash; '
    '`fund_fail` changes literally nothing; **every fundamental ranking axis loses by 2-22pp**. '
    'The screen\'s real product is **drawdown, not return**: −12.7 to −16.9pp of MaxDD on 12/12 '
    'offsets for −1.3pp of CAGR. **PART B (the 10-Oct review, done five weeks early, on r/161\'s '
    'byte-identical engine whose control reproduces 21.26 / −34.80 / 0.618 exactly): NOT ONE SCREEN '
    'WINS ON A SINGLE SEED OF THIRTY** on return, either window, either missing policy — b7 '
    '−9.71pp, quality-only −11.28pp, growth-only −5.80pp, the screen as written −17.77pp at 18.8% '
    'invested. Mechanism is starvation: qualifying signals 3,619 → 468 → 76, invested 87% → 63% → '
    '19%. **PART C: dilutive** — cash at the same weight beats it on **360/360 paths** at 10/20/33%, '
    'monthly correlation 0.717 to Base Age. 193 cells + 14 blends disclosed. Published '
    '`/app/backtest/quality-summit-optimisation-research162`. | 2026-09-12 | **A: CONCLUDED (no '
    'adoption) · B: NO EDGE (quality-screen line closed) · C: DILUTIVE** |\n')

TODO_ENTRY = '''## ✅ 2026-09-12 — research/162: Quality Summit could NOT be improved, and the quality screen does NOT belong inside Base Age

Arun said **"go"** at ~23:35 on 11-Sep and asked for the overlay review that research/160 had
booked for **10-Oct-2026** to be done now. Both were done the same night. **Nothing is deployed,
papered, or changed** — no engine, no live book, no `strategies.ts`, no `/app/mpf-report`. The one
operational consequence is subtraction: the October review slot is freed.

**Part A — can Quality Summit earn more and fall less? CONCLUDED, no adoption.**
The incumbent (r/160 Family B `b7`, k=0.90, 15 names) stands: **21.19% after tax / −37.1% /
Calmar 0.58**, reproduced bit-identically before anything was changed. The three axes r/160 never
tried were swept — ATR trails, alternative ranking axes, inverse-vol sizing (the sector cap was
dropped: **no sector field exists anywhere in this project**). The best cell — keep the screen,
widen the near-ATH band from 0.90 to 0.85, cut to ten names, size inverse-vol — beat the incumbent
by **+6.06pp CAGR and +0.282 Calmar on 12 of 12 rebalance offsets** in the fit window, on a
verified 24-cell plateau. **The pre-registered holdout returned −3.48pp on 3 of 12 and −0.152
Calmar on 1 of 12**, 9.22pp below its fit window against a 4pp limit written down in advance, and
it loses money without its ten best trades. Both plateau neighbours fail the same way.

**Part B — does the quality screen help inside Open Alpha · Base Age? NO EDGE. Review CLOSED.**
r/161's engine byte-identical (its no-mask control reproduces **21.26% / −34.80% / Calmar 0.618**
exactly). Applying each screen to ENTRIES only: **not one screen wins on a single seed out of
thirty** on return, in either window, under either missing-data policy. b7 costs −9.71pp; the
screen as Arun wrote it costs −17.77pp at 18.8% invested. It is starvation, not selection —
qualifying signals fall 3,619 → 468 → 76. **The quality-screen line is closed permanently**, as
that review's own text instructed on a fail.

**Part C — portfolio fit: DILUTIVE.** Against True North + Base Age 50-50 monthly (the honest pair,
24.42% / −13.71% / Calmar 1.769 on monthly marks), adding Quality Summit at 10/20/33% is beaten by
plain **cash at the same weight on 360 of 360 paths**. Monthly correlation to Base Age **0.717**.

**What this leaves for someone to pick up**

- Nothing is owed. One dated review was registered: **2027-09-12 — re-open the Quality Summit
  optimisation ONLY when the holdout has grown a year** (re-running it on the same window is
  holdout mining). The pass criterion is unchanged and written into the Ops Centre entry.
- **If Arun ever wants a lower-drawdown near-ATH momentum book**, the screen is the honest way to
  get it: it takes **12.7 to 16.9 points off the maximum drawdown on 12 of 12 offsets** for 1.3
  points of CAGR. That is insurance with a premium, not an edge, and it is not what was asked for.
- A method note worth keeping: without the pre-registered W1/W2 split and the 4pp rule, a 12-of-12
  offset sweep with a verified plateau would have been published as an improvement.

Study: **http://94.136.185.54:5000/app/backtest/quality-summit-optimisation-research162**
Verdicts + caveats: `research/162_quality_summit_optimisation/results/RESULTS.md`
Pre-registration + live log + crash recovery:
`research/162_quality_summit_optimisation/QUALITY_SUMMIT_OPTIMISATION_DAILY_SWEEP_STATUS.md`

---

'''

LABS_ENTRY = '''
## research/162 — Quality Summit optimisation and the Base Age quality overlay (added 2026-09-12)

Both questions are **answered and closed**; nothing here runs on a schedule.

| Item | State |
|---|---|
| r/160's "quality as an OVERLAY inside Open Alpha's entries" review (was due **2026-10-10**) | **DONE 12-Sep-2026, FAILED.** Not one screen wins on a single seed of thirty on return, either window, either missing-data policy. The quality-screen line is **closed permanently**, as that review's own text instructed on a fail. Ops Centre entry rewritten as a DONE record with the outcome. |
| **NEW dated review — 2027-09-12** | Re-open the Quality Summit optimisation **only when the holdout has grown a year**. r/162 Part A's candidate (b7 screen, near-ATH band k=0.85, ten names, inverse-vol) won the fit window on 12 of 12 offsets and lost the holdout on 1 of 12, 9.22pp below fit against a pre-registered 4pp limit. The holdout is only four years and is dominated by the 2023-25 smallcap boom, so the failure could in principle be regime rather than overfit — but re-running it on the same window is holdout mining. Pass criterion unchanged; if it fails again, close the line permanently. Cost ~2 h; every derived cache, mask and grid is committed. |

Manual re-run (nothing is scheduled):

```bash
cd /home/arun/quantifyd
venv/bin/python3 research/162_quality_summit_optimisation/scripts/build_aux.py        # ~90 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/build_masks162.py   # ~30 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/patch_engine.py     # regenerate qg_engine2.py
venv/bin/python3 research/162_quality_summit_optimisation/scripts/build_panel161.py   # ~45 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/partb_overlay.py    # ~30 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/finalize_a.py       # ~95 s
venv/bin/python3 research/162_quality_summit_optimisation/scripts/partc_blend.py
```
'''


def main():
    idx = ROOT / 'research/INDEX.md'
    s = io.open(idx, encoding='utf-8').read()
    if '| 162 |' not in s:
        if not s.endswith('\n'):
            s += '\n'
        io.open(idx, 'w', encoding='utf-8').write(s + INDEX_ROW)
        print('INDEX.md: row 162 appended')
    else:
        print('INDEX.md: row 162 already present')

    todo = ROOT / 'TODO.md'
    t = io.open(todo, encoding='utf-8').read()
    if 'research/162: Quality Summit could NOT be improved' not in t:
        anchor = '\n## '
        i = t.find(anchor)
        if i < 0:
            print('TODO.md: FAILED to find the first section header')
            return 2
        t = t[:i + 1] + TODO_ENTRY + t[i + 1:]
        io.open(todo, 'w', encoding='utf-8').write(t)
        print('TODO.md: r/162 entry inserted at the top')
    else:
        print('TODO.md: r/162 entry already present')

    labs = ROOT / 'docs/LABS_AND_JOBS_REFERENCE.md'
    l = io.open(labs, encoding='utf-8').read()
    if 'research/162 — Quality Summit optimisation' not in l:
        if not l.endswith('\n'):
            l += '\n'
        io.open(labs, 'w', encoding='utf-8').write(l + LABS_ENTRY)
        print('LABS_AND_JOBS_REFERENCE.md: r/162 section appended')
    else:
        print('LABS_AND_JOBS_REFERENCE.md: r/162 section already present')
    return 0


if __name__ == '__main__':
    sys.exit(main())
