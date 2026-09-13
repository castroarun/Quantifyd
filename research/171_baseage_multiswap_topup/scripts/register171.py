# -*- coding: utf-8 -*-
"""research/171 -- register the dated obligations in the Ops & Review Centre, add the
research/INDEX.md row, and append the TODO.md entry.

Idempotent: every write checks for its own marker first and each anchor must match exactly
once or the script aborts without touching the file.
"""
import sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
OPS = ROOT / 'research' / '111_sensex_manual_mgmt' / 'scripts' / 'ops_center.py'
INDEX = ROOT / 'research' / 'INDEX.md'
TODO = ROOT / 'TODO.md'

# ---------------------------------------------------------------- ops_center REVIEWS ----
# 1. amend the existing 26-Sep-2026 Base Age live-conversion review with what research/171
#    says the live queue should look like.
AMEND_ANCHOR = '     "Runbooks: "\n'
AMEND_NEW = (
    '     "ADDED 13-Sep-2026, research/171 (swap-more-than-one / top-up instead of a new "\n'
    '     "entrant): BOTH ideas are NO EDGE and nothing changed. What this review should "\n'
    '     "now ALSO check, because research/171 measured it: on how many evenings did TWO "\n'
    '     "holdings sit more than 10% under water at the same time while a signal was "\n'
    '     "refused? Over 21.7 years that happens 0.5 times a YEAR once OA-ROT-1 is running "\n'
    '     "(11 occasions in the whole history) because the rule keeps removing the loser. If "\n'
    '     "the live queue shows it materially more often than that, research/171 axis A "\n'
    '     "should be re-opened on the live event log. Do NOT read a second eligible loser as "\n'
    '     "a missed opportunity: on the full history, taking it was worth -0.21pp of CAGR and "\n'
    '     "-0.003 of Calmar, and a real loss at 40 bps. Evidence: "\n'
    '     "research/171_baseage_multiswap_topup/results/RESULTS.md. "\n'
    + AMEND_ANCHOR)

# 2. a NEW dated review: the unconditional -10% hard stop has now surfaced three times.
REVIEW_ANCHOR = 'REVIEWS = [\n'
REVIEW_NEW = REVIEW_ANCHOR + (
    '    ("OA . Base Age - the unconditional -10% hard stop has now surfaced THREE times as "\n'
    '     "the cheapest drawdown lever. Does it deserve its own study with its own bar?",\n'
    '     "2027-03-13", "PENDING",\n'
    '     "RAISED 13-Sep-2026 by research/171. A plain -10% stop, with NO rotation machinery "\n'
    '     "at all, returns 20.79% after tax at -30.46% drawdown and Calmar 0.683, against the "\n'
    '     "incumbent Base Age book 20.91% / -35.16% / 0.606 (30 seeds 7001-7030, 25 bps, 5.2% "\n'
    '     "idle cash, 2005-2026). It beats the incumbent on Calmar on 30 of 30 paired seeds "\n'
    '     "and is excluded ONLY by the pre-registered CAGR-eligibility clause, which it misses "\n'
    '     "by 0.12pp. research/166 found it first (caveat 4, Calmar 0.677), research/171 found "\n'
    '     "it twice more independently (0.683 as a hard stop, 0.692 routed through the "\n'
    '     "rotation path), and it supplies two thirds of the risk-adjusted gain of the best "\n'
    '     "cell research/171 could build. WHAT TO DECIDE on the due date, alongside the "\n'
    '     "OA-ROT-1 live review: whether Arun wants a lower-drawdown Base Age at all. If yes, "\n'
    '     "it needs its own STATUS doc and its own bar (a drawdown bar, not a Calmar bar - it "\n'
    '     "is insurance with a premium, not an edge) rather than being rediscovered a fourth "\n'
    '     "time as a by-product. Evidence: "\n'
    '     "research/171_baseage_multiswap_topup/results/RESULTS.md, the decomposition table."),\n')

INDEX_ROW = (
    '| 171 | [**OA . Base Age - swap MORE than one holding a night, and/or redeploy into '
    'EXISTING winners?**](171_baseage_multiswap_topup/results/RESULTS.md) - Arun 13-Sep-2026: '
    '*"what if we swap the last 2 ranks instead of 1? can we find out an optimized number? or '
    'maybe swap the lowest ranked one(s) and instead of new entrants, top up the highest '
    'running ones existing within the portfolio?"* 2005-01-03 to 2026-09-11, 30 fresh seeds '
    '(7001-7030), after tax, 25 bps, 5.2% idle cash; 51 selection cells of a 70 budget. '
    '**(A) NO EDGE - and the optimum k is 1.** Two holdings are more than 10% under water on '
    'the same refused-signal evening **0.5 times a year** once OA-ROT-1 is running (4.2 times '
    'a year on a book that never swaps - the rule destroys its own second opportunity); '
    '`k = 3, 4, 6` and all-eligible are **bit-identical on all 30 seeds**; `k = 2` costs '
    '-0.21pp CAGR / -0.003 Calmar and loses outright at 40 bps. '
    '**(B) NO EDGE - all 18 top-up constructions lose Calmar to doing nothing and none beats '
    'the staged rule on a single seed of 30** (best 0.587 vs incumbent 0.606 vs OA-ROT-1 '
    '0.702); a single position reaches 52-81% of NAV and the ten best trades come to supply '
    '55% of book profit. **(C) SIGNAL, NOT ADOPTED** - one invented cell clears the bar '
    '(+0.128 paired Calmar on 30/30, survives 40 bps) but the control that restricts it to '
    'refused-signal evenings returns **OA-ROT-1 to the digit with zero top-ups**: all of its '
    'edge is an unconditional -10% stop, which a plain stop delivers two thirds of at Calmar '
    '0.683 with no machinery, and 94% of its return edge sits in the 2016-2026 half. '
    'Live book **UNCHANGED**. Published `/app/backtest/baseage-multiswap-topup-research171`. '
    '**NO EDGE (A, B) / SIGNAL not adopted (C)** |\n')

TODO_NEW = """
## ✅ 2026-09-13 — OA · Base Age: "swap the last 2 instead of 1?" and "top up the winners we already hold?" — both NO EDGE, nothing changed

Arun: *"what if we swap the last 2 ranks instead of 1? can we find out an optimized number? or
maybe swap the lowest ranked one(s) and instead of new entrants, top up the highest running ones
existing within the portfolio?"* — research/171, published at
`/app/backtest/baseage-multiswap-topup-research171`.

**Swap two instead of one? No, and there is no optimum to find.** Once OA-ROT-1 is running, two
holdings are more than 10% under water on the same refused-signal evening **0.5 times a year** —
eleven occasions in 21.7 years, because the rule keeps removing the loser so a second never
accumulates. `k = 3, 4, 6` and "all eligible" are **bit-identical on all 30 seeds**. `k = 2` costs
−0.21pp of CAGR and −0.003 of Calmar and turns into a real loss at 40 bps.

**Top up existing winners instead of buying the new breakout? No — the clearest negative in the
study.** All 18 constructions lose Calmar to doing nothing and **not one beats the staged rule on
a single seed out of thirty**. A top-up spends the slot instead of refilling it, and a single
position reaches 52–81% of NAV while the ten best trades come to supply 55% of book profit.

**One cell the study invented clears the bar and is refused anyway.** Restrict it to the evenings
Arun described and it returns OA-ROT-1 to the digit with zero top-ups — all of its edge is an
unconditional −10% stop, 94% of it sits in the 2016-2026 half, and it takes one name to 40.8% of NAV.

**Nothing deployed. The live Base Age book keeps OA-ROT-1 exactly as research/165 staged it.**

Registered: the 26-Sep-2026 live-conversion review now also checks how often TWO holdings are
simultaneously eligible (expect ~0.5 a year); and a new 13-Mar-2027 review asks whether the
unconditional −10% hard stop — now surfaced three times at Calmar 0.683–0.692 with no machinery —
deserves its own study with its own drawdown bar.
"""

TODO_ANCHOR = '## ✅ 2026-09-13 — IPO Base: the live book now runs the spec it is funded on'


def main():
    ops = OPS.read_text(encoding='utf-8')
    if 'research/171_baseage_multiswap_topup' in ops:
        print('  ops_center: already registered')
    else:
        n1, n2 = ops.count(AMEND_ANCHOR), ops.count(REVIEW_ANCHOR)
        if n1 != 1 or n2 != 1:
            raise SystemExit('ops_center anchors matched %d and %d, expected 1 and 1' % (n1, n2))
        ops = ops.replace(AMEND_ANCHOR, AMEND_NEW).replace(REVIEW_ANCHOR, REVIEW_NEW)
        OPS.write_text(ops, encoding='utf-8')
        print('  ops_center: amended the 26-Sep review and added the 13-Mar-2027 hard-stop review')

    idx = INDEX.read_text(encoding='utf-8')
    if '171_baseage_multiswap_topup' in idx:
        print('  INDEX.md: already registered')
    else:
        if not idx.endswith('\n'):
            idx += '\n'
        INDEX.write_text(idx + INDEX_ROW, encoding='utf-8')
        print('  INDEX.md: row appended')

    todo = TODO.read_text(encoding='utf-8')
    if 'research/171' in todo:
        print('  TODO.md: already registered')
    else:
        n = todo.count(TODO_ANCHOR)
        if n != 1:
            raise SystemExit('TODO anchor matched %d times, expected 1' % n)
        TODO.write_text(todo.replace(TODO_ANCHOR, TODO_NEW.lstrip('\n') + '\n---\n\n' + TODO_ANCHOR),
                        encoding='utf-8')
        print('  TODO.md: entry inserted above the 13-Sep IPO entry')


if __name__ == '__main__':
    main()
