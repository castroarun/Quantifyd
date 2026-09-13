# -*- coding: utf-8 -*-
"""research/165 - register OA-ROT-1: strategies index, ops reviews, labs doc, TODO.

Idempotent and marker-guarded, like research/170's `register170.py`: every edit carries its OWN
marker and is skipped if that marker is already in the file, and every anchor must match EXACTLY
ONCE or the run aborts without writing anything. Safe to re-run, and safe to run after a
concurrent session has edited the same files - which is exactly what happened on 13-Sep-2026,
when register170.py landed the 2027-03-13 review between this script being written and run.

Note the edits AMEND research/170's own entries rather than adding rivals to them: that study
registered OA-ROT-1 as a PROPOSAL with a dated review, and Arun's adoption changes what those
entries say, not how many of them there are.

    python3 research/165_oa_baseage_live_conversion/scripts/register_rot1.py [--dry]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STRAT = ROOT / 'frontend' / 'src' / 'data' / 'strategies.ts'
OPS = ROOT / 'research' / '111_sensex_manual_mgmt' / 'scripts' / 'ops_center.py'
LABS = ROOT / 'docs' / 'LABS_AND_JOBS_REFERENCE.md'
TODO = ROOT / 'TODO.md'

# ───────────────────────────── strategies.ts ─────────────────────────────
STRAT_MARK = 'Swap rule OA-ROT-1 (staged)'

STRAT_ANCHOR = ("      ['Switch', 'OA_RULESET in services/oa_real.py: legacy (running) | "
                "baseage (staged).")

STRAT_ROW = (
    "      ['Swap rule OA-ROT-1 (staged)', 'On an evening when a qualifying signal is refused "
    "(no free slot, or cash below one slot): sell the holding with the largest loss versus its "
    "buy price IF that loss is worse than \\u221210% on the trigger close, and buy the refused "
    "signal with the highest 12-month relative strength (rs252, on the study\\u2019s NIFTYBEES "
    "master calendar). Both legs AMO for the next open, SELL first, tagged OA-ROT1-SELL / "
    "OA-ROT1-BUY. At most ONE swap per evening; a holding already being sold on the trail is an "
    "ordinary exit, not a swap; if no refused entrant can be funded from the proceeds, both legs "
    "are cancelled. research/170 Part B: 22.58% after tax / \\u221231.78% DD / Calmar 0.710 "
    "against the un-rotated 20.95 / \\u221234.05 / 0.611 \\u2014 +0.105 paired Calmar on 30/30 "
    "fresh seeds, +1.65pp CAGR on 60/60 pooled paths, beating a rate-matched random swap 60/60. "
    "ADOPTED by Arun 13-Sep-2026 over research/170\\u2019s own non-adoption verdict (it missed "
    "the pre-registered +0.10 Calmar bar by 0.004). Replication gate: 8,488 of 8,488 rotation "
    "decisions identical to research/170\\u2019s engine \\u2014 fire, name sold and name bought. "
    "Own OFF switch: OA_ROT1 in services/oa_real.py disables the swap and keeps Base Age.'],\n")

STRAT_LOG_ANCHOR = ("    changeLog: [{ date: '12 Sep 2026', text: 'Open Alpha \\u00b7 Base Age "
                    "conversion STAGED.")

STRAT_LOG = (
    "    changeLog: [{ date: '13 Sep 2026', text: 'SWAP RULE OA-ROT-1 BUILT INTO THE STAGED "
    "CONVERSION, switch still OFF. Arun adopted research/170 Part B: when a qualifying Base Age "
    "signal is refused for a slot or for cash, sell the holding more than 10% under water and "
    "buy the refused signal with the highest 12-month relative strength, both at the next open, "
    "at most one a night. 22.58% after tax / -31.78% drawdown / Calmar 0.710 against the "
    "un-rotated 20.95 / -34.05 / 0.611; +0.105 paired Calmar on 30 of 30 fresh seeds and +1.65pp "
    "of CAGR on 60 of 60 pooled paths. research/170 itself did NOT adopt it - it missed its own "
    "pre-registered +0.10 Calmar bar by 0.004 - so the rule carries its own OFF switch, OA_ROT1. "
    "Replication gate PASS: 8,488 of 8,488 rotation decisions across six engine paths agree with "
    "research/170s engine on fire, on the name sold and on the name bought. The gate also found "
    "and fixed a real defect before it could reach live: rs252 must be read on the studys "
    "NIFTYBEES master calendar, not on 252 of the symbols own bars (73.7% vs 99.7% reproduction "
    "of the frozen figures). HONEST CAVEAT: walked over the last 400 sessions on the live code "
    "the rule LOST on all three arms and fired about 12 swaps a year against the studys 4.5 - "
    "one 1.6-year path, but it is the only walk of the live code that exists. Dry run 11-Sep: no "
    "swap possible Monday (5 free slots), and none would fire even on a full book - the deepest "
    "loss is SBCL at -2.46%. research/165, OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md.' },\n"
    "      { date: '12 Sep 2026', text: 'Open Alpha \\u00b7 Base Age conversion STAGED.")

STRAT_SWITCH_OLD = ("baseage (staged). Entries paused 11-Sep-2026; Base Age entries staged "
                    "12-Sep-2026, switch OFF.")
STRAT_SWITCH_NEW = ("baseage (staged), and OA_ROT1: True | False, which disables the OA-ROT-1 "
                    "swap alone while keeping Base Age. Entries paused 11-Sep-2026; Base Age "
                    "entries staged 12-Sep-2026 and the swap rule 13-Sep-2026, switch OFF.")

# ───────────────────────── ops_center.py, research/170's own review ─────────────────────────
OPS_MARK = 'ADOPTED into the staged conversion 13-Sep-2026'

OPS_TITLE_OLD = ('    ("OA \\u00b7 Base Age - re-test rotation rule OA-ROT-1 on the LIVE entry '
                 'queue after six "\n     "months of real-money operation",')
OPS_TITLE_NEW = ('    ("OA \\u00b7 Base Age - rotation rule OA-ROT-1 ADOPTED: re-test it on the '
                 'LIVE entry queue "\n     "after six months of real-money operation",')

OPS_HEAD_OLD = ('     "research/170 (13-Sep-2026) CONFIRMED research/166\'s post-hoc pick on 30 '
                'seeds it had "')
OPS_HEAD_NEW = (
    '     "ADOPTED into the staged conversion 13-Sep-2026; review the first live swaps. "\n'
    '     "Arun read research/170 Part B and adopted the rule the same morning, overriding that '
    'study\'s own non-adoption verdict; research/165 built it into the staged live conversion '
    'behind its own OFF switch OA_ROT1 in services/oa_real.py. OA_RULESET is still \'legacy\', '
    'so nothing is running yet. Replication gate PASS: 8,488 of 8,488 rotation decisions across '
    'six engine paths identical to research/170\'s own engine on fire/no-fire, on the name sold '
    'and on the name bought; the gate also caught a real defect first - rs252 must be read on '
    'the study\'s NIFTYBEES master calendar, not on 252 of a symbol\'s own bars. Rule text, gate '
    'and runbook: research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md. "\n'
    '     "THE ORIGINAL FINDING, unchanged: "\n'
    + OPS_HEAD_OLD)

OPS_NOTADOPTED_OLD = ('     "holds its advantage at 40 and 60 bps. NOTHING WAS ADOPTED: the bar '
                      'was written down "\n     "before the run and the pooled figure misses it '
                      'by 0.004, the book has never traded "\n     "live, and the rule needs an '
                      'evening scorer plus a two-leg next-open order that does "\n     "not '
                      'exist. The exact proposed rule text is OA-ROT-1 in "')
OPS_NOTADOPTED_NEW = ('     "holds its advantage at 40 and 60 bps. research/170 DID NOT ADOPT IT: '
                      'the bar was written "\n     "down before the run and the pooled figure '
                      'misses it by 0.004, the book had never "\n     "traded live, and the rule '
                      'needed an evening scorer plus a two-leg next-open order "\n     "that did '
                      'not exist. Arun overrode the first reason on 13-Sep-2026 and research/165 '
                      'built "\n     "the third; the second is still true and is what this '
                      'review exists for. Rule text: OA-ROT-1 in "')

OPS_VERIFY_OLD = ('     "WHAT TO VERIFY ON THE DUE DATE, in order: (1) does the LIVE entry queue '
                  'show the "\n     "shape the rule needs - qualifying signals refused while a '
                  'holding sits more than 10% "\n     "under water, at roughly 4 occurrences a '
                  'year? If it does NOT, the rule is "\n     "inapplicable regardless of the '
                  'backtest and should be dropped, and this review "\n     "closes. (2) If it '
                  'does, re-run research/170\'s five Part-B cells on the LIVE event log "')
OPS_VERIFY_NEW = ('     "WHAT TO VERIFY ON THE DUE DATE, in order: (0) THE SWAP RATE, which is '
                  'now the first "\n     "question rather than an assumption: research/170 '
                  'measures ~4 swaps a year and this "\n     "criterion was written against '
                  'that, but research/165\'s 400-session walk of the LIVE "\n     "code produced '
                  '~12 a year and its 60-day walk ~46. Record the real rate, and look for "\n'
                  '     "churn chains - a name swapped in and swapped out again within ten '
                  'sessions. If the "\n     "live rate is far above 4, RE-READ this criterion '
                  'rather than quietly re-scaling it. "\n     "(0b) P&L ATTRIBUTION of both legs '
                  'of every swap: what the sold name did after it "\n     "was sold and what the '
                  'entrant did. Over the only window research/165 could walk, the "\n     "rule '
                  'LOST on all three arms (-Rs 23,082 from the live book over 400 sessions). "\n'
                  '     "(1) does the LIVE entry queue show the "\n     "shape the rule needs - '
                  'qualifying signals refused while a holding sits more than 10% "\n     "under '
                  'water? If it does NOT, the rule is "\n     "inapplicable regardless of the '
                  'backtest and should be switched off with OA_ROT1=False "\n     "and the '
                  'reason recorded, and this review "\n     "closes. (2) If it does, re-run '
                  'research/170\'s five Part-B cells on the LIVE event log "')

# ───────────────────────── ops_center.py, research/165's two reviews ─────────────────────────
OPS2_MARK = 'ADDED 13-Sep-2026, the OA-ROT-1 swap rule'

OPS_0926_OLD = ('"Also: is the 09:20 equity_executor OA top-up off the OA book. Runbook: "\n'
                '     "research/165_oa_baseage_live_conversion/'
                'OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md section 9."),')
OPS_0926_NEW = (
    '"Also: is the 09:20 equity_executor OA top-up off the OA book. "\n'
    '     "ADDED 13-Sep-2026, the OA-ROT-1 swap rule: (a) HOW MANY SWAPS in two weeks - '
    'research/170 expects ~4 a year, research/165\'s walk of the live code produced ~12; '
    '(b) HOW MANY REFUSALS WERE CONVERTED - refused-for-cash / no-free-slot lines followed by '
    'a swap, against those that were not; (c) did any swapped-IN name get swapped OUT again '
    'within ten sessions (a churn chain); (d) P&L attribution of both legs of every swap; '
    '(e) is the entry-sizing NAV-basis question still open - plan() sizes the slot off '
    'cost-plus-cash NAV while the study and rot1_pick() use the marked NAV, which is the '
    'difference between PAYTM x20 and x21. "\n'
    '     "Runbooks: "\n'
    '     "research/165_oa_baseage_live_conversion/'
    'OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md section 9 and '
    'OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md section 9."),')

OPS_0915_OLD = ('"and did the 09:20 equity_executor stay off the OA book. If the switch was '
                'not flipped, mark N/A and "\n     "carry to the next session."),')
OPS_0915_NEW = ('"and did the 09:20 equity_executor stay off the OA book. If the switch was '
                'not flipped, mark N/A and "\n     "carry to the next session. "\n'
                '     "ADDED 13-Sep-2026: the log must also show an OA-ROT-1 block, even if it '
                'only says there was nothing to swap - if that block is missing, the swap code '
                'did not run and the flip is only half applied. If a swap fired, check BOTH legs '
                'are in kite.orders() tagged OA-ROT1-SELL and OA-ROT1-BUY with the SELL first, '
                'that reconcile labelled the sale rot1_swap_out rather than rule_exit, that the '
                'name sold was the deepest loss in the book AND worse than -10%, and that the '
                'name bought was the highest-rs252 refused signal. Anything off: set '
                'OA_ROT1 = False, which stops the swap and leaves Base Age running."),')

# ─────────────────────────────── labs doc ────────────────────────────────
LABS_MARK = 'Review 2027-03-13 - OA-ROT-1'
LABS_ANCHOR = ('## Reviews 2026-09-15 and 2026-09-26 - Open Alpha - Base Age LIVE conversion '
               '(added 12-Sep-2026)\n')
LABS_ENTRY = '''## Review 2027-03-13 - OA-ROT-1, the Base Age best-entrant swap (added 13-Sep-2026)
Arun ADOPTED research/170 Part B on 13-Sep-2026: when a qualifying Base Age signal is refused
for a slot or for cash, sell the holding more than 10% under its buy price and buy the refused
signal with the highest 12-month relative strength, both at the next open, at most one a night.
Built into the staged conversion by research/165 behind its own OFF switch `OA_ROT1` in
`services/oa_real.py`; `OA_RULESET` is still `legacy`, so nothing runs yet. Replication gate:
8,488 of 8,488 rotation decisions identical to research/170's engine. The 2027-03-13 review
(research/170's own, amended) checks the live SWAP RATE first - ~4/yr expected, ~12/yr in
research/165's walk of the live code - then the P&L attribution of both legs, then re-runs
research/170's five Part-B cells against the unchanged +0.10 Calmar bar. The 15-Sep and 26-Sep
reviews below carry the swap checks too. Registered in `ops_center.py` REVIEWS. Runbook:
`research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md` section 9.

'''

# ──────────────────────────────── TODO.md ────────────────────────────────
TODO_MARK = 'OA-ROT-1 ADOPTED'

TODO_STANDING_OLD = '''**Standing decision, unchanged:** the live Base Age book converting under research/165 goes live
**exactly as research/161 adopted it** — 16 slots at 6.25%, SuperTrend(14,4) close trail, no
stop, no rotation, no trimming. The proposed rule is written out as **OA-ROT-1** in
`research/170_.../results/RESULTS.md` and is a **proposal, not a change**; no executor file was
touched. **Dated review 2027-03-13** registered in the Ops & Review Centre, with the pass
criterion on the **live entry queue** (it must actually refuse signals while a holding sits more
than 10% under water, ~4 times a year) and the **same +0.10 bar, unchanged**.'''

TODO_STANDING_NEW = '''**SUPERSEDED THE SAME DAY — OA-ROT-1 ADOPTED by Arun, 13-Sep-2026.** This study's standing
decision was that the live Base Age book converts *without* rotation. Arun read Part B and
overrode it: *"Swap, entrant by relative strength 22.58% / −31.8% — I love this. Let's make
changes to the live system later today, not now."* research/165 has built the rule into the
staged conversion behind its own OFF switch `OA_ROT1`; `OA_RULESET` is still `'legacy'` and
nothing is running. The **dated review 2027-03-13** stays, amended: it now asks about the live
SWAP RATE first (~4/yr expected, ~12/yr in research/165's walk of the live code) and the P&L
attribution of both legs, then re-runs the five Part-B cells against the **same +0.10 bar,
unchanged**. See the conversion entry below and
`research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md`.'''

TODO_CONV_ANCHOR = ('Register row updated (Open Alpha - Base Age (converting)); reviews '
                    '2026-09-15 and 2026-09-26 in the Ops Centre.')

TODO_CONV_ADD = TODO_CONV_ANCHOR + '''

**EXTENDED 13-Sep-2026 — the conversion now also carries OA-ROT-1, still switched off.** Arun
adopted research/170 Part B (*"Swap, entrant by relative strength 22.58% / −31.8% — I love this.
Let's make changes to the live system later today, not now."*). On an evening when a qualifying
signal is refused for a slot or for cash, the 18:50 job sells the holding whose loss against its
buy price is worse than **−10%** on the trigger close and buys the refused signal with the
**highest 12-month relative strength**, both as AMOs for the next open (SELL first), tagged
`OA-ROT1-SELL` / `OA-ROT1-BUY`, **one swap a night**. Study: 22.58% after tax / −31.78% DD /
Calmar 0.710 against the un-rotated 20.95 / −34.05 / 0.611; **+0.105 paired Calmar on 30/30 fresh
seeds, +1.65pp CAGR on 60/60 pooled paths**. research/170 itself did **not** adopt it (it missed
its own pre-registered +0.10 bar by 0.004), so the rule has its **own OFF switch `OA_ROT1`** —
set it to `False` to stop the swap and keep Base Age, no crontab change, no restart.

**Replication gate PASS, 100.00%:** 8,488 of 8,488 rotation decisions across six engine paths
agree with research/170's own engine on fire/no-fire, the name sold and the name bought. The gate
also caught a real defect before it could reach live: `rs252` must be read on the study's
**NIFTYBEES master calendar**, not on 252 of a symbol's own bars (73.7% vs 99.7% reproduction of
the frozen figures; 16 of 810 contested days would have bought a different name).

**Two cautions to carry into the flip.** (1) Walked over the **last 400 sessions on the live
code**, the rule **LOST** on all three arms (−₹23,082 from the live book) and fired **~12 swaps a
year against the study's 4.5** — one 1.6-year path with no seeds, but it is the only walk of the
live code that exists. (2) `plan()` sizes the entry slot off **cost-plus-cash NAV** while the
study and `rot1_pick()` use the **marked** NAV — the difference between Monday's PAYTM at ×20 and
the ×21 printed above. **Not changed**; a question for Arun before the flip.

**Dry run on the 11-Sep close:** no swap can fire Monday (5 free slots and ₹1.88L of cash take
PAYTM outright, so nothing is refused), and **none would fire even on a full book** — the deepest
loss is **SBCL at −2.46%**, which would have to fall a further 7.7% to qualify.

**Runbook delta + the swap's own rollback:**
`research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md` §9.'''


def edit(path, marker, fn, label, dry):
    s = path.read_text(encoding='utf-8')
    if marker in s:
        print('SKIP  %-18s (marker %r already present)' % (label, marker))
        return
    out = fn(s)
    if out == s:
        raise SystemExit('ABORT: %s produced no change' % label)
    if dry:
        print('DRY   %-18s would grow by %d bytes' % (label, len(out) - len(s)))
        return
    path.write_text(out, encoding='utf-8')
    print('WROTE %-18s (+%d bytes)' % (label, len(out) - len(s)))


def once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise SystemExit('ABORT: %s anchor matched %d times, expected 1:\n%r'
                         % (what, n, old[:110]))
    return s.replace(old, new, 1)


def main():
    dry = '--dry' in sys.argv

    def strat(s):
        s = once(s, STRAT_ANCHOR, STRAT_ROW + STRAT_ANCHOR, 'strategies rules row')
        s = once(s, STRAT_SWITCH_OLD, STRAT_SWITCH_NEW, 'strategies switch row')
        return once(s, STRAT_LOG_ANCHOR, STRAT_LOG, 'strategies changeLog')

    def ops170(s):
        s = once(s, OPS_TITLE_OLD, OPS_TITLE_NEW, 'ops 2027-03-13 title')
        s = once(s, OPS_HEAD_OLD, OPS_HEAD_NEW, 'ops 2027-03-13 head')
        s = once(s, OPS_NOTADOPTED_OLD, OPS_NOTADOPTED_NEW, 'ops 2027-03-13 not-adopted')
        return once(s, OPS_VERIFY_OLD, OPS_VERIFY_NEW, 'ops 2027-03-13 verify list')

    def ops165(s):
        s = once(s, OPS_0926_OLD, OPS_0926_NEW, 'ops 2026-09-26 review')
        return once(s, OPS_0915_OLD, OPS_0915_NEW, 'ops 2026-09-15 review')

    def labs(s):
        return once(s, LABS_ANCHOR, LABS_ENTRY + LABS_ANCHOR, 'labs doc')

    def todo(s):
        s = once(s, TODO_STANDING_OLD, TODO_STANDING_NEW, 'TODO r/170 standing decision')
        return once(s, TODO_CONV_ANCHOR, TODO_CONV_ADD, 'TODO conversion entry')

    edit(STRAT, STRAT_MARK, strat, 'strategies.ts', dry)
    edit(OPS, OPS_MARK, ops170, 'ops r/170 review', dry)
    edit(OPS, OPS2_MARK, ops165, 'ops r/165 reviews', dry)
    edit(LABS, LABS_MARK, labs, 'LABS reference', dry)
    edit(TODO, TODO_MARK, todo, 'TODO.md', dry)


if __name__ == '__main__':
    main()
