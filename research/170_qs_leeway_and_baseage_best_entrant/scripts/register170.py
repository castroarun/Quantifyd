# -*- coding: utf-8 -*-
"""research/170 -- close the loop: INDEX row, Ops & Review Centre entry, TODO entry.

Idempotent: each edit checks for its own marker first.
"""
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
INDEX = ROOT / 'research' / 'INDEX.md'
OPS = ROOT / 'research' / '111_sensex_manual_mgmt' / 'scripts' / 'ops_center.py'
TODO = ROOT / 'TODO.md'

# ------------------------------------------------------------------------ INDEX row ----
INDEX_ROW = (
    '| 170 | [**QS rank leeway · and OA·Base Age best-qualifying entrant**]'
    '(170_qs_leeway_and_baseage_best_entrant/results/RESULTS.md) — two unrelated questions '
    'Arun asked on 13-Sep-2026. **(A)** *"a stock falling to 16th rank being taken out now — '
    'how about giving it a leeway, say within top 25"* on Quality Summit (r/160 Family-B b7). '
    '**(B)** *"our case: if not any qualifying signal — the best candidate / highest-ranked '
    'one"* — confirm r/166\'s POST-HOC "give the freed slot to the highest-RS entrant" pick '
    'on seeds it had never seen | A: 2018-08-01→2026-09-10, 12 rebalance-day offsets, 54 '
    'selection cells; B: 2005-01-03→2026-09-11, 30 seeds on TWO independent seed sets '
    '(fresh 1001-1030 and r/166\'s 1-30), 5 selection cells. Both after tax, 25 bps/side, '
    'idle cash 5.2% post-tax | **A — the premise was wrong and the axis is dead.** The book '
    'already keeps a holding to **rank 23** (ceil(1.5×15)), not 15 — rank 16 is not a sale. '
    'Instrumenting the sale reason shows only **14.5% of sales are RANK sales**; 85.5% are '
    'the name leaving the near-ATH band / liquidity / screen, which no leeway can touch. '
    'Paired on 12 offsets every width loses: rank 15/20/26/30/38/45 give −2.02/−0.15/−0.09/'
    '−0.39/−0.57/−0.84 pp of CAGR on 3/4/5/4/3/2 of 12. Arun\'s own "top 25" (rank 26) is a '
    'dead wash. Widening DOES cut churn (74→54 trades/yr, hold 66→93 days) and CANNOT cut '
    'tax: trades held >365d rise only 0.3%→1.6%, so the 12.5% rate never arrives; buying it '
    'via quarterly cadence costs more than it saves (18.75%). The *other* leeway — keep a '
    'name after it leaves the band (retain=loose) — is the worst cell in the study: −1.42pp '
    'CAGR, −0.146 Calmar, +5.5pp DD. k=0.90 is the Calmar peak at every leeway; k=0.85 '
    'clears the fit window (+0.156 Calmar 9/12) and reverses in the holdout (−0.063 4/12) — '
    '**second independent replication of r/162\'s reversal**. Random-ranking null 11.73% vs '
    '21.39% (RS worth +9.7pp) and the leeway helps the RANDOM book too, so its benefit is '
    'churn, not selection. **B — the post-hoc pick REPLICATED.** On 30 fresh seeds the rule '
    '"sell the holding >10% under water, buy the refused entrant with the highest rs252" '
    'returns 22.58% / −31.78% / Calmar **0.710** vs the incumbent 20.95% / −34.05% / 0.611 '
    '— paired **+0.105 Calmar on 30/30** and +1.64pp CAGR on 30/30, beats a rate-matched '
    'random swap **+0.115 on 30/30**, wins BOTH windows 30/30, and holds at 40/60 bps '
    '(0.710/0.684/0.647 vs 0.611/0.589/0.561). Pooled over all 60 paths: **+0.096 Calmar on '
    '60/60** against a pre-registered bar of +0.100 — three evaluations now at +0.094, '
    '+0.096, +0.105. **Who leaves sets the return; who enters sets the drawdown**: all three '
    'entrant priorities earn +1.65 to +1.77pp, and the whole 5.5pp DD spread is the entrant '
    '(rs252 −31.78, tv20 −33.52, base age −37.23 — base age LOSES on Calmar). Tax +25% '
    '(₹84.1L→₹105.4L on ₹10L). Published `/app/backtest/'
    'qs-leeway-and-baseage-best-entrant-research170` | **A: CONCLUDED — NO ADOPTION. '
    'B: SIGNAL — confirmed, sits ON the bar, NOT adopted; live Base Age (r/165) goes live '
    'unchanged; proposal OA-ROT-1 + dated review 2027-03-13** |\n')

# ----------------------------------------------------------------------- Ops review ----
OPS_MARK = 'OA-ROT-1'
OPS_ENTRY = '''    ("OA \\u00b7 Base Age - re-test rotation rule OA-ROT-1 on the LIVE entry queue after six "
     "months of real-money operation",
     "2027-03-13", "PENDING",
     "research/170 (13-Sep-2026) CONFIRMED research/166's post-hoc pick on 30 seeds it had "
     "never been run on: when the Base Age book is full and a qualifying signal arrives, "
     "selling the holding more than 10% under water and giving its slot to the refused "
     "entrant with the HIGHEST 252-day relative strength returns 22.58% after tax at a "
     "-31.78% drawdown (Calmar 0.710) against the incumbent's 20.95% / -34.05% / 0.611. "
     "Paired: +0.105 Calmar on 30/30 fresh seeds, +0.094 on research/166's own seeds, "
     "+0.096 on all 60 paths pooled - against a pre-registered bar of +0.100. It beats a "
     "rate-matched random swap on 30/30, wins both pre-registered windows on 30/30, and "
     "holds its advantage at 40 and 60 bps. NOTHING WAS ADOPTED: the bar was written down "
     "before the run and the pooled figure misses it by 0.004, the book has never traded "
     "live, and the rule needs an evening scorer plus a two-leg next-open order that does "
     "not exist. The exact proposed rule text is OA-ROT-1 in "
     "research/170_qs_leeway_and_baseage_best_entrant/results/RESULTS.md. "
     "WHAT TO VERIFY ON THE DUE DATE, in order: (1) does the LIVE entry queue show the "
     "shape the rule needs - qualifying signals refused while a holding sits more than 10% "
     "under water, at roughly 4 occurrences a year? If it does NOT, the rule is "
     "inapplicable regardless of the backtest and should be dropped, and this review "
     "closes. (2) If it does, re-run research/170's five Part-B cells on the LIVE event log "
     "plus the extended history and apply the SAME +0.10 paired Calmar bar, UNCHANGED - do "
     "not move the bar because three evaluations have landed just under it. (3) Report the "
     "tax and tradeability cost alongside: the rule raises tax 25%, drops the win rate from "
     "48.9% to 47.3%, lengthens the worst losing streak from 14 to 15, and raises the share "
     "of profit coming from the ten best realisations from 37.5% to 43.7%. "
     "PASS = either a clean drop with the reason recorded, or a re-run that clears +0.10 on "
     "live-informed evidence and is then raised as its own strategy change with its own "
     "STATUS doc and an after-15:40 deploy. Study: /app/backtest/"
     "qs-leeway-and-baseage-best-entrant-research170"),
'''

# ------------------------------------------------------------------------ TODO entry ---
TODO_MARK = 'research/170'
TODO_ENTRY = '''## ✅ 2026-09-13 — research/170: QS rank leeway is a dead axis — and Base Age's "best entrant" pick **replicated on fresh seeds** and still sits ON the bar

Two unrelated questions Arun asked on the morning of 13-Sep-2026, on two different books.
Published at `/app/backtest/qs-leeway-and-baseage-best-entrant-research170`. **Nothing is deployed.**

**Part A — Quality Summit rank leeway: CONCLUDED, NO ADOPTION.** Arun asked for "a leeway,
say within top 25" for a holding that slips in the rank. **The book already keeps a holding to
rank 23** (`ceil(buffer 1.5 × N 15)`), so a name at rank 16 is not sold today. More to the
point, instrumenting the sale reason for the first time shows **only 14.5% of sales are rank
sales** — 85.5% are the name leaving the near-all-time-high band, the liquidity floor or the
screen, and no leeway of any width can touch those. Paired on the same 12 rebalance-day offsets,
**every** width loses: rank 15/20/26/30/38/45 give −2.02 / −0.15 / −0.09 / −0.39 / −0.57 /
−0.84 pp of CAGR, winning 3/4/5/4/3/2 of 12. Arun's own proposal (rank 26) is a dead wash.
Widening **does** cut churn (74→54 trades/yr, hold 66→93 days) and **cannot** cut tax: trades
held beyond 365 days rise only 0.3%→1.6%, so the 12.5% long-term rate never arrives. The
*other* leeway — keep a name after it leaves the band — is the **worst cell in the study**
(−1.42pp CAGR, −0.146 Calmar, 5.5 extra points of drawdown). k = 0.90 stands: it is the Calmar
peak at every leeway, and k = 0.85 clears the fit window and reverses in the holdout — the
**second independent replication** of research/162's reversal.

**Part B — Base Age best-qualifying entrant: SIGNAL, confirmed, NOT adopted.** research/166
found, *after* seeing results, that giving a freed slot to the highest-RS refused entrant lifts
the rotation rule to Calmar 0.715. research/170 named the rule and the three candidates
**before** running anything and re-ran them on **seeds 1001–1030, which no cell in r/164 or
r/166 had ever touched**. It came back: **22.58% after tax, −31.78% drawdown, Calmar 0.710**
against the incumbent's 20.95% / −34.05% / 0.611 — paired **+0.105 Calmar on 30/30**, beating
a rate-matched random swap **+0.115 on 30/30**, winning **both** windows on 30/30, and holding at
40 and 60 bps. Pooled over all 60 paths: **+0.096 against a pre-registered bar of +0.100**. Three
independent evaluations have now landed at +0.094, +0.096 and +0.105. **The effect is real and
small, and the threshold sits on top of it.**

**The new mechanical fact worth carrying forward: who leaves the book sets the return; who enters
it sets the drawdown.** All three entrant priorities earn the same +1.65 to +1.77pp of CAGR; the
entire 5.5-point drawdown spread is in which refused breakout you buy (rs252 −31.78%, tv20
−33.52%, oldest base −37.23% — and the oldest-base variant actually **loses** to the incumbent
on Calmar).

**Standing decision, unchanged:** the live Base Age book converting under research/165 goes live
**exactly as research/161 adopted it** — 16 slots at 6.25%, SuperTrend(14,4) close trail, no
stop, no rotation, no trimming. The proposed rule is written out as **OA-ROT-1** in
`research/170_.../results/RESULTS.md` and is a **proposal, not a change**; no executor file was
touched. **Dated review 2027-03-13** registered in the Ops & Review Centre, with the pass
criterion on the **live entry queue** (it must actually refuse signals while a holding sits more
than 10% under water, ~4 times a year) and the **same +0.10 bar, unchanged**.

'''


def edit(path, marker, fn, label):
    s = path.read_text(encoding='utf-8')
    if marker in s:
        print('skip %s (already present)' % label)
        return
    path.write_text(fn(s), encoding='utf-8')
    print('updated %s' % label)


def main():
    edit(INDEX, '| 170 |', lambda s: s.rstrip('\n') + '\n' + INDEX_ROW, 'research/INDEX.md')
    edit(OPS, OPS_MARK,
         lambda s: s.replace('REVIEWS = [\n', 'REVIEWS = [\n' + OPS_ENTRY, 1), 'ops_center.py')

    def todo_ins(s):
        anchor = '\n## '
        i = s.index(anchor)
        return s[:i + 1] + TODO_ENTRY + s[i + 1:]
    edit(TODO, TODO_MARK, todo_ins, 'TODO.md')


if __name__ == '__main__':
    main()
