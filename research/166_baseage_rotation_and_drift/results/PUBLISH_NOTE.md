# research/166 — PUBLISH NOTE

Two other agents held `frontend/` while this study ran (a 5.2% cash re-run and the research/165
live-conversion build). If `git log --oneline` shows both of their jobs committed, append the
entry below to `frontend/src/data/backtests.ts`, copy the chart, and
`cd frontend && npm run build` (frontend-only → no backend restart, safe at any hour). Verify by
grepping the built bundle for the slug, not by curling the route.

## Chart to copy into `frontend/public/`

| Source (on the VPS) | Destination | Served at |
|---|---|---|
| `research/166_baseage_rotation_and_drift/results/r166_curves.png` | `frontend/public/baseage-rotation-research166.png` | `/app/baseage-rotation-research166.png` |

It is a two-panel figure: growth of ₹100 on a log scale for the incumbent, the pre-registered
rotation rule, the post-hoc best rotation rule, the best drift rule and NIFTYBEES, with a
drawdown panel beneath. `results.charts[0]` below already points at it; if the copy is skipped,
delete that one line — every table in the entry is self-contained.

## Ops review to register (I did not touch `ops_center.py` — two other agents held the tree)

Add to `REVIEWS` in `research/111_sensex_manual_mgmt/scripts/ops_center.py`:

- **title:** "OA Base Age — re-test the under-water rotation rule on six months of live data (research/166)"
- **due:** `2027-03-13`
- **status:** `PENDING`
- **note:** "research/166 found rotation is a SIGNAL that misses the pre-registered bar by
  0.006 Calmar: swapping out a holding more than 10% under water for a qualifying new signal
  earns +1.68pp after-tax CAGR at a shallower drawdown on 30/30 paired seeds, both windows,
  25/40/60 bps, and beats a rate-matched random swap by +2.29pp. Not adopted, because the bar
  was +0.10 Calmar / +2pp CAGR and the best cell needed a post-hoc entrant-priority pick. After
  six months of live Base Age operation, check the live entry queue: if signals are being
  refused while a holding sits more than 10% under water, re-run axis A on the live event log
  and re-apply the SAME bar unchanged. PASS = clears +0.10 Calmar or +2pp CAGR on the live-
  extended history. Also mirror the research/164 correction: the tv20 tie-break keeps its CAGR
  claim, loses its Calmar claim."

Also worth a one-line amendment on the **research/164** study page (its `caveats`), since this
study corrects it — text in the verdict below under "A correction to research/164".

## The entry

```ts
  {
    slug: 'baseage-rotation-and-drift-research166',
    title: 'OA · Base Age — when the book is full, should a better signal swap a weak holding out? And should bloated winners be trimmed? (research/166)',
    verdict:
      'SIGNAL — rotation is real and misses the pre-registered bar by six thousandths of a Calmar point; trimming is NO EDGE; nothing is deployed and the live book converting under research/165 goes live unchanged.\n\n## What Arun asked\n\n"If we have some new entrants coming up and let us say we have all slots already taken, is there a way we could rank our positions, swap the better one in for a bad one out?" research/164 widened it: at every slot count MORE qualifying signals are refused for want of CASH (1,955) than for want of a SLOT (977), because the book never trims a winner. So rotation and position drift were tested as one family, on the frozen research/161 event list, 3,619 signals, 03-Jan-2005 to 11-Sep-2026, 16 slots at 6.25%, 30 seeds, after tax, 25 bps a side, idle cash 5.2% post-tax.\n\n## Yes — but only one ranking works, and it is the dullest one\n\n- The only score that beats the incumbent is SWAP OUT A HOLDING MORE THAN 10% UNDER WATER. It earns +1.68pp of after-tax CAGR (20.98% to 22.68%) at a SHALLOWER drawdown (-34.05% to -33.52%), Calmar 0.613 to 0.673, on 30 of 30 paired seeds for CAGR and 27 of 30 for Calmar, in BOTH windows, at 4.2 swaps a year.\n- Every intuitive ranking LOSES. Rotating into momentum, or out of whatever sits closest to its trailing stop, lifts CAGR as far as 24.08% and blows the drawdown out to -43.8%. Calmar falls in all twelve of those cells, winning on 0 to 2 of 30 seeds. That is leverage, not selection.\n- Base age and time held do nothing: the only versions that do not hurt are the ones whose threshold is so high that they fire 0.1 to 0.3 times a year — which is the incumbent wearing a disguise.\n- Swap at most ONCE a day. Allowing two or three takes the same rule from 0.673 to 0.599.\n\n## Two controls decide what is actually working\n\n- SELL the under-water holding on the same trigger but DO NOT buy the entrant: Calmar 0.603 — WORSE than never swapping (0.613). So this is not a stop-loss in disguise; the cash must go straight back into a fresh qualifying breakout at the same open.\n- A RATE-MATCHED RANDOM swap (same 4 swaps a year, random holding): 20.39% / -35.74% / 0.569. The ranked rule beats it by +2.29pp of CAGR and +0.104 of Calmar. Which holding leaves is not incidental either.\n- Honest counterpoint: a plain UNCONDITIONAL -10% hard stop, with no rotation at all, returns 20.62% / -30.46% / 0.677 on 29 of 30 seeds. It was excluded only by the pre-registered clause that a cell must not lose CAGR. It is insurance, not edge — but if a lower-drawdown Base Age is ever wanted, that is the cleaner lever and it needs no machinery.\n\n## Why it is still not adopted\n\n- The bar, written into the STATUS doc before the first cell ran, was +0.10 Calmar OR +2pp CAGR at no worse drawdown, on 20 of 30 seeds, in both windows, on a plateau, surviving 40 bps. The best cell in the study delivers +0.094 Calmar and +1.66pp CAGR. It misses.\n- That best cell needed a SECOND, post-hoc choice: giving the freed slot to the highest-RS entrant rather than the most liquid one lifts the rule from 0.673 to 0.715 at a -31.78% drawdown. Three entrant priorities were tried after the rotation result was known, on top of a 1-of-20 margin pick, inside 57 selection cells. Discount accordingly; the forward-looking number is the pre-registered 0.673.\n- Rotation makes the book HARDER to hold: win rate 49.1% to 47.9%, worst losing streak 14 to 15, profit concentrated in the ten best realisations 36.8% to 42.1%, and tax up 30.1% (Rs 8,466,719 to Rs 11,018,798 on a Rs 10 lakh book over 21.7 years — modelled exactly through the financial-year netting engine, never haircut). The +1.68pp is what survives all of it.\n- The book has never traded live. Adding a second untested mechanic on its first day of real money is the wrong order of operations.\n\n## Trimming winners: NO EDGE — and the reason is the finding\n\n- Month-end trimming has no plateau at all: 1.25x gives Calmar 0.595, 1.5x 0.625, 1.75x 0.613, 2x 0.593, 2.5x 0.646, 3x 0.613 — and the best of them, 2.5x, wins by NEVER FIRING (0.1 trims a year; at 3x no position in twenty-one years ever breaches the threshold at a month-end).\n- You CAN abolish the cash-refusal problem. Demand-trimming at 1.5x target plus letting a short entry take whatever cash there is drives cash refusals from 1,953 to 67 and lifts entries from 689 to 801. It buys +0.97pp of CAGR and +0.006 of Calmar.\n- Because the refusals do not disappear — they MIGRATE. Slot refusals rise from 978 to 2,796 and the total the book turns away barely moves, 2,931 to 2,863. A Base Age position is held for months, so 16 slots absorb roughly 700 to 800 entries in twenty-one years no matter how the money is arranged. CASH AND SLOTS ARE ONE CONSTRAINT — SLOT-TIME — WEARING TWO HATS. That refines research/164 closing claim: drift is what LABELS the refusal, not what causes it.\n- The interaction is subtractive: bolting the drift fixes onto the rotation rule takes it from 0.673 down to 0.641 and 0.625.\n\n## A correction to research/164\n\nresearch/164 recommended ranking a contested slot by the largest 20-day traded value, citing a -32.67% drawdown and 0.665 Calmar. A ranked tie-break is DETERMINISTIC: it consumes no randomness, so it has exactly ONE path and no ensemble. Re-run at five idle-cash rates, its CAGR is stable — 21.72, 21.76, 21.40, 21.70, 21.76% at 5.0 / 5.1 / 5.2 / 5.3 / 5.5%, always above the random draw — but its drawdown is a coin flip: -32.67, -32.63, -35.93, -32.54, -32.45. The rs252 rule moves the opposite way at exactly the same rate. Nothing about the market changes; only which entries the book can afford on a handful of days. ADOPT THE TV20 TIE-BREAK FOR ITS CAGR AND BECAUSE A LIVE BOOK CANNOT DRAW A SEED. STRIKE ITS DRAWDOWN AND CALMAR CLAIM.\n\n## Harness proof\n\nWith rotation and trimming switched off the engine reproduces research/164 PER SEED, not merely per median: all 30 CAGR, MaxDD, Calmar, trade-count and win-rate values identical to research/164 seedstats (maximum absolute difference 0.000000), and the 30 per-seed CAGRs identical to research/163 independently-written 5.2% implementation. It also reproduces research/164 axis D at 5.0% to the digit. The one new input, the SuperTrend(14,4) LINE values, self-checks against the panel stored exit signal on 5,122,891 of 5,122,891 bars — 100.0000%.\n\n## What happens next\n\nThe live Base Age book converting under research/165 goes live EXACTLY as research/161 adopted it: 16 slots at 6.25%, SuperTrend(14,4) close trail, no stop, no rotation, no trimming. A dated re-test is registered for 13-Mar-2027, after six months of live operation: if the live entry queue really does refuse signals while a holding sits more than 10% under water, re-run axis A on the live event log and re-apply the same bar, unchanged.',
    status: 'COMPLETE',
    date: '2026-09-13',
    cardBlurb:
      'Arun asked whether a full book should rank its holdings and swap a weak one out for a better new signal. It should — but only on the dullest ranking of the five tested, and by +0.094 Calmar against a bar of +0.10 written down before the run. Trimming bloated winners abolishes the cash-refusal problem and changes nothing, because cash and slots turn out to be one constraint wearing two hats.',
    cardStats: [
      { label: 'Verdict', value: 'SIGNAL — misses the bar by 0.006 Calmar' },
      { label: 'What works', value: 'Swap out a holding >10% under water: +1.68pp CAGR, 30/30 seeds' },
      { label: 'What does not', value: 'Trimming winners — refusals just move from the cash queue to the slot queue' },
    ],
    systemRules: {
      intro:
        'Every entry and exit rule is research/161 adopted spec, held completely fixed. This study adds TWO capabilities and nothing else: rotation (swap the weakest holding out for a signal the book cannot otherwise take) and drift control (trim a position above a multiple of its target weight, or size an entry to the cash there is). The harness reproduces research/164 per seed before any cell was run.',
      sharedCoreTitle: 'The book, unchanged from research/161',
      sharedCore: [
        { k: 'Signal', v: 'First close above the prior all-time-high close, where that prior high is at least 60 trading bars old AND the stock fell at least 20% below it in between.' },
        { k: 'Filters', v: 'No volume filter, no saucer filter. Liquidity floor Rs 2 crore of 20-day median traded value, funds excluded. 60-bar per-symbol re-arm.' },
        { k: 'Exit', v: 'SuperTrend(14,4) on the close. No hard stop, no target, no time stop.' },
        { k: 'Fill', v: 'Signalled on the close, filled at the NEXT day open, on BOTH legs — and on every new rule in this study too.' },
        { k: 'Book', v: 'Rs 10 lakh, NSE cash CNC, 16 slots at 6.25% of NAV, 25 bps a side, after tax at 20% short-term and 12.5% long-term with Indian financial-year loss netting.' },
        { k: 'Idle cash', v: '5.2% post-tax, credited DAILY on the cash balance, compounded, never routed through the tax settlement. Arun new standard as of 12-Sep-2026.' },
        { k: 'Window', v: '03-Jan-2005 to 11-Sep-2026, 3,619 qualifying events, 30 random-draw seeds, medians with the range and the worst seed.' },
        { k: 'What this study varies', v: 'Rotation (which holding leaves, which signal takes its place, and by what margin) and drift (trim multiple, trim timing, minimum fill size). Nothing else.' },
      ],
      riskLayer: {
        title: 'The adoption bar, pre-registered before any cell ran',
        caption: 'Strict, because this is a spec change to a book that is converting to real money right now. NOTHING cleared it — and the closest cell missed by 0.006 Calmar.',
        columns: ['#', 'Criterion', 'Best result achieved', 'Outcome'],
        rows: [
          ['1', 'At least +0.10 Calmar OR +2pp CAGR at no worse drawdown, paired on the same 30 seeds', '+0.094 Calmar and +1.66pp CAGR at a shallower drawdown', 'FAIL'],
          ['2', 'Wins on at least 20 of 30 paired seeds', '30/30 on Calmar AND 30/30 on CAGR', 'PASS'],
          ['3', 'Holds in BOTH windows', 'W1 2005-2015 wins 30/30, W2 2016-2026 wins 29/30; W2 CAGR is 4.83pp ABOVE W1', 'PASS'],
          ['4', 'Sits on a plateau — the margin neighbours within 2pp of CAGR', 'Neighbours at 7.5% and 12.5% under water run 21.82% and 21.48% against the peak 22.73%', 'PASS (a hump, not a shelf)'],
          ['5', 'Survives 40 AND 60 bps', 'Calmar 0.715 / 0.682 / 0.651 against the incumbent 0.613 / 0.574 / 0.560', 'PASS'],
          ['6', 'Beats a rate-matched random-swap null', 'Null at the same 4 swaps a year: 20.39% / -35.74% / 0.569. Beaten by +2.29pp CAGR and +0.104 Calmar', 'PASS'],
        ],
        highlightRows: [0],
      },
    },
    system: {
      intro:
        'Economic question: a slot-constrained breakout book refuses roughly four of every five qualifying signals. Some of what it holds instead is demonstrably worse than what it is refusing. Is that gap harvestable after the tax the early sale crystallises, and is the ranking doing the work or is it a stop-loss in disguise?',
      rows: [
        { k: 'Rotation trigger', v: 'A qualifying signal arrives and the book cannot take it — no free slot, OR a free slot but not enough cash. Both cases are handled by the same rule, because a swap frees a slot AND cash.' },
        { k: 'Rotation execution', v: 'Sell the weakest holding at the NEXT open and buy the entrant at that SAME open. Both legs placeable. A position bought today cannot be swapped out today.' },
        { k: 'Scores tested for the holding', v: 'Cushion above its SuperTrend line; 12-month relative strength; unrealised return since entry; bars held; distance below its own running high. All causal, read at the close of the signal bar.' },
        { k: 'Entrant priority', v: 'Largest 20-day traded value by default (research/164 live recommendation). Relative strength and base age tested as post-hoc variants.' },
        { k: 'Margins', v: 'Four per score, plus a finer grid around the winner: 2.5 / 5 / 7.5 / 10 / 12.5 / 15 / 20 percentage points under water.' },
        { k: 'Drift rules', v: 'Trim anything above 1.25x to 3x its 6.25% target back to target, at month-end or on demand when a signal is about to be refused for cash; and buy what the cash affords when it is at least 25 / 50 / 75% of a full slot.' },
        { k: 'Tax on churn', v: 'Every swap-out and every trim is a real realisation flowing through the same financial-year netting block as any exit. Reported in rupees per cell. Never approximated by a haircut.' },
        { k: 'Nulls and controls', v: 'Random swap-out at five rates including one matched to the winner 4 swaps a year; sell-without-buying; unconditional hard stops at -8 / -10 / -15%; and the three deterministic tie-breaks with no rotation at all.' },
      ],
    },
    conditions: {
      intro:
        'Everything ran on the VPS against the frozen research/161 event list. The harness is research/164 engine copy with rotation and drift added; with both switched off it reproduces research/164 per seed.',
      rows: [
        { k: 'Universe', v: '1,698 NSE daily series on a 5,378-day calendar, dead names included, no index-membership filter.' },
        { k: 'Events', v: '3,619 qualifying entries over 1,880 distinct signal days — the count research/163 and research/164 both found independently.' },
        { k: 'Fit window W1', v: '2005-01 to 2015-12. Every selection decision is checked here.' },
        { k: 'Holdout W2', v: '2016-01 to 2026-09, opened once at the end. A cell whose W2 CAGR falls more than 4pp below its W1 CAGR is declared not robust.' },
        { k: 'Ranking metric', v: 'After-tax Calmar over the full window, paired on the same 30 seeds, subject to after-tax CAGR at least the 5.2% baseline, with the tradeability gate shown every time.' },
        { k: 'Baseline', v: '20.975% CAGR (worst seed 20.010%), -34.05% MaxDD, Calmar 0.6135, 72.91% average invested — per-seed identical to research/163 independent 5.2% re-run.' },
        { k: 'Cells', v: '57 selection cells against a pre-registered budget of 120, plus 9 controls and 74 re-scorings. Roughly 4,800 simulations.' },
        { k: 'New input', v: 'SuperTrend(14,4) LINE values rebuilt for all 1,698 symbols; direction self-check against the panel stored exit signal agrees on 5,122,891 of 5,122,891 bars.' },
      ],
    },
    comparisons: [
      {
        title: 'The five rankings — which holding should leave?',
        caption: 'Best cell on each score, 30 seeds, paired against the incumbent. The two intuitive answers are the two worst answers: they raise CAGR by keeping the book in the freshest, most-extended breakouts, and pay for it with 8 to 10 extra points of drawdown.',
        columns: ['Rank the weakest holding by', 'Best margin', 'CAGR', 'MaxDD', 'Calmar', 'vs incumbent', 'Calmar seeds won', 'Swaps/yr'],
        rows: [
          ['Unrealised return since entry', 'more than 10% under water', '22.68%', '-33.52%', '0.673', '+0.049', '27/30', '4.2'],
          ['Cushion above its SuperTrend line', 'at least 7.5pp better', '24.08%', '-43.82%', '0.550', '-0.066', '0/30', '56.7'],
          ['12-month relative strength', 'at least 25pp better', '23.18%', '-41.38%', '0.560', '-0.062', '2/30', '23.8'],
          ['Bars held', 'at least 320 bars', '21.24%', '-33.81%', '0.621', '+0.009', '18/30', '0.3 — it barely fires'],
          ['Distance below its own running high', 'at least 30pp below', '20.77%', '-34.05%', '0.609', '-0.001', '8/30', '0.1 — it is the incumbent'],
          ['Never swap (incumbent)', '—', '20.98%', '-34.05%', '0.613', '—', '—', '0.0'],
        ],
        highlightRows: [0, 5],
      },
      {
        title: 'The controls — is the swap doing the work?',
        caption: 'Selling the loser without replacing it is WORSE than doing nothing, so this is not a stop-loss in disguise. A random swap at the same frequency loses 2.29pp of CAGR, so the ranking is not incidental either. And a plain -10% stop reaches a similar Calmar with none of the machinery — excluded only by the pre-registered CAGR clause.',
        columns: ['Control', 'What it isolates', 'CAGR', 'MaxDD', 'Calmar', 'Calmar seeds won'],
        rows: [
          ['The rule', 'Sell the >10% under-water holding AND buy the entrant', '22.68%', '-33.52%', '0.673', '27/30'],
          ['Sell-only', 'Sell it on the same trigger, do NOT buy', '21.81%', '-36.27%', '0.603', '14/30'],
          ['Random swap, rate-matched', 'Swap a random holding, same 4.0 swaps a year', '20.39%', '-35.74%', '0.569', '3/30'],
          ['Unconditional -10% hard stop', 'A stop with no rotation at all', '20.62%', '-30.46%', '0.677', '29/30 — but CAGR below baseline'],
          ['Unconditional -8% hard stop', 'A tighter stop', '19.80%', '-30.19%', '0.657', '26/30 — CAGR below baseline'],
          ['Incumbent', 'Never swap', '20.98%', '-34.05%', '0.613', '—'],
        ],
        highlightRows: [0, 1],
      },
      {
        title: 'The deterministic tie-break has no ensemble — a correction to research/164',
        caption: 'A ranked tie-break consumes no randomness, so it has exactly ONE path. Changing only the idle-cash rate re-orders which entries the book can afford and hands it a different path. CAGR is stable; the drawdown is a coin flip. Read the CAGR column; do not plan on the Calmar.',
        columns: ['Idle cash', 'Random draw (incumbent, 30 seeds)', 'Tie-break = tv20', 'Tie-break = rs252', 'Tie-break = base age'],
        rows: [
          ['5.0%', '20.93% / -35.50% / 0.601', '21.72% / -32.67% / 0.665', '20.59% / -36.05% / 0.571', '21.75% / -37.59% / 0.579'],
          ['5.1%', '20.98% / -34.91% / 0.605', '21.76% / -32.63% / 0.667', '20.62% / -36.02% / 0.573', '21.77% / -36.77% / 0.592'],
          ['5.2%', '20.98% / -34.05% / 0.613', '21.40% / -35.93% / 0.596', '20.90% / -32.64% / 0.640', '21.77% / -36.77% / 0.592'],
          ['5.3%', '21.00% / -34.85% / 0.611', '21.70% / -32.54% / 0.667', '20.81% / -35.97% / 0.578', '21.79% / -36.74% / 0.593'],
          ['5.5%', '21.26% / -34.80% / 0.618', '21.76% / -32.45% / 0.671', '21.11% / -32.51% / 0.649', '21.78% / -37.46% / 0.582'],
        ],
        highlightRows: [2],
      },
    ],
    results: {
      metrics: [
        { label: 'Incumbent', value: '20.98% / -34.05% / 0.613', hint: 'after tax, 25 bps, 5.2% idle cash, 30-seed median; worst seed 20.01%' },
        { label: 'Rotation, pre-registered', value: '22.68% / -33.52% / 0.673', tone: 'pos', hint: 'swap out a holding more than 10% under water; worst seed 21.70%' },
        { label: 'Rotation, post-hoc entrant', value: '22.73% / -31.78% / 0.715', tone: 'pos', hint: 'same rule, freed slot to the highest-RS entrant — 1 of 3 variants chosen after the fact' },
        { label: 'Clears the adoption bar?', value: 'NO — +0.094 Calmar against +0.10' },
        { label: 'Trimming winners', value: 'NO EDGE — +0.006 Calmar', tone: 'neg' },
        { label: 'Tax cost of the churn', value: '+Rs 2,552,079 (+30.1%)', hint: 'modelled through the FY-netting engine on a Rs 10 lakh book over 21.7 years' },
        { label: 'Cells disclosed', value: '57 selection cells, 30 seeds each' },
      ],
      charts: [
        { src: '/app/baseage-rotation-research166.png', caption: 'Growth of Rs 100 (log) — the incumbent, the pre-registered rotation rule, the post-hoc best rotation rule and the best drift rule against NIFTYBEES, with the drawdown panel beneath. Median-seed paths, after tax, 25 bps a side, 5.2% idle cash.' },
      ],
      tables: [
        {
          title: 'The margin curve — how far under water before you swap?',
          caption: 'A narrow hump, not a shelf. Below 7.5% the rule churns; above 12.5% it stops firing and the book reverts to the incumbent. The working band is roughly 7.5 to 12.5 percent, firing 2 to 6 swaps a year.',
          columns: ['Swap out a holding at least this far under water', 'CAGR', 'MaxDD', 'Calmar', 'vs incumbent', 'Swaps/yr', 'Turnover xNAV'],
          rows: [
            ['Any loss at all', '21.29%', '-40.74%', '0.522', '-0.111', '33.5', '5.78'],
            ['2.5%', '19.95%', '-39.56%', '0.504', '-0.121', '19.8', '4.19'],
            ['5%', '19.11%', '-35.20%', '0.543', '-0.070', '11.4', '3.48'],
            ['7.5%', '21.40%', '-35.61%', '0.601', '-0.019', '6.3', '2.91'],
            ['10% (the winner)', '22.68%', '-33.52%', '0.673', '+0.049', '4.2', '2.74'],
            ['12.5%', '21.68%', '-33.74%', '0.641', '+0.013', '2.4', '2.56'],
            ['15%', '20.36%', '-33.03%', '0.609', '-0.021', '1.0', '2.49'],
            ['20%', '20.94%', '-35.16%', '0.597', '-0.006', '0.2', '2.43'],
            ['Never (incumbent)', '20.98%', '-34.05%', '0.613', '—', '0.0', '2.48'],
          ],
          highlightRows: [4],
        },
        {
          title: 'Drift — does trimming fix the cash-refusal problem? Yes. Does it help? No.',
          caption: 'Read the last three columns together. Combining a demand trim with a partial fill drives cash refusals from 1,953 to 67 and lifts entries by 16% — and slot refusals rise from 978 to 2,796, so the TOTAL the book turns away barely moves. Cash and slots are one constraint wearing two hats.',
          columns: ['Rule', 'CAGR', 'MaxDD', 'Calmar', 'vs incumbent', 'Entries taken', 'Refused: no slot', 'Refused: no cash', 'Total refused'],
          rows: [
            ['Incumbent — never trim', '20.98%', '-34.05%', '0.613', '—', '689', '978', '1,953', '2,931'],
            ['Trim above 1.5x target, month-end', '20.96%', '-33.53%', '0.625', '-0.002', '717', '1,467', '1,432', '2,899'],
            ['Trim above 2x target, month-end', '20.48%', '-34.91%', '0.593', '-0.029', '695', '1,043', '1,883', '2,926'],
            ['Trim above 3x target, month-end', '20.98%', '-34.05%', '0.613', '+0.000', '689', '978', '1,953', '2,931'],
            ['Trim above 1.5x target, on demand', '21.59%', '-34.35%', '0.629', '+0.019', '720', '1,673', '1,223', '2,896'],
            ['Trim above 2x target, on demand', '21.33%', '-33.81%', '0.623', '+0.006', '697', '1,115', '1,809', '2,924'],
            ['Partial fill, min 25% of a slot', '21.11%', '-34.26%', '0.615', '-0.011', '746', '2,378', '496', '2,874'],
            ['Partial fill, min 50% of a slot', '20.68%', '-36.64%', '0.567', '-0.040', '726', '1,911', '980', '2,891'],
            ['Both — demand trim 1.5x + partial fill 25%', '22.05%', '-35.59%', '0.617', '+0.006', '801', '2,796', '67', '2,863'],
          ],
          highlightRows: [0, 8],
        },
        {
          title: 'Cost ladder and the tradeability gate',
          caption: 'The ranking is unchanged at every cost rung. What rotation buys in return, it partly pays for in holdability: a lower win rate, a longer losing streak, more profit concentrated in a few names, and 30% more tax.',
          columns: ['Cell', '25 bps', '40 bps', '60 bps', 'Win rate', 'Max losing streak', 'Trades/yr', 'Top-10 share of profit', 'Tax paid'],
          rows: [
            ['Incumbent', '20.98% / 0.613', '20.30% / 0.574', '19.46% / 0.560', '49.1%', '14', '31.8', '36.8%', 'Rs 8,466,719'],
            ['Rotation, pre-registered', '22.68% / 0.673', '21.84% / 0.639', '21.07% / 0.597', '47.9%', '15', '34.3', '42.1%', 'Rs 11,018,798'],
            ['Rotation, post-hoc entrant', '22.73% / 0.715', '21.92% / 0.682', '21.12% / 0.651', '47.2%', '15', '34.3', '43.0%', 'Rs 10,896,547'],
            ['Best drift that fires', '21.59% / 0.629', '—', '—', '—', '—', '—', '28.8%', 'Rs 9,391,322'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Outlier dependence — delete the ten best trades',
          caption: 'The ten best trades are found on the median-seed path, those ten EVENTS are removed from the event list, and the whole book is re-run on all 30 seeds so the slots they occupied are freed for whatever else qualified. A fair deletion, not a bookkeeping subtraction.',
          columns: ['Cell', 'Total book profit', 'Top-10 share', 'CAGR, all events', 'CAGR, top-10 deleted', 'Cost'],
          rows: [
            ['Incumbent', 'Rs 67,304,544', '40.3%', '20.98%', '19.18%', '-1.80pp'],
            ['Rotation, pre-registered', 'Rs 91,805,088', '48.9%', '22.69%', '20.77%', '-1.92pp'],
            ['Rotation, post-hoc entrant', 'Rs 93,275,768', '45.5%', '22.73%', '20.48%', '-2.25pp'],
            ['Demand trim 1.5x', 'Rs 76,355,104', '28.8%', '21.59%', '20.50%', '-1.10pp'],
          ],
          highlightRows: [0],
        },
      ],
    },
    winners: [
      {
        config: 'Swap out a holding more than 10% under water, one swap a day, freed slot to the most liquid entrant',
        summary:
          'The only ranking of five that beats the incumbent on the pre-registered metric. It is not a stop-loss — selling the loser without replacing it is worse than doing nothing — and it is not luck: a random swap at the same 4-a-year rate loses 2.29pp of CAGR to it. It still misses the adoption bar by 0.006 Calmar, so nothing changes.',
        metrics: [
          { k: 'After-tax CAGR', v: '22.68% (worst of 30 seeds 21.70%) against the incumbent 20.98%' },
          { k: 'MaxDD', v: '-33.52%, SHALLOWER than the incumbent -34.05%' },
          { k: 'Calmar', v: '0.673 against 0.613 — paired +0.049, 27 of 30 seeds' },
          { k: 'Both windows', v: 'W1 20.48% (30/30 seeds), W2 24.84% (25/30)' },
          { k: 'Cost ladder', v: '0.673 / 0.639 / 0.597 at 25 / 40 / 60 bps against 0.613 / 0.574 / 0.560' },
          { k: 'Churn', v: '4.2 swaps a year, turnover 2.74x NAV, tax up 30.1%' },
        ],
        rejected: [
          'Rotate into momentum (12-month relative strength): CAGR up to 23.18% but drawdown -41.38%, Calmar 0.560, beating the incumbent on 2 of 30 seeds.',
          'Rotate out of whatever is closest to its trailing stop (cushion): CAGR up to 24.08%, drawdown -43.82%, Calmar 0.550, 0 of 30 seeds. The most intuitive rule in the study is the worst.',
          'Rank by base age or by time held: only harmless at thresholds so high they fire 0.1 to 0.3 times a year, which is the incumbent under another name.',
          'More than one swap a day: 0.673 down to 0.599.',
          'Month-end trimming at any multiple from 1.25x to 3x: no plateau, and the best of them wins by never firing.',
          'Partial fills at 50% or 75% of a slot: -0.040 and -0.062 Calmar.',
          'Combining rotation with the drift fixes: subtractive, 0.673 down to 0.641 and 0.625.',
        ],
      },
    ],
    caveats: [
      'NOTHING HERE IS ADOPTED OR PAPERED. The live Base Age book converting under research/165 goes live exactly as research/161 left it: 16 slots at 6.25%, SuperTrend(14,4) close trail, no stop, no rotation, no trimming.',
      'THE WINNER MISSES THE BAR BY 0.006 CALMAR. The bar — +0.10 Calmar or +2pp CAGR at no worse drawdown, 20 of 30 seeds, both windows, a plateau, 40 bps — was written into the STATUS doc before the first cell ran, precisely so that this sentence would have to be written rather than the bar quietly moved.',
      'THE BEST CELL IS A PICK ON TOP OF A PICK. The 0.715 Calmar cell exists only because three entrant priorities were tried AFTER the rotation result was known, on top of a 1-of-20 margin choice, inside 57 selection cells. The forward-looking number is the pre-registered 0.673, not 0.715.',
      'THE PLATEAU IS A HUMP, NOT A SHELF. Calmar falls from 0.673 to 0.601 one notch below the peak margin and to 0.597 two notches above. The rule needs its threshold roughly right, which is exactly the property that does not travel well out of sample.',
      'A PLAIN -10% HARD STOP REACHES A SIMILAR CALMAR (0.677 on 29 of 30 seeds) WITH NONE OF THE MACHINERY, and was excluded only by the pre-registered clause that a cell must not lose CAGR. Anyone reading this study as "rotation is the only way to improve risk-adjusted return here" would be over-reading it.',
      'ROTATION MAKES THE BOOK HARDER TO HOLD. Win rate 49.1% to 47.9%, worst losing streak 14 to 15, ten-best-realisations share of profit 36.8% to 42.1%, tax up 30.1%, and it needs a live process to compare every holding unrealised P&L against a threshold each evening and then place two orders at the next open.',
      'THE CORRELATION AND BLEND TESTS WERE NOT RE-RUN, by construction: this study changes no entry signal and no exit rule. But rotation does tilt the held book toward younger positions, which is a real and unmeasured change to how it blends with True North and IPO Base — one more reason not to adopt it before a live soak.',
      'THE DISTANCE-BELOW-ITS-OWN-HIGH SCORE uses a running maximum over the panel window from 03-Jan-2005, not each symbol true all-time high, because the panel starts there. Exact for names listed after 2005. That score lost anyway.',
      'TRIMS ARE EXCLUDED from win rate, average win and loss, and losing-streak statistics — they are partial realisations, not round trips — but INCLUDED in turnover, tax and the ten-best-realisations share, so the tradeability columns stay comparable to research/161 and research/164.',
      'INHERITED DATA DEFECT: market_data.db is not retroactively split-adjusted, so the event builder truncates each series after any day-over-day fall worse than -35%. research/161 guard, carried over unchanged; a mitigation rather than a fix.',
      'INHERITED CONVENTION, DISCLOSED: research/161 sizes a new entry off a NAV marked at the same day CLOSE while buying at that day OPEN. It is kept verbatim so the baseline reproduces bit-exactly, and is NOT extended to any rule added here — every new rule decides on the close of the signal bar and executes at the next open.',
    ],
    githubLinks: [
      { label: 'research/166 — STATUS', href: 'https://github.com/castroarun/quantifyd/blob/main/research/166_baseage_rotation_and_drift/BASEAGE_ROTATION_AND_DRIFT_DAILY_SWEEP_STATUS.md' },
      { label: 'research/166 — RESULTS', href: 'https://github.com/castroarun/quantifyd/blob/main/research/166_baseage_rotation_and_drift/results/RESULTS.md' },
      { label: 'research/164 — the slot and sizing study this follows', href: 'https://github.com/castroarun/quantifyd/blob/main/research/164_baseage_slots_sizing/results/RESULTS.md' },
    ],
    projectPaths: [
      'research/166_baseage_rotation_and_drift/BASEAGE_ROTATION_AND_DRIFT_DAILY_SWEEP_STATUS.md',
      'research/166_baseage_rotation_and_drift/scripts/sim166.py',
      'research/166_baseage_rotation_and_drift/scripts/run166.py',
      'research/166_baseage_rotation_and_drift/scripts/report166.py',
      'research/166_baseage_rotation_and_drift/results/RESULTS.md',
      'research/166_baseage_rotation_and_drift/results/tables166.md',
      'research/166_baseage_rotation_and_drift/results/cells_full.csv',
      'research/166_baseage_rotation_and_drift/results/paired166.csv',
    ],
  },
```
