# research/164 — PUBLISH NOTE (NOT published by this agent)

The frontend was **deliberately not touched** — another agent held `frontend/`,
`static/app/` and `research/_utilities/mpf_report_build.py` while this study ran. Below is
the exact `BacktestStudy` entry to append to `frontend/src/data/backtests.ts`, plus the chart
list. Publish it once the other agent's frontend work is committed, then
`cd frontend && npm run build` (frontend-only → no backend restart, safe at any hour).

## Charts to produce and place in `frontend/public/`

None are rendered yet — this study's evidence is tabular, and the two figures worth drawing
are both cheap to generate from `results/navs_full/*.npz`:

| File | What it shows | Source |
|---|---|---|
| `baseage-slot-curve-research164.png` | the slot curve: after-tax CAGR and Calmar against slot count (6→30), with the ±range band across 30 seeds and the incumbent 16 marked | `results/cells_full.csv`, axis A rows |
| `baseage-slots-tearsheet-research164.png` | equity curve of the incumbent vs 11 slots vs NIFTYBEES with the drawdown panel beneath | `results/navs_full/A_s16_eq.npz`, `A_s11_eq.npz` + NIFTYBEES from `market_data.db` |

Set `results.charts[0] = '/app/baseage-slot-curve-research164.png'` once it exists. The study
page reads fine without them — every table below is self-contained.

## The entry

```ts
  {
    slug: 'baseage-slots-and-sizing-research164',
    title: 'OA · Base Age — how many slots, at what size, and who wins a contested slot? (research/164)',
    verdict:
      'CONCLUDED — the inherited 16 slots at 6.25% SURVIVES; no spec change before the 26-Sep paper-book call. The slot count and position size of Open Alpha · Base Age were never tested. They were inherited from the old Open Alpha, whose 680-cell sweep (research/142) was scored entirely against a same-bar look-ahead entry — and research/158 and research/159 showed those surfaces INVERT once the entry is made placeable. So this was a setting fitted on a surface we now know was the wrong one. Re-fitting it, holding every entry and exit rule fixed: 16 IS NOT THE OPTIMUM, BUT IT IS CLOSE ENOUGH THAT NOTHING CLEARS THE BAR. The Calmar curve has a genuine broad hump at 9-12 slots — 0.629 / 0.669 / 0.670 / 0.633 against the incumbent 0.601 — worth +1.1 to +1.4pp of after-tax CAGR. That advantage wins 25-27 of 30 paired seeds, appears in BOTH windows (29-30 of 30 in the fit window), sits on a real plateau and survives 25/40/60 bps. It is simply too small for a bar of +0.10 Calmar or +2pp CAGR that was written down before the run. THE TWO THINGS THE BAR DOES NOT MEASURE BOTH ARGUE THE OTHER WAY: concentrating from 16 slots to 10 raises the share of total profit coming from the ten best trades from 35.7% to 53.4%, and doubles the capacity footprint — median position 0.43% to 0.77% of the held name’s own 20-day traded value, 46% of trades above 1% versus 33%, and every figure ten times larger on a Rs 1 crore book. THE SIZE PER SLOT IS NOT A FREE PARAMETER. Holding the slot count fixed and shrinking the position only de-levers: 10 slots at 4/5/6.25/8/10% runs 12.82 / 14.68 / 16.94 / 20.05 / 22.46% CAGR against -17.12 / -21.25 / -26.13 / -32.45 / -33.57% drawdown. Calmar rises monotonically as you de-lever, which is what a zero-drawdown 5% cash sleeve does to a ratio — not alpha. A DELIBERATE CASH BUFFER DOES NOT EARN ITS PLACE AT 5%: the best buffered cell (8 slots at 6.25%, 45% invested) makes 15.46% with a -23.69% drawdown and Calmar 0.647, while 11 slots FULLY invested gives a better ratio (0.670) AND 6.6pp more return. ONE RULE DOES BEAT THE RANDOM DRAW FOR A CONTESTED SLOT: take the most liquid candidate. Ranking by 20-day traded value returns +2.60pp at 8 slots (beating 30/30 random seeds on CAGR and Calmar) and +0.78pp at 16 (29/30 on both), in both windows. The other three rules are noise — IBD-style relative strength wins at 8 slots and LOSES at 16 (5/30), settling research/160 vs research/158 in favour of research/158 for this book; least-extended is flat; longest-base-age raises CAGR but deepens drawdown and wins Calmar on only 9 of 30. The liquidity rule does not clear the adoption bar either (+0.064 Calmar), but it is free, it is deterministic — it removes the seed randomness that a LIVE book cannot reproduce anyway — and it helps capacity, so it is the one change worth a dedicated test. AND THE STRUCTURAL FINDING THAT REFRAMES THE QUESTION: at every slot count the commonest reason a qualifying signal is NOT taken is that the book has no CASH, not that it has no SLOT. At 16 slots, 3,619 qualifying events produce 688 entries, 977 refused for want of a slot and 1,955 refused for want of cash — because the book never trims a winner, so a few bloated positions absorb 95% of NAV while slots sit nominally free. The slot count is the second-order constraint; POSITION DRIFT is the first-order one, and it is untested. That is the study that should follow.',
    status: 'COMPLETE',
    date: '2026-09-12',
    cardBlurb:
      'The sixteen-slot, 6.25%-per-slot book was inherited from a sweep scored against a look-ahead entry, and never re-fitted. Re-fitting it: about ten slots is genuinely better, but not by enough to change a spec three weeks from a paper-book decision — and the outlier and capacity evidence argues the other way.',
    cardStats: [
      { label: 'Verdict', value: 'CONCLUDED — keep 16 slots at 6.25%' },
      { label: 'Honest optimum', value: '9-12 slots, worth +1.1 to +1.4pp CAGR — below the bar' },
      { label: 'Real constraint', value: 'CASH, not slots — 1,955 signals refused for cash vs 977 for slots' },
    ],
    metrics: [
      { label: 'Incumbent CAGR', value: '20.94%', hint: 'after tax, 25 bps, 30-seed median, 5.0% idle cash' },
      { label: 'Incumbent MaxDD', value: '-35.50%', tone: 'neg' },
      { label: 'Incumbent Calmar', value: '0.601' },
      { label: 'Best cell (11 slots)', value: '22.03% / -33.85% / 0.670', tone: 'pos' },
      { label: 'Clears the adoption bar?', value: 'NO — 0 of 32 cells' },
      { label: 'Cells disclosed', value: '32 selection cells, 30 seeds each' },
    ],
    systemRules: {
      intro:
        'Every entry and exit rule is research/161’s adopted spec, held completely fixed. This study varies THREE things only: how many slots, how big each position, and who wins a contested slot. The harness reproduces research/161’s published winner exactly before any cell was run.',
      sharedCoreTitle: 'The book, unchanged from research/161',
      sharedCore: [
        { k: 'Signal', v: 'First close above the prior all-time-high close, where that prior high is at least 60 trading bars old AND the stock fell at least 20% below it in between.' },
        { k: 'Filters', v: 'No volume filter, no saucer filter. Liquidity floor Rs 2 crore of 20-day median traded value. 60-bar per-symbol re-arm applied after the filters.' },
        { k: 'Exit', v: 'SuperTrend(14,4) on the close, no hard stop, no time stop.' },
        { k: 'Fill', v: 'Signalled on the close, filled at the NEXT day’s open, on BOTH legs.' },
        { k: 'Book', v: 'Rs 10 lakh, NSE cash CNC, 25 bps a side, after tax at 20% short-term and 12.5% long-term with Indian financial-year loss netting.' },
        { k: 'Idle cash', v: '5.0% post-tax, credited DAILY on the cash balance, compounded, and never routed through the tax settlement. NOT research/161’s 5.5% — Arun standardised the whole Momentum Portfolio on 5% on 12-Sep-2026. Worth 1.67pp of CAGR to this book.' },
        { k: 'Window', v: '03-Jan-2005 to 11-Sep-2026, 30 seeds, medians reported with the range and the worst seed.' },
        { k: 'What this study varies', v: 'slots, slot_pct, and the contested-slot selection rule. Nothing else.' },
      ],
      riskLayer: {
        title: 'The adoption bar, pre-registered before any cell ran',
        caption: 'Strict, because this is a spec change to a candidate book three weeks from a paper-book decision. NOTHING cleared it.',
        columns: ['#', 'Criterion', 'Best result achieved', 'Outcome'],
        rows: [
          ['1', 'At least +0.10 Calmar OR +2pp CAGR at no worse drawdown, paired on the same 30 seeds', '+0.064 Calmar (10 slots) / +2.03pp CAGR but at a deeper drawdown (liquidity rule at 8 slots)', 'FAIL'],
          ['2', 'Wins on at least 20 of 30 paired seeds', '29/30 (liquidity rule at 16 slots)', 'PASS'],
          ['3', 'Holds in BOTH windows', '9-12 slots wins 29-30/30 in the fit window, 21-22/30 in the holdout', 'PASS'],
          ['4', 'Sits on a plateau — both neighbouring slot counts within 3pp of CAGR', '9/10/11/12 slots span 21.33 to 22.46%', 'PASS'],
          ['5', 'Survives the 40 bps rung', '11 slots: 22.03% to 21.61%; incumbent 20.94% to 20.29%', 'PASS'],
          ['6', 'Passes the tradeability gate — losing streak, trades per year, capacity', 'Concentration WORSENS capacity: median position 0.43% to 0.77% of the name’s own traded value, trades above 1% 33% to 46%', 'FAIL'],
        ],
        highlightRows: [0, 5],
      },
    },
    system: {
      intro:
        'Economic question: the old Open Alpha’s 16-slot / 6.25% book was chosen on a parameter surface that research/158 and research/159 showed inverts once the entry is placeable. research/161 swept age, depth, volume, saucer shape, exits and hard stops — and has no slot column at all. This closes that gap.',
      rows: [
        { k: 'Events', v: '3,619 qualifying entries over 1,880 distinct signal days — the same count research/163 independently found. No day ever carries 16 simultaneous signals; the maximum is 13 and the median is 1.' },
        { k: 'Axis A — concentration', v: 'slot_pct = 1/slots so a full book is ~100% invested. slots in {6,7,8,9,10,11,12,13,14,16,18,20,24,30}.' },
        { k: 'Axis B — cash buffer', v: 'slot_pct fixed at 6.25%, slots in {8,10,12,14,16} so the maximum invested share is 50 / 62.5 / 75 / 87.5 / 100%. Never above 100% — no leverage anywhere.' },
        { k: 'Axis C — size independent of count', v: 'slots in {10,16,20} x slot_pct in {4, 5, 6.25, 8, 10%}, dropping anything above 100% invested. Separates "how many names" from "how big each bet".' },
        { k: 'Axis D — contested slots', v: 'seeded random draw (the incumbent AND the null control), IBD-style relative strength, least extended above the prior high, largest 20-day traded value, longest base age — each at 8 and 16 slots.' },
        { k: 'Cells disclosed', v: '32 selection cells at 30 seeds each: 26 pre-registered plus 6 extra slot counts (7, 9, 11, 13, 14, 18) added to execute the pre-registered plateau test. The top cell, 11 slots, is one of the six added AFTER pre-registration — stated because that is exactly the kind of winner multiple testing manufactures. Re-scorings that are not additional selection: 2 harness-proof cells, 32 at zero idle yield, 32 at 40 bps, 32 at 60 bps. 3,860 simulations.' },
        { k: 'Harness proof', v: 'At 5.5% idle cash the harness reproduces research/161’s published winner to the last digit: 21.26% CAGR, worst seed 19.87%, -34.80% drawdown, Calmar 0.618. At 5.0% it matches research/163’s independent re-run exactly: 20.94% / 19.81% / -35.50% / 0.601, 72.9% invested.' },
      ],
    },
    results: {
      charts: [],
      tables: [
        {
          title: 'Axis A — the slot curve, fully invested',
          caption: 'Full window, after tax, 25 bps a side, 5.0% idle cash, 30 seeds, medians. 16 slots is the incumbent. Read the last two columns: at every slot count more signals are refused for want of CASH than for want of a SLOT.',
          columns: ['Slots', 'Size/slot', 'CAGR', 'Worst seed', 'MaxDD', 'Calmar', 'Invested', 'Trades/yr', 'Max losing streak', 'Refused: no slot', 'Refused: no cash'],
          rows: [
            ['6', '16.67%', '21.89%', '20.69%', '-42.50%', '0.516', '74.1%', '12.3', '11', '1,785', '1,567'],
            ['8', '12.50%', '20.37%', '19.08%', '-35.64%', '0.571', '75.3%', '16.6', '11', '1,587', '1,671'],
            ['9', '11.11%', '21.56%', '20.89%', '-34.26%', '0.629', '75.0%', '18.7', '11', '1,533', '1,680'],
            ['10', '10.00%', '22.46%', '20.70%', '-33.57%', '0.669', '75.0%', '20.7', '11', '1,442', '1,733'],
            ['11', '9.09%', '22.03%', '20.45%', '-33.85%', '0.670', '74.9%', '22.5', '12', '1,371', '1,759'],
            ['12', '8.33%', '21.33%', '19.88%', '-33.60%', '0.633', '74.2%', '24.4', '13', '1,225', '1,863'],
            ['16 (incumbent)', '6.25%', '20.94%', '19.81%', '-35.50%', '0.601', '72.9%', '31.7', '14', '977', '1,955'],
            ['20', '5.00%', '20.57%', '19.42%', '-34.86%', '0.585', '71.0%', '38.3', '15', '781', '2,008'],
            ['24', '4.17%', '19.87%', '19.32%', '-35.12%', '0.566', '69.0%', '45.1', '15', '617', '2,024'],
            ['30', '3.33%', '17.55%', '17.26%', '-31.08%', '0.565', '66.3%', '55.1', '15', '423', '2,000'],
          ],
          highlightRows: [6],
        },
        {
          title: 'Axes B and C — the cash buffer and the size dial',
          caption: 'Not one of these beats the incumbent on CAGR. Calmar rises as the book de-levers, which is what a zero-drawdown 5% cash sleeve does to a ratio. "Cash sleeve" is the CAGR lost when the idle yield is switched to zero.',
          columns: ['Cell', 'Slots', 'Size/slot', 'Max invested', 'CAGR', 'MaxDD', 'Calmar', 'Avg invested', 'Cash sleeve'],
          rows: [
            ['B — buffer', '8', '6.25%', '50%', '15.46%', '-23.69%', '0.647', '45.3%', '2.89pp'],
            ['B — buffer', '10', '6.25%', '62%', '16.94%', '-26.13%', '0.651', '54.6%', '2.37pp'],
            ['B — buffer', '12', '6.25%', '75%', '18.52%', '-30.57%', '0.601', '63.3%', '2.44pp'],
            ['B — buffer', '14', '6.25%', '88%', '20.73%', '-33.45%', '0.615', '71.2%', '1.97pp'],
            ['C — size dial', '10', '4.00%', '40%', '12.82%', '-17.12%', '0.754', '35.9%', '3.30pp'],
            ['C — size dial', '10', '5.00%', '50%', '14.68%', '-21.25%', '0.691', '44.3%', '2.89pp'],
            ['C — size dial', '10', '8.00%', '80%', '20.05%', '-32.45%', '0.627', '68.5%', '1.64pp'],
            ['C — size dial', '16', '4.00%', '64%', '16.07%', '-25.84%', '0.621', '52.8%', '2.36pp'],
            ['C — size dial', '16', '5.00%', '80%', '18.68%', '-31.01%', '0.589', '65.0%', '1.69pp'],
            ['C — size dial', '20', '4.00%', '80%', '17.74%', '-29.88%', '0.593', '62.7%', '2.18pp'],
            ['Incumbent', '16', '6.25%', '100%', '20.94%', '-35.50%', '0.601', '72.9%', '1.67pp'],
          ],
          highlightRows: [10],
        },
        {
          title: 'Axis D — who wins a contested slot, against the random-draw null',
          caption: 'The ranked rules are DETERMINISTIC — they consume no randomness, so each has a single path. "Seeds beaten" is that one path against all 30 random-draw paths, which is the honest test of a fixed rule against the draw’s luck. Only the liquidity rule is consistent across both slot counts.',
          columns: ['Rule', 'Slots', 'CAGR', 'MaxDD', 'Calmar', 'vs random', 'Seeds beaten (CAGR)', 'Seeds beaten (Calmar)'],
          rows: [
            ['Random draw (null / incumbent)', '8', '20.37%', '-35.64%', '0.571', '—', '—', '—'],
            ['Relative strength (12-month)', '8', '21.62%', '-35.64%', '0.607', '+1.25pp', '25/30', '25/30'],
            ['Least extended above prior high', '8', '21.75%', '-36.46%', '0.596', '+1.38pp', '26/30', '23/30'],
            ['Largest 20-day traded value', '8', '22.97%', '-35.64%', '0.645', '+2.60pp', '30/30', '30/30'],
            ['Longest base age', '8', '19.75%', '-35.64%', '0.554', '-0.62pp', '8/30', '8/30'],
            ['Random draw (null / incumbent)', '16', '20.94%', '-35.50%', '0.601', '—', '—', '—'],
            ['Relative strength (12-month)', '16', '20.59%', '-36.05%', '0.571', '-0.35pp', '5/30', '5/30'],
            ['Least extended above prior high', '16', '20.86%', '-35.99%', '0.579', '-0.08pp', '9/30', '9/30'],
            ['Largest 20-day traded value', '16', '21.72%', '-32.67%', '0.665', '+0.78pp', '29/30', '29/30'],
            ['Longest base age', '16', '21.75%', '-37.59%', '0.579', '+0.81pp', '30/30', '9/30'],
          ],
          highlightRows: [3, 8],
        },
        {
          title: 'The shortlist against the incumbent — paired on the same 30 seeds, in both windows',
          caption: 'W1 (fit) 2005-01 to 2015-12; W2 (holdout) 2016-01 to 2026-09, opened once at the end. Window drawdowns are measured from the running peak of the FULL curve. No cell clears the bar.',
          columns: ['Cell', 'CAGR', 'MaxDD', 'Calmar', 'dCalmar', 'Calmar seeds won', 'dCAGR', 'W1 CAGR (seeds won)', 'W2 CAGR (seeds won)', 'Clears the bar?'],
          rows: [
            ['16 slots (incumbent)', '20.94%', '-35.50%', '0.601', '—', '—', '—', '18.94%', '22.90%', '—'],
            ['11 slots', '22.03%', '-33.85%', '0.670', '+0.058', '26/30', '+1.13pp', '19.72% (29/30)', '24.38% (22/30)', 'NO — below +0.10 Calmar and +2pp CAGR'],
            ['10 slots', '22.46%', '-33.57%', '0.669', '+0.064', '25/30', '+1.40pp', '20.20% (29/30)', '24.82% (21/30)', 'NO — below +0.10 Calmar and +2pp CAGR'],
            ['Liquidity rule, 16 slots', '21.72%', '-32.67%', '0.665', '+0.064', '29/30', '+0.78pp', '19.67% (30/30)', '23.73% (22/30)', 'NO — below +0.10 Calmar and +2pp CAGR'],
            ['Liquidity rule, 8 slots', '22.97%', '-35.64%', '0.645', '+0.044', '25/30', '+2.03pp', '18.27% (0/30)', '27.94% (30/30)', 'NO — deeper drawdown, and 0/30 in the fit window'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Cost ladder and outlier dependence',
          caption: 'The ranking is unchanged at every cost rung. The outlier test identifies the ten best trades on the median-seed path, DELETES those ten events from the event list, and re-runs all 30 seeds — so the slots they occupied are freed for whatever else qualified. Concentration makes the book MORE tail-carried.',
          columns: ['Cell', '25 bps', '40 bps', '60 bps', 'Profit from the 10 best trades', 'CAGR with those 10 events deleted'],
          rows: [
            ['16 slots (incumbent)', '20.94% / 0.601', '20.29% / 0.569', '19.39% / 0.560', '35.7%', '18.62% (-2.31pp)'],
            ['11 slots', '22.03% / 0.670', '21.61% / 0.639', '20.59% / 0.600', '48.5%', '19.99% (-2.04pp)'],
            ['10 slots', '22.46% / 0.669', '21.55% / 0.645', '20.76% / 0.617', '53.4%', '20.29% (-2.18pp)'],
            ['Liquidity rule, 16 slots', '21.72% / 0.665', '21.02% / 0.633', '20.08% / 0.536', '35.7%', '19.88% (-1.84pp)'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Capacity — the argument against concentrating',
          caption: 'Measured on a Rs 10 lakh book. Every percentage scales linearly with capital: at Rs 1 crore each figure is TEN TIMES larger, which is the number that matters for sizing this book up.',
          columns: ['Cell', 'Median position', 'As % of the held name’s own 20-day traded value', 'p95', 'Trades above 1% of it'],
          rows: [
            ['16 slots (incumbent)', 'Rs 5,42,587', '0.426%', '4.70%', '32.9%'],
            ['11 slots', 'Rs 8,88,504', '0.673%', '8.89%', '42.4%'],
            ['10 slots', 'Rs 9,11,877', '0.772%', '10.70%', '45.9%'],
            ['Liquidity rule, 16 slots', 'Rs 6,02,579', '0.435%', '5.17%', '34.4%'],
          ],
          highlightRows: [0],
        },
      ],
    },
    caveats: [
      'NOTHING HERE IS ADOPTED OR PAPERED. The adopted spec is unchanged: 16 slots at 6.25% goes into the 26-Sep-2026 paper-book call exactly as research/161 left it.',
      'THE TOP CELL WAS ADDED AFTER PRE-REGISTRATION. 11 slots is one of six slot counts (7, 9, 11, 13, 14, 18) added to execute the pre-registered plateau test, so 32 cells were selected over rather than the 26 pre-registered. An improvement of +0.058 Calmar found at the peak of a 32-cell grid is the size of thing multiple testing manufactures. 10 and 11 slots are statistically indistinguishable (0.669 vs 0.670) — the finding is "about ten", never "eleven".',
      'THE SLOT COUNT IS THE SECOND-ORDER KNOB. At every slot count more qualifying signals are refused for want of CASH than for want of a SLOT, because the book never trims a winner and a few bloated positions can absorb 95% of NAV while slots sit nominally free. Fixing position drift — trimming a bloated winner toward its target weight, or sizing the next entry to available cash instead of skipping it — is untested and is the study that should follow. A dated review is registered for 2026-10-10.',
      'THE LIQUIDITY TIE-BREAK IS 1 OF 8 CELLS ON ITS AXIS. Discount it for multiple testing. It is also the weakest of the shortlist at 60 bps (Calmar 0.536 against the incumbent’s 0.560). What recommends it is not its size but its shape: it is free, needs no new data, is deterministic — a LIVE book cannot "draw a seed", and the spec currently gives the operator no written tie-break at all — and it tilts toward larger names, which helps capacity.',
      'THE CORRELATION AND BLEND TESTS WERE NOT RE-RUN, by construction: this study changes no entry and no exit, so the book’s correlation with True North and IPO Base is research/161’s and research/154’s, unchanged.',
      'INHERITED DATA DEFECT: market_data.db is not retroactively split-adjusted, so the event builder truncates each series after any day-over-day fall worse than -35%. This is research/161’s guard, carried over unchanged, and it is a mitigation rather than a fix.',
    ],
  },
```
