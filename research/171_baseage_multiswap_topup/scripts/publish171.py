# -*- coding: utf-8 -*-
"""research/171 -- insert the study entry at the top of frontend/src/data/backtests.ts.

Idempotent: refuses to run twice (the slug is checked first).
"""
import sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
TS = ROOT / 'frontend' / 'src' / 'data' / 'backtests.ts'
MARK = 'export const BACKTEST_STUDIES: BacktestStudy[] = [\n'
SLUG = 'baseage-multiswap-topup-research171'

ENTRY = r"""  {
    slug: 'baseage-multiswap-topup-research171',
    title: 'Open Alpha - Base Age: should the book swap TWO holdings a night instead of one, and should the money buy more of what it already owns? (research/171)',
    verdict:
      'NO EDGE on both of the questions as asked. The live Base Age book keeps OA-ROT-1 exactly as staged and nothing is deployed.\n\n## Swapping two instead of one is not a choice the market offers\n\n- Once the single-swap rule is running, two holdings are more than 10% under water on the same refused-signal evening **eleven times in twenty-one and a half years** - one occasion every two years. On a book that never swaps it would be 4.2 times a year; the rule destroys its own second opportunity by removing the loser each time.\n- **k = 3, k = 4, k = 6 and all-eligible are bit-identical to each other on all 30 seeds.** A third swap in a single evening happens on one path out of thirty and never again. There is no optimum to find because the axis has only two distinguishable points.\n- k = 2 is a wash at 25 bps (-0.21pp of CAGR, -0.003 of Calmar, winning 11 of 30 paired seeds) and a real loss at 40 bps: Calmar 0.671 against the staged rule 0.684.\n\n## Topping up existing winners is the worst destination tested on this book\n\n- Eighteen constructions: the holding ranked by 12-month relative strength, unrealised return or cushion above its SuperTrend line, at k = 1 or 2, with a position cap of 2x, 3x or none.\n- **All eighteen lose risk-adjusted to doing nothing, and not one beats the staged rule on a single seed out of thirty.** The best of them reaches Calmar 0.587 against the incumbent 0.606 and OA-ROT-1 0.702.\n- The mechanism is visible: a top-up consumes no event and takes no slot, so the book drops to fifteen names and gives up an independent bet to buy more of one it already owns. A single position reaches **52% of NAV** under the relative-strength ranking with no cap and **81%** under the unrealised ranking, the drawdown goes to -36% to -42%, and the ten best trades come to supply 55% of all book profit against the incumbent 39%.\n- The two pre-registered follow-ups did not rescue it: splitting across the best two holdings gives 0.562, firing on any evening gives 0.587.\n\n## One construction clears the bar - and it is not the idea that was asked about\n\nSell the under-water holding, give the money to the refused entrant when a signal is going begging and otherwise buy more of the holding with the most cushion above its trailing stop. That cell returns 24.70% after tax at -33.62% and Calmar 0.735: **+0.128 paired Calmar on 30 of 30 seeds** against a pre-registered bar of +0.10, positive in both windows on 30/30, and it survives 40 bps. Four reasons it is still not an adoption:\n\n- **Its top-up leg does nothing on the evenings the question described.** Restrict the identical rule to refused-signal evenings only and it returns 22.30% / -31.78% / 0.702 - OA-ROT-1 to the digit, with zero top-ups. With one sale a night the entrant swap always consumes it. **Every point of the advantage comes from sales made on evenings when no signal fired at all**, which is an unconditional -10% stop bolted on to the book.\n- **Most of what that stop earns, a plain stop earns alone.** Incumbent 0.606 -> unconditional -10% hard stop with the money in cash 0.683 -> the same sale through the rotation path 0.692 -> plus the staged entrant swap 0.702 -> plus the redeployment 0.735. Two thirds of the risk-adjusted gain needs no rotation machinery at all. research/166 said this in its caveat 4; it has now reproduced on a third seed set.\n- **Ninety-four percent of its return edge lives in the holdout half of the history**: +0.44pp in 2005-2015 against +7.03pp in 2016-2026. The staged rule, by contrast, earns +1.26pp and +1.40pp - the same edge in both halves, which is what a durable mechanic looks like.\n- **It stops being a sixteen-name book.** One position reaches 40.8% of NAV, the win rate falls from 48.9% to 42.2%, the ten best trades supply 57.3% of profit, and the tax bill rises 59%.\n\n## What it changes\n\n- **Operationally nothing.** The live book keeps OA-ROT-1: one swap an evening, the holding more than 10% under water leaves, the refused entrant with the highest 252-day relative strength takes the slot.\n- Registered for the **26-Sep-2026** review: expect about 4.5 swaps a year and roughly **one occasion every two years** where a second holding would also have qualified. Do not read that second loser as a missed opportunity - on twenty-one years of history, taking it was worth -0.21pp.\n- **The unconditional -10% stop keeps turning up.** Third independent appearance, at Calmar 0.683 to 0.692 with no machinery, failing only the CAGR-eligibility clause by about a tenth of a percentage point. If a lower-drawdown Base Age is ever wanted, that is the lever, and it deserves its own study with its own bar.',
    status: 'COMPLETE',
    date: '2026-09-13',
    cardBlurb:
      'Arun asked whether the Base Age book should swap the last two ranks instead of one, and whether the proceeds should top up the holdings already running hardest rather than buy a new breakout. Two eligible losers coincide once every two years, so there is no second swap to make; and every one of eighteen top-up constructions loses to simply buying the new name.',
    cardStats: [
      { label: 'Swap two instead of one', value: 'NO EDGE - the case arises 0.5 times a year' },
      { label: 'Top up existing winners', value: 'NO EDGE - 0 of 18 beat the staged rule on any seed' },
      { label: 'Live book', value: 'UNCHANGED - OA-ROT-1 as staged' },
    ],

    systemRules: {
      intro:
        'The book is research/161 adopted Open Alpha - Base Age spec, held completely fixed. Only the rotation mechanic varies. The engine is generated from research/170 sim170.py by twelve exact-string patches that must each match exactly once, and at its defaults it reproduces research/170 incumbent and OA-ROT-1 cells bit for bit - identical NAV array, identical trade list, identical book dictionary.',
      sharedCoreTitle: 'The book, unchanged in every cell',
      sharedCore: [
        { k: 'Entry event', v: 'First close above the prior all-time-high close, where that prior high is at least 60 trading bars old and the stock fell at least 20% below it in between, with 20-day median traded value at least Rs 2 crore and a 60-bar per-symbol re-arm. Filled at the NEXT open.' },
        { k: 'Exit', v: 'SuperTrend(14, 4) turning down on the close, sold at the next open. No hard stop, no target, no time stop.' },
        { k: 'Slots and sizing', v: '16 slots at 6.25% of NAV, Rs 10 lakh book, NSE cash CNC, long only. Contested slots broken by a seeded random draw - the ensemble axis.' },
        { k: 'The incumbent', v: 'Never swap. This is the book as research/161 adopted it and the thing every cell must beat.' },
        { k: 'OA-ROT-1, the rule staged live today', v: 'On an evening when a qualifying signal is refused, sell the ONE holding more than 10% below its buy price (largest loss first) and buy the refused entrant with the highest 252-day relative strength. Both legs at the next open. At most one swap per evening.' },
        { k: 'What a top-up is, and is not', v: 'A top-up buys more of a holding the book already owns. It consumes no event, occupies no slot and cannot re-arm anything - the 60-bar re-arm lives in the event generator and is blind to the book. The position keeps its original entry index, its buy price becomes the weighted average so the -10% trigger is measured against what the book actually paid, and the purchase is recorded as its own tax lot so yesterday shares cannot inherit a two-year-old long-term clock. A position bought or topped up today cannot be sold today.' },
        { k: 'Costs and tax', v: '25 bps a side, with 40 and 60 bps as a sensitivity on the whole shortlist. 20% short-term / 12.5% long-term capital gains above 365 days with Indian financial-year loss netting, settled 1 April, modelled inside the engine and never haircut.' },
        { k: 'Idle cash', v: '5.2% post-tax, credited daily on the cash balance.' },
        { k: 'Paths', v: '30 seeds, seed base 7000 (seeds 7001 to 7030) - a set no cell in research/164, 166 or 170 has ever touched. Medians reported with the worst seed.' },
        { k: 'Windows', v: '2005-01-03 to 2026-09-11. Fit W1 to 2015-12-31, holdout W2 from 2016-01-01, both pre-registered with a 4-percentage-point deterioration rule. W3 covering 2025-01 to 2026-09 is a REPORTING window - the regime the live book is converting into - and was never used for selection.' },
      ],
      riskLayer: {
        title: 'The adoption bar, written into the STATUS doc before the first cell ran',
        caption: 'The bar is research/166 own, applied unchanged. The one cell that clears it is reported against it honestly, and then refused for reasons the bar does not encode.',
        columns: ['Criterion', 'Best result achieved', 'Outcome'],
        rows: [
          ['Paired at least +0.10 Calmar OR +2pp CAGR at no worse drawdown vs the INCUMBENT', 'Axis A best +0.087. Axis B best -0.023 (every cell negative). Axis C best +0.128', 'A FAIL / B FAIL / C PASS'],
          ['Wins on at least 20 of 30 seeds', 'Axis A 30/30 but identical to the staged rule. Axis B 10/30 at best. Axis C 30/30', 'A n/a / B FAIL / C PASS'],
          ['Beats the rule ALREADY STAGED (OA-ROT-1)', 'Axis A -0.003 on 11/30. Axis B -0.117 on 0/30. Axis C +0.030 on 29/30', 'A FAIL / B FAIL / C PASS'],
          ['Holds in BOTH pre-registered windows', 'Axis C winner +0.44pp in W1 against +7.03pp in W2, both on 30/30 - passes the letter, but a sixteen-fold asymmetry', 'PASS (flagged)'],
          ['Sits on a plateau', 'Margin 7.5 / 10 / 12.5% gives Calmar 0.700 / 0.735 / 0.617, and the tightening neighbour degrades to a wash. Position cap 1.5x / 2x / 3x / none gives 0.677 / 0.671 / 0.735 / 0.720 - a step at the loose end, not a peak', 'MARGINAL'],
          ['Survives 40 bps a side', 'Axis C rs252 variant collapses to 0.638, BELOW the staged rule 0.684. The cushion variant holds at 0.702', 'MIXED'],
          ['Beats a rate-matched random-swap null', '0.735 against 0.558 at the same 9 swaps a year', 'PASS'],
          ['The mechanic actually does what was asked', 'Restricted to refused-signal evenings, the winning cell returns OA-ROT-1 to the digit with ZERO top-ups', 'FAIL - decisive'],
        ],
        highlightRows: [7],
      },
    },

    system: {
      intro:
        'Three axes, all on the same book. A: how many holdings may leave in one evening. B: whether the proceeds buy a new breakout or more of an existing holding. C: mixtures of the two. Fifty-one selection cells against a budget of seventy, plus twelve controls and twenty-nine cost re-scorings.',
      rows: [
        { k: 'Axis A - how many leave', v: 'k = 1 (the staged rule), 2, 3, 4, 6 and all-eligible, where eligible means a holding more than the margin under water, ordered largest-loss-first. Crossed with margins of 7.5 and 12.5% at k = 2. Because each swap consumes one refused entrant, k is bounded by the SMALLER of the eligible losers and the refused signals - which is the right model of the live rule, since you cannot buy an entrant that did not signal.' },
        { k: 'Axis A - the spill variants', v: 'What happens to a second eligible loser when there is no second entrant to buy: keep it (the engine own behaviour), sell it and hold the cash at 5.2%, or sell it and top up the strongest holding. Built precisely to test the unbounded version of k.' },
        { k: 'Axis B - the destination', v: 'Same sell trigger, but the proceeds buy more of the holding ranked HIGHEST by 12-month relative strength, unrealised return since entry, or cushion above its SuperTrend line. Crossed with k = 1 or 2 and a position cap of 2x, 3x or none of the 6.25% target. Eighteen cells. If nothing can be bought, the sale is cancelled - the same cancel-both-legs discipline OA-ROT-1 uses.' },
        { k: 'Axis C - mixtures', v: 'A 50/50 split between the best refused entrant and the top existing holding, and entrant-when-a-signal-is-refused-otherwise-top-up, each at k = 1 and 2.' },
        { k: 'The eligibility histogram', v: 'A recording inside the engine, added for this study: on every evening when at least one qualifying signal was refused, how many open positions were simultaneously more than 10% below their average buy price. It decides nothing and draws no random number. This is the number that settles Axis A, and it is reported before any CAGR.' },
        { k: 'Controls and nulls', v: 'The incumbent; OA-ROT-1; a measure-only cell with k = 0 that must reproduce the incumbent exactly; sell-only at k = 1, 2 and 3 (research/166 found selling without redeploying LOSES to doing nothing - it must reproduce); random swap at the matched rate and at the elevated rate the Axis-C family fires at; and a plain unconditional -10% hard stop with no rotation machinery at all.' },
      ],
    },

    conditions: {
      intro:
        'Everything ran on the VPS against frozen inputs. No live file, no services module, no database and no other research folder was written by this study. research/170 sim170.py was never edited - sim171.py is generated from it.',
      rows: [
        { k: 'Universe and events', v: 'research/166 frozen event list: 3,619 qualifying all-time-high breakouts over 1,698 symbols, built from every NSE daily series in market_data.db with at least 90 bars, dead names included, no index-membership filter.' },
        { k: 'Price panel', v: 'research/164 panel164.pkl - 1,698 symbols over 5,378 trading days, 2005-01-03 to 2026-09-11, closes forward-filled for marking and opens left as NaN so a missing bar cannot be filled.' },
        { k: 'SuperTrend lines', v: 'research/166 st166.pkl, whose rebuilt direction agrees with the panel stored exit signal on 5,122,891 of 5,122,891 bars.' },
        { k: 'Harness proof, before any selection cell', v: 'sim171.py at its defaults returns an identical NAV array, trade list, invested series and book dictionary to research/170 sim170.py on the incumbent and on OA-ROT-1, on seeds 1001 and 1017. Run on research/170 own seed base it returns the incumbent at 20.945 / -34.045 / 0.611 and OA-ROT-1 at 22.58 / -31.78 / 0.710 - the published rows to the digit. A second check: the measure-only cell reproduces the incumbent on all 30 new seeds.' },
        { k: 'Data snapshot', v: 'backtest_data/market_data.db as of 12-Sep-2026. Not retroactively split-adjusted, so the inherited defence applies: each series is truncated after any day-over-day fall worse than -35%.' },
        { k: 'Compute', v: 'Two workers at nice 10 on the VPS. About 2,820 simulations; the main 38-cell grid finished in 249 seconds.' },
      ],
    },

    comparisons: [
      {
        title: 'THE NUMBER THAT SETTLES AXIS A - how often are two holdings eligible at once?',
        caption: 'On an evening when at least one qualifying signal was REFUSED, how many open positions were simultaneously more than 10% below their average buy price? Medians across 30 seeds over 21.7 years. Read the second row: once the single-swap rule is running it keeps removing the loser, so a second one almost never accumulates.',
        columns: ['Book', 'Refused-signal evenings', 'at least 1 eligible', 'at least 2 eligible', 'at least 3 eligible', 'Most ever at once', 'Evenings per year with 2 or more'],
        rows: [
          ['The INCUMBENT own path (never swaps)', '1,396', '418 (30.0%)', '92 (6.6%)', '16 (1.1%)', '4', '4.2'],
          ['OA-ROT-1 running (k = 1, staged live)', '1,412', '97 (6.9%)', '11 (0.8%)', '2 (0.1%)', '3', '0.5'],
          ['k = 2', '1,408', '95 (6.7%)', '9 (0.6%)', '2 (0.1%)', '4', '0.4'],
          ['k = all eligible', '1,407', '94 (6.7%)', '9 (0.6%)', '2 (0.1%)', '4', '0.4'],
        ],
        highlightRows: [1],
      },
      {
        title: 'AXIS A - how many holdings should leave per evening',
        caption: 'After tax, net of 25 bps, medians across 30 seeds. k = 3, k = 4, k = 6 and all-eligible are bit-identical to EACH OTHER on every seed; k = 2 differs from them on one seed out of thirty. The axis has two distinguishable points and one wins.',
        columns: ['Cell', 'CAGR', 'Worst seed', 'MaxDD', 'Calmar', 'dCalmar vs INCUMBENT', 'dCalmar vs OA-ROT-1', 'Swaps/yr', 'Calmar at 40 bps'],
        rows: [
          ['INCUMBENT - never swap', '20.91%', '20.16%', '-35.16%', '0.606', '-', '-0.090 on 0/30', '0.0', '0.600'],
          ['OA-ROT-1 - k = 1 (staged live)', '22.30%', '21.37%', '-31.78%', '0.702', '+0.090 on 30/30', '-', '4.5', '0.684'],
          ['k = 2 - sell the two weakest', '22.09%', '20.99%', '-31.78%', '0.695', '+0.087 on 30/30', '-0.003 on 11/30', '4.5', '0.671'],
          ['k = 3', '22.09%', '20.99%', '-31.78%', '0.695', '+0.087 on 30/30', '-0.003 on 11/30', '4.5', '-'],
          ['k = 4 (= all eligible capped at 25% of the book)', '22.09%', '20.99%', '-31.78%', '0.695', '+0.087 on 30/30', '-0.003 on 11/30', '4.5', '-'],
          ['k = 6', '22.09%', '20.99%', '-31.78%', '0.695', '+0.087 on 30/30', '-0.003 on 11/30', '4.5', '-'],
          ['k = all eligible', '22.09%', '20.99%', '-31.78%', '0.695', '+0.087 on 30/30', '-0.003 on 11/30', '4.5', '-'],
          ['k = 2, margin 7.5%', '22.22%', '20.93%', '-32.13%', '0.692', '+0.081 on 30/30', '-0.005 on 13/30', '6.6', '-'],
          ['k = 2, margin 12.5%', '21.45%', '20.28%', '-33.50%', '0.625', '+0.004 on 17/30', '-0.074 on 0/30', '2.5', '-'],
          ['k = 2, second loser sold to CASH', '22.55%', '21.14%', '-32.06%', '0.707', '+0.092 on 30/30', '+0.007 on 19/30', '4.6', '0.683'],
          ['k = 2, second loser sold and TOPS UP the best holding', '22.27%', '20.95%', '-31.78%', '0.698', '+0.083 on 30/30', '+0.001 on 15/30', '4.6', '-'],
        ],
        highlightRows: [1],
      },
      {
        title: 'AXIS B - top up an existing winner instead of buying the refused entrant',
        caption: 'All eighteen cells lose risk-adjusted to doing nothing, and not one beats the staged rule on a single seed out of thirty. The last column is why: a top-up converts diversification into leverage on one name.',
        columns: ['Ranked by', 'Cap', 'k', 'CAGR', 'MaxDD', 'Calmar', 'dCalmar vs INCUMBENT', 'dCalmar vs OA-ROT-1', 'Largest weight one position reached'],
        rows: [
          ['rs252', 'none', '1', '21.36%', '-35.75%', '0.587', '-0.029 on 10/30', '-0.117 on 0/30', '52.5%'],
          ['rs252', '3x', '1', '21.09%', '-41.78%', '0.511', '-0.095 on 0/30', '-0.189 on 0/30', '32.9%'],
          ['rs252', '2x', '1', '20.65%', '-37.78%', '0.558', '-0.037 on 6/30', '-0.137 on 0/30', '26.1%'],
          ['unrealised', 'none', '1', '21.34%', '-38.01%', '0.561', '-0.051 on 2/30', '-0.147 on 0/30', '81.3%'],
          ['unrealised', '3x', '1', '21.20%', '-37.69%', '0.560', '-0.045 on 3/30', '-0.138 on 0/30', '36.3%'],
          ['unrealised', '2x', '1', '21.07%', '-35.72%', '0.587', '-0.023 on 9/30', '-0.106 on 0/30', '30.1%'],
          ['cushion', 'none', '1', '19.42%', '-42.01%', '0.463', '-0.154 on 0/30', '-0.253 on 0/30', '41.1%'],
          ['cushion', '3x', '1', '20.05%', '-41.33%', '0.486', '-0.126 on 0/30', '-0.213 on 0/30', '36.2%'],
          ['cushion', '2x', '1', '20.20%', '-41.25%', '0.489', '-0.124 on 0/30', '-0.210 on 0/30', '25.9%'],
          ['rs252', 'none', '2', '21.21%', '-36.09%', '0.585', '-0.043 on 9/30', '-0.122 on 0/30', '60.7%'],
          ['rs252', '3x', '2', '20.96%', '-40.06%', '0.527', '-0.079 on 2/30', '-0.169 on 0/30', '32.9%'],
          ['rs252', '2x', '2', '20.38%', '-37.36%', '0.549', '-0.059 on 2/30', '-0.151 on 0/30', '26.1%'],
          ['unrealised', 'none', '2', '21.59%', '-37.45%', '0.579', '-0.029 on 5/30', '-0.122 on 0/30', '79.2%'],
          ['unrealised', '3x', '2', '21.06%', '-36.31%', '0.570', '-0.035 on 4/30', '-0.128 on 0/30', '43.6%'],
          ['unrealised', '2x', '2', '20.73%', '-35.70%', '0.581', '-0.026 on 6/30', '-0.116 on 0/30', '32.4%'],
          ['cushion', 'none', '2', '19.41%', '-42.15%', '0.460', '-0.155 on 0/30', '-0.238 on 0/30', '41.4%'],
          ['cushion', '3x', '2', '19.16%', '-42.09%', '0.453', '-0.156 on 0/30', '-0.254 on 0/30', '36.2%'],
          ['cushion', '2x', '2', '20.41%', '-40.34%', '0.505', '-0.103 on 0/30', '-0.191 on 0/30', '25.5%'],
          ['PRE-REGISTERED FOLLOW-UP: split across the best TWO holdings, k = 1', 'none', '1', '21.55%', '-38.25%', '0.562', '-0.044 on 2/30', '-0.144 on 0/30', '53.1%'],
          ['PRE-REGISTERED FOLLOW-UP: fire on ANY evening, k = 1', 'none', '1', '21.00%', '-35.57%', '0.587', '-0.035 on 6/30', '-0.113 on 0/30', '53.6%'],
        ],
      },
      {
        title: 'AXIS C - the mixtures, and the control that decides them',
        caption: 'One cell clears the pre-registered bar. The last row is the control that settles what it is doing: restrict the identical rule to refused-signal evenings - the evenings Arun described - and it returns OA-ROT-1 to the digit with zero top-ups. Every point of its advantage comes from selling on evenings when no signal fired.',
        columns: ['Cell', 'CAGR', 'Worst seed', 'MaxDD', 'Calmar', 'dCalmar vs INCUMBENT', 'dCalmar vs OA-ROT-1', 'Sales/yr', 'Top-ups/yr', 'Calmar 25 / 40 / 60 bps'],
        rows: [
          ['INCUMBENT', '20.91%', '20.16%', '-35.16%', '0.606', '-', '-0.090 on 0/30', '0.0', '0.0', '0.606 / 0.600 / 0.567'],
          ['OA-ROT-1 (staged live)', '22.30%', '21.37%', '-31.78%', '0.702', '+0.090 on 30/30', '-', '4.5', '0.0', '0.702 / 0.684 / 0.653'],
          ['50/50 between the entrant and the top holding, k = 1', '21.55%', '20.22%', '-32.79%', '0.644', '+0.045 on 23/30', '-0.047 on 4/30', '4.6', '4.4', '-'],
          ['Entrant if a signal is refused, else top up by rs252, k = 1', '23.23%', '22.43%', '-31.27%', '0.736', '+0.102 on 30/30', '+0.035 on 21/30', '9.1', '7.2', '0.736 / 0.638 / 0.580'],
          ['Entrant if a signal is refused, else top up by CUSHION, k = 1', '24.70%', '24.03%', '-33.62%', '0.735', '+0.128 on 30/30', '+0.030 on 29/30', '8.7', '7.0', '0.735 / 0.702 / 0.643'],
          ['The same, split across the best two holdings', '23.04%', '22.80%', '-31.46%', '0.730', '+0.126 on 30/30', '+0.027 on 30/30', '8.9', '13.7', '0.730 / 0.688 / 0.636'],
          ['The same, k = 2', '22.56%', '21.82%', '-34.96%', '0.645', '+0.037 on 26/30', '-0.053 on 0/30', '9.0', '7.2', '0.645 / 0.605 / 0.546'],
          ['CONTROL - the cushion cell restricted to REFUSED-SIGNAL evenings only', '22.30%', '21.37%', '-31.78%', '0.702', '+0.089 on 30/30', '+0.000 on 2/30', '4.5', '0.0', '-'],
        ],
        highlightRows: [4, 7],
      },
      {
        title: 'WHERE THE AXIS-C GAIN ACTUALLY COMES FROM - built up one layer at a time',
        caption: 'Two thirds of the risk-adjusted gain over the incumbent is an unconditional -10% stop that needs no rotation machinery at all. research/166 said this in its caveat 4 and excluded the hard stop only because its CAGR sits a hair BELOW the incumbent. That is the pre-registered eligibility clause and it still applies - but the finding has now reproduced independently on a third seed set.',
        columns: ['Build it up', 'CAGR', 'MaxDD', 'Calmar', 'Machinery needed'],
        rows: [
          ['INCUMBENT', '20.91%', '-35.16%', '0.606', 'none'],
          ['plus an unconditional -10% hard stop, money to cash', '20.79%', '-30.46%', '0.683', 'a price check'],
          ['plus the same sale run through the rotation path', '21.08%', '-30.47%', '0.692', 'a price check'],
          ['plus OA-ROT-1 entrant swap on refused-signal evenings', '22.30%', '-31.78%', '0.702', 'the staged rule'],
          ['plus the money redeployed into the best-cushion holding', '24.70%', '-33.62%', '0.735', '8.7 sales and 7.0 top-ups a year, and 40.8% single-name weight'],
        ],
        highlightRows: [1],
      },
      {
        title: 'THE PLATEAU CLAUSE on the Axis-C leader - a hump on one axis, a step on the other',
        caption: 'The margin is the same narrow hump this family always has, and the tightening neighbour degrades to a wash against doing nothing. The cap axis is a step rather than a peak: anything looser than 3x works and anything tighter does not, which reads less like a tuned parameter and more like the effect requiring concentration to exist. That is a reason for suspicion, not comfort.',
        columns: ['Axis', 'Setting', 'Calmar', 'dCalmar vs INCUMBENT', 'Largest single weight'],
        rows: [
          ['Margin - how far under water before it sells', '7.5%', '0.700', '+0.092 on 30/30', '44.8%'],
          ['Margin', '10% (the cell)', '0.735', '+0.128 on 30/30', '40.8%'],
          ['Margin', '12.5%', '0.617', '+0.004 on 16/30 - a wash', '34.4%'],
          ['Position cap', '1.5x target', '0.677', '+0.058 on 28/30', '26.2%'],
          ['Position cap', '2x', '0.671', '+0.057 on 25/30', '32.4%'],
          ['Position cap', '3x (the cell)', '0.735', '+0.128 on 30/30', '40.8%'],
          ['Position cap', 'none', '0.720', '+0.105 on 30/30', '55.4%'],
        ],
        highlightRows: [1, 5],
      },
      {
        title: 'WINDOWS - paired CAGR against the INCUMBENT on the same seed',
        caption: 'W1 is the fit window, W2 the holdout, W3 the 2025-01 to 2026-09 stretch the live book is converting into (a REPORTING window, never used for selection). The staged rule earns the same amount in both halves of the history. The Axis-C leader earns almost all of it in the holdout - a regime-dependence flag that the one-sided 4pp rule does not catch.',
        columns: ['Cell', 'W1 2005-2015', 'W2 2016-2026', 'W3 2025-01 to 2026-09'],
        rows: [
          ['OA-ROT-1 (staged live)', '+1.26pp on 30/30', '+1.40pp on 26/30', '+0.10pp on 15/30'],
          ['k = 2', '+1.26pp on 30/30', '+1.63pp on 27/30', '+0.32pp on 15/30'],
          ['k = all eligible', '+1.26pp on 30/30', '+1.53pp on 27/30', '+1.19pp on 16/30'],
          ['Best pure top-up (rs252, no cap)', '+0.00pp on 12/30', '+0.92pp on 21/30', '-3.09pp on 9/30'],
          ['Entrant else top up by rs252', '+1.45pp on 30/30', '+3.15pp on 30/30', '-1.81pp on 12/30'],
          ['Entrant else top up by CUSHION', '+0.44pp on 30/30', '+7.03pp on 30/30', '+7.76pp on 26/30'],
          ['Unconditional -10% stop to cash (control)', '+0.34pp on 30/30', '-0.04pp on 15/30', '-2.04pp on 10/30'],
        ],
        highlightRows: [0, 5],
      },
      {
        title: 'CONTROLS AND NULLS',
        caption: 'research/166 sell-only result reproduces exactly on the new seeds: selling the loser and NOT replacing it is worse than never swapping at all, at every k. The cash must go back to work at the same open or the rule is not a rule. And a random swap at the same rate is 0.13 of Calmar worse than the ranked one, so WHICH holding leaves is not incidental either.',
        columns: ['Control', 'What it isolates', 'CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['Measure-only (k = 0)', 'the engine own no-op', '20.91%', '-35.16%', '0.606 - identical to the incumbent'],
          ['Sell-only, k = 1', 'sell the loser, do not redeploy', '21.80%', '-36.14%', '0.603'],
          ['Sell-only, k = 2', 'the same at k = 2', '21.46%', '-36.13%', '0.603'],
          ['Sell-only, k = 3', 'the same at k = 3', '21.38%', '-36.13%', '0.601'],
          ['Random swap at the matched rate, k = 2', 'is the CHOICE of holding doing anything?', '20.06%', '-34.62%', '0.571'],
          ['Random swap at the matched rate, k = 3', 'the same at k = 3', '20.42%', '-34.86%', '0.571'],
          ['Random swap at the elevated rate (9 a year)', 'the same at the Axis-C family own rate', '19.69%', '-35.40%', '0.558'],
          ['Unconditional -10% hard stop, no rotation at all', 'the cheap lever', '20.79%', '-30.46%', '0.683'],
        ],
        highlightRows: [1, 7],
      },
      {
        title: 'YEAR BY YEAR - house format, after tax, medians across 30 seeds',
        caption: 'Each cell is the year return with the intra-year maximum drawdown beneath it in brackets, measured from the running peak of the FULL curve. Benchmarks are excluded from the best-of picks. Two things the year rows say that the summary cannot: the stop-to-cash control takes LEAST DD in sixteen of the twenty-two years - it is insurance, priced like insurance - and 2025 is one of only three years in which doing NOTHING takes BEST OVERALL, which is precisely the stretch the live book is converting into.',
        columns: ['Year', 'INCUMBENT', 'OA-ROT-1 k=1', 'k=2', 'TOP-UP only (best of 18)', 'Stop + TOP-UP by cushion', 'Stop to CASH (control)', 'NIFTYBEES', 'BEST CAGR', 'LEAST DD', 'BEST OVERALL'],
        rows: [
          ['2005', '+14.1 (-12.3)', '+15.0 (-12.3)', '+15.0 (-12.3)', '+14.1 (-12.3)', '+17.4 (-12.6)', '+17.6 (-11.6)', '+32.8 (-14.0)', 'Stop to CASH', 'Stop to CASH', 'Stop to CASH'],
          ['2006', '+41.5 (-20.6)', '+32.8 (-21.0)', '+32.8 (-21.0)', '+32.9 (-20.7)', '+41.4 (-20.9)', '+41.9 (-19.5)', '+41.3 (-29.9)', 'Stop to CASH', 'Stop to CASH', 'Stop to CASH'],
          ['2007', '+84.7 (-10.8)', '+82.4 (-9.8)', '+82.4 (-9.8)', '+82.1 (-10.9)', '+69.0 (-12.2)', '+71.5 (-10.9)', '+53.0 (-14.9)', 'INCUMBENT', 'OA-ROT-1', 'INCUMBENT'],
          ['2008', '-29.9 (-32.6)', '-28.8 (-31.8)', '-28.8 (-31.8)', '-29.6 (-32.4)', '-30.1 (-33.5)', '-26.4 (-30.5)', '-52.1 (-59.7)', 'Stop to CASH', 'Stop to CASH', 'Stop to CASH'],
          ['2009', '+62.8 (-31.7)', '+66.0 (-30.9)', '+66.0 (-30.9)', '+64.7 (-31.6)', '+64.9 (-32.5)', '+64.0 (-29.6)', '+75.6 (-59.1)', 'OA-ROT-1', 'Stop to CASH', 'OA-ROT-1'],
          ['2010', '+15.5 (-15.7)', '+26.8 (-14.5)', '+26.8 (-14.5)', '+17.0 (-16.0)', '+13.0 (-15.8)', '+21.3 (-14.7)', '+18.6 (-25.0)', 'OA-ROT-1', 'OA-ROT-1', 'OA-ROT-1'],
          ['2011', '-10.5 (-18.6)', '-11.7 (-18.7)', '-11.7 (-18.7)', '-10.7 (-18.9)', '-10.0 (-18.4)', '-10.2 (-16.7)', '-24.1 (-27.3)', 'Cushion', 'Stop to CASH', 'Stop to CASH'],
          ['2012', '+28.2 (-19.5)', '+29.1 (-19.6)', '+29.1 (-19.6)', '+28.7 (-19.8)', '+36.4 (-17.9)', '+25.2 (-17.0)', '+26.5 (-26.0)', 'Cushion', 'Stop to CASH', 'Cushion'],
          ['2013', '+4.4 (-9.3)', '+4.4 (-9.3)', '+4.4 (-9.3)', '+3.9 (-9.4)', '-1.3 (-12.5)', '+2.0 (-9.2)', '+7.2 (-16.0)', 'INCUMBENT', 'Stop to CASH', 'INCUMBENT'],
          ['2014', '+49.7 (-7.3)', '+59.1 (-7.7)', '+59.1 (-7.7)', '+60.1 (-7.9)', '+64.2 (-8.9)', '+56.4 (-6.9)', '+31.6 (-6.2)', 'Cushion', 'Stop to CASH', 'Cushion'],
          ['2015', '-3.0 (-22.9)', '-3.0 (-22.7)', '-3.0 (-22.7)', '-5.6 (-23.5)', '-2.5 (-24.4)', '-6.6 (-22.4)', '-4.3 (-15.0)', 'Cushion', 'Stop to CASH', 'OA-ROT-1'],
          ['2016', '+7.2 (-28.9)', '+7.0 (-28.5)', '+7.0 (-28.5)', '+7.5 (-29.4)', '+13.6 (-33.0)', '+14.0 (-27.8)', '+4.0 (-21.6)', 'Stop to CASH', 'Stop to CASH', 'Stop to CASH'],
          ['2017', '+61.0 (-12.6)', '+64.7 (-12.4)', '+68.0 (-12.4)', '+76.5 (-13.1)', '+75.4 (-12.8)', '+50.3 (-9.2)', '+29.9 (-8.5)', 'TOP-UP only', 'Stop to CASH', 'TOP-UP only'],
          ['2018', '-27.9 (-34.9)', '-24.5 (-30.3)', '-23.8 (-31.2)', '-29.0 (-35.5)', '-25.6 (-33.6)', '-21.9 (-26.5)', '+4.8 (-14.1)', 'Stop to CASH', 'Stop to CASH', 'Stop to CASH'],
          ['2019', '+30.1 (-35.2)', '+27.2 (-30.6)', '+24.7 (-31.5)', '+27.1 (-35.8)', '+28.5 (-32.3)', '+26.9 (-27.2)', '+13.6 (-10.5)', 'INCUMBENT', 'Stop to CASH', 'Stop to CASH'],
          ['2020', '+48.7 (-22.4)', '+55.7 (-18.1)', '+55.2 (-20.3)', '+66.3 (-23.6)', '+42.8 (-23.9)', '+50.0 (-16.0)', '+15.4 (-36.3)', 'TOP-UP only', 'Stop to CASH', 'TOP-UP only'],
          ['2021', '+83.7 (-10.9)', '+66.5 (-14.2)', '+70.2 (-13.7)', '+80.1 (-12.8)', '+91.8 (-13.1)', '+73.8 (-9.9)', '+26.0 (-9.5)', 'Cushion', 'Stop to CASH', 'Cushion'],
          ['2022', '-3.9 (-27.4)', '-1.9 (-24.8)', '-2.3 (-25.4)', '-13.6 (-33.9)', '+13.3 (-22.5)', '-11.1 (-26.6)', '+5.5 (-16.1)', 'Cushion', 'Cushion', 'Cushion'],
          ['2023', '+51.9 (-18.1)', '+49.9 (-19.6)', '+54.4 (-19.3)', '+54.0 (-26.8)', '+43.6 (-14.5)', '+33.7 (-24.6)', '+21.0 (-9.7)', 'k=2', 'Cushion', 'k=2'],
          ['2024', '+5.3 (-24.3)', '+12.9 (-23.4)', '+6.1 (-24.6)', '+4.7 (-23.4)', '+28.8 (-22.5)', '+23.7 (-19.7)', '+10.4 (-10.5)', 'Cushion', 'Stop to CASH', 'Cushion'],
          ['2025', '+3.2 (-16.8)', '-2.2 (-21.2)', '-1.5 (-22.8)', '-4.9 (-23.2)', '-0.5 (-15.9)', '-0.3 (-16.8)', '+11.7 (-15.2)', 'INCUMBENT', 'Cushion', 'INCUMBENT'],
          ['2026', '+38.2 (-20.1)', '+41.2 (-23.0)', '+41.8 (-24.2)', '+42.4 (-28.3)', '+57.8 (-22.5)', '+34.6 (-20.5)', '-9.4 (-14.8)', 'Cushion', 'INCUMBENT', 'Cushion'],
          ['CAGR / MaxDD', '20.91% / -35.2%', '22.30% / -31.8%', '22.09% / -31.8%', '21.35% / -35.8%', '24.71% / -33.6%', '21.08% / -30.5%', '12.30% / -59.7%', '', '', ''],
        ],
        highlightRows: [20, 22],
      },
    ],

    results: {
      metrics: [
        { label: 'Evenings a year with TWO eligible losers', value: '0.5', hint: 'once the single-swap rule is running. Eleven occasions in 21.7 years - the rule removes the loser, so a second never accumulates', tone: 'neg' },
        { label: 'k = 3, 4, 6 and all-eligible', value: 'identical', hint: 'bit-identical to each other on all 30 seeds. A third swap in one evening happens on one path in thirty', tone: 'neg' },
        { label: 'k = 2 vs the staged rule', value: '-0.003 Calmar', hint: 'winning 11 of 30 paired seeds at 25 bps, and a real loss at 40 bps (0.671 vs 0.684)', tone: 'neg' },
        { label: 'Top-up cells beating the staged rule', value: '0 of 18', hint: 'on zero seeds out of thirty. All eighteen also lose Calmar to doing nothing', tone: 'neg' },
        { label: 'Largest single position under a top-up', value: '52% to 81% of NAV', hint: 'the book stops being a sixteen-name book. The ten best trades come to supply 55% of all profit', tone: 'neg' },
        { label: 'The Axis-C leader, restricted to the evenings Arun described', value: '0.702', hint: 'OA-ROT-1 to the digit, with zero top-ups. Every point of its edge comes from selling on evenings when no signal fired', tone: 'neg' },
        { label: 'A plain -10% stop, no machinery', value: 'Calmar 0.683', hint: 'against the incumbent 0.606 and the full Axis-C construction 0.735. Two thirds of the gain, none of the machinery', tone: 'pos' },
        { label: 'Live book change', value: 'NONE', hint: 'OA-ROT-1 stays exactly as research/165 staged it', tone: 'pos' },
      ],
      tables: [],
      charts: [
        { src: '/app/r171-multiswap-topup.png', caption: 'Growth of Rs 100 on a log scale, median of 30 seeds, after tax and 25 bps a side, with the drawdown panel beneath. The two top-up curves are the ones that fall furthest; the stop-to-cash control is the shallowest and the flattest.' },
      ],
    },

    winners: [
      {
        config: 'NOTHING IS ADOPTED. The live Base Age book keeps OA-ROT-1 exactly as staged.',
        summary:
          'Both of the questions asked return NO EDGE. One construction the study invented clears the pre-registered bar, and the control shows it is an unconditional -10% stop wearing a top-up costume - a lever research/166 already identified and which a plain stop delivers two thirds of with none of the machinery.',
        metrics: [
          { k: 'Swap two instead of one', v: 'NO EDGE. The case arises 0.5 times a year; k = 2 costs -0.21pp of CAGR and loses outright at 40 bps' },
          { k: 'Is there an optimum k', v: 'Yes - one. k = 3 and above are bit-identical to each other and unreachable in practice' },
          { k: 'Top up existing winners', v: 'NO EDGE. 0 of 18 beat the staged rule on any seed; all 18 lose Calmar to doing nothing' },
          { k: 'Effect on the 2025-26 window', v: 'Every top-up construction except one LOSES there, and 2025 is a year in which doing nothing was the best answer outright' },
          { k: 'What IS worth its own study', v: 'The unconditional -10% hard stop: Calmar 0.683 with no machinery, third independent appearance, failing only the CAGR-eligibility clause by a tenth of a point' },
          { k: 'Registered for the 26-Sep-2026 review', v: 'Expect ~4.5 swaps a year and one occasion every two years where a second holding would also have qualified. Do not read it as a missed opportunity' },
        ],
        rejected: [
          'k = 2 - a wash at 25 bps (-0.003 Calmar on 11 of 30 seeds) and a loss at 40 bps. More churn, no return.',
          'k = 3, 4, 6 and all-eligible - not rejected so much as unreachable: bit-identical to each other on every seed.',
          'Top up by 12-month relative strength - best of its family at Calmar 0.587, still below doing nothing and beaten by the staged rule on 30 of 30 seeds.',
          'Top up by unrealised return - one position reaches 81% of NAV with no cap.',
          'Top up by cushion above the trailing stop - the worst ranking on both sides of the trade, exactly as research/166 found on the sell side. Calmar 0.453 to 0.505.',
          'Split the top-up across the best two holdings - 0.562, worse than putting it all in one.',
          'The 50/50 hybrid between entrant and top-up - 0.644, below the staged rule on 26 of 30 seeds.',
          'Entrant-else-top-up by rs252 - clears the bar at 25 bps (0.736) and collapses to 0.638 at 40 bps, BELOW the staged rule.',
          'Entrant-else-top-up by cushion - clears every letter of the bar and is refused anyway: its top-up leg does nothing on the evenings in question, 94% of its edge is in the holdout half, and it takes one position to 40.8% of NAV.',
        ],
      },
    ],

    caveats: [
      'k WAS NEVER A FREE AXIS. The engine caps swaps at k per evening, but each swap also needs its own refused entrant, so k is bounded by the SMALLER of the eligible losers and the refused signals. That is the right model of the live rule - you cannot buy an entrant that did not signal - but it means all-eligible was never reachable and the sweep could not have found a large k even if one existed. The spill variants were built precisely to test the unbounded version, and they are the wash reported above.',
      'THE ELIGIBILITY HISTOGRAM COUNTS HOLDINGS WITH A FINITE SCORE AND AN ENTRY BEFORE TODAY, but does not require a tradeable opening price. It is a frequency statistic, not a fill test; the true count of ACTIONABLE pairs is slightly lower than reported, which strengthens the conclusion rather than weakening it.',
      'THE ONE CELL THAT CLEARS THE BAR IS A POST-HOC PICK AND IS LABELLED AS ONE EVERYWHERE. The any-evening trigger came out of Axis C, the cushion ranking was 1 of 3 tried after that, and the 3x cap was 1 of 4. Its plateau was then measured, and passed on the margin axis only marginally.',
      'NINETY-FOUR PERCENT OF THAT CELL RETURN EDGE SITS IN THE HOLDOUT WINDOW: +0.44pp in 2005-2015 against +7.03pp in 2016-2026. Both are positive on 30 of 30 seeds, so the letter of the two-window rule passes, but a sixteen-fold asymmetry between the two halves of the history is a regime-dependence flag and the pre-registered 4pp rule only catches the mirror image of it.',
      'THE 2025-26 WINDOW IS REPORTED TWO WAYS AND THEY DISAGREE IN TONE. research/165 ran research/170 engine from the LIVE book eleven positions over 2025-01-31 to 2026-09-11 and found OA-ROT-1 firing at 7.9 swaps a year and losing on 27 of 30 paths. This study W3 row is a slice of the full compounded curve and shows a 15-of-30 coin flip. Different experiments; the restart-from-live-state version is the one that matches what the live book will actually do this year, and it is the more pessimistic of the two.',
      'CORRELATION AND BLEND VALUE WERE NOT RE-TESTED, and that is a gap. No entry, exit or universe changed, so the correlation to True North and IPO Base is research/161 and research/154. But a top-up construction tilts the book toward fewer, larger, more-extended names, which is a real and unmeasured change to the blend - one more reason nothing here is adopted.',
      'CAPACITY. The binding fact is not the median position (0.426% of the held name own 20-day traded value for the incumbent, 0.558% for the Axis-C leader on a Rs 10 lakh book, ten times those on Rs 1 crore) but the 40.8% single-name weight. At Rs 1 crore that is Rs 40 lakh in one mid-cap, bought in a single next-open order.',
      'market_data.db IS NOT RETROACTIVELY SPLIT-ADJUSTED. The inherited defence applies - each series is truncated after any day-over-day fall worse than -35% - and it makes the near-all-time-high state slightly EASIER to satisfy for affected names. Direction stated, not hidden.',
      'SURVIVORSHIP PRESSURE IS UPWARD ON EVERY ARM. The random-swap nulls carry the identical bias and are the control for the ranking claims, not for the levels.',
      'FIFTY-ONE SELECTION CELLS against a budget of seventy, plus twelve controls and twenty-nine cost re-scorings, about 2,820 simulations. Discount accordingly; that is why plateaus, paired tests, three windows, nulls and a decomposition control are reported rather than a single winner.',
      'NOTHING HERE WAS SOAKED ON LIVE DATA. The event list is research/164 frozen 3,619 events and the panel is a snapshot of market_data.db dated 12-Sep-2026.',
    ],

    reports: [
      { label: 'research/171 - RESULTS.md (full write-up)', href: 'https://github.com/castroarun/Quantifyd/blob/main/research/171_baseage_multiswap_topup/results/RESULTS.md' },
      { label: 'research/171 - STATUS doc (pre-registration, live log, crash recovery)', href: 'https://github.com/castroarun/Quantifyd/blob/main/research/171_baseage_multiswap_topup/BASEAGE_MULTISWAP_TOPUP_DAILY_SWEEP_STATUS.md' },
      { label: 'research/171 - the paired tests and the eligibility histogram', href: 'https://github.com/castroarun/Quantifyd/blob/main/research/171_baseage_multiswap_topup/results/paired171.md' },
      { label: 'research/170 - the study that confirmed OA-ROT-1', href: '/app/backtest/qs-leeway-and-baseage-best-entrant-research170' },
      { label: 'research/166 - the rotation family this extends', href: '/app/backtest/baseage-rotation-and-drift-research166' },
    ],

    githubLinks: [
      { label: 'research/171 - scripts', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/171_baseage_multiswap_topup/scripts' },
      { label: 'research/171 - results', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/171_baseage_multiswap_topup/results' },
    ],
    projectPaths: [
      'research/171_baseage_multiswap_topup/BASEAGE_MULTISWAP_TOPUP_DAILY_SWEEP_STATUS.md',
      'research/171_baseage_multiswap_topup/results/RESULTS.md',
      'research/171_baseage_multiswap_topup/results/paired171.md',
      'research/171_baseage_multiswap_topup/results/elig171.md',
      'research/171_baseage_multiswap_topup/results/yoy171.md',
      'research/171_baseage_multiswap_topup/scripts/',
    ],
  },
"""


def main():
    src = TS.read_text(encoding='utf-8')
    if SLUG in src:
        print('already published: %s' % SLUG)
        return
    if src.count(MARK) != 1:
        print('marker not found exactly once')
        sys.exit(1)
    TS.write_text(src.replace(MARK, MARK + ENTRY), encoding='utf-8')
    print('inserted %s (%d chars)' % (SLUG, len(ENTRY)))


if __name__ == '__main__':
    main()
