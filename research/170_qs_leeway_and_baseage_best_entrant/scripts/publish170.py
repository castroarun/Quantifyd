# -*- coding: utf-8 -*-
"""research/170 -- insert the study entry at the top of frontend/src/data/backtests.ts.

Idempotent: refuses to run twice (the slug is checked first).
"""
import sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
TS = ROOT / 'frontend' / 'src' / 'data' / 'backtests.ts'
MARK = 'export const BACKTEST_STUDIES: BacktestStudy[] = [\n'
SLUG = 'qs-leeway-and-baseage-best-entrant-research170'

ENTRY = r"""  {
    slug: 'qs-leeway-and-baseage-best-entrant-research170',
    title: 'Two questions, two books: does a slipping holding deserve rank leeway (Quality Summit), and which refused breakout should take a freed slot (Base Age)? (research/170)',
    verdict:
      'PART A: CONCLUDED, NO ADOPTION — the premise was wrong and the axis is dead. PART B: SIGNAL — the post-hoc pick from research/166 REPLICATED on seeds it had never seen, and still sits exactly on the adoption bar. Nothing is deployed.\n\n## PART A — Arun asked for a rank leeway. The book already has a wider one.\n\n"A stock falling to 16th rank being taken out now — how about giving it a leeway, say within top 25." The deployed Quality Summit spec keeps a holding while its relative-strength rank sits inside ceil(buffer x N) = ceil(1.5 x 15) = 23. A name at rank 16 is NOT sold. It is sold on rank only at 24th or worse.\n\n- **Six of every seven sales are not rank sales.** This study instrumented the engine to split the monthly sale reason. 85.5% of closed trades are STATE sales — the name left the near-all-time-high band, the liquidity floor, or the point-in-time screen — and only 14.5% are RANK sales. A rank leeway of any width can only ever touch one sale in seven.\n- **Every width tested loses, paired on the same 12 rebalance-day offsets.** Rank 15 / 20 / 26 / 30 / 38 / 45 against the incumbent rank 23 gives dCAGR of -2.02 / -0.15 / -0.09 / -0.39 / -0.57 / -0.84 percentage points, winning 3, 4, 5, 4, 3 and 2 of 12 offsets. Arun own proposal — rank 26, his "top 25" — is the closest thing to a tie in the table: a dead wash at -0.09pp and -0.003 Calmar.\n- **The only real signal on the axis is at the NARROW end, and it is negative.** Removing the leeway entirely costs 2.02pp of CAGR on 9 of 12 offsets. The leeway that already exists earns its keep. Widening it beyond rank 23 earns nothing more.\n\n## PART A — the tax argument does not survive contact with the data\n\nThe mechanism Arun is reaching for is real in one direction and absent in the other. Widening the leeway from rank 15 to rank 45 cuts trades from 74.2 to 53.6 a year, turnover from 5.06x to 3.59x NAV, and lengthens the average hold from 66 to 93 days.\n\n- **But the share of trades held beyond 365 days rises only from 0.3% to 1.6%.** Ninety-eight of every hundred round trips are still short-term at the widest leeway tested, so the 12.5% long-term rate never arrives. The rupee tax bill on a Rs 1 crore book over 8.1 years moves from Rs 70.3 lakh to Rs 65.7 lakh, and most of that fall is a smaller book making smaller gains.\n- **The reason is structural.** A monthly RS book with 15 slots replaces about four names a month. A name must stay in the top 45 of a fast-moving ranking AND inside its own near-ATH band for twelve consecutive months to reach long-term treatment. The rank band cannot manufacture that, because the rank is not what is ending most of these holdings.\n- **You can buy the tax rate on this book, and it costs more than it saves.** At quarterly cadence the long-term share does reach 9.7% — and the after-tax CAGR falls to 18.75%.\n\n## PART A — the OTHER leeway, which is the one that would actually bite\n\nIf the rank is not what ends holdings, the near-all-time-high state is. So the study ran it: keep a holding whose rank is fine even after it has fallen out of the band.\n\n- Incumbent: 21.39% after tax, -36.90% drawdown, Calmar 0.58, average hold 81 days.\n- Keep fallen names: 19.88%, -42.35%, Calmar 0.47, average hold 209 days.\n- **Paired: -1.42pp of CAGR and -0.146 of Calmar, winning 2 of 12 offsets.** It delivers exactly the longer holding period Arun wants and pays 5.5 points of drawdown and 1.5 points of return for it. The near-all-time-high requirement is doing protective work; loosening the grip is the wrong direction whichever form the loosening takes.\n\n## PART A — the band k = 0.90 stands, for the second time\n\nk = 0.90 is the Calmar peak at every leeway width tested. k = 0.85 earns more (22.32% against 21.39%) and in the FIT window clears the bar outright at +2.32pp of CAGR and +0.156 of Calmar on 9 of 12 offsets — then reverses in the HOLDOUT at -1.09pp and -0.063 on 4 of 12. research/162 found the same reversal at a different construction and called it a fit to the 2020-21 leg. **This is the second independent replication of that reversal. The 0.90 band is settled.**\n\nNull control: random ranking at the same leeway returns 11.73% against 21.39%, so relative strength is worth +9.7pp — and the leeway helps the random book too (Calmar 0.28 to 0.36 as it widens), which means its small benefit is churn reduction, not selection.\n\n## PART B — the post-hoc entrant pick, confirmed on fresh seeds\n\nresearch/166 found one rotation rule that works on Open Alpha - Base Age: when a qualifying signal arrives and the book is full, sell the holding that is more than 10% under water and give its slot to the newcomer. A POST-HOC cell did better — give the freed slot to the refused entrant with the highest 252-day relative strength rather than the most liquid one — reaching Calmar 0.715 against the incumbent 0.613. It was one of three entrant priorities tried after the result was known, so research/166 refused to bank it.\n\nThis study named the rule and the three candidates BEFORE running anything, and ran them on seeds 1001-1030, which no cell in research/164 or research/166 has ever touched.\n\n- **It came back.** On the fresh seeds: 22.58% after-tax CAGR, -31.78% drawdown, Calmar 0.710, against the incumbent 20.95% / -34.05% / 0.611 — paired +0.105 Calmar on 30 of 30 seeds and +1.64pp CAGR on 30 of 30, beating a rate-matched random swap by +0.115 Calmar on 30 of 30, and winning BOTH pre-registered windows on 30 of 30.\n- **Pooled across both seed sets, 60 paths: +0.096 Calmar on 60/60 and +1.65pp of CAGR on 60/60.** The pre-registered bar is +0.10. Three independent evaluations have now landed at +0.094, +0.096 and +0.105.\n- **The honest summary is neither "it works" nor "it does not".** The effect is real and small, and the threshold happens to sit on top of it. Moving the bar after seeing the number is what the bar exists to prevent.\n\n## PART B — who leaves sets the return; who enters sets the drawdown\n\nThis is the cleanest new fact in the study. All three entrant priorities earn the same uplift — +1.65pp (relative strength), +1.77pp (traded value), +1.75pp (base age) pooled across 60 paths. The entire spread between them, 5.5 points of drawdown and 0.105 of Calmar, sits in WHICH refused breakout you buy.\n\n- highest 252-day relative strength: -31.78% drawdown, Calmar 0.710\n- largest 20-day traded value: -33.52%, Calmar 0.671\n- oldest base: -37.23%, Calmar 0.605 — this one actually LOSES to the incumbent on Calmar, on 11 of 30 seeds, despite earning more\n\nThat spread cuts both ways for the multiple-testing question. A 1-of-3 axis that moves Calmar by 0.105 is a real lever, and it is exactly the kind of axis a grid search exploits. What tips it toward real is the 60-of-60 consistency and the fact that the winning variant is the one that REDUCES drawdown, which is not what a return-chasing overfit looks like.\n\n## PART B — costs, tax and what it costs to hold\n\nThe advantage over the incumbent is +0.099 / +0.095 / +0.086 of Calmar at 25 / 40 / 60 bps a side, so it is not a cost artefact. Tax is modelled through the Indian financial-year netting engine, never haircut: the rule pays Rs 105.4 lakh against the incumbent Rs 84.1 lakh on a Rs 10 lakh book over 21.7 years, up 25%, and the +1.65pp is what is left after paying it.\n\nEverything else moves the wrong way, modestly: win rate 48.9% to 47.3%, worst losing streak 14 to 15, profit from the ten best realisations 37.5% to 43.7%, median position 0.43% to 0.50% of the held name own 20-day traded value.\n\nThe margin plateau is a hump, not a shelf: 7.5% under water gives +0.072 Calmar, 10% gives +0.105, 12.5% gives +0.026. All three beat the incumbent — but only the 10% margin wins BOTH windows. The 7.5% variant earns its edge in W1 and loses W2; the 12.5% variant does the reverse.\n\n## Recommendation\n\n**Quality Summit keeps the spec research/160 published: k = 0.90, 15 names, RS ranking, monthly, leeway to rank 23.** It is not a deployed book, so Part A changes no live risk either way.\n\n**The live Base Age book converting under research/165 goes live UNCHANGED.** The rule misses the pre-registered bar; the book has never traded live and its first months are the only clean read of the base spec we will ever get; and the mechanic needs an evening scorer plus a two-leg next-open order that does not exist. The exact rule text is written out as proposal OA-ROT-1 in RESULTS.md, with a dated review on 13-Mar-2027 and a pass criterion on the LIVE entry queue: it must show signals refused while a holding sits more than 10% under water, about four times a year. If the live queue does not show that shape, the rule is inapplicable regardless of the backtest.\n\n## Harness proofs, run before any selection cell\n\n- Part A engine is research/160 frozen qg_engine.py, copied and patched by 11 exact-string patches. Run at research/160 own 5.0% idle-cash assumption it returns 21.19% / -37.07% / Calmar 0.58 — the published F_Bb7 row to the second decimal.\n- Part B harness is research/166 sim166.py, byte-identical. On research/166 own seeds it reproduces every published row: incumbent 20.98 / -34.05 / 0.613, pre-registered rotation winner 22.68 / -33.52 / 0.673, post-hoc entrant cell 22.73 / -31.78 / 0.715, random null 20.39 / -35.75 / 0.569.',
    status: 'COMPLETE',
    date: '2026-09-13',
    cardBlurb:
      'Arun asked for a rank leeway on Quality Summit — the book already had a wider one, and only one sale in seven is a rank sale at all. He also asked whether a full Base Age book should hand a freed slot to the best qualifying signal: research/166 said yes on a pick made after seeing results, and on 30 seeds it had never seen, the pick came back.',
    cardStats: [
      { label: 'Part A — QS rank leeway', value: 'NO ADOPTION · 85.5% of sales are not rank sales' },
      { label: 'Part B — best entrant', value: 'REPLICATED · +0.105 Calmar on 30/30 fresh seeds' },
      { label: 'Pooled over 60 paths', value: '+0.096 vs a bar of +0.100' },
    ],

    systemRules: {
      intro:
        'Two books, two questions, one folder. Neither engine was modified in any way that changes a trading decision: Part A generates its engine from research/160 frozen copy by exact-string patches that only ADD instrumentation, and Part B copies research/166 harness byte-identical. Both reproduce their parent study before any selection cell runs.',
      sharedCoreTitle: 'The two books, held completely fixed',
      sharedCore: [
        { k: 'PART A — Quality Summit', v: 'Close at least 0.90x the name own CAUSAL all-time-high close; 20-day median traded value at least Rs 2 crore; the point-in-time Screener b7 screen (profitable 3 filed years, 3-year average ROE above 15, ROCE above 15 or a lender, 3-year sales AND profit growth above 10, market cap above Rs 1,000 crore, NO debt test); top 15 by relative strength; rebalanced monthly; next-open fills; NO exit rule.' },
        { k: 'PART A — the leeway itself', v: 'A holding is kept while its rank sits inside ceil(buffer x N). At the deployed buffer 1.5 and N 15 that is rank 23. It is ALSO sold, at any rank, if it stops qualifying — below the band, below the liquidity floor, or off the screen. Both exits were instrumented separately for the first time in this study.' },
        { k: 'PART B — Open Alpha, Base Age', v: 'First close above the prior all-time-high close, where that prior high is at least 60 bars old and the stock fell at least 20% below it in between. SuperTrend(14,4) close trail. No hard stop, no target, no time stop. 16 slots at 6.25% of NAV, Rs 10 lakh book, contested slots broken at random.' },
        { k: 'PART B — the rule under test', v: 'On a day with at least one qualifying signal the book cannot take (no free slot OR not enough cash), rank the holdings by unrealised return. If the weakest is more than 10% under water, sell it at the next open and buy, at the SAME open, the refused entrant ranked highest by X. At most one swap a day. A position bought today cannot be swapped out today. X was pre-registered as one of rs252, tv20_cr, x_bars — three cells, and that is the whole selection question.' },
        { k: 'Costs and tax, both parts', v: '25 bps a side (40 and 60 on the finalists). 20% short-term / 12.5% long-term capital gains above 365 days, with Indian financial-year loss netting, settled 1 April, modelled through the engine and never haircut.' },
        { k: 'Idle cash, both parts', v: '5.2% post-tax, credited daily on the cash balance — Arun standard as of 12-Sep-2026.' },
        { k: 'Paths', v: 'Part A: 12 rebalance-day offsets per cell, medians with the range and the worst offset. Part B: 30 seeds per cell on EACH of two independent seed sets — 1001-1030 (fresh, the primary evidence) and 1-30 (research/166 own).' },
        { k: 'Windows', v: 'Part A: 2018-08-01 to 2026-09-10, fit W1 to 2022-06-30, holdout W2 from 2022-07-01. Part B: 2005-01-03 to 2026-09-11, W1 to 2015-12-31, W2 from 2016-01-01. Both pre-registered, with a 4-percentage-point W1-to-W2 deterioration rule.' },
      ],
      riskLayer: {
        title: 'The adoption bars, written into the STATUS doc before the first cell ran',
        caption: 'Part A bar is research/162. Part B bar is research/166, applied unchanged so that the confirmation is judged by the same rule that refused the original.',
        columns: ['Part', 'Criterion', 'Best result achieved', 'Outcome'],
        rows: [
          ['A', 'Paired at least +0.15 Calmar OR +2pp CAGR at no worse drawdown vs the incumbent leeway', 'Every leeway width is NEGATIVE on both. Arun rank-26 proposal: -0.09pp and -0.003', 'FAIL'],
          ['A', 'Wins on at least 8 of 12 rebalance-day offsets', 'Best leeway width wins 5 of 12', 'FAIL'],
          ['A', 'Holds in BOTH windows, W2 not more than 4pp below W1', 'The two cells that clear in the fit window (k 0.85, and N 10 quarterly) reverse in the holdout; N 10 quarterly fails the 4pp rule at -8.92pp', 'FAIL'],
          ['A', 'Beats a random-ranking null at the same leeway', 'RS 21.39% vs random 11.73% — but the leeway helps the RANDOM book too, so its benefit is churn, not selection', 'PASS (and uninformative)'],
          ['B', 'Paired at least +0.10 Calmar OR +2pp CAGR at no worse drawdown vs the incumbent', 'Fresh seeds +0.105 (CLEARS). Pooled over 60 paths +0.096 (MISSES by 0.004)', 'AMBIGUOUS'],
          ['B', 'Wins on at least 20 of 30 seeds', '30/30 on Calmar and 30/30 on CAGR, on BOTH seed sets — 60/60 pooled', 'PASS'],
          ['B', 'Holds in BOTH windows', 'W1 +1.26pp on 30/30, W2 +2.23pp on 30/30 (fresh seeds)', 'PASS'],
          ['B', 'Sits on a plateau', 'Margin neighbours at 7.5% and 12.5% give +0.072 and +0.026 Calmar — all positive, a hump not a shelf, and only the 10% margin wins both windows', 'PASS (marginal)'],
          ['B', 'Survives 40 and 60 bps', 'Calmar 0.710 / 0.684 / 0.647 against the incumbent 0.611 / 0.589 / 0.561', 'PASS'],
          ['B', 'Beats a rate-matched random-swap null', '+0.115 Calmar on 30/30 fresh seeds; the null itself is -0.037 against the incumbent', 'PASS'],
        ],
        highlightRows: [4],
      },
    },

    system: {
      intro:
        'Two economic questions. Part A: a monthly-rebalanced momentum book sells names that slip in the ranking — how much grace should a slipping holding get, and does the grace pay for itself in tax? Part B: a slot-constrained breakout book refuses roughly four of every five qualifying signals — when it frees a slot, which refused signal should get it?',
      rows: [
        { k: 'Part A, what varies', v: 'The leeway (7 widths, keeping to rank 15 / 20 / 23 / 26 / 30 / 38 / 45 at N 15), the book size (10 / 15 / 20), the cadence (monthly / quarterly) and the near-all-time-high band k (0.75 / 0.80 / 0.85 / 0.90 / 0.95). Nothing else.' },
        { k: 'Part A, new instrumentation', v: 'The monthly sale reason is split into reb_rank (the name still qualifies, its rank fell outside the leeway — the ONLY sale a leeway can prevent) and reb_state (it no longer qualifies at all). Rupee tax paid, average holding period and the share of trades held beyond 365 days are recorded per path.' },
        { k: 'Part A, the second leeway', v: 'A control that keeps a holding whose rank is fine even after it has left the near-all-time-high band. This is the leeway that would actually bite, given that 85.5% of sales are state sales.' },
        { k: 'Part B, what varies', v: 'Which refused entrant takes the freed slot — highest 252-day relative strength, largest 20-day traded value, or oldest base — and the under-water margin that fires the swap (7.5 / 10 / 12.5%). Five selection cells in total.' },
        { k: 'Part B, why fresh seeds', v: 'The cell under test was chosen in research/166 AFTER seeing results. A seed set the cell has never been run on removes the seed-selection component of that worry. It cannot remove the fact that the rule was chosen from a family of twenty across three studies of the same book — which is why the random-swap null and the 60-path consistency carry the argument, not the point estimate.' },
        { k: 'Nulls and controls', v: 'Part A: random ranking at three leeway widths, and the retain-loose probe. Part B: the incumbent that never swaps, and a random holding swapped out at the same 4 swaps a year.' },
      ],
    },

    conditions: {
      intro:
        'Everything ran on the VPS. Part A reads research/160 frozen panel and its b7 point-in-time Screener mask; Part B reads research/164 panel and research/166 frozen event list. No live state, no services module and no database was written by this study.',
      rows: [
        { k: 'Part A universe', v: 'Every NSE daily series in market_data.db with funds and ETFs excluded by the panel is_fund flag, 2,158 symbols, screened by the point-in-time Screener panel in fundamentals.db (features_pit_monthly, 4-month filing lag).' },
        { k: 'Part A window limit', v: 'The window cannot be extended. Screener serves about 12 fiscal years, so four filed years do not exist for most names until FY2018 is filed in August 2018 — coverage steps from 7% to 87% at that date. 8.1 years is all there is.' },
        { k: 'Part B universe', v: '1,698 NSE daily series on a 5,378-day calendar, dead names included, no index-membership filter; 3,619 qualifying entries over 1,880 distinct signal days.' },
        { k: 'Part A cells', v: '54 selection cells (42 leeway x book size x cadence, 12 band x leeway) against a pre-registered budget of 60, plus 2 harness proofs, 6 controls and 42 validation re-scorings. About 3,700 simulations.' },
        { k: 'Part B cells', v: '5 selection cells against a budget of 20, plus 2 controls and 8 cost re-scorings, each on 30 seeds on each of two seed sets. About 660 simulations.' },
        { k: 'Ranking metric', v: 'After-tax Calmar, paired on the same offset or the same seed, among cells whose after-tax CAGR is at least the incumbent. Percentage invested is reported in every table so that a Calmar earned by sitting in cash is visible.' },
        { k: 'Drawdown convention', v: 'Every per-year drawdown is measured from the running peak of the FULL curve, never from the year first bar — the research/154 correction.' },
      ],
    },

    comparisons: [
      {
        title: 'PART A — the rank leeway, at the incumbent construction (N 15, monthly)',
        caption: 'After tax, 12-offset medians. The leeway curve is flat with a shallow top at the incumbent rank 23. Note the two columns on the right: widening the leeway lengthens the hold exactly as expected and STILL cannot reach the 365-day line where the tax rate changes.',
        columns: ['Keep to rank', 'buffer', 'CAGR pre-tax', 'CAGR after tax', 'MaxDD', 'Calmar', 'Trades/yr', 'Turnover xNAV', 'Avg hold (days)', '% trades > 365d', 'Tax paid on Rs 1 cr'],
        rows: [
          ['15 — no leeway', '1.00', '22.82%', '19.80%', '-37.56%', '0.55', '74.2', '5.06', '66', '0.3%', 'Rs 64.1 L'],
          ['20', '1.33', '24.48%', '21.19%', '-37.02%', '0.58', '64.6', '4.42', '78', '0.6%', 'Rs 70.4 L'],
          ['23 — THE INCUMBENT', '1.50', '24.63%', '21.39%', '-36.90%', '0.58', '61.6', '4.21', '81', '0.7%', 'Rs 70.3 L'],
          ['26 — Arun "top 25"', '1.67', '23.95%', '20.76%', '-36.58%', '0.57', '59.8', '4.06', '83', '0.8%', 'Rs 68.3 L'],
          ['30', '2.00', '24.34%', '21.39%', '-38.63%', '0.51', '57.4', '3.93', '87', '1.0%', 'Rs 66.7 L'],
          ['38', '2.50', '24.32%', '21.40%', '-40.30%', '0.53', '54.9', '3.74', '91', '1.2%', 'Rs 65.7 L'],
          ['45', '3.00', '23.96%', '20.70%', '-37.91%', '0.57', '53.6', '3.59', '93', '1.6%', 'Rs 66.0 L'],
        ],
        highlightRows: [2],
      },
      {
        title: 'PART A — paired against the incumbent, same 12 offsets, full window',
        caption: 'The pre-registered bar was at least +0.15 Calmar or +2pp CAGR at no worse drawdown, on at least 8 of 12. Nothing comes close, and Arun own proposal is a dead wash.',
        columns: ['Candidate', 'dCAGR', 'CAGR wins', 'dCalmar', 'Calmar wins', 'dMaxDD', 'DD wins'],
        rows: [
          ['Keep to rank 15 — no leeway at all', '-2.02pp', '3/12', '-0.047', '2/12', '-1.09', '4/12'],
          ['Keep to rank 20', '-0.15pp', '4/12', '-0.011', '4/12', '-0.27', '3/12'],
          ['Keep to rank 26 — Arun "top 25"', '-0.09pp', '5/12', '-0.003', '5/12', '-0.62', '5/12'],
          ['Keep to rank 30', '-0.39pp', '4/12', '-0.024', '4/12', '-1.16', '3/12'],
          ['Keep to rank 38', '-0.57pp', '3/12', '-0.063', '2/12', '-2.11', '1/12'],
          ['Keep to rank 45', '-0.84pp', '2/12', '-0.026', '1/12', '-1.18', '3/12'],
          ['Band k 0.85 (not a leeway change)', '+0.60pp', '9/12', '+0.009', '7/12', '-0.97', '3/12'],
          ['N 10 + quarterly (not a leeway change)', '+2.97pp', '9/12', '-0.006', '6/12', '-4.46', '0/12'],
          ['KEEP FALLEN NAMES (the other leeway)', '-1.42pp', '4/12', '-0.146', '2/12', '-5.51', '1/12'],
        ],
        highlightRows: [2, 8],
      },
      {
        title: 'PART A — why the fit window is not enough: the two cells that DO beat the incumbent',
        caption: 'Both clear the bar in the fit window and both reverse in the holdout. research/162 found the same reversal from a different direction; this is the second independent replication.',
        columns: ['Cell', 'W1 fit dCAGR', 'wins', 'W1 dCalmar', 'wins', 'W2 holdout dCAGR', 'wins', 'W2 dCalmar', 'wins'],
        rows: [
          ['Band k 0.85', '+2.32pp', '8/12', '+0.156', '9/12', '-1.09pp', '5/12', '-0.063', '4/12'],
          ['Band k 0.85 + leeway rank 26', '+2.91pp', '9/12', '+0.133', '9/12', '-0.34pp', '5/12', '-0.030', '2/12'],
          ['N 10, quarterly', '+6.73pp', '12/12', '+0.135', '11/12', '-1.73pp', '5/12', '-0.091', '4/12'],
          ['THE INCUMBENT (for reference)', '20.29% W1', '—', '0.61 W1', '—', '21.44% W2', '—', '0.59 W2', '—'],
        ],
        highlightRows: [3],
      },
      {
        title: 'PART A — the band k, at three leeway widths',
        caption: 'After-tax CAGR / MaxDD / Calmar. k = 0.90 is the Calmar peak at every leeway. As k loosens, names stop failing the STATE test and start failing the RANK test: the share of rank sales runs 4.8% at k 0.95, 14.5% at k 0.90 and 50.7% at k 0.75. A leeway only has work to do once the band is loose — and a loose band is worse.',
        columns: ['Band k', 'Leeway rank 23 (incumbent)', 'Leeway rank 26', 'Leeway rank 38'],
        rows: [
          ['0.75', '22.35 / -41.72 / 0.53', '21.91 / -41.85 / 0.53', '19.03 / -42.58 / 0.45'],
          ['0.80', '20.87 / -40.15 / 0.52', '21.03 / -39.88 / 0.53', '18.25 / -39.53 / 0.46'],
          ['0.85', '22.32 / -37.62 / 0.57', '22.35 / -37.23 / 0.57', '21.03 / -37.52 / 0.53'],
          ['0.90 — THE INCUMBENT', '21.39 / -36.90 / 0.58', '20.76 / -36.58 / 0.57', '21.40 / -40.30 / 0.53'],
          ['0.95', '17.16 / -37.98 / 0.46', '17.15 / -37.96 / 0.46', '16.80 / -38.07 / 0.45'],
        ],
        highlightRows: [3],
      },
      {
        title: 'PART B — the confirmation, on seeds the cell had never been run on',
        caption: 'Open Alpha - Base Age, after tax, 25 bps a side, idle cash 5.2% post-tax, 2005-01-03 to 2026-09-11. Deltas are medians of the per-seed difference on the SAME seed.',
        columns: ['', 'Incumbent (never swap)', 'Entrant = rs252 (the pick under test)', 'Entrant = tv20 (r/166 pre-registered)', 'Entrant = base age', 'Rate-matched RANDOM swap'],
        rows: [
          ['CAGR after tax, FRESH seeds', '20.95%', '22.58%', '22.77%', '22.50%', '20.80%'],
          ['Worst of 30 fresh seeds', '19.42%', '21.78%', '21.45%', '20.52%', '18.79%'],
          ['MaxDD, FRESH seeds', '-34.05%', '-31.78%', '-33.52%', '-37.23%', '-34.42%'],
          ['Calmar, FRESH seeds', '0.611', '0.710', '0.671', '0.605', '0.603'],
          ['Paired dCalmar vs incumbent, FRESH', '—', '+0.105 on 30/30', '+0.058 on 28/30', '-0.018 on 11/30', '+0.001 on 15/30'],
          ['Paired dCAGR vs incumbent, FRESH', '—', '+1.64pp on 30/30', '+1.86pp on 30/30', '+1.60pp on 28/30', '+0.00pp on 15/30'],
          ['Paired dCalmar vs the NULL, FRESH', '—', '+0.115 on 30/30', '+0.058 on 27/30', '-0.015 on 14/30', '—'],
          ['Paired dCalmar vs incumbent, r/166 seeds', '—', '+0.094 on 30/30', '+0.049 on 27/30', '-0.004 on 15/30', '-0.052 on 3/30'],
          ['POOLED over all 60 paths — dCalmar', '—', '+0.096 on 60/60', '+0.051 on 55/60', '-0.013 on 26/60', '-0.037 on 18/60'],
          ['POOLED over all 60 paths — dCAGR', '—', '+1.65pp on 60/60', '+1.77pp on 60/60', '+1.75pp on 58/60', '-0.34pp on 22/60'],
          ['W1 2005-2015 paired dCAGR, fresh', '—', '+1.26pp on 30/30', '+1.49pp on 30/30', '+1.18pp on 30/30', '-0.37pp on 11/30'],
          ['W2 2016-2026 paired dCAGR, fresh', '—', '+2.23pp on 30/30', '+2.30pp on 26/30', '+2.04pp on 22/30', '+0.51pp on 18/30'],
          ['Swaps per year', '0.0', '4.5', '4.3', '4.4', '3.9'],
          ['Tax paid over the window, Rs 10 L book', 'Rs 84.1 L', 'Rs 105.4 L', 'Rs 111.5 L', 'Rs 108.2 L', 'Rs 86.8 L'],
        ],
        highlightRows: [8, 9],
      },
      {
        title: 'PART B — who leaves sets the return; who enters sets the drawdown',
        caption: 'All three entrant priorities earn the same uplift. The entire spread between them is in the drawdown. The oldest-base entrant LOSES to the incumbent on Calmar despite earning more — the same leverage-not-selection failure research/166 found on the sell side.',
        columns: ['Entrant priority', 'dCAGR vs incumbent (pooled 60 paths)', 'MaxDD (fresh seeds)', 'Calmar (fresh seeds)', 'Calmar seeds won vs incumbent'],
        rows: [
          ['Highest 252-day relative strength (rs252)', '+1.65pp', '-31.78%', '0.710', '30/30'],
          ['Largest 20-day traded value (tv20_cr)', '+1.77pp', '-33.52%', '0.671', '28/30'],
          ['Oldest base (x_bars)', '+1.75pp', '-37.23%', '0.605', '11/30'],
          ['Never swap (the incumbent)', '—', '-34.05%', '0.611', '—'],
        ],
        highlightRows: [0],
      },
      {
        title: 'PART B — the margin plateau and the cost ladder, fresh seeds',
        caption: 'A hump, not a shelf. All three margins beat the incumbent, but only the 10% margin wins BOTH windows: 7.5% earns its edge in W1 and loses W2; 12.5% does the reverse. On cost, the advantage is +0.099 / +0.095 / +0.086 of Calmar across the ladder.',
        columns: ['Cell', 'Calmar 25 bps', 'Calmar 40 bps', 'Calmar 60 bps', 'dCalmar vs incumbent at 25 bps', 'W1 dCAGR', 'W2 dCAGR'],
        rows: [
          ['Swap margin 7.5% under water, rs252 entrant', '0.683', '—', '—', '+0.072 on 30/30', '+2.46pp on 30/30', '-0.75pp on 9/30'],
          ['Swap margin 10% under water, rs252 entrant', '0.710', '0.684', '0.647', '+0.105 on 30/30', '+1.26pp on 30/30', '+2.23pp on 30/30'],
          ['Swap margin 12.5% under water, rs252 entrant', '0.645', '—', '—', '+0.026 on 26/30', '-0.22pp on 0/30', '+1.72pp on 30/30'],
          ['Incumbent (never swap)', '0.611', '0.589', '0.561', '—', '—', '—'],
        ],
        highlightRows: [1],
      },
    ],

    results: {
      metrics: [
        { label: 'Part A — share of QS sales caused by RANK', value: '14.5%', hint: 'the other 85.5% are the name leaving the near-all-time-high band, the liquidity floor or the screen — no leeway can touch those', tone: 'neg' },
        { label: 'Part A — best leeway found', value: 'rank 23', hint: 'the incumbent. Every other width loses on paired evidence', tone: 'neg' },
        { label: 'Part A — trades held beyond 365 days', value: '0.3% to 1.6%', hint: 'across the whole leeway axis. The tax argument does not arrive', tone: 'neg' },
        { label: 'Part B — rs252 entrant, fresh seeds', value: '22.58% / -31.78%', hint: 'Calmar 0.710 against the incumbent 20.95% / -34.05% / 0.611', tone: 'pos' },
        { label: 'Part B — paired Calmar, pooled 60 paths', value: '+0.096 on 60/60', hint: 'against a pre-registered bar of +0.100. Clears on the fresh set at +0.105, misses on research/166 set at +0.094', tone: 'pos' },
        { label: 'Part B — tax paid, Rs 10 L book, 21.7 years', value: 'Rs 105.4 L vs Rs 84.1 L', hint: 'up 25%, modelled through the financial-year netting engine. The +1.65pp is what survives it', tone: 'neg' },
      ],
      tables: [
        {
          title: 'PART A — YoY, house format',
          caption: 'After tax, net of costs, medians across 12 rebalance-day offsets. Each cell is the year return with the intra-year max drawdown in brackets, measured from the running peak of the FULL curve. Benchmarks are excluded from the best-of picks. The incumbent takes BEST OVERALL in five of nine years — more than any other column.',
          columns: ['Year', 'No leeway (15)', 'INCUMBENT (23)', 'Leeway 26', 'Leeway 38', 'Band k 0.85', 'NIFTY 50', 'Midcap 150', 'BEST CAGR', 'LEAST DD', 'BEST OVERALL'],
          rows: [
            ['2018', '-12.2 (-18.7)', '-12.2 (-18.7)', '-12.2 (-18.7)', '-12.2 (-18.7)', '-13.2 (-21.4)', '-4.3 (-14.6)', '-5.1 (-19.4)', 'no leeway', 'no leeway', 'no leeway'],
            ['2019', '+21.8 (-15.9)', '+24.3 (-16.2)', '+24.4 (-16.2)', '+25.3 (-16.2)', '+25.5 (-18.0)', '+12.0 (-11.4)', '-0.3 (-21.0)', 'k 0.85', 'no leeway', 'leeway 38'],
            ['2020', '+32.3 (-33.3)', '+39.9 (-33.4)', '+37.8 (-33.6)', '+35.1 (-34.2)', '+29.6 (-32.8)', '+14.9 (-38.4)', '+24.4 (-40.8)', 'INCUMBENT', 'k 0.85', 'INCUMBENT'],
            ['2021', '+81.1 (-13.7)', '+86.9 (-13.9)', '+86.5 (-13.6)', '+87.6 (-14.0)', '+90.6 (-13.4)', '+24.1 (-10.1)', '+46.8 (-10.4)', 'k 0.85', 'k 0.85', 'k 0.85'],
            ['2022', '-18.4 (-33.3)', '-19.0 (-33.0)', '-20.7 (-33.3)', '-18.7 (-33.8)', '-7.5 (-26.9)', '+4.3 (-17.2)', '+3.0 (-21.6)', 'k 0.85', 'k 0.85', 'k 0.85'],
            ['2023', '+52.4 (-29.2)', '+58.9 (-27.9)', '+58.5 (-28.7)', '+65.2 (-27.8)', '+61.6 (-19.2)', '+20.0 (-9.9)', '+43.7 (-10.4)', 'leeway 38', 'k 0.85', 'k 0.85'],
            ['2024', '+31.3 (-13.9)', '+36.0 (-14.1)', '+34.7 (-14.0)', '+26.9 (-14.3)', '+26.9 (-14.1)', '+8.8 (-10.9)', '+23.8 (-11.0)', 'INCUMBENT', 'no leeway', 'INCUMBENT'],
            ['2025', '-17.5 (-36.1)', '-16.8 (-34.9)', '-17.2 (-34.6)', '-21.4 (-37.0)', '-17.3 (-37.0)', '+10.5 (-15.8)', '+5.4 (-21.1)', 'INCUMBENT', 'leeway 26', 'INCUMBENT'],
            ['2026', '+15.0 (-35.6)', '+15.7 (-34.4)', '+15.1 (-34.5)', '+12.9 (-38.5)', '+15.5 (-35.1)', '-10.2 (-15.2)', '+2.8 (-14.1)', 'INCUMBENT', 'INCUMBENT', 'INCUMBENT'],
            ['CAGR / MaxDD', '19.80 / -37.6', '21.39 / -36.9', '20.76 / -36.6', '21.40 / -40.3', '22.32 / -37.6', '9.38 / -38.4', '16.41 / -40.8', '', '', ''],
          ],
          highlightRows: [9],
        },
        {
          title: 'PART B — YoY, house format',
          caption: 'After tax, net of costs, medians across 30 FRESH seeds. Each cell is the year return with the intra-year max drawdown in brackets, from the running peak of the FULL curve. rs252 takes BEST OVERALL in ten of twenty-two years and LEAST DD in eight — concentrated in the bad years (2018, 2019, 2020), which is the signature of a drawdown mechanic rather than a return mechanic.',
          columns: ['Year', 'Incumbent', 'ENTRANT rs252', 'Entrant tv20', 'Entrant base age', 'Random-swap null', 'NIFTYBEES', 'BEST CAGR', 'LEAST DD', 'BEST OVERALL'],
          rows: [
            ['2005', '+14.1 (-12.3)', '+15.0 (-12.3)', '+12.8 (-12.7)', '+14.9 (-12.5)', '+14.1 (-12.4)', '+32.8 (-14.0)', 'rs252', 'rs252', 'rs252'],
            ['2006', '+41.5 (-20.6)', '+32.8 (-21.0)', '+37.2 (-20.6)', '+31.5 (-21.6)', '+40.6 (-20.6)', '+41.3 (-29.9)', 'incumbent', 'incumbent', 'incumbent'],
            ['2007', '+84.7 (-10.8)', '+82.4 (-9.8)', '+82.3 (-9.8)', '+82.6 (-9.8)', '+79.6 (-11.4)', '+53.0 (-14.9)', 'incumbent', 'base age', 'incumbent'],
            ['2008', '-29.9 (-32.6)', '-28.8 (-31.8)', '-28.8 (-31.8)', '-28.8 (-31.8)', '-29.8 (-32.4)', '-52.1 (-59.7)', 'base age', 'base age', 'base age'],
            ['2009', '+62.8 (-31.7)', '+66.0 (-30.9)', '+66.0 (-30.9)', '+66.0 (-30.9)', '+62.8 (-31.6)', '+75.6 (-59.1)', 'base age', 'base age', 'base age'],
            ['2010', '+15.5 (-15.7)', '+26.8 (-14.5)', '+26.8 (-14.5)', '+26.8 (-14.5)', '+16.5 (-15.7)', '+18.6 (-25.0)', 'tv20', 'tv20', 'tv20'],
            ['2011', '-10.5 (-18.6)', '-11.7 (-18.7)', '-11.7 (-18.7)', '-11.7 (-18.7)', '-10.6 (-18.5)', '-24.1 (-27.3)', 'incumbent', 'null', 'incumbent'],
            ['2012', '+28.2 (-19.5)', '+29.1 (-19.6)', '+33.1 (-19.6)', '+33.1 (-19.6)', '+28.2 (-19.4)', '+26.5 (-26.0)', 'tv20', 'null', 'tv20'],
            ['2013', '+4.4 (-9.3)', '+4.4 (-9.3)', '+4.7 (-9.3)', '+4.4 (-9.6)', '+4.5 (-9.3)', '+7.2 (-16.0)', 'tv20', 'rs252', 'tv20'],
            ['2014', '+49.7 (-7.3)', '+59.1 (-7.7)', '+58.7 (-8.0)', '+58.7 (-8.0)', '+50.2 (-7.8)', '+31.6 (-6.2)', 'rs252', 'incumbent', 'rs252'],
            ['2015', '-3.0 (-22.9)', '-3.0 (-22.7)', '-5.5 (-23.4)', '-5.4 (-23.1)', '-3.3 (-22.7)', '-4.3 (-15.0)', 'incumbent', 'rs252', 'rs252'],
            ['2016', '+7.2 (-28.9)', '+7.0 (-28.5)', '+7.2 (-29.1)', '+7.2 (-29.1)', '+6.7 (-28.7)', '+4.0 (-21.6)', 'incumbent', 'rs252', 'rs252'],
            ['2017', '+61.1 (-12.6)', '+64.7 (-12.4)', '+61.5 (-13.0)', '+64.2 (-13.0)', '+62.2 (-12.3)', '+29.9 (-8.5)', 'rs252', 'null', 'rs252'],
            ['2018', '-27.2 (-33.7)', '-24.5 (-30.3)', '-27.0 (-33.2)', '-32.1 (-37.0)', '-28.6 (-33.9)', '+4.8 (-14.1)', 'rs252', 'rs252', 'rs252'],
            ['2019', '+30.1 (-34.0)', '+27.2 (-30.6)', '+24.8 (-33.5)', '+26.2 (-37.2)', '+30.0 (-34.3)', '+13.6 (-10.5)', 'incumbent', 'rs252', 'rs252'],
            ['2020', '+48.5 (-21.1)', '+55.8 (-18.1)', '+56.0 (-22.9)', '+56.0 (-26.5)', '+52.3 (-22.3)', '+15.4 (-36.3)', 'base age', 'rs252', 'rs252'],
            ['2021', '+83.7 (-10.9)', '+76.5 (-12.4)', '+70.6 (-13.5)', '+71.4 (-12.5)', '+76.3 (-12.0)', '+26.0 (-9.5)', 'incumbent', 'incumbent', 'incumbent'],
            ['2022', '-4.8 (-27.8)', '-2.2 (-24.2)', '+6.2 (-23.6)', '+8.3 (-23.6)', '-3.1 (-24.8)', '+5.5 (-16.1)', 'base age', 'tv20', 'base age'],
            ['2023', '+51.9 (-19.2)', '+50.8 (-18.7)', '+44.4 (-14.9)', '+50.5 (-14.6)', '+51.7 (-19.2)', '+21.0 (-9.7)', 'incumbent', 'base age', 'base age'],
            ['2024', '+4.6 (-24.9)', '+18.9 (-22.5)', '+19.0 (-20.8)', '+22.3 (-22.3)', '+8.0 (-21.4)', '+10.4 (-10.5)', 'base age', 'tv20', 'base age'],
            ['2025', '+2.1 (-17.1)', '-1.2 (-19.5)', '-1.5 (-18.4)', '-0.2 (-19.9)', '+3.9 (-19.3)', '+11.7 (-15.2)', 'null', 'incumbent', 'incumbent'],
            ['2026', '+32.7 (-20.3)', '+41.8 (-21.6)', '+39.1 (-21.4)', '+34.9 (-21.4)', '+29.4 (-20.9)', '-9.4 (-14.8)', 'rs252', 'incumbent', 'rs252'],
            ['CAGR / MaxDD', '20.95 / -34.0', '22.58 / -31.8', '22.77 / -33.5', '22.50 / -37.2', '20.80 / -34.4', '12.30 / -59.7', '', '', ''],
          ],
          highlightRows: [22],
        },
        {
          title: 'PART B — tradeability gate, fresh seeds',
          caption: 'Everything moves the wrong way, modestly. On a Rs 1 crore book every capacity figure is ten times these — a median position of 5.0% of the name own daily traded value, executable but no longer trivial.',
          columns: ['', 'Incumbent', 'Entrant = rs252'],
          rows: [
            ['Win rate', '48.9%', '47.3%'],
            ['Max losing streak', '14', '15'],
            ['Trades per year', '31.8', '34.4'],
            ['Turnover x NAV', '2.49', '2.80'],
            ['Profit from the ten best realisations', '37.5%', '43.7%'],
            ['Compounding proxy divided by the same with the ten best deleted', '1.16 x 10^5', '2.16 x 10^5'],
            ['Median position, % of the held name own 20-day traded value', '0.43%', '0.50%'],
            ['Trades above 1% of the name tv20', '33.1%', '37.5%'],
          ],
        },
      ],
      charts: [
        { src: '/app/r170-part-a.png', caption: 'PART A — Quality Summit at four rank leeways plus the keep-fallen-names control, log growth of 100 with the drawdown panel beneath, against NIFTY 50 and Midcap 150. Every curve is the cross-sectional median across 12 rebalance-day offsets. The leeway curves sit on top of one another; the keep-fallen-names curve is the one that separates, downward.' },
        { src: '/app/r170-part-b.png', caption: 'PART B — Open Alpha Base Age: the incumbent, the two working entrant priorities and the rate-matched random-swap null, log growth of 100 with the drawdown panel beneath. Every curve is the cross-sectional median across the 30 FRESH seeds. The rs252 curve separates in the drawdown panel, not the price panel — that is the whole finding.' },
      ],
    },

    winners: [
      {
        config: 'PART A — nothing. Quality Summit keeps k 0.90, 15 names, RS ranking, monthly, leeway to rank 23.',
        summary:
          'The question was answered by instrumentation rather than by a sweep: only 14.5% of this book sales are rank sales, so a rank leeway can touch one sale in seven, and every width tested loses on paired evidence. The wider leeway does cut churn — 74 to 54 trades a year, 66 to 93 days held — and cannot cut tax, because the holding period never approaches the 365-day line.',
        metrics: [
          { k: 'Incumbent, after tax', v: '21.39% CAGR, -36.90% MaxDD, Calmar 0.58, 91.3% invested' },
          { k: 'Arun proposal (rank 26), paired', v: '-0.09pp CAGR and -0.003 Calmar, winning 5 of 12 offsets — a dead wash' },
          { k: 'Removing the leeway entirely', v: '-2.02pp CAGR on 9 of 12 offsets — the existing leeway earns its keep' },
          { k: 'Keeping fallen names instead', v: '-1.42pp CAGR and -0.146 Calmar, 5.5 extra points of drawdown' },
          { k: 'RS ranking vs a coin toss', v: '21.39% vs 11.73% — the ranking is still the entire engine' },
        ],
        rejected: [
          'Leeway to rank 15, 20, 26, 30, 38 and 45 — all negative against the incumbent rank 23 on paired CAGR and paired Calmar',
          'Band k 0.85 — clears the bar in the fit window (+0.156 Calmar on 9/12) and reverses in the holdout (-0.063 on 4/12). Second replication of research/162 finding',
          'N 10 quarterly — +6.73pp in the fit window on 12 of 12 offsets, -1.73pp in the holdout, and it fails the 4pp deterioration rule at -8.92pp',
          'retain = loose (keep a name after it leaves the near-all-time-high band) — the leeway that would actually bite, and it is the worst cell in the study',
        ],
      },
      {
        config: 'PART B — proposal OA-ROT-1, NOT adopted: swap out a holding more than 10% under water and give the slot to the highest-RS refused entrant.',
        summary:
          'research/166 found this cell after looking at results. It was named in advance here and re-run on 30 seeds it had never seen, and it came back: +0.105 paired Calmar on 30 of 30, beating a rate-matched random swap on 30 of 30, winning both pre-registered windows on 30 of 30, and holding its advantage at 40 and 60 bps. Pooled over both seed sets it delivers +0.096 against a bar of +0.100. The effect is real and small, and the threshold sits on top of it. The live Base Age book converting under research/165 goes live unchanged.',
        metrics: [
          { k: 'Fresh seeds (1001-1030)', v: '22.58% CAGR, -31.78% MaxDD, Calmar 0.710, worst seed 21.78%' },
          { k: 'Incumbent, same seeds', v: '20.95% CAGR, -34.05% MaxDD, Calmar 0.611, worst seed 19.42%' },
          { k: 'Paired vs incumbent, pooled 60 paths', v: '+0.096 Calmar on 60/60 and +1.65pp CAGR on 60/60' },
          { k: 'Paired vs a rate-matched random swap', v: '+0.115 Calmar on 30/30 fresh seeds' },
          { k: 'Cost ladder', v: 'Calmar 0.710 / 0.684 / 0.647 at 25 / 40 / 60 bps vs the incumbent 0.611 / 0.589 / 0.561' },
          { k: 'Cost of holding it', v: 'Tax up 25% (Rs 84.1 L to Rs 105.4 L on a Rs 10 L book), win rate 48.9% to 47.3%, worst streak 14 to 15' },
          { k: 'Dated review', v: '13-Mar-2027, after six months of live Base Age operation, on the LIVE entry queue' },
        ],
        rejected: [
          'Entrant = oldest base — earns the same +1.75pp of CAGR and LOSES to the incumbent on Calmar (-0.013 pooled, winning 26 of 60 paths). The entrant choice is not free.',
          'Swap margin 7.5% under water — earns +2.46pp in W1 and loses 0.75pp in W2 on 9 of 30 seeds',
          'Swap margin 12.5% under water — loses 0.22pp in W1 on 0 of 30 seeds and earns +1.72pp in W2. Only the 10% margin wins both windows',
          'Adopting any of it into the live book now — the bar was written down first and it is not met',
        ],
      },
    ],

    caveats: [
      'PART B IS A CONFIRMATION, NOT A DISCOVERY. The cell being confirmed was found post-hoc in research/166. A fresh seed set removes the seed-selection component of that worry; it cannot remove the fact that this rule was chosen from a family of twenty across three studies of the same book. The 60-of-60 consistency and the rate-matched random null are the strongest defences available, and they are not proof.',
      'THE ADOPTION BAR IS NOT RESOLVING THE QUESTION, AND THAT IS UNCOMFORTABLE. Three independent evaluations have landed at +0.094, +0.096 and +0.105 paired Calmar against a bar of +0.100. Anyone reading this as either "it works" or "it does not work" is reading more than the data supports.',
      'PART A WINDOW IS 8.1 YEARS AND CANNOT BE EXTENDED. Screener serves about twelve fiscal years, so four filed years do not exist for most names until FY2018 is filed in August 2018 (coverage steps from 7% to 87% at that date). The window contains the 2023-25 smallcap boom and one real bear leg. Nothing here shows how the book behaves across a full cycle, because the data does not exist.',
      'QUALITY SUMMIT IS NOT A DEPLOYED BOOK. research/160 and research/162 both concluded it is dilutive to the live True North + Open Alpha pair at every weight, and it runs 0.62 to 0.73 monthly correlation with Open Alpha. This study changes nothing about that; it answers a mechanical question about a book nobody is running.',
      'SURVIVORSHIP PRESSURE IS UPWARD ON EVERY ARM. market_data.db keeps only 102 stopped series in 2,158 (4.7%) across eleven years. The random-ranking and random-swap nulls carry the identical bias and are the controls that neutralise it for the ranking claims, not for the levels.',
      'market_data.db IS NOT RETROACTIVELY SPLIT-ADJUSTED. Both engines carry their inherited defences — research/160 restarts the all-time-high cummax on a one-day collapse below 0.55x, research/161 truncates a series after a -35% day. Both make the near-all-time-high state slightly EASIER to satisfy for affected names. Direction stated, not hidden.',
      'PART A TAX MODEL IS AN APPROXIMATION OF THE STATUTE — one netted financial-year pool with per-trade rates and loss carry-forward, not the full short-term/long-term set-off ordering.',
      'CORRELATION AND BLEND VALUE WERE NOT RE-TESTED in either part, because neither part changes an entry signal or an exit rule. Rotation does tilt the Base Age book toward younger positions, which is a real and unmeasured change to the blend, and one more reason not to adopt it before a live soak.',
      'NOTHING IN EITHER PART WAS SOAKED ON LIVE DATA. Part B event list is research/164 frozen 3,619 events; Part A runs on a frozen panel snapshot dated 12-Sep-2026.',
      '59 SELECTION CELLS IN TOTAL (54 in Part A against a budget of 60, 5 in Part B against a budget of 20). Discount accordingly; that is why plateaus, paired tests, two windows and nulls are reported rather than a single winner.',
    ],

    reports: [
      { label: 'research/170 — RESULTS.md (full write-up, both parts)', href: 'https://github.com/castroarun/Quantifyd/blob/main/research/170_qs_leeway_and_baseage_best_entrant/results/RESULTS.md' },
      { label: 'research/170 — STATUS doc (pre-registration, live log, crash recovery)', href: 'https://github.com/castroarun/Quantifyd/blob/main/research/170_qs_leeway_and_baseage_best_entrant/QS_LEEWAY_AND_BASEAGE_BEST_ENTRANT_DAILY_SWEEP_STATUS.md' },
      { label: 'research/166 — the study this confirms', href: '/app/backtest/baseage-rotation-and-drift-research166' },
      { label: 'research/162 — Quality Summit optimisation (the k 0.85 reversal, first sighting)', href: '/app/backtest/quality-summit-optimisation-research162' },
    ],

    githubLinks: [
      { label: 'research/170 — scripts', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/170_qs_leeway_and_baseage_best_entrant/scripts' },
      { label: 'research/170 — results', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/170_qs_leeway_and_baseage_best_entrant/results' },
    ],
    projectPaths: [
      'research/170_qs_leeway_and_baseage_best_entrant/QS_LEEWAY_AND_BASEAGE_BEST_ENTRANT_DAILY_SWEEP_STATUS.md',
      'research/170_qs_leeway_and_baseage_best_entrant/results/RESULTS.md',
      'research/170_qs_leeway_and_baseage_best_entrant/results/pairedA.md',
      'research/170_qs_leeway_and_baseage_best_entrant/results/pairedB.md',
      'research/170_qs_leeway_and_baseage_best_entrant/scripts/',
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
