#!/usr/bin/env python3
"""research/174 — append the BacktestStudy entry to frontend/src/data/backtests.ts.

Run on the VPS. Frontend-only change: safe at any hour, no backend restart.
Idempotent — re-running replaces the existing r/174 entry.
"""
import re
from pathlib import Path

TS = Path("/home/arun/quantifyd/frontend/src/data/backtests.ts")
SLUG = "longdated-short-premium-research174"

ENTRY = r"""  {
    slug: 'longdated-short-premium-research174',
    title: 'NIFTY long-dated short premium - are 2, 3, 6 and 12-month straddles, strangles and condors better than the live 45-DTE book? (research/174)',
    verdict:
      'NO EDGE beyond 45 DTE - CONCLUDED. Nothing deployed, nothing live touched. The live 45-DTE book is already sitting at the only tenor that works, the longer tenors decay monotonically to nothing, the 1-year contracts do not trade, and an eighth independent test says no stop of any kind beats no stop.\n\n## What Arun asked\n\n"Now that we are live with 45 DTE, can we test for more like 2 months away straddles / strangles / condors with different DTE entries and DTE exits and different stop losses, same for other liquid ones 3 months, 6 months away, 1 year away etc?" - and on stops: "sls can be combined premium, single side, underlying price move, vix, relative vix or combinations or more that i cudnt think about."\n\n## The answers, in one block\n\n- 2 months away (60 DTE): NO. t 1.36 against 45 DTE t 2.96, drawdown three times deeper, loses money across the first half of the sample.\n- 3 months (90 DTE): NO. Net NEGATIVE at -5.5 points a trade, win rate 48%.\n- 6 months (180 DTE): NO. Net -70.4 points a trade, and only 22 non-overlapping trades exist in twelve years.\n- 1 year (365 DTE): UNTRADEABLE. The at-the-money call trades 54 contracts a day; the two-year contracts trade four. Killed on the data, not on the P and L.\n- Different DTE exits: NOT RESOLVABLE. At 140 trades with a per-trade standard deviation of 225 points, exits at 9, 14, 21 and 27 DTE have overlapping intervals. The live 21-DTE rule is defensible and unrefuted, but it is NOT demonstrated, and should not be described as optimised.\n- Strangles: lower variance, lower margin, LOWER RETURN. Paired on the same entry days the 5% strangle loses 26 points a trade and wins on 39% of them.\n- Condors and winged strangles: KILLED at every tenor, worsening as tenor grows.\n- Stops - premium, single side, underlying move, VIX level, VIX rank, VIX relative to entry: ALL REFUTED, paired, at every tenor.\n\n## Tenor is a monotone decay, not a peak\n\nFixed rule with no per-tenor cherry-picking - short at-the-money straddle entered at tenor T, exited at 21 DTE, no stop, 0.75% slippage, liquidity floor of 25 traded contracts on both legs, 2015 to 2026. Net points a trade: 45d +56.6 (t 2.96), 60d +46.3 (t 1.36), 75d +30.5 (t 0.78), 90d -5.5, 105d -32.2, 180d -70.4, 210d -164.3. Win rate falls smoothly 73.6 / 66.7 / 59.0 / 48.0. Drawdown grows 1,243 / 3,671 / 4,795 points. A monotone dose-response is stronger evidence than a peak, and it points one way.\n\nSELLING MORE PREMIUM EARNS LESS. The 365-day straddle collects 1,906 points against the 45-day straddle 640 - three times the credit - nets nothing, and blocks 2.7 times the margin for twelve times as long. Across the full 624-cell tenor sweep exactly TWO cells clear t = 2, and both of them are the live book.\n\n## Why every management rule fails - the decomposition that explains eight studies\n\nChop the 45 to 21 DTE hold into five five-day pieces, each re-picking the at-the-money strike at its own start. Same calendar exposure, re-centred instead of held. Held as one position: +68.5 gross, +56.6 net, t 2.96. Sum of the five pieces: +45.2 gross, MINUS 12.8 net. Re-centring gives up a third of the gross edge and pays four extra round trips.\n\nThere is no theta window to go and collect. Every individual DTE window across the whole contract life is statistically indistinguishable from zero. The return comes from HOLDING ONE STRIKE THROUGH THE DRIFT. That single fact explains research/119 phase E (move-triggered re-centring), phase G (premium-triggered stop) and all seven stop families tested here.\n\n## Stops - the eighth refutation, and the new families die too\n\nEvery family run paired against no-stop on identical trades. At 45 DTE with 141 trades the no-stop baseline earns +51.3 a trade at t 2.56. Best parameter that actually fires on at least 10% of trades: underlying move 5% gives -37.3 at paired t -2.84; VIX level 25 gives -26.0 at t -2.68; VIX rank 95 gives -19.2; single side at 2.5x leg entry gives -19.1; combined premium 1.5x gives -5.3. The damage is MONOTONE in how often the stop fires - move stop at 86% fired -49.7, at 61% -68.5, at 41% -65.2, at 25% -58.6, at 10% -22.7.\n\nThe single-side idea was the most promising new one and it is refuted specifically. The hypothesis was that keeping the untouched leg preserves the theta that pays for the loss. At matched fire rates single-side and whole-position premium stops are indistinguishable. All the damage is in cutting the threatened leg; keeping the other buys nothing back.\n\nStops ARE real insurance at a real premium. VIX rank above 60 cuts the worst trade from -1,049 to -464 points; a 2% move stop cuts it to -264. If the goal is a smaller tail rather than more return, that is the honest menu - but research/119 already showed the cheaper way to buy the same thing is to trade fewer lots.\n\n## A data defect found and permanently fixed\n\nnse_options_bhav was missing EVERY NIFTY expiry beyond about 75 DTE for 2016-01 to 2024-02, plus 2026-04/05/06 and 2026-09. Cause: MAX_DTE = 75 in the research/89 stock-bhav downloader, which resumed by trade-date, so every session it touched was marked done and the uncapped production downloader skipped it forever. 2,102 sessions re-downloaded with zero errors, 7,567,118 rows staged, PLUS 2,836,719 NEW ROWS MERGED. Every year from 2015 to 2026 now has complete long-dated coverage. The download was staged to a separate file during market hours and merged only after 15:40 IST, so it never took a write lock on the database the live executors read.\n\n## One correction I made to myself mid-study\n\nAn interim table showed the 45-DTE 5% strangle beating the live straddle on t-stat, worst trade and return on margin, and I reported it as a candidate improvement. Paired trade by trade on the same 140 entry days it LOSES: median delta -26.4 points, wins on 55 of 140. Every apparent advantage came from lower variance and a smaller margin block, not from earning more. That claim is retracted. The strangle is a lower-return, lower-variance, lower-margin version of the SAME BET at correlation 0.87 - interesting as a capital-allocation question, not as an edge.',
    status: 'COMPLETE',
    date: '2026-09-15',
    cardBlurb:
      'Arun asked whether 2, 3, 6 and 12-month short premium beats the live 45-DTE NIFTY straddle. It does not, at any tenor, by any structure, with any stop. Return decays monotonically as tenor grows, the 1-year contracts trade 54 lots a day, and the reason every management rule ever tested has failed turns out to be visible in one decomposition.',
    cardStats: [
      { label: 'Verdict', value: 'NO EDGE beyond 45 DTE - CONCLUDED' },
      { label: 'Cells above t=2', value: '2 of 624 - and both are the live book' },
      { label: 'Data repaired', value: 'plus 2,836,719 long-dated option rows, 8 missing years' },
    ],
    systemRules: {
      intro:
        'The live 45-DTE NIFTY short straddle (research/119, real money since 11-Sep-2026, 3 lots) is the incumbent and the control in every table. This study never touches it. Everything here is a candidate to run ALONGSIDE or INSTEAD, and no cell cleared the bar.',
      sharedCoreTitle: 'The base trade, held fixed while tenor, structure and stop vary',
      sharedCore: [
        { k: 'Entry', v: 'Daily close of the session on or before expiry minus T calendar days. T swept 30 / 45 / 60 / 75 / 90 / 105 / 120 / 150 / 180 / 210 / 240 / 270 / 300 / 365.' },
        { k: 'Strike', v: 'Nearest listed strike to spot where BOTH legs are fillable on that session.' },
        { k: 'Liquidity rule (binding, research/89)', v: 'A leg is fillable only if that strike and side has close above zero AND at least 25 traded contracts that session. Long-dated strikes carry stale settlement prints; filling them manufactures edge.' },
        { k: 'Exit', v: 'Daily close on or before expiry minus X days, X swept per tenor (five to six values each, always including 21); or buy back at 35 / 50 / 65% of credit; or no target.' },
        { k: 'Resolution', v: 'DAILY CLOSE ONLY. Expired-contract intraday option data cannot be obtained from Kite, and the 1-minute recorder starts 20-Apr-2026 at about 27 DTE. No intraday claim is made anywhere in this study.' },
        { k: 'Costs', v: 'Slippage charged PER LEG on gross premium turnover (not on net credit - that is how four-leg structures get flattered), plus STT 0.10% of sell-side premium, exchange 0.05% of turnover, Rs 20 per order per leg per side, GST 18% on brokerage and exchange. Slippage swept 0.25 / 0.75 / 1.5 / 3.0%.' },
        { k: 'Margin', v: 'MEASURED from Kite basket_order_margins, NRML, consider_positions false, 15-Sep-2026, NIFTY spot 23,190, lot 65. Not modelled. Long wings submitted FIRST in every basket to avoid the leg-ordering trap.' },
        { k: 'Lot', v: 'NIFTY lot 65, so 1 point is Rs 65 per lot. The option_chain table says 75 and is wrong.' },
        { k: 'Window', v: '01-Jan-2015 to 15-Sep-2026, on the repaired nse_options_bhav (36,054,242 rows).' },
      ],
      riskLayer: {
        title: 'The adoption bar, pre-registered before any cell ran',
        caption: 'Nothing came close. The bar is reproduced here so the failure is measured against a number written down in advance, not one chosen afterwards.',
        columns: ['#', 'Criterion for a new tenor or structure', 'Best result achieved', 'Outcome'],
        rows: [
          ['1', 'Net positive at t at least 2.5 on at least 40 non-overlapping trades', 'No tenor other than 45 DTE reaches t 1.6; the long tenors have 9 to 26 trades', 'FAIL'],
          ['2', 'At least 1.25x the live 45-DTE margin-time return on the same window', 'Best non-45 tenor returns 11.8% a year on margin against 24.5%', 'FAIL'],
          ['3', 'Survives slippage at 1.5% of premium', '60 DTE falls to t 0.17; everything past 90 DTE is already negative', 'FAIL'],
          ['4', 'Drawdown no worse than 1.5x the 45-DTE book on the same window', '60 DTE -3,671 points against -1,243; 210 DTE -8,554', 'FAIL'],
          ['5', 'For a stop family: beats no-stop PAIRED, median delta above zero, wins on 60% of trades, paired t at least 2.0', 'Best firing parameter in any family at any tenor has a NEGATIVE paired delta', 'FAIL'],
        ],
        highlightRows: [0, 4],
      },
    },
    system: {
      intro:
        'Economic question: a short straddle is paid for accepting gamma and vega. Intuition says a longer tenor sells more premium per contract and decays more slowly, so it should be gentler to hold. This study asks whether that intuition survives real traded prices, real margin and real liquidity - and where in a contract life the money is actually made.',
      rows: [
        { k: 'Structures tested', v: 'Short straddle; short strangle at 1.5 / 2.5 / 3.5 / 5.0% out; iron condor at 2.5 and 5.0% body with 3 / 5 / 7% wings; winged straddle with 5 / 7 / 10% wings. Fourteen in all, each re-tested at six tenors.' },
        { k: 'Stop families tested', v: 'NONE (the incumbent, and the baseline in every cell); combined premium at 1.3 / 1.5 / 1.75 / 2.0 / 2.5x credit; underlying move at 2 / 3 / 4 / 5 / 7%; SINGLE SIDE at 1.5 / 2 / 2.5 / 3 / 4x the threatened leg entry price; single side on strike crossed by 0 / 1 / 2 / 3%; VIX level 16 / 18 / 20 / 22 / 25 / 30; VIX percentile rank 60 / 70 / 80 / 90 / 95; VIX relative to entry-day VIX at 1.15 / 1.25 / 1.4 / 1.6 / 2.0x.' },
        { k: 'Why the new families were expected to differ', v: 'Every previously refuted stop cuts the WHOLE position on a mark-to-market trigger, realising the loss and forfeiting the theta that pays for it. A single-side stop keeps the surviving leg theta; a VIX stop fires on the price of risk rather than on our own mark. Different mechanisms, genuinely untested - and they failed anyway.' },
        { k: 'Entry filter', v: 'India VIX percentile rank against the previous 252 sessions, causal. Carried as an AXIS (off / above 25 / above 50), not as an assumption, even though above 25 is the live rule.' },
        { k: 'Pairing', v: 'Within an arm the trade set is built ONCE and every stop family runs over the SAME trades, so every comparison is paired by construction. Unpaired medians lie at small n.' },
        { k: 'Per-window drawdown convention', v: 'Intra-year drawdown is measured from the running peak of the FULL curve, never from the year first bar - the convention error research/154 had to retract.' },
      ],
    },
    conditions: {
      intro:
        'Everything ran read-only on the VPS against nse_options_bhav, the real NSE daily bhavcopy, except the one-off backfill that repaired it.',
      rows: [
        { k: 'Price source', v: 'nse_options_bhav in backtest_data/market_data.db - open, high, low, close, settle, contracts and open interest per strike per expiry per session. NIFTY from 03-Jan-2011.' },
        { k: 'Liquidity probe', v: '2,887 sessions x 9 target tenors, asking per session which listed contract a trader could actually have sold. A first version conflated dead weeklies with real monthlies and was discarded - NSE lists weeklies five to nine weeks out that never print a single contract.' },
        { k: 'Expiry classification', v: 'Derived from data as expiry minus first session seen, never hardcoded. The monthly weekday moved Thursday to Tuesday, and legacy far-dated Thursday contracts coexist with real monthlies, so any last-expiry-of-the-month rule is unsafe.' },
        { k: 'Cells', v: 'P1 tenor 624 cells over 36,506 trades; P1b 19 decay windows; P2 structures 168 cells; P3 stops about 340 paired cells; P5 finalists 5 specs x 3 slippages x 3 filters x 3 windows.' },
        { k: 'Engine validation', v: 'Rebuilding the live 45-DTE arm independently returns n 92, average credit 782.4, t 3.12 against research/119 published n 89, 786.3, t 3.12.' },
        { k: 'Market-hours discipline', v: 'The backfill download ran during the session but wrote to a STAGING database; the merge into market_data.db was held until 15:40 IST so it never took a write lock on the file the live executors read.' },
      ],
    },
    comparisons: [
      {
        title: 'Tenor bake-off - fixed rule, enter at T and exit at 21 DTE',
        caption: 'No per-tenor cherry-picking of the exit. 0.75% slippage, no stop, 50% target, liquidity floor 25 contracts. Margin measured from Kite on 15-Sep-2026. The decay is monotone in every column at once.',
        columns: ['Entry tenor', 'n', 'Days held', 'Avg credit', 'Avg net / trade', 't', 'Pts per lot-year', 'Entry margin', 'Return on margin', 'Max DD', 'Win %'],
        rows: [
          ['45 DTE (LIVE)', '140', '24.6', '639.5', '+56.6', '2.96', '838', 'Rs 2.22L', '24.5%/yr', '-1,243', '73.6%'],
          ['60 DTE', '138', '41.2', '745.0', '+46.3', '1.36', '410', 'Rs 2.26L', '11.8%', '-3,671', '66.7%'],
          ['75 DTE', '139', '54.0', '812.3', '+30.5', '0.78', '206', 'Rs 2.33L', '5.7%', '-4,795', '59.0%'],
          ['90 DTE', '50', '67.3', '923.1', '-5.5', '-0.08', '-30', 'Rs 2.44L', '-0.8%', '-4,415', '48.0%'],
          ['105 DTE', '35', '81.3', '1,056.5', '-32.2', '-0.31', '-145', 'Rs 2.55L', '-3.7%', '-4,277', '54.3%'],
          ['120 DTE', '32', '93.0', '1,117.7', '+0.2', '0.00', '1', 'Rs 2.66L', '0.0%', '-2,914', '62.5%'],
          ['150 DTE', '25', '118.3', '1,317.6', '-11.4', '-0.08', '-35', 'Rs 2.88L', '-0.8%', '-3,024', '52.0%'],
          ['180 DTE', '22', '149.1', '1,478.8', '-70.4', '-0.35', '-172', 'Rs 3.09L', '-3.6%', '-4,763', '54.5%'],
          ['210 DTE', '23', '169.0', '1,582.1', '-164.3', '-0.86', '-355', 'Rs 3.59L', '-6.4%', '-8,554', '52.2%'],
          ['240 DTE', '19', '187.9', '1,698.5', '-158.0', '-0.59', '-307', 'Rs 4.41L', '-4.5%', '-7,760', '52.6%'],
          ['270 DTE', '14', '231.5', '1,925.5', '-173.8', '-0.47', '-274', 'Rs 5.23L', '-3.4%', '-6,743', '57.1%'],
          ['300 DTE', '11', '259.5', '1,875.7', '-47.2', '-0.16', '-66', 'Rs 5.73L', '-0.8%', '-3,707', '36.4%'],
          ['365 DTE', '9', '307.2', '1,906.1', '+88.0', '0.25', '105', 'Rs 5.94L', '1.1%', '-1,446', '55.6%'],
        ],
        highlightRows: [0],
      },
      {
        title: 'The liquidity gate - can you sell it at all?',
        caption: '2,887 sessions, 2015 to 2026, best listed contract near each tenor on each session. The 120 and 180 gaps are CALENDAR gaps (beyond 75 days NSE lists only the quarterly and semi-annual series), not liquidity failures. Everything from 365 DTE outward is killed on volume.',
        columns: ['Target tenor', 'Sessions with a fillable near-ATM pair', 'Nearest strike distance from spot', 'ATM call contracts/day', 'ATM call OI', 'Verdict'],
        rows: [
          ['45 days', '99%', '0.10%', '1,245', '87,600', 'tradeable'],
          ['60 days', '100%', '0.12%', '566', '59,750', 'tradeable'],
          ['90 days', '97%', '0.27%', '256', '47,888', 'tradeable'],
          ['120 days', '70%', '1.03%', '462', '210,700', 'tradeable when listed'],
          ['180 days', '73% (100% since 2022)', '1.21%', '311', '175,600', 'tradeable when listed'],
          ['270 days', '62%', '1.20%', '125', '115,962', 'marginal'],
          ['365 days', '59%', '1.17%', '54', '67,150', 'TOO THIN'],
          ['545 days', '48%', '1.62%', '9', '18,375', 'DEAD'],
          ['730 days', '23%', '2.12%', '4', '6,188', 'DEAD'],
        ],
        highlightRows: [6, 7, 8],
      },
      {
        title: 'Why management fails - chopping the 45 to 21 DTE hold into re-centred pieces',
        caption: 'Same calendar exposure, strike re-picked at each piece start instead of held. Attribution, not an identity - the pieces have different trade counts. Re-centring gives up a third of the gross edge AND pays four extra round trips, which together is the entire result.',
        columns: ['Position', 'Gross pts / trade', 'Net pts / trade', 't'],
        rows: [
          ['HELD AS ONE, 45 to 21 DTE', '+68.5', '+56.6', '2.96'],
          ['45 to 40, fresh ATM', '+14.2', '+1.8', '0.33'],
          ['40 to 35, fresh ATM', '+15.0', '+3.3', '0.52'],
          ['35 to 30, fresh ATM', '-8.7', '-20.2', '-2.20'],
          ['30 to 25, fresh ATM', '+14.7', '+3.3', '0.47'],
          ['25 to 21, fresh ATM', '+10.1', '-1.1', '-0.20'],
          ['SUM OF THE FIVE PIECES', '+45.2', '-12.8', '-'],
        ],
        highlightRows: [0, 6],
      },
      {
        title: 'Stops at 45 DTE - every family, paired against no-stop on identical trades',
        caption: 'Best parameter in each family that actually fires on at least 10% of trades. Settings loose enough not to lose are settings that fire on 1 to 3% of trades, where the paired win rate is 1.1% - it helped once, by luck. The same table at 60, 90, 120, 180 and 240 DTE reaches the same conclusion.',
        columns: ['Family', 'Arun words', 'Parameter', '% of trades fired', 'Paired mean delta', 'Paired t', 'Paired win %'],
        rows: [
          ['NONE', '-', '-', '0%', 'baseline +51.3 avg net, t 2.56', '-', '-'],
          ['MOVE', 'underlying price move', '5%', '22.7%', '-37.3', '-2.84', '6.4%'],
          ['VIXL', 'vix', '25', '12.1%', '-26.0', '-2.68', '2.1%'],
          ['VIXR', 'relative vix (percentile rank)', '95', '14.2%', '-19.2', '-1.87', '2.8%'],
          ['SIDE', 'single side', '2.5x leg entry', '19.9%', '-19.1', '-1.88', '7.1%'],
          ['SIDEK', 'single side on strike cross', '2%', '83.7%', '-20.0', '-0.85', '33.3%'],
          ['PREM', 'combined premium', '1.5x credit', '12.1%', '-5.3', '-0.74', '3.5%'],
          ['VIXD', 'vix vs the vix we sold into', '1.25x', '12.1%', '-11.6', '-1.40', '2.1%'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Structures at 45 DTE - condors and wings are dead on the index',
        caption: 'Slippage charged per leg on gross premium turnover. Condors and winged straddles lose at 45, 60, 90, 120, 180 and 240 DTE and get worse with tenor. Consistent with research/128; NOT a contradiction of research/127, whose 2.5% body and 7% wings work on STOCKS, where tails are idiosyncratic.',
        columns: ['Structure', 'n', 'Win %', 'Avg credit', 'Avg net', 't', 'Worst trade', 'Entry margin', 'Return on margin'],
        rows: [
          ['5% strangle', '143', '82.5', '182.8', '+41.4', '3.62', '-615', 'Rs 1.57L', '30.6%'],
          ['3.5% strangle', '143', '75.5', '268.2', '+44.6', '3.20', '-830', 'Rs 1.74L', '27.1%'],
          ['ATM straddle (LIVE)', '140', '73.6', '639.5', '+56.6', '2.96', '-1,047', 'Rs 2.22L', '24.5%'],
          ['2.5% strangle', '142', '72.5', '343.2', '+44.4', '2.81', '-892', 'Rs 1.88L', '24.2%'],
          ['1.5% strangle', '141', '72.3', '440.4', '+47.9', '2.70', '-994', 'Rs 2.00L', '23.6%'],
          ['Winged straddle +/-10%', '139', '69.1', '577.1', '+21.4', '1.36', '-992', 'Rs 1.89L', '10.9%'],
          ['Condor 5% body / 7% wing', '139', '77.0', '125.7', '+10.2', '1.18', '-618', 'Rs 1.30L', '8.7%'],
          ['Condor 2.5% / 3%', '139', '59.7', '190.2', '-0.7', '-0.10', '-365', 'Rs 1.10L', '-0.7%'],
        ],
        highlightRows: [2],
      },
      {
        title: 'The correction - the strangle paired against the straddle',
        caption: 'An interim of this study called the 5% strangle a winner on the unpaired table. Paired on the same 140 entry days it loses on six trades in ten. The claim is retracted. Every apparent advantage came from lower variance and a smaller margin block, not from earning more.',
        columns: ['vs the live 45-DTE ATM straddle, same 140 entry days', '3.5% strangle', '5% strangle'],
        rows: [
          ['Median paired delta', '-6.8 pts', '-26.4 pts'],
          ['Mean paired delta', '-10.1', '-13.6'],
          ['Paired t', '-1.32', '-1.27'],
          ['Trades it wins on', '63 / 140 (45.0%)', '55 / 140 (39.3%)'],
          ['Correlation to the straddle', '0.94', '0.87'],
        ],
        highlightRows: [3],
      },
    ],
    results: {
      metrics: [
        { label: 'Cells above t = 2', value: '2 of 624', hint: 'both are the live 45-DTE book: 45 to 21 and 45 to 20' },
        { label: 'Live 45-DTE book', value: '+56.6 pts / trade', tone: 'pos', hint: 't 2.96, 140 trades, 73.6% win, 24.5%/yr on measured margin' },
        { label: '60-DTE book', value: '+46.3 pts / trade', hint: 't 1.36, drawdown 3x deeper, negative across the first half' },
        { label: '90-DTE book and beyond', value: 'negative', tone: 'neg', hint: '-5.5 then -32.2 then -70.4 then -164.3 pts a trade' },
        { label: '365-DTE ATM call liquidity', value: '54 contracts / day', tone: 'neg', hint: 'the 2-year contracts trade four; killed on data, not on P and L' },
        { label: 'Stop families beaten by no-stop', value: '7 of 7', hint: 'at every tenor; damage monotone in fire rate' },
        { label: 'Re-centring cost', value: '+56.6 becomes -12.8', tone: 'neg', hint: 'chopping the 45 to 21 hold into five re-centred pieces' },
        { label: 'Option rows repaired', value: '+2,836,719', tone: 'pos', hint: 'eight years of missing long-dated NIFTY history, now permanent' },
      ],
      tables: [
        {
          title: 'Per-year net points, 0.75% slippage, no VIX filter',
          caption: 'Intra-year drawdown in brackets, measured off the running peak of the FULL curve per research/154 - never off the year own first bar. The 45-DTE book loses in one year of eleven; the 60-DTE book loses in four.',
          columns: ['Year', '45 ATM (LIVE)', '60 ATM', '45 3.5% strangle', '45 5% strangle', '60 5% strangle'],
          rows: [
            ['2016', '+670.0 (-59)', '+22.1 (-449)', '+565.0 (-52)', '+533.2 (0)', '+103.3 (-212)'],
            ['2017', '+240.9 (-294)', '-442.2 (-764)', '+217.5 (-147)', '+167.8 (-106)', '-133.8 (-253)'],
            ['2018', '+109.4 (-578)', '-1,104.3 (-1,683)', '+263.7 (-337)', '+154.8 (-259)', '-446.7 (-680)'],
            ['2019', '+87.1 (-395)', '+658.1 (-1,534)', '+127.7 (-237)', '+159.8 (-134)', '+675.9 (-707)'],
            ['2020', '+1,035.0 (-363)', '-1,850.6 (-2,875)', '+1,196.0 (-317)', '+1,344.4 (-237)', '-778.2 (-1,326)'],
            ['2021', '+671.5 (-520)', '+1,406.9 (-3,254)', '+812.3 (-246)', '+905.9 (-145)', '-379.3 (-1,758)'],
            ['2022', '+1,224.8 (-464)', '+1,197.4 (-1,432)', '+1,146.1 (-356)', '+988.1 (-268)', '+800.3 (-1,233)'],
            ['2023', '-241.9 (-1,047)', '-1,674.9 (-3,392)', '-242.8 (-830)', '-203.9 (-624)', '-820.0 (-1,455)'],
            ['2024', '+1,044.9 (-1,243)', '+1,339.2 (-3,080)', '+800.9 (-913)', '+961.4 (-653)', '+802.8 (-1,420)'],
            ['2025', '+2,135.3 (-167)', '+2,361.7 (-1,696)', '+964.2 (-315)', '+614.9 (-221)', '+819.8 (-510)'],
            ['2026', '+1,249.1 (-644)', '+267.3 (-1,428)', '+732.6 (-543)', '+475.9 (-453)', '+120.5 (-798)'],
            ['Years positive', '10 of 11', '7 of 11', '10 of 11', '10 of 11', '7 of 11'],
            ['Best / worst year', '+2,135 / -242', '+2,362 / -1,851', '+1,196 / -243', '+1,344 / -204', '+820 / -820'],
          ],
          highlightRows: [11],
        },
        {
          title: 'Two-window split and the VIX entry filter on the live book',
          caption: 'Both halves positive in every filter setting, and the filter effect is monotone. This validates the rule already running live. The above-50 variant is better still in sample on 58 trades and is deliberately NOT proposed on that basis.',
          columns: ['VIX rank filter', 'Full window', 'First half', 'Second half', 'Return on margin'],
          rows: [
            ['off', '+64.3 / trade, t 3.18, n 128', '+42.1, t 1.80', '+82.6, t 2.63', '27.8%/yr'],
            ['above 25 (THE LIVE RULE)', '+86.5 / trade, t 3.64, n 87', '+56.6, t 1.85', '+113.2, t 3.18', '37.3%/yr'],
            ['above 50', '+95.5 / trade, t 2.97, n 58', '+77.5, t 1.90', '+112.3, t 2.26', '41.5%/yr'],
          ],
          highlightRows: [1],
        },
        {
          title: 'Cost sensitivity - return on measured margin',
          caption: 'The strangle one genuine advantage is here: it loses 8% of its return across this ladder where the straddle loses 22%, because it sells 189 points of premium instead of 663 and pays slippage on a third of the turnover. The 60-DTE book does not survive the ladder at all.',
          columns: ['Slippage per side', '45 ATM straddle', '45 5% strangle', '60 ATM straddle'],
          rows: [
            ['0.25%', '30.5%/yr (t 3.49)', '35.5%/yr (t 4.06)', '7.3%/yr (t 0.72)'],
            ['0.75%', '27.8%/yr (t 3.18)', '34.3%/yr (t 3.93)', '5.1%/yr (t 0.50)'],
            ['1.50%', '23.7%/yr (t 2.72)', '32.5%/yr (t 3.73)', '1.8%/yr (t 0.17)'],
          ],
        },
        {
          title: 'Tradeability gate',
          caption: 'Net of cost at 0.75% slippage, in index points. 1 point is Rs 65 per lot; the live book runs 3 lots, so 1 point is Rs 195.',
          columns: ['Spec', 'n', 'Win %', 'Avg win', 'Avg loss', 'Expectancy', 'Max losing streak', 'Trades / yr', 'Worst'],
          rows: [
            ['45 ATM straddle (LIVE)', '140', '73.6', '+156.6', '-221.6', '+56.6', '3', '11.7', '-1,047'],
            ['45 3.5% strangle', '143', '75.5', '+110.7', '-159.5', '+44.6', '2', '11.9', '-830'],
            ['45 5% strangle', '143', '82.5', '+84.0', '-159.9', '+41.4', '2', '11.9', '-615'],
            ['60 ATM straddle', '138', '69.6', '+199.0', '-372.7', '+25.0', '3', '11.5', '-2,000'],
            ['60 5% strangle', '140', '72.1', '+114.0', '-257.6', '+10.5', '3', '11.7', '-1,326'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Measured margin per lot, straight from Kite',
          caption: 'NRML, standalone (consider_positions false), 15-Sep-2026, NIFTY spot 23,190, lot 65. Margin roughly TRIPLES out to the 1-year tenor. Note also that long-dated strikes sit on a 1,000 to 1,500 point grid, so the nearest listed strike is 3.05% from spot - a 1-year ATM straddle cannot be placed at all.',
          columns: ['DTE', '42', '69', '105', '196', '287', '469', '651', '1015'],
          rows: [
            ['ATM straddle', 'Rs 2.21L', 'Rs 2.29L', 'Rs 2.55L', 'Rs 3.20L', 'Rs 5.69L', 'Rs 6.28L', 'Rs 6.81L', 'Rs 7.76L'],
            ['2.5% strangle', 'Rs 1.87L', 'Rs 1.97L', 'Rs 1.98L', 'Rs 2.37L', 'Rs 4.88L', 'Rs 5.34L', 'Rs 6.10L', 'Rs 7.10L'],
            ['2.5 body / 7 wing condor', 'Rs 1.48L', 'Rs 1.72L', 'Rs 1.41L', 'Rs 1.61L', 'Rs 3.10L', 'Rs 3.91L', 'Rs 5.51L', 'Rs 6.32L'],
          ],
        },
      ],
    },
    winners: [
      {
        config: 'NOTHING ADOPTED - the incumbent 45-DTE book wins its own bake-off',
        summary:
          'The live research/119 book is the only cell in the study that clears any bar, and it was not a candidate - it was the control. No longer tenor, no alternative structure and no stop family beat it. The study output is a closed line of enquiry and a repaired database.',
        metrics: [
          { k: 'Live spec, unchanged', v: 'Sell ATM NIFTY straddle at expiry minus 45 calendar days, VIX percentile rank above 25, exit at 21 DTE or 50% target or 200% stop, 3 lots' },
          { k: 'Measured performance', v: '+56.6 net points a trade, t 2.96, 140 trades, 73.6% win, 24.5% a year on Rs 2.22L of measured margin per lot' },
          { k: 'With the live VIX filter', v: '+86.5 net points a trade, t 3.64, 37.3% a year on margin, both halves positive' },
          { k: 'Independent reproduction', v: 'n 92, average credit 782.4, t 3.12 against research/119 published n 89, 786.3, t 3.12' },
        ],
        rejected: [
          '60-DTE straddle - t 1.36, drawdown 3x deeper, loses across the first half',
          '90 / 105 / 150 / 180 / 210 / 240 / 270 / 300-DTE straddles - all net NEGATIVE',
          '365-DTE and longer - untradeable, 54 and 4 contracts a day; also unplaceable, nearest strike 3% from spot',
          'Iron condors at every body and wing width, at every tenor',
          'Winged straddles at 5, 7 and 10%, at every tenor',
          'Combined-premium stop - refuted again, ninth time for this family counting research/119 phase G',
          'Underlying-move stop - refuted again, paired t -2.84',
          'SINGLE-SIDE stop (new) - refuted; identical to whole-position stops at matched fire rates',
          'VIX-level, VIX-rank and VIX-relative-to-entry stops (new) - all refuted, paired t down to -3.5',
          '5% strangle - retracted after the paired test; lower return, not a better trade',
        ],
      },
    ],
    caveats: [
      'DAILY CLOSE ONLY. Expired-contract intraday option data cannot be obtained from Kite, and the 1-minute recorder starts 20-Apr-2026 at about 27 DTE. A stop that would have fired intraday and reverted by the close is invisible here - which means the stop families are tested in their MOST FAVOURABLE form and still lose.',
      'Slippage at the long end is assumed, not measured - bhavcopy carries no bid and ask. For a strike trading 54 contracts a day, 0.75% of premium is probably generous. The long tenors are reported at 0.25, 0.75 and 1.5% and are negative at all three, so the assumption is not load-bearing.',
      'Sample size is the binding limit at the long end and no method fixes it: a 365-DTE book yields nine non-overlapping trades in twelve years. Those cells are reported for completeness and should not be read as evidence either way. The kill rests on liquidity and on the monotone decay from 45 outward, not on their point estimates.',
      'Multiple testing: 624 cells in the tenor sweep, 168 in structures, about 340 in stops. The 45 to 21 result is not a discovery of this study - it is the incumbent, independently reproduced, and it sits at the top of a monotone ordering rather than on a lone peak.',
      'Margin is a single-day snapshot (15-Sep-2026, spot 23,190). SPAN scans move with volatility, and the long-tenor requirements in particular will be higher in a stressed market - which makes the long end worse, not better.',
      'The window 2015 to 2026 contains one genuine volatility crisis (2020) and one bad grind (2023). It does not contain 2008.',
      'The strangle finding is in sample, selected from 14 structures, and unadopted. Its plateau across 1.5 / 2.5 / 3.5 / 5.0% is reassuring, but it loses paired on points and is a replacement rather than an addition, at correlation 0.87.',
      'The exit-DTE conclusion is a statement about resolution, not about equivalence: 9, 14, 21 and 27 DTE cannot be told apart at n 140, and only the 21-DTE interval excludes zero.',
    ],
    githubLinks: [
      { label: 'research/174 - STATUS', href: 'https://github.com/castroarun/quantifyd/blob/main/research/174_long_dated_short_premium/NIFTY_LONGDATED_SHORT_PREMIUM_DAILY_SWEEP_STATUS.md' },
      { label: 'research/174 - RESULTS', href: 'https://github.com/castroarun/quantifyd/blob/main/research/174_long_dated_short_premium/results/RESULTS.md' },
      { label: 'research/119 - the live 45-DTE book this is measured against', href: 'https://github.com/castroarun/quantifyd/blob/main/research/119_45dte_short_straddle/results/RESULTS.md' },
    ],
    projectPaths: [
      'research/174_long_dated_short_premium/NIFTY_LONGDATED_SHORT_PREMIUM_DAILY_SWEEP_STATUS.md',
      'research/174_long_dated_short_premium/scripts/engine_lt.py',
      'research/174_long_dated_short_premium/scripts/probe_liquidity_v2.py',
      'research/174_long_dated_short_premium/scripts/backfill_longdated_bhav.py',
      'research/174_long_dated_short_premium/scripts/margin_by_tenor_live.py',
      'research/174_long_dated_short_premium/scripts/run_p1_tenor.py',
      'research/174_long_dated_short_premium/scripts/run_p1b_decay_windows.py',
      'research/174_long_dated_short_premium/scripts/run_p2_structures.py',
      'research/174_long_dated_short_premium/scripts/run_p3_stops.py',
      'research/174_long_dated_short_premium/scripts/run_p5_finals.py',
      'research/174_long_dated_short_premium/scripts/paired_struct.py',
      'research/174_long_dated_short_premium/results/RESULTS.md',
      'research/174_long_dated_short_premium/results/liquidity_v2.csv',
      'research/174_long_dated_short_premium/results/p1_summary.csv',
      'research/174_long_dated_short_premium/results/p2_structures.csv',
      'research/174_long_dated_short_premium/results/p3_stops.csv',
      'research/174_long_dated_short_premium/results/margin_by_tenor.json',
    ],
  },
"""


def main():
    src = TS.read_text(encoding="utf-8")
    if SLUG in src:
        # remove the existing entry: from its opening brace line to the line before the next
        # top-level "  {" or the closing "];"
        start = src.index("  {\n    slug: '%s'" % SLUG)
        rest = src[start + 4:]
        nxt = rest.find("\n  {\n    slug: '")
        end = (start + 4 + nxt + 1) if nxt != -1 else src.index("\n];", start) + 1
        src = src[:start] + src[end:]
        print("removed existing entry")
    marker = "\n];\n\nexport function getStudy"
    assert marker in src, "could not find the array terminator"
    src = src.replace(marker, "\n" + ENTRY + "];\n\nexport function getStudy")
    TS.write_text(src, encoding="utf-8")
    print("appended r/174 study -> %s" % TS)


if __name__ == "__main__":
    main()
