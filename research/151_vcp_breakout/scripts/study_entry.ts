  {
    slug: 'vcp-breakout-research151',
    title: 'BananaPatterns "VCP" screen — the pattern that is not there',
    verdict:
      'NO EDGE. The site publishes a VCP (volatility contraction pattern) screen claiming Rs 10L into Rs 2.6Cr = 25.99x, +72.1% CAGR and a worst fall of only -14.8% over 2020-2025. Three separate questions were tested. THE RULES: their exit engine reproduces 31 of 32 ground-truth trades exactly on both date and price (8% stop on the CLOSE, exit at the close that breaks the 50-day SMA) — the same engine research/142 decoded behind their Blue Sky screen, which became our live Open Alpha book. Their entry pivot is an exact prior CLOSE in 36 of 37 trades, but it is NOT the all-time high (median buy sits 6% below it) and it contains NO volatility contraction: pivot ages run from 1 to 157 bars, 11 of 37 bases have zero measurable contractions, and no fixed lookback can fit them (N would need to be at least 157 and at most 11 simultaneously). Best of 68 candidate reconstructions is a 30-day rolling closing high at 62.2% joint trade match — a PARTIAL pass of the replication gate. THE CLAIM: refuted. Replaying their own dials honestly (realistic fills, 25 bps, after tax, 5% idle cash, 30 seeds) gives 32.4% CAGR with a seed range of 6.5 to 61.6 percent, at a -34.5% drawdown. Their -14.8% worst fall is unreachable; our shallowest path is -21%. Trade counts DO match (121 vs their 164), so the machine is right and the number is not. Their concentrated sizing (2% risk over a 7% stop = 28.6% of capital per position) is what makes their single path so wild. THE STRATEGY: killed by its own null control. Shrinking the pivot lookback toward no-breakout-at-all monotonically IMPROVES the book — 30 days gives Calmar 1.28, 10 days 1.59, 3 days 1.91, 2 days 2.63. Requiring more of the pattern always made things worse, so the screen contributes negative value; everything it earns comes from the surrounding machinery (relative-strength ranking, the liquidity floor, a 15-day moving-average trail and equal-weight slots). The stop is inert too: 6, 8, 10, 15 percent and NO STOP all return the same, because the trail always fires first. PORTFOLIO: rejected on every limb. Correlation to the live Open Alpha book is 0.749 daily and 0.759 monthly against a pre-registered bar of 0.4. The best blend weight adds +0.033 Calmar (bar: +0.10) while making drawdown worse, every heavier weight loses on the paired test, and a plain CASH sleeve at the same weight beats it. Standalone it buys +2.1pp of CAGR over Open Alpha for +16pp of drawdown, and it dies on cost: Calmar 0.90 at 25 bps, 0.71 at 40, 0.51 at 60, on a book that turns over roughly 37 times its NAV per year.',
    status: 'COMPLETE',
    date: '2026-09-05',
    cardBlurb:
      'A screen called VCP whose own published trades contain no volatility contraction. The exit engine replicates to the paisa, the headline 72.1% CAGR does not (32.4% honest, at more than double the claimed drawdown), and a null control shows the breakout pattern subtracts value rather than adding it. Correlation 0.75 to the live Open Alpha book; loses the blend test to plain cash.',
    cardStats: [
      { label: 'Verdict', value: 'NO EDGE — rejected' },
      { label: 'Claimed vs honest CAGR', value: '72.1% vs 32.4%' },
      { label: 'Correlation to Open Alpha', value: '0.75 (bar: <0.40)' },
    ],

    systemRules: {
      intro:
        'The site does not publish its VCP definition, so the rules below are the best of 68 candidate reconstructions, arbitrated against 40 trades transcribed from one of its own VCP backtest runs. The exit half is theirs exactly; the entry half is a 62% match.',
      sharedCoreTitle: 'The reconstructed system',
      sharedCore: [
        { k: 'Universe', v: 'All NSE cash equities with at least 260 daily bars (2,321 symbols). ETFs excluded by name filter.' },
        { k: 'Liquidity floor', v: '20-day median traded value of at least Rs 5 crore, measured as of the previous day.' },
        { k: 'Leader filter', v: 'IBD-weighted relative strength (2 x 63-day + 126-day + 189-day + 252-day return) ranked across eligible names, percentile 70 or better. Swept: this is the only dial of theirs that does real work.' },
        { k: 'Pivot', v: 'The highest CLOSE of the previous 30 trading days. Their published pivots are exact prior closes (36 of 37), never the all-time high, with no contraction structure.' },
        { k: 'Trigger', v: 'The day CLOSE finishes above that pivot, having sat below it and within 20% of it the day before.' },
        { k: 'Fill', v: 'Buy-stop at the pivot, filled at max(pivot, open). Booking the pivot on a gap-up day is fill inflation and is reported as a separate arm.' },
        { k: 'Cut a loser', v: 'CLOSE 8% below the fill. Verified against their trade list, whose own exit labels read stop_8pct even though the panel dial reads 7%.' },
        { k: 'Sell winners', v: 'The first CLOSE below the 15-day simple moving average (their dial is the 50-day; the joint entry-by-exit sweep prefers 15).' },
        { k: 'Book', v: '16 slots at 6.25% of NAV each, cash-constrained, no leverage, no market-regime gate.' },
        { k: 'Their sizing, for contrast', v: 'Position value = risk percent divided by stop distance, capped at 30% of capital. At 2% risk over a 7% stop that is 28.6% per position, so a 5-slot book holds only about 3.5 names and the outcome swings wildly with fill order.' },
      ],
      riskLayer: {
        title: 'The null control — how much work does the VCP pattern actually do?',
        caption:
          '2012-2026, 16 slots x 6.25%, 15-day trail, after tax, 25 bps per side, 10 seeds. Shrinking the lookback toward no-pattern-at-all improves every metric.',
        columns: ['Pivot lookback', 'CAGR', 'Max drawdown', 'Calmar', 'Trades'],
        rows: [
          ['2 days (the null)', '56.6%', '-22.1%', '2.63', '6,319'],
          ['3 days', '49.3%', '-25.6%', '1.91', '5,806'],
          ['5 days', '46.3%', '-26.3%', '1.70', '5,015'],
          ['7 days', '43.9%', '-27.1%', '1.65', '4,618'],
          ['10 days', '43.2%', '-26.9%', '1.59', '4,321'],
          ['30 days (their pattern)', '37.5%', '-29.5%', '1.28', '4,008'],
        ],
        highlightRows: [0],
      },
    },

    system: {
      intro:
        'Economic hypothesis as marketed: a stock coiling through successively shallower pullbacks on drying volume has absorbed its supply, so a break of the pattern high runs on late-comer flow. The test was whether their published trades actually contain that structure. They do not.',
      rows: [
        { k: 'What was replicated first', v: 'The 40-trade ground truth transcribed from one of their VCP backtest runs (37 usable; BONDADA and E2E are not in our database).' },
        { k: 'Entry fingerprint', v: 'Buy price versus every rolling maximum of prior highs and closes, over 16 windows x 8 gaps, plus fractal swing highs and the all-time high — 36 of 37 land on an exact prior CLOSE.' },
        { k: 'Pattern audit', v: 'Pivot ages 1 to 157 bars; 11 of 37 bases contain zero measurable contractions; volume dry-up ratio spans 0.27 to 1.53. There is no minimum base length, no contraction count and no volume condition in their trades.' },
        { k: 'Why no single rule fits', v: 'A fixed lookback would need to be at least 157 bars (deepest pivot age) and at most 11 (shortest run since a higher close) at the same time. Their pivot is structural and, from 37 trades, not identifiable.' },
        { k: 'Falsification declared up front', v: 'Below a 60% joint match the rules would be declared not reproducible. The best reconstruction reached 62.2%, so the study proceeded — but as OUR reconstruction, not as their engine.' },
        { k: 'Ranking metric, pre-registered', v: 'After-tax Calmar at 25 bps per side on the long window, 30-seed median, worst-seed CAGR as tie-break.' },
      ],
    },

    conditions: {
      intro:
        'Everything runs on the VPS daily database. Rolling statistics are computed on each symbol own de-duplicated series and only then aligned, so a missing or phantom row cannot poison a window — the exact bug that silently disabled a gate in research/142.',
      rows: [
        { k: 'Data', v: 'market_data_unified daily bars, 2,321 symbols, 5,528 trading days from 2004-06 (550 days of warm-up before the 2006 study start).' },
        { k: 'Windows', v: 'Their window 2020-2025, plus 2012-2026 and 2006-2026. Adoption figures use 2006-04 to 2026-09.' },
        { k: 'Costs', v: '25 bps per side headline, with a full ladder at 40 and 60 bps. Real explicit cost in NSE delivery is about 13 bps; the rest is slippage headroom.' },
        { k: 'Tax', v: '20% short-term / 12.5% long-term above 365 days, netted within the Indian financial year with losses carried forward.' },
        { k: 'Idle cash', v: '5% per annum accrued daily on uninvested capital.' },
        { k: 'Path dependence', v: 'Random selection among same-day candidates when slots are scarce; 30 seeds for every adoption number, 10 for scanning. Medians with the full range, and the worst seed, are reported.' },
        { k: 'Multiple testing', v: 'About 230 sweep cells across seven axes. Disclosed so any single winner can be discounted.' },
      ],
    },

    comparisons: [
      {
        title: 'Their published claim versus an honest replay',
        caption: '2020-01-01 to 2025-12-31, their dials: 5 positions, 2% risk, trail 50-day, weak-market skip off, Rs 10L.',
        columns: ['Arm', 'Terminal', 'CAGR (median [min..max])', 'Max drawdown', 'Trades'],
        rows: [
          ['Published on the site (marked provisional)', '25.99x', '+72.1%', '-14.8%', '164'],
          ['Faithful replica: their optimistic pivot fills, no costs, no tax', '7.64x', '40.0% [23.5..65.0]', '-26.8%', '124'],
          ['Honest: realistic fills, 25 bps, after tax, 5% idle cash, 30 seeds', '5.38x', '32.4% [6.5..61.6]', '-34.5%', '121'],
        ],
        highlightRows: [2],
      },
      {
        title: 'Their risk-based sizing versus fixed weights',
        caption: 'After tax, 25 bps, realistic fills, 30 seeds. Fixed weights win on Calmar in every window and halve the seed spread.',
        columns: ['Window', 'Their dials (5 slots, risk-sized)', 'Fixed 16 x 6.25%'],
        rows: [
          ['2020-2025', '32.4% / -34.5% / Calmar 0.96', '36.5% / -29.4% / Calmar 1.23'],
          ['2012-2026', '31.6% / -44.0% / Calmar 0.71', '31.0% / -34.2% / Calmar 0.89'],
          ['2006-2026', '25.5% / -54.9% / Calmar 0.45', '27.5% / -51.6% / Calmar 0.53'],
        ],
      },
      {
        title: 'Which of their dials actually matter',
        caption: '2012-2026, 16 slots x 6.25%, after tax, 25 bps, 10 seeds. Two of their three headline dials are inert.',
        columns: ['Dial', 'Setting', 'CAGR', 'Calmar', 'Read'],
        rows: [
          ['Cut a loser at', '6% / 8% / 10% / 15% / none', '43.0 / 43.2 / 43.2 / 43.2 / 43.2', '1.63 / 1.59 / 1.64 / 1.62 / 1.62', 'INERT — the trail always fires first'],
          ['Near the trigger', '10% / 20% / 50%', '43.3 / 43.2 / 43.2', '1.68 / 1.59 / 1.59', 'INERT above 10%'],
          ['Relative strength', '0 / 50 / 70 / 85', '32.4 / 38.9 / 43.3 / 51.2', '1.12 / 1.37 / 1.68 / 1.47', 'THE real filter; 70 is about the Calmar optimum'],
          ['Sell winners by', '50-day / 20-day / 15-day trail', '30.9 / 34.5 / 37.5', '0.88 / 1.08 / 1.28', 'Tighter trails dominate; their 50-day is the weakest'],
          ['Skip weak markets', 'off / NIFTY 200-day / 100-day', '43.2 / 34.1 / 35.4', '1.59 / 1.27 / 1.38', 'The gate costs more than it saves here'],
        ],
        highlightRows: [0, 2],
      },
    ],

    results: {
      metrics: [
        { label: 'Verdict', value: 'NO EDGE', tone: 'neg' },
        { label: 'Replication gate', value: '62.2% joint' },
        { label: 'Standalone CAGR (30 seeds, after tax)', value: '36.1%' },
        { label: 'Standalone max drawdown', value: '-40.8%', tone: 'neg' },
        { label: 'Standalone Calmar', value: '0.89' },
        { label: 'Correlation to Open Alpha', value: '0.75', tone: 'neg' },
        { label: 'Best blend Calmar uplift', value: '+0.033 (bar +0.10)', tone: 'neg' },
        { label: 'Calmar at 60 bps per side', value: '0.51', tone: 'neg' },
      ],
      tables: [
        {
          title: 'Adopted spec versus the deployed book, 2006-04 to 2026-09',
          caption: 'After tax, 25 bps per side, idle cash 5%. Medians across 30 VCP seeds, 10 Open Alpha seeds and 3 True North rebalance offsets.',
          columns: ['Book', 'CAGR', 'Max drawdown', 'Calmar'],
          rows: [
            ['VCP (this study)', '35.6%', '-41.1%', '0.87'],
            ['Open Alpha (live)', '33.5%', '-24.9%', '1.35'],
            ['True North (live)', '19.6%', '-24.1%', '0.81'],
            ['TN + OA 50-50 (the deployed pair)', '27.2%', '-16.8%', '1.62'],
            ['TN + OA + VCP 40/40/20', '29.1%', '-19.0%', '1.53'],
            ['NIFTY 50', '9.1%', '-38.4%', '0.24'],
            ['Midcap 150', '14.7%', '-44.2%', '0.33'],
            ['Smallcap 250', '12.2%', '-60.8%', '0.20'],
          ],
          highlightRows: [3],
        },
        {
          title: 'The blend test — and the cash-null that beats it',
          caption: 'Paired across 10 Open Alpha seeds x 3 True North offsets, monthly rebalanced, after tax. Pre-registered bar: +0.10 Calmar or -2pp drawdown at equal-or-better CAGR, correlation under 0.4, and it must beat plain cash at the same weight.',
          columns: ['Blend', 'CAGR', 'Max drawdown', 'Calmar', 'Paired change in Calmar', 'Paths won'],
          rows: [
            ['TN + OA 50-50 (baseline)', '27.17%', '-16.42%', '1.597', '-', '-'],
            ['+ VCP 10%', '28.14%', '-16.77%', '1.642', '+0.033', '27 of 30'],
            ['+ VCP 15%', '28.62%', '-17.86%', '1.584', '-0.029', '12 of 30'],
            ['+ VCP 20%', '29.10%', '-18.96%', '1.520', '-0.089', '6 of 30'],
            ['+ VCP 25%', '29.66%', '-20.05%', '1.463', '-0.146', '1 of 30'],
            ['+ VCP 33%', '30.48%', '-21.82%', '1.384', '-0.223', '0 of 30'],
            ['+ CASH 10% (the null)', '24.94%', '-14.48%', '1.659', '-', '-'],
            ['+ CASH 20% (the null)', '22.72%', '-12.63%', '1.745', '-', '-'],
          ],
          highlightRows: [6, 7],
        },
        {
          title: 'Where the third sleeve actually lands — crash and grind windows',
          caption: 'Blend medians. The deployed pair has already stripped the crash tail; adding VCP puts it back and is worse in the grinds too.',
          columns: ['Blend', '2008 crash', '2018 grind', '2022 H1 grind'],
          rows: [
            ['TN + OA 50-50', '+1.2% (dd -2.6%)', '-9.9% (dd -11.2%)', '-5.9% (dd -9.1%)'],
            ['+ VCP 20%', '-5.0% (dd -7.0%)', '-11.3% (dd -12.3%)', '-7.7% (dd -10.5%)'],
            ['+ CASH 20%', '+2.1% (dd -1.9%)', '-7.2% (dd -8.6%)', '-4.3% (dd -7.1%)'],
          ],
          highlightRows: [2],
        },
        {
          title: 'Cost ladder and robustness on the adopted spec',
          caption: '2006-2026, 16 slots x 6.25%, after tax. The book trades about 37 times its NAV per year, so it loses roughly 6.8pp of CAGR per extra 15 bps per side.',
          columns: ['Arm', 'CAGR', 'Max drawdown', 'Calmar'],
          rows: [
            ['25 bps per side (headline)', '37.0%', '-41.1%', '0.90'],
            ['40 bps per side', '30.9%', '-43.2%', '0.71'],
            ['60 bps per side', '23.4%', '-45.9%', '0.51'],
            ['Delete the 10 best trades', '37.0%', '-41.1%', '0.90'],
            ['Cap every winner at +50%', '32.0%', '-40.6%', '0.79'],
            ['First window 2006-2015', '27.7%', '-41.1%', '0.68'],
            ['Second window 2016-2026', '43.4%', '-29.2%', '1.50'],
          ],
          highlightRows: [2],
        },
        {
          title: 'Year by year — return with the intra-year drawdown beneath it',
          caption: 'After tax, 25 bps, seed and offset medians. Best-of picks exclude benchmarks.',
          columns: ['Year', 'VCP', 'Open Alpha', 'True North', 'TN+OA 50-50', 'TN+OA+VCP 40/40/20', 'NIFTY 50', 'Best overall'],
          rows: [
            ['2006', '31.8 (-15.3)', '19.0 (-12.3)', '9.0 (-10.9)', '16.1 (-3.8)', '21.0 (-3.1)', '-', 'TN+OA+VCP'],
            ['2007', '128.5 (-15.7)', '131.9 (-13.9)', '54.8 (-15.6)', '92.9 (-6.5)', '99.4 (-5.9)', '-', 'Open Alpha'],
            ['2008', '-30.2 (-36.8)', '-16.0 (-14.9)', '-12.8 (-19.6)', '-14.1 (-2.5)', '-17.5 (-6.2)', '-', 'TN+OA 50-50'],
            ['2009', '68.3 (-11.0)', '54.9 (-11.3)', '67.7 (-10.8)', '59.2 (-3.1)', '60.9 (-3.4)', '-', 'TN+OA+VCP'],
            ['2010', '24.6 (-10.1)', '12.5 (-8.9)', '4.6 (-15.4)', '8.7 (-3.6)', '12.2 (-4.0)', '-', 'VCP'],
            ['2011', '-16.4 (-16.7)', '-6.2 (-11.3)', '-9.5 (-12.8)', '-7.4 (-3.8)', '-8.8 (-5.4)', '-24.9 (-26.2)', 'TN+OA 50-50'],
            ['2012', '11.6 (-16.2)', '12.9 (-7.8)', '20.7 (-6.8)', '16.8 (-4.0)', '15.4 (-4.0)', '27.7 (-13.8)', 'True North'],
            ['2013', '7.8 (-12.8)', '13.7 (-7.2)', '6.4 (-6.2)', '8.8 (-2.9)', '9.5 (-2.5)', '6.8 (-14.6)', 'TN+OA+VCP'],
            ['2014', '92.3 (-11.4)', '79.7 (-10.0)', '40.7 (-11.9)', '61.2 (-5.5)', '66.6 (-4.3)', '31.4 (-6.5)', 'VCP'],
            ['2015', '9.9 (-19.4)', '7.2 (-9.9)', '-2.3 (-9.2)', '3.4 (-7.0)', '4.6 (-7.1)', '-4.1 (-16.0)', 'TN+OA+VCP'],
            ['2016', '17.2 (-16.6)', '12.5 (-11.9)', '27.8 (-4.5)', '20.2 (-1.9)', '21.3 (-3.2)', '3.0 (-12.5)', 'True North'],
            ['2017', '114.7 (-17.9)', '105.3 (-18.6)', '36.6 (-7.1)', '69.2 (-0.4)', '73.2 (0.0)', '28.6 (-4.1)', 'VCP'],
            ['2018', '-20.2 (-27.9)', '-19.9 (-18.4)', '-7.6 (-22.8)', '-13.0 (-13.4)', '-12.0 (-12.7)', '3.2 (-14.6)', 'TN+OA+VCP'],
            ['2019', '18.5 (-11.5)', '7.1 (-8.4)', '-1.1 (-7.5)', '3.6 (-6.1)', '4.8 (-6.6)', '12.0 (-11.4)', 'VCP'],
            ['2020', '81.9 (-11.5)', '118.4 (-10.4)', '66.0 (-7.0)', '90.0 (-2.3)', '88.9 (-3.5)', '14.9 (-38.4)', 'Open Alpha'],
            ['2021', '143.6 (-9.5)', '148.3 (-7.5)', '62.5 (-11.1)', '102.6 (0.0)', '108.7 (0.0)', '24.1 (-10.1)', 'Open Alpha'],
            ['2022', '30.2 (-25.1)', '0.6 (-20.6)', '17.2 (-8.3)', '8.3 (-10.3)', '12.5 (-10.1)', '4.3 (-16.5)', 'True North'],
            ['2023', '68.6 (-12.0)', '53.1 (-10.5)', '39.2 (-10.3)', '48.4 (-1.7)', '53.9 (-1.7)', '20.0 (-7.1)', 'VCP'],
            ['2024', '58.9 (-15.9)', '60.0 (-12.3)', '9.9 (-17.9)', '35.1 (-8.2)', '42.4 (-7.3)', '8.8 (-10.9)', 'Open Alpha'],
            ['2025', '21.9 (-17.2)', '9.0 (-18.4)', '10.7 (-8.3)', '10.5 (-3.5)', '12.4 (-4.0)', '10.5 (-8.7)', 'TN+OA+VCP'],
            ['2026 YTD', '19.6 (-24.3)', '35.6 (-21.6)', '9.7 (-6.2)', '22.5 (-5.4)', '20.2 (-7.5)', '-7.5 (-15.2)', 'TN+OA 50-50'],
          ],
        },
      ],
      charts: [
        {
          src: '/app/vcp-research151.png',
          caption:
            'Growth of Rs 100 on a log scale with the drawdown panel beneath. The VCP book (orange) outruns the deployed pair on return and pays for it with roughly two and a half times the drawdown. After tax, 25 bps per side, seed and offset medians, 2006-04 to 2026-09.',
        },
      ],
    },

    winners: [
      {
        config: 'Nothing is adopted',
        summary:
          'The screen is rejected. What survives is knowledge: their exit engine is now decoded twice over, and the null control tells us where the returns in this whole family actually come from.',
        metrics: [
          { k: 'Exit engine match', v: '31 of 32 ground-truth trades, exact on date and price' },
          { k: 'Entry reconstruction', v: '30-day rolling closing high, 62.2% joint match, best of 68 candidates' },
          { k: 'What actually earns', v: 'Relative-strength ranking, the liquidity floor, a 15-day trail and equal-weight slots — not the pattern' },
          { k: 'Deliverables for the correlation study', v: '30 after-tax daily equity curves plus the spec, in research/151 results' },
        ],
        rejected: [
          'The published 25.99x / +72.1% CAGR / -14.8% claim — refuted at 32.4% and -34.5% on their own dials',
          'The VCP pattern itself — the null control shows less pattern is always better',
          'The 7% cut-a-loser dial — inert, the trail always fires first',
          'A third sleeve at any weight from 10% to 33% — beaten by plain cash and by the pre-registered Calmar bar',
        ],
      },
    ],

    caveats: [
      'Survivorship on both sides: Kite lists only current instruments, so delisted names are absent from our universe. Their backtest very likely shares this. 2006 coverage is about 528 priced symbols, so the early window is survivorship-flattered.',
      'The replication gate is a 62% match, not the trade-exact 95% research/142 achieved on their Blue Sky screen. Their VCP definition is unpublished and, from 37 trades, not identifiable — a different reconstruction could score differently, although the null control makes it hard to see how any pattern definition rescues the family.',
      'The daily database is not retroactively split-adjusted, which matters for a screen built on highs. Research/142 repaired 72 scale-broken symbols; no scale anomaly appeared in the 37 ground-truth trades.',
      'Capacity was estimated, not measured against real fills. At 254 trades a year and 6.25% of NAV per position on names with a Rs 5 crore daily turnover floor, a Rs 1 crore book would be taking roughly 12% of a qualifying name daily volume.',
      'No paper soak was run — the honest consequence of a kill verdict rather than an omission.',
      'Not tested: intraday fill behaviour and circuit limits on breakout days, a market-capitalisation floor (we have no shares-outstanding history), and any options or leveraged expression of the signal.',
      'About 230 sweep cells were run across seven axes. Any single cell that looks good should be discounted for multiple testing; the conclusions above rest on monotone axes and null controls, not on cell peaks.',
    ],

    githubLinks: [
      {
        label: 'research/151_vcp_breakout',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/151_vcp_breakout',
      },
      {
        label: 'research/142 — the Blue Sky replication this builds on',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/142_bananapatterns_replication',
      },
    ],
    projectPaths: [
      'research/151_vcp_breakout/VCP_BREAKOUT_DAILY_SWEEP_STATUS.md',
      'research/151_vcp_breakout/results/RESULTS.md',
      'research/151_vcp_breakout/results/vcp_adopted_spec.json',
      'research/151_vcp_breakout/results/vcp_equity_seeds.csv',
      'research/151_vcp_breakout/scripts/vcp_replay.py',
    ],
  },
