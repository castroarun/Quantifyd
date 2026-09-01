  {
    slug: 'bluesky-ath-breakout-research142',
    title: 'BananaPatterns "Blue Sky" ATH Breakout — forensic replication + 20-year robustness',
    verdict:
      'bananapatterns.com showcases a blue-sky breakout backtest at 79.8% CAGR / -11.4% worst fall (2020-25, marked PROVISIONAL by the site itself). We decoded the engine trade-exactly (37/39 exits reproduced to the day and paisa; entries = buy-stop at the prior all-time-high CLOSE; RS = IBD-weighted momentum percentile >= 70) — the rules are real, the published numbers are not reproducible: six honest selection paths land 6.5-15.7x vs their 33.7x, and their -11.4% drawdown is unreachable at ANY marking frequency (best honest: -22% daily / -15% monthly). The construction itself, however, survives 20 years: 2006-2025 net of 25bps with realistic fills, a point-in-time Rs 500cr mcap floor and the NIFTYBEES<SMA200 gate, a 10-seed selection ensemble medians 30.4% CAGR (range 27.9-34.4%) at -31.5% median MaxDD — ~203x vs NIFTYBEES 10.3x (12.3%, -59.7%). Same tier as the research/75 momentum book (31.9% net). STRATEGY family credible; single-path CAGRs above ~40% from this design are selection luck, not edge.',
    status: 'COMPLETE',
    date: '2026-09-02',
    cardBlurb:
      'Reverse-engineered a viral backtest site trade-by-trade, proved its rules and refuted its headline numbers, then gave the decoded system an honest 20-year, net-of-cost, seed-ensemble examination against Nifty 50, Midcap 150 and Smallcap 250.',
    cardStats: [
      { label: 'CAGR (net, median)', value: '30.4%' },
      { label: 'MaxDD (median)', value: '-31.5%' },
      { label: '2006-25 multiple', value: '~203x' },
    ],
    system: {
      intro:
        'The decoded rules (validated against 90+ of their published trades before any long backtest was run):',
      rows: [
        { k: 'Universe', v: 'All NSE EQ dailies (2,321 syms post repair); liquidity floor 20d-median traded value >= Rs 5cr; ETFs excluded; headline config adds mcap >= Rs 500cr point-in-time (constant-adjusted-shares proxy).' },
        { k: 'Setup', v: 'Prev close within 20% of the all-time-high CLOSE and below it; IBD-weighted RS percentile (2xr63+r126+r189+r252) >= 70.' },
        { k: 'Signal', v: 'A CLOSE above the prior ATH-close ("a close into new highs") — 44/45 of their entry days confirm; fill booked AT the pivot (their convention; our honest runs fill at max(open, pivot)).' },
        { k: 'Exits', v: '-8% stop on the close (gap-aware); trail = exit at the close that breaks the 50-SMA.' },
        { k: 'Book', v: '8 slots, 18.75% of equity per position (1.5% risk / 8% stop), cash-constrained, pyramiding allowed; selection among simultaneous signals is UNDISCLOSED -> we report a 10-seed random-selection ensemble, not one lucky path.' },
        { k: 'Gate', v: 'Headline config blocks new entries while NIFTYBEES < its SMA200.' },
      ],
    },
    conditions: {
      intro: 'Two examinations: faithful 2020-25 replication vs their claims, then 2006-2025 robustness.',
      rows: [
        { k: 'Phase 2 (their window)', v: '2020-01-01 to 2025-12-31, Rs 10L, their fills, no costs — apples-to-apples vs the site.' },
        { k: 'Phase 3 (robustness)', v: '2006-01-01 to 2025-12-31, realistic fills, 25bps/side, 10 random-selection seeds per config.' },
        { k: 'Host / data', v: 'VPS market_data.db (post split-adjustment repair, 2026-09-01); scripts in research/142.' },
      ],
    },
    comparisons: [
      {
        title: 'Their window (2020-25): published vs honest replication of the SAME rules',
        columns: ['Run', 'Terminal', 'CAGR', 'MaxDD', 'Trades', 'Win%'],
        rows: [
          ['Site published (PROVISIONAL)', '33.74x', '79.8%', '-11.4% "worst fall"', '272', '52%'],
          ['Faithful replica (RS-desc picks)', '11.01x', '49.2%', '-31.5%', '175', '42%'],
          ['Random-selection seeds (range of 5)', '6.5-15.1x', '37-57%', '-22 to -32%', '~180', '44-46%'],
          ['Best variant: weak-market gate ON', '15.73x', '58.3%', '-22.0%', '141', '48%'],
        ],
        highlightRows: [3],
        heatmap: false,
      },
      {
        title: '20-year robustness (2006-25) — 10-seed ensemble medians [min..max]',
        columns: ['Config', 'Terminal x', 'CAGR', 'MaxDD', 'Signals'],
        rows: [
          ['A: gate ON, their fills, gross', '398 [228..813]', '34.9% [31.2..39.8]', '-44.0%', '16,612'],
          ['B: gate ON, real fills, net 25bps', '287 [138..758]', '32.7% [28.0..39.3]', '-45.7%', '16,612'],
          ['C: gate OFF, real fills, net 25bps', '225 [108..413]', '31.1% [26.4..35.2]', '-45.0%', '16,612'],
          ['D (HEADLINE): B + mcap >= 500cr PIT', '203 [136..367]', '30.4% [27.9..34.4]', '-31.5%', '8,069'],
          ['Benchmark: NIFTYBEES B&H', '10.25', '12.3%', '-59.7%', '—'],
          ['Reference: research/75 momentum (net)', '—', '31.9%', '-31.6%', '—'],
        ],
        highlightRows: [3],
        heatmap: false,
      },
      {
        title: 'Per-year, headline config D (ensemble median) vs Nifty 50',
        columns: ['Year', 'BlueSky D %', 'NIFTYBEES %'],
        rows: [
          ['2006', '+43.5', '—'], ['2007', '+116.9', '—'], ['2008', '-20.1', '—'],
          ['2009', '+35.1', '—'], ['2010', '+23.0', '—'], ['2011', '+10.0', '—'],
          ['2012', '+25.7', '+26.5'], ['2013', '+9.1', '+7.2'], ['2014', '+91.1', '+31.6'],
          ['2015', '+31.1', '-4.3'], ['2016', '-2.5', '+4.0'], ['2017', '+101.3', '+29.9'],
          ['2018', '-13.4', '+4.8'], ['2019', '+16.9', '+13.6'], ['2020', '+38.2', '+15.4'],
          ['2021', '+117.3', '+26.0'], ['2022', '+6.7', '+5.5'], ['2023', '+46.2', '+21.0'],
          ['2024', '+52.8', '+10.4'], ['2025', '+10.3', '+11.7'],
        ],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR net (median of 10 seeds)', value: '30.4%', tone: 'pos', hint: 'range 27.9-34.4%' },
        { label: 'MaxDD (median)', value: '-31.5%', tone: 'neg', hint: 'worst seed -33.6%' },
        { label: '20y multiple', value: '~203x', hint: 'vs NIFTYBEES 10.25x' },
        { label: 'Excess CAGR vs Nifty 50', value: '+18.1pp', tone: 'pos' },
        { label: 'Site claim reproduced?', value: 'NO', tone: 'neg', hint: '79.8%/-11.4% unreachable under any honest variant' },
        { label: 'Trade-rule match', value: '37/39 exits exact', tone: 'pos', hint: 'to the day and paisa' },
      ],
      tables: [],
      charts: [
        { src: '/app/bluesky-breakout-tearsheet.png', caption: 'Client factsheet — headline config D (median seed of the 10-seed ensemble) vs NIFTYBEES, 2006-2025, net of 25bps/side.' },
        { src: '/app/bluesky-vs-indices.png', caption: 'Growth of Rs 100 (log), 2011-2025: BlueSky replica (median seed, net) vs NIFTYBEES, NIFTYMIDCAP150, NIFTYSMLCAP250.' },
      ],
      embeds: [
        { src: '/app/bluesky-breakout-tearsheet.html', height: 900, caption: 'Interactive tearsheet (self-contained).' },
      ],
    },
    winners: [
      {
        config: 'Config D — gate ON + real fills + 25bps + mcap>=500cr (PIT proxy)',
        summary: 'The mcap floor is a RISK filter, not a return filter: -0.7pp CAGR for 14pp less drawdown vs config B. Calmar ~0.96, the same tier as research/75 momentum.',
        metrics: [
          { k: 'CAGR (median, net)', v: '30.4% [27.9..34.4]' },
          { k: 'MaxDD (median)', v: '-31.5% (worst -33.6%)' },
          { k: 'Worst years', v: '2008 -20.1, 2018 -13.4, 2016 -2.5' },
        ],
        rejected: [
          'The site\'s -11.4% risk claim (unreachable at any marking frequency)',
          'One-shot-per-pivot dedupe (collapsed performance: 2.1x, 32% win)',
          'Plain 252d RS (fails 9 of their own published trades)',
          'Single-path CAGRs >40% from this design (selection luck: seed spread 6.5-15.7x on 2020-25)',
        ],
      },
    ],
    caveats: [
      'SURVIVORSHIP: Kite lists only current instruments — 2006 coverage is 528 symbols, all of which survived to 2026. Pre-~2015 years (esp. 2006-07 at +43/+117%) are inflated by this. The DD and post-2015 years are the more trustworthy part.',
      'Mcap floor is a proxy: constant adjusted-shares from a 2026 yfinance snapshot (split-safe; wrong for heavy diluters), known for only 925/2,321 symbols — unknowns excluded.',
      'Selection among simultaneous signals is undisclosed by the site; all our numbers are 10-seed ensembles — trust the medians and ranges, not any single path.',
      'No STCG tax modelled (median hold is weeks — tax materially reduces net for a taxable account).',
      'Their published trade list remains only partially recallable (2-6/54 same-day matches) due to slot path-divergence, even though 48/51 of their trades pass every decoded condition.',
      '2025 in config D is +10.3% with the 2026 YTD tape negative — the smallcap-breadth regime this feeds on has cooled; paper-trade before capital.',
    ],
    projectPaths: [
      'research\\142_bananapatterns_replication\\BANANAPATTERNS_BLUESKY_TRADE_MATCH_DAILY_FORENSIC_STATUS.md',
      'research\\142_bananapatterns_replication\\scripts\\ (validate_trades, entry_diag, repair_data, extend_universe, bluesky_replay, make_report).py',
      'research\\142_bananapatterns_replication\\results\\RESULTS.md',
    ],
  },
