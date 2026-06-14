  {
    slug: 'factor-index-rotation',
    title: 'Nifty Factor-Index Rotation — does "diversify, don\'t select" transfer from assets to factors?',
    verdict:
      'Follow-on to the GTAA ETF study: does the "diversify, don\'t select" result transfer to the Nifty single factors (Momentum/Quality/Value/Low-Vol/Alpha)? No — on clean data the factors are ~0.8 correlated to each other and to the Nifty (mostly the same equity bet), so diversifying ACROSS factors fails (factor-only book Calmar 0.55). The value is purely swapping the SINGLE equity sleeve of the GTAA trio (Nifty → one factor) + Gold + Nasdaq, inverse-vol. Best clean sleeve = the VALUE factor: Calmar 1.83, CAGR 17.4%, MaxDD −9.5% (full 2015–26 window) — beating the Nifty book (1.50–1.57). Momentum is the higher-return alternative (20.0% / 1.60) but window-sensitive. Use ONE factor, not two. DATA-INTEGRITY NOTE: the Quality & Low-Vol Kite INDEX series were found corrupt (bad prints, 150%/308% daily vol) and excluded; the earlier "Low-Vol is the lone diversifier" claim is retracted (real Low-Vol ETF is 0.93-correlated to the Nifty). STRATEGY candidate — an incremental single-sleeve upgrade to research/63, not a standalone factor edge. ALL figures below are the consistent full 2015–26 window, inverse-vol, net 20 bps.',
    status: 'COMPLETE',
    date: '2026-06-14',
    cardBlurb:
      'Tests whether the GTAA "equal-weight beats selection" result extends to the Nifty factor indices. It doesn\'t — factors are ~0.8 correlated (the same Nifty bet), so factor-only books fail (Calmar 0.55). The win is swapping the single equity sleeve of the GTAA trio from Nifty to the Value factor, inverse-vol: Calmar 1.83 vs the Nifty book 1.50–1.57. Full 2015–26 window, net 20 bps.',
    cardStats: [
      { label: 'CAGR', value: '17.4%' },
      { label: 'MaxDD', value: '−9.5%' },
      { label: 'Calmar', value: '1.83' },
    ],
    system: {
      intro: 'Three families tested with the research/63 engine; the winner and why. All on the full 2015–26 window, inverse-vol, net 20 bps:',
      rows: [
        { k: 'Factor universe', v: 'Nifty200 Momentum 30, Nifty100 Quality 30, Nifty50 Value 20, Nifty100 LowVol 30, Nifty Alpha 50 (NSE index series). Quality & Low-Vol index series were CORRUPT (bad prints) → excluded; clean = Momentum/Value/Alpha.' },
        { k: '(a) Rotation', v: 'Top-N factors by momentum + trend gate — selection beats Nifty but is drawdown-bound (best ~0.67). Factors are too correlated for selection to add much.' },
        { k: '(b) Diversify across factors', v: 'Equal-weight / inverse-vol factor baskets FAIL: clean factor-only (Value+Momentum+Alpha) = Calmar 0.55, −24.9% DD — all the same equity beta.' },
        { k: '(c) Single-sleeve swap (WINNER)', v: 'Keep Gold + Nasdaq for diversification; swap the equity sleeve Nifty → the VALUE factor, inverse-vol, monthly. Calmar 1.83, CAGR 17.4%, MaxDD −9.5%.' },
        { k: 'Why it wins', v: 'The 1.5+ Calmar tier needs the cross-asset diversifiers (Gold+Nasdaq). Value is the lowest-vol/lowest-DD equity factor, so inverse-vol leans into it → the book\'s drawdown drops to −9.5%. Momentum is higher-return (20.0% / 1.60) but deeper DD. Adding ALL factors or a second factor dilutes the diversifiers.' },
        { k: 'Costs', v: '20 bps/side; low turnover. Synthetic 6%/yr cash leg where used.' },
      ],
    },
    conditions: {
      intro: 'Window and benchmark.',
      rows: [
        { k: 'Period', v: 'Full 2015–26 (Gold/Nasdaq ETF era; ~11.4y) for every combined-book figure on this page. Factor indices exist from 2010 but the combined book is gated by Gold/Nasdaq from 2015.' },
        { k: 'Benchmark', v: 'NIFTYBEES (Nifty 50) buy & hold; and the research/63 Nifty+Gold+Nasdaq equal-weight book.' },
        { k: 'Host', v: 'VPS market_data.db snapshot 2026-06-12 + Kite factor-index history; one canonical script (canonical.py) drives every number here.' },
      ],
    },
    comparisons: [
      {
        title: 'Equity sleeve in {sleeve + Gold + Nasdaq} — all 2015–26, inverse-vol, net 20 bps',
        caption: 'One window, one method, one source — directly comparable. Value is the best clean sleeve; Momentum the higher-return one; factor-only (no Gold/Nasdaq) collapses.',
        columns: ['Book', 'CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['Value + Gold + Nasdaq (WINNER)', '17.4%', '−9.5%', '1.83'],
          ['Momentum + Gold + Nasdaq', '20.0%', '−12.5%', '1.60'],
          ['Nifty + Gold + Nasdaq, equal (research/63)', '17.6%', '−11.3%', '1.57'],
          ['Alpha + Gold + Nasdaq', '20.8%', '−13.7%', '1.52'],
          ['Nifty + Gold + Nasdaq, inverse-vol', '16.8%', '−11.2%', '1.50'],
          ['Factor-only (Value+Momentum+Alpha, NO assets)', '13.8%', '−24.9%', '0.55'],
          ['Nifty 50 alone', '10.1%', '−28.8%', '0.35'],
        ],
        highlightRows: [0, 5],
        heatmap: false,
      },
      {
        title: 'Per-year: Value winner vs NIFTY 50 (2015–26)',
        columns: ['Year', 'Winner %', 'NIFTYBEES %', 'Excess pp'],
        rows: [
          ['2015', '+0.1', '−5.6', '+5.7'],
          ['2016', '+15.8', '+4.0', '+11.8'],
          ['2017', '+18.2', '+29.9', '−11.7'],
          ['2018', '−0.1', '+4.8', '−4.9'],
          ['2019', '+18.2', '+13.6', '+4.6'],
          ['2020', '+30.5', '+15.4', '+15.1'],
          ['2021', '+17.7', '+26.0', '−8.3'],
          ['2022', '−1.8', '+5.5', '−7.3'],
          ['2023', '+28.0', '+21.0', '+7.0'],
          ['2024', '+31.6', '+10.4', '+21.2'],
          ['2025', '+26.4', '+11.7', '+14.7'],
          ['2026*', '+10.4', '−9.3', '+19.7'],
        ],
        highlightRows: [0],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR', value: '17.4%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '11.4%' },
        { label: 'Excess / yr', value: '+6.0%', tone: 'pos' },
        { label: 'Sharpe', value: '1.53', tone: 'pos' },
        { label: 'Max Drawdown', value: '−9.5%', tone: 'neg', hint: 'vs NIFTYBEES −28.8%' },
        { label: 'Calmar', value: '1.83', tone: 'pos' },
        { label: 'vs research/63', value: '1.83 vs 1.57', hint: 'Value sleeve > Nifty sleeve' },
        { label: 'Mean factor corr', value: '~0.8', hint: 'clean; vs asset trio ~0.1' },
      ],
      tables: [
        {
          title: 'Winner vs research/63 GTAA (both 2015–26, inverse-vol vs equal)',
          columns: ['Metric', 'Value+Gold+Nasdaq (inv-vol)', 'Nifty+Gold+Nasdaq (equal)'],
          rows: [
            ['CAGR', '17.4%', '17.6%'],
            ['MaxDD', '−9.5%', '−11.3%'],
            ['Calmar', '1.83', '1.57'],
            ['Sharpe', '1.53', '1.45'],
          ],
          highlightRows: [1, 2],
        },
        {
          title: 'Factor cross-correlation (monthly, CLEAN data) — factors are mostly the same Nifty bet',
          caption: 'Quality & Low-Vol index series were CORRUPT and are excluded; clean factors are ~0.8 correlated to each other and to the Nifty — so diversifying across factors does not cut drawdown.',
          columns: ['', 'Mom', 'Value', 'Alpha', 'Nifty'],
          rows: [
            ['Momentum', '1.00', '0.77', '0.91', '0.84'],
            ['Value', '0.77', '1.00', '0.73', '0.89'],
            ['Alpha', '0.91', '0.73', '1.00', '0.80'],
            ['Nifty', '0.84', '0.89', '0.80', '1.00'],
          ],
          highlightRows: [],
          heatmap: true,
        },
        {
          title: 'Real factor-ETF check (2022-08→2026-06, ~3.8y BULL — read ranks, not levels)',
          caption: 'Pulled the real factor ETFs (max history) to test Low-Vol/Quality properly. Short bull window inflates all Calmars (3.5–4.9); the takeaways are the RANKING and that real Low-Vol = 13.6% vol & 0.93 corr to Nifty → NOT a diversifier (buries the corrupt-index claim). Factor-only stays poor: Calmar 0.61 vs 4.49 baseline.',
          columns: ['Equity sleeve (+Gold+Nasdaq, equal)', 'CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Value (MOVALUE)', '36%', '−7%', '4.92'],
            ['Low-Vol (LOWVOL1)', '29%', '−6%', '4.71'],
            ['Nifty (baseline)', '28%', '−6%', '4.49'],
            ['Quality (SBIETFQLTY)', '28%', '−7%', '3.95'],
            ['Momentum (MOMOMENTUM)', '29%', '−8%', '3.53'],
          ],
          highlightRows: [0],
        },
      ],
      charts: [
        {
          src: '/app/factor-gtaa-factsheet.png',
          caption:
            'CLIENT FACTSHEET — Factor GTAA: Value + Gold + Nasdaq (inverse-vol) vs NIFTY 50, full 2015–2026, net 20 bps. KPI strip, growth-of-₹1 (log), drawdown-vs-index, annual bars, monthly heatmap, rolling 12m, stat tables. 17.4% CAGR vs 11.4%, Calmar 1.83, MaxDD −9.5%. Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'Value factor + Gold + Nasdaq · inverse-vol · monthly (full 2015–26)',
        summary: 'The 1.5+ Calmar tier requires the cross-asset diversifiers (Gold+Nasdaq); given those, the Value factor is the best CLEAN equity sleeve (lowest vol/DD → inverse-vol leans into it → −9.5% book DD). Momentum is the higher-return alternative (20.0% / 1.60). Both beat the Nifty sleeve. Use one factor, not two.',
        metrics: [
          { k: 'CAGR', v: '17.4%' },
          { k: 'Excess', v: '+6.0%/yr vs NIFTYBEES' },
          { k: 'Sharpe', v: '1.53' },
          { k: 'MaxDD', v: '−9.5%' },
          { k: 'Calmar', v: '1.83' },
        ],
        rejected: [
          'Diversify across factors: clean factor-only (Value+Momentum+Alpha) = Calmar 0.55 — factors are ~0.8 correlated (the same Nifty bet), so equal-weighting them does not cut the −25% drawdown.',
          'Two factors instead of one: a second factor just adds correlated equity beta and crowds out Gold/Nasdaq — worse than a single factor sleeve.',
          'Sensex / BSE500 / Nifty500 as the sleeve: corr 0.97–1.00 to the Nifty — literally the same bet, no change. (Sector/thematic sleeves explored separately; the apparent defensive-sector winners are an overfit multiple-testing artifact — see EXPLORATORY_indices_sleeve.md, not published.)',
          'Quality & Low-Vol index series: CORRUPT (150%/308% daily vol) — excluded; the corrupt-data "Low-Vol diversifier" claim is retracted.',
        ],
      },
    ],
    caveats: [
      'DATA INTEGRITY (2026-06-14): the Kite INDEX series for Quality and Low-Vol were CORRUPT (bad prints — 150% / 308% annualised daily vol, single-day prints to +472%) and are excluded; the earlier "Low-Vol is the lone diversifier (0.42–0.47 corr)" claim is retracted (real Low-Vol ETF is 0.93-correlated to the Nifty). Also caught the Commodities index as corrupt.',
      'Period dependence: 2015–26 was a benign decade. The research/63 21-year through-cycle test (with proxies) showed the asset-trio Calmar drops from ~1.7 to ~0.8 and MaxDD widens to ~−24% in a real crisis (2008). Treat the 1.83 here similarly — the structural finding (Value sleeve > Nifty sleeve; factor diversification fails) is the durable part, not the absolute level.',
      'Window note: an earlier draft of this page featured Momentum on a 2016–26 window (which dropped a flat 2015 warm-up year) and showed 22.1% / Calmar 1.77. Every figure here is now the consistent full 2015–26 window from one canonical script; on that basis Momentum is 20.0% / 1.60 and Value (1.83) is the best clean sleeve.',
      'Mixed data: factor sleeves are PRICE-return indices (understate dividends ~1.5%/yr); Gold/Nasdaq are ETF prices. A live version must use the factor ETF NAV (MOVALUE etc., short history → recheck tracking/capacity).',
      'Backtest, net of 20 bps modelled cost, synthetic 6%/yr cash leg. Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS.md (verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/64_factor_index_rotation/results/RESULTS.md',
      },
      {
        label: 'canonical.py (single source of truth)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/64_factor_index_rotation/scripts/canonical.py',
      },
    ],
    projectPaths: [
      'research\\64_factor_index_rotation\\FACTOR_INDEX_ROTATION_MONTHLY_SWEEP_STATUS.md',
      'research\\64_factor_index_rotation\\scripts\\ (canonical, g1_probe, g2_sweep, replace_nifty_test, factor_etf_test).py',
      'research\\64_factor_index_rotation\\results\\ (factor_corr_CLEAN.csv, RESULTS.md, EXPLORATORY_indices_sleeve.md)',
    ],
  },
