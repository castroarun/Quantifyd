  {
    slug: 'factor-index-rotation',
    title: 'Nifty Factor-Index Rotation — does "diversify, don’t select" transfer from assets to factors?',
    verdict:
      'Follow-on to the GTAA ETF study: apply the same switching / equal-weight / risk-parity toolkit to the Nifty single-factor indices (Momentum, Quality, Value, Low-Vol, Alpha). It does NOT transfer — the factors are mostly the same bet: mean cross-correlation 0.65 and 0.79–0.91 vs the Nifty itself, so equal-weighting them tops out at Calmar 0.76 (best pure-factor book = Momentum+Low-Vol, 17.4% CAGR but −22.9% DD). The real win is COMBINING: use the strongest single factor (Momentum) as the equity sleeve inside the GTAA asset trio and weight by inverse-vol → Momentum + Gold + Nasdaq (inverse-vol), monthly = Calmar 1.77, CAGR 22.1%, MaxDD −12.5%, cost-insensitive. That marginally beats the GTAA Nifty book (1.75) by upgrading the equity sleeve and taming Nasdaq’s 24% vol. Piling in ALL factors dilutes it to 1.18. STRATEGY candidate — an incremental upgrade to research/63, not a standalone factor edge; factor selection/diversification alone is a SIGNAL.',
    status: 'COMPLETE',
    date: '2026-06-14',
    cardBlurb:
      'Tests whether the GTAA "equal-weight beats momentum-selection" result extends to the Nifty factor indices. It doesn’t — factors are too correlated (0.65) to diversify. But swapping the equity sleeve of the GTAA trio from Nifty to the Momentum factor, weighted inverse-vol, edges past it: Calmar 1.77 vs 1.75. Net 20bps; factors 2010–26, combined book 2016–26.',
    cardStats: [
      { label: 'CAGR', value: '22.1%' },
      { label: 'MaxDD', value: '−12.5%' },
      { label: 'Calmar', value: '1.77' },
    ],
    system: {
      intro: 'Three families tested with the research/63 engine; the winner and why:',
      rows: [
        { k: 'Factor universe', v: 'Nifty200 Momentum 30, Nifty100 Quality 30, Nifty50 Value 20, Nifty100 LowVol 30, Nifty Alpha 50 (NSE index series, 2010→2026).' },
        { k: '(a) Rotation', v: 'Top-N factors by ROC(3/6/12) + MA trend gate (does picking the leading factor pay?). Best Calmar 0.67 — selection beats Nifty but is drawdown-bound.' },
        { k: '(b) Baskets', v: 'Equal-weight & inverse-vol factor baskets. Best = Momentum+LowVol equal, Calmar 0.76 — Low-Vol is the only real in-set diversifier (0.42–0.47 corr).' },
        { k: '(c) Combined (WINNER)', v: 'Momentum factor + Gold + Nasdaq, weighted INVERSE-VOL, rebalanced monthly. = the GTAA trio with a better equity sleeve.' },
        { k: 'Why it wins', v: 'The 1.7-Calmar tier needs the cross-asset diversifiers (Gold+Nasdaq). Momentum > Nifty as the equity sleeve, and inverse-vol down-weights Nasdaq’s 24% vol. Adding ALL factors dilutes the diversifiers (1.18).' },
        { k: 'Costs', v: '20 bps/side; cost-insensitive (Calmar 1.80→1.75 across 0–40bps). Synthetic 6%/yr cash leg.' },
      ],
    },
    conditions: {
      intro: 'Windows and benchmark.',
      rows: [
        { k: 'Period', v: 'Factor-only families 2010–2026 (192m); combined book 2016–2026 (Gold/Nasdaq ETF era).' },
        { k: 'Benchmark', v: 'NIFTYBEES (Nifty 50) buy & hold; and the research/63 Nifty+Gold+Nasdaq equal-weight book.' },
        { k: 'Host', v: 'VPS market_data.db snapshot 2026-06-12 + Kite factor-index history; reproducible from committed scripts.' },
      ],
    },
    comparisons: [
      {
        title: 'The three families (best of each, net 20 bps)',
        columns: ['Family', 'Best config', 'CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['(c) Combined factor+asset', 'Momentum + Gold + Nasdaq, inverse-vol', '22.1%', '−12.5%', '1.77'],
          ['(b) Factor baskets', 'Momentum + Low-Vol, equal-weight', '17.4%', '−22.9%', '0.76'],
          ['(a) Factor rotation', 'top-3 blend + trend-gate-to-cash', '12.1%', '−18.1%', '0.67'],
          ['reference (research/63)', 'Nifty + Gold + Nasdaq, equal-weight', '19.7%', '−11.3%', '1.75'],
          ['benchmark', 'NIFTYBEES buy & hold', '10.1%', '−28.8%', '0.35'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
      {
        title: 'Per-year: winner vs NIFTY 50',
        columns: ['Year', 'Winner %', 'NIFTYBEES %', 'Excess pp'],
        rows: [
          ['2016', '+14.8', '+5.6', '+9.2'],
          ['2017', '+22.8', '+29.9', '−7.1'],
          ['2018', '−1.6', '+4.8', '−6.4'],
          ['2019', '+22.0', '+13.6', '+8.3'],
          ['2020', '+29.1', '+15.4', '+13.7'],
          ['2021', '+22.5', '+26.0', '−3.5'],
          ['2022', '−4.2', '+5.5', '−9.6'],
          ['2023', '+31.7', '+21.0', '+10.7'],
          ['2024', '+33.5', '+10.4', '+23.1'],
          ['2025', '+28.5', '+11.7', '+16.9'],
          ['2026*', '+17.9', '−9.3', '+27.2'],
        ],
        highlightRows: [0],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR', value: '22.1%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '10.1%' },
        { label: 'Excess / yr', value: '+12.0%', tone: 'pos' },
        { label: 'Sharpe', value: '1.68', tone: 'pos' },
        { label: 'Max Drawdown', value: '−12.5%', tone: 'neg', hint: 'vs NIFTYBEES −28.8%' },
        { label: 'Calmar', value: '1.77', tone: 'pos' },
        { label: 'vs research/63', value: '1.77 vs 1.75', hint: 'marginal upgrade' },
        { label: 'Mean factor corr', value: '0.65', hint: 'vs asset trio ~0.1' },
      ],
      tables: [
        {
          title: 'Winner vs research/63 GTAA',
          columns: ['Metric', 'Mom+Gold+Nasdaq (inv-vol)', 'Nifty+Gold+Nasdaq (EW)'],
          rows: [
            ['CAGR', '22.1%', '19.7%'],
            ['MaxDD', '−12.5%', '−11.3%'],
            ['Calmar', '1.77', '1.75'],
            ['Sharpe', '1.68', '1.57'],
          ],
          highlightRows: [0, 2, 3],
        },
        {
          title: 'Factor cross-correlation (monthly returns, 2010–26) — why diversification fails',
          columns: ['', 'Mom', 'Qual', 'Value', 'LowVol', 'Alpha'],
          rows: [
            ['Momentum', '1.00', '0.80', '0.76', '0.44', '0.90'],
            ['Quality', '0.80', '1.00', '0.85', '0.46', '0.76'],
            ['Value', '0.76', '0.85', '1.00', '0.42', '0.71'],
            ['LowVol', '0.44', '0.46', '0.42', '1.00', '0.42'],
            ['Alpha', '0.90', '0.76', '0.71', '0.42', '1.00'],
          ],
          highlightRows: [3],
          heatmap: true,
        },
      ],
      charts: [
        {
          src: '/app/factor-gtaa-factsheet.png',
          caption:
            'CLIENT FACTSHEET — Factor GTAA: Momentum + Gold + Nasdaq (inverse-vol) vs NIFTY 50, 2016–2026, net 20 bps. KPI strip, growth-of-₹1 (log), drawdown-vs-index, annual bars, monthly heatmap, rolling 12m, stat tables. 22.1% CAGR vs ~10%, Calmar 1.77, MaxDD −12.5%. Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'Momentum factor + Gold + Nasdaq · inverse-vol · monthly',
        summary: 'Best of 56 configs. The 1.7-Calmar tier requires the cross-asset diversifiers; given those, the Momentum factor is a better equity sleeve than plain Nifty and inverse-vol tames Nasdaq’s vol. Marginal over research/63 (1.77 vs 1.75) but structurally sound.',
        metrics: [
          { k: 'CAGR', v: '22.1%' },
          { k: 'Excess', v: '+12.0%/yr vs NIFTYBEES' },
          { k: 'Sharpe', v: '1.68' },
          { k: 'MaxDD', v: '−12.5%' },
          { k: 'Calmar', v: '1.77' },
        ],
        rejected: [
          'Pure factor diversification: equal-weight 5 factors = Calmar 0.76 — factors are 0.65 correlated, so equal-weighting them barely cuts the −23% drawdown. The research/63 "diversify > select" result does NOT transfer.',
          'Factor rotation/selection standalone: best 0.67 (top-3 + trend gate) — better than Nifty but drawdown-bound by equity beta.',
          'All-factors + Gold + Nasdaq: dilutes to Calmar 1.18 — too much correlated equity beta crowds out the diversifiers. Concentrate the equity sleeve into ONE factor.',
        ],
      },
    ],
    caveats: [
      'Period dependence (biggest): the combined book is 2016–26 — the same golden decade as research/63 (Nasdaq +24.6% INR, Momentum’s strong run). Treat ~22% CAGR as an upper bound; the low-DD/Calmar property is the robust takeaway.',
      'Mixed data: the Momentum sleeve is a PRICE-return index (understates dividends ~1.5%/yr, so its real edge over Nifty is larger); Gold/Nasdaq are ETF prices. A live version must use the Momentum ETF NAV (MOMOMENTUM / MOM30IETF, listed ~2022 → short history; recheck tracking/capacity).',
      'The improvement over research/63 (1.77 vs 1.75) is inside period-noise. The defensible claims are structural — factors don’t diversify; Momentum > Nifty as a sleeve; inverse-vol tames the book — not the 2nd-decimal Calmar.',
      '56 configs searched (multiple-testing): the winner sits on a stable plateau (any cross-asset combo with a momentum-ish equity sleeve lands ~1.6–1.8) but haircut the headline.',
      'Backtest, net of 20 bps modelled cost, synthetic 6%/yr cash leg. Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS.md (verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/64_factor_index_rotation/results/RESULTS.md',
      },
      {
        label: 'g2_sweep.py (sweep)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/64_factor_index_rotation/scripts/g2_sweep.py',
      },
    ],
    projectPaths: [
      'research\\64_factor_index_rotation\\FACTOR_INDEX_ROTATION_MONTHLY_SWEEP_STATUS.md',
      'research\\64_factor_index_rotation\\scripts\\ (g1_probe, g2_sweep, g2_finalize).py',
      'research\\64_factor_index_rotation\\results\\ (factor_corr.csv, g2_sweep.csv, RESULTS.md, tearsheet.png)',
    ],
  },
