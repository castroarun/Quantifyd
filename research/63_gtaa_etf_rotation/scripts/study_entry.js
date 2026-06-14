  {
    slug: 'gtaa-etf-rotation',
    title: 'GTAA Multi-Asset ETF Rotation — validating (and beating) the Upstox "Strategy 1"',
    verdict:
      'A trading-course slide pitched a monthly top-1 momentum rotation over 3 ETFs (Nifty / Gold / Nasdaq-100) at "Calmar 0.93". We could not reproduce 0.93 (Kite serves these ETFs only from 2015) and in all testable history the top-1 design is WEAK: Calmar 0.30 (raw) / 0.44 (trend-gated), −34%/−25% drawdown. The fix is almost embarrassingly simple — drop the momentum SELECTION and just hold all three EQUAL-WEIGHT, rebalanced monthly: Calmar ~1.73, CAGR 19.5%, MaxDD only −11.3%, turnover ~0, and completely cost-insensitive. Three uncorrelated sleeves (Nifty/Gold −0.08, Nifty/Nasdaq +0.25, Gold/Nasdaq +0.04) mean diversification, not rotation, is the edge. STRATEGY candidate for a simple low-DD core mandate — same ~1.7 Calmar tier as our concentrated equity books (research/41, /62) at a fraction of their drawdown, turnover, tax and complexity.',
    status: 'COMPLETE',
    date: '2026-06-14',
    cardBlurb:
      'Validate a popular trading-course GTAA strategy (top-1 of Nifty/Gold/Nasdaq-100 by 12m momentum, monthly) and try to beat it. Finding: the top-1 selection underperforms; naive equal-weight of the same 3 ETFs, monthly-rebalanced, more than doubles the Calmar (1.73 vs 0.30–0.44) at −11% drawdown. Net of 20bps, 2016–2026.',
    cardStats: [
      { label: 'CAGR', value: '19.5%' },
      { label: 'MaxDD', value: '−11.3%' },
      { label: 'Calmar', value: '1.73' },
    ],
    system: {
      intro: 'The slide’s rules (what we validated) and the winning construction (what beats it):',
      rows: [
        { k: 'Universe', v: 'NIFTYBEES (Nifty 50), GOLDBEES (gold), MON100 (Motilal Oswal Nasdaq-100) — three low-correlation sleeves.' },
        { k: 'Slide signal', v: 'Monthly: rank by ROC(12), hold the single top asset; "bullish" = close > 6-month MA.' },
        { k: 'Winner', v: 'Drop selection entirely — hold all 3 equal-weight (1/3 each), rebalance monthly. (= top-N where N = universe size, so ROC/MA become irrelevant.)' },
        { k: 'Why it wins', v: 'Monthly-return correlations: Nifty/Gold −0.08, Nifty/Nasdaq +0.25, Gold/Nasdaq +0.04. Equal-weight harvests the diversification + rebalancing premium; top-1 concentrates into the hot asset then eats the reversal.' },
        { k: 'Rotation', v: 'Monthly rebalance on the month-end close; returns realised t→t+1 (no same-bar leak).' },
        { k: 'Costs', v: '20 bps/side modelled (winner turnover ≈ 5%/yr → cost-insensitive: same Calmar at 0/10/20/40 bps).' },
        { k: 'Backtest window', v: '2016-02 → 2026-06 (~10.4y) — Kite serves these ETFs only from 2015, so the slide’s longer/older window is not reproducible.' },
      ],
    },
    conditions: {
      intro: 'Backtest window and benchmark.',
      rows: [
        { k: 'Period', v: 'Feb 2016 – Jun 2026 (~10.4 years), after a 12-month momentum warm-up.' },
        { k: 'Benchmark', v: 'NIFTYBEES (Nifty 50) buy & hold, same window.' },
        { k: 'Host', v: 'VPS market_data.db snapshot 2026-06-12; reproducible from committed scripts.' },
      ],
    },
    comparisons: [
      {
        title: 'Finalists — net 20 bps/side (2016–2026)',
        columns: ['Strategy', 'CAGR', 'MaxDD', 'Calmar', 'Sharpe', 'Turnover/yr'],
        rows: [
          ['Equal-weight 3-asset (WINNER)', '19.5%', '−11.3%', '1.73', '1.55', '0.05'],
          ['Equal-weight + trend filter (defensive)', '11.8%', '−8.5%', '1.40', '1.38', '2.16'],
          ['Momentum top-2 (gated) — best tactical', '16.4%', '−12.5%', '1.31', '1.34', '2.38'],
          ['Slide top-1 (trend-gated)', '10.9%', '−24.9%', '0.44', '0.62', '3.31'],
          ['Slide top-1 (raw)', '10.5%', '−34.4%', '0.30', '0.57', '2.83'],
          ['Benchmark: NIFTYBEES B&H', '10.0%', '−28.8%', '0.35', '—', '0'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
      {
        title: 'Per-year: winner vs NIFTY 50',
        columns: ['Year', 'Winner %', 'NIFTYBEES %', 'Excess pp'],
        rows: [
          ['2016', '+12.8', '+4.0', '+8.9'],
          ['2017', '+24.0', '+29.9', '−5.9'],
          ['2018', '+1.5', '+4.8', '−3.3'],
          ['2019', '+26.2', '+13.6', '+12.6'],
          ['2020', '+32.0', '+15.4', '+16.6'],
          ['2021', '+16.7', '+26.0', '−9.3'],
          ['2022', '−3.5', '+5.5', '−9.0'],
          ['2023', '+29.4', '+21.0', '+8.4'],
          ['2024', '+28.6', '+10.4', '+18.3'],
          ['2025', '+28.7', '+11.7', '+17.1'],
          ['2026*', '+12.8', '−9.3', '+22.1'],
        ],
        highlightRows: [0],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR', value: '19.5%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '10.0%' },
        { label: 'Excess / yr', value: '+9.5%', tone: 'pos' },
        { label: 'Sharpe', value: '1.55', tone: 'pos' },
        { label: 'Max Drawdown', value: '−11.3%', tone: 'neg', hint: 'vs NIFTYBEES −28.8%' },
        { label: 'Calmar', value: '1.73', tone: 'pos' },
        { label: 'Turnover / yr', value: '~5%', hint: 'cost-insensitive' },
        { label: 'Yrs beating index', value: '7 / 11' },
      ],
      tables: [
        {
          title: 'Strategy vs benchmark',
          columns: ['Metric', 'EqualWeight 3-ETF', 'NIFTYBEES'],
          rows: [
            ['CAGR', '19.5%', '10.0%'],
            ['Total return', '6.4x', '~2.7x'],
            ['Sharpe', '1.55', '~0.6'],
            ['Max Drawdown', '−11.3%', '−28.8%'],
            ['Calmar', '1.73', '0.35'],
          ],
          highlightRows: [0, 3, 4],
        },
        {
          title: 'Cost sensitivity — Calmar by cost (the winner barely trades)',
          columns: ['Strategy', '0 bps', '10 bps', '20 bps', '40 bps'],
          rows: [
            ['Equal-weight (WINNER)', '1.73', '1.73', '1.73', '1.73'],
            ['Equal-weight + trend filter', '1.73', '1.56', '1.40', '1.12'],
            ['Momentum top-2 (gated)', '1.53', '1.42', '1.31', '1.13'],
            ['Slide top-1 (gated)', '0.53', '0.48', '0.44', '0.35'],
          ],
          highlightRows: [0],
        },
      ],
      charts: [
        {
          src: '/app/gtaa-etf-rotation-factsheet.png',
          caption:
            'CLIENT FACTSHEET — GTAA Equal-Weight 3-Asset (Nifty + Gold + Nasdaq-100) vs NIFTY 50, 2016–2026, net of 20 bps. KPI strip, growth-of-₹1 (log), drawdown-vs-index, annual bars, monthly heatmap, rolling 12m, stat tables. 19.7% CAGR vs 14% (this view), 6.4x, Sharpe 1.04 (rf-adj), Calmar 1.75, MaxDD −11.3%, 55% of years beating the index. Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'Equal-weight · 3 ETFs (Nifty/Gold/Nasdaq-100) · monthly rebalance',
        summary: 'Best of a 108-cell sweep, and the SIMPLEST cell — no momentum selection, no parameters to fit. Diversification across three uncorrelated sleeves beats every momentum-rotation variant on Calmar at the lowest drawdown and near-zero turnover.',
        metrics: [
          { k: 'CAGR', v: '19.5%' },
          { k: 'Excess', v: '+9.5%/yr vs NIFTYBEES' },
          { k: 'Sharpe', v: '1.55' },
          { k: 'MaxDD', v: '−11.3%' },
          { k: 'Calmar', v: '1.73' },
        ],
        rejected: [
          'The slide’s top-1 momentum selection: concentrates into the hot asset and eats the reversal — Calmar 0.30 (raw) / 0.44 (gated), −34%/−25% DD.',
          'Adding more ETFs (Next-50, Bank): ext5 universe peaked at Calmar 0.83 — diluting the gold/Nasdaq diversification HURT.',
          'Momentum top-2/top-3 with a cash gate: all underperformed plain equal-weight on net Calmar; selection added nothing here.',
        ],
      },
    ],
    caveats: [
      'Period dependence (biggest): 2016–2026 was a golden decade for this trio — MON100’s +24.6% INR CAGR (Nasdaq mega-bull + ~3%/yr INR depreciation) carries much of the absolute return and will NOT repeat at that rate. The low-DD / diversification / Calmar property is the robust takeaway; treat ~19% CAGR as an upper bound, not a forecast (forward-realistic ≈ 10–13% CAGR, −15 to −20% DD in a less benign regime).',
      'No all-3 simultaneous crash in sample: 2008 isn’t testable (no data), COVID-2020 was V-shaped. A global risk-off hitting equity AND gold AND tech together is under-represented → real MaxDD could exceed −11.3%.',
      'MON100 capacity/regulatory: overseas-ETF flows hit RBI/SEBI caps in 2022 (creation halted, premium to NAV). At size the Nasdaq sleeve carries tracking/capacity risk.',
      'Single 11-year window, no true OOS / walk-forward — mitigated only by the winner being the zero-parameter, simplest config (no knife-edge to overfit; 108 configs searched).',
      'Backtest, net of 20 bps modelled cost. LIQUIDBEES price-return ≈0% (daily-dividend ETF) understates the defensive variant’s cash yield by ~6%/yr; the winner uses no cash leg so is unaffected. Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS.md (verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/63_gtaa_etf_rotation/results/RESULTS.md',
      },
      {
        label: 'gtaa_engine.py (engine)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/63_gtaa_etf_rotation/scripts/gtaa_engine.py',
      },
    ],
    projectPaths: [
      'research\\63_gtaa_etf_rotation\\GTAA_ETF_ROTATION_MONTHLY_SWEEP_STATUS.md',
      'research\\63_gtaa_etf_rotation\\scripts\\ (download_etfs, gtaa_engine, run_gtaa_sweep, finalists).py',
      'research\\63_gtaa_etf_rotation\\results\\ (gtaa_sweep.csv, finalists.csv, RESULTS.md, tearsheet.png)',
    ],
  },
