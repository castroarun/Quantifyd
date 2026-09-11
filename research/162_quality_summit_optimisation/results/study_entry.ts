  {
    slug: 'quality-summit-optimisation-research162',
    title: 'Quality Summit, optimised — and the quality screen tested inside Open Alpha · Base Age (research/162)',
    verdict:
      'PART A: CONCLUDED, NOTHING ADOPTED. PART B: NO EDGE — the 10-Oct-2026 review is closed with a NO. PART C: DILUTIVE — cash beats it on 360 of 360 paths. Arun asked two things: make Quality Summit earn more and fall less, and bring forward the overlay test research/160 had booked for 10-Oct-2026. THE HONEST ANSWER TO THE FIRST IS THAT IT COULD NOT BE DONE. Quality Summit is research/160’s Family-B book — near its own all-time high, profitable, mcap over Rs 1,000 crore, ROE and ROCE over 15, growth over 10, no debt test, fifteen names ranked by relative strength, monthly, no exit — 21.19% after tax, -37.1% drawdown, Calmar 0.58. r/160 had already swept 588 cells of exits, gates, slot counts and screen dials without beating it. This study attacked the three axes r/160 never tried: ATR-scaled trailing exits, ranking axes other than relative strength, and position sizing. THE BEST CELL LOOKED LIKE A DISCOVERY AND WAS NOT. Keeping the screen but widening the near-all-time-high band from 0.90 to 0.85 and cutting the book from fifteen names to ten, sized inverse-volatility, beat the incumbent by +6.06 points of CAGR and +0.282 Calmar ON 12 OF 12 REBALANCE OFFSETS in the fit window, on a verified 24-cell plateau, surviving the cost ladder. The holdout — pre-registered before any cell ran, and opened once, at the end — returned -3.48 points on 3 of 12 offsets and -0.152 Calmar on 1 of 12. Its holdout CAGR sits 9.22 points below its fit CAGR against a 4-point limit written down in advance. Over the whole window it buys 0.01 points of CAGR for 3.9 extra points of drawdown, and deleting its ten best trades takes its compounding proxy BELOW ONE. Its two plateau neighbours fail the same way, so this is the construction, not a sizing detail. THE ATR TRAILS DO NOT RESCUE THE BOOK EITHER. The best of six, SuperTrend(20,3), is worth +0.06 Calmar at -1.0 point of CAGR — and gets there partly by sitting 36% in cash. The fundamental-failure exit changes literally nothing. And the trail’s value is construction-dependent: helpful at fifteen slots, destructive at ten, which is the interaction trap in miniature. EVERY FUNDAMENTAL RANKING AXIS LOSES, by 2 to 22 points of CAGR — ranking the same qualifying names by profit growth, by margin slope, by a blend of either with relative strength, or by market cap, all subtract. Relative strength is the ranking. THE SCREEN’S REAL PRODUCT IS DRAWDOWN, NOT RETURN: paired against the same book on the same screenable universe it takes 12.7 to 16.9 points off the maximum drawdown on 12 of 12 offsets, and pays 1.3 points of CAGR for it. PART B ANSWERS THE 10-OCTOBER QUESTION AND THE ANSWER IS NO. Running research/161’s engine byte-identically — its no-mask control reproduces the published 21.26% / -34.80% / Calmar 0.618 exactly — and applying each screen to ENTRIES only, NOT ONE SCREEN WINS ON A SINGLE SEED OUT OF THIRTY on return, in either window, under either missing-data policy. Quality Summit’s own screen costs 9.7 points of CAGR; the screen as Arun originally wrote it costs 17.8. The mechanism is starvation: the screen cuts Base Age’s qualifying signals from 3,619 to 468, or to 76, and the invested fraction from 87% to 63%, or to 19%. PART C: added to the honest True North + Base Age pair at 10, 20 or 33%, Quality Summit is beaten by plain CASH at the same weight on 360 of 360 paths, and its monthly correlation to Base Age is 0.717 — it is the same family, sampled worse. NOTHING IS DEPLOYED, PAPERED OR CHANGED. The one operational consequence is subtraction: an October review slot is freed.',
    status: 'COMPLETE',
    date: '2026-09-12',
    cardBlurb:
      'Two questions in one night: can Quality Summit be improved, and does its quality screen help inside Open Alpha · Base Age? A 12-of-12 offset sweep with a verified plateau looked like a +6-point improvement — and the pre-registered holdout turned it into a 1-of-12 loss. The overlay test, booked for 10-Oct, loses on 0 of 30 seeds. Nothing adopted.',
    cardStats: [
      { label: 'Part A — optimisation', value: 'Fit +6.06pp on 12/12 → holdout −3.48pp on 3/12. NO ADOPTION' },
      { label: 'Part B — screen inside Base Age', value: '0 of 30 seeds win, every screen, both windows. NO EDGE' },
      { label: 'Part C — portfolio fit', value: 'Cash beats it 360/360 at every weight. DILUTIVE' },
    ],

    systemRules: {
      intro:
        'Three questions, three engines, one rule shared by all of them: every selection decision was made on the FIT window only, and the HOLDOUT was opened once, at the end, for the chosen cells. That rule — and the 4-point gap limit attached to it — was written into the STATUS document before a single cell ran, and it is the reason this study reports a kill instead of an improvement.',
      sharedCoreTitle: 'Shared core — identical across every cell',
      sharedCore: [
        { k: 'Window', v: '2018-08-01 → 2026-09-10. FIT window W1 = 2018-08-01 → 2022-06-30; HOLDOUT W2 = 2022-07-01 → 2026-09-10. Not a choice: four filed fiscal years do not exist for most names until FY2018 is filed in Aug-2018' },
        { k: 'Quality Summit book', v: 'close ≥ k × causal all-time-high close, 20-day median traded value ≥ ₹2 cr, the point-in-time b7 screen, top N by relative strength, monthly rebalance, next-open fill, no exit. Incumbent: k = 0.90, N = 15' },
        { k: 'Base Age book', v: 'research/161 unchanged — new all-time-high CLOSE, previous high ≥ 60 bars old, ≥ 20% base depth, 16 slots at 6.25% of NAV on ₹10 L, SuperTrend(14,4) close trail, no hard stop, 60-bar re-arm' },
        { k: 'Costs / tax / cash', v: '25 bps a side (ladder to 40 and 60), 20% STCG / 12.5% LTCG netted within the Indian FY with loss carry-forward, idle cash 5.0% (Quality Summit) and 5.5% (Base Age)' },
        { k: 'Ensemble', v: '12 rebalance-day offsets (Quality Summit) or 30 selection seeds (Base Age). Every A-vs-B number is PAIRED on the same offset or seed' },
        { k: 'Missing fundamentals', v: 'every masked cell run BOTH ways — missing = ineligible and missing = eligible — because the gap between them is the coverage bias measured rather than assumed' },
        { k: 'Engines', v: 'r/160’s engine, extended by 14 auditable exact-string patches (r/160’s own file left frozen), and a byte-identical md5-verified copy of r/161’s engine' },
        { k: 'Identity checks passed first', v: 'the Quality Summit incumbent reproduces r/160’s 21.19 / −37.07 / 0.58 to the second decimal; the Base Age control reproduces r/161’s 21.26 / −34.80 / 0.618 with worst seed 19.87' },
      ],
      riskLayer: {
        title: 'The pre-registered adoption bars — written before any cell ran',
        caption:
          'Deciding the bar after seeing the results is how sweeps lie. All three bars below, and the 4-point fit-to-holdout gap limit, were locked in the STATUS document at 23:40 IST on 11-Sep-2026, before the first cell.',
        columns: ['Test', 'Bar', 'Outcome'],
        rows: [
          ['A new Quality Summit spec', '≥ +0.15 Calmar OR ≥ +2pp CAGR at no worse drawdown, on ≥ 8 of 12 offsets, in W1 AND W2, on a plateau, surviving 40 bps', 'PASSED in W1 on 12/12. FAILED in W2 on 1/12 — not adopted'],
          ['A cell’s robustness', 'W2 CAGR may not fall more than 4pp below W1 CAGR', 'Candidate fell 9.22pp. Both plateau neighbours fell 7.4 and 9.7pp'],
          ['The screen as an overlay on Base Age', '≥ +0.10 Calmar or −3pp drawdown at ≥ equal CAGR, on ≥ 20 of 30 seeds, both windows', 'FAILED — no screen wins on any seed on return'],
          ['A portfolio addition', '+0.10 Calmar or −2pp drawdown at ≥ equal CAGR vs the pair, correlation < 0.40 to both legs, and beats the cash null', 'FAILED — correlation 0.72 to Base Age; cash wins 360/360'],
        ],
        highlightRows: [1],
      },
    },

    system: {
      intro:
        'research/160 swept 588 cells and found nothing that beat simply holding the Quality Summit book. It closed with one recommendation it could not test itself: try the quality screen INSIDE Open Alpha’s own entries rather than as a book of its own, and it booked that as a dated review for 10-Oct-2026. Arun asked for both — the optimisation and the October test — tonight. This study runs the axes r/160 never tried, and answers the October question five weeks early.',
      rows: [
        { k: 'What Arun asked', v: '“Can Quality Summit earn more and fall less? And does its quality screen help inside Open Alpha · Base Age? Do the October review now.”' },
        { k: 'Axis r/160 never tried #1', v: 'ATR-scaled trailing exits — SuperTrend at (7,3), (10,3), (14,4), (20,3) and chandelier at 22-day high minus 2 and 3 ATR14, each with and without a fundamental-failure exit' },
        { k: 'Axis r/160 never tried #2', v: 'ranking inside the qualifying set — 3-year profit growth, 3-year operating-margin slope, two cross-sectional composites with relative strength, and market cap, each at 8 / 10 / 15 names' },
        { k: 'Axis r/160 never tried #3', v: 'position sizing — inverse 60-day volatility, capped at 2× and floored at 0.25× the equal-weight target' },
        { k: 'Axis dropped, and why', v: 'the 25%-per-sector cap. There is no sector field anywhere in this project: not in fundamentals.db, not in the 2,116 cached Screener pages, not in holdings_meta.db. Stated rather than faked' },
        { k: 'Part B design', v: 'r/161’s engine unchanged; ONE line added to the event filter — a candidate all-time-high close is dropped unless its symbol passes the mask on the SIGNAL day. Exits, sizing, slot contention and the 60-bar re-arm untouched' },
        { k: 'Screens tested as overlays', v: 'none (control) · b7 (Quality Summit’s own) · b3 (quality only, no growth test) · growth_only (growth > 20, nothing else) · arun_strict (the screen exactly as Arun wrote it) — each under both missing-data policies' },
        { k: 'Cells run', v: '157 in Part A · 36 in Part B · 14 blend constructions in Part C. Disclosed for the multiple-testing haircut' },
      ],
    },

    conditions: {
      intro:
        'What the data can and cannot support, stated before the numbers rather than after them.',
      rows: [
        { k: 'Fundamentals coverage', v: 'Screener serves ~12 fiscal years, so a genuine 3-year growth rate needs four filed years, which most names first have in Aug-2018 (coverage steps 7% → 87% at that date). The window is not a choice' },
        { k: 'Holdout size', v: 'four years (2022-07 → 2026-09), dominated by the 2023-25 smallcap boom. A candidate could fail it for regime reasons rather than for being overfit — the pre-registered rule was applied as written regardless' },
        { k: 'Price universe', v: 'not point-in-time: market_data.db keeps 102 stopped series in 2,158 (4.7%), fewer than NSE actually delisted. Survivorship pressure is upward on EVERY arm, benchmarks included — which is why every headline here is a PAIRED number against a control carrying the identical bias' },
        { k: 'Split adjustment', v: 'not retroactive. The near-ATH state restarts its cummax on a one-day collapse below 0.55×, and this study’s new trail frames truncate a symbol after the last single-day fall worse than −40% (106 symbols). Both guards also fire on genuine crashes' },
        { k: 'Fundamentals are restated', v: 'the filing lag controls WHEN a fiscal year becomes visible; it cannot undo a later restatement. That residual look-ahead flatters every screened arm — including the ones that lost' },
        { k: 'Cash-yield inconsistency in Part C', v: 'True North carries idle cash at 6.5% (r/159), Base Age 5.5% (r/161), Quality Summit 5.0% (r/160). The pair is flattered by a few tenths of a point — i.e. the bias runs AGAINST the candidate' },
        { k: 'Drawdown convention', v: 'always from the running peak of the FULL curve, never from a window’s own first bar. Note the pair reads −19.9% on daily marks and −13.7% on monthly marks: both correct, different measurements' },
      ],
    },

    comparisons: [
      {
        title: 'Part A · A1 — the ATR-scaled trail family research/160 never tried (fit window)',
        caption:
          'On the incumbent construction, one axis changed at a time. The best trail buys 6.3 points of drawdown for 1.0 point of CAGR — and does it partly by sitting in cash, which is why the invested column belongs in the table.',
        columns: ['Exit', 'CAGR W1', 'MaxDD', 'Calmar', '% invested', 'trades/yr'],
        rows: [
          ['none — the incumbent', '20.11%', '−34.43%', '0.61', '85.1', '49.2'],
          ['SuperTrend(20, 3) close trail', '19.14%', '−28.15%', '0.67', '63.6', '72.6'],
          ['SuperTrend(10, 3) close trail', '19.28%', '−29.06%', '0.65', '65.1', '70.2'],
          ['SuperTrend(14, 4) — research/161’s own winner', '18.47%', '−33.79%', '0.56', '72.3', '59.3'],
          ['SuperTrend(7, 3) close trail', '17.47%', '−29.96%', '0.60', '64.6', '72.6'],
          ['Chandelier, 22-day high − 3×ATR14', '14.46%', '−26.91%', '0.52', '57.0', '90.1'],
          ['Chandelier, 22-day high − 2×ATR14', '4.74%', '−13.78%', '0.32', '34.0', '129.8'],
          ['fund_fail — sell when a name stops passing the screen', '20.11%', '−34.43%', '0.61', '85.1', '49.2'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Part A · A2 — ranking axes inside the qualifying set (fit window, after-tax CAGR)',
        caption:
          'Every fundamental ranking axis loses, by 2 to 22 points. Blending a fundamental into the relative-strength score dilutes it rather than sharpening it. Market cap is a shares-constant proxy and is quoted as one.',
        columns: ['Ranking axis', 'N = 8', 'N = 10', 'N = 15'],
        rows: [
          ['Relative strength — the incumbent', '21.80%', '21.78%', '20.11%'],
          ['3-year net-profit growth', '12.71%', '14.07%', '15.57%'],
          ['3-year operating-margin slope', '19.99%', '19.80%', '17.49%'],
          ['composite z(RS) + z(profit growth)', '15.49%', '16.83%', '16.92%'],
          ['composite z(RS) + z(margin slope)', '21.47%', '19.03%', '19.42%'],
          ['Market cap, largest first (proxy)', '−0.07%', '2.73%', '7.06%'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Part A · A5 — the plateau probe: near-ATH band k × number of names (fit window, after-tax CAGR)',
        caption:
          'A winner whose neighbours disagree is noise, so the band was swept before anything was called a result. Every neighbour of the k = 0.85 / N = 10 cell sits inside ±3 points — a genuine plateau. It still did not survive the holdout.',
        columns: ['k \\ N', '8', '10', '12', '15'],
        rows: [
          ['0.80', '21.75', '23.44', '23.68', '21.43'],
          ['0.825', '22.51', '23.00', '24.09', '22.17'],
          ['0.85', '24.13', '24.48', '23.53', '21.80'],
          ['0.875', '21.65', '22.31', '21.92', '21.35'],
          ['0.90 (incumbent band)', '21.80', '21.78', '21.50', '20.11'],
          ['0.95', '20.03', '20.53', '19.81', '17.53'],
        ],
        heatmap: true,
        highlightRows: [2],
      },
      {
        title: 'Part A — the holdout, opened once, as pre-registered',
        caption:
          'The rule written before the run: a cell whose holdout CAGR falls more than 4 points below its fit CAGR is declared not robust. All three finalists fail it, and they fail it together — so it is the construction, not the sizing.',
        columns: ['Book', 'W1 fit', 'W2 holdout', 'gap', 'full window'],
        rows: [
          ['Quality Summit incumbent (k 0.90, N 15)', '20.11% / −34.4 / 0.61', '20.95% / −35.5 / 0.59', '+0.84', '21.19% / −37.1 / 0.58'],
          ['QS-v2 (k 0.85, N 10, inverse-vol)', '25.59% / −30.9 / 0.83', '16.37% / −40.9 / 0.41', '−9.22', '21.43% / −40.9 / 0.53'],
          ['QS-v2 equal-weight', '24.48% / −32.0 / 0.80', '14.78% / −42.2 / 0.35', '−9.70', '20.91% / −41.7 / 0.50'],
          ['QS-v2 plateau neighbour (k 0.825, N 12)', '24.09% / −32.8 / 0.74', '16.67% / −42.0 / 0.43', '−7.42', '21.24% / −41.8 / 0.52'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Part A — the paired test, same rebalance offset on both sides',
        caption:
          'A 12-of-12 sweep in the fit window and a 1-of-12 rout in the holdout is the signature of a fit to one leg of the market, not of an edge. Unpaired medians would have hidden it.',
        columns: ['QS-v2 vs the incumbent', 'ΔCAGR', 'CAGR wins', 'ΔCalmar', 'Calmar wins', 'ΔMaxDD', 'DD wins'],
        rows: [
          ['W1 — the fit window', '+6.06pp', '12/12', '+0.282', '12/12', '+4.94pp', '10/12'],
          ['W2 — the holdout', '−3.48pp', '3/12', '−0.152', '1/12', '−4.43pp', '0/12'],
          ['Full window', '+0.01pp', '6/12', '−0.055', '2/12', '−3.93pp', '1/12'],
        ],
        highlightRows: [1],
      },
      {
        title: 'Part A — does the SCREEN add value? Paired against the same book on the screenable sub-universe',
        caption:
          'The honest control is every name with four filed fiscal years and nothing else required, because it carries the identical survivorship bias. The first row reproduces research/160’s published paired result exactly.',
        columns: ['Construction', 'ΔCAGR', 'wins', 'ΔCalmar', 'wins', 'ΔMaxDD', 'wins'],
        rows: [
          ['Incumbent (k 0.90, N 15) — full window', '−1.32pp', '2/12', '+0.106', '10/12', '+12.74pp', '12/12'],
          ['Candidate (k 0.85, N 10) — full window', '+4.75pp', '9/12', '+0.244', '11/12', '+16.94pp', '12/12'],
        ],
      },
    ],

    results: {
      metrics: [
        { label: 'QS incumbent (unchanged)', value: '21.19%', hint: 'after tax, −37.1% DD, Calmar 0.58, 12 offsets' },
        { label: 'Candidate — fit window', value: '+6.06pp', hint: '12 of 12 offsets, Calmar +0.282', tone: 'pos' },
        { label: 'Candidate — holdout', value: '−3.48pp', hint: '3 of 12 offsets, Calmar −0.152 on 1 of 12', tone: 'neg' },
        { label: 'Base Age, no screen', value: '26.57%', hint: '2018-08→2026-09, −26.6% DD, Calmar 0.999, 30 seeds' },
        { label: 'Base Age + the b7 screen', value: '16.79%', hint: '−9.71pp paired, 0 of 30 seeds win', tone: 'neg' },
        { label: 'Blend value vs the pair', value: '0.001', hint: 'ΔCalmar at 10%; cash at the same weight gives +0.052 on 360/360', tone: 'neg' },
      ],
      tables: [
        {
          title: 'Part B — the quality screen as an ENTRY filter inside Open Alpha · Base Age (2018-08 → 2026-09, 30 seeds)',
          caption:
            'research/161’s engine byte-identical; one line added to the event filter. The invested and events columns are the mechanism: the screen does not pick better names, it starves the book of signals.',
          columns: ['Entry filter', 'CAGR', 'worst seed', 'MaxDD', 'Calmar', '% invested', 'events', 'trades/yr'],
          rows: [
            ['none — Base Age as it stands', '26.57%', '23.07%', '−26.57%', '0.999', '87.0', '3,619', '41.9'],
            ['+ b7 (Quality Summit’s own), missing = fail', '16.79%', '16.78%', '−22.39%', '0.750', '63.3', '468', '30.0'],
            ['+ b7, missing = pass', '16.38%', '16.35%', '−21.72%', '0.754', '63.6', '1,148', '30.2'],
            ['+ b3 (quality only, no growth test), fail', '15.02%', '14.59%', '−30.69%', '0.491', '76.0', '771', '35.5'],
            ['+ b3, pass', '15.70%', '15.34%', '−31.02%', '0.506', '76.5', '1,451', '35.7'],
            ['+ growth_only (growth > 20, nothing else), fail', '20.77%', '20.77%', '−20.50%', '1.013', '53.9', '388', '24.5'],
            ['+ growth_only, pass', '21.60%', '21.59%', '−20.95%', '1.031', '54.4', '1,068', '24.8'],
            ['+ arun_strict (the screen as written), fail', '8.80%', '8.80%', '−10.10%', '0.872', '18.8', '76', '9.2'],
            ['+ arun_strict, pass', '11.08%', '11.07%', '−12.87%', '0.861', '24.9', '756', '12.0'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Part B — the paired test, same selection seed on both sides, against the no-mask control',
          caption:
            'Not one screen wins on a single seed out of thirty on return, in any window, under either missing-data policy. growth_only gets within touching distance on Calmar in the full sub-window and is negative on 0 of 30 in the fit window — that is not a pass on any reading of the bar.',
          columns: ['Overlay', 'Window', 'ΔCAGR', 'CAGR wins', 'ΔCalmar', 'Calmar wins', 'Verdict'],
          rows: [
            ['b7, missing = fail', '2018-08 →', '−9.71pp', '0/30', '−0.248', '0/30', 'FAIL'],
            ['b7, missing = fail', 'W1 fit', '−16.16pp', '0/30', '−0.407', '0/30', 'FAIL'],
            ['b7, missing = fail', 'W2 holdout', '−3.58pp', '2/30', '−0.037', '9/30', 'FAIL'],
            ['b7, missing = pass', '2018-08 →', '−10.09pp', '0/30', '−0.245', '0/30', 'FAIL'],
            ['b3 quality-only, fail', '2018-08 →', '−11.28pp', '0/30', '−0.478', '0/30', 'FAIL'],
            ['growth_only, fail', '2018-08 →', '−5.80pp', '0/30', '+0.014', '19/30', 'FAIL'],
            ['growth_only, pass', '2018-08 →', '−4.84pp', '0/30', '+0.039', '22/30', 'FAIL (W1 −0.121 on 0/30)'],
            ['arun_strict, fail', '2018-08 →', '−17.77pp', '0/30', '−0.127', '0/30', 'FAIL'],
          ],
        },
        {
          title: 'Part C — blend value against the honest pair (True North + Base Age 50-50, monthly, 360 paths)',
          caption:
            'research/160 ran this against the PUBLISHED Open Alpha, which research/159 has since shown rests on a same-bar look-ahead fill. Re-run here against the pair Arun’s money is actually in. The cash null is the row that decides it.',
          columns: ['Book', 'CAGR', '[min..max]', 'MaxDD', 'Calmar', 'ΔCalmar vs pair', 'paths improved'],
          rows: [
            ['TN + Base Age 50-50 — the honest pair', '24.42', '[19.00..27.37]', '−13.71', '1.769', '—', '—'],
            ['+ QS incumbent at 10%', '24.12', '[19.48..27.11]', '−13.98', '1.742', '+0.001', '181/360'],
            ['+ QS incumbent at 20%', '23.74', '[19.93..26.80]', '−15.03', '1.565', '−0.188', '70/360'],
            ['+ QS incumbent at 33%', '23.15', '[20.47..26.35]', '−17.06', '1.357', '−0.389', '27/360'],
            ['+ QS-v2 at 10%', '24.36', '[19.37..27.41]', '−13.48', '1.834', '+0.054', '237/360'],
            ['+ QS-v2 at 20%', '24.31', '[19.70..27.38]', '−14.13', '1.717', '−0.064', '125/360'],
            ['+ QS-v2 at 33%', '24.02', '[20.07..27.26]', '−17.06', '1.383', '−0.343', '8/360'],
            ['+ CASH at 10% — the null', '22.49', '[17.67..25.11]', '−12.21', '1.826', '+0.052', '360/360'],
            ['+ CASH at 20% — the null', '20.55', '[16.32..22.86]', '−10.74', '1.900', '+0.118', '360/360'],
            ['+ CASH at 33% — the null', '18.04', '[14.55..19.94]', '−8.87', '2.040', '+0.238', '360/360'],
            ['Base Age standalone', '25.40', '[22.28..27.43]', '−19.46', '1.317', '−0.434', '34/360'],
            ['True North standalone', '21.58', '[14.51..25.60]', '−16.03', '1.380', '−0.431', '0/360'],
            ['QS incumbent standalone', '21.02', '[17.04..23.77]', '−31.10', '0.671', '−1.135', '0/360'],
          ],
          highlightRows: [7, 8, 9],
        },
        {
          title: 'Part C — correlation, and the stress windows',
          caption:
            'A complement wants to sit below ~0.40 monthly. Quality Summit sits at 0.72 against Base Age: a near-all-time-high momentum book added to a book of all-time-high breakouts. The two legs already in the pair are 0.337 to each other.',
          columns: ['Pair', 'Monthly correlation', 'Daily correlation'],
          rows: [
            ['QS incumbent vs Base Age', '0.717', '0.671'],
            ['QS-v2 vs Base Age', '0.651', '0.649'],
            ['QS incumbent vs True North', '0.374', '0.440'],
            ['True North vs Base Age', '0.337', '0.430'],
          ],
        },
        {
          title: 'Part C — stress windows: return % (intra-window drawdown from the full-curve peak)',
          caption:
            'The pair’s known weakness is the grind, and this sleeve makes it three points worse. It does not earn in the grind and it does not cushion the crash — it deepens both.',
          columns: ['Window', 'The pair', '+ QS incumbent 20%', '+ QS-v2 20%', '+ CASH 20%'],
          rows: [
            ['2020 crash (Feb–Apr)', '−10.0 (−10.0)', '−10.2 (−11.1)', '−9.1 (−10.4)', '−7.8 (−7.8)'],
            ['2022H1 grind (Jan–Jun)', '−11.8 (−11.8)', '−14.8 (−14.9)', '−12.2 (−12.6)', '−9.1 (−9.1)'],
          ],
        },
        {
          title: 'Year on year — return with the intra-year drawdown beneath, medians across the ensemble',
          caption:
            'House format. Each cell is the calendar-year return with the worst intra-year drawdown in parentheses, measured from the running peak of the FULL curve. Two things the summary row hides: QS-v2 beats the incumbent in 2021 and 2022 — both inside the fit window — and loses in 2023, 2024 and 2025, all inside the holdout; and the b7 overlay’s best-looking years on Base Age are the years it was barely invested.',
          columns: ['Year', 'QS incumbent', 'QS-v2 (fit winner)', 'Base Age', 'Base Age + b7 overlay', 'True North', 'TN + BA pair', 'NIFTYBEES'],
          rows: [
            ['2018', '−12.2 (−18.7)', '−13.1 (−22.7)', '−10.2 (−16.3)', '−0.6 (−2.0)', '−5.7 (−9.6)', '−8.2 (−12.4)', '−3.8 (−14.1)'],
            ['2019', '+24.3 (−16.2)', '+26.3 (−19.0)', '+30.1 (−16.6)', '+16.0 (−4.9)', '+1.1 (−11.6)', '+14.9 (−12.5)', '+13.6 (−10.5)'],
            ['2020', '+39.8 (−33.4)', '+38.8 (−30.0)', '+48.8 (−15.5)', '+26.4 (−12.7)', '+58.4 (−12.3)', '+53.8 (−11.4)', '+15.4 (−36.3)'],
            ['2021', '+87.2 (−14.0)', '+89.1 (−16.7)', '+83.7 (−10.9)', '+18.3 (−10.0)', '+57.9 (−12.4)', '+70.1 (−7.2)', '+26.0 (−9.5)'],
            ['2022', '−19.1 (−33.2)', '+4.1 (−28.1)', '−2.5 (−26.6)', '−10.6 (−20.0)', '+6.0 (−16.2)', '+2.4 (−17.3)', '+5.5 (−16.1)'],
            ['2023', '+59.4 (−28.0)', '+45.2 (−23.0)', '+51.3 (−17.5)', '+63.0 (−17.6)', '+47.9 (−12.2)', '+50.2 (−10.6)', '+21.0 (−9.7)'],
            ['2024', '+35.9 (−14.0)', '+17.7 (−16.5)', '+5.6 (−24.8)', '+4.2 (−22.4)', '+25.1 (−18.6)', '+15.0 (−19.9)', '+10.4 (−10.5)'],
            ['2025', '−16.1 (−34.9)', '−17.8 (−40.3)', '+1.9 (−16.6)', '−0.1 (−19.9)', '+2.3 (−19.7)', '+3.4 (−11.5)', '+11.7 (−15.2)'],
            ['2026 (to Sep)', '+14.6 (−34.4)', '+20.3 (−38.4)', '+34.6 (−20.0)', '+34.8 (−14.3)', '+4.7 (−14.9)', '+18.1 (−11.8)', '−9.5 (−14.8)'],
            ['CAGR / MaxDD / Calmar', '21.2 / −37.1 / 0.58', '21.4 / −40.9 / 0.53', '26.2 / −26.6 / 0.99', '16.8 / −22.4 / 0.75', '21.4 / −21.2 / 1.01', '24.4 / −19.9 / 1.24', '10.6 / −36.3 / 0.29'],
          ],
          highlightRows: [9],
        },
        {
          title: 'Robustness and the tradeability gate — full window, after tax',
          caption:
            'Both books survive the cost ladder with the same shallow slope and neither is a cash-yield artefact. A 17-trade losing streak on a book that trades 38-61 times a year is four to six months of nothing but losers. Multiply the capacity ratio by the real book size before reading it as a constraint.',
          columns: ['Book', '25 bps', '40 bps', '60 bps', '0% cash yield', 'win %', 'expectancy/trade', 'max losing streak', 'trades/yr', 'capacity at ₹1 cr'],
          rows: [
            ['QS incumbent', '21.19%', '20.25%', '19.10%', '21.01%', '46.3', '+7.26%', '17', '61.4', '0.7%'],
            ['QS-v2', '21.43%', '20.54%', '19.19%', '21.15%', '45.9', '+9.00%', '17', '38.3', '1.2%'],
            ['Screenable-sub-universe control', '17.46%', '—', '—', '—', '42.4', '+5.79%', '16', '56.4', '1.4%'],
          ],
        },
        {
          title: 'Outlier dependence — trade level, one path, net of costs',
          caption:
            'The candidate’s compounding proxy falls BELOW ONE when its ten best trades are deleted: without ten names it loses money. The incumbent’s stands at 355×. This is the second, independent reason not to adopt it.',
          columns: ['Book', 'trades', 'mean/trade', 'full', 'ex top-10', 'ratio', 'capped at +50%'],
          rows: [
            ['QS incumbent', '520', '+6.02%', '2.11e6', '355', '5,938×', '2.42e3'],
            ['QS-v2', '332', '+6.62%', '4.60e3', '0.636 — LOSES MONEY', '7,234×', '10'],
            ['QS-v2 equal-weight', '334', '+6.63%', '5.62e3', '0.777 — LOSES MONEY', '7,234×', '12.3'],
            ['Screenable-sub-universe control', '455', '+5.73%', '444', '0.017', '26,280×', '0.011'],
          ],
          highlightRows: [1],
        },
      ],
      charts: [
        {
          src: '/app/quality-summit-optimisation-research162.png',
          caption:
            'Log growth of 100, after tax and net of costs, 2018-08 to 2026-09, with the drawdown panel beneath. Quality Summit’s incumbent and the fit-window winner sit almost on top of each other over the full period — the candidate’s +6-point fit-window lead is entirely given back in the holdout. Open Alpha · Base Age runs above both with materially less drawdown; the b7 overlay on Base Age is the flat line underneath it, which is what starving a 16-slot book of 87% of its signals looks like. The True North + Base Age pair is the line with the shallowest drawdown panel.',
        },
        {
          src: '/app/quality-summit-optimisation-research162-tearsheet.png',
          caption:
            'Client factsheet for the book that survived this study unchanged: Quality Summit as research/160 published it — near its own all-time high at k = 0.90, profitable, market cap over ₹1,000 crore, ROE and ROCE over 15, growth over 10, no debt test, fifteen names ranked by relative strength, rebalanced monthly, no exit rule.',
        },
      ],
    },

    winners: [
      {
        config: 'Nothing is adopted. Quality Summit keeps the spec research/160 published, and the 10-Oct-2026 overlay review is closed with a NO.',
        summary:
          'The optimisation found a real-looking improvement and the pre-registered holdout destroyed it; the overlay test lost on every seed; the portfolio test lost to cash on every path. The incumbent is, notably, the most window-stable book in the study — 20.11% in the fit window, 20.95% in the holdout — and that stability is exactly why nothing beat it out of sample.',
        metrics: [
          { k: 'Quality Summit, unchanged', v: '21.19% after tax / −37.1% DD / Calmar 0.58 / 91% invested' },
          { k: 'The fit-window winner', v: '+6.06pp CAGR and +0.282 Calmar on 12/12 offsets — and −3.48pp on 3/12 in the holdout' },
          { k: 'Base Age, unchanged', v: '26.57% / −26.6% / Calmar 0.999 on 2018-08→2026-09; every screen overlay loses on 0 of 30 seeds' },
          { k: 'What the screen IS worth', v: '12.7 to 16.9 points off the maximum drawdown on 12 of 12 offsets, for 1.3 points of CAGR — insurance, not an edge' },
          { k: 'Operational consequence', v: 'subtraction only: the October review slot is freed. No engine, book, register or page changes' },
        ],
        rejected: [
          'QS-v2 (k 0.85, ten names, inverse-vol) — the best cell in the fit window; holdout CAGR 9.22 points below fit against a pre-registered 4-point limit, and it loses money without its ten best trades',
          'Both of QS-v2’s plateau neighbours — they fail the holdout the same way, which is what proves it is the construction and not the sizing',
          'All four SuperTrend close trails and both chandelier trails on Quality Summit — the best is worth +0.06 Calmar at −1.0pp CAGR, and part of that is idle cash',
          'The fundamental-failure exit — changes literally nothing; every paired row is identical',
          'All five alternative ranking axes, including both composites with relative strength — they subtract 2 to 22 points of CAGR',
          'The NIFTYBEES 100-week index gate on Quality Summit — it lowers return at both constructions',
          'Inverse-volatility sizing as a standalone change — worth +0.02 Calmar on the incumbent',
          'Every quality screen as an entry filter inside Open Alpha · Base Age, under both missing-data policies',
          'Adding Quality Summit to the True North + Base Age pair at 10%, 20% or 33% — cash at the same weight wins on 360 of 360 paths',
          'The 25%-per-sector cap — not tested, because no sector data exists anywhere in this project. Said rather than faked',
        ],
      },
    ],

    caveats: [
      'EIGHT YEARS IS SHORT AND THE HOLDOUT IS FOUR OF THEM, dominated by the 2023-25 smallcap boom. The window cannot be extended — Screener serves about twelve fiscal years, so four filed years do not exist for most names until FY2018 is filed in Aug-2018, and coverage steps from 7% to 87% at exactly that date. A candidate could in principle fail this holdout for regime reasons rather than for being overfit. The pre-registered rule was applied as written regardless, but a four-year holdout is a four-year holdout and that is the single biggest limitation on this page.',
      'THE PRICE UNIVERSE IS NOT POINT-IN-TIME. market_data.db keeps only 102 stopped series in 2,158 (4.7%) across eleven years, fewer than NSE actually delisted or suspended. Survivorship pressure is upward on EVERY arm, benchmarks included. That is why every headline here is a PAIRED number against a control carrying the identical bias, rather than a standalone level.',
      'THE DATABASE IS NOT RETROACTIVELY SPLIT-ADJUSTED. The near-all-time-high state restarts its cummax on a one-day collapse below 0.55×, and this study’s new SuperTrend and chandelier frames truncate a symbol’s series after the last single-day fall worse than −40% (106 symbols affected). Both guards also fire on genuine crashes, which makes the near-ATH state EASIER to satisfy for those names. Direction stated, not hidden.',
      'THE FUNDAMENTALS ARE RESTATED, NOT AS-REPORTED. The four-month filing lag controls when a fiscal year becomes visible; it cannot undo a later restatement. That residual look-ahead flatters every screened arm — including the ones that lost.',
      'THE MARKET-CAP RANKING AXIS IS A PROXY — research/142’s shares-constant snapshot, today’s share count back-projected on historical price. Its −0.07% result is so far from everything else that the proxy cannot be the explanation, but it is a proxy and is labelled one.',
      'THE NEW TRAIL FRAMES USE FORWARD-FILL-WITHIN-SPAN, so a missing session counts as a repeated bar inside the ATR window rather than being skipped. Holes are rare after the phantom-row purge and the effect is immaterial, but it is a choice and not the strict dropna-and-reindex recipe.',
      '193 CELLS WERE RUN — 157 in Part A, 36 in Part B — plus 14 Part-C blend constructions. Discount the fit-window winner accordingly, which is precisely what the holdout did. The count is disclosed so a reader can apply the haircut rather than take our word that one was applied.',
      'THE PART-C CASH-YIELD INCONSISTENCY IS REAL AND UNCORRECTED. True North’s curve carries idle cash at 6.5% a year (research/159’s convention), Base Age at 5.5% (research/161) and Quality Summit at 5.0% (research/160). The pair is therefore flattered by a few tenths of a point relative to the candidate. The bias runs AGAINST the candidate, which is the safe direction, and the curves are other studies’ artefacts so they were not rewritten.',
      'DRAWDOWN CONVENTION MATTERS HERE. The pair reads −19.9% on daily marks in the year table and −13.7% on monthly marks in the blend table. Both are correct and they are different measurements: blend tables in this project are built on monthly returns (the research/154 convention), year tables on daily curves. Quote the daily number when someone asks what the book felt like to hold.',
      'WHAT WAS NOT TESTED, AND WHY: a sector cap (no sector field exists in fundamentals.db, in the 2,116 cached Screener pages, or in holdings_meta.db); a point-in-time index-membership universe (not reconstructable from our data); intraday or stop-based exits (the Quality Summit engine is close-only by construction, which is also why research/142’s trigger/fill trap cannot be expressed in it at all); and screens built from quarterly figures (Screener carries about thirteen quarters, so they only exist from mid-2023).',
    ],

    reports: [
      { label: 'RESULTS.md — the full verdicts', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/results/RESULTS.md' },
      { label: 'STATUS doc — pre-registration, live log, crash recovery', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/QUALITY_SUMMIT_OPTIMISATION_DAILY_SWEEP_STATUS.md' },
      { label: 'Year-on-year table (house format)', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/results/yoy162.md' },
      { label: 'Part C — correlation and blend value', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/results/partC_blend.md' },
      { label: 'research/160 — the study this one optimises', href: '/app/backtest/quality-growth-near-ath-research160' },
    ],

    githubLinks: [
      { label: 'patch_engine.py — the 14 auditable patches on research/160’s frozen engine', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/scripts/patch_engine.py' },
      { label: 'build_aux.py — SuperTrend / chandelier / volatility / point-in-time ranking frames', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/scripts/build_aux.py' },
      { label: 'partb_overlay.py — the screen as an entry filter inside Base Age', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/scripts/partb_overlay.py' },
      { label: 'finalize_a.py — the holdout, the paired tests and the outlier deletion', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/scripts/finalize_a.py' },
      { label: 'partc_blend.py — correlation and blend value vs the honest pair', href: 'https://github.com/castroarun/quantifyd/blob/main/research/162_quality_summit_optimisation/scripts/partc_blend.py' },
    ],

    projectPaths: [
      'research/162_quality_summit_optimisation/QUALITY_SUMMIT_OPTIMISATION_DAILY_SWEEP_STATUS.md',
      'research/162_quality_summit_optimisation/results/RESULTS.md',
      'research/162_quality_summit_optimisation/results/cells_a.csv',
      'research/162_quality_summit_optimisation/results/partA_final.csv',
      'research/162_quality_summit_optimisation/results/partA_paired.csv',
      'research/162_quality_summit_optimisation/results/partB_cells.csv',
      'research/162_quality_summit_optimisation/results/partB_paired.csv',
      'research/162_quality_summit_optimisation/results/partC_blend.md',
      'research/162_quality_summit_optimisation/results/yoy162.md',
      'research/162_quality_summit_optimisation/scripts/',
    ],
  },
