"""research/154 - publish the study to the React app registry (frontend-only, no restart).

Inserts one BacktestStudy object at the END of BACKTEST_STUDIES in
frontend/src/data/backtests.ts and copies the figure into frontend/public/.
Idempotent: if the slug is already present the file is left untouched.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
TS = ROOT / "frontend/src/data/backtests.ts"
RES = ROOT / "research/154_multi_system_blends/results"
SLUG = "multi-system-blends-research154"

ENTRY = r"""  {
    slug: 'multi-system-blends-research154',
    title: 'Six sleeves, every combination — the correlation and blend matrix',
    verdict:
      'STRATEGY (candidate) — the deployed TN+OA pair is under-diversified, and the fix is TWO satellites, not a third breakout sleeve. Three findings, in order of how much they should change what we do. FIRST, A CORRECTION: the deployed pair’s worst drawdown in twenty years is the 2008 crash at -16.5% (monthly marks) or -17.15% (daily marks), NOT the -2.4% that research/146 and /151 reported. Those studies measured the 2008 window starting 2008-01-01, which is AFTER the December-2007 peak, so the drawdown from that peak was invisible. The standing conclusion that the True North gate plus Open Alpha’s stops have already stripped the crash tail is withdrawn — the pair has a crash tail, and it is its single deepest hole. SECOND, CORRELATION SAYS THE BOOK OWNS ONE FACTOR. Open Alpha to VCP is 0.749 daily and 0.767 monthly; at position level 87.0% of Open Alpha’s signals are also VCP signals and the two hold the same stock on the same day 42-49% of the time. VCP is Open Alpha wearing a different screen. MYB shares 90.2% of its signals with VCP. Only two sleeves are genuinely different things: IPO-Base (0.21 daily to Open Alpha, 0.22 to True North, and 0.0% signal AND 0.0% holding-day overlap — not one shared symbol-day in sixteen years) and GOLD (about zero to everything, negative on monthly returns). THIRD, THE FRONTIER. Every weight vector on a 5% grid over the four survivors was enumerated — 1,767 vectors on three windows, 5,301 cells, each over 360 paired paths (30 Open Alpha seeds x 12 True North rebalance-day offsets). 197 vectors clear the pre-registered bar on ALL THREE windows at equal-or-better CAGR, beating the pair, a cash null AND an IPO beta-matched null on at least 288 of 360 paired paths each. Every admitted vector holds gold; almost all hold IPO; the ones that win most cut True North. That is a broad contiguous plateau rather than a peak, which is the robustness evidence and also the honest warning that the exact weights are not the finding — the direction is. RECOMMENDED, WITH CONSTRAINTS APPLIED: OA 40 / TN 25 / IPO 20 / GOLD 15 gives 28.21% CAGR at -10.77% drawdown, Calmar 2.61, against the deployed pair’s 27.74% / -17.01% / 1.68 over 2006-04 to 2026-08, winning 360/360 paths against the pair, 360/360 against the cash null and 358/360 against the beta-matched null. But IPO has never traded, not live and not even on paper, and it is invested only 19.6% of the time — beyond about 20% weight its extra Calmar is indistinguishable from de-levering on two of three windows. The deployable-today step is therefore the gold leg alone, with IPO going to a paper soak first.',
    status: 'COMPLETE',
    date: '2026-09-05',
    cardBlurb:
      'All 57 subsets of six sleeves, plus a 5,301-cell weight frontier, paired across 360 paths. Open Alpha and VCP share 87% of their signals; Open Alpha and IPO-Base have never held the same stock on the same day. Corrects a measurement error that hid the deployed pair’s true 2008 drawdown (-16.5%, not -2.4%), and refutes research/152’s four-sleeve claim against a properly specified gold-only null.',
    cardStats: [
      { label: 'Verdict', value: 'STRATEGY candidate — add TWO satellites' },
      { label: 'Recommended vs deployed', value: '28.2% / -10.8% / 2.61  vs  27.7% / -17.0% / 1.68' },
      { label: 'Admitted weight vectors', value: '197 of 1,767, all three windows' },
    ],

    systemRules: {
      intro:
        'Nothing new is traded here. Six existing sleeves are combined; the question is which combination, at what weights, and whether the improvement survives three nulls. A PATH is one (Open Alpha seed, True North rebalance-day offset) pair — 30 x 12 = 360 of them — and within a path the stochastic research sleeves reuse the same seed index, so every A-versus-B number below is a distribution of PAIRED differences, never an unpaired median.',
      sharedCoreTitle: 'The six sleeves and how their curves were made',
      sharedCore: [
        { k: 'OA — Open Alpha (LIVE, Rs 10L)', v: 'Close above the prior all-time-high close; relative strength 70+; 20-day median traded value at least Rs 5 crore; buy-stop at the pivot filled at max(pivot, open); -8% close stop; exit on the first close below the 15-day SMA; 16 slots at 6.25% of NAV; no market gate. Regenerated here across 30 random-selection seeds.' },
        { k: 'TN — True North (LIVE, momentum)', v: 'Nifty-200, top 8 equal weight, monthly rebalance, NIFTYBEES 100-SMA weekly liquidate-all gate, 15-day-low Donchian stop. Deterministic, so its ensemble is the 12 rebalance-day offsets. Across the full 12 its CAGR runs 14.9% to 25.0% — research/146 cached only offsets 0, 4 and 8, which happens to miss both tails.' },
        { k: 'VCP — research/151 (NO EDGE)', v: 'Close above the highest close of the prior 30 days. Included deliberately: knowing WHY a rejected sleeve fails in a blend is part of the answer.' },
        { k: 'MYB — research/152 (SIGNAL, not adopted)', v: 'Close above the highest close of the prior three years, where that level is below the stock’s all-time high. History cannot begin before 2010 by construction.' },
        { k: 'IPO — research/153 (STRATEGY candidate)', v: 'Listed within six months, 25-day base, pivot = highest close, depth at most 30%, 8 slots at 18.75%, -8% stop, SMA-20 trail, +25% take-profit. Invested only 19.6% of NAV on average; it took zero trades in 2013 and in 2014.' },
        { k: 'GOLD — research/147 (candidate)', v: 'GOLDBEES buy-and-hold from 2015. Before that, a REBUILT daily gold-in-rupee reconstruction (COMEX front gold times USDINR, aligned to the NSE calendar), labelled everywhere it is used.' },
        { k: 'Blend mechanics', v: 'Monthly rebalance to target weights. Headline figures use month-end marks for comparability with research/146 to /153; a daily-marked panel is reported as a robustness check and changes no ranking.' },
        { k: 'Costs, tax, cash', v: '25 bps per side inside every equity sleeve; 20% short-term / 12.5% long-term capital gains with Indian financial-year netting and loss carry-forward; idle cash at 5% per annum.' },
      ],
      riskLayer: {
        title: 'Three windows, stated explicitly, never mixed',
        caption:
          'The sleeves do not share a start date. Mixing a blend that contains 2008 with one that does not is exactly the error caught and corrected inside research/152. Every comparison lives inside one panel.',
        columns: ['Panel', 'Window', 'Sleeves available', 'Contains', 'Deployed pair on it (CAGR / MaxDD / Calmar)'],
        rows: [
          ['A — master', '2010-01 to 2026-08', 'all six (gold 2010-14 = reconstruction)', '2020 crash, 2018 and 2022H1 grinds. No 2008.', '26.63% / -15.69% / 1.685'],
          ['B — crash-honest', '2006-04 to 2026-08', 'five; MYB cannot exist', '2008 GFC, 2020, both grinds', '27.74% / -17.01% / 1.678'],
          ['C — real data only', '2015-01 to 2026-08', 'all six, nothing reconstructed', '2018, 2020, 2022H1', '30.54% / -15.69% / 1.925'],
        ],
        highlightRows: [1],
      },
    },

    system: {
      intro:
        'The ask was for the correlations between every system including Open Alpha and True North, in all possible combinations. With six sleeves there are 57 subsets of size two or more; all of them were run at equal weight on every panel where their members exist, then a weight sweep, then a full frontier enumeration. 8,172 cells in total, disclosed so any single winner can be discounted.',
      rows: [
        { k: 'Correlations', v: '15 pairs x daily and monthly x 3 panels = 90 cells, reported as the seed-median with the full seed range (up to 900 seed pairs per cell).' },
        { k: 'Equal-weight subsets', v: '57 on panel A + 26 on panel B + 57 on panel C = 140 cells, each over 360 paired paths.' },
        { k: 'Weight sweep', v: '646 further cells — pair sweeps at 10/20/25/33/50/67/75/80/90% and satellite sweeps at 10/20/25/33% around a TN+OA core.' },
        { k: 'Frontier enumeration', v: '1,767 weight vectors on a 5% grid over OA/TN/IPO/GOLD x 3 panels = 5,301 cells, plus 996 vectors on a 10% grid including MYB x 2 panels = 1,992.' },
        { k: 'Position-level overlap', v: 'Signal overlap computed on the raw screens (seed-free) and holding-day overlap across 5 seeds, for every pair whose trade list can be reconstructed.' },
        { k: 'Pre-registered adoption bar', v: 'Median CAGR at least the deployed pair’s, AND beating the pair, the cash null and the IPO beta-matched null on at least 288 of 360 paired paths (80%), on ALL THREE panels. Set before the run; applied literally, with no post-hoc softening.' },
        { k: 'Plateau requirement', v: 'A winning weight whose neighbours disagree is noise. The admitted set had to be contiguous, and it is.' },
      ],
    },

    conditions: {
      intro:
        'A data defect was found and fixed before anything was computed. Research/147’s cached gold-in-rupee reference series — the one that extends gold before GOLDBEES’s 2015 start — is a MONTHLY series missing 40 of its 274 months, 14.6%. Yahoo’s monthly candles drop months, and their epoch stamps carry a US/UTC offset so bars such as 2004-03-31 23:00 land in the wrong month, collide with the real March bar and get deleted as duplicates. A sparse monthly series makes a percentage-change calculation silently span two months, mis-stating every pre-2015 gold return.',
      rows: [
        { k: 'The fix', v: 'Pull the DAILY series instead (COMEX gold from 2000-08, USDINR from 2003-12), stamp months with a +12h offset so a timezone shift cannot cross a month boundary, and align onto the NSE trading calendar. Result: zero missing months, 2005-01-03 to 2026-09-04.' },
        { k: 'Reconstruction validated, not assumed', v: 'Against real GOLDBEES over 2,889 overlapping days / 140 months: MONTHLY return correlation 0.878 (the old sparse series scored 0.788), DAILY correlation only 0.390, annualised drift -1.00pp per year.' },
        { k: 'What that licenses', v: 'The reconstruction is used for monthly-rebalanced blends and yearly cells. It is NOT used for daily correlations — the 0.39 is a COMEX-close versus NSE-close timing mismatch, so daily gold correlations are computed on real GOLDBEES data only, from 2015.' },
        { k: 'Where it lives', v: 'results/gold_nav.csv inside the study folder. It is never written into market_data.db, and every figure that touches pre-2015 gold is labelled.' },
        { k: 'Window drawdowns, measured correctly', v: 'Per-window drawdown is measured from the running peak of the FULL curve, not from the window’s own first bar. Starting the 2008 window on 2008-01-01 hides the December-2007 peak; that is the artefact this study corrects.' },
        { k: 'Data snapshot', v: 'market_data.db on the VPS, maximum date 2026-09-04. Research only — no live engine, crontab or deployed spec was touched.' },
      ],
    },

    comparisons: [
      {
        title: 'Pairwise correlation — panel B, 2006-04 to 2026-08, daily returns, seed-median',
        caption:
          'The reference point every new pair is judged against is Open Alpha to True North at 0.421. Anything materially above it is the same bet twice. Seed ranges are tight — Open Alpha to VCP spans 0.719 to 0.776 across 900 seed pairs.',
        columns: ['', 'OA', 'TN', 'VCP', 'IPO', 'GOLD'],
        rows: [
          ['OA', '1.000', '0.421', '0.749', '0.211', '0.076'],
          ['TN', '0.421', '1.000', '0.473', '0.220', '-0.037'],
          ['VCP', '0.749', '0.473', '1.000', '0.269', '0.041'],
          ['IPO', '0.211', '0.220', '0.269', '1.000', '-0.003'],
          ['GOLD', '0.076', '-0.037', '0.041', '-0.003', '1.000'],
        ],
        heatmap: true,
      },
      {
        title: 'Pairwise correlation — panel A, 2010-01 to 2026-08, all six sleeves, MONTHLY returns',
        caption:
          'MYB only exists from 2010, so this is the only panel where it appears. Gold is negative to every equity sleeve on monthly returns. Daily correlations involving gold use real GOLDBEES from 2015 only.',
        columns: ['', 'OA', 'TN', 'VCP', 'MYB', 'IPO', 'GOLD'],
        rows: [
          ['OA', '1.000', '0.432', '0.750', '0.502', '0.353', '-0.080'],
          ['TN', '0.432', '1.000', '0.466', '0.409', '0.271', '-0.126'],
          ['VCP', '0.750', '0.466', '1.000', '0.612', '0.377', '-0.096'],
          ['MYB', '0.502', '0.409', '0.612', '1.000', '0.408', '-0.067'],
          ['IPO', '0.353', '0.271', '0.377', '0.408', '1.000', '-0.067'],
          ['GOLD', '-0.080', '-0.126', '-0.096', '-0.067', '-0.067', '1.000'],
        ],
        heatmap: true,
      },
      {
        title: 'Position-level overlap — two sleeves can correlate modestly and still be the same trades',
        caption:
          'Signal overlap is computed on the raw screens before slot competition, so it is seed-free; holding-day overlap is the median across 5 seeds. Window 2010-01 to 2026-09. IPO’s trading calendar was reconstructed from the database and validated: 100.00% of its 20,244 recorded holding periods reproduce exactly.',
        columns: ['Pair', 'Shared signals', '% of A’s signals', '% of B’s signals', 'Holding-day overlap (A / B)'],
        rows: [
          ['OA ~ VCP', '14,893', '87.0%', '51.1%', '48.6% / 41.5%'],
          ['VCP ~ MYB', '1,740', '6.0%', '90.2%', '4.0% / 22.0%'],
          ['OA ~ MYB', '0', '0.0%', '0.0%', '0.6% / 2.9%'],
          ['OA ~ IPO', '0', '0.0%', '0.0%', '0.0% / 0.0%'],
          ['VCP ~ IPO', '0', '0.0%', '0.0%', '0.0% / 0.0%'],
          ['MYB ~ IPO', '0', '0.0%', '0.0%', '0.0% / 0.0%'],
        ],
        highlightRows: [0, 3],
      },
      {
        title: 'All 57 subsets at equal weight — panel B (2006-04 to 2026-08), sorted by Calmar',
        caption:
          'Every subset containing VCP is beaten by its own cash null. Every subset containing gold improves the drawdown; every subset containing IPO improves both. Wins are out of 360 paired paths.',
        columns: ['Subset (equal weight)', 'CAGR', 'MaxDD', 'Calmar', 'dCAGR', 'dDD', 'dCalmar', 'Calmar wins', 'vs cash-null'],
        rows: [
          ['OA+TN+IPO+GOLD', '26.06%', '-8.03%', '3.224', '-1.62', '+8.66', '+1.552', '360/360', '+0.977 (360/360)'],
          ['OA+IPO+GOLD', '27.89%', '-8.99%', '3.103', '+0.02', '+7.94', '+1.458', '360/360', '+1.173 (360/360)'],
          ['OA+VCP+IPO+GOLD', '30.13%', '-10.11%', '2.976', '+2.37', '+6.81', '+1.338', '360/360', '+0.806 (360/360)'],
          ['TN+IPO+GOLD', '22.65%', '-8.92%', '2.489', '-4.99', '+8.34', '+0.895', '360/360', '+0.664 (320/360)'],
          ['OA+TN+GOLD', '23.71%', '-10.64%', '2.207', '-4.04', '+6.57', '+0.648', '360/360', '+0.352 (360/360)'],
          ['OA+TN+IPO', '29.59%', '-13.69%', '2.152', '+1.95', '+3.31', '+0.537', '360/360', '+0.230 (330/360)'],
          ['OA+GOLD', '24.92%', '-12.71%', '1.957', '-2.81', '+4.28', '+0.294', '360/360', '+0.289 (360/360)'],
          ['TN+OA — THE DEPLOYED PAIR', '27.74%', '-17.01%', '1.678', '—', '—', '—', '—', '—'],
          ['OA+TN+VCP', '30.60%', '-22.00%', '1.398', '+3.01', '-5.24', '-0.269', '13/360', '-0.545 (0/360)'],
          ['OA+VCP', '35.77%', '-26.61%', '1.347', '+7.99', '-9.76', '-0.328', '33/360', '-0.350 (0/360)'],
          ['TN+VCP', '28.09%', '-24.54%', '1.152', '+0.46', '-7.74', '-0.504', '0/360', '-0.437 (14/360)'],
        ],
        highlightRows: [0, 1, 7],
      },
      {
        title: 'The control that changes the answer — IPO is 80% cash, so a cash null is not enough',
        caption:
          'Research/153’s own print records IPO as invested only 19.6% of NAV on average, and it took zero trades in 2013 and 2014. The BETA-MATCHED null replaces IPO with 19.6% Open Alpha plus 80.4% cash at the same weight, reproducing its average exposure but none of its selection or timing. A tick means it beats that null on at least 288 of 360 paired paths.',
        columns: ['Book', 'Panel A (2010+)', 'Panel B (2006+)', 'Panel C (2015+)'],
        rows: [
          ['OA+TN+IPO+GOLD 25/25/25/25', '+0.239, 232/360  FAIL', '+0.812, 360/360  pass', '+0.083, 224/360  FAIL'],
          ['OA+TN+IPO+GOLD 30/30/20/20', '+0.100, 237/360  FAIL', '+0.556, 356/360  pass', '+0.148, 296/360  pass'],
          ['OA+TN+IPO+GOLD 40/40/10/10', '+0.198, 346/360  pass', '+0.247, 359/360  pass', '+0.273, 341/360  pass'],
          ['OA+IPO+GOLD 33/33/33', '+0.301, 336/360  pass', '+0.950, 360/360  pass', '+0.592, 360/360  pass'],
        ],
        highlightRows: [2, 3],
      },
      {
        title: 'Per-window behaviour — and the 2008 correction',
        caption:
          'Median across 360 paths, panel B. Drawdown is measured from the running peak of the full curve. The deployed pair’s 2008 drawdown is -16.5%, not the -2.4% previously reported: research/146 and /151 started the window on 2008-01-01, after the December-2007 peak. MYB cannot be evaluated in 2008 at all. Gold before 2015 is the labelled reconstruction, so read the 2008 column as directional.',
        columns: ['Book', '2008 crash ret / dd', '2020 crash ret / dd', '2018 grind ret / dd', '2022H1 grind ret / dd'],
        rows: [
          ['TN+OA 50-50 (deployed)', '+0.8 / -16.5', '-1.4 / -8.3', '-10.2 / -12.7', '-5.3 / -11.0'],
          ['OA 45 / TN 35 / IPO 10 / GOLD 10', '+4.0 / -11.5', '-0.3 / -1.1', '-7.3 / -10.2', '-3.9 / -9.3'],
          ['OA 40 / TN 25 / IPO 20 / GOLD 15', '+7.3 / -7.5', '+0.8 / -0.4', '-4.4 / -8.7', '-2.5 / -7.5'],
          ['OA+IPO+GOLD 33/33/33', '+12.2 / -4.1', '+1.5 / -0.3', '-0.7 / -8.3', '-1.1 / -7.4'],
          ['OA+TN+VCP 33 each', '-9.1 / -22.0', '-1.3 / -6.8', '-12.7 / -13.8', '-8.5 / -11.3'],
          ['OA alone', '+1.5 / -17.7', '-3.7 / -9.7', '-10.8 / -18.3', '-7.2 / -17.4'],
          ['TN alone', '-0.5 / -16.4', '+1.0 / -13.7', '-10.7 / -14.7', '-4.7 / -10.9'],
          ['VCP alone', '-26.5 / -35.6', '-1.1 / -8.1', '-18.4 / -18.6', '-14.3 / -16.8'],
          ['IPO alone', '+3.8 / -2.7', '-5.5 / -5.9', '+3.2 / -11.9', '-3.1 / -11.9'],
          ['GOLD alone', '+29.2 / -13.1', '+13.9 / 0.0', '+4.6 / -14.4', '+6.1 / -12.5'],
        ],
        highlightRows: [0],
      },
      {
        title: 'The two registered open questions from research/152',
        caption:
          'Both are answered here rather than left standing. Q2 is REFUTED: its original reference was gold at HALF the weight, which is a smaller allocation rather than a null.',
        columns: ['Question', 'What was claimed', 'What the correct test says', 'Verdict'],
        rows: [
          [
            'Q1 — does MYB+OA beat the deployed TN+OA pair?',
            '28.71% / -14.5% / 1.98 versus 26.16% / -16.1% / 1.56 on 2010-2026',
            'Reproduced: 29.13% / -14.50% / 2.017 versus 26.63% / -15.69% / 1.685, winning 314 of 360 paired paths. But MYB’s three-year pivot means it CANNOT produce a signal before 2010, and 2008 is where the pair’s worst drawdown lives — so the deciding window is unreachable by construction, not by lack of data. Every 2006-testable substitute for True North that DOES win (IPO, gold) wins on being uncorrelated, at 0.21 and about 0. MYB sits at 0.412 daily to Open Alpha and shares 90% of its signals with VCP, which is Open Alpha.',
            'Real on its window; NOT a reason to change the book. Third time a second smallcap-breakout sleeve has looked good post-2010 (research/62, /145, /152) and third time the reason is the missing crash.',
          ],
          [
            'Q2 — is 80% TN+OA / 10% gold / 10% MYB better than gold alone?',
            'Calmar 2.43 versus 2.08 gold-only',
            'Against gold at the SAME total satellite weight (OA 40 / TN 40 / GOLD 20) the mix LOSES: -0.094 Calmar on panel A winning only 91 of 360 paths, and -0.092 on panel C winning 60 of 360. It does beat a cash null (+0.302, 360/360) and a MYB-only null (+0.135, 355/360) — it is simply not better than spending the whole satellite budget on gold.',
            'REFUTED. Do not label it a finding.',
          ],
        ],
        highlightRows: [1],
      },
    ],

    results: {
      metrics: [
        { label: 'Recommended blend CAGR', value: '28.21%', hint: 'OA 40 / TN 25 / IPO 20 / GOLD 15, 2006-04 to 2026-08, after tax, 360 paths', tone: 'pos' },
        { label: 'Recommended blend MaxDD', value: '-10.77%', hint: 'against the deployed pair’s -17.01%', tone: 'pos' },
        { label: 'Recommended blend Calmar', value: '2.61', hint: 'deployed pair 1.68', tone: 'pos' },
        { label: 'Paths won vs the pair', value: '360/360', hint: 'paired on the same OA seed and TN offset' },
        { label: 'Paths won vs the beta-matched null', value: '358/360', hint: 'the null that catches de-levering' },
        { label: 'OA to VCP signal overlap', value: '87.0%', hint: 'VCP is Open Alpha wearing a different screen', tone: 'neg' },
        { label: 'OA to IPO holding-day overlap', value: '0.0%', hint: 'not one shared symbol-day in sixteen years', tone: 'pos' },
        { label: 'Deployed pair 2008 drawdown', value: '-16.5%', hint: 'previously reported as -2.4%; the window started after the peak', tone: 'neg' },
        { label: 'Cells run and disclosed', value: '8,172', hint: 'each over up to 360 paired paths' },
      ],
      tables: [
        {
          title: 'YoY house-format table — panel B, 2006-04 to 2026-08',
          caption:
            'After tax, net of 25 bps a side, median of 360 paired paths, monthly rebalanced. Each cell is the annual return with the intra-year maximum drawdown, measured from the running peak, in brackets. 2026 is eight months. The benchmark is excluded from the best-of picks. All four books finish within 0.5pp of each other on twenty-year CAGR and the diversified ones do it with half the drawdown: the deployed pair wins the biggest up years, the diversified books win every bad one.',
          columns: ['Year', 'TN+OA 50-50 (deployed)', 'OA45 TN35 IPO10 GOLD10', 'OA40 TN25 IPO20 GOLD15', 'OA+IPO+GOLD 33/33/33', 'NIFTY 50', 'BEST CAGR', 'LEAST DD', 'BEST OVERALL'],
          rows: [
            ['2006', '+15.0 (-4.8)', '+17.6 (-3.8)', '+20.4 (-3.8)', '+23.6 (-3.2)', '+15.9 (-11.8)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2007', '+97.4 (-5.9)', '+91.7 (-4.8)', '+88.7 (-3.9)', '+77.7 (-1.8)', '+53.0 (-7.8)', 'deployed', 'OA+IPO+GOLD', 'deployed'],
            ['2008', '-13.9 (-15.9)', '-8.3 (-11.0)', '-4.5 (-8.3)', '+5.6 (-4.1)', '-52.1 (-55.2)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2009', '+62.5 (-14.3)', '+51.0 (-7.9)', '+42.5 (-3.6)', '+26.1 (-4.1)', '+75.6 (-54.9)', 'deployed', 'OA40 TN25', 'deployed'],
            ['2010', '+10.5 (-15.0)', '+15.1 (-10.5)', '+18.6 (-7.9)', '+25.3 (-5.7)', '+18.6 (-20.6)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2011', '-6.9 (-9.0)', '-1.3 (-7.2)', '+2.3 (-6.1)', '+12.3 (-4.4)', '-24.0 (-24.3)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2012', '+14.9 (-7.8)', '+13.2 (-3.2)', '+11.9 (-2.5)', '+8.9 (-1.2)', '+26.5 (-19.7)', 'deployed', 'OA+IPO+GOLD', 'OA45 TN35'],
            ['2013', '+9.0 (-4.3)', '+6.2 (-4.4)', '+4.9 (-4.0)', '-0.0 (-6.0)', '+7.2 (-11.3)', 'deployed', 'OA40 TN25', 'deployed'],
            ['2014', '+66.4 (-4.5)', '+53.0 (-3.7)', '+43.4 (-2.9)', '+24.9 (-3.4)', '+31.6 (-3.5)', 'deployed', 'OA40 TN25', 'deployed'],
            ['2015', '+2.4 (-9.2)', '+3.6 (-7.2)', '+4.6 (-5.8)', '+5.7 (-4.1)', '-4.3 (-10.4)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2016', '+21.6 (-6.8)', '+22.5 (-5.0)', '+24.2 (-4.1)', '+25.3 (-3.0)', '+4.0 (-20.8)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2017', '+67.9 (-1.9)', '+63.1 (-1.2)', '+60.0 (-0.3)', '+51.8 (0.0)', '+29.9 (-2.7)', 'deployed', 'OA+IPO+GOLD', 'deployed'],
            ['2018', '-12.9 (-12.9)', '-10.3 (-10.7)', '-8.1 (-10.0)', '-4.4 (-8.3)', '+4.8 (-11.0)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2019', '+4.1 (-15.2)', '+7.3 (-11.8)', '+9.5 (-9.1)', '+15.0 (-4.4)', '+13.6 (-7.0)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2020', '+84.4 (-8.3)', '+82.7 (-1.2)', '+81.9 (-0.8)', '+74.3 (-0.5)', '+15.4 (-28.8)', 'deployed', 'OA+IPO+GOLD', 'OA45 TN35'],
            ['2021', '+103.9 (-2.0)', '+87.3 (-3.5)', '+75.5 (-4.9)', '+53.3 (-7.1)', '+26.0 (-3.9)', 'deployed', 'deployed', 'deployed'],
            ['2022', '+6.5 (-11.7)', '+7.3 (-9.9)', '+9.5 (-8.7)', '+10.9 (-6.9)', '+5.5 (-10.2)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2023', '+48.0 (-9.0)', '+48.1 (-7.2)', '+47.8 (-6.1)', '+45.2 (-3.9)', '+21.0 (-7.3)', 'OA45 TN35', 'OA+IPO+GOLD', 'OA40 TN25'],
            ['2024', '+46.1 (-8.0)', '+48.9 (-5.6)', '+51.6 (-4.2)', '+52.9 (-1.1)', '+10.4 (-8.3)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2025', '+8.0 (-12.2)', '+13.0 (-10.8)', '+14.9 (-10.4)', '+26.7 (-8.5)', '+11.7 (-13.8)', 'OA+IPO+GOLD', 'OA+IPO+GOLD', 'OA+IPO+GOLD'],
            ['2026 (8m)', '+20.3 (-5.3)', '+25.3 (-5.4)', '+31.4 (-4.5)', '+44.2 (-4.5)', '-6.9 (-14.5)', 'OA+IPO+GOLD', 'OA40 TN25', 'OA+IPO+GOLD'],
            ['FULL: CAGR / MaxDD / Calmar', '27.74 / -17.01 / 1.68', '27.90 / -12.34 / 2.28', '28.21 / -10.77 / 2.61', '27.89 / -8.99 / 3.10', '10.67 / -55.16 / 0.19', '', '', ''],
          ],
          highlightRows: [21],
        },
        {
          title: 'The weight frontier — 197 of 1,767 vectors admitted on ALL THREE panels',
          caption:
            'Admitted means: median CAGR at least the deployed pair’s, and beating the pair, the cash null AND the IPO beta-matched null on at least 288 of 360 paired paths each, on every panel. The unconstrained optimum drops True North entirely and is reported rather than hidden; the recommendation applies operational constraints (keep both live books, cap the never-traded sleeve at 20%, cap gold at 20%).',
          columns: ['Weights', 'A: CAGR / DD / Calmar', 'B: CAGR / DD / Calmar', 'C: CAGR / DD / Calmar', 'Note'],
          rows: [
            ['OA 20 / IPO 50 / GOLD 30', '26.85 / -8.48 / 3.158', '27.89 / -8.48 / 3.266', '33.72 / -8.48 / 3.956', 'Unconstrained frontier optimum. NOT recommended — 50% in a never-traded sleeve, no True North.'],
            ['OA 40 / TN 25 / IPO 20 / GOLD 15', '27.10 / -10.77 / 2.506', '28.21 / -10.77 / 2.612', '32.10 / -10.77 / 2.966', 'RECOMMENDED — best vector inside the operational constraints.'],
            ['OA 45 / TN 20 / IPO 15 / GOLD 20', '27.03 / -10.78 / 2.489', '28.08 / -10.78 / 2.590', '31.73 / -10.78 / 2.927', 'Neighbour — also admitted. The plateau, not a peak.'],
            ['OA 45 / TN 25 / IPO 15 / GOLD 15', '27.26 / -11.37 / 2.382', '28.31 / -11.37 / 2.474', '32.01 / -11.37 / 2.802', 'Neighbour — also admitted.'],
            ['OA 45 / TN 35 / IPO 10 / GOLD 10', '26.82 / -12.18 / 2.204', '27.90 / -12.34 / 2.277', '31.31 / -12.18 / 2.554', 'MINIMUM CHANGE from today’s book that still clears the bar.'],
            ['OA 60 / TN 15 / GOLD 25', '27.10 / -12.79 / 2.113', '28.02 / -13.31 / 2.095', '31.16 / -12.79 / 2.426', 'BEST WITHOUT IPO — the actionable-today step, no unproven sleeve.'],
            ['OA 55 / TN 30 / GOLD 15', '26.83 / -13.55 / 1.970', '27.84 / -13.78 / 2.004', '30.69 / -13.55 / 2.243', 'Gold-only, gentler tilt away from True North.'],
            ['OA 45 / TN 45 / GOLD 10 (research/147’s pick)', '25.54 / -12.91 / 1.968', '26.61 / -13.78 / 1.989', '29.21 / -12.90 / 2.240', 'NOT ADMITTED — fails CAGR-at-least-the-pair by -1.13pp on 2006+. Gold pays for its drawdown cut with return unless Open Alpha’s weight rises to fund it.'],
          ],
          highlightRows: [1, 5],
        },
        {
          title: 'Daily-marked robustness — the same books re-marked every day instead of at month end',
          caption:
            'Month-end marking (the convention kept above for comparability with research/146 to /153) cannot see an intra-month trough. Daily marking deepens every drawdown, as it should, and changes not one ranking. 120 paths. The 2015+ panel uses real GOLDBEES only.',
          columns: ['Book', '2006+ CAGR / daily MaxDD / Calmar', '2015+ CAGR / daily MaxDD / Calmar'],
          rows: [
            ['TN+OA 50-50 (deployed)', '26.52 / -17.15 / 1.536', '30.00 / -16.84 / 1.797'],
            ['OA 45 / TN 35 / IPO 10 / GOLD 10', '26.97 / -13.78 / 1.994', '30.75 / -12.57 / 2.417'],
            ['OA 40 / TN 25 / IPO 20 / GOLD 15', '27.56 / -12.90 / 2.188', '31.70 / -10.98 / 2.862'],
            ['OA+IPO+GOLD 33/33/33', '27.56 / -11.79 / 2.318', '32.07 / -10.50 / 3.073'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Sleeve exposure audit — how much of the diversification is simply idle cash',
          caption:
            'A year returning about 5.0% with near-zero intra-year drawdown is a book sitting entirely in cash at the 5% idle yield. IPO does this repeatedly. That is a real and repeatable mechanism — no company lists into a bear market, so the sleeve is automatically flat in a crash — but it is having nothing to buy, not alpha, and the beta-matched null exists to keep the two apart.',
          columns: ['Year', 'OA', 'TN', 'VCP', 'MYB', 'IPO', 'GOLD'],
          rows: [
            ['2008', '-16.2', '-12.9', '-29.8', 'n/a', '+4.2  (cash)', '+27.5'],
            ['2009', '+55.7', '+70.3', '+69.1', 'n/a', '+3.1  (cash)', '+21.6'],
            ['2012', '+12.6', '+21.8', '+12.6', '+3.5', '+2.5  (cash)', '+10.5'],
            ['2013', '+14.0', '+2.4', '+9.2', '+11.9', '+5.1  (cash)', '-19.0'],
            ['2014', '+77.7', '+52.5', '+89.4', '-0.1', '+5.0  (cash)', '+1.1'],
            ['2018', '-19.6', '-6.7', '-17.3', '+2.1', '+0.0', '+6.7'],
            ['2020', '+117.8', '+58.4', '+85.9', '+50.5', '+88.1', '+27.0'],
            ['2022', '+2.5', '+6.0', '+33.8', '+28.6', '+15.0', '+12.8'],
            ['2025', '+11.6', '+2.3', '+18.5', '+15.4', '-1.4', '+71.8'],
          ],
          heatmap: true,
        },
      ],
      charts: [
        {
          src: '/app/multi_system_blends_research154.png',
          caption:
            'Growth of Rs 100 (log) with the drawdown panel beneath, 2006-04 to 2026-08, after Indian tax and 25 bps a side, median of 360 paired paths (30 Open Alpha seeds x 12 True North rebalance-day offsets), monthly rebalanced. The four books finish within half a percentage point of each other on twenty-year CAGR; the drawdown panel is where they differ. Note the deployed pair’s 2008 hole, which prior studies did not see because they started the window after the peak.',
        },
      ],
    },

    winners: [
      {
        config: 'OA 40% / TN 25% / IPO 20% / GOLD 15% — the recommended portfolio',
        summary:
          'The best weight vector that keeps both live books, caps the never-traded sleeve at 20% and gold at 15%. Same twenty-year CAGR as the deployed pair, at a drawdown a third smaller, on every window and against every null.',
        metrics: [
          { k: 'CAGR (2006-04 to 2026-08)', v: '28.21% vs the pair’s 27.74%' },
          { k: 'MaxDD (monthly / daily marks)', v: '-10.77% / -12.90% vs -17.01% / -17.15%' },
          { k: 'Calmar', v: '2.61 vs 1.68' },
          { k: 'Paired paths won vs the pair', v: '360/360' },
          { k: 'Versus the cash null', v: '+0.977 Calmar, 360/360' },
          { k: 'Versus the IPO beta-matched null', v: '+0.556 Calmar, 358/360' },
          { k: '2008 crash', v: '+7.3% return at a -7.5% drawdown, versus +0.8% at -16.5%' },
          { k: 'Correlation of the added sleeves', v: 'IPO 0.211 daily to OA and 0.220 to TN; GOLD 0.076 and -0.037. Bar was 0.40.' },
        ],
        rejected: [
          'VCP in any weight — 87% of Open Alpha’s signals are VCP signals, 0.749 daily correlation, and every VCP subset loses to its own cash null.',
          'MYB as a third sleeve — 0.412 daily to Open Alpha and 90% of its signals are VCP’s. At equal satellite budget, adding MYB is WORSE than adding more gold (-0.094 Calmar, 91/360).',
          'The unconstrained frontier optimum (OA 20 / IPO 50 / GOLD 30) — highest Calmar of all, and not a recommendation: it puts half the book into a sleeve that has never placed an order and deletes a live system.',
          'Research/147’s 45/45/10 — not admitted; it fails CAGR-at-least-the-pair by 1.13pp on the 2006+ window.',
        ],
      },
      {
        config: 'OA 60% / TN 15% / GOLD 25% — the deployable-today step',
        summary:
          'The best admitted vector that uses no unproven sleeve. Gold is a listed ETF that can be bought tomorrow; IPO is a three-hour-old backtest. This is what the evidence supports acting on now, with IPO going to a paper soak first.',
        metrics: [
          { k: 'CAGR (2006-04 to 2026-08)', v: '28.02% vs the pair’s 27.74%' },
          { k: 'MaxDD', v: '-13.31% vs -17.01%' },
          { k: 'Calmar', v: '2.095 vs 1.678' },
          { k: 'Paired paths won', v: '360/360 against the pair and against the cash null, on all three panels' },
          { k: 'What it also does', v: 'Raises Open Alpha’s weight and cuts True North’s, which is independently what research/144 concluded when asked where more return comes from.' },
        ],
      },
    ],

    caveats: [
      'IPO HAS NEVER TRADED. Not live, not on paper. Its listing-date table was built in research/153 hours before this study from a heuristic that rejects bulk data-onboarding waves; the table is validated on a 60-name test set, not on all 1,293 accepted listings. A 20% allocation to it is a research recommendation, not a deployment instruction.',
      'IPO IS 80% CASH AND HAS MULTI-YEAR DEAD ZONES. It is invested only 19.6% of NAV on average and took zero trades in 2013 and in 2014. Beyond about 20% weight the beta-matched null shows the extra Calmar is indistinguishable from de-levering on two of the three windows. Its crash protection is structural — the IPO window closes in a bear market — which is real and repeatable, but it is having nothing to buy, not alpha.',
      'IPO CAPACITY IS THE BINDING CONSTRAINT AT SCALE. Young listings are small and thin. At 20% of a Rs 10L book that is Rs 2L across 8 slots and is fine; at Rs 1cr it is not obviously executable, and research/153’s capacity note must be re-derived before any size increase.',
      'GOLD BEFORE 2015 IS RECONSTRUCTED. COMEX gold times USDINR, daily, monthly correlation 0.878 to the real instrument with -1.0pp per year of drift. The 2008 column — the single most persuasive row in this study — rests on it. Treat 2008 as directional, not decision-grade. Separately, 2015-2026 was a strong gold decade; gold has no carry, and a 1980-to-2000-style dead period sits outside any data we hold.',
      'SURVIVORSHIP. All equity sleeves run on Kite’s current instrument list, so delisted names are absent; 2006 carries only about 528 priced symbols. This inflates absolute CAGR across the board. It should bias the RELATIVE comparison less, since all four books draw from the same universe, but it is not zero.',
      'MULTIPLE TESTING: 8,172 cells. The defence is the plateau — 197 admitted vectors forming one contiguous region rather than isolated winners — plus paired win counts of at least 288 of 360 and three independent nulls. It is not the size of any single number.',
      'MONTHLY REBALANCING IS ASSUMED FRICTIONLESS AT THE BLEND LEVEL. Costs and tax are modelled inside each sleeve, but moving money between four sleeves every month is real turnover with real tax events that are NOT modelled here. A quarterly-rebalance sensitivity is owed before deployment.',
      'PRIOR-FINDING RETRACTION AND ITS BLAST RADIUS. Every per-window drawdown figure in research/146 through /153 used the same window-start convention and may understate the same way. Those studies should be re-audited. In particular, the standing structural claim that crash-alpha candidates solve a problem the deployed pair does not have is false, and candidates rejected on that basis deserve one re-look.',
      'NOT TESTED, AND WHY: blend-level rebalancing cost and tax; whether IPO’s edge survives a live paper soak (that is the next gate, not a backtest question); any sleeve outside the six named; leverage or a cash-drag-financed overlay.',
      'NOTHING WAS DEPLOYED. No live engine, crontab, sizing or spec was touched by this study.',
    ],
    githubLinks: [{ label: 'research/154 (repo)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/154_multi_system_blends' }],
    projectPaths: [
      'research\\154_multi_system_blends\\MULTI_SYSTEM_CORRELATION_BLEND_DAILY_SWEEP_STATUS.md',
      'research\\154_multi_system_blends\\results\\RESULTS.md',
    ],
  },
"""


def main():
    src = TS.read_text(encoding="utf-8")
    if SLUG in src:
        print("already published, nothing to do")
    else:
        anchor = "];\n\nexport function getStudy"
        assert anchor in src, "could not find the end of BACKTEST_STUDIES"
        src = src.replace(anchor, ENTRY + anchor, 1)
        TS.write_text(src, encoding="utf-8")
        print(f"inserted {SLUG} into backtests.ts")
    dst = ROOT / "frontend/public/multi_system_blends_research154.png"
    shutil.copy(RES / "multi_system_blends_research154.png", dst)
    print(f"figure copied to {dst}")


if __name__ == "__main__":
    main()
