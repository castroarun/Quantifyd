# -*- coding: utf-8 -*-
"""Insert the research/75 BacktestStudy into frontend/src/data/backtests.ts IN PLACE
on the VPS (no scp-overwrite — per the 2026-06-11 stale-overwrite incident rule),
then report slug count before/after."""
from pathlib import Path
import re, shutil

TS = Path("/home/arun/quantifyd/frontend/src/data/backtests.ts")
SLUG = "nifty250-momentum-video-research75"

STUDY = r"""  {
    slug: 'nifty250-momentum-video-research75',
    title: 'Nifty-250 Momentum Top-15 — Faithful Replication of the "Only Momentum Strategy" Video',
    verdict:
      'Faithful survivorship-free replication of the Quantinuous "Only Momentum Strategy You Need for Nifty 250 Stocks" video (top-15 by 12-month momentum, per-stock 50>100>200 EMA filter, Nifty-100EMA cash gate, monthly). The video’s claimed 27% CAGR REPLICATES and is beaten — 31.8% net / 29.1% post-tax, 292×, 2006–2026 — but its claimed −23% MaxDD does NOT hold: the honest daily-marked drawdown is −31.6% (our 20-year window includes 2008). The NIFTYBEES-100EMA cash gate is the ENTIRE risk story (removing it → −51% DD) and is IRREPLACEABLE — no per-stock quality / ATH-proximity / MA-exit combination substitutes for it (best gate-less DD −46%). The per-stock EMA-stack filter is inert-to-harmful. Best RISK-ADJUSTED knob = midcap + 6-month relative strength (Calmar 1.26, −29% DD); highest raw CAGR = mid+small combo (43.5% net, but −42% DD). Same family as Aurum’s midcap_smoothest — corroborates the live design, does not add new alpha. STRATEGY-family candidate (already harvested).',
    status: 'COMPLETE',
    date: '2026-07-21',
    cardBlurb:
      'Replicated a popular YouTube momentum system on our own survivorship-free NSE data, 2006–2026, net of 0.3% cost, daily-marked. The 27% CAGR claim replicates (31.8% net, 292×); the −23% drawdown claim does not (−31.6%). Attribution: the index cash-gate is the whole risk story and cannot be replaced by per-stock rules; midcap + 6-month RS is the best knob.',
    cardStats: [
      { label: 'CAGR (net)', value: '31.8%' },
      { label: 'MaxDD (daily)', value: '−31.6%' },
      { label: 'Total return', value: '292×' },
    ],
    system: {
      intro: 'Long-only concentrated momentum with a market-regime cash gate — the video’s rules, exactly:',
      rows: [
        { k: 'Universe', v: 'Nifty LargeMidcap 250 — survivorship-free PIT proxy = top-250 NSE names by trailing-6-month median traded value, rebuilt monthly (ETFs/index excluded).' },
        { k: 'Selection', v: 'Rank the universe by momentum, hold the TOP 15 equal-weight, 100% invested when risk-on.' },
        { k: 'Momentum', v: 'Faithful default = plain 12-month price return (also tested 12−1, and 6m/12m relative strength — conclusion unchanged).' },
        { k: 'Per-stock trend filter', v: 'Eligible only if EMA50 > EMA100 > EMA200 on the stock’s own close (causal).' },
        { k: 'Market regime gate', v: 'If NIFTYBEES close ≤ its own EMA100 → liquidate to cash; NIFTYBEES = full-history Nifty-50 proxy (raw NIFTY50 daily only starts 2023).' },
        { k: 'Rotation', v: 'Monthly; daily-marked NAV for honest drawdown. Idle cash earns 6.5% p.a. (sensitivity: 4% → −1pp CAGR, 0% → −2.6pp).' },
        { k: 'Costs / tax', v: '0.3% round-trip on turnover; post-tax = 20% STCG on lots < 365 days (shown separately).' },
        { k: 'Window', v: '2006–2026 (~20.5y, incl. the 2008 crash — the real test of the −23% DD claim).' },
      ],
    },
    conditions: {
      intro: 'Backtest window, benchmark, host.',
      rows: [
        { k: 'Period', v: 'Jan 2006 – Jul 2026 (~20.5 years).' },
        { k: 'Benchmark', v: 'NIFTY-50 (NIFTYBEES), same window, excluded from the investable universe.' },
        { k: 'Host', v: 'VPS market_data.db snapshot 2026-07-08; reproducible from committed scripts.' },
      ],
    },
    comparisons: [
      {
        title: 'Universe × momentum-angle — net CAGR / MaxDD / Calmar (2006–2026)',
        columns: ['Universe · momentum', 'Net CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['midcap · 6m RS (rs126)', '37.2%', '−29.6%', '1.26'],
          ['midcap · 120d RS', '36.3%', '−29.2%', '1.25'],
          ['midcap · 6m+12m blend', '38.8%', '−32.8%', '1.18'],
          ['large-mid 250 · blend', '36.2%', '−32.8%', '1.11'],
          ['large-mid 250 · 12m (faithful, no stack)', '34.7%', '−32.2%', '1.08'],
          ['mid+small combo · 12m', '43.5%', '−42.2%', '1.03'],
          ['smallcap · 12m', '33.2%', '−46.2%', '0.72'],
        ],
        highlightRows: [0, 1],
        heatmap: true,
      },
    ],
    results: {
      metrics: [
        { label: 'CAGR (net)', value: '31.8%', tone: 'pos' },
        { label: 'CAGR (post-tax 20%)', value: '29.1%', tone: 'pos' },
        { label: 'NIFTYBEES CAGR', value: '11.6%' },
        { label: 'Excess / yr', value: '+20.2%', tone: 'pos' },
        { label: 'Sharpe', value: '1.14', tone: 'pos' },
        { label: 'Max Drawdown', value: '−31.6%', tone: 'neg', hint: 'vs NIFTYBEES −59.7%; video claimed −23%' },
        { label: 'Calmar', value: '1.01', tone: 'pos' },
        { label: 'Yrs beating index', value: '71%' },
      ],
      tables: [
        {
          title: 'Faithful base vs benchmark',
          columns: ['Metric', 'Nifty-250 Momentum', 'NIFTYBEES'],
          rows: [
            ['CAGR', '31.8%', '11.6%'],
            ['Total return', '291.6x', '9.5x'],
            ['Sharpe', '1.14', '0.34'],
            ['Max Drawdown', '−31.6%', '−59.7%'],
            ['Calmar', '1.01', '0.19'],
          ],
          highlightRows: [0, 1, 3],
        },
        {
          title: 'Rule attribution — the gate is the whole risk story; the EMA-stack is inert',
          columns: ['Configuration', 'Net CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Faithful base (gate ON, EMA-stack ON)', '31.8%', '−31.6%', '1.01'],
            ['No EMA-stack (gate ON)', '34.7%', '−32.2%', '1.08'],
            ['No index gate (stack ON)', '~20%', '−51.3%', '0.38'],
            ['Pure momentum (no gate, no stack)', '~21%', '−51.4%', '0.40'],
          ],
          highlightRows: [0],
        },
        {
          title: 'Can per-stock controls REPLACE the gate? (midcap RS120) — No.',
          columns: ['Risk control (no gate)', 'Net CAGR', 'MaxDD', 'Calmar'],
          rows: [
            ['Gate ON (baseline)', '36.2%', '−29.2%', '1.24'],
            ['No gate, nothing', '32.9%', '−65.2%', '0.50'],
            ['+ quality filter', '36.3%', '−64.6%', '0.56'],
            ['+ ATH-proximity', '23.1%', '−58.4%', '0.39'],
            ['+ SMA100 exit', '32.7%', '−53.9%', '0.61'],
            ['+ Donchian-15 exit (best gate-less)', '31.0%', '−46.2%', '0.67'],
          ],
          highlightRows: [0],
        },
      ],
      charts: [
        {
          src: '/app/nifty250-momentum-research75-factsheet.png',
          caption:
            'CLIENT FACTSHEET — Nifty-250 Momentum Top-15 (faithful video replication) vs NIFTY 50, 2006–2026, survivorship-free, net of 0.3% cost, daily-marked. 31.8% CAGR (29.1% post-tax), 291.6× vs 9.5×, Sharpe 1.14, MaxDD −31.6% vs −59.7%, 71% of years beating the index. Note the drawdown panel: the gate held the strategy to −31.6% while NIFTY fell −59.7% in 2008. Generated by research/_utilities/tearsheet.py.',
        },
      ],
    },
    winners: [
      {
        config: 'Faithful video base · large-mid 250 · 12m momentum · EMA-stack · 100EMA gate · monthly',
        summary: 'The video’s CAGR claim replicates and is beaten (31.8% net / 292×); its −23% drawdown claim does not (−31.6%, because our honest window includes 2008). Best risk-adjusted variant = midcap + 6-month RS (Calmar 1.26).',
        metrics: [
          { k: 'CAGR', v: '31.8% net / 29.1% post-tax' },
          { k: 'Excess', v: '+20.2%/yr vs NIFTYBEES' },
          { k: 'Sharpe', v: '1.14' },
          { k: 'MaxDD', v: '−31.6%' },
          { k: 'Calmar', v: '1.01' },
        ],
        rejected: [
          'Dropping the 100EMA gate: drawdown blows out to −51% — the gate is the entire risk story and IRREPLACEABLE.',
          'Per-stock quality / ATH-proximity / MA-exit as a gate substitute: best gate-less DD is still −46% (Donchian), Calmar ≤ 0.67 vs the gated 1.24. Stocks break one at a time — a per-stock rule cannot be a market circuit-breaker.',
          'The per-stock 50>100>200 EMA filter: inert-to-harmful (removing it RAISES CAGR 31.8% → 34.7% at the same drawdown).',
          'mid+small combo (highest CAGR 43.5%, survives cost-stress to 0.7%) — rejected on drawdown: −42% is uninvestable.',
        ],
      },
    ],
    caveats: [
      'The video’s −23% MaxDD is not reproducible on an honest 20-year, daily-marked, survivorship-free basis — we get −31.6%. The likely reasons the video looks shallower: a shorter/post-2008 window, monthly (not daily) marking, and/or index-provided constituent data. Its 27% CAGR, by contrast, is conservative — we beat it (31.8%).',
      'A data-integrity bug was caught and fixed mid-study: the first run reused a helper that hard-coded a 2014 rebalance start, parking the book in cash for 2006–2013 and never testing 2008 (it reported a spurious 21.5% with a +111% 2014 jump). Fixed to trade the full 2006–2026 calendar; all headline numbers are post-fix.',
      'Cash-yield assumption: idle cash earns a modeled 6.5% p.a. (liquid fund). At a realistic 4% the CAGR is ~1pp lower; at 0% it is ~2.6pp lower. A modest but real tailwind.',
      'Momentum definition was the one thing the video left unspecified; the plain-12m default and two alternatives (12−1, risk-adjusted 6m/12m) all give the same conclusion, so the result does not hinge on it.',
      'Cost realism: 0.3% round-trip is defensible for large-mid and midcap; smallcap/combo cells are optimistic (real 0.5–0.7%+) — cost-stressed, they survive on CAGR (combo 43%→39%) but are killed by drawdown, not fees.',
      'Redundancy: this is the same family as the live momentum-paper book and Aurum’s midcap_smoothest (research/41/62). The study CORROBORATES those; it is not new alpha.',
      'Backtest, net of modelled costs (post-tax where stated). Nothing wired live. Past performance is not indicative of future results.',
    ],
    githubLinks: [
      {
        label: 'RESULTS (phase 2 verdict + tables)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/75_nifty250_momentum_top15/results/RESULTS_P2.md',
      },
      {
        label: 'run_nifty250_momentum.py (faithful engine)',
        href: 'https://github.com/castroarun/Quantifyd/tree/main/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py',
      },
    ],
    projectPaths: [
      'research\\75_nifty250_momentum_top15\\NIFTY250_MOMENTUM_TOP15_DAILY_SWEEP_STATUS.md',
      'research\\75_nifty250_momentum_top15\\scripts\\ (run_nifty250_momentum.py, run_variants_phase2.py, run_phase3_combos.py)',
      'research\\75_nifty250_momentum_top15\\results\\ (ranking*.csv, RESULTS_P2.md, tearsheet.png)',
    ],
  },
"""


def main():
    src = TS.read_text(encoding="utf-8")
    before = src.count("slug:")
    if SLUG in src:
        print(f"ALREADY PRESENT ({SLUG}); slug count={before}. No change."); return
    # insert before the array's closing "];" that precedes getStudy
    marker = "\n];\n\nexport function getStudy"
    idx = src.find(marker)
    if idx < 0:
        print("MARKER NOT FOUND — aborting, no write."); return
    shutil.copy(TS, str(TS) + ".bak_r75")
    new = src[:idx] + "\n" + STUDY + src[idx + 1:]   # keep the "];" ; insert study + comma before it
    TS.write_text(new, encoding="utf-8")
    after = TS.read_text(encoding="utf-8").count("slug:")
    print(f"INSERTED. slug count {before} -> {after}. Backup: {TS}.bak_r75")


if __name__ == "__main__":
    main()
