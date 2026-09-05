"""research/155 — publish the study page: copy the factsheet into frontend/public/ and insert
the BacktestStudy entry at the top of frontend/src/data/backtests.ts (idempotent)."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/home/arun/quantifyd")
RES = ROOT / "research" / "155_ipo_cash_redeployment" / "results"
TS = ROOT / "frontend" / "src" / "data" / "backtests.ts"
PUB = ROOT / "frontend" / "public"

shutil.copy(RES / "ipo_cash_redeployment_research155.png",
            PUB / "ipo_cash_redeployment_research155.png")
print("copied factsheet ->", PUB / "ipo_cash_redeployment_research155.png")

ENTRY = r"""  {
    slug: 'ipo-idle-cash-redeployment-research155',
    title: 'Should the IPO sleeve’s idle cash be working? — redeploying it into Open Alpha and True North',
    verdict:
      'CONCLUDED — the idle cash stays in cash. The research/153 IPO-Base sleeve is invested only 32.7% of the time; two thirds of it sits in cash at 5% a year, and in 2013 and 2014 it took no trades at all while Open Alpha made +14.0% and +77.7%. The proposal tested was Arun’s: because a name listed a day or two ago is structurally ineligible for about five weeks (the spec needs 25 trading bars of history and a 25-day base window), we can see the whole candidate pool for the next 25 sessions with NO look-ahead — so park the idle cash in Open Alpha or True North while the pool is empty and pull it back when supply returns. FIRST, THE PREMISE IS CONFIRMED AND THE MECHANISM WORKS. Built at position level with an external cash sink, the gated arm missed ZERO IPO entries on any of 30 paired paths, needed only about 30 pull-backs in twenty years, and left the sleeve’s 674 trades intact. It also delivers +0.10 percentage points of blend CAGR, winning 30 of 30 paired paths — consistent, and one tenth of one percent. Median paired Calmar change is +0.006 winning 21 of 30, against a bar pre-registered before the first run of +0.10 Calmar on at least 26 of 30. REJECT. SECOND, THE ARITHMETIC SAYS THERE WAS NEVER ROOM. The sleeve is 20% of the blend and 67.3% cash, so idle cash is 13.5% of the portfolio; the candidate pool is empty on 19.0% of days — identically for horizons of 25, 50 and 100 days, because Indian IPO droughts last months not weeks. Time-averaged, the gate can only touch 2.7% of the portfolio. A 2.7% tilt cannot move a Calmar. THIRD, REDEPLOYING CONTINUOUSLY DOES MOVE IT, THE WRONG WAY. Parking all idle cash in Open Alpha lifts blend CAGR by 1.54pp and deepens blend drawdown by 3.85pp — Calmar -0.375, losing 30 of 30 — and the sleeve’s daily correlation to Open Alpha goes from 0.21 to 0.90. It stops being the uncorrelated thing that earned it a place in the book. FOURTH, FRICTION IS LARGE BUT IS NOT WHAT KILLS IT. Modelled in full (25 bps per side both ways, tax on the realised gain, T+1 settlement, an explicit lot-selection policy), friction costs 0.28pp of blend CAGR on the gated arm — 73% of its gross benefit — and 5.26pp on the continuous arm, flipping +3.30pp into -1.95pp. But frictionless, continuous parking STILL loses 28 of 30 paths on Calmar. FIFTH, A STATIC WEIGHT BEATS THE WHOLE MACHINE. A plain TN 35 / OA 35 / IPO 30 blend returns 29.39% at -13.64% against the gated mechanism’s 29.02% at -13.66% — more return, equal drawdown, and zero new operating complexity, settlement risk or pull-back tax. The idle cash is not waste; it is the sleeve’s drawdown brake, and every configuration that converts more of it earns more CAGR and gives back more than that in drawdown.',
    status: 'COMPLETE',
    date: '2026-09-05',
    cardBlurb:
      'Arun asked whether the IPO sleeve’s two-thirds idle cash could work in Open Alpha or True North during the multi-year droughts, with a mechanism to pull it back when listings resume. The forward visibility is real and carries no look-ahead, the mechanism was built at position level with every pull-back friction charged, and it never missed a single entry — it just moves 2.7% of the portfolio and buys one tenth of a percentage point. Continuous redeployment does more and costs a Calmar.',
    cardStats: [
      { label: 'Verdict', value: 'CONCLUDED — idle cash stays in cash' },
      { label: 'Gated arm vs incumbent', value: '+0.10pp CAGR, +0.006 Calmar (bar: +0.10)' },
      { label: 'Friction drag', value: '0.28pp of blend CAGR — 73% of the gross benefit' },
    ],

    systemRules: {
      intro:
        'Nothing in the IPO-Base signal, sizing or exit logic was changed. The only new machinery is what happens to cash the sleeve is not using. NAV-level blending cannot answer this question — redeployment changes the sleeve’s own cash path, its position sizes and its trade set — so the sleeve is simulated at POSITION level with an external cash sink and source, and only the finished sleeve NAV is blended. A PATH is one (Open Alpha seed, IPO seed, True North rebalance-day offset) triple; there are 30 of them and every A-versus-B figure below is a distribution of PAIRED differences, never an unpaired median.',
      sharedCoreTitle: 'The sleeve (unchanged from research/153) and the new cash machinery',
      sharedCore: [
        { k: 'The sleeve', v: 'IPO-Base MID: listed within 6 months AND at least 25 trading bars old; 25-day base; pivot = highest close; depth at most 30%; 20-day median traded value at least Rs 5 crore; buy-stop AT the pivot filled at max(pivot, open); -8% close stop; SMA-20 close trail; +25% take-profit; 8 slots at 18.75% of sleeve NAV; no market gate.' },
        { k: 'Why forward visibility is causal', v: 'A name listed one or two days ago cannot signal for about five weeks, because the spec needs 25 bars of history and a 25-day base window. So the set of names that can possibly trigger over the next 25 sessions is fully determined by listings that have ALREADY happened. The gate uses liquidity as of today plus bars-since-listing and calendar age, both deterministic once a listing is in the past. No future price is consulted.' },
        { k: 'Parking', v: 'Idle cash above a reserve is parked in an external NAV — Open Alpha (the same seed as the blend’s OA leg), True North (the same offset), a daily-rebalanced 50/50 of the two, or NIFTYBEES as a plain-beta null.' },
        { k: 'Pull-back friction, charged in full', v: '25 bps per side on BOTH the redemption and the re-parking (ladder 25 / 40 / 60); tax on the realised gain at 20% short-term and 12.5% long-term with Indian financial-year netting; T+1 settlement, so cash sold on day t arrives on t+1 and the entry that forced the sale is MISSED; and an explicit lot-selection policy — pro-rata, LIFO or FIFO.' },
        { k: 'The smarter mechanic, tested as its own axis', v: 'A liquidity reserve of k slot-sizes held in settled cash plus a slower re-park cadence, so entries are funded from the sleeve’s own natural exits and uninvested cash first and force-trimming happens only when short. Reserve 0, 1 or 2 slots; cadence daily, weekly or monthly.' },
        { k: 'A note on tax, stated plainly', v: 'The cached Open Alpha and True North NAV series are ALREADY after tax, so taxing the NAV-lot gain again double-counts. Both are reported: tax=full is Arun’s literal instruction and a strict UPPER bound on friction, and it is the arm the adoption decision was made on; tax=txn is transaction cost only and the economically correct lower bound. They differ by 0.11pp of blend CAGR on the gated arm.' },
        { k: 'Blend and benchmark', v: '40 / 40 / 20 True North + Open Alpha + IPO, monthly rebalanced, after tax, idle cash 5% a year. The incumbent to beat is that same blend with the sleeve’s idle cash left in cash.' },
      ],
      riskLayer: {
        title: 'The pre-registered adoption bar, fixed before the first run and not relaxed afterwards',
        caption:
          'Six criteria, all versus the 40/40/20 incumbent on the same 30 paired paths, after tax. Sub-window drawdowns are measured from the running peak of the FULL curve, never from the window’s first bar (the research/154 correction).',
        columns: ['#', 'Criterion', 'Gated (Open Alpha)', 'Continuous, best T+1', 'Continuous, T+0'],
        rows: [
          ['1', '+0.10 Calmar OR -2pp MaxDD at equal-or-better CAGR', 'FAIL (+0.006)', 'FAIL (-0.358)', 'FAIL (-0.375)'],
          ['2', 'Wins on at least 26 of 30 paired paths', 'FAIL (21/30)', 'FAIL (0/30)', 'FAIL (0/30)'],
          ['3', 'Drought-window MaxDD not worse by more than 1.5pp', 'PASS', 'FAIL', 'FAIL'],
          ['4', 'Survives the 25 / 40 / 60 bps cost ladder', 'FAIL', 'FAIL', 'not run'],
          ['5', 'Not dominated by a plain static weight vector', 'FAIL (1 of 49 dominates)', 'FAIL (16 of 49)', 'FAIL (11 of 49)'],
          ['6', 'Correlation below 0.40 to BOTH live legs', 'PASS (0.31 / 0.27)', 'FAIL (0.63 / 0.38)', 'FAIL (0.60 / 0.39)'],
        ],
        highlightRows: [],
      },
    },

    system: {
      intro:
        'Economic hypothesis: the Indian IPO pipeline is a policy and market-cycle variable, not a price series, and it swings from 8 usable listings in 2014 to 182 in 2025. During a drought the sleeve has nothing to buy, so its capital earns only the cash yield. If the drought is visible in advance — and it is — that capital could ride an existing book instead. The counter-hypothesis, which is what the evidence supports, is that the cash is not idle capital but a risk budget: it is what makes a 20% allocation to a volatile young-stock breakout sleeve LOWER the blend’s drawdown rather than raise it.',
      rows: [
        { k: 'Why the idle cash is genuinely idle', v: 'IPO supply is not a market-regime signal. Correlation of listings to NIFTY in the same year is -0.01 and to Open Alpha +0.18; 2014 had 8 listings while NIFTY rose 31.6%, and 2009 had 15 while NIFTY rose 75.6%. The cash is not conditionally protective — it is simply unused.' },
        { k: 'What research/153 had NOT tested', v: 'Fixed weights only. Its cash-null was a CONTROL, replacing the sleeve entirely with cash, not a redeployment arm. Nothing dynamic had ever been simulated before this study.' },
        { k: 'Arms', v: 'A incumbent (idle to cash) · B idle to Open Alpha · C idle to True North · D idle to a 50/50 · N idle to NIFTYBEES (the plain-beta null) · E the forward-visibility gate, parking only while no candidate can exist for the next N trading days, N in 25 / 50 / 100.' },
        { k: 'Windows', v: 'Full 2006-01 to 2026-09. Drought sub-windows 2008-01 to 2009-12 and 2012-01 to 2014-12, with 2013-2014 isolated (zero IPO trades). Boom sub-window 2021 to 2026.' },
        { k: 'Cells disclosed', v: '114, each a 30-path ensemble — 3,420 position-level sleeve simulations. The conclusion does not rest on a best cell: every cell failed, and the surface is monotone rather than noisy.' },
      ],
    },

    conditions: {
      intro:
        'The replication gate was run first, and had to pass before anything new was tested.',
      rows: [
        { k: 'Replication of the sleeve', v: 'Arm A, with the parking machinery switched off, reproduces research/153’s ipo_equity_seeds.csv BIT FOR BIT — maximum absolute NAV difference 0.0 across all 30 seeds and 5,128 days.' },
        { k: 'Replication of the published blend', v: 'research/153’s 40/40/20 numbers reproduced exactly: 28.27% CAGR / -12.79% MaxDD / Calmar 2.21, and 27.14 / -16.42 / 1.65 at zero IPO weight.' },
        { k: 'Drawdown convention', v: 'Confirmed on this study’s own numbers: the 40/40/20 blend’s 2008 drawdown is -1.66% measured inside the 2008 calendar slice and -12.23% measured from the full curve’s running peak. Everything here uses the latter.' },
        { k: 'The paired baseline to beat', v: '40/40/20 with idle cash: 28.92% CAGR / -13.59% MaxDD / Calmar 2.181. The deployed TN+OA pair on the same paths: 27.85 / -17.18 / 1.67.' },
        { k: 'Path construction', v: '30 Open Alpha seeds x 30 IPO seeds paired 1:1, with the 12 True North rebalance-day offsets cycled across them. This departs from a 10x3 grid deliberately: research/154 showed that offsets 0, 4 and 8 alone miss both tails of True North’s 14.9% to 25.0% CAGR range.' },
      ],
    },

    comparisons: [
      {
        title: 'Phase 1 — the naive bound: park all idle cash, every day',
        caption:
          'T+1 settlement, no cash reserve, daily re-park, pro-rata lots, tax=full, 25 bps. Medians over the same 30 paired paths. With no reserve EVERY entry needs a pull-back — 1,206 events per path — and under T+1 every one of them missed its trade.',
        columns: ['Arm', 'Sleeve CAGR', 'Sleeve trades', '% invested in IPOs', 'Blend CAGR', 'Blend MaxDD', 'Blend Calmar', 'Calmar wins', 'Corr to Open Alpha'],
        rows: [
          ['A — incumbent (idle to cash)', '31.38%', '674', '32.7%', '28.92%', '-13.59%', '2.181', '—', '0.21'],
          ['B — idle to Open Alpha', '22.45%', '183', '8.6%', '26.86%', '-19.55%', '1.378', '0/30', '0.90'],
          ['C — idle to True North', '15.01%', '196', '9.2%', '25.35%', '-18.67%', '1.409', '0/30', '0.40'],
          ['D — idle to 50/50', '19.11%', '187', '8.7%', '25.98%', '-19.18%', '1.401', '0/30', '0.76'],
          ['N — idle to NIFTYBEES (null)', '7.80%', '191', '9.2%', '23.93%', '-25.29%', '0.952', '0/30', '0.32'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Phase 2 — 54 mechanics. None of them rescues it',
        caption:
          'Reserve, cadence, lot policy and settlement, swept jointly on the Open Alpha parking asset. The axis is monotone in one thing only: how much cash you convert. The best possible mechanic is the limit of parking nothing, which IS the incumbent. Lot policy is inert — pro-rata, LIFO and FIFO differ by at most 0.01 Calmar in every family.',
        columns: ['Mechanic', 'Blend CAGR', 'Blend MaxDD', 'Blend Calmar', 'Paired change in Calmar', 'Calmar wins'],
        rows: [
          ['Incumbent (no parking)', '28.92%', '-13.59%', '2.181', '—', '—'],
          ['T+1, 2-slot reserve, monthly, LIFO (best realistic)', '28.95%', '-15.75%', '1.843', '-0.358', '0/30'],
          ['T+1, no reserve, monthly, LIFO', '29.16%', '-17.24%', '1.691', '-0.486', '0/30'],
          ['T+1, no reserve, daily, pro-rata (naive)', '26.86%', '-19.55%', '1.378', '-0.773', '0/30'],
          ['T+0, 1-slot reserve, weekly, pro-rata (settlement waived)', '30.44%', '-17.03%', '1.771', '-0.375', '0/30'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Phase 3 — the forward-visibility gate, which is Arun’s actual proposal',
        caption:
          'Park only while no name can possibly become an eligible candidate for the next N trading days. The pool is empty on 19.0% of days, and IDENTICALLY so for N = 25, 50 and 100 — Indian IPO droughts last months, not weeks, so the horizon is irrelevant. Note that the best gated cell parks into the NIFTYBEES NULL, not into either live book: there is no Open-Alpha-specific magic, only a little more equity beta.',
        columns: ['Arm', '% of sleeve parked', 'Pull-backs / 20 yrs', 'Entries missed', 'Blend CAGR', 'Blend MaxDD', 'Change in CAGR', 'CAGR wins', 'Change in Calmar', 'Calmar wins'],
        rows: [
          ['Incumbent', '0%', '0', '0', '28.918%', '-13.591%', '—', '—', '—', '—'],
          ['GATED — Open Alpha, no reserve, monthly', '13.4%', '31', '0', '29.025%', '-13.662%', '+0.105pp', '30/30', '+0.006', '21/30'],
          ['GATED — Open Alpha, 2-slot reserve, monthly', '8.3%', '30', '0', '29.012%', '-13.646%', '+0.096pp', '30/30', '+0.007', '21/30'],
          ['GATED — NIFTYBEES (null), no reserve, daily', '18.5%', '43', '0', '29.324%', '-13.669%', '+0.400pp', '30/30', '+0.028', '21/30'],
          ['GATED — True North, no reserve, monthly', '12.9%', '30', '0', '28.856%', '-13.726%', '-0.066pp', '8/30', '-0.005', '6/30'],
        ],
        highlightRows: [0],
      },
      {
        title: 'Cost ladder — each arm paired against the incumbent AT THE SAME COST',
        caption:
          'A 40 bps arm is never scored against a 25 bps incumbent. The gated arm’s entire advantage is gone by 40 bps and negative by 60. Criterion 4 fails.',
        columns: ['Arm', 'bps per side', 'Blend CAGR', 'Blend Calmar', 'Change in CAGR', 'CAGR wins', 'Change in Calmar'],
        rows: [
          ['Incumbent', '25', '28.918%', '2.181', '—', '—', '—'],
          ['GATED (tax=full)', '25', '29.025%', '2.126', '+0.105pp', '30/30', '+0.006'],
          ['GATED (tax=full)', '40', '28.477%', '2.064', '+0.005pp', '18/30', '-0.001'],
          ['GATED (tax=full)', '60', '27.808%', '1.956', '-0.128pp', '0/30', '-0.024'],
          ['GATED (tax=txn, lower bound)', '25', '29.134%', '2.139', '+0.213pp', '30/30', '+0.014'],
          ['GATED (tax=txn, lower bound)', '60', '27.917%', '1.969', '-0.018pp', '9/30', '-0.011'],
          ['CONTINUOUS best (tax=full)', '25', '28.945%', '1.843', '-0.066pp', '12/30', '-0.358'],
          ['CONTINUOUS best (tax=full)', '60', '27.991%', '1.723', '-0.033pp', '13/30', '-0.347'],
        ],
        highlightRows: [1, 3],
      },
      {
        title: 'The static-tilt null — a plain weight in a spreadsheet beats the machinery',
        caption:
          '49 STATIC weight vectors over the same 30 paired paths, same tax and cost. Calmar here is median CAGR divided by the absolute median MaxDD, one estimator across the whole grid. This is a NULL CONTROL, not a weight recommendation — research/154’s 1,767-vector frontier, which also holds gold, remains the reference for weights.',
        columns: ['Static vector (TN / OA / IPO)', 'CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['45 / 30 / 25', '28.10%', '-12.54%', '2.241'],
          ['40 / 30 / 30', '28.62%', '-12.87%', '2.224'],
          ['35 / 35 / 30', '29.39%', '-13.64%', '2.155'],
          ['35 / 40 / 25', '29.54%', '-13.84%', '2.134'],
          ['40 / 40 / 20 (the incumbent)', '28.92%', '-13.59%', '2.128'],
          ['the GATED dynamic arm, for comparison', '29.02%', '-13.66%', '—'],
        ],
        highlightRows: [2],
      },
    ],

    results: {
      metrics: [
        { label: 'Gated arm, change in blend CAGR', value: '+0.105pp', hint: 'wins 30 of 30 paired paths — consistent, and one tenth of one percent', tone: 'pos' },
        { label: 'Gated arm, change in blend Calmar', value: '+0.006', hint: 'pre-registered bar was +0.10 on at least 26 of 30; achieved 21 of 30', tone: 'neg' },
        { label: 'Friction drag, gated arm', value: '0.28pp of CAGR', hint: '73% of the gross benefit, of which the transaction cost is 0.17pp and the tax layer 0.11pp', tone: 'neg' },
        { label: 'Friction drag, continuous arm', value: '5.26pp of CAGR', hint: 'flips +3.30pp into -1.95pp' , tone: 'neg' },
        { label: 'Continuous arm, change in blend Calmar', value: '-0.375', hint: 'losing 30 of 30 paths, with settlement waived in its favour', tone: 'neg' },
        { label: 'Sleeve correlation to Open Alpha', value: '0.21 to 0.90', hint: 'continuous parking destroys the property that admitted the sleeve', tone: 'neg' },
        { label: 'Candidate pool empty', value: '19.0% of days', hint: 'identical for horizons of 25, 50 and 100 days — droughts last months, not weeks' },
        { label: 'What the gate can actually move', value: '2.7% of the portfolio', hint: '20% sleeve x 13.4% of sleeve NAV parked' },
      ],
      tables: [
        {
          title: 'Where the gate actually fires — per year, median of 30 paths',
          caption:
            'The gate does exactly what it was designed to do: it fires in 2009 and through the 2012-2014 drought and is completely silent through the 2020-2026 IPO boom. It earns +3.9pp in 2009 and +7.7pp in 2014, gives back -2.1pp in 2006 and -2.4pp in 2015, and does nothing at all for the last seven years. Two good years in twenty-one is not an edge.',
          columns: ['Year', '% of sleeve parked', 'Pull-backs', 'Blend return, incumbent', 'Blend return, gated'],
          rows: [
            ['2006', '21.3', '4', '21.5%', '19.4%'],
            ['2008', '2.4', '2', '0.8%', '0.6%'],
            ['2009', '50.3', '4', '49.4%', '53.3%'],
            ['2010', '4.9', '2', '22.3%', '21.4%'],
            ['2011', '7.3', '1', '2.2%', '1.8%'],
            ['2012', '49.4', '3', '12.8%', '12.6%'],
            ['2013', '46.8', '4', '8.6%', '8.6%'],
            ['2014', '64.3', '3', '52.0%', '59.7%'],
            ['2015', '21.5', '3', '7.0%', '4.6%'],
            ['2016-2019', '0 to 6.5', '0 to 2', 'unchanged', 'unchanged'],
            ['2020-2026', '0.0', '0', 'identical', 'identical'],
          ],
          highlightRows: [2, 7],
        },
        {
          title: 'Friction, decomposed — what the mechanism costs before you ask what it earns',
          caption:
            'All figures are the change in blend CAGR versus the incumbent, on the gated arm. Settlement is nearly free HERE precisely because the gate guarantees, N days ahead, that nothing can trigger — so a pull-back never races an entry. On the continuous arm the same T+1 pipe caused 1,206 missed entries per path.',
          columns: ['Frictions applied', 'Change in blend CAGR', 'Cost of that layer'],
          rows: [
            ['none (frictionless)', '+0.386pp', '—'],
            ['+ 25 bps transaction cost', '+0.213pp', '0.173pp'],
            ['+ tax on the realised NAV-lot gain', '+0.105pp', '0.108pp'],
            ['+ T+1 settlement', '+0.105pp', '0.007pp'],
          ],
          highlightRows: [3],
        },
      ],
      charts: [
        { src: '/app/ipo_cash_redeployment_research155.png', caption: 'Top: growth of Rs 100 on a log scale for the 40/40/20 blend under three cash policies — idle cash left in cash (the incumbent), Arun’s forward-visibility gate, and naive continuous parking in Open Alpha — median of 30 paired paths, after tax, 25 bps a side, 2006 to Sep-2026, with the drawdown panel beneath. The incumbent and the gated arm are visually indistinguishable, which IS the finding; continuous parking sits visibly lower and drops visibly deeper. Bottom left: how much of the sleeve each mechanism can actually move, per year — the gate is active only in 2006, 2009 and 2012-2015 and is silent from 2020. Bottom right: the paired verdict, median change in blend Calmar against the pre-registered +0.10 bar, with the number of winning paths out of 30 beside each bar.' },
      ],
    },

    winners: [
      {
        config: 'The incumbent — leave the idle cash in cash',
        summary:
          'Nothing tested beat it. The idle cash is not dead weight; it is the sleeve’s drawdown brake, and it is what lets a volatile young-stock breakout sleeve LOWER the blend’s drawdown at a 20% weight instead of raising it.',
        metrics: [
          { k: 'Blend, 40/40/20 with idle cash', v: '28.92% CAGR / -13.59% MaxDD / Calmar 2.181' },
          { k: 'Best redeployment arm', v: '29.02% / -13.66% / +0.006 Calmar — inside the noise' },
          { k: 'Best static alternative, if more return is wanted', v: 'TN 35 / OA 35 / IPO 30 gives 29.39% at -13.64%, with no new machinery at all' },
          { k: 'Worth keeping on file', v: 'T+1 settlement costs the gated design NOTHING — zero missed entries in twenty years — because the gate knows N days ahead that nothing can trigger. It is the only structure in which the settlement pipe is not a killer, so if the sleeve is ever run at a much larger weight the arithmetic could change.' },
        ],
        rejected: [
          'Continuous parking in Open Alpha — Calmar -0.773, losing 30/30, correlation to Open Alpha 0.90.',
          'Continuous parking in True North — Calmar -0.746, losing 30/30. Its own NIFTYBEES 100-SMA gate makes it partly self-neutralising in bear markets, which helps a little and nowhere near enough.',
          'Continuous parking in a 50/50 — Calmar -0.752, losing 30/30.',
          'Continuous parking in NIFTYBEES — Calmar -1.206, the worst arm tested.',
          'The forward-visibility gate at every horizon (25, 50, 100 days) and in every asset — the best of them buys +0.006 Calmar on 21 of 30 paths and is gone by 40 bps.',
          'All 54 mechanics — reserve, cadence and lot policy. Lot policy is inert; the reserve and the cadence only matter through how much cash they stop you converting.',
        ],
      },
    ],

    caveats: [
      'THE PARKED LEG IS MODELLED AT NAV LEVEL, NOT POSITION LEVEL. Parking in Open Alpha means holding units of its after-tax NAV series, not simulating extra Open Alpha positions, so a forced pull-back liquidates a slice rather than named positions. This is the right approximation for the question but it means the forced-exit tax is modelled on wrapper lots. Both bounds are reported — tax=full double-counts (the NAV is already after tax) and tax=txn ignores the early-realisation timing penalty — and the adoption call was made on the conservative one.',
      'T+1 WAS APPLIED ONLY TO THE PARKED LEG, not to the sleeve’s own equity trades, so the comparison against research/153’s incumbent stays exactly paired. Modelling T+1 on the sleeve’s own trades would penalise both arms roughly equally.',
      'THE GATE ASSUMES the exchange trading calendar is known ahead (it is published) and that a name liquid today stays liquid over the horizon. Neither is a price look-ahead, but both are assumptions.',
      'EVERY RESEARCH/153 CAVEAT STILL APPLIES, inherited unchanged: the entire IPO-Base edge lives in the pivot buy-stop fill (31.0% CAGR versus 17.0% on a close fill); no replication gate was ever run against the source site’s own dials; survivorship inside the database is small but the residual — IPOs that died before ever being onboarded to Kite — is unmeasurable and biases upward; market_data.db is NOT retroactively split-adjusted, mitigated by research/153’s masking of 42 suspects; and the 2025 cohort’s ends-early rate is a feed-freshness artefact rather than delisting.',
      'NIFTYBEES IS USED AS A PRICE SERIES, so dividends are not reinvested and the index null is slightly understated. It still won the gated bake-off, which only strengthens the reading that the effect is plain beta.',
      '114 CELLS DISCLOSED. The best cell should be discounted for multiple testing — but the conclusion does not rest on a best cell: every cell in the sweep failed and the surface is monotone rather than noisy.',
      'THIS STUDY DOES NOT CHANGE ANY SYSTEM’S VERDICT. IPO-Base MID remains a research/153 STRATEGY candidate at 20% weight, and it has still never traded, not even on paper.',
      'NOTHING WAS DEPLOYED. No live engine, crontab, sizing or spec was touched, and no backend restart was required at any point.',
    ],
    githubLinks: [{ label: 'research/155 (repo)', href: 'https://github.com/castroarun/Quantifyd/tree/main/research/155_ipo_cash_redeployment' }],
    projectPaths: [
      'research\\155_ipo_cash_redeployment\\IPO_CASH_REDEPLOYMENT_DAILY_SWEEP_STATUS.md',
      'research\\155_ipo_cash_redeployment\\results\\RESULTS.md',
    ],
  },
"""

src = TS.read_text(encoding="utf-8")
if "ipo-idle-cash-redeployment-research155" in src:
    print("entry already present — replacing is not attempted; nothing done")
else:
    anchor = "export const BACKTEST_STUDIES: BacktestStudy[] = [\n"
    assert anchor in src, "anchor not found"
    src = src.replace(anchor, anchor + ENTRY, 1)
    TS.write_text(src, encoding="utf-8")
    print("inserted BacktestStudy entry at the head of BACKTEST_STUDIES")
