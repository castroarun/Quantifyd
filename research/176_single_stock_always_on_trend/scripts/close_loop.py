# -*- coding: utf-8 -*-
"""research/176 — close the loop: INDEX.md row, TODO.md entry, ops review."""
import io
import os
import re

ROOT = '/home/arun/quantifyd'

# ---------------------------------------------------------------- research/INDEX.md
IDX = os.path.join(ROOT, 'research', 'INDEX.md')
ROW = (
    "\n| 176 | [**Single-stock always-on directional trend**](176_single_stock_always_on_trend/results/RESULTS.md) - "
    "SuperTrend / EMA crossover / the MST master(7,5)+child(7,2) pair Arun hand-traded on MARUTI, futures long/flat "
    "and long/short, every signal taken | 146 liquid F&O names; daily 2006-2026 (20.7y) and 60-min + 30-min resampled "
    "from 5-min 2015-2026 (11.5y); 327,840 name-cells | "
    "**research/48 confirmed on five times the data.** The pre-registered gate was 55% of names beating BUY-AND-HOLD of "
    "the same stock in both window halves. Best cell anywhere: **0.432** (daily EMA 9,21 long/flat), falling to **0.329** "
    "in 2016-2026. 60-min 0.336, 30-min 0.295 - the decay is **monotone in turnover**, the research/56 result on stocks. "
    "**Long/short is a catastrophe** (beat-rate 0.062, median CAGR -3.4% vs B&H +12.7%) and the short leg has negative "
    "median expectancy per trade on every family and timeframe - the 4th kill after r/81, r/82, r/83. **On Arun's own "
    "names: 0 of 40 daily cells beat B&H on RELIANCE, 0 of 40 on HDFCBANK, 3 of 40 on MARUTI**; ST(7,3) on RELIANCE "
    "returns +8.06% against the stock's +14.07% at a -63.7% drawdown. The SuperTrend surface is a flat failing plane "
    "(0.151-0.342 long/flat; 0.007-0.048 long/short), not a spike. **One real effect:** a new block-permutation null "
    "(preserves time-in-market, trade count and both run-length distributions, destroys only the timing) shows the rule "
    "beats its own matched shuffle on **69.9%** of names for CAGR and **73.3%** for Calmar - genuine timing skill that "
    "buys DRAWDOWN, not return, and decays (Calmar-beat 0.659 in 2006-15, 0.521 in 2016-26). The beat concentrates on "
    "names that FELL (75% of the negative-CAGR names, 35% of the 10-20% names) - a loss-avoider selectable only after "
    "the fact. Arun's literal MST lot-stack is the best drawdown tool in the study (3-name book -14.8% vs B&H -54.7%, "
    "2008 -9.4% vs -45.1%) and the lowest-returning positive arm (**6.86%**, below NIFTY's 8.79%). **G4 kills it:** "
    "against the live short-vol book every directional sleeve is beaten by a **plain cash sleeve** (+0.54 Calmar at 40% "
    "vs +0.14 for the best trend book) and by **RELIANCE buy-and-hold** (+0.41), and every blend cuts CAGR - "
    "research/134's conclusion reproduced at single-stock level. Options expression never opened (pre-registered as "
    "gated on G1-G4; r/56 and r/150 have killed that family four times). Best construction found = 13.88% / -28.0% / "
    "Calmar 0.495 against the deployed TN+OA pair at 1.68. | **NO EDGE - CONCLUDED** |\n"
)


def do_index():
    s = io.open(IDX, encoding='utf-8').read()
    if '| 176 |' in s:
        print('INDEX: already has 176')
        return
    # append after the last table row in the file
    s = s.rstrip('\n') + '\n' + ROW
    io.open(IDX, 'w', encoding='utf-8').write(s)
    print('INDEX: row 176 appended')


# ------------------------------------------------------------------------- TODO.md
TODO = os.path.join(ROOT, 'TODO.md')
DONE = """## DONE 2026-09-15 - Single-stock always-on directional trend (MARUTI MST, ST 7,3 on RELIANCE, EMA cross on HDFCBANK): NO EDGE on every timeframe, and plain cash is a better sleeve

Arun: *"earlier i used to trade manually on maruti always on... master super trend 7,5, child ST 7,2...
with futures long/short on mst, options selling hedge on cst... it can be on a single stock / few stocks -
like taking ALL signals... say we might end up finding supertrend 7,3 all signals works on reliance."*
research/176, published at `/app/backtest/single-stock-alwayson-trend-research176`.
**NO EDGE - CONCLUDED. Nothing deployed, nothing live touched.**

**This is research/48 again, on five times the data.** r/48 killed always-on SuperTrend on a 381-name
basket but only had 2.3 years of 15-minute history. research/176 ran the same family - SuperTrend,
EMA crossover and the MST master+child pair - over **146 liquid F&O names, 20.7 years of daily bars and
11.5 years of intraday bars resampled from 5-minute**, 327,840 name-cells. **The result gets worse with
the longer window.**

**The gate was pre-registered before the first cell ran: beat BUY-AND-HOLD of the same stock on 55% of
names, in both halves.** Best cell anywhere: **0.432** (daily EMA 9,21 long/flat), and **0.329** in
2016-2026. 60-min 0.336, 30-min 0.295 - a **monotone decay with turnover**, which is research/56's
cost-per-flip finding reproduced on stocks. At 40 bps: 0.363 / 0.267 / 0.212.

**Long/short is a catastrophe and the short leg dies for the fourth time.** Daily long/short best cell:
beat-rate 0.062, median CAGR **-3.4%** against buy-and-hold's +12.7%. Short-only has **negative median
expectancy per trade on every family and every timeframe**. Arun's long/short MST on HDFCBANK: -9.70% a
year at -91.2%. Reproduces r/81, r/82, r/83.

**On the three names he asked about: 0 of 40 daily cells beat buy-and-hold on RELIANCE, 0 of 40 on
HDFCBANK, 3 of 40 on MARUTI.** ST(7,3) on RELIANCE returns +8.06% against the stock's +14.07%, at a
-63.7% drawdown.

**One thing IS real, and it is not a return edge.** A new **block-permutation null** - preserves time in
market, trade count and both run-length distributions, destroys only WHEN the long spells happen, 200
draws per name per cell - shows the rule beats its own matched shuffle on **69.9%** of names for CAGR and
**73.3%** for Calmar. Genuine timing skill. What it buys is **drawdown** (median -51.8% vs the stock's
-75.7%), not return, and the risk edge decays: Calmar-beat 0.659 in 2006-15, **0.521 in 2016-26**. The
beat also concentrates on the names that FELL (75% of the negative-CAGR names, 35% of the 10-20% names) -
a loss-avoider you can only select after the fact.

**Arun's literal MST machine with lot stacking** (master flip resets, each child flip adds a lot to five)
is the **best drawdown tool in the study**: the three-name book goes -14.8% instead of -54.7%, and 2008
-9.4% instead of -45.1%. It is also the lowest-returning positive arm - **6.86% a year, below NIFTY 50's
8.79%** and barely above the 5.2% post-tax cash standard. A ramp, not a system.

**The deciding test kills it.** Against the live short-vol book (C1 + 45-DTE, 75 months), **every**
directional sleeve is beaten by a **plain cash sleeve** (+0.54 blend Calmar at 40% weight versus +0.14
for the best trend book) and by **RELIANCE buy-and-hold** (+0.41) - and every blend cuts CAGR, so nothing
clears the "at equal or better return" clause. That is research/134's conclusion - *the diversifier is
plain long equity and trend timing on top of it hurts* - reproduced independently on single stocks.

**The options leg was never opened**, by pre-registration: it was gated on the futures signal surviving.
research/56 already ran the MST/CST credit book (-Rs 17k to -Rs 62k per six weeks against a ~10 bps
per-flip break-even) and research/150 killed five option structures on high-win-rate signals - an overlay
changes the payoff shape, it does not create expectancy.

**Best construction found anywhere: 13.88% CAGR / -28.0% drawdown / Calmar 0.495**, against the deployed
TN+OA pair at Calmar 1.68.

- Status doc: `research/176_single_stock_always_on_trend/SINGLE_STOCK_ALWAYS_ON_TREND_MULTITF_SWEEP_STATUS.md`
- Full write-up: `research/176_single_stock_always_on_trend/results/RESULTS.md`
- Registered: 15-Mar-2027 review - any future single-name trend-following proposal must cite research/48
  and research/176 and state what is different, before any compute is spent.

---

"""


def do_todo():
    s = io.open(TODO, encoding='utf-8').read()
    if 'research/176' in s:
        print('TODO: already mentions research/176')
    else:
        anchor = '# Covered_Calls'
        i = s.index('\n', s.index(anchor))
        j = s.index('\n', i + 1)
        # insert the DONE block right after the two header lines
        k = s.index('## ', j)
        s = s[:k] + DONE + s[k:]
    # remove the QUEUED block now that it is done
    m = re.search(r'\n## QUEUED 2026-09-15 -- NEXT STUDY after research/174 closes.*?(?=\n## )',
                  s, re.S)
    if m:
        s = s[:m.start()] + '\n' + s[m.end():]
        print('TODO: queued block removed')
    io.open(TODO, 'w', encoding='utf-8').write(s)
    print('TODO: updated')


# --------------------------------------------------------------------- ops review
OPS = os.path.join(ROOT, 'research', '111_sensex_manual_mgmt', 'scripts', 'ops_center.py')
REVIEW = """    ('Single-name trend-following (SuperTrend / EMA-cross / MST master+child on one or a few stocks) - '
     'is any new proposal citing research/48 AND research/176 and saying what is different?',
     '2027-03-15', 'PENDING',
     'RAISED 15-Sep-2026 by research/176, which CLOSED this line. The family has now failed three times: '
     'research/48 on 1 name x 2.3 years of 15-min (looked superb) and on 381 names x 2.3 years (median '
     'Sharpe -0.37, beat buy-and-hold on 30% of names), and research/176 on 146 names x 20.7 years daily '
     'plus 11.5 years of 60-min and 30-min - 327,840 name-cells. The pre-registered gate was 55% of names '
     'beating BUY-AND-HOLD of the same stock in both window halves; the best cell anywhere reached 0.432 '
     'and fell to 0.329 in 2016-2026, and the decay is monotone in turnover (daily 0.432, 60-min 0.336, '
     '30-min 0.295). Long/short reached 0.062 and the short leg has negative median expectancy per trade '
     'on every family and timeframe. On the names Arun named, 0 of 40 daily cells beat buy-and-hold on '
     'RELIANCE and 0 of 40 on HDFCBANK. As a sleeve against the live short-vol book a PLAIN CASH sleeve '
     'beats every directional variant (+0.54 blend Calmar at 40% versus +0.14). The reusable tool banked '
     'there is the BLOCK-PERMUTATION null (preserves time in market, trade count and both run-length '
     'distributions, destroys only the timing) - it showed real timing skill worth drawdown and not '
     'return. CHECK at the due date: any SuperTrend / EMA-cross / MST proposal raised since must cite '
     'both studies and state what is different before a sweep is launched. Evidence: '
     'research/176_single_stock_always_on_trend/results/RESULTS.md.'),
"""


def do_ops():
    s = io.open(OPS, encoding='utf-8').read()
    if 'research/176' in s:
        print('OPS: already registered')
        return
    m = re.search(r'^REVIEWS\s*=\s*\[\s*\n', s, re.M)
    if not m:
        print('OPS: REVIEWS list not found - SKIPPED, register manually')
        return
    s = s[:m.end()] + REVIEW + s[m.end():]
    io.open(OPS, 'w', encoding='utf-8').write(s)
    print('OPS: review registered for 2027-03-15')


if __name__ == '__main__':
    do_index()
    do_todo()
    do_ops()
