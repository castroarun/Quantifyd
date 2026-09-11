# -*- coding: utf-8 -*-
"""OA: collect every breakout candidate in the last ~2 years, then fetch their annual books.

Two steps, both resumable, because the fetch is the slow and fragile part.

STEP 1  Which names does the OA signal actually produce between 2024-08-01 and 2026-09-04?
        The confirmed-breakout condition (close clears the prior all-time-high close, in a
        name inside the 20% base, RS >= 70, Rs 5cr/day liquid) - the honest trigger, not the
        touch. Distinct symbols only.

STEP 2  For each, the raw ANNUAL income statement and balance sheet lines needed for Arun's
        five criteria. Raw values are cached per symbol, per fiscal year, so the criteria can
        be recomputed later without refetching:

            revenue, net income, EBIT
            total debt, shareholder equity
            total assets, total current liabilities

        ROE and ROCE are computed from these rather than read from a vendor field, because
        the existing service's own ROE field was populated for only 13 of 137 symbols while
        the underlying data was there all along.

WHY THE WINDOW IS TWO YEARS. yfinance returns about four fiscal years. A decision needs the
three most recent years whose results were PUBLISHED before the decision date, so with a
four-month filing lag on a 31-March year end, the earliest decision we can evaluate honestly
is around August 2024. Going further back would mean reading results that had not been filed
when the trade was taken, which is the error this whole audit is about.

Cache: research/158_oa_arming_width/results/fund_cache/<SYMBOL>.json
Resumable: an existing file is skipped, so the script can be re-run after a rate-limit stop.
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/158_oa_arming_width/scripts'))
sys.path.insert(0, str(ROOT))

CACHE = ROOT / 'research/158_oa_arming_width/results/fund_cache'
CACHE.mkdir(parents=True, exist_ok=True)
SYMS_FILE = ROOT / 'research/158_oa_arming_width/results/oa_candidates_2y.json'
WIN = ('2024-08-01', '2026-09-04')

ROWS = {
    'revenue': ['Total Revenue', 'Operating Revenue'],
    'net_income': ['Net Income', 'Net Income Common Stockholders'],
    'ebit': ['EBIT', 'Operating Income'],
    'debt': ['Total Debt'],
    'equity': ['Stockholders Equity', 'Total Equity Gross Minority Interest'],
    'assets': ['Total Assets'],
    'cur_liab': ['Current Liabilities', 'Total Current Liabilities'],
}


def candidates():
    if SYMS_FILE.exists():
        return json.load(open(SYMS_FILE))
    import oa_entry_mechanics as em
    print('building frames for the candidate scan ...', flush=True)
    w = em.load_frames('2022-01-01', trail_sma=15)
    close, high, tv20 = w['close'], w['high'], w['tv20']
    athcp = w['athcp']
    etf = [c for c in close.columns if em.ETF_RE.search(c)]
    tv_prev, prev_close = tv20.shift(1), close.shift(1)
    elig = tv_prev >= em.TV_FLOOR
    elig[etf] = False
    r63 = close / close.shift(63) - 1
    r126 = close / close.shift(126) - 1
    r189 = close / close.shift(189) - 1
    r252 = close / close.shift(252) - 1
    rs = ((2 * r63 + r126 + r189 + r252).where(elig)
          .rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = setup & (close > athcp) & athcp.notna()
    m = (trig.index >= WIN[0]) & (trig.index <= WIN[1])
    hit = trig.loc[m]
    syms = sorted([c for c in hit.columns if bool(hit[c].any())])
    json.dump(syms, open(SYMS_FILE, 'w'))
    print('%d distinct OA breakout candidates in %s..%s' % (len(syms), *WIN), flush=True)
    return syms


def pick(df, names):
    if df is None or getattr(df, 'empty', True):
        return None
    for n in names:
        for idx in df.index:
            if str(idx).strip().lower() == n.lower():
                return df.loc[idx]
    return None


def fetch(sym):
    import yfinance as yf
    t = yf.Ticker(sym + '.NS')
    inc, bs = t.income_stmt, t.balance_sheet
    out = {}
    for key, names in ROWS.items():
        s = pick(inc if key in ('revenue', 'net_income', 'ebit') else bs, names)
        if s is None:
            continue
        for fy, v in s.items():
            if pd.isna(v):
                continue
            k = str(pd.Timestamp(fy).date())
            out.setdefault(k, {})[key] = float(v)
    return out


def main():
    syms = candidates()
    todo = [s for s in syms if not (CACHE / ('%s.json' % s)).exists()]
    print('%d cached, %d to fetch' % (len(syms) - len(todo), len(todo)), flush=True)
    ok = fail = 0
    for i, s in enumerate(todo, 1):
        try:
            d = fetch(s)
            json.dump(dict(symbol=s, fy=d, fetched=time.strftime('%Y-%m-%dT%H:%M:%S')),
                      open(CACHE / ('%s.json' % s), 'w'), indent=1)
            ok += 1 if d else 0
            fail += 0 if d else 1
        except Exception as e:
            json.dump(dict(symbol=s, fy={}, error=str(e)[:120]),
                      open(CACHE / ('%s.json' % s), 'w'))
            fail += 1
        if i % 25 == 0:
            print('  %d/%d  (%d with data, %d without)' % (i, len(todo), ok, fail),
                  flush=True)
        time.sleep(1.4)
    print('DONE fetch: %d with data, %d without' % (ok, fail), flush=True)


if __name__ == '__main__':
    main()
