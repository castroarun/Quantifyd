# -*- coding: utf-8 -*-
"""Open Alpha - Base Age: the ONE place the adopted spec lives.

research/161 adopted it, research/164 froze it (`build164.py`), and this module is the
single implementation both the live entry scanner and the live exit checker read, so the
two halves of the book can never drift apart from each other or from the study.

THE SPEC, in one paragraph. On the day's official close, a symbol signals when its close is
the first above the PRIOR all-time-high close, that prior high is at least 60 trading bars
old, and the close fell at least 20% below it in between. Liquidity floor: 20-day median
traded value >= Rs 2 cr. No volume filter, no saucer filter. A symbol re-arms only 60 bars
after its last kept signal. The entry fills at the NEXT day's open. The exit is
SuperTrend(14,4) on the daily close, filled at the next day's open; no hard stop, no time
stop. Sixteen slots at 6.25% of NAV.

TWO THINGS THAT LOOK LIKE DETAILS AND ARE NOT.

1. THE SPLIT CUT. `market_data.db` is not retroactively split-adjusted, so a pre-split row
   sits at the old price scale and fakes an all-time high that can never be reached again.
   research/161 truncates each series to the bars AFTER the last day-over-day fall worse
   than -35%, and so does this. Genuine highs on names that split are lost along with the
   fakes; that is the price of not trading a data artefact.

2. THE FUND EXCLUSION IS BY NAME, NOT BY PATTERN. research/161's own scan used a substring
   list ('GOLD', 'SILVER', 'NIFTY', ...). research/158 showed that rule both lets the
   2023-25 gold/silver fund wave through as if it were a company (EGOLD, TATAGOLD,
   SILVER1 ...) and deletes real companies whose names merely contain the word (SKYGOLD,
   GOLDIAM, SILVERTUC, GOLDTECH). The live book uses the curated list in
   `backtest_data/etf_exclusions.json`. It is a deliberate, declared deviation and the
   replication gate quantifies it as its own bucket rather than burying it.

Nothing in this module places an order, writes state, or touches the network.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
ETF_JSON = ROOT / 'backtest_data' / 'etf_exclusions.json'

# ---- the adopted cell (research/161 §1, frozen by research/164 build164.py) --------
X_MIN = 60              # prior ATH close must be at least this many bars old
DEPTH_MIN = 20.0        # and the close must have fallen at least this % below it
LIQ_MIN_CR = 2.0        # 20-day median traded value, Rs cr
REARM = 60              # bars before the same symbol may signal again
SPLIT_DOWN = -0.35      # series truncated after the last fall worse than this
SPLIT_UP = 0.50         # a trigger-day move bigger than this is a data event, not a trade
VOL_MED = 20
MIN_BARS = 90
SLOTS = 16
SLOT_PCT = 0.0625
ST_PERIOD, ST_MULT = 14, 4.0
DATA_EVENT_DROP = -0.40  # hold + alert, never sell into a split (services/ipo_paper.py)

# research/161 `ath_events.py` dropped this one name: its series is identical to CRESTO's.
DROP = {'SILLYMONKS'}

# research/161's OWN substring rule, kept ONLY so the replication gate can run the engine
# under the study's exact universe before switching to the curated list.
STUDY_ETF_PAT = ('BEES', 'ETF', 'IETF', 'GOLD', 'SILVER', 'LIQUID', 'NIFTY', 'SENSEX',
                 'BANKNIFTY', 'MIDCAP', 'SMALLCAP', 'INDEX', 'MAFANG', 'HNGSNGBEES')


def excluded_names():
    """The curated fund list. Falls back to the study pattern only if the file is missing."""
    try:
        return set(json.load(open(ETF_JSON))['by_name'])
    except Exception:
        return set()


def is_fund(sym, names=None, study_mode=False):
    if study_mode:
        return any(p in sym.upper() for p in STUDY_ETF_PAT)
    return sym in (names if names is not None else excluded_names())


# ------------------------------------------------------------------ indicators
def supertrend_dir(high, low, close, period=ST_PERIOD, mult=ST_MULT):
    """Direction array (+1 long / -1 out). Copied from research/161 bt_core.supertrend_dir.

    Byte-for-byte the study's function, deliberately: an exit that is 'basically the same
    SuperTrend' is a different book. Wilder-smoothed ATR seeded with the simple mean of the
    first `period` true ranges, bands on hl2, the usual ratchet, and the first period+1 bars
    forced long because the band does not exist yet.
    """
    n = len(close)
    out = np.zeros(n, dtype=np.int8)
    if n <= period + 2:
        return out
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.full(n, np.nan)
    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = (high + low) / 2.0
    up, dn = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = np.copy(up), np.copy(dn)
    d = np.ones(n, dtype=np.int8)
    for i in range(period + 1, n):
        fu[i] = up[i] if (up[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = dn[i] if (dn[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        d[i] = (-1 if close[i] < fl[i] else 1) if d[i - 1] == 1 else (1 if close[i] > fu[i] else -1)
    d[:period + 1] = 1
    return d


def supertrend_full(high, low, close, period=ST_PERIOD, mult=ST_MULT):
    """(direction, line) - the same computation, also returning the visible trail level.

    The line is what Arun reads on the dry-run table ('how far is it from the stop'), so it
    has to be the band the direction actually flipped on: the lower band while long, the
    upper band while out.
    """
    n = len(close)
    d = supertrend_dir(high, low, close, period, mult)
    line = np.full(n, np.nan)
    if n <= period + 2:
        return d, line
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.full(n, np.nan)
    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = (high + low) / 2.0
    up, dn = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = np.copy(up), np.copy(dn)
    for i in range(period + 1, n):
        fu[i] = up[i] if (up[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = dn[i] if (dn[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        line[i] = fl[i] if d[i] == 1 else fu[i]
    return d, line


# ------------------------------------------------------------------ data access
def connect():
    return sqlite3.connect('file:%s?mode=ro' % DB, uri=True)


def universe(con, study_mode=False, min_bars=MIN_BARS):
    """Symbols with enough history, funds removed. Sorted, so every run is reproducible."""
    syms = [r[0] for r in con.execute(
        "SELECT symbol FROM market_data_unified WHERE timeframe='day' "
        "GROUP BY symbol HAVING COUNT(*)>=? ORDER BY symbol", (min_bars,))]
    names = None if study_mode else excluded_names()
    return [s for s in syms
            if s not in DROP and not is_fund(s, names, study_mode)]


def load_bars(con, sym, asof=None):
    """One symbol's daily bars, cleaned and split-cut. Returns (df, was_cut) or (None, _)."""
    d = pd.read_sql_query(
        "SELECT date,open,high,low,close,volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date", con, params=(sym,))
    if d.empty:
        return None, 0
    # De-duplicate on the RAW date string, exactly as research/161 `ath_events.py` does.
    # Truncating to 10 chars first would merge rows the study kept apart, and the whole
    # point of this module is that the live scan and the study see the same series.
    if asof:
        d = d[d['date'].astype(str).str[:10] <= asof]
    d = d[(d['volume'] > 0) & (d['close'] > 0)].drop_duplicates(subset='date', keep='last')
    if len(d) < MIN_BARS:
        return None, 0
    r = d['close'].pct_change()
    hits = r.index[r < SPLIT_DOWN]
    cut = int(len(hits) > 0)
    if cut:
        d = d.loc[hits.max():]
    d = d.reset_index(drop=True)
    if len(d) < MIN_BARS:
        return None, cut
    return d, cut


# ------------------------------------------------------------------ the signal
def raw_events(d, sym, cut, allow_last_bar=False):
    """Every NEW all-time-high CLOSE in `d`, with the attributes the filters need.

    This is research/161 `ath_events.py`'s inner loop, unchanged except for
    `allow_last_bar`: the study needed a next bar to exist because it booked the entry
    open, whereas the live scan fires on TODAY's close and the fill has not happened yet.
    With `allow_last_bar=False` the two produce identical rows.
    """
    dates = d['date'].to_numpy()
    o = d['open'].to_numpy(np.float64)
    c = d['close'].to_numpy(np.float64)
    v = d['volume'].to_numpy(np.float64)
    n = len(c)
    ret1 = np.empty(n)
    ret1[0] = 0.0
    ret1[1:] = c[1:] / c[:-1] - 1.0
    med = pd.Series(v).rolling(VOL_MED, min_periods=VOL_MED).median().shift(1).to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        volx = np.where(med > 0, v / med, np.nan)
    tv20 = pd.Series(c * v).rolling(20, min_periods=10).median().to_numpy()

    out = []
    ath_idx = 0
    min_since = c[0]
    for t in range(1, n):
        prev_ath = c[ath_idx]
        if c[t] > prev_ath:
            has_next = t + 1 < n
            if ((has_next or (allow_last_bar and t == n - 1))
                    and np.isfinite(tv20[t]) and tv20[t] >= LIQ_MIN_CR * 1e7
                    and abs(ret1[t]) <= SPLIT_UP and np.isfinite(volx[t])):
                out.append(dict(
                    symbol=sym, trigger_date=str(dates[t])[:10],
                    trigger_close=round(c[t], 2), prev_ath=round(prev_ath, 2),
                    prev_ath_date=str(dates[ath_idx])[:10], x_bars=t - ath_idx,
                    depth_pct=round(100.0 * (prev_ath - min_since) / prev_ath, 2),
                    vol_mult=round(float(volx[t]), 2),
                    tv20_cr=round(float(tv20[t]) / 1e7, 2),
                    trigger_day_move_pct=round(100 * ret1[t], 2),
                    entry_date=str(dates[t + 1])[:10] if has_next else None,
                    entry_open=round(o[t + 1], 2) if has_next else None,
                    hist_bars=t, split_cut=cut))
            ath_idx = t
            min_since = c[t]
        elif c[t] < min_since:
            min_since = c[t]
    return out


def qualify(rows):
    """Apply the adopted filters and the 60-bar re-arm to ONE symbol's raw events.

    Order matters and is the study's: filter FIRST, re-arm SECOND (`build164.rearm`),
    because which event is 'first' depends on the filter. Greedy, earliest-first, on
    `hist_bars` - the bar index inside the split-cut series.
    """
    keep, last = [], -10 ** 9
    for r in sorted(rows, key=lambda x: x['hist_bars']):
        if (r['tv20_cr'] >= LIQ_MIN_CR and r['x_bars'] >= X_MIN
                and r['depth_pct'] >= DEPTH_MIN and r['hist_bars'] - last >= REARM):
            keep.append(r)
            last = r['hist_bars']
    return keep


def scan(asof=None, study_mode=False, symbols=None, allow_last_bar=True, progress=None):
    """Every qualifying Base Age signal in the whole universe, up to and including `asof`.

    `study_mode=True` swaps the curated fund list for research/161's substring pattern and
    turns off `allow_last_bar`, which is exactly the study's configuration - that is what
    the replication gate runs.
    """
    con = connect()
    try:
        syms = symbols if symbols is not None else universe(con, study_mode)
        out = []
        for i, s in enumerate(syms, 1):
            d, cut = load_bars(con, s, asof)
            if d is None:
                continue
            out.extend(qualify(raw_events(d, s, cut,
                                          allow_last_bar=allow_last_bar and not study_mode)))
            if progress and i % progress == 0:
                print('  [%d/%d] %d signals so far' % (i, len(syms), len(out)), flush=True)
    finally:
        con.close()
    return out


def last_session(con=None):
    """The most recent date any daily bar exists for. The scan's 'today'."""
    own = con is None
    con = con or connect()
    try:
        return con.execute(
            "SELECT MAX(date) FROM market_data_unified WHERE timeframe='day'").fetchone()[0][:10]
    finally:
        if own:
            con.close()


def st_state(sym, asof=None, proxy=None, con=None):
    """(direction, line, last_close, last_date) for one symbol under SuperTrend(14,4).

    `proxy` is an optional (open, high, low, close) tuple appended as TODAY's still-forming
    bar, which is how the 15:18 check reads the rule before the close exists. Without it the
    answer is the official one. The series is split-cut exactly as the entry scan is, so the
    exit and the entry agree about what the price history even is.
    """
    own = con is None
    con = con or connect()
    try:
        d, _ = load_bars(con, sym, asof)
    finally:
        if own:
            con.close()
    if d is None or len(d) <= ST_PERIOD + 2:
        return None, None, None, None
    h = d['high'].to_numpy(np.float64)
    l = d['low'].to_numpy(np.float64)
    c = d['close'].to_numpy(np.float64)
    last_date = str(d['date'].iloc[-1])[:10]
    if proxy is not None:
        po, ph, pl, pc = proxy
        h = np.append(h, ph)
        l = np.append(l, pl)
        c = np.append(c, pc)
        last_date = 'proxy'
    dirs, line = supertrend_full(h, l, c)
    return int(dirs[-1]), float(line[-1]), float(c[-1]), last_date
