# -*- coding: utf-8 -*-
"""research/172 - 52-week channel (Donchian) on Nifty 100: the price panel.

Modelled on research/169 xpanel.py, so the same data defences apply and are toggleable:
  * split back-adjustment  (single-day close ratio <=0.60 or >=1.80 -> adjust the past)
  * phantom-holiday drop   (sparse date where >90% of names print zero volume)
  * NaN-robust rolling     (every window computed on the symbol's own dropna'd series
                            and scattered back onto the master calendar)
  * fund exclusion by NAME (backtest_data/etf_exclusions.json) not by ticker regex
  * causal liquidity floor (20-day median traded value, shifted one day)
  * point-in-time size universes from a monthly traded-value rank (the research/41 method
    re-used by research/169) - the survivorship control for the official Nifty-100 list.

Everything is causal. No array indexed at t uses a bar after t.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
HERE = ROOT / 'research/172_52wk_channel_n100'
RES = HERE / 'results'
RES.mkdir(parents=True, exist_ok=True)

DB = ROOT / 'backtest_data/market_data.db'
PANEL_START = '2003-01-01'          # the DB has 2 symbols before 2003; nothing to lose
TRADE_START = '2006-01-02'          # matches research/168 window "wa/wb" so blends align
TRADE_END = '2026-09-11'
W1 = ('2006-01-02', '2015-12-31')
W2 = ('2016-01-01', '2026-09-11')

TV_FLOOR = 5e7                      # Rs 5 crore, 20-day median traded value
ADJ_DOWN, ADJ_UP = 0.60, 1.80

ENTRY_LOOKBACKS = (63, 126, 189, 252, 378, 504)
EXIT_LOOKBACKS = (21, 42, 63, 126, 189, 252)
SMA_WINDOWS = (15, 50)
BENCH = 'NIFTYBEES'


def shift1(a):
    b = np.full_like(a, np.nan) if a.dtype.kind == 'f' else np.zeros_like(a)
    b[1:] = a[:-1]
    return b


def supertrend_dir(high, low, close, period, mult):
    """+1 uptrend / -1 downtrend, computed on one symbol's own contiguous series."""
    n = len(close)
    out = np.ones(n, dtype=np.int8)
    if n <= period + 2:
        return out
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
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


def wilder_atr(high, low, close, period=14):
    n = len(close)
    atr = np.full(n, np.nan)
    if n <= period:
        return atr
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Panel:
    """Build once, reuse for every cell."""

    def __init__(self, adjust=True, drop_phantom=True, robust=True, verbose=True,
                 keep_rank=550):
        t0 = time.time()
        con = sqlite3.connect(str(DB))
        meta = pd.read_sql_query(
            "select symbol, min(substr(date,1,10)) d0, count(*) n from market_data_unified "
            "where timeframe='day' group by symbol", con)
        meta = meta[meta.n >= 120]
        q = pd.read_sql_query(
            "select symbol, substr(date,1,10) d, open o, high h, low l, close c, volume v "
            "from market_data_unified where timeframe='day' and date >= ? "
            "order by symbol, date", con, params=(PANEL_START,))
        con.close()
        q = q[q.symbol.isin(set(meta.symbol))]
        q = q.drop_duplicates(['symbol', 'd'], keep='first')

        # ---- universes (official CSVs = CURRENT membership -> survivorship, named sin)
        def off(fn):
            return set(pd.read_csv(ROOT / 'backtest_data' / fn).Symbol.astype(str))

        self.N50 = off('nifty50_official.csv')
        self.NN50 = off('niftynext50_official.csv')
        self.N100 = self.N50 | self.NN50
        self.N500 = off('nifty500_proxy.csv') if (ROOT / 'backtest_data/nifty500_proxy.csv').exists() else set()

        # ---- funds, by instrument NAME not ticker regex
        funds = set(json.load(open(ROOT / 'backtest_data/etf_exclusions.json'))['symbols'])
        self.funds = funds

        # ---- phantom holiday dates
        g = q.assign(z=(q.v == 0)).groupby('d').agg(n=('z', 'size'), z=('z', 'sum'))
        med = g.n.rolling(21, center=True, min_periods=5).median()
        self.phantom = list(g.index[(g.z / g.n > 0.9) & (g.n < 0.5 * med)])
        if drop_phantom and self.phantom:
            q = q[~q.d.isin(self.phantom)]

        # ---- restrict columns: everything we could possibly need, to fit in RAM.
        #      We need ALL symbols to compute the liquidity rank fairly, so rank first
        #      on a cheap traded-value pivot, then keep the union of (rank<=keep_rank
        #      ever) and the official lists and the benchmark.
        tv = (q.c * q.v).groupby([q.d, q.symbol]).first().unstack()
        tvr = tv.rolling(126, min_periods=60).median().shift(1)
        tvr = tvr.drop(columns=[c for c in tvr.columns if c in funds], errors='ignore')
        mon = pd.PeriodIndex(pd.to_datetime(tvr.index), freq='M')
        starts = np.nonzero(np.r_[True, mon[1:] != mon[:-1]])[0]
        ever = set()
        rank_rows = {}
        for a in starts:
            row = tvr.iloc[a].dropna()
            if not len(row):
                continue
            order = row.sort_values(ascending=False)
            rk = pd.Series(np.arange(1, len(order) + 1), index=order.index)
            rank_rows[tvr.index[a]] = rk
            ever |= set(order.index[:keep_rank])
        keep = sorted((ever | self.N100 | self.N500 | {BENCH}) & set(q.symbol.unique()))
        del tv, tvr
        q = q[q.symbol.isin(keep)]

        W = {k: q.pivot(index='d', columns='symbol', values=k)
             for k in ('o', 'h', 'l', 'c', 'v')}
        cols = list(W['c'].columns)
        dates = pd.to_datetime(W['c'].index)
        RC = W['c'].to_numpy(np.float64)
        RO, RH, RL = (W[k].to_numpy(np.float64) for k in ('o', 'h', 'l'))
        RV = W['v'].to_numpy(np.float64)
        nb_raw = W['c'][BENCH].copy()
        nb_raw.index = dates
        del W
        T, N = RC.shape

        def f32():
            return np.full((T, N), np.nan, dtype=np.float32)

        C, O, H, L_, TVp, PREVC = f32(), f32(), f32(), f32(), f32(), f32()
        ATR = f32()
        ATHC = f32()                                   # running all-time-high close
        BARS = np.zeros((T, N), dtype=np.int32)
        RS252 = f32()                                  # 252-day relative strength
        PIV_C = {b: f32() for b in ENTRY_LOOKBACKS}    # prior-close max
        PIV_H = {b: f32() for b in ENTRY_LOOKBACKS}    # prior-high max
        LO_C = {b: f32() for b in EXIT_LOOKBACKS}      # prior-close min
        LO_L = {b: f32() for b in EXIT_LOOKBACKS}      # prior-low min
        SMA = {n_: f32() for n_ in SMA_WINDOWS}
        EMA50 = f32()
        ST = {k: np.zeros((T, N), dtype=np.int8) for k in ('ST_7_3', 'ST_10_3', 'ST_14_4')}
        events = []

        for j, s in enumerate(cols):
            idx = np.nonzero(np.isfinite(RC[:, j]))[0]
            if len(idx) < 60:
                continue
            c = RC[idx, j]
            o, h, l = RO[idx, j], RH[idx, j], RL[idx, j]
            v = np.nan_to_num(RV[idx, j])
            tvs = pd.Series(c * v)
            tv20p = tvs.rolling(20, min_periods=20).median().shift(1).to_numpy()
            fac = np.ones(len(c))
            if adjust:
                with np.errstate(divide='ignore', invalid='ignore'):
                    r = c[1:] / c[:-1]
                ev = np.nonzero(((r <= ADJ_DOWN) | (r >= ADJ_UP)) & np.isfinite(r)
                                & (c[:-1] > 0) & (c[1:] > 0))[0] + 1
                if len(ev):
                    e = np.ones(len(c))
                    e[ev] = r[ev - 1]
                    rc = np.cumprod(e[::-1])[::-1]
                    fac[:-1] = rc[1:]
                    for k in ev:
                        events.append(dict(symbol=s, date=str(dates[idx[k]].date()),
                                           ratio=round(float(r[k - 1]), 4),
                                           tv20_cr=(round(float(tv20p[k]) / 1e7, 2)
                                                    if np.isfinite(tv20p[k]) else None)))
            ca, oa, ha, la = c * fac, o * fac, h * fac, l * fac
            C[idx, j] = ca
            O[idx, j] = oa
            H[idx, j] = ha
            L_[idx, j] = la
            TVp[idx, j] = tv20p
            BARS[idx, j] = np.arange(1, len(idx) + 1)
            cs, hs, ls_ = pd.Series(ca), pd.Series(ha), pd.Series(la)
            PREVC[idx, j] = cs.shift(1).to_numpy()
            ATHC[idx, j] = cs.cummax().to_numpy()
            RS252[idx, j] = (cs / cs.shift(252) - 1.0).to_numpy()
            for b in ENTRY_LOOKBACKS:
                PIV_C[b][idx, j] = cs.rolling(b, min_periods=b).max().shift(1).to_numpy()
                PIV_H[b][idx, j] = hs.rolling(b, min_periods=b).max().shift(1).to_numpy()
            for b in EXIT_LOOKBACKS:
                LO_C[b][idx, j] = cs.rolling(b, min_periods=b).min().shift(1).to_numpy()
                LO_L[b][idx, j] = ls_.rolling(b, min_periods=b).min().shift(1).to_numpy()
            for n_ in SMA_WINDOWS:
                SMA[n_][idx, j] = cs.rolling(n_, min_periods=n_).mean().to_numpy()
            EMA50[idx, j] = cs.ewm(span=50, adjust=False, min_periods=50).mean().to_numpy()
            ATR[idx, j] = wilder_atr(ha, la, ca, 14)
            ST['ST_7_3'][idx, j] = supertrend_dir(ha, la, ca, 7, 3.0)
            ST['ST_10_3'][idx, j] = supertrend_dir(ha, la, ca, 10, 3.0)
            ST['ST_14_4'][idx, j] = supertrend_dir(ha, la, ca, 14, 4.0)

        del RC, RO, RH, RL, RV

        # ---- point-in-time liquidity rank over the retained columns
        RANK = np.full((T, N), np.inf, dtype=np.float32)
        colpos = {s: j for j, s in enumerate(cols)}
        dstr = np.array([str(d.date()) for d in dates])
        rank_dates = sorted(rank_rows)
        ri = 0
        cur = np.full(N, np.inf, dtype=np.float32)
        for i in range(T):
            while ri < len(rank_dates) and rank_dates[ri] <= dstr[i]:
                rk = rank_rows[rank_dates[ri]]
                cur = np.full(N, np.inf, dtype=np.float32)
                for s, v_ in rk.items():
                    j = colpos.get(s)
                    if j is not None:
                        cur[j] = v_
                ri += 1
            RANK[i] = cur

        self.dates, self.cols = dates, cols
        self.dstr = dstr
        self.dnum = np.asarray((dates - pd.Timestamp('1970-01-01')).days, dtype=np.int64)
        self.T, self.N = T, N
        self.C, self.O, self.H, self.L = C, O, H, L_
        # CM = MARK price: the close forward-filled per symbol, so a holiday or a
        # missing bar marks the position at its last real price instead of at zero.
        # Signals keep using the raw C (NaN -> no signal); only the mark uses CM.
        # research/161 bt_core.py had to do exactly this; without it every gap in a
        # held name prints a fake -100% NAV spike.
        self.CM = pd.DataFrame(C).ffill().to_numpy(np.float32)
        self.TVp, self.PREVC, self.BARS, self.ATR = TVp, PREVC, BARS, ATR
        self.ATHC, self.RS252, self.RANK = ATHC, RS252, RANK
        self.PIV_C, self.PIV_H, self.LO_C, self.LO_L = PIV_C, PIV_H, LO_C, LO_L
        self.SMA, self.EMA50, self.ST = SMA, EMA50, ST
        self.events = pd.DataFrame(events)
        self.FUND = np.array([s in self.funds for s in cols])

        # ---- membership masks
        self.UNI = {
            'n50': np.array([s in self.N50 for s in cols]),
            'nn50': np.array([s in self.NN50 for s in cols]),
            'n100': np.array([s in self.N100 for s in cols]),
            'n500': np.array([s in self.N500 for s in cols]),
        }

        # ---- benchmark + gates (NIFTYBEES; NIFTY50 index only starts 2011)
        nb = nb_raw.reindex(dates).ffill()
        self.bench = nb.to_numpy(np.float64)
        for w in (100, 200):
            g_ = (nb > nb.rolling(w, min_periods=w).mean()).shift(1)
            setattr(self, 'GATE%d' % w, g_.reindex(dates).ffill().fillna(False).to_numpy(bool))
        self.GATE_NONE = np.ones(T, dtype=bool)

        self.days = {'full': self._slice(TRADE_START, TRADE_END),
                     'w1': self._slice(*W1), 'w2': self._slice(*W2)}
        self.flags = dict(adjust=adjust, drop_phantom=drop_phantom, robust=robust)
        if verbose:
            liq = int((self.events.tv20_cr.fillna(0) >= 5).sum()) if len(self.events) else 0
            print('[p172] %d x %d panel %s..%s | %s | split events %d (%d on >=Rs5cr names) '
                  '| phantom dates %s | %.0fs'
                  % (T, N, dstr[0], dstr[-1], self.flags, len(self.events), liq,
                     [str(x) for x in self.phantom], time.time() - t0), flush=True)

    def _slice(self, a, b):
        return np.nonzero((self.dstr >= a) & (self.dstr <= b))[0]

    # ------------------------------------------------------------------ masks
    def universe_mask(self, uni):
        """(T,N) bool - is symbol s a member of `uni` on day t."""
        if uni in self.UNI:
            return np.broadcast_to(self.UNI[uni][None, :], (self.T, self.N))
        if uni.startswith('pit'):                 # pit100 / pit50 / pit500
            k = int(uni[3:])
            return (self.RANK <= k)
        raise ValueError(uni)

    def eligible(self, uni, lookback):
        with np.errstate(invalid='ignore'):
            e = (self.TVp >= TV_FLOOR) & (self.BARS >= lookback + 1) & ~self.FUND[None, :]
        return e & self.universe_mask(uni)
