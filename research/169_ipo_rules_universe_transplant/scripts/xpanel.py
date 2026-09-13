# -*- coding: utf-8 -*-
"""research/169 - IPO Spec A rules transplanted to size universes: panel + signal + null + runner.

The book simulator is research/153's ipo_replay.simulate_ipo, UNCHANGED (the same one r/167 and
r/168 use). What is new is the panel: every non-fund NSE symbol, a causal monthly traded-value
rank (the research/41 method) for the size universes, per-symbol NaN-robust rolling windows,
phantom-holiday dates dropped, and split back-adjustment. Each defense can be toggled off so the
r/167 panel can be reproduced.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
HERE = ROOT / 'research/169_ipo_rules_universe_transplant'
RES = HERE / 'results'
RES.mkdir(parents=True, exist_ok=True)
(RES / 'navs').mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'research/153_ipo_base/scripts'))
sys.path.insert(0, str(ROOT / 'research/142_bananapatterns_replication/scripts'))
import bluesky_replay as br   # noqa: E402
import ipo_replay as ir       # noqa: E402

DB = ROOT / 'backtest_data/market_data.db'
PANEL_START = '2005-06-01'
TV_FLOOR = 5e7
SEEDS = list(range(1, 31))
WINDOWS = {'w2': ('2006-01-01', '2026-09-04'),
           'wa': ('2006-01-01', '2015-12-31'),
           'wb': ('2016-01-01', '2026-09-04')}
NULL_RNG = 20260912
TRAILS = (20, 30, 50, 75, 100)
LS = (25, 50)
UNIVERSES = {'top50': (1, 50), 'top100': (1, 100), 'top200': (1, 200), 'top500': (1, 500),
             'mid101_250': (101, 250), 'small251_500': (251, 500),
             'rank501plus': (501, 10 ** 9), 'all': None}
ADJ_DOWN, ADJ_UP = 0.60, 1.80


def shift(a):
    b = np.zeros_like(a) if a.dtype == bool else np.full_like(a, np.nan)
    b[1:] = a[:-1]
    return b


class Panel:
    def __init__(self, verbose=True):
        t0 = time.time()
        ld = pd.read_csv(ROOT / 'research/153_ipo_base/results/listing_dates.csv')
        ld = ld[ld.accepted]
        self.listing = dict(zip(ld.symbol, pd.to_datetime(ld.list_date)))
        con = sqlite3.connect(str(DB))
        meta = pd.read_sql_query(
            "select symbol, min(substr(date,1,10)) d0, count(*) n, "
            "sum(case when substr(date,1,10) < ? then 1 else 0 end) npre "
            "from market_data_unified where timeframe='day' group by symbol", con,
            params=(PANEL_START,))
        self.meta = meta[meta.n >= 60].set_index('symbol')
        q = pd.read_sql_query(
            "select symbol, substr(date,1,10) d, open o, high h, low l, close c, volume v "
            "from market_data_unified where timeframe='day' and date >= ? "
            "order by symbol, date", con, params=(PANEL_START,))
        con.close()
        q = q[q.symbol.isin(self.meta.index) & (q.d >= PANEL_START)]
        q = q.drop_duplicates(['symbol', 'd'], keep='first')
        q['d'] = pd.to_datetime(q['d'])
        lst = q.symbol.map(self.listing)
        q = q[lst.isna() | (q.d >= lst)]
        g = q.assign(z=(q.v == 0)).groupby('d').agg(n=('z', 'size'), z=('z', 'sum'))
        med = g.n.rolling(21, center=True, min_periods=5).median()
        self.phantom = list(g.index[(g.z / g.n > 0.9) & (g.n < 0.5 * med)])
        self.q = q
        funds = set(json.load(open(ROOT / 'backtest_data/etf_exclusions.json'))['symbols'])
        self.funds = funds | {s for s in self.meta.index if br.ETF_RE.search(s)}
        if verbose:
            print('[panel] %d rows, %d symbols, phantom dates %s (%.0fs)'
                  % (len(q), q.symbol.nunique(), [str(x.date()) for x in self.phantom],
                     time.time() - t0), flush=True)

    def build(self, adjust=True, robust=True, drop_phantom=True, verbose=True):
        t0 = time.time()
        for k in ('C', 'O', 'H', 'L', 'PIV', 'LO', 'SMA', 'TVp', 'PREVC', 'RANK', 'AGE',
                  'ELIG', 'BARS'):
            if hasattr(self, k):
                delattr(self, k)
        q = self.q
        if drop_phantom and self.phantom:
            q = q[~q.d.isin(self.phantom)]
        W = {k: q.pivot(index='d', columns='symbol', values=k)
             for k in ('o', 'h', 'l', 'c', 'v')}
        cols = list(W['c'].columns)
        dates = W['c'].index
        RC = W['c'].to_numpy(np.float64)
        RO, RH, RL = (W[k].to_numpy(np.float64) for k in ('o', 'h', 'l'))
        RV = W['v'].to_numpy(np.float64)
        nb_raw = W['c']['NIFTYBEES'].dropna()
        del W
        T, N = RC.shape

        def f32():
            return np.full((T, N), np.nan, dtype=np.float32)

        C, O, H, L_, TVp, PREVC, TVR = f32(), f32(), f32(), f32(), f32(), f32(), f32()
        BARS = np.zeros((T, N), dtype=np.float32)
        PIV = {b: f32() for b in LS}
        LO = {b: f32() for b in LS}
        SMA = {n: f32() for n in TRAILS}
        events = []
        npre = self.meta.npre.to_dict()
        for j, s in enumerate(cols):
            idx = np.nonzero(np.isfinite(RC[:, j]))[0]
            if len(idx) < 2:
                continue
            c = RC[idx, j]
            o, h, l = RO[idx, j], RH[idx, j], RL[idx, j]
            v = np.nan_to_num(RV[idx, j])
            tv = pd.Series(c * v)
            tv20p = tv.rolling(20).median().shift(1).to_numpy()
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
                                           tv20_cr=round(float(tv20p[k]) / 1e7, 2)
                                           if np.isfinite(tv20p[k]) else None))
            ca, la = c * fac, l * fac
            C[idx, j] = ca
            O[idx, j] = o * fac
            H[idx, j] = h * fac
            L_[idx, j] = la
            TVp[idx, j] = tv20p
            TVR[idx, j] = tv.rolling(126, min_periods=60).median().shift(1).to_numpy()
            BARS[idx, j] = np.arange(1, len(idx) + 1) + (0 if s in self.listing
                                                        else npre.get(s, 0))
            if robust:
                cs, ls_ = pd.Series(ca), pd.Series(la)
                PREVC[idx, j] = cs.shift(1).to_numpy()
                for b in LS:
                    PIV[b][idx, j] = cs.rolling(b).max().shift(1).to_numpy()
                    LO[b][idx, j] = ls_.rolling(b).min().shift(1).to_numpy()
                for n in TRAILS:
                    SMA[n][idx, j] = cs.rolling(n).mean().to_numpy()
        if not robust:
            Cd, Ld = pd.DataFrame(C, index=dates), pd.DataFrame(L_, index=dates)
            PREVC = Cd.shift(1).to_numpy(np.float32)
            for b in LS:
                PIV[b] = Cd.rolling(b).max().shift(1).to_numpy(np.float32)
                LO[b] = Ld.rolling(b).min().shift(1).to_numpy(np.float32)
            for n in TRAILS:
                SMA[n] = Cd.rolling(n).mean().to_numpy(np.float32)
            TVp = pd.DataFrame(RC * RV, index=dates).rolling(20).median().shift(1) \
                .to_numpy(np.float32)
            BARS = np.isfinite(C).cumsum(axis=0).astype(np.float32)
            del Cd, Ld
        del RC, RO, RH, RL, RV

        EPOCH = pd.Timestamp('1970-01-01')
        dnum = np.asarray((dates - EPOCH).days, dtype=np.float64)
        lday = np.empty(N)
        for j, s in enumerate(cols):
            if s in self.listing:
                lday[j] = (self.listing[s] - EPOCH).days
            elif npre.get(s, 0) > 0 or str(self.meta.d0.get(s, '9999')) <= '2005-06-30':
                lday[j] = -1e6
            else:
                lday[j] = (pd.Timestamp(self.meta.d0[s]) - EPOCH).days
        self.AGE = (dnum[:, None] - lday[None, :]).astype(np.float32)
        self.VET = np.array([s in self.listing for s in cols])
        self.FUND = np.array([s in self.funds for s in cols])
        with np.errstate(invalid='ignore'):
            self.ELIG = (TVp >= TV_FLOOR) & ~self.FUND[None, :]

        mon = dates.to_period('M')
        starts = np.nonzero(np.r_[True, mon[1:] != mon[:-1]])[0]
        TVRf = pd.DataFrame(TVR).ffill(limit=5).to_numpy(np.float64)
        RANK = np.full((T, N), np.inf, dtype=np.float32)
        ends = list(starts[1:]) + [T]
        for a, b in zip(starts, ends):
            sc = TVRf[a].copy()
            sc[self.FUND] = np.nan
            ok = np.nonzero(np.isfinite(sc))[0]
            order = np.argsort(-sc[ok], kind='stable')
            row = np.full(N, np.inf)
            row[ok[order]] = np.arange(1, len(ok) + 1)
            RANK[a:b] = row
        del TVR, TVRf

        self.dates, self.cols = dates, cols
        self.C, self.O, self.H, self.L = C, O, H, L_
        self.TVp, self.PREVC, self.BARS, self.RANK = TVp, PREVC, BARS, RANK
        self.PIV, self.LO, self.SMA = PIV, LO, SMA
        self.events = pd.DataFrame(events)
        w = (nb_raw < nb_raw.rolling(150).mean()).shift(1)
        self.WEAK = w.reindex(dates).ffill().fillna(False).to_numpy(bool)
        self.NOWEAK = np.zeros(T, dtype=bool)
        ds = np.array([str(d.date()) for d in dates])
        self.days = {k: np.nonzero((ds >= a) & (ds <= b))[0] for k, (a, b) in WINDOWS.items()}
        self.flags = dict(adjust=adjust, robust=robust, drop_phantom=drop_phantom)
        if verbose:
            liq = int((self.events.tv20_cr.fillna(0) >= 5).sum()) if len(self.events) else 0
            print('[panel] built %s: %d x %d, split events %d (%d on >= Rs5cr names) (%.0fs)'
                  % (self.flags, T, N, len(self.events), liq, time.time() - t0), flush=True)
        return self


# ───────────────────────────────────────────── signals and nulls
def age_mask(P, age):
    if age == 'none':
        return None
    if age == 'vet_any':
        return P.VET[None, :] & (P.AGE > 0)
    kind, m = age[:2], float(age[2:])
    days = m * 30.44
    if kind == 'le':
        return P.VET[None, :] & (P.AGE > 0) & (P.AGE <= days)
    if kind == 'gt':
        return P.AGE > days
    raise ValueError(age)


def build_signals(P, uni, age, L=25, min_bars=60, max_depth=0.30):
    base = P.ELIG & (P.BARS >= min_bars)
    if UNIVERSES.get(uni) is not None:
        lo_, hi_ = UNIVERSES[uni]
        base &= (P.RANK >= lo_) & (P.RANK <= hi_)
    am = age_mask(P, age)
    if am is not None:
        base &= am
    piv, lo = P.PIV[L], P.LO[L]
    with np.errstate(invalid='ignore', divide='ignore'):
        depth = (piv - lo) / np.where(piv > 0, piv, np.nan)
        setup = base & (depth <= max_depth) & (P.PREVC < piv) & np.isfinite(piv)
        sig = setup & (P.C > piv)
    pvn, lon = shift(piv), shift(lo)
    with np.errstate(invalid='ignore'):
        trig = shift(sig) & (P.H >= pvn) & np.isfinite(pvn)
    return trig, pvn, lon, base


def build_null(P, trig, pvn, base):
    rng = np.random.default_rng(NULL_RNG)
    bsh = shift(base)
    with np.errstate(invalid='ignore'):
        reach = P.H >= pvn
    nper = trig.sum(axis=1)
    null = np.zeros_like(trig)
    for i in np.nonzero(nper)[0]:
        pool = np.nonzero(bsh[i] & reach[i] & np.isfinite(pvn[i]))[0]
        if not len(pool):
            continue
        k = min(int(nper[i]), len(pool))
        null[i, rng.choice(pool, size=k, replace=False)] = True
    return null


# ───────────────────────────────────────────── runner
def run(P, trig, lvl, lo, trail=50, windows=('w2', 'wa', 'wb'), seeds=SEEDS, cost=0.0025,
        stop=0.10, target=0.25, cash_yield=0.052, gate=True, keep=False):
    sma = P.SMA[trail]
    weak = P.WEAK if gate else P.NOWEAK
    res = {}
    for wk in windows:
        days = P.days[wk]
        du = P.dates[days]
        rows, navs, trs = [], [], []
        for sd in seeds:
            eq, trd, _, inv = ir.simulate_ipo(
                sd, days, P.dates, P.C, P.O, lvl, lo, sma, None, P.TVp, trig, weak,
                cost=cost, stop=stop, slots=8, size_pct=0.1875, target=target,
                fill_close=False, cash_yield=cash_yield)
            st, e = ir.stats_from(eq, du, trd, invested=inv)
            st.pop('yearly', None)
            rows.append(st)
            if keep and wk == 'w2':
                navs.append(e.to_numpy(np.float64))
                trs.append(trd)
        res[wk] = pd.DataFrame(rows)
        if keep and wk == 'w2':
            res['navs'] = np.vstack(navs)
            res['trades'] = trs
            res['dates'] = du
    return res


def summarize(res, cost=0.0025, null=None):
    out = {}
    for wk in ('w2', 'wa', 'wb'):
        if wk not in res:
            continue
        d = res[wk]
        mdd = float(d.dd.median())
        out.update({
            f'{wk}_cagr': round(float(d.cagr.median()), 2),
            f'{wk}_cagr_lo': round(float(d.cagr.min()), 2),
            f'{wk}_cagr_hi': round(float(d.cagr.max()), 2),
            f'{wk}_dd': round(mdd, 2), f'{wk}_dd_worst': round(float(d.dd.min()), 2),
            f'{wk}_calmar': round(float(d.cagr.median()) / abs(mdd), 3) if mdd else np.nan,
            f'{wk}_n': int(d.n.median()), f'{wk}_tpy': round(float(d.tpy.median()), 1),
            f'{wk}_win': round(float(d.win.median()), 1),
            f'{wk}_mean': round(float(d['mean'].median()), 3),
            f'{wk}_netexp': round(float(d['mean'].median()) - 200 * cost, 3),
            f'{wk}_avg_win': round(float(d.avg_win.median()), 2),
            f'{wk}_avg_loss': round(float(d.avg_loss.median()), 2),
            f'{wk}_hold_mean': round(float(d.hold.median()), 0),
            f'{wk}_inv': round(float(d.invested_pct.median()), 1),
            f'{wk}_streak': int(d.max_loss_streak.median())})
        if null is not None and wk in null:
            dl = d.cagr.values - null[wk].cagr.values
            out.update({
                f'{wk}_null_cagr': round(float(null[wk].cagr.median()), 2),
                f'{wk}_null_dd': round(float(null[wk].dd.median()), 2),
                f'{wk}_null_mean': round(float(null[wk]['mean'].median()), 3),
                f'{wk}_edge': round(float(np.median(dl)), 2),
                f'{wk}_edge_lo': round(float(dl.min()), 2),
                f'{wk}_edge_hi': round(float(dl.max()), 2),
                f'{wk}_wins': int((dl > 0).sum())})
    return out


def trade_diag(P, res):
    rows = [dict(x, seed=s) for s, t in zip(SEEDS, res['trades']) for x in t]
    if not rows:
        return {}
    tr = pd.DataFrame(rows)
    tr['frac_tv'] = 100 * tr.notional / tr.tv
    ages = P.AGE[tr.ei.values, tr.col.values]
    young = P.VET[tr.col.values] & (ages > 0) & (ages <= 6 * 30.44)
    last = P.days['w2'][-1]
    stale = tr[(tr.reason == 'open_marked').values
               & ~np.isfinite(P.C[last, tr.col.values])]
    p90 = float(tr.frac_tv.quantile(0.9))
    tops = tr.groupby('seed')['ret'].apply(lambda s: s.nlargest(10).sum())
    tot = tr.groupby('seed')['ret'].sum()
    ex10 = pd.Series({s: g.drop(g['ret'].nlargest(10).index)['ret'].mean()
                      for s, g in tr.groupby('seed')})
    rk = P.RANK[tr.ei.values, tr.col.values]
    return dict(hold_median=float(tr.held.median()),
                cap_med_pct_tv_10L=round(float(tr.frac_tv.median()), 3),
                cap_p90_pct_tv_10L=round(p90, 3),
                cap_p90_pct_tv_1cr=round(10 * p90, 2),
                book_L_at_p90_5pct=round(10 * 5.0 / p90, 1) if p90 > 0 else None,
                young6m_share_pct=round(100 * float(young.mean()), 1),
                rank_median=float(np.nanmedian(np.where(np.isfinite(rk), rk, np.nan)))
                if np.isfinite(rk).any() else None,
                distinct_names=int(tr.groupby('seed').col.nunique().median()),
                stale_open_per_seed=float(stale.groupby('seed').size().reindex(SEEDS)
                                          .fillna(0).median()),
                top10_share_pct=round(100 * float(tops.median() / tot.median()), 1)
                if tot.median() else None,
                mean_ex_top10=round(100 * float(ex10.median()), 2))


def save_navs(label, res):
    np.savez_compressed(RES / 'navs' / f'{label}.npz', navs=res['navs'].astype(np.float32),
                        dates=np.array([str(d.date()) for d in res['dates']]))
