"""
Blue-sky replica v2 (research/142 Phase 3) — decoded rules, seed-ensemble, 2006-2025.

Decoded rules (validated trade-exact in Phases 1-2):
  signal : a CLOSE above the prior all-time-high close (setup: prev close within 20%
           of ATH-close and below it; IBD-RS percentile >= 70; 20d median traded
           value >= Rs 5cr) -> fill AT the pivot (ATH-close)
  stop   : close <= buy*0.92 -> exit at close
  trail  : close < SMA50 -> exit at that close (not on entry day)
  book   : 8 slots, size = min(18.75%, 30%) of equity, cash-constrained, pyramiding ok

v2 additions: --start/--end, --ensemble N (N random-selection seeds, one load),
weak-market gate (NIFTYBEES < SMA200 blocks entries), ETF exclusion, NIFTYBEES
benchmark, coverage-by-year print, numpy sim core.
"""
import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'

CAPITAL = 1_000_000
SLOTS = 8
SIZE_PCT, CAP_PCT = 0.1875, 0.30
STOP = 0.08
TV_FLOOR = 5e7
ETF_RE = re.compile(r'(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)')


def load_frames(base_start, trail_sma=50):
    conn = sqlite3.connect(str(DB))
    syms = [r[0] for r in conn.execute(
        "select symbol from (select symbol, count(*) n from market_data_unified "
        "where timeframe='day' group by symbol) where n >= 260")]
    print(f'{len(syms)} symbols with >=260 daily rows', flush=True)
    cols = {}
    t0 = time.time()
    for i, s in enumerate(syms):
        df = pd.read_sql_query(
            "select date, open, high, close, volume from market_data_unified "
            "where symbol=? and timeframe='day' order by date", conn, params=(s,))
        df['date'] = pd.to_datetime(df['date'].str[:10])
        df = df.drop_duplicates('date').set_index('date').sort_index()
        ath_prev = df['close'].shift(1).cummax()
        s50 = df['close'].rolling(trail_sma).mean()
        tv = (df['close'] * df['volume']).rolling(20).median()
        m = df.index >= base_start
        if not m.any():
            continue
        cols[s] = dict(close=df.loc[m, 'close'], high=df.loc[m, 'high'],
                       open=df.loc[m, 'open'], athcp=ath_prev[m],
                       sma50=s50[m], tv20=tv[m])
        if (i + 1) % 600 == 0:
            print(f'  loaded {i+1}/{len(syms)} ({time.time()-t0:.0f}s)', flush=True)
    conn.close()
    w = {k: pd.DataFrame({s: v[k] for s, v in cols.items()}).astype('float32')
         for k in ('close', 'high', 'open', 'athcp', 'sma50', 'tv20')}
    print(f'wide frames: {w["close"].shape} ({time.time()-t0:.0f}s)', flush=True)
    return w


def simulate(seed, sel, days_idx, dates, C, H, O, ATH, S50, RS, TVp, TRIG, weak_arr,
             fill_realistic, cost, stop=STOP, slots=SLOTS, size_pct=SIZE_PCT,
             stcg=0.0, ltcg=0.125, fill_close=False):
    rng = np.random.default_rng(seed)
    cash = float(CAPITAL)
    positions = []      # (col, entry_i, buy, qty)
    trades = []
    equity = np.empty(len(days_idx), dtype=float)
    passed_up = 0
    tax_yr_gain = 0.0        # net realized gains this calendar year (Rs)
    cur_year = dates[days_idx[0]].year

    for k, i in enumerate(days_idx):
        yr = dates[i].year
        if stcg and yr != cur_year:
            if tax_yr_gain > 0:
                cash -= tax_yr_gain            # tax accrued at trade level below
            tax_yr_gain = 0.0
            cur_year = yr
        # entries
        if not weak_arr[i]:
            cand = np.nonzero(TRIG[i])[0]
            if len(cand):
                mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                          for c, _, b, q in positions)
                eq = cash + mtm
                if sel == 'rs':
                    cand = cand[np.argsort(-np.nan_to_num(RS[i, cand]))]
                elif sel == 'tv':
                    cand = cand[np.argsort(-np.nan_to_num(TVp[i, cand]))]
                elif sel == 'random':
                    cand = rng.permutation(cand)
                for c in cand:
                    if len(positions) >= slots:
                        passed_up += 1
                        continue
                    piv = float(ATH[i, c])
                    if fill_close:
                        fill = float(C[i, c])   # buy at the signal day's CLOSE (EOD-buy mechanic)
                    else:
                        fill = max(piv, float(O[i, c])) if fill_realistic else piv
                    size = size_pct * eq
                    qty = int(size / fill)
                    if qty < 1 or cash < qty * fill * (1 + cost):
                        passed_up += 1
                        continue
                    cash -= qty * fill * (1 + cost)
                    positions.append((c, i, fill, qty))
        # exits at close
        still = []
        for c, ei, b, q in positions:
            cl = C[i, c]
            if np.isnan(cl):
                still.append((c, ei, b, q))
                continue
            reason = None
            if cl <= b * (1 - stop):
                reason = 'stop_8pct'
            elif i > ei and not np.isnan(S50[i, c]) and cl < S50[i, c]:
                reason = 'trail_50d'
            if reason:
                cash += q * float(cl) * (1 - cost)
                if stcg:
                    pnl = q * (float(cl) - b)
                    held = (dates[i] - dates[ei]).days
                    rate = ltcg if held > 365 else stcg
                    tax_yr_gain += rate * pnl   # negative pnl offsets within the year
                trades.append((c, ei, i, b, float(cl), reason))
            else:
                still.append((c, ei, b, q))
        positions = still
        mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                  for c, _, b, q in positions)
        equity[k] = cash + mtm

    last = days_idx[-1]
    for c, ei, b, q in positions:
        cl = C[last, c]
        trades.append((c, ei, last, b, float(cl) if not np.isnan(cl) else b, 'open_marked'))
    return equity, trades, passed_up


def stats_from(equity, dates_used, trades, capital):
    e = pd.Series(equity, index=dates_used)
    yrs = (dates_used[-1] - dates_used[0]).days / 365.25
    cagr = (e.iloc[-1] / capital) ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    rets = np.array([t[4] / t[3] - 1 for t in trades])
    yearly = e.groupby(e.index.year).last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / capital - 1
    return dict(final=e.iloc[-1], x=e.iloc[-1] / capital, cagr=cagr * 100, dd=dd * 100,
                n=len(trades), win=(rets > 0).mean() * 100 if len(rets) else 0,
                mean=rets.mean() * 100 if len(rets) else 0,
                median=np.median(rets) * 100 if len(rets) else 0,
                yearly={int(k): round(v * 100, 1) for k, v in yr.items()}), e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--end', default='2025-12-31')
    ap.add_argument('--ensemble', type=int, default=0, help='N random-selection seeds')
    ap.add_argument('--selection', choices=['rs', 'tv', 'alpha', 'random'], default='rs')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fill-realistic', action='store_true')
    ap.add_argument('--cost', type=float, default=0.0, help='bps per side')
    ap.add_argument('--rs-min', type=float, default=70.0)
    ap.add_argument('--fill-close', action='store_true', help='entry at the signal-day close instead of the pivot')
    ap.add_argument('--stcg', action='store_true', help='model 20% STCG / 12.5% LTCG on net realized gains')
    ap.add_argument('--stop', type=float, default=8.0, help='stop %')
    ap.add_argument('--trail-sma', type=int, default=50)
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--gate-sma', type=int, default=200)
    ap.add_argument('--mcap-floor', type=float, default=0,
                    help='point-in-time mcap floor in CRORES (shares-const proxy from snapshot)')
    ap.add_argument('--skip-weak', action='store_true')
    ap.add_argument('--poke-trigger', action='store_true')
    ap.add_argument('--tag', default='p3')
    a = ap.parse_args()
    cost = a.cost / 10000.0

    base_start = (pd.Timestamp(a.start) - pd.Timedelta(days=550)).strftime('%Y-%m-%d')
    w = load_frames(base_start, trail_sma=a.trail_sma)
    close, high, open_, athcp, sma50, tv20 = (w[k] for k in
                                              ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
    # coverage by year (survivorship visibility)
    for y in range(int(a.start[:4]), 2026, 3):
        d0 = close.index[close.index.searchsorted(pd.Timestamp(f'{y}-01-05'))]
        print(f'coverage {y}: {int(close.loc[d0].notna().sum())} symbols priced', flush=True)

    etf_cols = [c for c in close.columns if ETF_RE.search(c)]
    tv_prev = tv20.shift(1)
    prev_close = close.shift(1)
    eligible = tv_prev >= TV_FLOOR
    eligible[etf_cols] = False
    if a.mcap_floor:
        import json
        snap = json.load(open(STUDY / 'results' / 'mcap_snapshot.json'))
        shares = pd.Series({s: v['mcap'] / v['px'] for s, v in snap.items()
                            if v.get('mcap') and v.get('px')}).reindex(close.columns)
        known = int(shares.notna().sum())
        print(f'mcap proxy: {known}/{len(close.columns)} symbols with shares; '
              f'unknowns EXCLUDED under the floor', flush=True)
        mcap_prev = prev_close.mul(shares, axis=1)
        eligible &= (mcap_prev >= a.mcap_floor * 1e7)

    r63 = close / close.shift(63) - 1
    r126 = close / close.shift(126) - 1
    r189 = close / close.shift(189) - 1
    r252 = close / close.shift(252) - 1
    score = (2 * r63 + r126 + r189 + r252).where(eligible)
    rs = (score.rank(axis=1, pct=True) * 100).shift(1)

    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & eligible & (rs >= a.rs_min)
    trig = setup & ((high >= athcp) if a.poke_trigger else (close > athcp)) & athcp.notna()

    nb = close.get('NIFTYBEES')
    if a.skip_weak:
        if nb is None:
            sys.exit('NIFTYBEES missing for --skip-weak')
        # NaN-robust: compute on the traded series only, then re-align. A single
        # phantom/missing row in the union index otherwise NaN-poisons every
        # rolling window after it and silently disables the gate (found
        # 2026-09-03: Kite holiday placeholder rows on 2026-01-15 killed the
        # gate from late-Apr-2026 in every prior run).
        nbs = nb.dropna()
        weak = (nbs < nbs.rolling(a.gate_sma).mean()).shift(1)
        weak = weak.reindex(close.index).ffill().fillna(False).astype(bool)
        weak_arr = weak.values
    else:
        weak_arr = np.zeros(len(close.index), dtype=bool)

    dates = close.index
    days_idx = np.array([i for i, d in enumerate(dates)
                         if a.start <= str(d.date()) <= a.end])
    dates_used = dates[days_idx]
    C, H, O = close.values, high.values, open_.values
    ATH, S50 = athcp.values, sma50.values
    RSv, TVv, TRIGv = rs.values, tv_prev.values, trig.fillna(False).values

    n_signals = int(TRIGv[days_idx].sum())
    print(f'\n=== {a.tag} ({a.start}->{a.end} gate={"ON" if a.skip_weak else "OFF"} '
          f'fill={"real" if a.fill_realistic else "pivot"} cost={a.cost}bps) ===')
    print(f'signals in period: {n_signals}')

    runs = ([(s, 'random') for s in range(1, a.ensemble + 1)] if a.ensemble
            else [(a.seed, a.selection)])
    all_stats, eq_curves = [], {}
    for seed, sel in runs:
        t0 = time.time()
        equity, trades, passed = simulate(seed, sel, days_idx, dates, C, H, O, ATH,
                                          S50, RSv, TVv, TRIGv, weak_arr,
                                          a.fill_realistic, cost,
                                          stop=a.stop / 100.0, slots=a.slots,
                                          stcg=0.20 if a.stcg else 0.0,
                                          fill_close=a.fill_close)
        st, e = stats_from(equity, dates_used, trades, CAPITAL)
        st['seed'] = seed
        all_stats.append(st)
        eq_curves[f'seed{seed}'] = e
        print(f"seed {seed}: {st['x']:8.2f}x  CAGR {st['cagr']:5.1f}%  DD {st['dd']:6.1f}%  "
              f"trades {st['n']:4d}  win {st['win']:.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(eq_curves).to_csv(STUDY / 'results' / f'replica_{a.tag}_equity.csv')
    sdf = pd.DataFrame(all_stats)
    if len(sdf) > 1:
        print(f"\nENSEMBLE ({len(sdf)} seeds): terminal x median {sdf.x.median():.2f} "
              f"[{sdf.x.min():.2f} .. {sdf.x.max():.2f}]  CAGR median {sdf.cagr.median():.1f}% "
              f"[{sdf.cagr.min():.1f} .. {sdf.cagr.max():.1f}]  DD median {sdf.dd.median():.1f}% "
              f"worst {sdf.dd.min():.1f}%")
        ymed = pd.DataFrame([s['yearly'] for s in all_stats]).median()
        print('median yearly %:', {int(k): round(v, 1) for k, v in ymed.items()})

    if nb is not None:
        b = nb.loc[dates_used].dropna()
        byrs = (b.index[-1] - b.index[0]).days / 365.25
        bcagr = (b.iloc[-1] / b.iloc[0]) ** (1 / byrs) - 1
        bdd = (b / b.cummax() - 1).min()
        print(f'benchmark NIFTYBEES B&H: {b.iloc[-1]/b.iloc[0]:.2f}x  CAGR {bcagr*100:.1f}%  '
              f'maxDD {bdd*100:.1f}%')
    print('reference research/75 momentum: 31.9% net CAGR, DD -31.6% (2006-2026)')


if __name__ == '__main__':
    main()
