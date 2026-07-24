#!/usr/bin/env python3
"""RSI momentum-regime timing engine (research/72).

Long-only, close-basis daily system:
  ENTRY: RSI(len) close >= entry_th  -> buy at that close (if flat)
  EXIT : RSI(len) close <  exit_th   -> sell at that close (if long)
Flat = cash. Costs modelled as round-trip bps. Self-contained: reads the
canonical market_data.db. Designed to run on the VPS.

Optional entry gates (marginal-add filters, all default OFF):
  --ma_len N       : only ENTER if close > SMA(N)
  --adx_min X      : only ENTER if ADX(14) >= X
  --wrsi_min X     : only ENTER if weekly RSI(14) >= X
  --st_regime      : only ENTER if Supertrend(10,3) is bullish; also exit on ST flip
  --donchian N     : also exit if close < lowest-low(N)  (trail)
Filters gate ENTRIES only (except where noted); exits stay RSI<exit unless an
alternate exit is enabled, so we can isolate each overlay's marginal effect.
"""
import argparse, json, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd

DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'
TRADING_DAYS = 252


def load(symbol, timeframe='day', start=None, end=None):
    con = sqlite3.connect(str(DB))
    q = ("SELECT date, open, high, low, close, volume FROM market_data_unified "
         "WHERE symbol=? AND timeframe=? ORDER BY date")
    df = pd.read_sql_query(q, con, params=(symbol, timeframe))
    con.close()
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')
    df = df.drop_duplicates('date').set_index('date').sort_index()
    if start: df = df[df.index >= pd.Timestamp(start)]
    if end:   df = df[df.index <= pd.Timestamp(end)]
    return df


def rsi(close, length=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    # Wilder's smoothing (RMA)
    roll_up = up.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    roll_dn = dn.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    rs = roll_up / roll_dn
    out = 100.0 - (100.0 / (1.0 + rs))
    out[roll_dn == 0] = 100.0
    return out


def adx(df, length=14):
    h, l, c = df['high'], df['low'], df['close']
    up = h.diff(); dn = -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()


def supertrend(df, period=10, mult=3.0):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    hl2 = (h + l) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st_dir = pd.Series(index=df.index, dtype=float)  # 1 bull, -1 bear
    fu = fl = np.nan; dirn = 1
    fu_prev = fl_prev = np.nan
    for i in range(len(df)):
        cl = c.iloc[i]
        cu, clw = upper.iloc[i], lower.iloc[i]
        if i == 0 or np.isnan(fu_prev):
            fu, fl, dirn = cu, clw, 1
        else:
            fu = min(cu, fu_prev) if c.iloc[i-1] <= fu_prev else cu
            fl = max(clw, fl_prev) if c.iloc[i-1] >= fl_prev else clw
            if dirn == 1 and cl < fl:
                dirn = -1
            elif dirn == -1 and cl > fu:
                dirn = 1
        st_dir.iloc[i] = dirn
        fu_prev, fl_prev = fu, fl
    return st_dir


def weekly_rsi_daily(df, length=14):
    """Weekly RSI aligned to daily index (as-of, causal: uses last completed weekly bar)."""
    wk = df['close'].resample('W-FRI').last().dropna()
    wr = rsi(wk, length)
    # shift so a given week's RSI is only known at that week's close, then ffill to daily
    return wr.reindex(df.index, method='ffill')


def run(df, entry_th=70, exit_th=40, rsi_len=14, cost_bps=15.0,
        ma_len=0, adx_min=0.0, wrsi_min=0.0, st_regime=False, donchian=0):
    r = rsi(df['close'], rsi_len)
    c = df['close']
    gate = pd.Series(True, index=df.index)
    if ma_len:
        gate &= c > c.rolling(ma_len).mean()
    if adx_min:
        gate &= adx(df) >= adx_min
    if wrsi_min:
        gate &= weekly_rsi_daily(df) >= wrsi_min
    st_dir = supertrend(df) if st_regime else None
    if st_regime:
        gate &= (st_dir == 1)
    dch_low = c.rolling(donchian).min().shift(1) if donchian else None

    in_pos = False
    entry_price = 0.0
    equity = 1.0            # NAV multiplier (strategy)
    eq_curve = []
    trades = []
    rt_cost = cost_bps / 10000.0
    entry_date = None
    prev_c = None
    for dt, row in df.iterrows():
        px = row['close']; rv = r.get(dt, np.nan)
        # mark-to-market if in position (using close-to-close return)
        if in_pos and prev_c is not None:
            equity *= px / prev_c
        # decide on close
        if in_pos:
            exit_sig = (not np.isnan(rv) and rv < exit_th)
            if st_regime and st_dir is not None and st_dir.get(dt, 1) == -1:
                exit_sig = True
            if donchian and not np.isnan(dch_low.get(dt, np.nan)) and px < dch_low.get(dt):
                exit_sig = True
            if exit_sig:
                equity *= (1 - rt_cost / 2)   # exit-side cost
                ret = px / entry_price - 1
                trades.append(dict(entry=str(entry_date.date()), exit=str(dt.date()),
                                   entry_px=round(entry_price, 2), exit_px=round(px, 2),
                                   ret_pct=round(ret * 100, 2),
                                   hold_days=(dt - entry_date).days))
                in_pos = False
        else:
            if (not np.isnan(rv) and rv >= entry_th) and bool(gate.get(dt, False)):
                in_pos = True
                entry_price = px
                entry_date = dt
                equity *= (1 - rt_cost / 2)   # entry-side cost
        eq_curve.append((dt, equity))
        prev_c = px
    eqs = pd.Series(dict(eq_curve))
    return eqs, trades, r


def metrics(eqs, name='strategy'):
    if len(eqs) < 2:
        return {}
    rets = eqs.pct_change().dropna()
    yrs = (eqs.index[-1] - eqs.index[0]).days / 365.25
    total = eqs.iloc[-1] / eqs.iloc[0]
    cagr = total ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = (rets.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (rets.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan
    roll_max = eqs.cummax()
    dd = eqs / roll_max - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return dict(name=name, cagr=round(cagr*100,2), total_return_x=round(total,3),
                sharpe=round(sharpe,2), sortino=round(sortino,2),
                maxdd=round(maxdd*100,2), calmar=round(calmar,2),
                vol=round(vol*100,2), years=round(yrs,2))


def per_year(eqs):
    y = eqs.resample('YE').last()
    y0 = eqs.iloc[0]
    prev = y0; out = {}
    for dt, v in y.items():
        out[dt.year] = round((v/prev - 1)*100, 2)
        prev = v
    return out


def buyhold(df):
    eqs = df['close'] / df['close'].iloc[0]
    return eqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='RELIANCE')
    ap.add_argument('--bench', default='NIFTYBEES')
    ap.add_argument('--entry', type=float, default=70)
    ap.add_argument('--exit', type=float, default=40)
    ap.add_argument('--rsi_len', type=int, default=14)
    ap.add_argument('--cost_bps', type=float, default=15)
    ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default=None)
    ap.add_argument('--ma_len', type=int, default=0)
    ap.add_argument('--adx_min', type=float, default=0)
    ap.add_argument('--wrsi_min', type=float, default=0)
    ap.add_argument('--st_regime', action='store_true')
    ap.add_argument('--donchian', type=int, default=0)
    ap.add_argument('--json_out', default=None)
    a = ap.parse_args()

    df = load(a.symbol, start=a.start, end=a.end)
    if df.empty:
        print('NO DATA for', a.symbol); sys.exit(1)
    bench = load(a.bench, start=max(df.index[0], pd.Timestamp(a.start)), end=a.end)

    eqs, trades, r = run(df, a.entry, a.exit, a.rsi_len, a.cost_bps,
                         a.ma_len, a.adx_min, a.wrsi_min, a.st_regime, a.donchian)
    # net already includes cost; gross = re-run cost 0
    eqs_g, _, _ = run(df, a.entry, a.exit, a.rsi_len, 0,
                      a.ma_len, a.adx_min, a.wrsi_min, a.st_regime, a.donchian)
    m_net = metrics(eqs, 'RSI_net'); m_gross = metrics(eqs_g, 'RSI_gross')
    m_bh_stock = metrics(buyhold(df.loc[eqs.index[0]:]), a.symbol+'_BH')
    m_bench = metrics(buyhold(bench), a.bench+'_BH') if not bench.empty else {}

    wins = [t for t in trades if t['ret_pct'] > 0]
    exposure = float((eqs.pct_change().abs() > 0).mean())  # rough
    summary = dict(
        symbol=a.symbol, entry=a.entry, exit=a.exit, rsi_len=a.rsi_len,
        cost_bps=a.cost_bps, start=str(df.index[0].date()), end=str(df.index[-1].date()),
        n_trades=len(trades),
        win_rate=round(100*len(wins)/len(trades),1) if trades else 0,
        avg_win=round(np.mean([t['ret_pct'] for t in wins]),2) if wins else 0,
        avg_loss=round(np.mean([t['ret_pct'] for t in trades if t['ret_pct']<=0]),2) if len(wins)<len(trades) else 0,
        avg_hold_days=round(np.mean([t['hold_days'] for t in trades]),1) if trades else 0,
        net=m_net, gross=m_gross, stock_bh=m_bh_stock, bench_bh=m_bench,
        per_year_net=per_year(eqs),
        per_year_bench=per_year(buyhold(bench)) if not bench.empty else {},
        filters=dict(ma_len=a.ma_len, adx_min=a.adx_min, wrsi_min=a.wrsi_min,
                     st_regime=a.st_regime, donchian=a.donchian),
    )
    print(json.dumps(summary, indent=2, default=str))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(dict(summary=summary, trades=trades), indent=2, default=str))
    return summary


if __name__ == '__main__':
    main()
