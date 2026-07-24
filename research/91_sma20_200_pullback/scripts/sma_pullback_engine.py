"""
20/200-SMA "Picture of Power" retrace-break engine (iFundTraders RBI&GO).
research/91 — causal, look-ahead-safe. See STATUS-MD sections 1-4 for the spec.

Setup (LONG): trend up (SMA20>SMA200, SMA20 rising, close>SMA200); a RED pause bar
after a GREEN, with its low NEAR (+/- near_pct) the SMA20; enter stop-buy above the
red's high within entry_win bars; SL below red's low; exit when price extends
atr_mult*ATR above SMA20 (drift away). SHORT is the mirror.

All features at bar i use only data <= i. Trigger/fills happen on bars > i.
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'


# ---------------------------------------------------------------- data loading
def load_bars(symbol, timeframe, db=DB, start=None, end=None):
    con = sqlite3.connect(str(db))
    q = ("SELECT date, open, high, low, close, volume FROM market_data_unified "
         "WHERE symbol=? AND timeframe=? ")
    args = [symbol, timeframe]
    if start:
        q += "AND date>=? "; args.append(start)
    if end:
        q += "AND date<=? "; args.append(end)
    q += "ORDER BY date"
    df = pd.read_sql_query(q, con, params=args, parse_dates=['date'])
    con.close()
    if df.empty:
        return df
    df = df.drop_duplicates('date').reset_index(drop=True)
    # basic hygiene: positive prices, valid bar
    df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)].reset_index(drop=True)
    return df


def resample_intraday(df, rule):
    """Resample 5-min bars to e.g. '15min'/'30min'/'60min' within each session."""
    if df.empty:
        return df
    d = df.set_index('date')
    day = d.index.normalize()
    out = []
    for _, g in d.groupby(day):
        r = g.resample(rule, label='left', closed='left', origin='start_day').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        r = r.dropna(subset=['open'])
        out.append(r)
    res = pd.concat(out).reset_index()
    return res


# ---------------------------------------------------------------- indicators
def add_indicators(df, sma_fast=20, sma_slow=200, atr_len=14):
    c, h, l = df['close'], df['high'], df['low']
    df['sma_f'] = c.rolling(sma_fast).mean()
    df['sma_s'] = c.rolling(sma_slow).mean()
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_len).mean()          # simple ATR (causal)
    df['green'] = c > df['open']
    df['red'] = c < df['open']
    return df


# ---------------------------------------------------------------- backtest
def backtest(df, direction='long', slope_lb=3, near_pct=0.005, atr_mult=2.5,
             entry_win=3, hold='intraday', cost_bps=0.0, exit_mode='extend',
             max_hold=0):
    # exit_mode: 'extend' (drift atr_mult*ATR from SMA20, user's rule),
    #            'sma_cross' (exit when close crosses back through SMA20 = ride the trend),
    #            'both' (whichever fires first). max_hold>0 = hard time-stop in bars.
    """Return list of trade dicts. df must have indicators + a 'date' column."""
    if df.empty or len(df) < 210:
        return []
    o = df['open'].values; h = df['high'].values; l = df['low'].values; c = df['close'].values
    smf = df['sma_f'].values; sms = df['sma_s'].values; atr = df['atr'].values
    green = df['green'].values; red = df['red'].values
    dates = df['date'].values
    sess = pd.to_datetime(df['date']).dt.normalize().values   # session id (day)
    n = len(df)
    trades = []
    i = 200
    long = direction == 'long'

    while i < n - 1:
        # need valid indicators at signal bar i and its predecessor
        if np.isnan(smf[i]) or np.isnan(sms[i]) or np.isnan(atr[i]) or i - slope_lb < 0:
            i += 1; continue

        if long:
            regime = (smf[i] > sms[i]) and (smf[i] > smf[i - slope_lb]) and (c[i] > sms[i])
            pause = red[i] and green[i - 1]
            near = (smf[i] * (1 - near_pct) <= l[i] <= smf[i] * (1 + near_pct))
            setup = regime and pause and near
        else:
            regime = (smf[i] < sms[i]) and (smf[i] < smf[i - slope_lb]) and (c[i] < sms[i])
            pause = green[i] and red[i - 1]
            near = (smf[i] * (1 - near_pct) <= h[i] <= smf[i] * (1 + near_pct))
            setup = regime and pause and near

        if not setup:
            i += 1; continue

        trig = h[i] if long else l[i]          # break of the pause bar's extreme
        stop0 = l[i] if long else h[i]         # initial SL just beyond the other side
        # ---- look for trigger in the next entry_win bars
        entry_price = None; ent = None
        for j in range(i + 1, min(i + 1 + entry_win, n)):
            if long:
                if l[j] < stop0:               # invalidated before trigger
                    break
                if h[j] >= trig:
                    entry_price = max(trig, o[j]); ent = j; break
            else:
                if h[j] > stop0:
                    break
                if l[j] <= trig:
                    entry_price = min(trig, o[j]); ent = j; break
        if ent is None:
            i += 1; continue

        # ---- manage the position from entry bar onward
        exit_price = None; exj = None; reason = None
        use_ext = exit_mode in ('extend', 'both')
        use_cross = exit_mode in ('sma_cross', 'both')
        for k in range(ent, n):
            # intraday hold: force exit at last bar of the entry session
            if hold == 'intraday' and sess[k] != sess[ent]:
                exj = k - 1; exit_price = c[exj]; reason = 'eod'; break
            if max_hold and (k - ent) >= max_hold:
                exit_price = c[k]; exj = k; reason = 'time'; break
            if long:
                if l[k] <= stop0:                                   # stop (gap-through)
                    exit_price = min(o[k], stop0) if o[k] < stop0 else stop0
                    exj = k; reason = 'sl'; break
                if use_ext and not np.isnan(smf[k]) and (h[k] - smf[k]) >= atr_mult * atr[k]:
                    exit_price = c[k]; exj = k; reason = 'extend'; break
                if use_cross and k > ent and not np.isnan(smf[k]) and c[k] < smf[k]:
                    exit_price = c[k]; exj = k; reason = 'cross'; break
            else:
                if h[k] >= stop0:
                    exit_price = max(o[k], stop0) if o[k] > stop0 else stop0
                    exj = k; reason = 'sl'; break
                if use_ext and not np.isnan(smf[k]) and (smf[k] - l[k]) >= atr_mult * atr[k]:
                    exit_price = c[k]; exj = k; reason = 'extend'; break
                if use_cross and k > ent and not np.isnan(smf[k]) and c[k] > smf[k]:
                    exit_price = c[k]; exj = k; reason = 'cross'; break
        if exit_price is None:                                       # ran to end of data
            exj = n - 1; exit_price = c[exj]; reason = 'end'

        # ---- P&L
        if long:
            gross = (exit_price - entry_price) / entry_price
            risk = (entry_price - stop0) / entry_price
        else:
            gross = (entry_price - exit_price) / entry_price
            risk = (stop0 - entry_price) / entry_price
        net = gross - cost_bps / 1e4
        R = net / risk if risk > 1e-6 else np.nan
        trades.append(dict(
            entry_date=str(dates[ent])[:19], exit_date=str(dates[exj])[:19],
            entry=round(float(entry_price), 4), exit=round(float(exit_price), 4),
            stop=round(float(stop0), 4), reason=reason, hold_bars=int(exj - ent),
            gross_pct=round(float(gross) * 100, 4), net_pct=round(float(net) * 100, 4),
            risk_pct=round(float(risk) * 100, 4), R=round(float(R), 3) if not np.isnan(R) else None))
        i = exj + 1          # no overlapping trades on the same series
    return trades


# ---------------------------------------------------------------- metrics
def summarize(trades):
    if not trades:
        return dict(n=0)
    net = np.array([t['net_pct'] for t in trades]) / 100.0
    gross = np.array([t['gross_pct'] for t in trades]) / 100.0
    Rs = np.array([t['R'] for t in trades if t['R'] is not None])
    wins = net[net > 0]; losses = net[net <= 0]
    n = len(net)
    t_stat = float(net.mean() / (net.std(ddof=1) / np.sqrt(n))) if n > 1 and net.std() > 0 else 0.0
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float('inf')
    return dict(
        n=n,
        gross_exp_pct=round(float(gross.mean()) * 100, 4),
        net_exp_pct=round(float(net.mean()) * 100, 4),
        net_exp_R=round(float(Rs.mean()), 3) if len(Rs) else None,
        win_pct=round(float((net > 0).mean()) * 100, 1),
        pf=round(pf, 2),
        t_stat=round(t_stat, 2),
        avg_hold=round(float(np.mean([t['hold_bars'] for t in trades])), 1),
        total_net_pct=round(float(net.sum()) * 100, 2))


def run_symbol(symbol, timeframe, direction='long', resample=None, **params):
    df = load_bars(symbol, timeframe)
    if df.empty:
        return [], dict(n=0, err='no data')
    if resample:
        df = resample_intraday(df, resample)
    df = add_indicators(df,
                        sma_fast=params.pop('sma_fast', 20),
                        sma_slow=params.pop('sma_slow', 200),
                        atr_len=params.pop('atr_len', 14))
    tr = backtest(df, direction=direction, **params)
    s = summarize(tr)
    return tr, s
