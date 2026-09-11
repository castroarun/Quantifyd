"""
research/159 — Rounding-base (saucer) + volume accumulation -> rim breakout.
STRICTLY CAUSAL detector. Read-only against market_data.db.

On every day t only bars with date <= t are used. The rim level R is fixed on the
base-qualify day q and frozen; the entry signal is the first close above that frozen R.
A breakout day is NEVER located first and a base fitted backwards to it.

Spec: research/159_rounding_base_breakout/ROUNDING_BASE_BREAKOUT_DAILY_SCREEN_STATUS.md
Forward-return and SuperTrend columns are INFORMATION ONLY and must not be used to
select or rank events.
"""
import csv
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUTDIR = Path(__file__).resolve().parents[1] / 'results'
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / 'rounding_base_events.csv'

# ---------------------------------------------------------------- thresholds (locked)
MIN_BARS          = 90
MODES             = [('L120', 120), ('L180', 180), ('L250', 250), ('VAR', 250)]
VAR_MIN_LEN       = 90
VAR_MAX_LEN       = 400
TROUGH_POS_LO     = 0.30
TROUGH_POS_HI     = 0.70
DEPTH_LO          = 0.20
DEPTH_HI          = 0.70
R2_MIN            = 0.70
R2_VARIANTS       = (0.60, 0.80)
VERTEX_LO         = 0.33
VERTEX_HI         = 0.67
# No-V test, DEPTH-relative (revised 11-Sep-2026, see STATUS 3.2 / deviation log).
# Band = trough + FLAT_DEPTH_FRAC * (rim - trough) = the bottom third of the base's range.
# Reference geometry: a LINEAR V spends exactly 0.333 of its bars below that line;
# an ideal PARABOLA spends 0.577. The 0.40 floor sits between the two, so it rejects a V
# on geometry rather than on a price-relative band tuned to any one example.
FLAT_DEPTH_FRAC   = 0.3333
FLAT_MIN_FRAC     = 0.40
RECOVERY_BAND     = 0.95      # close >= 0.95 * R  => base-qualify day q
MAX_WAIT_BARS     = 60        # bars after q to wait for the breakout close
VOID_BAND         = 0.85      # close < 0.85 * R while waiting => base voided
REARM_BARS        = 60        # suppression after a fired event
REARM_BARS_FAIL   = 20        # suppression after an expiry/void
LIQ_MIN_TV        = 2e7       # Rs 2 crore, 20-day median traded value at q
SPLIT_DOWN        = -0.35     # one-day close-to-close move rejecting the window
SPLIT_UP          = 0.50
VOL_RATIO_MIN     = 1.20
ST_PERIOD, ST_MULT = 7, 3.0
FWD_HORIZONS      = (60, 120, 250)

ETF_PAT = ('BEES', 'ETF', 'IETF', 'GOLD', 'SILVER', 'LIQUID', 'NIFTY', 'SENSEX',
           'BANKNIFTY', 'MIDCAP', 'SMALLCAP', 'INDEX', 'MAFANG', 'HNGSNGBEES')

FIELDS = [
    'symbol', 'mode', 'ipo_short_window',
    'base_qualify_date', 'rim_date', 'rim_level', 'trough_date', 'trough_close',
    'depth_pct', 'base_len_bars', 'trough_pos', 'fit_r2', 'fit_curvature',
    'vertex_frac', 'flat_frac', 'r2_ge_060', 'r2_ge_080',
    'vol_ratio', 'obv_gain', 'obv_slope_norm', 'obv_filter_pass',
    'tv20_cr_at_q', 'tv20_cr_at_breakout',
    'breakout_date', 'breakout_close', 'breakout_day_move_pct', 'days_q_to_breakout',
    'fill_a_nextopen_date', 'fill_a_nextopen', 'fill_b_buystop', 'gap_pct_a_vs_rim',
    'pattern_quality',
    # ---- INFORMATION ONLY BELOW THIS LINE — never used for selection/ranking ----
    'info_fwd60_pct', 'info_fwd120_pct', 'info_fwd250_pct',
    'info_st73_exit_date', 'info_st73_exit_close', 'info_st73_ret_pct',
    'info_st73_bars_held', 'info_st73_still_open', 'info_mfe_250_pct', 'info_mae_250_pct',
]


def supertrend(high, low, close, period=ST_PERIOD, mult=ST_MULT):
    """Classic SuperTrend on daily closes. Returns direction array (+1 up / -1 down)."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr[1:] = np.maximum(hl, np.maximum(hc, lc))
    atr = np.empty(n)
    atr[:] = np.nan
    if n <= period:
        return np.zeros(n, dtype=np.int8)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):                      # Wilder smoothing
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = (high + low) / 2.0
    up = hl2 + mult * atr
    dn = hl2 - mult * atr
    fu = np.copy(up)
    fl = np.copy(dn)
    direction = np.ones(n, dtype=np.int8)
    for i in range(period, n):
        if i == period:
            continue
        fu[i] = up[i] if (up[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = dn[i] if (dn[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < fl[i] else 1
        else:
            direction[i] = 1 if close[i] > fu[i] else -1
    direction[:period] = 0
    return direction


def quad_fit(logc):
    """Fit log(close) = a x^2 + b x + c. Returns (a, r2, vertex_frac)."""
    m = len(logc)
    x = np.arange(m, dtype=np.float64)
    try:
        a, b, c = np.polyfit(x, logc, 2)
    except Exception:
        return None
    if not np.isfinite(a) or a == 0:
        return None
    yh = a * x * x + b * x + c
    sst = ((logc - logc.mean()) ** 2).sum()
    if sst <= 0:
        return None
    r2 = 1.0 - ((logc - yh) ** 2).sum() / sst
    vertex = (-b / (2.0 * a)) / max(m - 1, 1)
    return float(a), float(r2), float(vertex)


def clip01(v):
    return float(min(max(v, 0.0), 1.0))


def score_pattern(r2, trough_pos, depth, vol_ratio, obv_gain, flat_frac):
    fit = clip01((r2 - 0.70) / 0.25)
    sym = 1.0 - min(abs(trough_pos - 0.50) / 0.20, 1.0)
    dep = 1.0 - min(abs(depth - 0.375) / 0.25, 1.0)
    acc = 0.5 * clip01((vol_ratio - 1.0) / 2.0) + 0.5 * clip01(obv_gain / 0.5)
    flat = clip01((flat_frac - 0.40) / 0.18)   # 0.40 = V-reject floor, 0.58 = ideal parabola
    return round(0.30 * fit + 0.20 * sym + 0.15 * dep + 0.20 * acc + 0.15 * flat, 4)


def load_symbol(con, sym):
    df = pd.read_sql_query(
        "SELECT date,open,high,low,close,volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date",
        con, params=(sym,))
    if df.empty:
        return None
    # phantom holiday rows (O=H=L=C, volume 0) and NULL volume
    df = df[(df['volume'].notna()) & (df['volume'] > 0)]
    df = df[(df['close'].notna()) & (df['close'] > 0)]
    df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)
    if len(df) < MIN_BARS:
        return None
    return df


def process_symbol(sym, df, writer, fh, stats):
    date = df['date'].to_numpy()
    o = df['open'].to_numpy(np.float64)
    h = df['high'].to_numpy(np.float64)
    lo = df['low'].to_numpy(np.float64)
    c = df['close'].to_numpy(np.float64)
    v = df['volume'].to_numpy(np.float64)
    n = len(c)
    logc = np.log(c)
    ret1 = np.empty(n); ret1[0] = 0.0
    ret1[1:] = c[1:] / c[:-1] - 1.0
    obv = np.concatenate(([0.0], np.cumsum(np.sign(np.diff(c)) * v[1:])))
    tv = c * v
    tv20 = pd.Series(tv).rolling(20, min_periods=10).median().to_numpy()
    st_dir = supertrend(h, lo, c)

    # ---- vectorised cheap gates per mode -------------------------------------
    pre = {}
    for name, L in MODES:
        if n < MIN_BARS:
            continue
        if name == 'VAR':
            W = 250
            if n >= W:
                sw = sliding_window_view(c, W)
                rim_off = sw.argmax(axis=1)                 # index within window
                rim_idx = np.arange(n - W + 1) + rim_off     # absolute rim index, aligned to t = W-1..n-1
                full_rim = np.full(n, -1, dtype=np.int64)
                full_rim[W - 1:] = rim_idx
            else:
                full_rim = np.full(n, -1, dtype=np.int64)
            # expanding fallback for young symbols / early bars
            exp_rim = np.maximum.accumulate(np.where(np.r_[True, c[1:] > np.maximum.accumulate(c)[:-1]],
                                                     np.arange(n), 0))
            # robust expanding argmax
            exp_rim = np.zeros(n, dtype=np.int64)
            best = 0
            for i in range(n):
                if c[i] > c[best]:
                    best = i
                exp_rim[i] = best
            full_rim = np.where(full_rim >= 0, full_rim, exp_rim)
            pre[name] = ('VAR', full_rim)
        else:
            third = max(L // 3, 5)
            s = pd.Series(c)
            rmax_third = s.rolling(third, min_periods=third).max().shift(L - third).to_numpy()
            rmin_full = s.rolling(L, min_periods=L).min().to_numpy()
            pre[name] = ('FIX', L, third, rmax_third, rmin_full)

    for name, L in MODES:
        if name not in pre:
            continue
        state = 'idle'
        armed = None
        block_until = -1
        for t in range(MIN_BARS - 1, n):
            if state == 'armed':
                R = armed['rim_level']
                if c[t] > R:
                    if abs(ret1[t]) <= SPLIT_UP:
                        emit(sym, name, armed, t, date, o, c, h, lo, st_dir, tv20, ret1, n, writer, stats)
                        stats['events'] += 1
                    else:
                        stats['bo_split_rejected'] += 1
                    state, armed, block_until = 'idle', None, t + REARM_BARS
                elif c[t] < VOID_BAND * R:
                    stats['voided'] += 1
                    state, armed, block_until = 'idle', None, t + REARM_BARS_FAIL
                elif t - armed['q'] > MAX_WAIT_BARS:
                    stats['expired'] += 1
                    state, armed, block_until = 'idle', None, t + REARM_BARS_FAIL
                continue
            if t < block_until:
                continue
            if not np.isfinite(tv20[t]) or tv20[t] < LIQ_MIN_TV:
                continue

            # ---- determine window [w0..t] and rim R, causally -----------------
            p = pre[name]
            if p[0] == 'VAR':
                i_rim = int(p[1][t])
                if i_rim < 0:
                    continue
                wlen = t - i_rim + 1
                if wlen < VAR_MIN_LEN or wlen > VAR_MAX_LEN:
                    continue
                w0 = i_rim
                R = c[i_rim]
                ipo_short = 1 if (t + 1) < 250 else 0
            else:
                _, Lf, third, rmax_third, rmin_full = p
                if (t + 1) >= Lf:
                    w0 = t - Lf + 1
                    Rv = rmax_third[t]
                    minv = rmin_full[t]
                    if not np.isfinite(Rv) or not np.isfinite(minv):
                        continue
                    if c[t] < RECOVERY_BAND * Rv:
                        continue
                    d = (Rv - minv) / Rv
                    if d < DEPTH_LO or d > DEPTH_HI:
                        continue
                    ipo_short = 0
                else:
                    w0 = 0
                    if (t + 1) < MIN_BARS:
                        continue
                    ipo_short = 1
                seg0 = c[w0:t + 1]
                m0 = len(seg0)
                th = max(m0 // 3, 5)
                i_rim = w0 + int(np.argmax(seg0[:th]))
                R = c[i_rim]

            if c[t] < RECOVERY_BAND * R:
                continue

            seg = c[w0:t + 1]
            m = len(seg)
            if m < MIN_BARS:
                continue
            i_tr_loc = int(np.argmin(seg))
            i_tr = w0 + i_tr_loc
            if i_tr <= i_rim:
                continue
            trough_pos = i_tr_loc / (m - 1)
            if trough_pos < TROUGH_POS_LO or trough_pos > TROUGH_POS_HI:
                continue
            depth = (R - seg[i_tr_loc]) / R
            if depth < DEPTH_LO or depth > DEPTH_HI:
                continue
            # split-artifact guard
            wret = ret1[w0 + 1:t + 1]
            if wret.size and (wret.min() < SPLIT_DOWN or wret.max() > SPLIT_UP):
                stats['split_rejected'] += 1
                continue
            # no-V test
            band = seg[i_tr_loc] + FLAT_DEPTH_FRAC * (R - seg[i_tr_loc])
            flat_frac = float((seg <= band).sum()) / m
            if flat_frac < FLAT_MIN_FRAC:
                continue
            # roundness
            qf = quad_fit(logc[w0:t + 1])
            if qf is None:
                continue
            a, r2, vertex = qf
            if a <= 0 or r2 < R2_MIN:
                continue
            if vertex < VERTEX_LO or vertex > VERTEX_HI:
                continue
            # volume / OBV accumulation
            lv = v[w0:i_tr + 1]
            rv = v[i_tr:t + 1]
            lmed = np.median(lv) if lv.size else 0.0
            rmed = np.median(rv) if rv.size else 0.0
            vol_ratio = float(rmed / lmed) if lmed > 0 else np.nan
            rsum = float(rv.sum())
            obv_gain = float((obv[t] - obv[i_tr]) / rsum) if rsum > 0 else np.nan
            k = t - i_tr + 1
            if k >= 3:
                xx = np.arange(k, dtype=np.float64)
                yy = obv[i_tr:t + 1]
                sl = float(np.polyfit(xx, yy, 1)[0])
                mv = float(rv.mean()) if rv.size else np.nan
                obv_slope_norm = sl / mv if mv and mv > 0 else np.nan
            else:
                obv_slope_norm = np.nan
            obv_pass = int(bool(
                (np.isfinite(obv_slope_norm) and obv_slope_norm > 0) and
                (np.isfinite(obv_gain) and obv_gain > 0) and
                (np.isfinite(vol_ratio) and vol_ratio >= VOL_RATIO_MIN)))

            armed = dict(
                q=t, w0=w0, i_rim=i_rim, i_tr=i_tr, rim_level=float(R),
                depth=float(depth), base_len=m, trough_pos=float(trough_pos),
                r2=float(r2), curv=float(a), vertex=float(vertex), flat_frac=float(flat_frac),
                vol_ratio=vol_ratio, obv_gain=obv_gain, obv_slope=obv_slope_norm,
                obv_pass=obv_pass, ipo_short=int(ipo_short), tv20_q=float(tv20[t]),
            )
            state = 'armed'
            stats['qualified'] += 1
            # A fixed-window rim can already be exceeded on the qualify day itself.
            # That is still strictly causal (R comes from the window's first third, and
            # close[t] is known at t; the fill is the NEXT bar), so allow it.
            if c[t] > R:
                if abs(ret1[t]) <= SPLIT_UP:
                    emit(sym, name, armed, t, date, o, c, h, lo, st_dir, tv20, ret1, n, writer, stats)
                    stats['events'] += 1
                else:
                    stats['bo_split_rejected'] += 1
                state, armed, block_until = 'idle', None, t + REARM_BARS
        # end day loop
    fh.flush()


def emit(sym, mode, a, t, date, o, c, h, lo, st_dir, tv20, ret1, n, writer, stats):
    R = a['rim_level']
    nxt = t + 1
    if nxt < n:
        fill_a = float(o[nxt])
        fill_a_date = str(date[nxt])[:10]
        fill_b = float(max(R, o[nxt]))
        gap = 100.0 * (fill_a - R) / R
    else:
        fill_a, fill_a_date, fill_b, gap = np.nan, '', np.nan, np.nan

    def fwd(hz):
        if not np.isfinite(fill_a):
            return np.nan
        j = min(nxt + hz - 1, n - 1)
        if j <= nxt:
            return np.nan
        return round(100.0 * (c[j] / fill_a - 1.0), 2)

    # SuperTrend(7,3) exit — INFORMATION ONLY
    st_exit_date, st_exit_close, st_ret, st_bars, st_open = '', np.nan, np.nan, np.nan, 1
    if np.isfinite(fill_a):
        j = nxt
        while j < n and st_dir[j] != -1:
            j += 1
        if j < n:
            st_exit_date = str(date[j])[:10]
            st_exit_close = float(c[j])
            st_ret = round(100.0 * (st_exit_close / fill_a - 1.0), 2)
            st_bars = j - nxt + 1
            st_open = 0
        else:
            st_bars = n - nxt
            st_ret = round(100.0 * (c[n - 1] / fill_a - 1.0), 2)
    mfe = mae = np.nan
    if np.isfinite(fill_a) and nxt < n:
        j = min(nxt + 250, n)
        mfe = round(100.0 * (h[nxt:j].max() / fill_a - 1.0), 2)
        mae = round(100.0 * (lo[nxt:j].min() / fill_a - 1.0), 2)

    row = {
        'symbol': sym, 'mode': mode, 'ipo_short_window': a['ipo_short'],
        'base_qualify_date': str(date[a['q']])[:10],
        'rim_date': str(date[a['i_rim']])[:10], 'rim_level': round(R, 2),
        'trough_date': str(date[a['i_tr']])[:10], 'trough_close': round(float(c[a['i_tr']]), 2),
        'depth_pct': round(100 * a['depth'], 2), 'base_len_bars': a['base_len'],
        'trough_pos': round(a['trough_pos'], 3), 'fit_r2': round(a['r2'], 3),
        'fit_curvature': '%.3e' % a['curv'], 'vertex_frac': round(a['vertex'], 3),
        'flat_frac': round(a['flat_frac'], 3),
        'r2_ge_060': int(a['r2'] >= 0.60), 'r2_ge_080': int(a['r2'] >= 0.80),
        'vol_ratio': round(a['vol_ratio'], 3) if np.isfinite(a['vol_ratio']) else '',
        'obv_gain': round(a['obv_gain'], 4) if np.isfinite(a['obv_gain']) else '',
        'obv_slope_norm': round(a['obv_slope'], 4) if np.isfinite(a['obv_slope']) else '',
        'obv_filter_pass': a['obv_pass'],
        'tv20_cr_at_q': round(a['tv20_q'] / 1e7, 2),
        'tv20_cr_at_breakout': round(float(tv20[t]) / 1e7, 2) if np.isfinite(tv20[t]) else '',
        'breakout_date': str(date[t])[:10], 'breakout_close': round(float(c[t]), 2),
        'breakout_day_move_pct': round(100 * float(ret1[t]), 2),
        'days_q_to_breakout': t - a['q'],
        'fill_a_nextopen_date': fill_a_date,
        'fill_a_nextopen': round(fill_a, 2) if np.isfinite(fill_a) else '',
        'fill_b_buystop': round(fill_b, 2) if np.isfinite(fill_b) else '',
        'gap_pct_a_vs_rim': round(gap, 2) if np.isfinite(gap) else '',
        'pattern_quality': score_pattern(a['r2'], a['trough_pos'], a['depth'],
                                         a['vol_ratio'] if np.isfinite(a['vol_ratio']) else 0.0,
                                         a['obv_gain'] if np.isfinite(a['obv_gain']) else 0.0,
                                         a['flat_frac']),
        'info_fwd60_pct': fwd(60), 'info_fwd120_pct': fwd(120), 'info_fwd250_pct': fwd(250),
        'info_st73_exit_date': st_exit_date,
        'info_st73_exit_close': round(st_exit_close, 2) if np.isfinite(st_exit_close) else '',
        'info_st73_ret_pct': st_ret if st_ret == st_ret else '',
        'info_st73_bars_held': st_bars, 'info_st73_still_open': st_open,
        'info_mfe_250_pct': mfe if mfe == mfe else '', 'info_mae_250_pct': mae if mae == mae else '',
    }
    writer.writerow(row)


def main():
    t0 = time.time()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    syms = [r[0] for r in con.execute(
        "SELECT symbol FROM market_data_unified WHERE timeframe='day' "
        "GROUP BY symbol HAVING COUNT(*)>=%d ORDER BY symbol" % MIN_BARS)]
    print('candidate symbols: %d' % len(syms), flush=True)
    syms = [s for s in syms if not any(p in s.upper() for p in ETF_PAT)]
    print('after ETF/index exclusion: %d' % len(syms), flush=True)
    if len(sys.argv) > 1 and sys.argv[1].startswith('--symbols='):
        only = set(sys.argv[1].split('=', 1)[1].split(','))
        syms = [s for s in syms if s in only]
        print('SMOKE TEST restricted to: %s' % syms, flush=True)

    stats = dict(events=0, qualified=0, expired=0, voided=0, split_rejected=0,
                 bo_split_rejected=0, skipped=0)
    out = OUT_CSV
    if len(sys.argv) > 1 and sys.argv[1].startswith('--symbols='):
        out = OUTDIR / 'smoke_events.csv'
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, s in enumerate(syms, 1):
            try:
                df = load_symbol(con, s)
                if df is None:
                    stats['skipped'] += 1
                    continue
                process_symbol(s, df, w, fh, stats)
            except Exception as e:
                print('  ERR %s: %r' % (s, e), flush=True)
            if i % 200 == 0:
                el = time.time() - t0
                print('[%d/%d] %.0fs elapsed | events=%d qualified=%d expired=%d voided=%d '
                      'split_rej=%d | ETA %.0fs'
                      % (i, len(syms), el, stats['events'], stats['qualified'],
                         stats['expired'], stats['voided'], stats['split_rejected'],
                         el / i * (len(syms) - i)), flush=True)
    con.close()
    print('DONE in %.0fs' % (time.time() - t0), flush=True)
    print('stats:', stats, flush=True)
    print('csv:', out, flush=True)


if __name__ == '__main__':
    main()
