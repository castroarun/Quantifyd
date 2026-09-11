"""
research/159 — Rounding-base (saucer) detector **v2**.

WHAT CHANGED FROM v1, AND WHY
-----------------------------
Arun manually checked the v1 short-list and rejected the entry definition on SKFINDIA:
v1 entered 03-Jul-2025 at Rs2,352 when the real breakout was 16-May-2025 at Rs2,144.70
(a gap-up on 9.5x the 20-day median volume, clearing the flat Mar-May shelf at
Rs1,750-1,860). v1 entered +15% into the move.

The v1 trigger was "first close above the LEFT RIM" -- the far left lip of the decline.
That is the wrong level. The breakout Arun trades is out of the **BASE CEILING**: the
ceiling of the consolidation at the BOTTOM of the saucer. The left rim is not the
trigger, it is the first overhead SUPPLY the trade must later chew through -- so v2
records the distance to it as information instead of using it as the entry level.

v2 therefore:
  * KEEPS all the saucer-shape machinery (quadratic curvature > 0, R2 floor, trough in
    the middle, depth-relative no-V >= 0.40, depth >= 20% rim->trough, split guard,
    phantom-row purge, IPO-age exception).
  * DROPS the "close back within 5% of the left rim" completion test. It is replaced by
    a **minimal lift-off**: close >= trough + 15% of (rim - trough). The base only has
    to be recognisable, not recovered.
  * ENTERS on a **base-ceiling breakout**: first close above the highest close of the
    prior N bars (N=60 default) on volume >= K x the prior 20-bar median (K=3 default),
    and the day must be an up-candle. Fill = next-day open.

Still strictly causal: on day t only bars with date <= t are used; the trigger compares
close[t] against the prior N bars, never against anything at or after t.
Forward returns and SuperTrend columns are INFORMATION ONLY and are never used to
select or rank anything.

v1 lives on unchanged in detect_rounding_base.py / results/rounding_base_events.csv.
"""
import csv
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
OUT_CSV = OUTDIR / 'rounding_base_events_v2.csv'

# ------------------------------------------------- shape thresholds (unchanged from v1)
MIN_BARS          = 90
MODES             = [('L120', 120), ('L180', 180), ('L250', 250), ('VAR', 250)]
VAR_MIN_LEN       = 90
VAR_MAX_LEN       = 400
TROUGH_POS_LO     = 0.30
TROUGH_POS_HI     = 0.70
DEPTH_LO          = 0.20
DEPTH_HI          = 0.70
R2_MIN            = 0.70
# v2 deviation D4: the vertex bound is ALIGNED with the trough-position bound (0.30-0.70)
# instead of the tighter 0.33-0.67 inherited from v1. Both gates test the same geometric
# property - where the bottom of the bowl sits - and having two different tolerances for it
# was an inconsistency, with the fitted vertex the noisier of the two estimators (it drifts
# monotonically as bars are appended). On SKFINDIA the v1 bound rejected Arun's own
# 16-May-2025 breakout by 0.004 (vertex 0.674 vs 0.67) while trough_pos sat at a comfortable
# 0.664. Cost of the change is reported in the STATUS doc as an event-count delta.
VERTEX_LO         = 0.30
VERTEX_HI         = 0.70
FLAT_DEPTH_FRAC   = 0.3333     # bottom third of the base's vertical range
FLAT_MIN_FRAC     = 0.40       # linear V = 0.333, ideal parabola = 0.577
LIQ_MIN_TV        = 2e7        # Rs 2 crore, 20-day median traded value
SPLIT_DOWN        = -0.35
SPLIT_UP          = 0.50
VOL_RATIO_MIN     = 1.20

# ------------------------------------------------- v2: base completion + entry trigger
LIFTOFF_FRAC      = 0.15       # close >= trough + 15% of (rim - trough)
TRIG_N            = 60         # prior-N-bar highest close = the base ceiling
TRIG_N_FLAGS      = (40, 60, 100)
TRIG_K            = 3.0        # volume >= K x prior 20-bar median
VOL_MED_LOOKBACK  = 20
MAX_WAIT_BARS     = 120        # bars after q to wait for the trigger
REARM_BARS        = 60
REARM_BARS_FAIL   = 20
ST_PERIOD, ST_MULT = 7, 3.0

ETF_PAT = ('BEES', 'ETF', 'IETF', 'GOLD', 'SILVER', 'LIQUID', 'NIFTY', 'SENSEX',
           'BANKNIFTY', 'MIDCAP', 'SMALLCAP', 'INDEX', 'MAFANG', 'HNGSNGBEES')

FIELDS = [
    'symbol', 'mode', 'ipo_short_window',
    'left_rim_date', 'left_rim_level', 'trough_date', 'trough_close',
    'depth_pct', 'base_len_bars', 'trough_pos', 'fit_r2', 'fit_curvature',
    'vertex_frac', 'flat_frac', 'r2_ge_060', 'r2_ge_080',
    'vol_ratio', 'obv_gain', 'obv_slope_norm', 'obv_filter_pass',
    'base_qualify_date', 'tv20_cr_at_q',
    'trigger_date', 'trigger_close', 'trigger_level', 'trigger_N',
    'vol_multiple', 'volx_ge_2', 'volx_ge_5',
    'fired_n40', 'fired_n60', 'fired_n100',
    'days_q_to_trigger', 'trigger_day_move_pct', 'tv20_cr_at_trigger',
    'entry_date', 'entry_open', 'dist_to_left_rim_pct',
    'pattern_quality',
    # ---- INFORMATION ONLY BELOW THIS LINE - never used for selection or ranking ----
    'info_fwd60_pct', 'info_fwd120_pct', 'info_fwd250_pct',
    'info_st73_exit_date', 'info_st73_exit_close', 'info_st73_ret_pct',
    'info_st73_bars_held', 'info_st73_still_open', 'info_mfe_250_pct', 'info_mae_250_pct',
]


def supertrend(high, low, close, period=ST_PERIOD, mult=ST_MULT):
    n = len(close)
    tr = np.empty(n); tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.full(n, np.nan)
    if n <= period:
        return np.zeros(n, dtype=np.int8)
    atr[period - 1] = tr[:period].mean()
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
    d[:period] = 0
    return d


def quad_fit(logc):
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
    return float(a), float(1.0 - ((logc - yh) ** 2).sum() / sst), float((-b / (2.0 * a)) / max(m - 1, 1))


def clip01(v):
    return float(min(max(v, 0.0), 1.0))


def score_pattern(r2, trough_pos, depth, vol_ratio, obv_gain, flat_frac):
    fit = clip01((r2 - 0.70) / 0.25)
    sym = 1.0 - min(abs(trough_pos - 0.50) / 0.20, 1.0)
    dep = 1.0 - min(abs(depth - 0.375) / 0.25, 1.0)
    acc = 0.5 * clip01((vol_ratio - 1.0) / 2.0) + 0.5 * clip01(obv_gain / 0.5)
    flat = clip01((flat_frac - 0.40) / 0.18)
    return round(0.30 * fit + 0.20 * sym + 0.15 * dep + 0.20 * acc + 0.15 * flat, 4)


def load_symbol(con, sym):
    df = pd.read_sql_query(
        "SELECT date,open,high,low,close,volume FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date", con, params=(sym,))
    if df.empty:
        return None
    df = df[(df['volume'].notna()) & (df['volume'] > 0)]
    df = df[(df['close'].notna()) & (df['close'] > 0)]
    df = df.drop_duplicates(subset='date', keep='last').reset_index(drop=True)
    return df if len(df) >= MIN_BARS else None


def process_symbol(sym, df, writer, stats):
    date = df['date'].to_numpy()
    o = df['open'].to_numpy(np.float64); h = df['high'].to_numpy(np.float64)
    lo = df['low'].to_numpy(np.float64); c = df['close'].to_numpy(np.float64)
    v = df['volume'].to_numpy(np.float64)
    n = len(c)
    logc = np.log(c)
    ret1 = np.empty(n); ret1[0] = 0.0; ret1[1:] = c[1:] / c[:-1] - 1.0
    obv = np.concatenate(([0.0], np.cumsum(np.sign(np.diff(c)) * v[1:])))
    tv20 = pd.Series(c * v).rolling(20, min_periods=10).median().to_numpy()
    st_dir = supertrend(h, lo, c)

    # --- trigger-side precomputation (all causal: prior-bar windows, shifted by 1) -----
    cs = pd.Series(c)
    prior_hi = {N: cs.rolling(N, min_periods=20).max().shift(1).to_numpy() for N in TRIG_N_FLAGS}
    med20 = pd.Series(v).rolling(VOL_MED_LOOKBACK, min_periods=10).median().shift(1).to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        volx = np.where(med20 > 0, v / med20, np.nan)

    # --- window-mode precomputation ---------------------------------------------------
    pre = {}
    for name, L in MODES:
        if name == 'VAR':
            if n >= 250:
                rim_off = sliding_window_view(c, 250).argmax(axis=1)
                full_rim = np.full(n, -1, dtype=np.int64)
                full_rim[249:] = np.arange(n - 249) + rim_off
            else:
                full_rim = np.full(n, -1, dtype=np.int64)
            exp_rim = np.zeros(n, dtype=np.int64); best = 0
            for i in range(n):
                if c[i] > c[best]:
                    best = i
                exp_rim[i] = best
            pre[name] = ('VAR', np.where(full_rim >= 0, full_rim, exp_rim))
        else:
            third = max(L // 3, 5)
            rmax_third = cs.rolling(third, min_periods=third).max().shift(L - third).to_numpy()
            if n >= L:
                sw = sliding_window_view(c, L)
                amin = sw.argmin(axis=1)
                rmin = sw.min(axis=1)
                argmin_pos = np.full(n, np.nan); min_full = np.full(n, np.nan)
                argmin_pos[L - 1:] = amin / (L - 1.0)
                min_full[L - 1:] = rmin
            else:
                argmin_pos = np.full(n, np.nan); min_full = np.full(n, np.nan)
            pre[name] = ('FIX', L, third, rmax_third, min_full, argmin_pos)

    for name, L in MODES:
        p = pre[name]
        state, armed, block_until = 'idle', None, -1
        for t in range(MIN_BARS - 1, n):
            if state == 'armed':
                # expiry is checked BEFORE the trigger, so the wait window is a hard
                # <= MAX_WAIT_BARS (checking it after let 9 events fire on bar 121)
                if t - armed['q'] > MAX_WAIT_BARS:
                    stats['expired'] += 1
                    state, armed, block_until = 'idle', None, t + REARM_BARS_FAIL
                    continue
                fired = _check_trigger(t, c, v, ret1, prior_hi, volx, n)
                if fired is not None:
                    if abs(ret1[t]) <= SPLIT_UP:
                        emit(sym, name, armed, t, fired, date, o, c, h, lo,
                             st_dir, tv20, ret1, volx, prior_hi, n, writer)
                        stats['events'] += 1
                    else:
                        stats['trig_split_rejected'] += 1
                    state, armed, block_until = 'idle', None, t + REARM_BARS
                continue
            if t < block_until or not np.isfinite(tv20[t]) or tv20[t] < LIQ_MIN_TV:
                continue

            # ---- window + rim, causally ---------------------------------------------
            if p[0] == 'VAR':
                i_rim = int(p[1][t])
                if i_rim < 0:
                    continue
                wlen = t - i_rim + 1
                if wlen < VAR_MIN_LEN or wlen > VAR_MAX_LEN:
                    continue
                w0, R = i_rim, c[i_rim]
                ipo_short = 1 if (t + 1) < 250 else 0
            else:
                _, Lf, third, rmax_third, min_full, argmin_pos = p
                if (t + 1) >= Lf:
                    if not np.isfinite(min_full[t]) or not np.isfinite(rmax_third[t]):
                        continue
                    pos = argmin_pos[t]
                    if pos < TROUGH_POS_LO or pos > TROUGH_POS_HI:
                        continue          # cheap vectorised symmetry gate
                    Rv, minv = rmax_third[t], min_full[t]
                    if Rv <= 0:
                        continue
                    d = (Rv - minv) / Rv
                    if d < DEPTH_LO or d > DEPTH_HI:
                        continue
                    if c[t] < minv + LIFTOFF_FRAC * (Rv - minv):
                        continue          # cheap lift-off gate
                    w0, ipo_short = t - Lf + 1, 0
                else:
                    if (t + 1) < MIN_BARS:
                        continue
                    w0, ipo_short = 0, 1
                seg0 = c[w0:t + 1]
                th = max(len(seg0) // 3, 5)
                i_rim = w0 + int(np.argmax(seg0[:th]))
                R = c[i_rim]

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
            trough = seg[i_tr_loc]
            depth = (R - trough) / R
            if depth < DEPTH_LO or depth > DEPTH_HI:
                continue
            # v2 base completion: minimal lift-off off the low (NOT a return to the rim)
            if c[t] < trough + LIFTOFF_FRAC * (R - trough):
                continue
            wret = ret1[w0 + 1:t + 1]
            if wret.size and (wret.min() < SPLIT_DOWN or wret.max() > SPLIT_UP):
                stats['split_rejected'] += 1
                continue
            flat_frac = float((seg <= trough + FLAT_DEPTH_FRAC * (R - trough)).sum()) / m
            if flat_frac < FLAT_MIN_FRAC:
                continue
            qf = quad_fit(logc[w0:t + 1])
            if qf is None:
                continue
            a, r2, vertex = qf
            if a <= 0 or r2 < R2_MIN or not (VERTEX_LO <= vertex <= VERTEX_HI):
                continue
            lv, rv = v[w0:i_tr + 1], v[i_tr:t + 1]
            lmed, rmed = (np.median(lv) if lv.size else 0.0), (np.median(rv) if rv.size else 0.0)
            vol_ratio = float(rmed / lmed) if lmed > 0 else np.nan
            rsum = float(rv.sum())
            obv_gain = float((obv[t] - obv[i_tr]) / rsum) if rsum > 0 else np.nan
            k = t - i_tr + 1
            if k >= 3:
                mv = float(rv.mean()) if rv.size else np.nan
                sl = float(np.polyfit(np.arange(k, dtype=np.float64), obv[i_tr:t + 1], 1)[0])
                obv_slope = sl / mv if mv and mv > 0 else np.nan
            else:
                obv_slope = np.nan
            obv_pass = int(bool(np.isfinite(obv_slope) and obv_slope > 0 and
                                np.isfinite(obv_gain) and obv_gain > 0 and
                                np.isfinite(vol_ratio) and vol_ratio >= VOL_RATIO_MIN))
            armed = dict(q=t, w0=w0, i_rim=i_rim, i_tr=i_tr, rim_level=float(R),
                         depth=float(depth), base_len=m, trough_pos=float(trough_pos),
                         r2=float(r2), curv=float(a), vertex=float(vertex),
                         flat_frac=float(flat_frac), vol_ratio=vol_ratio, obv_gain=obv_gain,
                         obv_slope=obv_slope, obv_pass=obv_pass, ipo_short=int(ipo_short),
                         tv20_q=float(tv20[t]))
            state = 'armed'
            stats['qualified'] += 1


def _check_trigger(t, c, v, ret1, prior_hi, volx, n):
    """v2 entry trigger, strictly causal. Returns the cleared level or None."""
    lvl = prior_hi[TRIG_N][t]
    if not np.isfinite(lvl) or c[t] <= lvl:
        return None
    if not np.isfinite(volx[t]) or volx[t] < TRIG_K:
        return None
    if t == 0 or c[t] <= c[t - 1]:          # must be an up-candle
        return None
    return float(lvl)


def emit(sym, mode, a, t, lvl, date, o, c, h, lo, st_dir, tv20, ret1, volx, prior_hi, n, writer):
    nxt = t + 1
    if nxt < n:
        entry = float(o[nxt]); entry_date = str(date[nxt])[:10]
    else:
        entry, entry_date = np.nan, ''

    def fwd(hz):
        if not np.isfinite(entry):
            return ''
        j = min(nxt + hz - 1, n - 1)
        return round(100.0 * (c[j] / entry - 1.0), 2) if j > nxt else ''

    st_date, st_close, st_ret, st_bars, st_open = '', '', '', '', 1
    if np.isfinite(entry):
        j = nxt
        while j < n and st_dir[j] != -1:
            j += 1
        if j < n:
            st_date, st_close = str(date[j])[:10], round(float(c[j]), 2)
            st_ret, st_bars, st_open = round(100.0 * (c[j] / entry - 1.0), 2), j - nxt + 1, 0
        else:
            st_bars, st_ret = n - nxt, round(100.0 * (c[n - 1] / entry - 1.0), 2)
    mfe = mae = ''
    if np.isfinite(entry) and nxt < n:
        j = min(nxt + 250, n)
        mfe = round(100.0 * (h[nxt:j].max() / entry - 1.0), 2)
        mae = round(100.0 * (lo[nxt:j].min() / entry - 1.0), 2)

    R = a['rim_level']
    writer.writerow({
        'symbol': sym, 'mode': mode, 'ipo_short_window': a['ipo_short'],
        'left_rim_date': str(date[a['i_rim']])[:10], 'left_rim_level': round(R, 2),
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
        'base_qualify_date': str(date[a['q']])[:10],
        'tv20_cr_at_q': round(a['tv20_q'] / 1e7, 2),
        'trigger_date': str(date[t])[:10], 'trigger_close': round(float(c[t]), 2),
        'trigger_level': round(lvl, 2), 'trigger_N': TRIG_N,
        'vol_multiple': round(float(volx[t]), 2),
        'volx_ge_2': int(volx[t] >= 2), 'volx_ge_5': int(volx[t] >= 5),
        'fired_n40': int(np.isfinite(prior_hi[40][t]) and c[t] > prior_hi[40][t]),
        'fired_n60': int(np.isfinite(prior_hi[60][t]) and c[t] > prior_hi[60][t]),
        'fired_n100': int(np.isfinite(prior_hi[100][t]) and c[t] > prior_hi[100][t]),
        'days_q_to_trigger': t - a['q'],
        'trigger_day_move_pct': round(100 * float(ret1[t]), 2),
        'tv20_cr_at_trigger': round(float(tv20[t]) / 1e7, 2) if np.isfinite(tv20[t]) else '',
        'entry_date': entry_date,
        'entry_open': round(entry, 2) if np.isfinite(entry) else '',
        'dist_to_left_rim_pct': round(100.0 * (R - c[t]) / R, 2),
        'pattern_quality': score_pattern(
            a['r2'], a['trough_pos'], a['depth'],
            a['vol_ratio'] if np.isfinite(a['vol_ratio']) else 0.0,
            a['obv_gain'] if np.isfinite(a['obv_gain']) else 0.0, a['flat_frac']),
        'info_fwd60_pct': fwd(60), 'info_fwd120_pct': fwd(120), 'info_fwd250_pct': fwd(250),
        'info_st73_exit_date': st_date, 'info_st73_exit_close': st_close,
        'info_st73_ret_pct': st_ret, 'info_st73_bars_held': st_bars,
        'info_st73_still_open': st_open, 'info_mfe_250_pct': mfe, 'info_mae_250_pct': mae,
    })


def main():
    global TRIG_K, TRIG_N
    t0 = time.time()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    syms = [r[0] for r in con.execute(
        "SELECT symbol FROM market_data_unified WHERE timeframe='day' "
        "GROUP BY symbol HAVING COUNT(*)>=%d ORDER BY symbol" % MIN_BARS)]
    syms = [s for s in syms if not any(p in s.upper() for p in ETF_PAT)]
    print('v2 universe after ETF/index exclusion: %d' % len(syms), flush=True)
    print('trigger: close > prior %d-bar high AND vol >= %.1fx prior %d-bar median AND up-candle'
          % (TRIG_N, TRIG_K, VOL_MED_LOOKBACK), flush=True)

    out = OUT_CSV
    for arg in sys.argv[1:]:
        if arg.startswith('--symbols='):
            only = set(arg.split('=', 1)[1].split(','))
            syms = [s for s in syms if s in only]
            out = OUTDIR / 'smoke_events_v2.csv'
            print('SMOKE TEST restricted to: %s' % syms, flush=True)
        elif arg.startswith('--k='):          # sensitivity probe only
            TRIG_K = float(arg.split('=', 1)[1])
            out = OUTDIR / 'smoke_events_v2.csv'
            print('TRIG_K overridden to %.1f' % TRIG_K, flush=True)
        elif arg.startswith('--n='):          # sensitivity probe only
            TRIG_N = int(arg.split('=', 1)[1])
            out = OUTDIR / 'smoke_events_v2.csv'
            print('TRIG_N overridden to %d' % TRIG_N, flush=True)

    stats = dict(events=0, qualified=0, expired=0, split_rejected=0,
                 trig_split_rejected=0, skipped=0)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, s in enumerate(syms, 1):
            try:
                df = load_symbol(con, s)
                if df is None:
                    stats['skipped'] += 1
                    continue
                process_symbol(s, df, w, stats)
            except Exception as e:
                print('  ERR %s: %r' % (s, e), flush=True)
            fh.flush()
            if i % 200 == 0:
                el = time.time() - t0
                print('[%d/%d] %.0fs | events=%d qualified=%d expired=%d split_rej=%d | ETA %.0fs'
                      % (i, len(syms), el, stats['events'], stats['qualified'],
                         stats['expired'], stats['split_rejected'], el / i * (len(syms) - i)),
                      flush=True)
    con.close()
    print('DONE in %.0fs' % (time.time() - t0), flush=True)
    print('stats:', stats, flush=True)
    print('csv:', out, flush=True)


if __name__ == '__main__':
    main()
