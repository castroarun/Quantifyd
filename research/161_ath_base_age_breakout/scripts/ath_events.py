"""
research/161 — every NEW ALL-TIME-HIGH CLOSE, with the attributes the sweep filters on.

One row per (symbol, trigger day) where close[t] > max(close[:t]). NO re-arm and NO
filtering is applied here: the sweep applies X / depth / K / saucer and only THEN the
60-bar re-arm, because which event is "first" depends on the filter.

Strictly causal: ATH is the running maximum of closes STRICTLY BEFORE t; the volume
median and the liquidity measure are shifted back one bar; the fill is the next open.

Split guard: market_data.db is not retroactively split adjusted, so a pre-split row sits
at the old price scale and would fake an unreachable all-time high. The series is
truncated to the bars AFTER the last day-over-day fall worse than -35%.

Saucer flag: the research/159 v3 base recognition evaluated on the window from the bar
that set the previous ATH to t. That window must be 90-400 bars, so **saucer=on
implicitly forces X >= 89** - an interaction recorded in the STATUS doc, not hidden.
"""
import csv
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUTDIR = Path(__file__).resolve().parents[1] / 'results'
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / 'ath_events.csv'

MIN_BARS = 90
SPLIT_DOWN, SPLIT_UP = -0.35, 0.50
VOL_MED = 20
LIQ_MIN_TV = 2e7                     # Rs 2 cr default floor; Rs 5 cr re-run filters later
SAUCER_MIN, SAUCER_MAX = 90, 400
R2_MIN = 0.70
TROUGH_LO, TROUGH_HI = 0.30, 0.70
DEPTH_LO, DEPTH_HI = 0.20, 0.70
FLAT_DEPTH_FRAC, FLAT_MIN_FRAC = 0.3333, 0.40
VERTEX_LO, VERTEX_HI = 0.30, 0.70

ETF_PAT = ('BEES', 'ETF', 'IETF', 'GOLD', 'SILVER', 'LIQUID', 'NIFTY', 'SENSEX',
           'BANKNIFTY', 'MIDCAP', 'SMALLCAP', 'INDEX', 'MAFANG', 'HNGSNGBEES')
DROP = {'SILLYMONKS'}                # identical series to CRESTO (research/159)

FIELDS = ['symbol', 'trigger_date', 'trigger_close', 'prev_ath', 'prev_ath_date',
          'x_bars', 'depth_pct', 'vol_mult', 'tv20_cr', 'saucer_ok', 'saucer_r2',
          'gap_through', 'trigger_day_move_pct', 'entry_date', 'entry_open',
          'hist_bars', 'split_cut']


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


def saucer_check(seg, logseg):
    """research/159 v3 base recognition on the window [prev ATH .. t]."""
    m = len(seg)
    R = seg[0]
    itl = int(np.argmin(seg))
    if m < 2:
        return 0, np.nan
    tpos = itl / (m - 1)
    if not (TROUGH_LO <= tpos <= TROUGH_HI):
        return 0, np.nan
    trough = seg[itl]
    depth = (R - trough) / R
    if not (DEPTH_LO <= depth <= DEPTH_HI):
        return 0, np.nan
    flat = float((seg <= trough + FLAT_DEPTH_FRAC * (R - trough)).sum()) / m
    if flat < FLAT_MIN_FRAC:
        return 0, np.nan
    qf = quad_fit(logseg)
    if qf is None:
        return 0, np.nan
    a, r2, vertex = qf
    if a <= 0 or r2 < R2_MIN or not (VERTEX_LO <= vertex <= VERTEX_HI):
        return 0, round(r2, 3)
    return 1, round(r2, 3)


def main():
    t0 = time.time()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    syms = [r[0] for r in con.execute(
        "SELECT symbol FROM market_data_unified WHERE timeframe='day' "
        "GROUP BY symbol HAVING COUNT(*)>=%d ORDER BY symbol" % MIN_BARS)]
    syms = [s for s in syms if s not in DROP and not any(p in s.upper() for p in ETF_PAT)]
    print('universe after exclusions: %d symbols' % len(syms), flush=True)

    only = None
    out = OUT_CSV
    for a in sys.argv[1:]:
        if a.startswith('--symbols='):
            only = set(a.split('=', 1)[1].split(','))
            syms = [s for s in syms if s in only]
            out = OUTDIR / 'smoke_ath_events.csv'
            print('SMOKE: %s' % syms, flush=True)

    n_ev = 0
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for si, sym in enumerate(syms, 1):
            try:
                d = pd.read_sql_query(
                    "SELECT date,open,close,volume FROM market_data_unified WHERE symbol=? "
                    "AND timeframe='day' ORDER BY date", con, params=(sym,))
                d = d[(d['volume'] > 0) & (d['close'] > 0)].drop_duplicates(subset='date', keep='last')
                if len(d) < MIN_BARS:
                    continue
                r = d['close'].pct_change()
                hits = r.index[r < SPLIT_DOWN]
                cut = int(len(hits) > 0)
                if cut:
                    d = d.loc[hits.max():]
                d = d.reset_index(drop=True)
                if len(d) < MIN_BARS:
                    continue
                dates = d['date'].to_numpy()
                o = d['open'].to_numpy(np.float64)
                c = d['close'].to_numpy(np.float64)
                v = d['volume'].to_numpy(np.float64)
                n = len(c)
                logc = np.log(c)
                ret1 = np.empty(n); ret1[0] = 0.0; ret1[1:] = c[1:] / c[:-1] - 1.0
                med = pd.Series(v).rolling(VOL_MED, min_periods=VOL_MED).median().shift(1).to_numpy()
                with np.errstate(divide='ignore', invalid='ignore'):
                    volx = np.where(med > 0, v / med, np.nan)
                tv20 = pd.Series(c * v).rolling(20, min_periods=10).median().to_numpy()

                # running: index of the ATH close strictly before t, and the min close since it
                ath_idx = 0
                min_since = c[0]
                for t in range(1, n):
                    prev_ath = c[ath_idx]
                    if c[t] > prev_ath:
                        # ---- a NEW all-time-high close -------------------------------
                        if (t + 1 < n and np.isfinite(tv20[t]) and tv20[t] >= LIQ_MIN_TV
                                and abs(ret1[t]) <= SPLIT_UP and np.isfinite(volx[t])):
                            x_bars = t - ath_idx
                            depth = 100.0 * (prev_ath - min_since) / prev_ath
                            sok, sr2 = 0, ''
                            if SAUCER_MIN <= x_bars <= SAUCER_MAX:
                                sok, sr2 = saucer_check(c[ath_idx:t + 1], logc[ath_idx:t + 1])
                            w.writerow(dict(
                                symbol=sym, trigger_date=str(dates[t])[:10],
                                trigger_close=round(c[t], 2), prev_ath=round(prev_ath, 2),
                                prev_ath_date=str(dates[ath_idx])[:10], x_bars=x_bars,
                                depth_pct=round(depth, 2), vol_mult=round(float(volx[t]), 2),
                                tv20_cr=round(float(tv20[t]) / 1e7, 2),
                                saucer_ok=sok, saucer_r2=sr2,
                                gap_through=int(o[t + 1] > prev_ath),
                                trigger_day_move_pct=round(100 * ret1[t], 2),
                                entry_date=str(dates[t + 1])[:10],
                                entry_open=round(o[t + 1], 2),
                                hist_bars=t, split_cut=cut))
                            n_ev += 1
                        ath_idx = t
                        min_since = c[t]
                    elif c[t] < min_since:
                        min_since = c[t]
            except Exception as e:
                print('  ERR %s: %r' % (sym, e), flush=True)
            if si % 300 == 0:
                el = time.time() - t0
                print('[%d/%d] %.0fs  events=%d  ETA %.0fs'
                      % (si, len(syms), el, n_ev, el / si * (len(syms) - si)), flush=True)
    con.close()
    print('DONE %.0fs: %d new-ATH-close events -> %s' % (time.time() - t0, n_ev, out), flush=True)


if __name__ == '__main__':
    main()
