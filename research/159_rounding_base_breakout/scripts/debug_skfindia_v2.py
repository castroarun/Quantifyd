"""research/159 v2 — why does SKFINDIA's base not qualify until 19-May-2025,
one bar AFTER Arun's 16-May-2025 breakout? Trace every gate, day by day."""
import sqlite3, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_rounding_base_v2 as D

SYM = sys.argv[1] if len(sys.argv) > 1 else 'SKFINDIA'
con = sqlite3.connect(f'file:{D.DB}?mode=ro', uri=True)
df = D.load_symbol(con, SYM); con.close()
c = df['close'].to_numpy(float); v = df['volume'].to_numpy(float)
date = pd.to_datetime(df['date']).to_numpy(); n = len(c)
logc = np.log(c)
ret1 = np.empty(n); ret1[0] = 0.0; ret1[1:] = c[1:]/c[:-1]-1
tv20 = pd.Series(c*v).rolling(20, min_periods=10).median().to_numpy()
ds = [str(x)[:10] for x in date]

lo_d, hi_d = '2025-04-25', '2025-06-06'
idx = [i for i in range(n) if lo_d <= ds[i] <= hi_d]
print('%s: tracing L120 base gates %s .. %s' % (SYM, lo_d, hi_d))
print('%-11s %6s %7s %6s %6s %6s %6s %6s %7s  %s' %
      ('date', 'close', 'rim', 'tpos', 'depth', 'liftok', 'flat', 'r2', 'vertex', 'FAILS'))
for t in idx:
    L = 120
    if (t+1) < L: continue
    w0 = t-L+1
    seg = c[w0:t+1]; m = len(seg)
    th = max(m//3, 5)
    i_rim = w0 + int(np.argmax(seg[:th])); R = c[i_rim]
    itl = int(np.argmin(seg)); i_tr = w0+itl
    tpos = itl/(m-1); trough = seg[itl]
    depth = (R-trough)/R
    liftok = c[t] >= trough + D.LIFTOFF_FRAC*(R-trough)
    flat = float((seg <= trough + D.FLAT_DEPTH_FRAC*(R-trough)).sum())/m
    qf = D.quad_fit(logc[w0:t+1])
    a, r2, vtx = qf if qf else (0, -9, -9)
    wret = ret1[w0+1:t+1]
    fails = []
    if not (D.TROUGH_POS_LO <= tpos <= D.TROUGH_POS_HI): fails.append('trough_pos')
    if not (D.DEPTH_LO <= depth <= D.DEPTH_HI): fails.append('depth')
    if not liftok: fails.append('liftoff')
    if flat < D.FLAT_MIN_FRAC: fails.append('no-V(flat=%.3f)' % flat)
    if a <= 0: fails.append('curvature')
    elif r2 < D.R2_MIN: fails.append('R2=%.3f' % r2)
    elif not (D.VERTEX_LO <= vtx <= D.VERTEX_HI): fails.append('vertex=%.3f' % vtx)
    if wret.size and (wret.min() < D.SPLIT_DOWN or wret.max() > D.SPLIT_UP): fails.append('split')
    if not (np.isfinite(tv20[t]) and tv20[t] >= D.LIQ_MIN_TV): fails.append('liquidity')
    if i_tr <= i_rim: fails.append('trough<=rim')
    print('%-11s %6.1f %7.1f %6.3f %6.3f %6s %6.3f %6.3f %7.3f  %s' %
          (ds[t], c[t], R, tpos, depth, liftok, flat, r2, vtx,
           ','.join(fails) if fails else '--- QUALIFIES ---'))
