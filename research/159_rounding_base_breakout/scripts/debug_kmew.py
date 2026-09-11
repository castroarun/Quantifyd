"""research/159 — trace every gate for KMEW to find why the detector did not fire."""
import sqlite3, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_rounding_base as D

con = sqlite3.connect(f'file:{D.DB}?mode=ro', uri=True)
df = D.load_symbol(con, 'KMEW'); con.close()
c = df['close'].to_numpy(float); v = df['volume'].to_numpy(float)
date = df['date'].to_numpy(); n = len(c)
logc = np.log(c)
ret1 = np.empty(n); ret1[0] = 0.0; ret1[1:] = c[1:]/c[:-1]-1
tv = c*v
tv20 = pd.Series(tv).rolling(20, min_periods=10).median().to_numpy()

i_rim = 13; R = c[i_rim]
print('rim idx=%d %s %.2f' % (i_rim, str(date[i_rim])[:10], R))
print('one-day moves in [13..207]: min=%.3f max=%.3f' % (ret1[14:208].min(), ret1[14:208].max()))
bad = [(str(date[i])[:10], round(ret1[i],3)) for i in range(14,208) if ret1[i] < D.SPLIT_DOWN or ret1[i] > D.SPLIT_UP]
print('split-guard offenders:', bad)

# first day close >= 0.95R after the trough
i_tr = int(np.argmin(c[:250])); print('trough idx=%d %s %.2f depth=%.2f%%' % (i_tr, str(date[i_tr])[:10], c[i_tr], 100*(R-c[i_tr])/R))
cand = [t for t in range(i_tr+1, n) if c[t] >= D.RECOVERY_BAND*R]
print('first 5 days with close >= 0.95*R:', [(t, str(date[t])[:10], round(c[t],1)) for t in cand[:5]])

for t in cand[:6]:
    w0 = i_rim; seg = c[w0:t+1]; m = len(seg)
    itl = int(np.argmin(seg)); tp = itl/(m-1)
    depth = (R-seg[itl])/R
    flat = float((seg <= seg[itl] + D.FLAT_DEPTH_FRAC*(R - seg[itl])).sum())/m  # depth-relative no-V test
    qf = D.quad_fit(logc[w0:t+1])
    wret = ret1[w0+1:t+1]
    lv=v[w0:w0+itl+1]; rv=v[w0+itl:t+1]
    vr = np.median(rv)/np.median(lv) if np.median(lv)>0 else float('nan')
    print('t=%d %s m=%d tp=%.3f depth=%.3f flat=%.3f r2=%.3f curv=%+.2e vtx=%.3f tv20=%.2fcr split_ok=%s volratio=%.2f'
          % (t, str(date[t])[:10], m, tp, depth, flat,
             qf[1] if qf else -9, qf[0] if qf else 0, qf[2] if qf else -9,
             tv20[t]/1e7, (wret.min()>=D.SPLIT_DOWN and wret.max()<=D.SPLIT_UP), vr))
    fails=[]
    if not (D.TROUGH_POS_LO<=tp<=D.TROUGH_POS_HI): fails.append('trough_pos')
    if not (D.DEPTH_LO<=depth<=D.DEPTH_HI): fails.append('depth')
    if flat < D.FLAT_MIN_FRAC: fails.append('flat/noV')
    if qf is None or qf[0]<=0: fails.append('curvature')
    elif qf[1] < D.R2_MIN: fails.append('r2')
    elif not (D.VERTEX_LO<=qf[2]<=D.VERTEX_HI): fails.append('vertex')
    if not (wret.min()>=D.SPLIT_DOWN and wret.max()<=D.SPLIT_UP): fails.append('split')
    if not (tv20[t]>=D.LIQ_MIN_TV): fails.append('liquidity')
    if m < D.MIN_BARS: fails.append('min_bars')
    print('   FAILS:', fails if fails else 'NONE -> would qualify')
