"""research/64 P5e — BEARISH lean (mirror of the jade). Reverse jade lizard: SELL +2% call + SELL -1%
put + BUY -2.5% put (no downside risk, wins flat-to-down, loses on UP). Defined version adds long +4%
call to cap the up tail. Compare to the bullish jade. Unconditional + day-1-DOWN-confirmed. Proxy
premiums, VIX 13-22, per 10-lot. Tests whether a bear skew can work against NIFTY's up-drift."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
C = n.close; clv = C.values
vx = vix.reindex(n.index, method="ffill").shift(1)
HH = 5
r0 = np.full(len(C), np.nan); r1 = np.full(len(C), np.nan); day1 = np.full(len(C), np.nan)
for i in range(1, len(C)-HH):
    es = clv[i-1]; r0[i] = (clv[i+HH-1]-es)/es*100; day1[i] = (clv[i]-es)/es*100
    r1[i] = (clv[i+HH-1]-clv[i])/clv[i]*100
d = pd.DataFrame({"r0": r0, "r1": r1, "day1": day1, "vix": vx.values}, index=n.index).dropna()
d = d[(d.vix >= 13) & (d.vix <= 22)]
SPOT, QTY = 23500, 650; RUP = SPOT*QTY/100
straddle = lambda v: 0.8*(v/100)*np.sqrt(HH/252)*100

def bull_jade(r, v, lp=4.0, lp_frac=0.18):     # SELL -2%put SELL +1%call BUY +2.5%call (+ long lp%put)
    s = straddle(v); cr = 0.42*s + 0.30 - lp_frac*s
    pay = cr + min(0.0, r + 2.0) - min(max(r - 1.0, 0.0), 1.5)
    if lp: pay += max(-lp - r, 0.0)
    return pay

def bear_jade(r, v, lc=None, lc_frac=0.0):     # SELL +2%call SELL -1%put BUY -2.5%put (+ long lc%call)
    s = straddle(v); cr = 0.42*s + 0.30 - lc_frac*s
    pay = cr - max(r - 2.0, 0.0) - min(max(-1.0 - r, 0.0), 1.5)
    if lc: pay += max(r - lc, 0.0)
    return pay

def summ(name, series):
    rup = series*RUP
    print(f"  {name:38s} EV=Rs{rup.mean():8,.0f}  win={(series>0).mean()*100:4.0f}%  worst=Rs{rup.min():9,.0f}")

print(f"N={len(d)} weeks (VIX 13-22)  per 10-lot\n=== UNCONDITIONAL ===")
summ("BULL jade + 4% put (reference)", d.apply(lambda x: bull_jade(x.r0, x.vix), axis=1))
summ("BEAR jade NAKED (short 2% call)", d.apply(lambda x: bear_jade(x.r0, x.vix), axis=1))
summ("BEAR jade + long 4% call (defined)", d.apply(lambda x: bear_jade(x.r0, x.vix, lc=4.0, lc_frac=0.18), axis=1))

print("\n=== win-zone by move bucket: BULL jade+4put vs BEAR jade+4call ===")
for lab, lo, hi in [("strong_bear<-3", -100, -3), ("mild_bear -3..-1.5", -3, -1.5), ("calm +-1.5", -1.5, 1.5), ("mild_bull 1.5..3", 1.5, 3), ("strong_bull>3", 3, 100)]:
    s = d[(d.r0 > lo) & (d.r0 <= hi)] if (lo > -100 and hi < 100) else (d[d.r0 <= hi] if lo == -100 else d[d.r0 > lo])
    if not len(s): continue
    bp = s.apply(lambda x: bull_jade(x.r0, x.vix), axis=1).mean()*RUP
    rp = s.apply(lambda x: bear_jade(x.r0, x.vix, lc=4.0, lc_frac=0.18), axis=1).mean()*RUP
    print(f"  {lab:20s} ({len(s):4d}, {len(s)/len(d)*100:4.0f}%)  BULL Rs{bp:8,.0f}   BEAR Rs{rp:8,.0f}")

print("\n=== DAY-1 CONFIRMED: bull after day-1 UP>0.5% vs bear after day-1 DOWN>0.5% (move r1) ===")
cu = d[d.day1 > 0.5]; cd = d[d.day1 < -0.5]
print(f"  up-trigger {len(cu)/len(d)*100:.0f}% of weeks | down-trigger {len(cd)/len(d)*100:.0f}% of weeks")
summ("BULL jade+4put | day-1 UP>0.5%", cu.apply(lambda x: bull_jade(x.r1, x.vix), axis=1))
summ("BEAR jade+4call | day-1 DOWN>0.5%", cd.apply(lambda x: bear_jade(x.r1, x.vix, lc=4.0, lc_frac=0.18), axis=1))
