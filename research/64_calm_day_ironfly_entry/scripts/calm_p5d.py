"""research/64 P5d — DEFINED-RISK bullish skew: cap the jade-lizard crash tail with a long protective
put (turns the naked short put into a put spread). Sweep the long-put strike (-3.5/-4/-5%) for the
EV-vs-tail sweet spot; compare to naked jade, tuned broken-wing, and the day-1-confirmed entry (re-centred
at day-1 close, the actual hold). Proxy premiums (VIX-scaled). VIX 13-22. Per 10-lot. AlgoTest for exact."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
C = n.close; clv = C.values
vx = vix.reindex(n.index, method="ffill").shift(1)
HH = 5
r0 = np.full(len(C), np.nan); r1 = np.full(len(C), np.nan); day1 = np.full(len(C), np.nan)
for i in range(1, len(C)-HH):
    es = clv[i-1]
    r0[i] = (clv[i+HH-1]-es)/es*100                 # entry day-0 -> day-5
    day1[i] = (clv[i]-es)/es*100                     # day-1 realised
    r1[i] = (clv[i+HH-1]-clv[i])/clv[i]*100          # entry day-1 -> day-5 (confirmed hold)
d = pd.DataFrame({"r0": r0, "r1": r1, "day1": day1, "vix": vx.values}, index=n.index).dropna()
d = d[(d.vix >= 13) & (d.vix <= 22)]
SPOT, QTY = 23500, 650; RUP = SPOT*QTY/100
straddle = lambda v: 0.8*(v/100)*np.sqrt(HH/252)*100

def jade(r, v, lp=None, lp_frac=0.0):
    s = straddle(v); cr = 0.42*s + 0.30 - lp_frac*s
    pay = cr + min(0.0, r + 2.0) - min(max(r - 1.0, 0.0), 1.5)   # short 2% put + short 1/2.5 call spread
    if lp is not None:
        pay += max(-lp - r, 0.0)                                  # long lp% put -> caps downside
    return pay

def bwf(r, v):
    s = straddle(v); pay = 0.45*s - abs(r)
    if r < -2.5: pay += (-2.5 - r)
    if r > 4.0:  pay += (r - 4.0)
    return pay

def summ(name, series):
    rup = series*RUP
    print(f"  {name:36s} EV=Rs{rup.mean():8,.0f}  win={(series>0).mean()*100:4.0f}%  worst=Rs{rup.min():9,.0f}  std=Rs{rup.std():8,.0f}")

print(f"N={len(d)} weeks (VIX 13-22)  per 10-lot; 1% spot=Rs{RUP:,.0f}\n=== UNCONDITIONAL (enter day-0) ===")
summ("Jade NAKED (short 2% put)", d.apply(lambda x: jade(x.r0, x.vix), axis=1))
summ("Jade + long 5% put (wide)", d.apply(lambda x: jade(x.r0, x.vix, lp=5.0, lp_frac=0.12), axis=1))
summ("Jade + long 4% put", d.apply(lambda x: jade(x.r0, x.vix, lp=4.0, lp_frac=0.18), axis=1))
summ("Jade + long 3.5% put (tight)", d.apply(lambda x: jade(x.r0, x.vix, lp=3.5, lp_frac=0.24), axis=1))
summ("Broken-wing fly (tuned -2.5/+4)", d.apply(lambda x: bwf(x.r0, x.vix), axis=1))

print("\n=== DAY-1 CONFIRMED (enter day-1 close after day-1 up>0.5%, hold to day-5; move r1) ===")
conf = d[d.day1 > 0.5]
print(f"  triggers {len(conf)/len(d)*100:.0f}% of weeks (n={len(conf)})")
summ("Jade NAKED | day-1 up>0.5%", conf.apply(lambda x: jade(x.r1, x.vix), axis=1))
summ("Jade + long 4% put | day-1 up>0.5%", conf.apply(lambda x: jade(x.r1, x.vix, lp=4.0, lp_frac=0.18), axis=1))

print("\n=== tail detail: worst-5 weeks for naked vs +4%put (unconditional) ===")
d["naked"] = d.apply(lambda x: jade(x.r0, x.vix)*RUP, axis=1)
d["def4"] = d.apply(lambda x: jade(x.r0, x.vix, lp=4.0, lp_frac=0.18)*RUP, axis=1)
w = d.nsmallest(5, "naked")[["r0", "vix", "naked", "def4"]]
print(w.round(0).to_string())
