"""research/64 P5c — skewed-structure EV comparison on the actual NIFTY weekly move distribution.
Symmetric iron fly vs JADE LIZARD (bullish, no upside risk) vs BROKEN-WING fly (bullish lean, defined
risk). Payoffs at the 5-day horizon (~weekly expiry), credits scaled by VIX (proxy). Also the day-1
confirmed bullish variant (P5b follow-through). Per-10-lot (qty 650). PROXY premiums — AlgoTest for exact."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
C = n.close; clv = C.values
vx = vix.reindex(n.index, method="ffill").shift(1)   # causal entry VIX
HH = 5
end_ret = np.full(len(C), np.nan); day1 = np.full(len(C), np.nan)
for i in range(1, len(C)-HH):
    es = clv[i-1]; end_ret[i] = (clv[i+HH-1]-es)/es*100; day1[i] = (clv[i]-es)/es*100
d = pd.DataFrame({"r": end_ret, "day1": day1, "vix": vx.values}, index=n.index).dropna()
d = d[(d.vix >= 13) & (d.vix <= 22)]    # the tradeable VIX band (from P3)
SPOT, QTY = 23500, 650                  # rupees per 1% of spot per 10 lots = 23500*650/100 = 152,750
RUP = SPOT*QTY/100

def straddle(vixv): return 0.8*(vixv/100)*np.sqrt(HH/252)*100   # ATM straddle, % of spot

def fly(r, v):                          # symmetric iron fly +-2% wings
    return 0.55*straddle(v) - min(abs(r), 2.0)

def jade(r, v):                         # short 2% put + short 1%/2.5% call spread (bullish, ~no upside risk)
    cr = 0.42*straddle(v) + 0.30
    return cr + min(0.0, r + 2.0) - min(max(r - 1.0, 0.0), 1.5)

def bwf(r, v):                          # broken-wing fly: short straddle, put wing -1.5, call wing +3.5
    cr = 0.45*straddle(v); pay = cr - abs(r)
    if r < -1.5: pay += (-1.5 - r)      # downside capped at -1.5
    if r > 3.5:  pay += (r - 3.5)       # upside capped at -3.5
    return pay

def summ(name, pnl_pct):
    rup = pnl_pct*RUP
    print(f"  {name:34s} EV=Rs{rup.mean():8,.0f}  win={ (pnl_pct>0).mean()*100:4.0f}%  "
          f"worst=Rs{rup.min():9,.0f}  best=Rs{rup.max():8,.0f}  std=Rs{rup.std():8,.0f}")

print(f"N={len(d)} weeks (VIX 13-22)  | per 10-lot; 1% spot = Rs{RUP:,.0f}")
print("\n=== ALL weeks (unconditional) ===")
summ("Iron fly (symmetric)", d.apply(lambda x: fly(x.r, x.vix), axis=1))
summ("Jade lizard (bullish, ~no up-risk)", d.apply(lambda x: jade(x.r, x.vix), axis=1))
summ("Broken-wing fly (bullish lean)", d.apply(lambda x: bwf(x.r, x.vix), axis=1))

print("\n=== win-zone check: P&L sign by move bucket (jade vs fly) ===")
for lab, lo, hi in [("strong_bear<-3", -100, -3), ("mild_bear -3..-1.5", -3, -1.5), ("calm +-1.5", -1.5, 1.5), ("mild_bull 1.5..3", 1.5, 3), ("strong_bull>3", 3, 100)]:
    s = d[(d.r > lo) & (d.r <= hi)] if lo > -100 else d[d.r <= hi]
    if lo == -100: s = d[d.r <= hi]
    elif hi == 100: s = d[d.r > lo]
    else: s = d[(d.r > lo) & (d.r <= hi)]
    if not len(s): continue
    fp = s.apply(lambda x: fly(x.r, x.vix), axis=1).mean()*RUP
    jp = s.apply(lambda x: jade(x.r, x.vix), axis=1).mean()*RUP
    print(f"  {lab:20s} ({len(s):4d}, {len(s)/len(d)*100:4.0f}%)  fly Rs{fp:8,.0f}   jade Rs{jp:8,.0f}")

print("\n=== day-1 CONFIRMED bullish (P5b): enter jade only when day-1 move > +0.5% ===")
conf = d[d.day1 > 0.5]
print(f"  triggers on {len(conf)/len(d)*100:.0f}% of weeks (n={len(conf)})")
summ("Jade | day-1 up>0.5%", conf.apply(lambda x: jade(x.r, x.vix), axis=1))
summ("Iron fly | day-1 up>0.5% (compare)", conf.apply(lambda x: fly(x.r, x.vix), axis=1))
print("\n(PROXY credits scaled by VIX; jade downside is a short put = fat left tail — see 'worst'. AlgoTest for exact Rs.)")
