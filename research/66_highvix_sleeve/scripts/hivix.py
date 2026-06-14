"""research/66 (G1) — high-VIX (>22) sleeve for the chaos tail (the last idle). Test buy-the-fear /
mean-reversion longs (and filters) on NIFTY during VIX>22. Causal. Cached NIFTY+VIX daily."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
C = n.close; r = C.pct_change(); Cp = C.shift(1)
vxp = vix.reindex(n.index, method="ffill").shift(1)
ma20 = C.rolling(20).mean().shift(1); ma5 = C.rolling(5).mean().shift(1)
delta = C.diff(); rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
rsi = (100-100/(1+rma(delta.clip(lower=0), 14)/rma((-delta).clip(lower=0), 14))).shift(1)
vix5ago = vix.reindex(n.index, method="ffill").shift(6)   # VIX 5 sessions ago (prior close basis)
hi = vxp > 22
systems = {
    "Long every VIX>22 day (buy-fear)": hi.astype(float),
    "Long VIX>22 + RSI<35 (oversold)": (hi & (rsi < 35)).astype(float),
    "Long VIX>22 + VIX falling": (hi & (vxp < vix5ago)).astype(float),
    "Long VIX>22 + close>20DMA (no knife)": (hi & (Cp > ma20)).astype(float),
    "Long VIX>22 + close>5DMA": (hi & (Cp > ma5)).astype(float),
    "SHORT every VIX>22 day (momentum)": -hi.astype(float),
}
nyears = (n.index[-1]-n.index[0]).days/365.25
print(f"VIX>22 days: {int(hi.sum())} ({hi.mean()*100:.0f}% of days)\n")
print(f"{'system':38s} {'days':>5} {'tot%':>6} {'annIn%':>7} {'Sharpe':>7} {'maxDD':>7} {'win%':>6}")
for name, pos in systems.items():
    pos = pos.fillna(0); pnl = (pos*r).fillna(0); eq = (1+pnl).cumprod()
    days = int((pos.abs() > 0).sum()); dd = (eq/eq.cummax()-1).min()
    shp = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() > 0 else 0
    inwin = ((1+pnl[pos.abs() > 0]).prod()-1) if days else 0
    annin = ((1+inwin)**(252/days)-1)*100 if days > 20 else 0
    winr = (pnl[pos.abs() > 0] > 0).mean()*100 if days else 0
    print(f"{name:38s} {days:5d} {(eq.iloc[-1]-1)*100:5.0f}% {annin:6.0f}% {shp:6.2f} {dd*100:6.0f}% {winr:5.0f}%")
print("\nannIn% = annualised return DURING deployed (VIX>22) days. Buy-the-fear works if VIX-spikes mark")
print("NIFTY bottoms (mean-reversion). Compare vs debt; defined-risk in practice (this is the underlying).")
