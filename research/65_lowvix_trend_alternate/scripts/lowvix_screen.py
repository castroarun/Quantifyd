"""research/65 (G1) — low-VIX (<13) trend/momentum ALTERNATE, scoped to the fly/jade idle windows.
Does a long-NIFTY trend system make money exactly when VIX<13 (the premium-sellers sleep)? Causal
(positions from prior-close info). NIFTY daily 2015-2026. Net-of-cost note below."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
C = n.close; r = C.pct_change()
vxp = vix.reindex(n.index, method="ffill").shift(1)          # prior-close VIX (causal)
Cp = C.shift(1); ma50 = C.rolling(50).mean().shift(1); ma200 = C.rolling(200).mean().shift(1); mom20 = C.pct_change(20).shift(1)
lowvix = vxp < 13
systems = {
    "BuyHold NIFTY (always long, ref)": pd.Series(1.0, index=n.index),
    "Long every low-VIX day (VIX<13)": lowvix.astype(float),
    "Low-VIX + price>50DMA (trend)": (lowvix & (Cp > ma50)).astype(float),
    "Low-VIX + price>200DMA (trend)": (lowvix & (Cp > ma200)).astype(float),
    "Low-VIX + 20d momentum>0": (lowvix & (mom20 > 0)).astype(float),
    "Low-VIX + >50DMA + mom>0": (lowvix & (Cp > ma50) & (mom20 > 0)).astype(float),
}
nyears = (n.index[-1]-n.index[0]).days/365.25
print(f"NIFTY daily {n.index[0].date()}→{n.index[-1].date()} ({nyears:.1f}y). low-VIX(<13) days: {int(lowvix.sum())} ({lowvix.mean()*100:.0f}%)\n")
print(f"{'system':34s} {'days':>5} {'%yr':>4} {'totRet':>8} {'annRet':>7} {'Sharpe':>7} {'maxDD':>7} {'winDay%':>8}")
for name, pos in systems.items():
    pos = pos.fillna(0)
    pnl = (pos * r).fillna(0)
    eq = (1+pnl).cumprod()
    days = int((pos > 0).sum()); dd = (eq/eq.cummax()-1).min()
    ann = pnl.mean()*252; shp = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() > 0 else 0
    winr = (pnl[pos > 0] > 0).mean()*100 if days else 0
    print(f"{name:34s} {days:5d} {days/nyears/252*100:3.0f}% {(eq.iloc[-1]-1)*100:7.0f}% {ann*100:6.1f}% {shp:6.2f} {dd*100:6.0f}% {winr:7.0f}%")
print("""
annRet = strategy annualised (flat = in cash/debt on non-deployed days, earning 0 here).
Compare to ~6-7% debt. If a low-VIX trend long clears debt with a sane Sharpe/DD, it is the productive
fill for the idle (VIX<13) windows. Cost: NIFTY-future/ETF long, ~cheap; daily-rebalance turnover low.""")
