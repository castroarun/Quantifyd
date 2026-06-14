"""research/65 G3 — refine the low-VIX long: trend-confirmation entry + a trailing stop, to cut the
regime-ending-spike drag (2019/2026) while keeping the sustained-grind upside (2023/2025). State machine,
daily, causal. Cached NIFTY+VIX."""
import numpy as np, pandas as pd
n, vix = pd.read_pickle("/tmp/nifty_vix_cache.pkl")
C = n.close.values; r = n.close.pct_change().fillna(0).values
vxp = vix.reindex(n.index, method="ffill").shift(1).values
Cp = n.close.shift(1).values
ma20 = n.close.rolling(20).mean().shift(1).values
ma50 = n.close.rolling(50).mean().shift(1).values
yr = np.array(n.index.year); N = len(C)

def sim(trend=None, trail=None):
    pos = np.zeros(N); inpos = False; peak = 0.0
    for i in range(1, N):
        if np.isnan(vxp[i]): continue
        if not inpos:
            cond = vxp[i] < 13
            if trend == 20: cond = cond and (Cp[i] > ma20[i])
            if trend == 50: cond = cond and (Cp[i] > ma50[i])
            if cond:
                inpos = True; peak = Cp[i]
        if inpos:
            peak = max(peak, Cp[i])
            exit_vix = not (vxp[i] < 13)
            exit_stop = trail is not None and Cp[i] < peak*(1-trail)
            if exit_vix or exit_stop:
                inpos = False
        pos[i] = 1.0 if inpos else 0.0
    return pos, pos*r

nyears = (n.index[-1]-n.index[0]).days/365.25
variants = [("naive (VIX<13)", None, None), ("+20DMA trend", 20, None), ("+2% trail stop", None, 0.02),
            ("+3% trail stop", None, 0.03), ("+20DMA +2% trail", 20, 0.02), ("+50DMA +3% trail", 50, 0.03)]
print(f"{'variant':20s} {'days':>5} {'tot%':>6} {'Sharpe':>7} {'maxDD':>7}  | per-year ret% (19→26)")
peryear_all = {}
for name, tr, st in variants:
    pos, pnl = sim(tr, st); eq = np.cumprod(1+pnl)
    days = int((pos > 0).sum()); dd = (eq/np.maximum.accumulate(eq)-1).min()
    shp = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() > 0 else 0
    py = {}
    for y in range(2019, 2027):
        mask = yr == y; py[y] = (np.prod(1+pnl[mask])-1)*100
    peryear_all[name] = py
    pys = " ".join(f"{py[y]:+5.1f}" for y in range(2019, 2027))
    print(f"{name:20s} {days:5d} {(eq[-1]-1)*100:5.0f}% {shp:6.2f} {dd*100:6.0f}%  | {pys}")
print("\nyears: 2019 2020 2021 2022 2023 2024 2025 2026  (target: keep 2023/2025 big, cut 2019/2026 red)")
