"""Research 62 — PHASE 2e: does a diversified midcap sleeve DE-CORRELATE the top200 book?

Standalone the diversified midcap-momentum sleeve is inferior (net Cal ~0.44 < top200). But a
worse sleeve can still IMPROVE a portfolio if it's de-correlated (lowers combined drawdown). Test:
run top200 winner (NIFTYBEES gate, N8) and the diversified midcap sleeve (fair gate, N30) SEPARATELY,
overlay their NAVs at weights 100/0…50/50 (continuously-rebalanced), report correlation + blended
Calmar/DD. Do it GROSS (pure diversification benefit) and NET @₹10cr (does it survive capacity?).

Reuses 62g (run, synth_index).
"""
from __future__ import annotations
import importlib.util, time
from pathlib import Path
import numpy as np
import pandas as pd
import logging
logging.disable(logging.WARNING)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_g = importlib.util.spec_from_file_location(
    "m62g", str(HERE / "scripts" / "62g_fairgate_diversified.py"))
g = importlib.util.module_from_spec(_g); _g.loader.exec_module(g)
rs2 = g.rs2
BENCH = rs2.NIFTY_SYM


def metrics(n):
    n = n.dropna()
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1 if n.iloc[-1] > 0 else -1
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd = ((n - n.cummax()) / n.cummax()).min()
    cal = cagr / abs(dd) if dd < 0 else np.nan
    return cagr * 100, dd * 100, sh, cal


def blend(navA, navB, w):
    """Continuously-rebalanced blend: daily return = w*rA + (1-w)*rB; cumulate."""
    idx = navA.index.intersection(navB.index)
    rA = navA.reindex(idx).pct_change().fillna(0)
    rB = navB.reindex(idx).pct_change().fillna(0)
    rr = w * rA + (1 - w) * rB
    return (1 + rr).cumprod()


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    synth_mid = g.synth_index(close, tv, 100, 250)
    print(f"  synth_mid final {synth_mid.iloc[-1]:.1f}x", flush=True)

    for alab, aum in [("GROSS", None), ("NET@10cr", 1e8)]:
        print(f"\n========== {alab} ==========", flush=True)
        A = g.run(close, tv, "top200", close[BENCH], N=8, buffer=22, aum=aum)['nav']
        B = g.run(close, tv, "mid", synth_mid, N=30, buffer=45, aum=aum)['nav']
        idx = A.index.intersection(B.index)
        rA = A.reindex(idx).pct_change().dropna()
        rB = B.reindex(idx).pct_change().dropna()
        corr = rA.corr(rB.reindex(rA.index))
        cA = metrics(A); cB = metrics(B)
        print(f"  top200      : CAGR {cA[0]:5.1f}%  DD {cA[1]:6.1f}%  Sharpe {cA[2]:.2f}  Calmar {cA[3]:.2f}",
              flush=True)
        print(f"  div-midcap  : CAGR {cB[0]:5.1f}%  DD {cB[1]:6.1f}%  Sharpe {cB[2]:.2f}  Calmar {cB[3]:.2f}",
              flush=True)
        print(f"  return correlation = {corr:.2f}", flush=True)
        print(f"  {'weight(top200/mid)':20} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7}", flush=True)
        rows = []
        for w in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            bn = blend(A, B, w)
            c = metrics(bn)
            tag = "  <- pure top200" if w == 1.0 else ""
            print(f"  {int(w*100):3}/{int((1-w)*100):<3}              "
                  f"{c[0]:7.1f} {c[1]:8.1f} {c[2]:7.2f} {c[3]:7.2f}{tag}", flush=True)
            rows.append(dict(scope=alab, w_top200=w, cagr=round(c[0], 1),
                             maxdd=round(c[1], 1), sharpe=round(c[2], 2),
                             calmar=round(c[3], 2), corr=round(corr, 2)))
        pd.DataFrame(rows).to_csv(RES / f"p2e_blend_{alab.replace('@','_')}.csv", index=False)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
