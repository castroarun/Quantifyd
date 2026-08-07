"""User capital model: futures-margin leverage (no MTF cash loan). Financing = 7% basis on (L-1)*capital;
collateral yield = 2.7% on capital always (45% liquid @6% + 5% cash + 50% stocks-excluded). radj_z N8, gated."""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
M = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
_s = importlib.util.spec_from_file_location("mom", str(M)); mom = importlib.util.module_from_spec(_s); _s.loader.exec_module(mom)
rs2, BENCH, EXCLUDE, RT, momentum = mom.rs2, mom.BENCH, mom.EXCLUDE, mom.RT, mom.momentum
START, MODERN = mom.START, mom.MODERN
FIN, COLL = 0.07, 0.45*0.06        # 7% futures basis on (L-1); 2.7% collateral yield on capital always
def run_uc(close, tv, nbx_ema100, N, lev, score="radj_z"):
    idx = close.index[close.index >= START]
    _ss = pd.Series(idx, index=idx); me = set(_ss.groupby([idx.year, idx.month]).max().values)
    nbx = close[BENCH].ffill()
    coll_d = COLL/252; fin_d = (lev-1)*FIN/252
    nav = 1.0; navser = []; held = []; ppx = {}; prev = None; deployed = False; st = dict(fills=0, gate_off=0, dep=0)
    def gate_on(d): return nbx.loc[d] > nbx_ema100.loc[d] if pd.notna(nbx_ema100.loc[d]) else True
    for d in idx:
        r_book = 0.0
        if deployed and held and prev is not None:
            rs = []
            for s in held:
                p1 = close.at[d, s] if s in close.columns else np.nan; p0 = ppx.get(s)
                if pd.notna(p1) and p0 and p0 > 0: rs.append(p1/p0-1.0)
                if pd.notna(p1): ppx[s] = p1
            if rs: r_book = float(np.mean(rs))
        day_ret = coll_d + (lev*r_book - fin_d if deployed else 0.0)
        nav *= (1 + day_ret)
        if d in me:
            if gate_on(d):
                sc = momentum(close, d, score)
                if sc is not None:
                    uni = rs2.pit_universe(tv, close, d, "n250")
                    cand = [x for x in sc.index if x in uni and x not in EXCLUDE and x != BENCH and pd.notna(close.at[d, x] if x in close.columns else np.nan)]
                    if cand:
                        top = list(sc[cand].sort_values(ascending=False).index[:N])
                        turn = len(set(held).symmetric_difference(set(top)))/max(1, 2*len(top))
                        nav *= (1 - lev*RT*turn*2)
                        held = top; ppx = {s: close.at[d, s] for s in top if s in close.columns and pd.notna(close.at[d, s])}
                        deployed = True; st['fills'] += 1; st['dep'] += 1
            else:
                if held: st['gate_off'] += 1
                deployed = False; held = []
        navser.append((d, nav)); prev = d
    return mom._stats(navser, dict(fills=st['fills'], turn_sum=0.0, gate_off=st['gate_off'], mcalls=0), START)
def main():
    close, tv = mom.load()
    nbx100 = close[BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()
    bn = close[BENCH].loc[close.index >= START].dropna(); bn = bn/bn.iloc[0]
    bstat = mom._stats(list(zip(bn.index, bn.values)), dict(fills=0, turn_sum=0.0, gate_off=0, mcalls=0), START)
    print(f"NIFTYBEES B&H: CAGR {bstat['cagr']:.1f}%  DD {bstat['dd']:.1f}%  Calmar {bstat['calmar']:.2f}\n")
    print(f"USER CAPITAL MODEL — futures margin, 7% basis on (L-1), 2.7% collateral yield always, radj_z N8:")
    print(f"{'leverage':>10} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'growth':>9}")
    for lev in [1.0, 1.3, 1.6, 2.0]:
        r = run_uc(close, tv, nbx100, 8, lev)
        m = r['modern']
        print(f"{lev:>9}x {r['cagr']:>6.1f}% {r['dd']:>7.1f}% {r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['mult']:>8.1f}x  | modern(2014+) {m['cagr']:.1f}%/{m['dd']:.1f}%" if m else "")
if __name__ == "__main__": main()
