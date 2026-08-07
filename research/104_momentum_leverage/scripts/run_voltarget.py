"""Research 104b — VOL-TARGETED leverage on the gated Nifty-250 momentum book. Leverage is set from the
book's own trailing realized vol (target constant vol), re-scaled WEEKLY (auto-deleverage in turbulence),
names rebalanced monthly. Should lift the levered Calmar vs static leverage. radj_z N8, net, daily, 2006-26."""
import importlib.util, time, csv, sys
from pathlib import Path
import numpy as np, pandas as pd
M = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
_s = importlib.util.spec_from_file_location("mom", str(M)); mom = importlib.util.module_from_spec(_s); _s.loader.exec_module(mom)
rs2, BENCH, EXCLUDE, RT, DAY_CASH = mom.rs2, mom.BENCH, mom.EXCLUDE, mom.RT, mom.DAY_CASH
START, momentum = mom.START, mom.momentum
RES = Path("/home/arun/quantifyd/research/104_momentum_leverage/results"); RES.mkdir(parents=True, exist_ok=True)
MAINT, BORROW = 0.25, 0.08

def run_vt(close, tv, ema, nbx_ema100, score, N, target_vol, lookback, max_lev, min_lev=0.5, borrow=BORROW, rt=RT, start=START):
    day_borrow = (1+borrow)**(1/252)
    idx = close.index[close.index >= start]
    _ss = pd.Series(idx, index=idx); me = set(_ss.groupby([idx.year, idx.month]).max().values)
    iso = idx.isocalendar(); wk_last = set(pd.Series(idx, index=idx).groupby([iso.year.values, iso.week.values]).last().values)
    nbx = close[BENCH].ffill()
    cash, held, nav, prev = 1.0, {}, [], None
    st = dict(cost=0.0, fills=0, turn_sum=0.0, gate_off=0, mcalls=0, lev_sum=0.0, lev_n=0)
    rets = []
    def E(): return sum(v[0] for v in held.values())
    def cur_lev():
        if len(rets) < lookback: return max(min_lev, min(max_lev, 1.0))
        v = np.std(rets[-lookback:], ddof=0)*np.sqrt(252)
        if v <= 0: return max_lev
        return max(min_lev, min(max_lev, target_vol/v))
    def liquidate(d):
        nonlocal cash, held
        for s in list(held): cash += held.pop(s)[0]
    def gate_on(d):
        return nbx.loc[d] > nbx_ema100.loc[d] if pd.notna(nbx_ema100.loc[d]) else True
    def do_fill(d, px, lev):
        nonlocal cash, held
        sc = momentum(close, d, score)
        if sc is None: return
        uni = rs2.pit_universe(tv, close, d, "n250")
        cand = [s for s in sc.index if s in uni and s not in EXCLUDE and s != BENCH]
        cand = [s for s in cand if pd.notna(px.get(s, np.nan))]
        if not cand: liquidate(d); return
        top = list(sc[cand].sort_values(ascending=False).index[:N])
        tot = E() + cash
        if tot <= 0: liquidate(d); return
        notional = lev*tot; w = notional/len(top)
        turn = len(set(held).symmetric_difference(set(top)))/max(1, 2*len(top))
        nh = {}
        for s in top:
            nh[s] = held[s] if s in held else [w, w, d]; nh[s][0] = w
        held = nh
        c = notional*rt*turn*2; cash = tot - notional - c
        st['cost'] += c; st['fills'] += 1; st['turn_sum'] += turn
    def rescale(lev):                                   # adjust leverage, keep names
        nonlocal cash, held
        gross = E()
        if gross <= 0: return
        tot = gross + cash
        if tot <= 0: return
        desired = lev*tot; f = desired/gross
        for s in held: held[s][0] *= f
        c = abs(desired-gross)*rt; cash = tot - desired - c; st['cost'] += c
    for d in idx:
        px = close.loc[d]; g0 = E()
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0: stt[0] += stt[0]*(p1/p0-1.0)
            if g0 > 0: rets.append(E()/g0 - 1.0)
        cash *= DAY_CASH if cash > 0 else day_borrow
        if held:
            gross = E(); eq = gross + cash
            if gross > 0 and eq/gross < MAINT: liquidate(d); st['mcalls'] += 1
        if d in me and not gate_on(d) and held: liquidate(d); st['gate_off'] += 1
        if d in wk_last and held and gate_on(d): rescale(cur_lev())     # weekly vol re-lever
        if d in me:
            if gate_on(d):
                lev = cur_lev(); st['lev_sum'] += lev; st['lev_n'] += 1; do_fill(d, px, lev)
            elif held: liquidate(d)
        nav.append((d, E()+cash)); prev = d
    r = mom._stats(nav, st, start); r['avg_lev'] = st['lev_sum']/max(1, st['lev_n']); return r

FIELDS = ["config","target_vol","lookback","max_lev","avg_lev","cagr","maxdd","sharpe","calmar","mult","mcalls","mod_cagr","mod_maxdd","mod_calmar"]
def main():
    t0 = time.time(); print("Loading ...", flush=True)
    close, tv = mom.load()
    ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50,100,200)}
    nbx_ema100 = close[BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()
    bn = close[BENCH].loc[close.index >= START].dropna(); bn = bn/bn.iloc[0]
    bstat = mom._stats(list(zip(bn.index, bn.values)), dict(fills=0, turn_sum=0.0, gate_off=0, mcalls=0), START)
    print(f"  [ref] B&H {bstat['cagr']:.1f}%/{bstat['dd']:.1f}%/Cal{bstat['calmar']:.2f} | static radj_z N8: L1.0 33.8%/-30%/1.13  L1.6 48.7%/-48%/1.02  L2.0 57.8%/-60%/0.97", flush=True)
    out = RES/"voltarget.csv"
    with open(out,"w",newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    print(f"{'config':>22} {'avgLev':>6} {'CAGR':>6} {'MaxDD':>7} {'Sh':>5} {'Cal':>5} {'mcalls':>6}", flush=True)
    for tvv in [0.20,0.25,0.30,0.35]:
        for lb in [20,60]:
            for mx in [2.0,2.5]:
                lbl = f"vt{int(tvv*100)}_lb{lb}_mx{mx}"
                r = run_vt(close, tv, ema, nbx_ema100, "radj_z", 8, tvv, lb, mx)
                m = r['modern']
                row = dict(config=lbl, target_vol=tvv, lookback=lb, max_lev=mx, avg_lev=round(r['avg_lev'],2),
                    cagr=round(r['cagr'],1), maxdd=round(r['dd'],1), sharpe=round(r['sharpe'],2),
                    calmar=round(r['calmar'],2) if pd.notna(r['calmar']) else "", mult=round(r['mult'],1), mcalls=r['st']['mcalls'],
                    mod_cagr=round(m['cagr'],1) if m else "", mod_maxdd=round(m['dd'],1) if m else "", mod_calmar=round(m['calmar'],2) if m else "")
                with open(out,"a",newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                print(f"{lbl:>22} {row['avg_lev']:>6} {row['cagr']:>5.1f}% {row['maxdd']:>6.1f}% {row['sharpe']:>5.2f} {str(row['calmar']):>5} {row['mcalls']:>6}", flush=True)
                sys.stdout.flush()
    print(f"\nDone {time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__": main()
