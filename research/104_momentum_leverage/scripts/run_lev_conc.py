"""Research 104 — Leverage x Concentration on the Nifty-250 momentum book (research/75 engine).
Leverage applied ONLY while the index gate is risk-on; borrowed cash accrues financing; a maintenance
margin-call force-liquidates on ruin. Sweep N x lev x score. Net-of-cost, daily-marked, 2006-2026."""
import importlib.util, time, csv, sys
from pathlib import Path
import numpy as np, pandas as pd
M = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
_s = importlib.util.spec_from_file_location("mom", str(M)); mom = importlib.util.module_from_spec(_s); _s.loader.exec_module(mom)
rs2, BENCH, EXCLUDE, RT, DAY_CASH = mom.rs2, mom.BENCH, mom.EXCLUDE, mom.RT, mom.DAY_CASH
START, MODERN, momentum = mom.START, mom.MODERN, mom.momentum
HERE = Path("/home/arun/quantifyd/research/104_momentum_leverage"); RES = HERE/"results"; RES.mkdir(parents=True, exist_ok=True)
MAINT = 0.25            # own-equity / gross-notional maintenance floor -> margin call
BORROW = 0.08          # financing on borrowed leg, p.a.

def run_lev(close, tv, ema, nbx_ema100, score, N, lev, borrow=BORROW, stack=False, gate=True, rt=RT, start=START):
    day_borrow = (1+borrow)**(1/252)
    idx = close.index[close.index >= start]
    _ss = pd.Series(idx, index=idx); me = set(_ss.groupby([idx.year, idx.month]).max().values)
    nbx = close[BENCH].ffill(); e50, e100, e200 = ema[50], ema[100], ema[200]
    cash, held, nav, prev = 1.0, {}, [], None
    st = dict(cost=0.0, fills=0, turn_sum=0.0, gate_off=0, mcalls=0)
    def E(): return sum(v[0] for v in held.values())
    def liquidate(d):
        nonlocal cash, held
        for s in list(held): cash += held.pop(s)[0]
    def gate_on(d):
        return (not gate) or (nbx.loc[d] > nbx_ema100.loc[d] if pd.notna(nbx_ema100.loc[d]) else True)
    def do_fill(d, px):
        nonlocal cash, held
        sc = momentum(close, d, score)
        if sc is None: return
        uni = rs2.pit_universe(tv, close, d, "n250")
        cand = [s for s in sc.index if s in uni and s not in EXCLUDE and s != BENCH]
        if stack:
            cand = [s for s in cand if pd.notna(e50[s].loc[d]) and pd.notna(e200[s].loc[d]) and e50[s].loc[d] > e100[s].loc[d] > e200[s].loc[d]]
        cand = [s for s in cand if pd.notna(px.get(s, np.nan))]
        if not cand: liquidate(d); return
        top = list(sc[cand].sort_values(ascending=False).index[:N])
        tot = E() + cash
        if tot <= 0: liquidate(d); return           # ruined
        notional = lev * tot; w = notional/len(top)
        turn = len(set(held).symmetric_difference(set(top)))/max(1, 2*len(top))
        nh = {}
        for s in top:
            nh[s] = held[s] if s in held else [w, w, d]; nh[s][0] = w
        held = nh
        c = notional*rt*turn*2; cash = tot - notional - c
        st['cost'] += c; st['fills'] += 1; st['turn_sum'] += turn
    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0: stt[0] += stt[0]*(p1/p0-1.0)
        cash *= DAY_CASH if cash > 0 else day_borrow
        # maintenance margin call (only meaningful when levered / cash negative)
        if held:
            gross = E(); eq = gross + cash
            if gross > 0 and eq/gross < MAINT:
                liquidate(d); st['mcalls'] += 1
        if gate and d in me and not gate_on(d) and held:
            liquidate(d); st['gate_off'] += 1
        if d in me:
            if gate_on(d): do_fill(d, px)
            elif held: liquidate(d)
        nav.append((d, E()+cash)); prev = d
    return mom._stats(nav, st, start)

FIELDS = ["config","score","N","lev","borrow_pct","cagr","maxdd","sharpe","sortino","calmar","mult","fills","gate_off","mcalls","mod_cagr","mod_maxdd","mod_calmar"]
def row_of(lbl, score, N, lev, borrow, r):
    m = r['modern']
    return dict(config=lbl, score=score, N=N, lev=lev, borrow_pct=round(borrow*100,1),
        cagr=round(r['cagr'],1), maxdd=round(r['dd'],1), sharpe=round(r['sharpe'],2),
        sortino=round(r['sortino'],2) if pd.notna(r['sortino']) else "",
        calmar=round(r['calmar'],2) if pd.notna(r['calmar']) else "", mult=round(r['mult'],1),
        fills=r['st']['fills'], gate_off=r['st']['gate_off'], mcalls=r['st']['mcalls'],
        mod_cagr=round(m['cagr'],1) if m else "", mod_maxdd=round(m['dd'],1) if m else "",
        mod_calmar=round(m['calmar'],2) if m else "")

def main():
    t0 = time.time(); print("Loading price+tv ...", flush=True)
    close, tv = mom.load()
    print(f"  {close.shape[1]} symbols {close.index.min().date()}..{close.index.max().date()}", flush=True)
    ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean() for p in (50,100,200)}
    nbx_ema100 = close[BENCH].ffill().ewm(span=100, adjust=False, min_periods=100).mean()
    bn = close[BENCH].loc[close.index >= START].dropna(); bn = bn/bn.iloc[0]
    bstat = mom._stats(list(zip(bn.index, bn.values)), dict(fills=0, turn_sum=0.0, gate_off=0, mcalls=0), START)
    print(f"  [ref] NIFTYBEES B&H CAGR={bstat['cagr']:.1f}% DD={bstat['dd']:.1f}% Cal={bstat['calmar']:.2f} {bstat['mult']:.1f}x", flush=True)
    out = RES/"lev_conc.csv"; done = set()
    if out.exists():
        with open(out) as f: done = {r['config'] for r in csv.DictReader(f)}
    else:
        with open(out,"w",newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    grid = []
    for score in ["radj_z","ret252"]:
        for N in [5,8,12]:
            for lev in [1.0,1.3,1.6,2.0]:
                if score == "ret252" and N != 8: continue   # ret252 control only at N8
                grid.append((f"{score}_N{N}_L{lev}", score, N, lev))
    for lbl, score, N, lev in grid:
        if lbl in done: print(f"  skip {lbl}", flush=True); continue
        ts = time.time()
        r = run_lev(close, tv, ema, nbx_ema100, score, N, lev)
        row = row_of(lbl, score, N, lev, BORROW, r)
        with open(out,"a",newline="") as f: csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"  {lbl:18} CAGR={row['cagr']:5.1f}% DD={row['maxdd']:6.1f}% Sh={row['sharpe']:.2f} Cal={row['calmar']} {row['mult']}x mcalls={row['mcalls']} [{time.time()-ts:.0f}s]", flush=True)
        sys.stdout.flush()
    print(f"\nDone {time.time()-t0:.0f}s -> {out}", flush=True)

if __name__ == "__main__": main()
