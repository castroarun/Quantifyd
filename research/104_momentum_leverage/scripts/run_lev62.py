"""Leverage frontier on the EXISTING live book's exact rules (rsblend + top-22 buffer + 15d Donchian +
weekly NIFTYBEES-100 gate), full cycle 2006-2026 (through 2008), MTF financing 10.5%, margin-call modelled.
Reuses research/62 scoring helpers; loads 2004+ via research/75 load so 2006 start has warmup."""
import importlib.util, csv, sys
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
P62 = Path("/home/arun/quantifyd/research/62_momentum_etf_subselect/scripts/62_mom30_subselect.py")
P75 = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
_a = importlib.util.spec_from_file_location("r62", str(P62)); r62 = importlib.util.module_from_spec(_a); _a.loader.exec_module(r62)
_b = importlib.util.spec_from_file_location("mom75", str(P75)); mom75 = importlib.util.module_from_spec(_b); _b.loader.exec_module(mom75)
rs2, BENCH = r62.rs2, r62.BENCH
START = pd.Timestamp("2006-01-01"); MAINT = 0.25
def run_lev62(close, tv, score_fn, N, buffer, gate, donchian, etf_size, lev, borrow=0.105, rt=0.003):
    day_cash = (1.065)**(1/252); day_borrow = (1+borrow)**(1/252)
    idx = close.index[close.index >= START]
    me = set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
    iso = idx.isocalendar(); wk = set(pd.Series(idx, index=idx).groupby([iso.year.values, iso.week.values]).last().values)
    nbx = close[BENCH].ffill()
    roll_low = (close.rolling(donchian, min_periods=donchian).min().shift(1) if donchian else None)
    cash, held, nav, prev, derisked = 1.0, {}, [], None, False
    st = dict(cost=0.0, fills=0, donchian_exits=0, gate_derisk=0, mcalls=0, turn_sum=0.0)
    def E(): return sum(v[0] for v in held.values())
    def liq():
        nonlocal cash, held
        for s in list(held): cash += held.pop(s)[0]
    def do_fill(d, px):
        nonlocal cash, held, derisked
        etf = r62.eligible_etf(close, tv, d, score_fn, etf_size)
        if not etf: return
        top = etf[:N]; buf = set(etf[:buffer])
        for s in list(held):
            if s not in buf: cash += held.pop(s)[0]
        keep = [s for s in held if s in buf]; add = [s for s in top if s not in keep]
        target = [s for s in (keep + add)[:N] if pd.notna(px.get(s, np.nan))]
        if not target: return
        tot = E() + cash
        if tot <= 0: liq(); return
        notional = lev*tot; w = notional/len(target)
        turn = len(set(held).symmetric_difference(set(target)))/max(1, 2*N)
        nh = {}
        for s in target:
            if s in held: stt = held[s]; stt[0] = w; nh[s] = stt
            else: nh[s] = [w, w, d, px[s]]
        held = nh; c = notional*rt*turn*2; cash = tot - notional - c
        st['cost'] += c; st['fills'] += 1; st['turn_sum'] += turn; derisked = False
    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan); p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0: stt[0] += stt[0]*(p1/p0-1.0); stt[3] = max(stt[3], p1)
        cash *= day_cash if cash > 0 else day_borrow
        if donchian and held:
            lows = roll_low.loc[d]
            for s in list(held):
                p1 = px.get(s, np.nan); ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln: cash += held.pop(s)[0]; st['donchian_exits'] += 1
        if held:
            gross = E(); eq = gross + cash
            if gross > 0 and eq/gross < MAINT: liq(); st['mcalls'] += 1
        if gate and d in wk:
            b = nbx.loc[:d].dropna()
            roff = len(b) >= gate and b.iloc[-1] < b.tail(gate).mean()
            if roff:
                if held: liq(); st['gate_derisk'] += 1
                derisked = True
            else: derisked = False
        if d in me and not (gate and derisked): do_fill(d, px)
        nav.append((d, E()+cash)); prev = d
    return r62._stats(nav, st)
def main():
    close, tv = mom75.load()
    print(f"data {close.index.min().date()}..{close.index.max().date()}  frontier from {START.date()}, MTF 10.5%\n", flush=True)
    bn = close[BENCH].loc[close.index >= START].dropna(); bn = bn/bn.iloc[0]
    bm = r62.stats_from_nav(bn)
    print(f"NIFTYBEES B&H: CAGR {bm['cagr']:.1f}%  DD {bm['dd']:.1f}%  Calmar {bm['calmar']:.2f}\n")
    print(f"EXISTING BOOK (rsblend + buffer22 + Donchian15 + weekly gate), leverage frontier, 2006-2026:")
    print(f"{'leverage':>9} {'CAGR':>7} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7} {'donchX':>7} {'mcalls':>7}", flush=True)
    out = "/home/arun/quantifyd/research/104_momentum_leverage/results/existing_leverage.csv"
    with open(out, "w", newline="") as f: csv.writer(f).writerow(["lev","cagr","maxdd","sharpe","calmar","donch_exits","mcalls"])
    for lev in [1.0, 1.3, 1.6, 2.0]:
        g = run_lev62(close, tv, "rsblend", 8, 22, 100, 15, 30, lev)
        cal = g['calmar'] if pd.notna(g['calmar']) else 0
        print(f"{lev:>8}x {g['cagr']:>6.1f}% {g['dd']:>7.1f}% {g['sharpe']:>7.2f} {cal:>7.2f} {g['st']['donchian_exits']:>7} {g['st']['mcalls']:>7}", flush=True)
        with open(out, "a", newline="") as f: csv.writer(f).writerow([lev, round(g['cagr'],1), round(g['dd'],1), round(g['sharpe'],2), round(cal,2), g['st']['donchian_exits'], g['st']['mcalls']])
        sys.stdout.flush()
    print("\nwrote existing_leverage.csv", flush=True)
if __name__ == "__main__": main()
