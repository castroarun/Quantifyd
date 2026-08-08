"""Research 106 — does widening the universe 200 -> 500 help, and does it survive a liquidity screen?
Universe = top-N by trailing-6m median traded value (survivorship-free PIT proxy), with an explicit
min-ADV screen + cost sensitivity. Book rules unchanged (top-8, buffer22, Donchian15, weekly gate cash
exit, monthly rotate-only). 2011-2026, net of cost, STCG tracked."""
import importlib.util, csv, sys, time
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location("m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH = m75.BENCH; EXCLUDE = m75.EXCLUDE
RES = Path("/home/arun/quantifyd/research/106_nifty500_universe/results"); RES.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2011-01-01"); CASH_Y, STCG = 0.065, 0.20
print("loading ...", flush=True)
close, tv = m75.load()
idx = close.index[(close.index >= START) & (close.index <= pd.Timestamp("2026-08-01"))]
ME = sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values))
_iso = idx.isocalendar(); WK = set(pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1); NBX = close[BENCH].ffill()
ME = [pd.Timestamp(x) for x in ME]

print("precomputing per-month scores + ADV (once) ...", flush=True)
SNAP = {}
for d in ME:
    h = close.loc[:d].ffill()
    if len(h) <= 253: continue
    adv = tv.loc[:d].tail(126).median()                       # avg daily traded value, Rs
    sc = None
    for L, w in ((126, 0.5), (252, 0.5)):
        p0, p1 = h.iloc[-L-1], h.iloc[-1]
        nf = p1[BENCH]/p0[BENCH]
        r = (p1/p0)/nf * w
        sc = r if sc is None else sc.add(r, fill_value=np.nan)
    ok = [s for s in sc.index if s not in EXCLUDE and s != BENCH and pd.notna(sc[s]) and pd.notna(adv.get(s, np.nan))]
    SNAP[d] = (sc[ok], adv[ok])
print(f"  {len(SNAP)} month-ends", flush=True)

def basket(d, band, min_adv):
    sc, adv = SNAP[d]
    pool = adv.sort_values(ascending=False).index[:band]        # top-N by traded value
    if min_adv > 0:
        pool = [s for s in pool if adv[s] >= min_adv]
    sub = sc[[s for s in pool if s in sc.index]]
    return list(sub.sort_values(ascending=False).index[:30])

def run(band, min_adv, rt_leg):
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    nav_pre, nav_post = [], []
    def E(): return sum(v[0] for v in held.values())
    def sell(s, d):
        nonlocal cash, tax
        v, b, bd = held.pop(s)
        g = v - b
        if g > 0 and (d - bd).days < 365: tax += g * STCG
        cash += v * (1 - rt_leg)
    for d in idx:
        if held and prev is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0: v[0] += v[0]*(p1/p0-1.0)
        if cash > 0: cash *= (1+CASH_Y)**(1/252)
        if held:
            lows = ROLL_LOW.loc[d]
            for s in list(held):
                p1 = close.at[d, s] if s in close.columns else np.nan
                ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln: sell(s, d)
        if d in WK:
            b = NBX.loc[:d].dropna()
            if len(b) >= 100 and b.iloc[-1] < b.tail(100).mean():
                for s in list(held): sell(s, d)
                derisked = True
            else: derisked = False
        if d in SNAP and not derisked:
            etf = basket(d, band, min_adv)
            if etf:
                top, buf = etf[:8], set(etf[:22])
                for s in list(held):
                    if s not in buf: sell(s, d)
                cand = (list(held) + [x for x in top if x not in held])[:8]
                tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
                per = (E()+cash)/8
                for s in tgt:
                    if s not in held and cash > 0:
                        buy = min(per, cash)
                        if buy > 0: held[s] = [buy*(1-rt_leg), buy, d]; cash -= buy
        nav_pre.append((d, E()+cash)); nav_post.append((d, E()+cash-tax)); prev = d
    def blk(p):
        n = pd.DataFrame(p, columns=["d","v"]).set_index("d")["v"]
        yrs = (n.index[-1]-n.index[0]).days/365.25
        cagr = (n.iloc[-1]/n.iloc[0])**(1/yrs)-1
        dr = n.pct_change().dropna(); sh = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
        dd = ((n-n.cummax())/n.cummax()).min()
        return cagr*100, dd*100, sh, (cagr/abs(dd) if dd<0 else 0)
    c1,d1,s1,k1 = blk(nav_pre); c2,d2,s2,k2 = blk(nav_post)
    return dict(cagr=round(c1,1), maxdd=round(d1,1), sharpe=round(s1,2), calmar=round(k1,2),
                net_cagr=round(c2,1), net_calmar=round(k2,2))
CR = 10_000_000
grid = []
for band in (200, 500):
    for adv, atag in ((0,"none"), (5*CR,"5cr"), (10*CR,"10cr"), (25*CR,"25cr")):
        grid.append((f"N{band}_adv{atag}_cost0.15", band, adv, 0.0015))
for cost, ctag in ((0.003,"0.30"), (0.005,"0.50")):     # cost sensitivity on the two headline configs
    grid.append((f"N200_adv10cr_cost{ctag}", 200, 10*CR, cost))
    grid.append((f"N500_adv10cr_cost{ctag}", 500, 10*CR, cost))
F=["config","band","min_adv_cr","cost_leg","cagr","net_cagr","maxdd","sharpe","calmar","net_calmar"]
w=csv.DictWriter(open(RES/"n500_sweep.csv","w",newline=""),fieldnames=F); w.writeheader()
print(f"\n{'config':>26} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} {'Sharpe':>6} {'netCal':>7}", flush=True)
for lbl, band, adv, cost in grid:
    ts=time.time(); r = run(band, adv, cost)
    w.writerow(dict(config=lbl, band=band, min_adv_cr=adv/CR, cost_leg=cost, **r))
    print(f"{lbl:>26} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% {r['sharpe']:>6.2f} {r['net_calmar']:>7.2f}  [{time.time()-ts:.0f}s]", flush=True)
    sys.stdout.flush()
print("done", flush=True)
