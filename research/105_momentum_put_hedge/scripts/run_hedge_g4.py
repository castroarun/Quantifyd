"""G4 — FIXES the over-hedge flaw (dynamic re-sizing as stocks stop out) + a proper trailing-exit sweep.

Flaw found: hedge units were fixed at entry and only re-sized at the next roll. If stocks stop out on their
Donchian trails, equity shrinks while the put notional stays — turning a 2x hedge into a 3-4x naked short.
Weekly rolls masked this (re-sized every ~7d); monthly let it run for weeks — so the earlier "weekly beats
monthly" result may have been a RE-SIZING effect, not a tenor effect. This re-tests both with a daily
re-size, and sweeps SuperTrend (6 configs), premium-trailing and EMA-cross exits.
"""
import importlib.util, csv, sys
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location(
    "hs", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_sweep.py")
hs = importlib.util.module_from_spec(spec); spec.loader.exec_module(hs)
close, idx, ME, WK = hs.close, hs.idx, hs.ME, hs.WK
ROLL_LOW, NBX, ETF, S, SH, BENCH = hs.ROLL_LOW, hs.NBX, hs.ETF, hs.S, hs.SH, hs.BENCH
RT_LEG, CASH_Y, STCG, OPT_SLIP = 0.0015, 0.065, 0.20, 0.003
OUT = "/home/arun/quantifyd/research/105_momentum_put_hedge/results/hedge_g4.csv"


def supertrend(P, M):
    ds = sorted(SH); atr = {}; tr = 0.0
    for i in range(1, len(ds)):
        d, p = ds[i], ds[i - 1]
        h, l, c = SH[d]; pc = SH[p][2]
        t = max(h - l, abs(h - pc), abs(l - pc))
        tr = t if i == 1 else (tr * (P - 1) + t) / P
        atr[d] = tr
    bull = {}; pub = plb = None; pb = True
    for d in ds:
        if d not in atr:
            continue
        h, l, c = SH[d]; mid = (h + l) / 2; a = M * atr[d]
        ub, lb = mid + a, mid - a
        if pub is not None:
            ub = min(ub, pub) if c <= pub else ub
            lb = max(lb, plb) if c >= plb else lb
        b = (c >= lb) if pb else (c > (pub if pub is not None else ub))
        bull[d] = b; pb, pub, plb = b, ub, lb
    return bull


ST = {k: supertrend(*k) for k in [(7, 2), (7, 3), (10, 2), (10, 3), (14, 3), (20, 4)]}
EMA = {n: close[BENCH].ffill().ewm(span=n, adjust=False).mean() for n in (10, 20, 50)}


def run2(mode="hedge", tenor="weekly", moneyness=0.0, ratio=2.0, struct="long",
         resize=0.25, trail=None, start=pd.Timestamp("2019-02-01")):
    lo, hi, want = (22, 45, 30) if tenor == "monthly" else (3, 12, 7)
    ix = idx[idx >= start]
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    hedge = None; blocked = False
    nav_pre, nav_post = [], []
    stt = dict(resizes=0, hedges=0, rolls=0, prem=0.0)

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d, frac=1.0):
        nonlocal cash, tax
        v, b, bd = held[s]; cut = v * frac; g = (v - b) * frac
        if g > 0 and (d - bd).days < 365:
            tax += g * STCG
        cash += cut * (1 - RT_LEG)
        if frac >= 0.999:
            del held[s]
        else:
            held[s] = [v - cut, b * (1 - frac), bd]

    def fund(amt, d):
        nonlocal cash
        if cash >= amt:
            cash -= amt; return amt
        need = amt - max(0.0, cash); cash = max(0.0, cash); eq = E()
        if eq <= 0:
            return 0.0
        fr = min(0.98, need / eq / (1 - RT_LEG))
        for s in list(held):
            sell(s, d, fr)
        got = min(amt, cash); cash -= got; return got

    def hval_unit(d):
        if not hedge:
            return 0.0
        v = 0.0
        for (K, sg, rt, ser, last) in hedge['legs']:
            px = ser.get(d.strftime("%Y-%m-%d"))
            if px is None:
                px = last[0]
            else:
                last[0] = px
            v += sg * rt * px
        return v

    def open_hedge(d):
        nonlocal hedge
        eq = E()
        if eq <= 0 or d not in S:
            return
        Ex = hs.pick_expiry(d, lo, hi, want)
        if not Ex:
            return
        sp = S[d]; specs = [(moneyness, +1, 1.0)]
        if struct == "spread":
            specs.append((moneyness - 0.10, -1, 1.0))
        legs = []; net = 0.0
        for (m, sg, rt) in specs:
            r = hs.pick_strike(d, Ex, sp * (1 + m))
            if not r:
                return
            K, px = r
            legs.append((K, sg, rt, hs.series(Ex, K), [px])); net += sg * rt * px
        if net <= 0:
            return
        units = ratio * eq / sp; cost = net * units * (1 + OPT_SLIP)
        if fund(cost, d) < cost * 0.9:
            return
        hedge = dict(legs=legs, units=units, expiry=Ex, cost=cost, peak=net * units)
        stt['prem'] += cost; stt['hedges'] += 1

    def close_hedge(d, roll=False):
        nonlocal hedge, cash
        if not hedge:
            return
        cash += hval_unit(d) * hedge['units'] * (1 - OPT_SLIP)
        hedge = None
        if roll:
            stt['rolls'] += 1

    def resize_hedge(d):
        """THE FIX: keep hedge notional == ratio x CURRENT equity as stocks stop out."""
        nonlocal hedge, cash
        if not hedge or d not in S:
            return
        eq = E()
        if eq <= 0:
            close_hedge(d); return
        tgt = ratio * eq / S[d]; u = hedge['units']
        if tgt <= 0 or abs(u - tgt) / tgt <= resize:
            return
        pu = hval_unit(d)
        if tgt < u:
            cash += (u - tgt) * pu * (1 - OPT_SLIP); hedge['cost'] *= tgt / u
        else:
            add = (tgt - u) * pu * (1 + OPT_SLIP)
            if fund(add, d) < add * 0.9:
                return
            hedge['cost'] += add
        hedge['units'] = tgt; stt['resizes'] += 1

    for d in ix:
        if held and prev is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    v[0] += v[0] * (p1 / p0 - 1.0)
        if cash > 0:
            cash *= (1 + CASH_Y) ** (1 / 252)
        if held:
            lows = ROLL_LOW.loc[d]
            for s in list(held):
                p1 = close.at[d, s] if s in close.columns else np.nan
                ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    sell(s, d)
        if hedge and resize > 0:
            resize_hedge(d)
        if hedge:
            hv = hval_unit(d) * hedge['units']
            hedge['peak'] = max(hedge['peak'], hv)
            if d >= pd.Timestamp(hedge['expiry']):
                b = NBX.loc[:d].dropna()
                off = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
                close_hedge(d, roll=True)
                if off and not blocked:
                    open_hedge(d)
            elif trail:
                t, arg = trail
                if t == "st":
                    hit = ST[arg].get(d, False)
                elif t == "prem":
                    hit = hv > 0 and hv <= hedge['peak'] * (1 - arg)
                elif t == "ema":
                    hit = NBX.loc[d] > EMA[arg].loc[d]
                else:
                    hit = False
                if hit:
                    close_hedge(d); blocked = True
        if d in WK:
            b = NBX.loc[:d].dropna()
            off = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if off:
                if mode == "cash_exit":
                    for s in list(held):
                        sell(s, d)
                    derisked = True
                elif mode == "hedge" and hedge is None and not blocked:
                    open_hedge(d)
            else:
                derisked = False; blocked = False
                if hedge:
                    close_hedge(d)
        if d in ME and not derisked:
            etf = ETF.get(d) or []
            if etf:
                top, buf = etf[:8], set(etf[:22])
                for s in list(held):
                    if s not in buf:
                        sell(s, d)
                cand = (list(held) + [x for x in top if x not in held])[:8]
                tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
                per = (E() + cash) / 8
                for s in tgt:
                    if s not in held and cash > 0:
                        buy = min(per, cash)
                        if buy > 0:
                            held[s] = [buy * (1 - RT_LEG), buy, d]; cash -= buy
        hv = hval_unit(d) * hedge['units'] if hedge else 0.0
        nav_pre.append((d, E() + cash + hv))
        nav_post.append((d, E() + cash + hv - tax))
        prev = d

    def blk(p):
        n = pd.DataFrame(p, columns=["d", "v"]).set_index("d")["v"]
        y = (n.index[-1] - n.index[0]).days / 365.25
        c = (n.iloc[-1] / n.iloc[0]) ** (1 / y) - 1
        dr = n.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        dd = ((n - n.cummax()) / n.cummax()).min()
        return c * 100, dd * 100, sh, (c / abs(dd) if dd < 0 else 0)

    c1, d1, s1, k1 = blk(nav_pre); c2, d2, s2, k2 = blk(nav_post)
    return dict(cagr=round(c1, 1), maxdd=round(d1, 1), sharpe=round(s1, 2), calmar=round(k1, 2),
                net_cagr=round(c2, 1), net_calmar=round(k2, 2), resizes=stt['resizes'],
                hedges=stt['hedges'], prem=round(stt['prem'] * 100, 1))


st19 = pd.Timestamp("2019-02-01"); st11 = pd.Timestamp("2011-01-01")
grid = [
    ("BASE_cash_exit_2019", dict(mode="cash_exit", start=st19)),
    ("WK_r2.0_NOresize_OLDFLAWED", dict(tenor="weekly", ratio=2.0, resize=0, start=st19)),
    ("WK_r2.0_resize25", dict(tenor="weekly", ratio=2.0, resize=0.25, start=st19)),
    ("WK_r2.0_resize10", dict(tenor="weekly", ratio=2.0, resize=0.10, start=st19)),
    ("MN_r2.0_NOresize_OLDFLAWED", dict(tenor="monthly", ratio=2.0, resize=0, start=st19)),
    ("MN_r2.0_resize25", dict(tenor="monthly", ratio=2.0, resize=0.25, start=st19)),
    ("MN_r2.0_resize10", dict(tenor="monthly", ratio=2.0, resize=0.10, start=st19)),
    ("BASE_cash_exit_2011", dict(mode="cash_exit", start=st11)),
    ("MN_r2.0_resize25_2011", dict(tenor="monthly", ratio=2.0, resize=0.25, start=st11)),
    ("MN_r1.0_resize25_2011", dict(tenor="monthly", ratio=1.0, resize=0.25, start=st11)),
]
for p, m in [(7, 2), (7, 3), (10, 2), (10, 3), (14, 3), (20, 4)]:
    grid.append((f"WK_r2.0_rs25_ST{p}-{m}",
                 dict(tenor="weekly", ratio=2.0, resize=0.25, trail=("st", (p, m)), start=st19)))
for g in (0.30, 0.50):
    grid.append((f"WK_r2.0_rs25_premtrail{int(g*100)}",
                 dict(tenor="weekly", ratio=2.0, resize=0.25, trail=("prem", g), start=st19)))
for n in (10, 20, 50):
    grid.append((f"WK_r2.0_rs25_emaX{n}",
                 dict(tenor="weekly", ratio=2.0, resize=0.25, trail=("ema", n), start=st19)))

F = ["config", "cagr", "net_cagr", "maxdd", "sharpe", "calmar", "net_calmar", "resizes", "hedges", "prem"]
w = csv.DictWriter(open(OUT, "w", newline=""), fieldnames=F); w.writeheader()
print(f"\n{'config':>30} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} {'netCal':>7} {'resz':>5} {'prem':>7}", flush=True)
for lbl, kw in grid:
    try:
        r = run2(**kw)
    except Exception as e:
        print(f"  {lbl} FAILED {e}", flush=True); continue
    w.writerow({k: (lbl if k == "config" else r[k]) for k in F})
    print(f"{lbl:>30} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% "
          f"{r['net_calmar']:>7.2f} {r['resizes']:>5} {r['prem']:>7.1f}", flush=True)
    sys.stdout.flush()
print("done", flush=True)
