"""G5 — FAIR trailing-exit test (fixes the degenerate G4 trail logic) + ST diagnostic.

G4 flaw: a trail exit set blocked=True, which suppressed re-entry for the WHOLE risk-off episode. So any
trail that fired early effectively disabled the hedge — which is why all six SuperTrend configs returned
identical numbers. That was a broken test, not evidence against SuperTrend.

G5 fix: one unified rule — want_hedge(d) = (gate risk-off) AND (trail signal is not bullish). The hedge is
opened/closed to track that state, so a trail exit naturally RE-ENTERS when the signal turns bearish again.
Keeps the G4 daily re-sizing fix. Also reports days-hedged so the trail's effect is visible.
"""
import importlib.util, csv, sys
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location(
    "hs", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_sweep.py")
hs = importlib.util.module_from_spec(spec); spec.loader.exec_module(hs)
close, idx, ME, WK = hs.close, hs.idx, hs.ME, hs.WK
ROLL_LOW, NBX, ETF, S, SH, BENCH = hs.ROLL_LOW, hs.NBX, hs.ETF, hs.S, hs.SH, hs.BENCH
RT_LEG, CASH_Y, STCG, OPT_SLIP = 0.0015, 0.065, 0.20, 0.003
OUT = "/home/arun/quantifyd/research/105_momentum_put_hedge/results/hedge_g5.csv"


def supertrend(P, M):
    """Standard SuperTrend. Final bands ratchet using the PREVIOUS close; the trend flips against
    those final bands (the earlier version reset the band instead of flipping -> always bullish)."""
    ds = sorted(SH)
    atr = {}; tr = 0.0
    for i in range(1, len(ds)):
        d, p = ds[i], ds[i - 1]
        h, l, c = SH[d]; pc = SH[p][2]
        t = max(h - l, abs(h - pc), abs(l - pc))
        tr = t if i == 1 else (tr * (P - 1) + t) / P
        atr[d] = tr
    bull = {}; pfub = pflb = None; pb = True; pc = None
    for d in ds:
        h, l, c = SH[d]
        if d not in atr:
            pc = c; continue
        mid = (h + l) / 2; a = M * atr[d]
        bub, blb = mid + a, mid - a
        if pfub is None:
            fub, flb = bub, blb
        else:
            fub = bub if (bub < pfub or (pc is not None and pc > pfub)) else pfub
            flb = blb if (blb > pflb or (pc is not None and pc < pflb)) else pflb
        b = (c >= flb) if pb else (c > fub)
        bull[d] = b
        pb, pfub, pflb, pc = b, fub, flb, c
    return bull


STC = [(7, 2), (7, 3), (10, 2), (10, 3), (14, 3), (20, 4)]
ST = {k: supertrend(*k) for k in STC}
EMA = {n: close[BENCH].ffill().ewm(span=n, adjust=False).mean() for n in (10, 20, 50)}

# ── diagnostic: are the ST configs actually different, and what do they say at gate-off times? ──
gate_off = {}
for d in idx:
    b = NBX.loc[:d].dropna()
    gate_off[d] = bool(len(b) >= 100 and b.iloc[-1] < b.tail(100).mean())
state, cur = {}, False
for d in idx:
    if d in WK:
        cur = gate_off[d]
    state[d] = cur
off_days = [d for d in idx if state[d] and d >= pd.Timestamp("2019-02-01")]
print("ST DIAGNOSTIC (2019+): % of days bullish overall, and % bullish DURING gate-off "
      "(a high number here means the trail kills the hedge immediately)", flush=True)
for k in STC:
    b = ST[k]
    allb = np.mean([b.get(d, False) for d in idx if d >= pd.Timestamp("2019-02-01")]) * 100
    offb = np.mean([b.get(d, False) for d in off_days]) * 100 if off_days else 0
    flips = sum(1 for i in range(1, len(off_days)) if b.get(off_days[i]) != b.get(off_days[i - 1]))
    print(f"  ST{k}: bullish {allb:5.1f}% of all days | {offb:5.1f}% of gate-off days | "
          f"{flips} flips during gate-off", flush=True)
print(flush=True)


def run3(mode="hedge", tenor="weekly", moneyness=0.0, ratio=2.0, resize=0.25,
         trail=None, start=pd.Timestamp("2019-02-01")):
    lo, hi, want = (22, 45, 30) if tenor == "monthly" else (3, 12, 7)
    ix = idx[idx >= start]
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    hedge = None; gate_state = False; prem_block_until = None
    nav_pre, nav_post = [], []
    stt = dict(resizes=0, hedges=0, rolls=0, prem=0.0, hdays=0)

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
        sp = S[d]
        r = hs.pick_strike(d, Ex, sp * (1 + moneyness))
        if not r:
            return
        K, px = r
        if px <= 0:
            return
        units = ratio * eq / sp; cost = px * units * (1 + OPT_SLIP)
        if fund(cost, d) < cost * 0.9:
            return
        hedge = dict(legs=[(K, +1, 1.0, hs.series(Ex, K), [px])], units=units,
                     expiry=Ex, cost=cost, peak=px * units)
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

    def trail_bullish(d):
        """True = the trail says 'market recovering, drop the hedge'."""
        if not trail:
            return False
        t, arg = trail
        if t == "st":
            return ST[arg].get(d, False)
        if t == "ema":
            return bool(NBX.loc[d] > EMA[arg].loc[d])
        return False

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
        if d in WK:
            b = NBX.loc[:d].dropna()
            gate_state = bool(len(b) >= 100 and b.iloc[-1] < b.tail(100).mean())
            if mode == "cash_exit":
                if gate_state:
                    for s in list(held):
                        sell(s, d)
                    derisked = True
                else:
                    derisked = False
        if hedge and resize > 0:
            resize_hedge(d)
        if hedge:
            hv = hval_unit(d) * hedge['units']
            hedge['peak'] = max(hedge['peak'], hv)
            if trail and trail[0] == "prem" and hv > 0 and hv <= hedge['peak'] * (1 - trail[1]):
                close_hedge(d)                       # profit give-back; re-enter at the next expiry
                prem_block_until = pd.Timestamp(hs.pick_expiry(d, lo, hi, want) or d)
        # ── unified want-hedge rule: gate off AND trail not bullish -> natural re-entry ──
        if mode == "hedge":
            blocked = (prem_block_until is not None and d < prem_block_until)
            want_h = gate_state and not trail_bullish(d) and not blocked
            if hedge and pd.Timestamp(hedge['expiry']) <= d:
                close_hedge(d, roll=True)
                if want_h:
                    open_hedge(d)
            elif want_h and hedge is None:
                open_hedge(d)
            elif not want_h and hedge is not None:
                close_hedge(d)
        if hedge:
            stt['hdays'] += 1
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
                net_cagr=round(c2, 1), net_calmar=round(k2, 2), hedges=stt['hedges'],
                hdays=stt['hdays'], prem=round(stt['prem'] * 100, 1))


st19 = pd.Timestamp("2019-02-01")
grid = [("BASE_cash_exit", dict(mode="cash_exit", start=st19)),
        ("WK_r2.0_NOtrail (G4 ref)", dict(tenor="weekly", ratio=2.0, start=st19))]
for k in STC:
    grid.append((f"WK_r2.0_ST{k[0]}-{k[1]}", dict(tenor="weekly", ratio=2.0, trail=("st", k), start=st19)))
for n in (10, 20, 50):
    grid.append((f"WK_r2.0_emaX{n}", dict(tenor="weekly", ratio=2.0, trail=("ema", n), start=st19)))
for g in (0.30, 0.50):
    grid.append((f"WK_r2.0_premgive{int(g*100)}", dict(tenor="weekly", ratio=2.0, trail=("prem", g), start=st19)))
grid.append(("MN_r2.0_NOtrail", dict(tenor="monthly", ratio=2.0, start=st19)))
grid.append(("MN_r2.0_ST10-3", dict(tenor="monthly", ratio=2.0, trail=("st", (10, 3)), start=st19)))

F = ["config", "cagr", "net_cagr", "maxdd", "sharpe", "calmar", "net_calmar", "hedges", "hdays", "prem"]
w = csv.DictWriter(open(OUT, "w", newline=""), fieldnames=F); w.writeheader()
print(f"{'config':>28} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} {'netCal':>7} {'#hdg':>5} {'hdays':>6} {'prem':>7}", flush=True)
for lbl, kw in grid:
    try:
        r = run3(**kw)
    except Exception as e:
        print(f"  {lbl} FAILED {e}", flush=True); continue
    w.writerow({k: (lbl if k == "config" else r[k]) for k in F})
    print(f"{lbl:>28} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% "
          f"{r['net_calmar']:>7.2f} {r['hedges']:>5} {r['hdays']:>6} {r['prem']:>7.1f}", flush=True)
    sys.stdout.flush()
print("done", flush=True)
