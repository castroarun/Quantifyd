"""research/108b — two questions against the CURRENT system (hold + bi-weekly 2x put at risk-off):

  Q1. When a stock trips its DONCHIAN channel, refill that slot at the WEEKLY job or wait for month-end?
  Q2. Should the put be bought/sold on a DAILY EOD 100-SMA check instead of only the weekly routine?

Window 2019-2026 (bi-weekly options exist only from 2019). Net of costs and 20% STCG.
"""
import importlib.util, sys
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "hs", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_sweep.py")
hs = importlib.util.module_from_spec(spec); spec.loader.exec_module(hs)
close, idx, ME, WK = hs.close, hs.idx, hs.ME, hs.WK
ROLL_LOW, NBX, ETF, S = hs.ROLL_LOW, hs.NBX, hs.ETF, hs.S
RT, CASH_Y, STCG, OPT_SLIP = 0.0015, 0.065, 0.20, 0.003
START = pd.Timestamp("2019-02-01")
LO, HI, WANT = 8, 20, 14        # bi-weekly


def run(refill, gate_freq="weekly", ratio=2.0, resize=0.25):
    ix = idx[idx >= START]
    cash, held, prev, tax = 1.0, {}, None, 0.0
    hedge = None; gate_off = False
    pre, post, idle = [], [], []
    st = dict(hedges=0, refills=0)

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d):
        nonlocal cash, tax
        v, b, bd = held.pop(s)
        if v > b and (d - bd).days < 365:
            tax += (v - b) * STCG
        cash += v * (1 - RT)

    def hval(d):
        if not hedge:
            return 0.0
        p = hedge["ser"].get(d.strftime("%Y-%m-%d"))
        if p is None:
            p = hedge["last"][0]
        else:
            hedge["last"][0] = p
        return p * hedge["units"]

    def hopen(d):
        nonlocal hedge, cash
        eq = E()
        if eq <= 0 or d not in S or hedge:
            return
        Ex = hs.pick_expiry(d, LO, HI, WANT)
        if not Ex:
            return
        r = hs.pick_strike(d, Ex, S[d])
        if not r:
            return
        K, px = r
        if px <= 0:
            return
        u = ratio * eq / S[d]
        cost = px * u * (1 + OPT_SLIP)
        if cost > cash:
            return
        cash -= cost
        hedge = dict(K=K, E=Ex, units=u, ser=hs.series(Ex, K), last=[px], cost=cost)
        st["hedges"] += 1

    def hclose(d):
        nonlocal hedge, cash
        if not hedge:
            return
        cash += hval(d) * (1 - OPT_SLIP)
        hedge = None

    def refill_slots(d, full):
        nonlocal cash
        key = d if full else max([x for x in ETF if x <= d], default=None)
        etf = ETF.get(key) if key else None
        if not etf:
            return
        if full:
            buf = set(etf[:22])
            for s in list(held):
                if s not in buf:
                    sell(s, d)
        empty = 8 - len(held)
        if empty <= 0 or cash <= 0:
            return
        cands = [s for s in etf if s not in held and s in close.columns
                 and pd.notna(close.at[d, s])][:empty]
        if not cands:
            return
        per = (E() + cash) / 8
        for s in cands:
            buy = min(per, cash)
            if buy <= 0:
                break
            held[s] = [buy * (1 - RT), buy, d]
            cash -= buy
        st["refills"] += 1

    for d in ix:
        if held and prev is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    v[0] += v[0] * (p1 / p0 - 1.0)
        if cash > 0:
            cash *= (1 + CASH_Y) ** (1 / 252)
        # DONCHIAN stop-outs create the empty slots in question
        if held:
            lows = ROLL_LOW.loc[d]
            for s in list(held):
                p1 = close.at[d, s] if s in close.columns else np.nan
                ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    sell(s, d)
        # keep the put sized to current equity as stocks drop out
        if hedge and resize > 0 and d in S:
            eq = E()
            if eq <= 0:
                hclose(d)
            else:
                tgt = ratio * eq / S[d]
                if hedge and abs(hedge["units"] - tgt) / max(tgt, 1e-9) > resize:
                    p = hval(d) / hedge["units"] if hedge["units"] else 0.0
                    if tgt < hedge["units"]:
                        cash += (hedge["units"] - tgt) * p * (1 - OPT_SLIP)
                    hedge["units"] = tgt
        if hedge and pd.Timestamp(hedge["E"]) <= d:
            hclose(d)
            if gate_off:
                hopen(d)
        # Q2: evaluate the 100-SMA gate DAILY or only on the weekly bar
        if gate_freq == "daily" or d in WK:
            b = NBX.loc[:d].dropna()
            gate_off = bool(len(b) >= 100 and b.iloc[-1] < b.tail(100).mean())
            if gate_off:
                hopen(d)
            elif hedge:
                hclose(d)
        # Q1: refill Donchian-emptied slots weekly, or wait for month-end
        if refill == "weekly" and d in WK and not gate_off:
            refill_slots(d, False)
        if d in ME:
            refill_slots(d, True)
        nav = E() + cash + (hval(d) if hedge else 0.0)
        pre.append((d, nav)); post.append((d, nav - tax))
        idle.append(cash / nav if nav > 0 else 0)
        prev = d

    def blk(p):
        n = pd.DataFrame(p, columns=["d", "v"]).set_index("d")["v"]
        y = (n.index[-1] - n.index[0]).days / 365.25
        c = (n.iloc[-1] / n.iloc[0]) ** (1 / y) - 1
        dr = n.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        dd = ((n - n.cummax()) / n.cummax()).min()
        return c * 100, dd * 100, sh, (c / abs(dd) if dd < 0 else 0)

    c1, d1, s1, k1 = blk(pre)
    c2, d2, s2, k2 = blk(post)
    return dict(net_cagr=round(c2, 1), maxdd=round(d1, 1), sharpe=round(s1, 2),
                net_calmar=round(k2, 2), idle=round(100 * float(np.mean(idle)), 1),
                hedges=st["hedges"])


print("CURRENT SYSTEM: hold the 8 stocks + bi-weekly 2x NIFTY put at risk-off. 2019-2026, net of STCG.")
print()
print(f"{'Donchian slot refill':>22} {'put gate check':>16} {'netCAGR':>8} {'MaxDD':>7} "
      f"{'Sharpe':>7} {'netCal':>7} {'idle':>6} {'#puts':>6}")
arms = [("monthly", "weekly", "MONTH-END (current)", "WEEKLY (current)"),
        ("weekly", "weekly", "WEEKLY job", "WEEKLY (current)"),
        ("monthly", "daily", "MONTH-END (current)", "DAILY EOD"),
        ("weekly", "daily", "WEEKLY job", "DAILY EOD")]
res = {}
for r, gf, l1, l2 in arms:
    x = run(r, gate_freq=gf)
    res[(r, gf)] = x
    print(f"{l1:>22} {l2:>16} {x['net_cagr']:>7.1f}% {x['maxdd']:>6.1f}% {x['sharpe']:>7.2f} "
          f"{x['net_calmar']:>7.2f} {x['idle']:>5.1f}% {x['hedges']:>6}", flush=True)
base = res[("monthly", "weekly")]
print()
print(f"vs the current configuration (net Calmar {base['net_calmar']}):")
for (r, gf), x in res.items():
    if (r, gf) == ("monthly", "weekly"):
        continue
    print(f"  refill={r:<8} gate={gf:<7} -> netCal {x['net_calmar']:.2f} "
          f"({x['net_calmar'] - base['net_calmar']:+.2f})  CAGR {x['net_cagr'] - base['net_cagr']:+.1f}pp  "
          f"DD {x['maxdd'] - base['maxdd']:+.1f}pp")
