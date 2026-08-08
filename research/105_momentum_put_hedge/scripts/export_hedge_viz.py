"""Export everything needed to VISUALISE the put-hedge study: NAV curves for the hedged book and the
cash-exit baseline, the gate risk-off (regime) periods for shading, and every individual put purchase
with its strike, premium paid, exit value and P&L.

Config = the study's best arm: WEEKLY ATM puts at 2.0x equity, daily re-sized, exit on gate reversal,
rolled at expiry. 2019-2026 (weekly options only exist from 2019).
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "hs", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_sweep.py")
hs = importlib.util.module_from_spec(spec); spec.loader.exec_module(hs)
close, idx, ME, WK = hs.close, hs.idx, hs.ME, hs.WK
ROLL_LOW, NBX, ETF, S, BENCH = hs.ROLL_LOW, hs.NBX, hs.ETF, hs.S, hs.BENCH
RT_LEG, CASH_Y, STCG, OPT_SLIP = 0.0015, 0.065, 0.20, 0.003
RES = Path("/home/arun/quantifyd/research/105_momentum_put_hedge/results")
START = pd.Timestamp("2019-02-01")


def run(mode, ratio=2.0, resize=0.25):
    ix = idx[idx >= START]
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    hedge = None; gate_state = False
    nav, gate_log, episodes = [], [], []

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

    def px_now(d):
        if not hedge:
            return 0.0
        p = hedge['ser'].get(d.strftime("%Y-%m-%d"))
        if p is None:
            return hedge['last'][0]
        hedge['last'][0] = p
        return p

    def open_hedge(d):
        nonlocal hedge
        eq = E()
        if eq <= 0 or d not in S:
            return
        Ex = hs.pick_expiry(d, 3, 12, 7)
        if not Ex:
            return
        sp = S[d]
        r = hs.pick_strike(d, Ex, sp)
        if not r:
            return
        K, p = r
        if p <= 0:
            return
        units = ratio * eq / sp
        cost = p * units * (1 + OPT_SLIP)
        if fund(cost, d) < cost * 0.9:
            return
        hedge = dict(K=K, E=Ex, units=units, ser=hs.series(Ex, K), last=[p],
                     cost=cost, entry_px=p, entry_d=d, entry_spot=sp, entry_nav=E() + cash)

    def close_hedge(d, why):
        nonlocal hedge, cash
        if not hedge:
            return
        p = px_now(d)
        proceeds = p * hedge['units'] * (1 - OPT_SLIP)
        cash += proceeds
        episodes.append(dict(
            entry=hedge['entry_d'].strftime("%Y-%m-%d"), exit=d.strftime("%Y-%m-%d"),
            strike=int(hedge['K']), expiry=hedge['E'], spot=round(hedge['entry_spot']),
            prem_in=round(hedge['entry_px'], 1), prem_out=round(p, 1),
            cost_pct=round(hedge['cost'] * 100, 3), pnl_pct=round((proceeds - hedge['cost']) * 100, 3),
            ret=round((proceeds / hedge['cost'] - 1) * 100, 1) if hedge['cost'] > 0 else 0,
            days=(d - hedge['entry_d']).days, why=why))
        hedge = None

    def resize_hedge(d):
        nonlocal hedge, cash
        if not hedge or d not in S:
            return
        eq = E()
        if eq <= 0:
            close_hedge(d, "all stocks stopped out"); return
        tgt = ratio * eq / S[d]; u = hedge['units']
        if tgt <= 0 or abs(u - tgt) / tgt <= resize:
            return
        p = px_now(d)
        if tgt < u:
            cash += (u - tgt) * p * (1 - OPT_SLIP); hedge['cost'] *= tgt / u
        else:
            add = (tgt - u) * p * (1 + OPT_SLIP)
            if fund(add, d) < add * 0.9:
                return
            hedge['cost'] += add
        hedge['units'] = tgt

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
        if mode == "hedge":
            if hedge and pd.Timestamp(hedge['E']) <= d:
                close_hedge(d, "expiry roll")
                if gate_state:
                    open_hedge(d)
            elif gate_state and hedge is None:
                open_hedge(d)
            elif not gate_state and hedge is not None:
                close_hedge(d, "gate turned risk-on")
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
        hv = px_now(d) * hedge['units'] if hedge else 0.0
        nav.append((d.strftime("%Y-%m-%d"), round(E() + cash + hv, 5), round(E() + cash + hv - tax, 5)))
        gate_log.append((d.strftime("%Y-%m-%d"), 1 if gate_state else 0, round(hv * 100, 3)))
        prev = d
    return nav, gate_log, episodes


print("running hedged book ...", flush=True)
navH, gate, eps = run("hedge")
print(f"  {len(eps)} put purchases recorded", flush=True)
print("running cash-exit baseline ...", flush=True)
navB, _, _ = run("cash_exit")
nb = close[BENCH].loc[close.index >= START].dropna(); nb = nb / nb.iloc[0]
out = dict(hedged=navH, baseline=navB, gate=gate, episodes=eps,
           bench=[[d.strftime("%Y-%m-%d"), round(float(v), 5)] for d, v in nb.items()])
json.dump(out, open(RES / "hedge_viz.json", "w"))
tot_cost = sum(e['cost_pct'] for e in eps); tot_pnl = sum(e['pnl_pct'] for e in eps)
win = sum(1 for e in eps if e['pnl_pct'] > 0)
print(f"\nput purchases: {len(eps)}   winners: {win} ({100*win/max(1,len(eps)):.0f}%)")
print(f"total premium spent: {tot_cost:.1f}% of initial capital")
print(f"total hedge P&L:     {tot_pnl:+.1f}% of initial capital")
print(f"risk-off days: {sum(g[1] for g in gate)} of {len(gate)}")
print("wrote hedge_viz.json", flush=True)
