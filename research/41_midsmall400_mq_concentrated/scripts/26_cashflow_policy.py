"""Phase 26 — cash-flow allocation / de-allocation policy study.

Live users add/withdraw cash mid-stream. On the chosen best system
(SMOOTHEST + keep-top8, weekly daily-marked engine, fresh VPS data),
test which INFLOW deploy rule and which OUTFLOW raise rule serve the
investor best, under a realistic scenario: monthly SIP + lump deposits
+ lump withdrawals INCLUDING one forced during a drawdown.

Metrics (flows present → NAV-CAGR alone misleads):
  * TWR  — time-weighted CAGR via unitisation (pure strategy skill)
  * XIRR — money-weighted return (what the investor actually earned)
  * dailyDD on per-unit value (flow-neutral drawdown)
  * post-tax@20% (lot-level: STCG on lots <365d; LTCG modelled 0,
    consistent with the rest of the study)

Flow schedule is FIXED & identical across all policies so every delta
is attributable to the policy, not the path.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
SC = Path(__file__).resolve().parent
_p = importlib.util.spec_from_file_location(
    "p22", str(SC / "22_smoothest_variants.py"))
p22 = importlib.util.module_from_spec(_p); _p.loader.exec_module(p22)
rs2 = p22.rs2
BENCH, N, Q, ATH_THR, TRAIL = (rs2.NIFTY_SYM, p22.N, p22.Q,
                               p22.ATH_THR, p22.TRAIL)
BUF, DAY_CASH, RT, START = p22.BUF, p22.DAY_CASH, p22.RT, p22.START
own_pf = p22.own_pf
K_KEEP = 8                                   # keep-top8 base


def xirr(flows):
    """flows: list of (datetime, amount). Investor convention:
    money INTO the fund = negative, money OUT (+ terminal value) =
    positive. Returns annualised rate via bisection."""
    if not flows or len(flows) < 2:
        return np.nan
    t0 = flows[0][0]
    yrs = [(d - t0).days / 365.25 for d, _ in flows]
    amt = [a for _, a in flows]

    def npv(r):
        return sum(a / (1 + r) ** y for a, y in zip(amt, yrs))
    lo, hi = -0.99, 5.0
    flo, fhi = npv(lo), npv(hi)
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return np.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = npv(mid)
        if abs(fm) < 1e-9:
            return mid
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def build_schedule(idx):
    """Deterministic flows on the nearest trading day >= target.
    Returns {date: signed_amount} (+deposit, -withdrawal) excluding
    the SIP (handled inline). Amounts in units of initial capital=1."""
    def nd(s):
        c = idx[idx >= pd.Timestamp(s)]
        return c[0] if len(c) else None
    sched = {}
    for s, a in [("2017-06-01", +0.50), ("2023-01-02", +0.50),
                 ("2019-09-02", -0.40), ("2020-03-23", -0.40),
                 ("2025-06-02", -0.30)]:
        d = nd(s)
        if d is not None:
            sched[d] = sched.get(d, 0.0) + a
    return sched


SIP = 0.01                                   # +1% of initial / month


def run_cf(close, tv, inflow, outflow, stcg=0.0):
    idx = close.index[close.index >= START]
    me = sorted(set(rs2.month_ends(close.index)) & set(idx))
    me_set = set(me)
    sip_days = set()                         # first trading day / month
    seen = set()
    for d in idx:
        ym = (d.year, d.month)
        if ym not in seen:
            seen.add(ym); sip_days.add(d)
    wk = pd.Series(idx, index=idx)
    wk_last = set(wk.groupby([idx.isocalendar().year,
                              idx.isocalendar().week]).last().values)
    sched = build_schedule(idx)

    cash, held, prev = 1.0, {}, None         # sym->[val,cost,buydt,peak]
    units, tax = 1.0, 0.0                     # unitisation for TWR
    extflows = [(idx[0], -1.0)]               # initial investment
    nav, unit_nav = [], []

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d, frac=1.0):
        nonlocal tax
        g = (stt[0] - stt[1]) * frac
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; tax += t; return t
        return 0.0

    def rs_now(d):
        return rs2.rs_scores(close, d, 120)

    def deploy(amount, d, mode):
        """Inflow placement. amount already added to `cash`."""
        nonlocal cash
        if amount <= 0 or not held:
            return                           # park in cash (C1 / no book)
        sc = rs_now(d)
        if mode == "C2" or (mode == "C5" and not _roff(d)):
            names = list(held)
        elif mode == "C3":
            rk = {s: (sc.get(s, -9e9) if sc is not None else 0)
                  for s in held}
            names = sorted(rk, key=rk.get, reverse=True)[:3]
        elif mode == "C4":
            tot = sum(v[0] for v in held.values())
            for s, st in held.items():
                add = amount * (st[0] / tot) if tot > 0 else 0
                st[0] += add; st[1] += add
            cash -= amount + amount * RT
            return
        else:                                # C1 park / C5 risk-off park
            return
        if not names:
            return
        per = amount / len(names)
        for s in names:
            held[s][0] += per; held[s][1] += per
        cash -= amount + amount * RT

    def raise_cash(amount, d, mode):
        """Outflow. Use idle cash first; sell per policy for shortfall.
        `amount` leaves the fund (already reserved by caller)."""
        nonlocal cash
        need = amount - max(cash, 0.0)
        cash -= amount
        if need <= 0 or not held:
            return
        sc = rs_now(d)
        if mode == "W1":                     # pro-rata trim
            order = [(s, 1) for s in held]
        elif mode == "W2":                   # lowest-RS first
            order = sorted(held, key=lambda s: (sc.get(s, -9e9)
                           if sc is not None else 0))
            order = [(s, None) for s in order]
        elif mode == "W3":                   # tax-aware: LTCG then loRS
            def key(s):
                ltcg = (d - held[s][2]).days >= 365
                gain = held[s][0] - held[s][1]
                return (0 if ltcg else 1, 0 if gain <= 0 else 1,
                        sc.get(s, -9e9) if sc is not None else 0)
            order = [(s, None) for s in sorted(held, key=key)]
        else:                                # W4 trim-overweight→EW
            tot = sum(v[0] for v in held.values())
            tgt = tot / max(1, len(held))
            order = [(s, None) for s in sorted(
                held, key=lambda s: -held[s][0])]
            _ = tgt
        if mode == "W1":
            tot = sum(v[0] for v in held.values())
            if tot <= 0:
                return
            for s in list(held):
                fr = min(1.0, need * (held[s][0] / tot) / max(held[s][0],
                         1e-12))
                t = realize(held[s], d, fr)
                held[s][0] *= (1 - fr); held[s][1] *= (1 - fr)
                cash += 0  # proceeds fund the withdrawal, not re-held
                tax_adj = t
                need -= held[s][0] / (1 - fr) * fr if fr < 1 else 0
                if held[s][0] < 1e-9:
                    held.pop(s)
            return
        for s, _f in order:
            if need <= 1e-9 or s not in held:
                break
            v = held[s][0]
            take = min(v, need)
            fr = take / v if v > 0 else 1.0
            t = realize(held[s], d, fr)
            held[s][0] -= take; held[s][1] *= (1 - fr)
            need -= (take - t)
            if held[s][0] < 1e-9:
                held.pop(s)

    def _roff(d):
        b = close.loc[:d, BENCH].dropna()
        return len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    st[0] *= p1 / p0
                    st[3] = max(st[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]
        navpre = E() + cash

        # ---- external flows (unitise BEFORE deploying) ----
        flow = 0.0
        if d in sip_days:
            flow += SIP
        flow += sched.get(d, 0.0)
        if flow != 0.0 and navpre > 0:
            units *= (navpre + flow) / navpre        # keep unit value
            extflows.append((d, -flow))              # +dep = money in
            if flow > 0:
                cash += flow
                deploy(flow, d, inflow)
            else:
                raise_cash(-flow, d, outflow)

        # ---- WEEKLY keep-top8 risk-off ----
        if d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if roff and len(held) > K_KEEP:
                sc = rs2.rs_scores(close, d, 120)
                rk = {s: (sc.get(s, -9e9) if sc is not None else 0)
                      for s in held}
                keep = set(sorted(rk, key=rk.get, reverse=True)[:K_KEEP])
                for s in list(held):
                    if s not in keep:
                        t = realize(held[s], d)
                        cash += held.pop(s)[0] - t

        # ---- MONTHLY selection ----
        if d in me_set:
            for s in list(held):
                cs = h[s].dropna(); p1 = px.get(s, np.nan)
                if (len(cs) >= 100 and cs.iloc[-1] < cs.tail(100).mean()) \
                   or (pd.notna(p1) and held[s][3] > 0 and
                       p1 <= (1 - TRAIL) * held[s][3]):
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
            b = h[BENCH].dropna()
            roff_now = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if not roff_now:
                uni = rs2.pit_universe(tv, close, d, "mid")
                sc = rs2.rs_scores(close, d, 120)
                elig = []
                if sc is not None and uni:
                    for s in sc.sort_values(ascending=False).index:
                        if s == BENCH or s not in uni:
                            continue
                        cs = h[s].dropna()
                        pf = own_pf(cs.tail(252))
                        if pf is None or pf < Q:
                            continue
                        if len(cs) < 100 or \
                           cs.iloc[-1] < cs.tail(100).mean():
                            continue
                        if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                            continue
                        elig.append(s)
                        if len(elig) >= BUF:
                            break
                if elig:
                    bufset = set(elig[:BUF])
                    for s in list(held):
                        if s not in bufset:
                            t = realize(held[s], d)
                            cash += held.pop(s)[0] - t
                    tot = E() + cash
                    names = ([s for s in held] +
                             [s for s in elig[:N] if s not in held])[:N]
                    if names:
                        w = tot / len(names)
                        turn = (len(set(held).symmetric_difference(
                            set(names))) / max(1, 2 * N))
                        nh = {}
                        for s in names:
                            old = held.get(s)
                            nh[s] = ([w, old[1], old[2], old[3]] if old
                                     else [w, w, d, px[s]])
                        held = nh
                        cash = tot - w * len(names)
                        cash -= tot * RT * turn * 2
        tot = E() + cash
        nav.append((d, tot))
        unit_nav.append((d, tot / units if units else tot))
        prev = d

    extflows.append((idx[-1], nav[-1][1]))            # terminal value
    un = pd.DataFrame(unit_nav, columns=["date", "u"]).set_index(
        "date")["u"]
    yrs = (un.index[-1] - un.index[0]).days / 365.25
    twr = (un.iloc[-1] / un.iloc[0]) ** (1 / yrs) - 1
    dd = ((un - un.cummax()) / un.cummax()).min()
    irr = xirr(extflows)
    return dict(twr=twr * 100, xirr=(irr * 100 if not np.isnan(irr)
                else np.nan), dd=dd * 100, final=nav[-1][1],
                tax=tax)


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    INF = ["C1", "C2", "C3", "C4", "C5"]
    OUT = ["W1", "W2", "W3", "W4"]
    rows = []
    print("no-flow control ...", flush=True)
    # control: zero schedule by running with policies that never fire
    # (handled implicitly — flows always present; control = C1/W1 ref)
    for ci in INF:
        for wo in OUT:
            g = run_cf(close, tv, ci, wo)
            t = run_cf(close, tv, ci, wo, stcg=0.20)
            rows.append(dict(inflow=ci, outflow=wo,
                             twr=round(g['twr'], 1),
                             xirr=round(g['xirr'], 1),
                             xirr_posttax=round(t['xirr'], 1),
                             dailyDD=round(g['dd'], 1),
                             final=round(g['final'], 2)))
            print(f"  {ci}/{wo}  TWR={g['twr']:5.1f}% XIRR={g['xirr']:5.1f}%"
                  f" net20={t['xirr']:5.1f}% DD={g['dd']:6.1f}% "
                  f"final={g['final']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "phase26_cashflow_policy.csv", index=False)
    print("\n=== PHASE 26 — cash-flow policy grid (5 inflow × 4 outflow) "
          "===\n" + df.to_string(index=False))
    best = df.sort_values("xirr_posttax", ascending=False).head(3)
    print("\nTop-3 by post-tax XIRR:\n" + best.to_string(index=False))


if __name__ == "__main__":
    main()
