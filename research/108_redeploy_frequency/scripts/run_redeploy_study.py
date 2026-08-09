"""Research 108 — should stopped-out cash be redeployed WEEKLY instead of waiting for month-end?

Today the book only refills empty slots at the monthly rebalance; the weekly gate re-entry fires only
from a FULLY-cash state. The code cites research/41 phase-27 ("false-dawn penalty applies to the
partially-held state") — but that was the midcap-RS book, not this Nifty-200 top-8 book. Re-test here.

Arms (identical in every other respect):
  monthly  — refill empty slots at month-end only            (CURRENT LIVE BEHAVIOUR)
  weekly   — also refill empty slots at each weekly gate check when risk-on
  daily    — refill empty slots any day cash is sufficient
Plus a cash-drag diagnostic: average % of the book sitting idle.
"""
import importlib.util, csv, sys
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
RES = Path("/home/arun/quantifyd/research/108_redeploy_frequency/results"); RES.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2011-01-01"); CASH_Y, STCG, RT = 0.065, 0.20, 0.0015
N_HOLD, BUFFER, POOL = 8, 22, 30
close, tv = m75.load()
idx = close.index[(close.index >= START) & (close.index <= pd.Timestamp("2026-08-01"))]
ME = set(pd.Timestamp(x) for x in
         pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values)
_iso = idx.isocalendar()
WK = set(pd.Timestamp(x) for x in
         pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1); NBX = close[BENCH].ffill()

print("precomputing rankings on all weekly + monthly dates (needed for weekly refill) ...", flush=True)
RANKD = sorted(ME | WK)
SNAP = {}
for d in RANKD:
    h = close.loc[:d].ffill()
    if len(h) <= 253:
        continue
    adv = tv.loc[:d].tail(126).median().dropna()
    adv = adv[[s for s in adv.index if s not in EXCLUDE and s != BENCH]]
    uni = adv.sort_values(ascending=False).index[:200]
    sc = None
    for L, w in ((126, 0.5), (252, 0.5)):
        p0, p1 = h.iloc[-L - 1], h.iloc[-1]
        r = (p1 / p0) / (p1[BENCH] / p0[BENCH]) * w
        sc = r if sc is None else sc.add(r, fill_value=np.nan)
    sub = sc[[s for s in uni if s in sc.index and pd.notna(sc[s])]]
    SNAP[d] = list(sub.sort_values(ascending=False).index[:POOL])
print(f"  {len(SNAP)} ranking dates", flush=True)


def run(refill):
    """refill: 'monthly' | 'weekly' | 'daily' — when empty slots get refilled from cash."""
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    pre, post, idle = [], [], []
    st = dict(refills=0, buys=0)

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d):
        nonlocal cash, tax
        v, b, bd = held.pop(s)
        if v > b and (d - bd).days < 365:
            tax += (v - b) * STCG
        cash += v * (1 - RT)

    def do_refill(d, full_rerank):
        """full_rerank=True at month-end (rotate out of buffer + refill).
        Otherwise only FILL EMPTY SLOTS with the best un-owned names."""
        nonlocal cash
        etf = SNAP.get(d)
        if not etf:
            return
        if full_rerank:
            buf = set(etf[:BUFFER])
            for s in list(held):
                if s not in buf:
                    sell(s, d)
        empty = N_HOLD - len(held)
        if empty <= 0 or cash <= 0:
            return
        cands = [s for s in etf[:N_HOLD] if s not in held]
        if not full_rerank:
            # weekly/daily refill may need to look past the top-8 to find un-owned names
            cands = [s for s in etf if s not in held]
        cands = [s for s in cands if s in close.columns and pd.notna(close.at[d, s])][:empty]
        if not cands:
            return
        nav = E() + cash
        per = nav / N_HOLD
        for s in cands:
            buy = min(per, cash)
            if buy <= 0:
                break
            held[s] = [buy * (1 - RT), buy, d]; cash -= buy; st['buys'] += 1
        st['refills'] += 1

    for d in idx:
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
        risk_off = False
        if d in WK:
            b = NBX.loc[:d].dropna()
            risk_off = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if risk_off:
                for s in list(held):
                    sell(s, d)
                derisked = True
            else:
                derisked = False
        if not derisked:
            if d in ME:
                do_refill(d, True)                       # full monthly re-rank
            elif refill == "weekly" and d in WK and d in SNAP:
                do_refill(d, False)                      # top up empty slots
            elif refill == "daily" and len(held) < N_HOLD:
                dd = max([x for x in SNAP if x <= d], default=None)
                if dd:
                    etf = SNAP[dd]
                    empty = N_HOLD - len(held)
                    cands = [s for s in etf if s not in held
                             and s in close.columns and pd.notna(close.at[d, s])][:empty]
                    nav = E() + cash; per = nav / N_HOLD
                    for s in cands:
                        buy = min(per, cash)
                        if buy <= 0:
                            break
                        held[s] = [buy * (1 - RT), buy, d]; cash -= buy; st['buys'] += 1
                    if cands:
                        st['refills'] += 1
        nav = E() + cash
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
    c1, d1, s1, k1 = blk(pre); c2, d2, s2, k2 = blk(post)
    return dict(cagr=round(c1, 1), maxdd=round(d1, 1), sharpe=round(s1, 2), calmar=round(k1, 2),
                net_cagr=round(c2, 1), net_calmar=round(k2, 2),
                idle_pct=round(100 * float(np.mean(idle)), 1), refills=st['refills'], buys=st['buys'])


print(f"\n{'refill policy':>26} {'netCAGR':>8} {'MaxDD':>7} {'Sharpe':>7} {'netCal':>7} {'avg idle':>9} {'buys':>6}")
out = {}
for r, lbl in (("monthly", "MONTHLY (current live)"), ("weekly", "WEEKLY refill"), ("daily", "DAILY refill")):
    x = run(r); out[r] = x
    print(f"{lbl:>26} {x['net_cagr']:>7.1f}% {x['maxdd']:>6.1f}% {x['sharpe']:>7.2f} "
          f"{x['net_calmar']:>7.2f} {x['idle_pct']:>8.1f}% {x['buys']:>6}", flush=True)
w = csv.DictWriter(open(RES / "redeploy.csv", "w", newline=""),
                   fieldnames=["policy"] + list(out["monthly"].keys()))
w.writeheader()
for k, v in out.items():
    w.writerow(dict(policy=k, **v))
b = max(out.items(), key=lambda kv: kv[1]['net_calmar'])
print(f"\nBEST by net Calmar: {b[0]} ({b[1]['net_calmar']}) vs current monthly ({out['monthly']['net_calmar']})")
print(f"cash drag reduced from {out['monthly']['idle_pct']}% to {out['weekly']['idle_pct']}% (weekly)")
