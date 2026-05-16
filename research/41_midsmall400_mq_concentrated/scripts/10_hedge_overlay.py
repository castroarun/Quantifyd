"""Phase 10 — drawdown hedge overlay vs the Phase-09 winner.

Phase-09 best = mid_120d_N15 + q0.5 + SMA100 regime + ATH<=10% entry
(35.2% gross / 29.3% post-tax / -15.1% DD / Calmar 2.33). The hedge
question: in risk-off, instead of going to CASH (throwing away RS
stock-alpha), KEEP the stocks but SHORT a Nifty notional -> retain
stock-specific alpha, neutralise market beta. Also test a PERMANENT
partial Nifty short with no regime gate (addresses "OFF has higher
CAGR but -30% DD — can a constant hedge tame it?").

Hedge model (no options data needed): in a hedged month the book
return r_p = r_stock - hr * r_nifty (short NIFTYBEES notional =
hr * equity). Short carry (~riskfree - div) IGNORED -> slightly
overstates hedge cost in an up-drifting market = conservative.
Covered calls deliberately NOT modelled (caps the right-tail that is
the CAGR; mid-cap holdings mostly lack liquid options).
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_s = importlib.util.spec_from_file_location(
    "rs2", str(Path(__file__).resolve().parent / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR = 15, 0.50, 0.10
BUF = int(N * rs2.BUFFER_MULT)
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)


def sma100_off(h):
    b = h[BENCH].dropna()
    return len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()


def own_pf(h, s):
    ss = h[s].dropna().tail(252)
    if len(ss) < 120:
        return None
    bl = [ss.iloc[i:i + 21] for i in range(0, len(ss) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def backtest(close, tv, mode, hr=0.0, stcg=0.0):
    """mode: 'cash'  = SMA100 risk-off -> 6.5% cash (Phase-09 winner)
             'beta'  = SMA100 risk-off -> stay in stocks, short hr*Nifty
             'always'= no gate; every month short hr*Nifty (permanent)
             'off'   = no gate, no hedge (reference)"""
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp("2014-01-01")) &
            (me <= pd.Timestamp("2026-12-31"))]
    cash, eq = 1.0, 0.0
    held, last = {}, {}
    nav, tax = [], 0.0
    prev_nb = None
    for dt in me:
        h = close.loc[:dt]
        px = h.iloc[-1]
        nb = h[BENCH].dropna()
        nb_now = nb.iloc[-1] if len(nb) else np.nan
        nf_ret = ((nb_now / prev_nb - 1)
                  if prev_nb and pd.notna(nb_now) else 0.0)

        if held:
            ne = 0.0
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and s in last and last[s] > 0:
                    st[0] *= (1 + p1 / last[s] - 1)
                ne += st[0]
            eq = ne
        tot = cash * CASH_M + eq

        risk_off = (mode in ("cash", "beta")) and sma100_off(h)
        hedged_this = (mode == "always") or (mode == "beta" and risk_off)

        # apply short-Nifty overlay on the prior month's move
        if hedged_this and prev_nb is not None and eq > 0:
            tot += eq * (-hr * nf_ret)        # short hr*equity of Nifty

        if mode == "cash" and risk_off:
            if stcg and held:
                for s, st in held.items():
                    g = st[0] - st[1]
                    if g > 0 and (dt - st[2]).days < 365:
                        tot -= g * stcg; tax += g * stcg
            cash, eq, held, last = tot, 0.0, {}, {}
            nav.append((dt, tot)); prev_nb = nb_now; continue

        uni = rs2.pit_universe(tv, close, dt, "mid")
        sc = rs2.rs_scores(close, dt, 120)
        if sc is None or not uni:
            nav.append((dt, tot)); prev_nb = nb_now; continue
        cand = [s for s in sc.index if s in uni and s != BENCH]
        sc = sc[cand].sort_values(ascending=False)
        pick = []
        for s in sc.index:
            pf = own_pf(h, s)
            if pf is None or pf < Q:
                continue
            hh = h[s].dropna()
            if len(hh) < 60 or px.get(s, np.nan) < (1 - ATH_THR) * hh.max():
                continue
            pick.append(s)
            if len(pick) >= BUF:
                break
        if not pick:
            cash, eq, held, last = tot, 0.0, {}, {}
            nav.append((dt, tot)); prev_nb = nb_now; continue
        top = pick[:N]; buf = set(pick[:BUF])
        keep = [s for s in held if s in buf]
        fill = [s for s in top if s not in keep]
        tgt = [s for s in (keep + fill)[:N]
               if pd.notna(px.get(s, np.nan))]
        if not tgt:
            nav.append((dt, tot)); prev_nb = nb_now; continue
        drop = [s for s in held if s not in set(tgt)]
        if stcg:
            for s in drop:
                st = held[s]; g = st[0] - st[1]
                if g > 0 and (dt - st[2]).days < 365:
                    tot -= g * stcg; tax += g * stcg
        turn = len(set(held).symmetric_difference(set(tgt))) / max(1, 2 * N)
        tot *= (1 - rs2.RT_COST * turn * 2)
        w = tot / len(tgt)
        nh = {}
        for s in tgt:
            nh[s] = ([w, held[s][1], held[s][2]]
                     if (s in held and s not in drop) else [w, w, dt])
        held = nh
        last = {s: px[s] for s in tgt}
        cash, eq = 0.0, tot
        nav.append((dt, tot)); prev_nb = nb_now
    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    mr = n.pct_change().dropna()
    sh = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0
    py = n.groupby(n.index.year).last().pct_change()
    py.iloc[0] = n.groupby(n.index.year).last().iloc[0] / 1.0 - 1
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                py=(py * 100).round(1))


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    cfgs = [
        ("SMA100->cash (Ph09 best)", "cash", 0.0),
        ("OFF no-hedge (ref)",       "off",  0.0),
        ("SMA100->beta hr1.0",       "beta", 1.0),
        ("SMA100->beta hr0.5",       "beta", 0.5),
        ("always hedge hr0.25",      "always", 0.25),
        ("always hedge hr0.40",      "always", 0.40),
        ("always hedge hr0.60",      "always", 0.60),
    ]
    rows, py = [], {}
    for lbl, mode, hr in cfgs:
        m = backtest(close, tv, mode, hr)
        mt = backtest(close, tv, mode, hr, stcg=0.20)
        rows.append(dict(config=lbl, cagr=round(m['cagr'], 1),
                         post_tax20=round(mt['cagr'], 1),
                         maxdd=round(m['dd'], 1),
                         sharpe=round(m['sh'], 2),
                         calmar=round(m['cal'], 2)))
        py[lbl] = m['py']
        print(f"  {lbl:26} CAGR={m['cagr']:5.1f}% net20={mt['cagr']:5.1f}%"
              f" DD={m['dd']:6.1f}% Sh={m['sh']:.2f} Cal={m['cal']:.2f}",
              flush=True)
    pd.DataFrame(rows).to_csv(RES / "phase10_hedge.csv", index=False)
    pt = pd.DataFrame({k: py[k] for k in py}).reindex(range(2014, 2027))
    pt.to_csv(RES / "phase10_hedge_peryear.csv")
    print("\n=== PER-YEAR % ===\n" + pt.to_string())
    print("\nRef: Ph09 SMA100->cash = 35.1/net29.5/DD-16.4/Cal2.14 ; "
          "+ATH entry 35.2/29.3/-15.1/2.33 ; OFF 37.0/30.9/-29.6/1.25")
    print("VERDICT RULE: a hedge wins only if it beats SMA100->cash on "
          "Calmar AND keeps post-tax CAGR >= ~29%. Covered calls NOT "
          "modelled (rejected — caps the CAGR tail; holdings illiquid).")


if __name__ == "__main__":
    main()
