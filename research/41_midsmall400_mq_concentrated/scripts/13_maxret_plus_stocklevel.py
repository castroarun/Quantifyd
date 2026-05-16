"""Phase 13 — COMBINED: Max-Return beta-hedge + Smoothest stock-level exits.

User ask: take MAX-RETURN (SMA100 regime; in risk-off DON'T go to cash —
keep the stocks and short 1x Nifty) and ALSO add SMOOTHEST's two
stock-level exits (per-stock 100-SMA exit + 12% trailing stop). Does the
hybrid beat either parent?

Core (constant): mid PIT band, RS-120 vs NIFTYBEES, N=15, monthly,
top-22 buffer, q0.5 quality, ATH<=10% entry, 0.4% RT, 6.5% cash,
2014-2026. Per-stock-SMA100 is BOTH an entry filter and a monthly exit;
12% trailing stop is a monthly exit. Regime = NIFTYBEES vs its 100-SMA;
risk-off => keep (post-stop) book + short 1.0x Nifty for that month.

Monthly order: mark-to-mkt -> 12% trail exits -> per-stock-SMA exits ->
compute regime + (if risk-off) apply -1.0*Nifty hedge on equity ->
universe -> RS rank -> keep q0.5 & ATH<=10% & price>=own100SMA ->
top-22 buffer/fill to 15 -> turnover cost -> equal weight.
Reports gross, post-tax@20%, per-year return, per-year max-DD, vs the
two parents.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
SC = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("rs2", str(SC / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR, TRAIL, HR = 15, 0.50, 0.10, 0.12, 1.0
BUF = int(N * rs2.BUFFER_MULT)
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)


def sma_off(h, n=100):
    b = h[BENCH].dropna()
    return len(b) >= n and b.iloc[-1] < b.tail(n).mean()


def own_pf(h, s):
    ss = h[s].dropna().tail(252)
    if len(ss) < 120:
        return None
    bl = [ss.iloc[i:i + 21] for i in range(0, len(ss) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def backtest(close, tv, hedge_in_riskoff, stcg=0.0):
    """hedge_in_riskoff True = COMBINED (beta-hedge); False = parent
    Smoothest (cash in risk-off) — for sanity parity check."""
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp("2014-01-01")) &
            (me <= pd.Timestamp("2026-12-31"))]
    cash, eq = 1.0, 0.0
    held, last = {}, {}            # sym -> [val, cost, buydate, peakpx]
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
                if pd.notna(p1):
                    st[3] = max(st[3], p1)
                ne += st[0]
            eq = ne
        tot = cash * CASH_M + eq

        def _realize(stt):
            nonlocal tot, tax
            g = stt[0] - stt[1]
            if stcg and g > 0 and (dt - stt[2]).days < 365:
                tot -= g * stcg; tax += g * stcg

        # 1) 12% trailing-stop exits
        for s in list(held):
            p1 = px.get(s, np.nan)
            if pd.notna(p1) and held[s][3] > 0 and \
               p1 <= (1 - TRAIL) * held[s][3]:
                _realize(held.pop(s))
        # 2) per-stock SMA100 exits
        for s in list(held):
            cs = h[s].dropna()
            if len(cs) >= 100 and cs.iloc[-1] < cs.tail(100).mean():
                _realize(held.pop(s))

        risk_off = sma_off(h)
        # 3) regime action
        if risk_off and not hedge_in_riskoff:                 # parent: cash
            for s in list(held):
                _realize(held[s])
            cash, eq, held, last = tot, 0.0, {}, {}
            nav.append((dt, tot)); prev_nb = nb_now; continue
        if risk_off and hedge_in_riskoff and prev_nb is not None \
           and eq > 0:                                        # COMBINED hedge
            tot += eq * (-HR * nf_ret)

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
            cs = h[s].dropna()
            if len(cs) < 60 or px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                continue
            if len(cs) < 100 or cs.iloc[-1] < cs.tail(100).mean():
                continue                                      # entry: uptrend
            pick.append(s)
            if len(pick) >= BUF:
                break
        if not pick:
            if not hedge_in_riskoff:
                for s in list(held):
                    _realize(held[s])
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
        for s in drop:
            _realize(held[s])
        turn = len(set(held).symmetric_difference(set(tgt))) / max(1, 2 * N)
        tot *= (1 - rs2.RT_COST * turn * 2)
        w = tot / len(tgt)
        nh = {}
        for s in tgt:
            if s in held and s not in drop:
                nh[s] = [w, held[s][1], held[s][2], held[s][3]]
            else:
                nh[s] = [w, w, dt, px[s]]
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
    pyr = n.groupby(n.index.year).last().pct_change()
    pyr.iloc[0] = n.groupby(n.index.year).last().iloc[0] / 1.0 - 1
    pdd = {}
    for y, g in n.groupby(n.index.year):
        rm = g.cummax(); pdd[y] = round(((g - rm) / rm).min() * 100, 1)
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                pyr=(pyr * 100).round(1), pdd=pd.Series(pdd))


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print("COMBINED (beta-hedge + per-stock-SMA + 12% trail) ...",
          flush=True)
    cmb = backtest(close, tv, True)
    cmbt = backtest(close, tv, True, stcg=0.20)
    print("Parity: Smoothest (cash + stock-level) ...", flush=True)
    smo = backtest(close, tv, False)

    print("\n=== COMBINED vs the two parents (post-tax @20% STCG) ===")
    print(f"{'System':36}{'CAGR':>7}{'net20':>7}{'MaxDD':>8}"
          f"{'Sharpe':>7}{'Calmar':>7}")
    print(f"{'SMOOTHEST (Ph11: cash+stocklvl)':36}"
          f"{'35.6':>7}{'29.6':>7}{'-15.1':>8}{'1.80':>7}{'2.36':>7}")
    print(f"{'MAX-RETURN (Ph10: hedge,no stoplvl)':36}"
          f"{'42.8':>7}{'34.0':>7}{'-22.7':>8}{'1.83':>7}{'1.89':>7}")
    print(f"{'COMBINED (hedge + stocklvl)':36}"
          f"{cmb['cagr']:>7.1f}{cmbt['cagr']:>7.1f}{cmb['dd']:>8.1f}"
          f"{cmb['sh']:>7.2f}{cmb['cal']:>7.2f}")
    print(f"{'  (parity chk: my Smoothest)':36}"
          f"{smo['cagr']:>7.1f}{'-':>7}{smo['dd']:>8.1f}"
          f"{smo['sh']:>7.2f}{smo['cal']:>7.2f}")

    yrs = list(range(2014, 2027))
    t = pd.DataFrame(index=yrs)
    t["COMBINED_ret%"] = cmb["pyr"].reindex(yrs)
    t["COMBINED_maxDD%"] = cmb["pdd"].reindex(yrs)
    t.to_csv(RES / "phase13_combined.csv")
    print("\n=== COMBINED per-year ===\n" + t.fillna("n/a").to_string())
    print("\nVERDICT RULE: combined wins only if it beats SMOOTHEST on "
          "Calmar OR beats MAX-RETURN on post-tax CAGR at <= its DD. "
          "Else it's a dominated middle.")


if __name__ == "__main__":
    main()
