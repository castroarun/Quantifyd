"""Phase 15 — decoupled regime clock: monthly selection, faster gate.

User critique: the regime gate fires only at month-end, so by the time
Nifty is < its 100-SMA we may already be deep in a drop -> late hedge/
exit. Fix: keep STOCK selection + rotation MONTHLY (and the per-stock-
SMA / 12% trail exits at month-ends, unchanged), but evaluate the
MARKET REGIME GATE daily (or weekly) — act the day NIFTYBEES closes
below its 100-day SMA, not at the next month-end.

Single DAILY-marked engine (true daily MaxDD — stricter & more honest
than the month-end NAV used earlier). 3 systems x 3 regime cadences:

  SMOOTHEST  = risk-off -> liquidate to cash; per-stock-SMA + 12% trail
               at month-ends; re-enter only at a risk-on month-end.
  MAX-RETURN = risk-off -> hold stocks + short 1x Nifty (daily overlay);
               no stock-level stops; rotate monthly.
  FORTIFIED  = MAX-RETURN + per-stock-SMA + 12% trail at month-ends.

  cadence M = regime checked only at month-ends (baseline, reproduces
              the earlier monthly behaviour on this daily engine)
  cadence W = checked weekly (last bar each ISO week)
  cadence D = checked every trading day

Core (constant): mid PIT band, RS-120 vs NIFTYBEES, q0.5, ATH<=10%
entry, N=15, top-22 buffer, 0.4% RT/turnover, 6.5% p.a. cash,
2014-2026. Reports CAGR, daily-MaxDD, Sharpe, Calmar, post-tax@20%,
and regime-flip count (whipsaw proxy).
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
N, Q, ATH_THR, TRAIL = 15, 0.50, 0.10, 0.12
BUF = int(N * rs2.BUFFER_MULT)
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
START = pd.Timestamp("2014-01-01")


def own_pf(hist_tail):
    ss = hist_tail
    if len(ss) < 120:
        return None
    bl = [ss.iloc[i:i + 21] for i in range(0, len(ss) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def run(close, tv, mode, stocklevel, cadence, stcg=0.0):
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    nb = close[BENCH].reindex(close.index).ffill()
    sma100 = nb.rolling(100).mean()
    # weekly check days = last trading bar of each ISO week
    wk = pd.Series(idx, index=idx)
    wk_last = set(wk.groupby([idx.isocalendar().year,
                              idx.isocalendar().week]).last().values)

    cash, held = 1.0, {}          # held: sym -> [val, cost, buydt, peak]
    last_px = {}
    hedge_on, in_cash = False, False
    nav, tax, flips = [], 0.0, 0
    prev_regime_off = False
    prev_d = None

    def realize(stt, d):
        nonlocal tax
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            tax_amt = g * stcg
            return tax_amt
        return 0.0

    def equity():
        return sum(v[0] for v in held.values())

    for d in idx:
        px = close.loc[d]
        # ---- daily mark of holdings ----
        if held and prev_d is not None:
            for s, st in held.items():
                p1, p0 = px.get(s, np.nan), close.at[prev_d, s] \
                    if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    st[0] *= p1 / p0
                    st[3] = max(st[3], p1)
        eq0 = equity()
        # ---- daily short-Nifty overlay when hedged ----
        if hedge_on and prev_d is not None and eq0 > 0:
            nf0, nf1 = nb.get(prev_d, np.nan), nb.get(d, np.nan)
            if pd.notna(nf0) and pd.notna(nf1) and nf0 > 0:
                # short 1x equity: pnl = -(nifty daily ret) * equity
                hedge_pnl = -(nf1 / nf0 - 1) * eq0
                # distribute pnl proportionally onto holdings' values
                tot_now = equity()
                if tot_now > 0:
                    for st in held.values():
                        st[0] += hedge_pnl * (st[0] / tot_now)
        if cash > 0:
            cash *= DAY_CASH

        # ---- regime evaluation on this cadence ----
        check = (cadence == "D" or (cadence == "W" and d in wk_last)
                 or (cadence == "M" and d in me))
        if check and pd.notna(sma100.get(d, np.nan)):
            roff = nb[d] < sma100[d]
            if roff != prev_regime_off:
                flips += 1
            prev_regime_off = roff
            if mode == "cash":
                if roff and held:                     # exit to cash NOW
                    tot = equity() + cash
                    for s, st in list(held.items()):
                        tot -= 0  # value already in equity()
                    t = sum(realize(st, d) for st in held.values())
                    cash = equity() + cash - t
                    tax += t
                    held, last_px, in_cash = {}, {}, True
            else:  # beta
                hedge_on = roff

        # ---- monthly stock selection / rotation ----
        if d in me:
            h = close.loc[:d]
            roff_now = (pd.notna(sma100.get(d, np.nan)) and
                        nb[d] < sma100[d])
            tot = equity() + cash
            do_select = True
            if mode == "cash" and roff_now:
                do_select = False                     # stay in cash
            if do_select:
                # stock-level exits first (SMOOTHEST/FORTIFIED)
                if stocklevel and held:
                    for s in list(held):
                        cs = h[s].dropna()
                        p1 = px.get(s, np.nan)
                        hit_tr = (pd.notna(p1) and held[s][3] > 0 and
                                  p1 <= (1 - TRAIL) * held[s][3])
                        hit_sma = (len(cs) >= 100 and
                                   cs.iloc[-1] < cs.tail(100).mean())
                        if hit_tr or hit_sma:
                            t = realize(held[s], d); tax += t
                            cash += held.pop(s)[0] - t
                uni = rs2.pit_universe(tv, close, d, "mid")
                sc = rs2.rs_scores(close, d, 120)
                if sc is not None and uni:
                    cand = [s for s in sc.index
                            if s in uni and s != BENCH]
                    sc = sc[cand].sort_values(ascending=False)
                    pick = []
                    for s in sc.index:
                        cs = h[s].dropna()
                        pf = own_pf(cs.tail(252))
                        if pf is None or pf < Q:
                            continue
                        if len(cs) < 60 or px.get(s, np.nan) < \
                           (1 - ATH_THR) * cs.max():
                            continue
                        if stocklevel and (len(cs) < 100 or
                           cs.iloc[-1] < cs.tail(100).mean()):
                            continue
                        pick.append(s)
                        if len(pick) >= BUF:
                            break
                    if pick:
                        tot = equity() + cash
                        top, bufset = pick[:N], set(pick[:BUF])
                        keep = [s for s in held if s in bufset]
                        fill = [s for s in top if s not in keep]
                        tgt = [s for s in (keep + fill)[:N]
                               if pd.notna(px.get(s, np.nan))]
                        if tgt:
                            drop = [s for s in held
                                    if s not in set(tgt)]
                            for s in drop:
                                t = realize(held[s], d); tax += t
                                cash += held.pop(s)[0] - t
                            turn = (len(set(held).symmetric_difference(
                                set(tgt))) / max(1, 2 * N))
                            tot = equity() + cash
                            tot *= (1 - rs2.RT_COST * turn * 2)
                            w = tot / len(tgt)
                            nh = {}
                            for s in tgt:
                                if s in held and s not in drop:
                                    nh[s] = [w, held[s][1],
                                             held[s][2], held[s][3]]
                                else:
                                    nh[s] = [w, w, d, px[s]]
                            held, cash, in_cash = nh, 0.0, False
        last_px = {s: px.get(s, np.nan) for s in held}
        nav.append((d, equity() + cash))
        prev_d = d

    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                flips=flips, navlen=len(n))


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    SYS = [("SMOOTHEST", "cash", True),
           ("MAX-RETURN", "beta", False),
           ("FORTIFIED", "beta", True)]
    rows = []
    for sname, mode, sl in SYS:
        for cad in ("M", "W", "D"):
            g = run(close, tv, mode, sl, cad)
            t = run(close, tv, mode, sl, cad, stcg=0.20)
            rows.append(dict(system=sname, regime=cad,
                             cagr=round(g['cagr'], 1),
                             post_tax20=round(t['cagr'], 1),
                             maxdd_daily=round(g['dd'], 1),
                             sharpe=round(g['sh'], 2),
                             calmar=round(g['cal'], 2),
                             regime_flips=g['flips']))
            print(f"  {sname:11} {cad}  CAGR={g['cagr']:5.1f}% "
                  f"net20={t['cagr']:5.1f}% dailyDD={g['dd']:6.1f}% "
                  f"Sh={g['sh']:.2f} Cal={g['cal']:.2f} "
                  f"flips={g['flips']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "phase15_decoupled_regime.csv", index=False)
    print("\n=== PHASE 15 TABLE ===\n" + df.to_string(index=False))
    print("\nNOTE: MaxDD here is DAILY-measured (stricter than the "
          "month-end MaxDD in Ph09-13; numbers not directly comparable "
          "to those — compare WITHIN this table across M/W/D). "
          "VERDICT: faster regime helps if D/W cut daily-MaxDD "
          "materially vs M without killing CAGR or exploding flips.")


if __name__ == "__main__":
    main()
