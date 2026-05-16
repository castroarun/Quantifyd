"""Phase 11 — stock-level risk control vs the market-level SMA100 gate.

Phase-09 showed the 20% trailing stop was INERT (RS top-22 buffer
rotates losers out before -20%). So: (a) test BINDING tighter
stock-level trailing stops (15/12/10%); (b) a per-stock SMA100 trend
filter (hold a name only while its close > its OWN 100d SMA). Question:
can stock-level control REPLACE the market Nifty-SMA100 regime, or is
the combination better?

Core selection held constant = mid_120d_N15 + q0.5 + ATH<=10% entry
(Phase-09 winning stack). Only the risk-control layer changes.
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


def mkt_sma100_off(h):
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


def backtest(close, tv, market_gate, per_stock_sma, trail, stcg=0.0):
    """market_gate: True -> Nifty SMA100 risk-off to cash.
       per_stock_sma: True -> only hold a name while close>own SMA100.
       trail: 0 disables; else drop a name if it falls trail below its
              peak-since-entry (binding stock-level stop)."""
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp("2014-01-01")) &
            (me <= pd.Timestamp("2026-12-31"))]
    cash, eq = 1.0, 0.0
    held, last = {}, {}            # sym -> [val, cost, buydate, peakpx]
    nav, tax = [], 0.0
    for dt in me:
        h = close.loc[:dt]
        px = h.iloc[-1]
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

        # binding trailing stop (stock-level)
        if trail and held:
            for s in list(held):
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and held[s][3] > 0 and \
                   p1 <= (1 - trail) * held[s][3]:
                    _realize(held.pop(s))

        # per-stock SMA100 trend exit (drop names below own SMA100)
        if per_stock_sma and held:
            for s in list(held):
                cs = h[s].dropna()
                if len(cs) >= 100 and cs.iloc[-1] < cs.tail(100).mean():
                    _realize(held.pop(s))

        if market_gate and mkt_sma100_off(h):
            if held:
                for s in list(held):
                    _realize(held[s])
            cash, eq, held, last = tot, 0.0, {}, {}
            nav.append((dt, tot)); continue

        uni = rs2.pit_universe(tv, close, dt, "mid")
        sc = rs2.rs_scores(close, dt, 120)
        if sc is None or not uni:
            nav.append((dt, tot)); continue
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
            if per_stock_sma and (len(cs) < 100 or
                                  cs.iloc[-1] < cs.tail(100).mean()):
                continue                      # only buy uptrending names
            pick.append(s)
            if len(pick) >= BUF:
                break
        if not pick:
            cash, eq, held, last = tot, 0.0, {}, {}
            nav.append((dt, tot)); continue
        top = pick[:N]; buf = set(pick[:BUF])
        keep = [s for s in held if s in buf]
        fill = [s for s in top if s not in keep]
        tgt = [s for s in (keep + fill)[:N]
               if pd.notna(px.get(s, np.nan))]
        if not tgt:
            nav.append((dt, tot)); continue
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
        nav.append((dt, tot))
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
                py=(py * 100).round(1), nav=n)


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    cfgs = [
        ("SMA100 mkt (Ph09 winner)",  True,  False, 0.0),
        ("OFF + trail15",             False, False, 0.15),
        ("OFF + trail12",             False, False, 0.12),
        ("OFF + trail10",             False, False, 0.10),
        ("OFF + perStockSMA100",      False, True,  0.0),
        ("OFF + perStockSMA + tr12",  False, True,  0.12),
        ("perStockSMA only (no mkt)", False, True,  0.0),
        ("SMA100 mkt + perStockSMA",  True,  True,  0.0),
        ("SMA100 mkt + trail12",      True,  False, 0.12),
        ("SMA100 + perStock + tr12",  True,  True,  0.12),
    ]
    rows, py = [], {}
    for lbl, mg, ps, tr in cfgs:
        m = backtest(close, tv, mg, ps, tr)
        mt = backtest(close, tv, mg, ps, tr, stcg=0.20)
        rows.append(dict(config=lbl, market_gate=mg, perStockSMA=ps,
                         trail=tr, cagr=round(m['cagr'], 1),
                         post_tax20=round(mt['cagr'], 1),
                         maxdd=round(m['dd'], 1),
                         sharpe=round(m['sh'], 2),
                         calmar=round(m['cal'], 2)))
        py[lbl] = m['py']
        print(f"  {lbl:28} CAGR={m['cagr']:5.1f}% net20={mt['cagr']:5.1f}%"
              f" DD={m['dd']:6.1f}% Sh={m['sh']:.2f} Cal={m['cal']:.2f}",
              flush=True)
    pd.DataFrame(rows).to_csv(RES / "phase11_stocklevel.csv", index=False)
    pt = pd.DataFrame({k: py[k] for k in py}).reindex(range(2014, 2027))
    pt.to_csv(RES / "phase11_peryear.csv")
    print("\n=== PER-YEAR % ===\n" + pt.to_string())
    print("\nRef: Ph09 SMA100+ATH = 35.2 gross / 29.3 net20 / DD -15.1 / "
          "Cal 2.33. VERDICT: does any stock-level-only config match/beat "
          "that Calmar WITHOUT the market gate?")


if __name__ == "__main__":
    main()
