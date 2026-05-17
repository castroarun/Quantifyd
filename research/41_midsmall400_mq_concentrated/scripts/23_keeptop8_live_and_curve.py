"""Phase 23 — adopt keep-top8 into SMOOTHEST: month-end-engine NAV
(for the heatmap/equity-curve comparison) + today's 15 live holdings.

SMOOTHEST-KT8 = SMOOTHEST core (PIT mid-cap band, RS-120 vs NIFTYBEES,
q0.5, ATH≤10% entry, above-own-100SMA, N=15, top-22 buffer, monthly
rotation, per-stock-100SMA + 12% trail) but on a risk-off month-end
(Nifty < its 100-SMA) it KEEPS the 8 highest-RS holdings and cashes the
weaker 7 (vs SMOOTHEST's all-to-cash); no refill while risk-off; full
refill when risk-on. Computed on the SAME month-end engine the app
heatmap/equity-curve already use (p11/p10/p13 family) so the new column/
curve is directly comparable. 0.4% RT, 6.5% cash, 2014→2026.

Part B: today's would-be 15 holdings = the system's actual selection on
the latest available data (PIT mid band + RS-120 + q0.5 + ATH≤10% +
above-own-100SMA, ranked), with the regime state and, if risk-off, the
keep-top8 subset flagged. Laptop = frozen snapshot; rerun on VPS for a
true today-dated list.
"""
from __future__ import annotations
import importlib.util, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_s = importlib.util.spec_from_file_location(
    "rs2", str(Path(__file__).resolve().parent / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR, TRAIL, LSMA = 15, 0.50, 0.10, 0.12, 100
BUF = int(N * rs2.BUFFER_MULT)
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)


def own_pf(cs):
    if len(cs) < 120:
        return None
    bl = [cs.iloc[i:i + 21] for i in range(0, len(cs) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def backtest(close, tv, riskoff="cash", stcg=0.0):
    """riskoff: 'cash' = SMOOTHEST base; 'keeptop8' = adopt KT8.
    Month-end engine (regime + selection both at month-ends), matching
    the p11 family used by the app heatmap/curve."""
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp("2014-01-01")) &
            (me <= pd.Timestamp("2026-12-31"))]
    cash, eq = 1.0, 0.0
    held, last = {}, {}            # sym -> [val, cost, buydt, peak]
    nav = []
    for dt in me:
        h = close.loc[:dt]; px = h.iloc[-1]
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

        def _real(stt):
            nonlocal tot
            g = stt[0] - stt[1]
            if stcg and g > 0 and (dt - stt[2]).days < 365:
                tot -= g * stcg

        # stock-level exits (both modes)
        for s in list(held):
            cs = h[s].dropna(); p1 = px.get(s, np.nan)
            if (len(cs) >= LSMA and cs.iloc[-1] < cs.tail(LSMA).mean()) or \
               (pd.notna(p1) and held[s][3] > 0 and
                    p1 <= (1 - TRAIL) * held[s][3]):
                _real(held[s]); cash += held.pop(s)[0]; eq = sum(
                    v[0] for v in held.values())

        b = h[BENCH].dropna()
        roff = len(b) >= LSMA and b.iloc[-1] < b.tail(LSMA).mean()
        sc = rs2.rs_scores(close, dt, 120)

        if roff:
            if riskoff == "cash":
                for s in list(held):
                    _real(held[s])
                cash, eq, held, last = tot, 0.0, {}, {}
                nav.append((dt, tot)); continue
            else:                                  # keeptop8
                if held and sc is not None and len(held) > 8:
                    rk = {s: sc.get(s, -9e9) for s in held}
                    keep = sorted(rk, key=rk.get, reverse=True)[:8]
                    for s in list(held):
                        if s not in keep:
                            _real(held[s]); cash += held.pop(s)[0]
                # no refill while risk-off; rescale equity stays as-is
                eq = sum(v[0] for v in held.values())
                last = {s: px[s] for s in held}
                nav.append((dt, eq + (tot - eq if not held else
                            cash * 0 + (tot - eq))))
                cash = tot - eq
                continue

        # risk-on: full SMOOTHEST selection / refill
        uni = rs2.pit_universe(tv, close, dt, "mid")
        if sc is None or not uni:
            nav.append((dt, tot)); continue
        cand = [s for s in sc.index if s in uni and s != BENCH]
        scc = sc[cand].sort_values(ascending=False)
        pick = []
        for s in scc.index:
            cs = h[s].dropna()
            pf = own_pf(cs.tail(252))
            if pf is None or pf < Q:
                continue
            if len(cs) < LSMA or cs.iloc[-1] < cs.tail(LSMA).mean():
                continue
            if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                continue
            pick.append(s)
            if len(pick) >= BUF:
                break
        if not pick:
            nav.append((dt, tot)); continue
        top = pick[:N]; bufset = set(pick[:BUF])
        keepn = [s for s in held if s in bufset]
        fill = [s for s in top if s not in keepn]
        tgt = [s for s in (keepn + fill)[:N] if pd.notna(px.get(s, np.nan))]
        if not tgt:
            nav.append((dt, tot)); continue
        drop = [s for s in held if s not in set(tgt)]
        for s in drop:
            _real(held[s])
        turn = len(set(held).symmetric_difference(set(tgt))) / max(1, 2 * N)
        tot *= (1 - rs2.RT_COST * turn * 2)
        w = tot / len(tgt)
        nh = {}
        for s in tgt:
            nh[s] = ([w, held[s][1], held[s][2], held[s][3]]
                     if (s in held and s not in drop) else [w, w, dt, px[s]])
        held = nh; last = {s: px[s] for s in tgt}
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


def live_holstings(close, tv):
    asof = close.index.max()
    h = close.loc[:asof]; px = h.iloc[-1]
    uni = rs2.pit_universe(tv, close, asof, "mid")
    sc = rs2.rs_scores(close, asof, 120)
    b = h[BENCH].dropna()
    sma100 = b.tail(100).mean()
    roff = b.iloc[-1] < sma100
    rows = []
    for s in sc.sort_values(ascending=False).index:
        if s == BENCH or s not in uni:
            continue
        cs = h[s].dropna()
        pf = own_pf(cs.tail(252))
        if pf is None or pf < Q:
            continue
        if len(cs) < 100 or cs.iloc[-1] < cs.tail(100).mean():
            continue
        athv = cs.max()
        if px.get(s, np.nan) < (1 - ATH_THR) * athv:
            continue
        rows.append(dict(symbol=s, rs=round(float(sc[s]), 3),
                         pct_from_ATH=round((px[s] / athv - 1) * 100, 1),
                         posfrac=round(pf, 2),
                         last_close=round(float(px[s]), 2),
                         above_own_100sma="yes"))
        if len(rows) >= N:
            break
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df, asof, roff, float(b.iloc[-1]), float(sma100)


def last_broad_date(close, min_syms=200):
    """Last date with broad universe coverage AND a benchmark print.
    Two snapshot quirks bite here: (1) trailing rows are 2-symbol index
    stubs (real stock data ends 2026-02-17), and (2) NIFTYBEES — the RS
    benchmark — trades on a sparser grid and its last print is
    2026-02-16, NOT 2026-02-17. rs_scores divides by the benchmark
    ratio, so a date missing NIFTYBEES yields an all-NaN (empty) RS
    series. Require BOTH gates. On the VPS (live data) this resolves to
    the true latest trading day where stocks + NIFTYBEES both print."""
    cnt = close.notna().sum(axis=1)
    bench_ok = close[rs2.NIFTY_SYM].notna() if rs2.NIFTY_SYM in close else True
    good = close.index[(cnt >= min_syms) & bench_ok]
    return good.max() if len(good) else close.index.max()


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    good = last_broad_date(close)
    print(f"Clipping to last broad-coverage date: {good.date()} "
          f"(snapshot tail past this is index-stub rows)", flush=True)
    close = close.loc[:good]
    tv = tv.loc[:good]
    base = backtest(close, tv, "cash")
    kt8 = backtest(close, tv, "keeptop8")
    bt = backtest(close, tv, "cash", stcg=0.20)
    kt = backtest(close, tv, "keeptop8", stcg=0.20)
    print("\n=== SMOOTHEST base vs SMOOTHEST-KT8 (month-end engine) ===")
    print(f"  SMOOTHEST(cash)  CAGR={base['cagr']:.1f}% net20="
          f"{bt['cagr']:.1f}% DD={base['dd']:.1f}% Sh={base['sh']:.2f} "
          f"Cal={base['cal']:.2f}")
    print(f"  SMOOTHEST-KT8    CAGR={kt8['cagr']:.1f}% net20="
          f"{kt['cagr']:.1f}% DD={kt8['dd']:.1f}% Sh={kt8['sh']:.2f} "
          f"Cal={kt8['cal']:.2f}")
    base["nav"].to_csv(RES / "nav_smoothest_base.csv")
    kt8["nav"].to_csv(RES / "nav_smoothest_kt8.csv")
    pd.DataFrame({"SMOOTHEST": base["py"], "SMOOTHEST-KT8": kt8["py"]}
                 ).to_csv(RES / "phase23_kt8_peryear.csv")

    df, asof, roff, nbx, sma = live_holstings(close, tv)
    df.to_csv(RES / "live_holdings_today.csv")
    print(f"\n=== TODAY'S 15 HOLDINGS — SMOOTHEST-KT8 ===")
    print(f"As-of (snapshot): {asof.date()}  [laptop frozen — rerun on "
          f"VPS for true today]")
    print(f"Regime: NIFTYBEES {nbx:.2f} vs 100-SMA {sma:.2f} -> "
          f"{'RISK-OFF -> hold only the top-8 below' if roff else 'RISK-ON -> hold all 15'}")
    print(df.to_string())
    if roff:
        print("\nKT8 subset (top-8 by RS, held in risk-off):")
        print(df.head(8)[["symbol", "rs", "pct_from_ATH",
                          "last_close"]].to_string())


if __name__ == "__main__":
    main()
