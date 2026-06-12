"""Phase 28 — EVENT-DRIVEN per-stock trailing-MA exit, with/without Nifty gate.

User idea: trail each stock by its own 100/150/200 MA; the week it closes below,
sell it and IMMEDIATELY backfill the slot from the next passing RS candidate
(above its own MA). Test standalone (no market gate) AND on top of the Nifty
100-SMA gate. v1 exit = close-below-MA (optimal-exit refinements are G2).

Prior art: Phase 11 showed per-stock-SMA100 at MONTH-ENDS, standalone = −30.2%
DD / Calmar 1.10. New axis here = EVENT-DRIVEN weekly cadence + immediate
backfill + longer MAs. REF config reproduces Phase 27/22 BASE (34.2/−22.2) via
the trusted p22 engine = sanity anchor.

Core: PIT mid band, RS-120 vs NIFTYBEES, q0.5, ATH<=10% & above-own-L-MA entry,
N=15, top-22 buffer, monthly RS rotation, per-stock-L-MA + 12% trail at month-ends,
0.4% RT, 6.5% cash, 2014-2026, daily-marked.
"""
from __future__ import annotations
import importlib.util, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
SCR = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("rs2", str(SCR / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rs2)
_p = importlib.util.spec_from_file_location(
    "p22", str(SCR / "22_smoothest_variants.py"))
p22 = importlib.util.module_from_spec(_p); _p.loader.exec_module(p22)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR, TRAIL = 15, 0.50, 0.10, 0.12
BUF = int(N * rs2.BUFFER_MULT)
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
RT = rs2.RT_COST
START = pd.Timestamp("2014-01-01")
own_pf = p22.own_pf


def _metrics(nav, st):
    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    py = n.groupby(n.index.year).last().pct_change()
    py.iloc[0] = n.groupby(n.index.year).last().iloc[0] / n.iloc[0] - 1
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                py=(py * 100).round(1), st=st, nav=n)


def run(close, tv, L, use_gate, stcg=0.0):
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last().values)
    cash, held = 1.0, {}
    nav, prev = [], None
    st = dict(tax=0.0, cost=0.0, stock_exit=0, gate_exit=0, backfill=0, mfills=0)

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
        return 0.0

    def eligibles(d, h, px):
        uni = rs2.pit_universe(tv, close, d, "mid")
        sc = rs2.rs_scores(close, d, 120)
        out = []
        if sc is not None and uni:
            for s in sc.sort_values(ascending=False).index:
                if s == BENCH or s not in uni:
                    continue
                cs = h[s].dropna()
                pf = own_pf(cs.tail(252))
                if pf is None or pf < Q:
                    continue
                if len(cs) < L or cs.iloc[-1] < cs.tail(L).mean():
                    continue
                if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                    continue
                out.append(s)
                if len(out) >= BUF:
                    break
        return out

    def monthly_fill(d, px, h):
        nonlocal cash, held
        elig = eligibles(d, h, px)
        if not elig:
            return
        bufset = set(elig[:BUF])
        for s in list(held):
            if s not in bufset:
                t = realize(held[s], d); cash += held.pop(s)[0] - t
        tot = E() + cash
        names = ([s for s in held] +
                 [s for s in elig[:N] if s not in held])[:N]
        if not names:
            return
        w = tot / len(names)
        turn = (len(set(held).symmetric_difference(set(names)))
                / max(1, 2 * N))
        nh = {}
        for s in names:
            base = held[s] if s in held else [0, 0, d, px[s]]
            nh[s] = [w, (base[1] if s in held else w),
                     (base[2] if s in held else d),
                     (base[3] if s in held else px[s])]
        held = nh
        c = tot * RT * turn * 2
        cash = -c                       # fully deployed; cost drag (matches p22)
        st['cost'] += c; st['mfills'] += 1

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    stt[0] *= p1 / p0
                    stt[3] = max(stt[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]

        # ---- WEEKLY: market gate (optional) + per-stock event exit + backfill ----
        if d in wk_last:
            roff = False
            if use_gate:
                b = h[BENCH].dropna()
                roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
                if roff and held:
                    for s in list(held):
                        t = realize(held[s], d); cash += held.pop(s)[0] - t
                    st['gate_exit'] += 1
            if not roff:
                # per-stock event exit (close below own L-MA)
                for s in list(held):
                    cs = h[s].dropna()
                    if len(cs) >= L and cs.iloc[-1] < cs.tail(L).mean():
                        t = realize(held[s], d)
                        cash += held.pop(s)[0] - t
                        st['stock_exit'] += 1
                # immediate one-for-one backfill of freed slots
                empties = N - len(held)
                if empties > 0 and cash > 1e-9:
                    cands = [s for s in eligibles(d, h, px)
                             if s not in held][:empties]
                    if cands:
                        share = cash / empties      # per-slot budget
                        bought = share * len(cands)
                        c = bought * RT
                        w = (bought - c) / len(cands)
                        for s in cands:
                            held[s] = [w, w, d, px[s]]
                        cash -= bought
                        st['cost'] += c; st['backfill'] += 1

        # ---- MONTHLY: per-stock-L-MA + trail exits, then buffer fill ----
        if d in me:
            for s in list(held):
                cs = h[s].dropna(); p1 = px.get(s, np.nan)
                if (len(cs) >= L and cs.iloc[-1] < cs.tail(L).mean()) or \
                   (pd.notna(p1) and held[s][3] > 0 and
                        p1 <= (1 - TRAIL) * held[s][3]):
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
            do_fill = True
            if use_gate:
                b = h[BENCH].dropna()
                if len(b) >= 100 and b.iloc[-1] < b.tail(100).mean():
                    do_fill = False
            if do_fill:
                monthly_fill(d, px, h)
        nav.append((d, E() + cash))
        prev = d

    return _metrics(nav, st)


def main():
    t0 = time.time()
    print("Loading market data ...", flush=True)
    close, tv = rs2.load()
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    rows, pys = [], {}

    # REF anchor from the trusted p22 engine (locked allcash + gate)
    rg = p22.run(close, tv, "allcash", 0.0, 0, 100)
    rt = p22.run(close, tv, "allcash", 0.0, 0, 100, stcg=0.20)
    rows.append(dict(config="REF allcash+gate (locked)", cagr=round(rg['cagr'], 1),
                     post_tax20=round(rt['cagr'], 1), maxdd=round(rg['dd'], 1),
                     sharpe=round(rg['sh'], 2), calmar=round(rg['cal'], 2),
                     stock_exit=0, backfill=0, gate_exit=0))
    pys["REF allcash+gate (locked)"] = rg['py']
    r = rows[-1]
    print(f"  {'REF allcash+gate (locked)':30} CAGR={r['cagr']:5.1f}% "
          f"net20={r['post_tax20']:5.1f}% DD={r['maxdd']:6.1f}% "
          f"Cal={r['calmar']:.2f}  [SANITY ~34.2/-22.2]", flush=True)

    cfgs = [
        ("perStock L100  NO gate", 100, False),
        ("perStock L150  NO gate", 150, False),
        ("perStock L200  NO gate", 200, False),
        ("perStock L100  +gate",   100, True),
        ("perStock L150  +gate",   150, True),
        ("perStock L200  +gate",   200, True),
    ]
    for lbl, L, ug in cfgs:
        ts = time.time()
        g = run(close, tv, L, ug)
        t = run(close, tv, L, ug, stcg=0.20)
        row = dict(config=lbl, cagr=round(g['cagr'], 1),
                   post_tax20=round(t['cagr'], 1), maxdd=round(g['dd'], 1),
                   sharpe=round(g['sh'], 2), calmar=round(g['cal'], 2),
                   stock_exit=g['st']['stock_exit'],
                   backfill=g['st']['backfill'], gate_exit=g['st']['gate_exit'])
        rows.append(row); pys[lbl] = g['py']
        print(f"  {lbl:30} CAGR={row['cagr']:5.1f}% net20={row['post_tax20']:5.1f}%"
              f" DD={row['maxdd']:6.1f}% Sh={row['sharpe']:.2f}"
              f" Cal={row['calmar']:.2f} stkX={row['stock_exit']:>3}"
              f" bf={row['backfill']:>3} [{time.time() - ts:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(RES / "phase28_perstock.csv", index=False)
        pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(
            RES / "phase28_perstock_peryear.csv")

    print("\n=== PER-STOCK TRAILING-MA (event-driven) ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print(f"\nTotal {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
