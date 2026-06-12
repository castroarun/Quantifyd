"""Phase 34 — per-year + sub-period robustness for the two NAMED finalists:
'Keep-8 + Bear Trend-Trim' (=Ph32 A-L100) and 'Always-On Trend-Guard' (=V2-L100),
vs keep-top8 and Nifty 50. Unified engine (returns nav) so all share one ruler.
Outputs per-year CSV, sub-period CSV, and a yearly-vs-Nifty heatmap PNG.
"""
from __future__ import annotations
import importlib.util, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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


def run(close, tv, mode, L=100, stcg=0.0):
    """mode: 'A' (bear-only keep8+MA), 'V2' (always-on MA+gate), 'KT8' (keep8 only)."""
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last().values)
    cash, held = 1.0, {}
    nav, prev = [], None
    tax = [0.0]

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; tax[0] += t; return t
        return 0.0

    def monthly_fill(d, px, h):
        nonlocal cash, held
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
                if len(cs) < L or cs.iloc[-1] < cs.tail(L).mean():
                    continue
                if px.get(s, np.nan) < (1 - ATH_THR) * cs.max():
                    continue
                elig.append(s)
                if len(elig) >= BUF:
                    break
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
        turn = (len(set(held).symmetric_difference(set(names))) / max(1, 2 * N))
        nh = {}
        for s in names:
            base = held[s] if s in held else [0, 0, d, px[s]]
            nh[s] = [w, (base[1] if s in held else w),
                     (base[2] if s in held else d),
                     (base[3] if s in held else px[s])]
        held = nh
        cash = -tot * RT * turn * 2

    for d in idx:
        px = close.loc[d]
        if held and prev is not None:
            for s, stt in held.items():
                p1 = px.get(s, np.nan)
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    stt[0] *= p1 / p0; stt[3] = max(stt[3], p1)
        if cash > 0:
            cash *= DAY_CASH
        h = close.loc[:d]

        if d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if roff and len(held) > 8:                 # gate keep-top8 trim (all modes)
                sc = rs2.rs_scores(close, d, 120)
                if sc is not None:
                    rk = {s: sc.get(s, -9e9) for s in held}
                    keep = sorted(rk, key=rk.get, reverse=True)[:8]
                    for s in list(held):
                        if s not in keep:
                            t = realize(held[s], d); cash += held.pop(s)[0] - t
            do_ma = (mode == "A" and roff) or (mode == "V2")
            if do_ma:
                for s in list(held):
                    cs = h[s].dropna()
                    if len(cs) >= L and cs.iloc[-1] < cs.tail(L).mean():
                        t = realize(held[s], d); cash += held.pop(s)[0] - t

        if d in me:
            for s in list(held):
                cs = h[s].dropna(); p1 = px.get(s, np.nan)
                if (len(cs) >= L and cs.iloc[-1] < cs.tail(L).mean()) or \
                   (pd.notna(p1) and held[s][3] > 0 and p1 <= (1 - TRAIL) * held[s][3]):
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
            b = h[BENCH].dropna()
            if not (len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()):
                monthly_fill(d, px, h)
        nav.append((d, E() + cash))
        prev = d

    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    return n


def metrics(n, lo=None, hi=None):
    if lo is not None:
        n = n[(n.index >= lo) & (n.index <= hi)]
    n = n.dropna()
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    return cg * 100, dd * 100, sh


def peryear(n):
    yl = n.groupby(n.index.year).last(); py = yl.pct_change()
    py.iloc[0] = yl.iloc[0] / n.iloc[0] - 1
    return (py * 100).round(1)


def main():
    t0 = time.time()
    print("Loading ...", flush=True); close, tv = rs2.load()
    print(f"  loaded {time.time()-t0:.0f}s", flush=True)
    idx = close.index[close.index >= START]

    navs = {}
    print("  Keep-8 + Bear Trend-Trim ...", flush=True)
    navs["Keep-8 + Bear Trend-Trim"] = run(close, tv, "A", 100)
    print("  Always-On Trend-Guard ...", flush=True)
    navs["Always-On Trend-Guard"] = run(close, tv, "V2", 100)
    print("  keep-top8 ...", flush=True)
    navs["keep-top8"] = run(close, tv, "KT8", 100)
    bser = close[BENCH].reindex(idx).ffill().dropna()
    navs["Nifty 50"] = bser / bser.iloc[0]
    order = ["Keep-8 + Bear Trend-Trim", "Always-On Trend-Guard", "keep-top8", "Nifty 50"]

    py = pd.DataFrame({k: peryear(navs[k]) for k in order}).reindex(range(2014, 2027))
    py.to_csv(RES / "phase34_peryear.csv")
    print("\n=== PER-YEAR (%) ===\n" + py.to_string())
    for k in order[:-1]:
        w = int((py[k] > py["Nifty 50"]).sum()); tot = int(py[k].notna().sum())
        print(f"  {k:26} beats Nifty50 {w}/{tot} yrs", flush=True)

    wins = [("Full", START, idx[-1]),
            ("H1 14-19", pd.Timestamp("2014-01-01"), pd.Timestamp("2019-12-31")),
            ("H2 20-26", pd.Timestamp("2020-01-01"), idx[-1]),
            ("T1 14-17", pd.Timestamp("2014-01-01"), pd.Timestamp("2017-12-31")),
            ("T2 18-21", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")),
            ("T3 22-26", pd.Timestamp("2022-01-01"), idx[-1])]
    rows = []
    for k in order:
        d = dict(variant=k)
        for wl, lo, hi in wins:
            cg, dd, sh = metrics(navs[k], lo, hi)
            d[wl] = f"{cg:.1f}/{dd:.0f}/{sh:.2f}"
        rows.append(d)
    sp = pd.DataFrame(rows); sp.to_csv(RES / "phase34_subperiod.csv", index=False)
    print("\n=== SUB-PERIOD (CAGR/MaxDD/Sharpe) ===\n" + sp.to_string(index=False))

    yrs = list(range(2014, 2027))
    M = py.T.reindex(order)[yrs].values.astype(float)
    fig, ax = plt.subplots(figsize=(14, 4.2))
    norm = TwoSlopeNorm(vmin=np.nanmin(M), vcenter=0, vmax=np.nanmax(M))
    ax.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(yrs))); ax.set_xticklabels(yrs)
    full = {k: metrics(navs[k])[0] for k in order}
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{k}\n(CAGR {full[k]:.1f}%)" for k in order], fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=9)
    ax.set_title("Named finalists — yearly returns vs Nifty 50 (gross, daily-marked, 2014-2026)\n"
                 "Keep-8 + Bear Trend-Trim (recommended) vs Always-On Trend-Guard", fontsize=12)
    plt.tight_layout()
    fig.savefig(RES / "phase34_finalists_heatmap.png", dpi=130)
    print("\nSaved phase34_finalists_heatmap.png")
    print(f"Total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
