"""Phase 22 — SMOOTHEST de-risk variants (user-proposed).

Daily-marked engine, WEEKLY regime check, monthly stock selection.
Core = SMOOTHEST: PIT mid-cap band, RS-120 vs NIFTYBEES, q0.5,
ATH≤10% entry, N=15, top-22 buffer, monthly RS rotation, per-stock
own-SMA exit (length L_sma) + 12% trail at month-ends. 0.4% round-trip,
6.5% cash, 2014-2026. Market gate = Nifty vs its 100-day SMA (separate
from the per-stock SMA length).

Risk-off action `mode`:
  allcash  : liquidate 100% to cash            (= SMOOTHEST baseline)
  none     : NO market gate at all (only stock-level exits + RS rot)
  trim<f>  : ONE-SHOT — hold (1-f) invested while risk-off, restore
             to full when risk-on (NOT staggered-to-zero)
  keeptop<K>: keep only the top-K current holdings by RS, rest→cash;
             refill to 15 when risk-on
`L_sma` : per-stock trend-exit SMA length (100 base; 80/60 = tighter).
Each: gross + post-tax@20% + per-year + daily MaxDD.
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
RT = rs2.RT_COST
START = pd.Timestamp("2014-01-01")


def own_pf(cs):
    if len(cs) < 120:
        return None
    bl = [cs.iloc[i:i + 21] for i in range(0, len(cs) - 1, 21)]
    r = [x.iloc[-1] / x.iloc[0] - 1 for x in bl
         if len(x) >= 2 and x.iloc[0] > 0]
    return float(np.mean([z > 0 for z in r])) if r else 0.0


def run(close, tv, mode, f=0.0, K=0, L_sma=100, stcg=0.0):
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk = pd.Series(idx, index=idx)
    wk_last = set(wk.groupby([idx.isocalendar().year,
                              idx.isocalendar().week]).last().values)
    cash, held, last = 1.0, {}, {}     # sym->[val,cost,buydt,peak]
    tgt = 1.0
    nav, tax, prev = [], 0.0, None

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d):
        nonlocal tax
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; tax += t; return t
        return 0.0

    def scale_to(frac, d):              # pro-rata reduce equity→frac*total
        nonlocal cash
        eq = E(); tot = eq + cash
        want = frac * tot
        if eq > want + 1e-9 and eq > 0:
            k = want / eq
            for s, st in list(held.items()):
                sv = st[0] * (1 - k)
                t = realize([sv, st[1] * (1 - k), st[2], st[3]], d)
                st[0] *= k; st[1] *= k
                cash += sv - t
                if st[0] < 1e-9:
                    held.pop(s)
            cash -= (eq - want) * RT

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

        # ---- WEEKLY regime action ----
        if mode != "none" and d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if mode == "allcash":
                if roff and held:
                    for s in list(held):
                        t = realize(held[s], d); cash += held.pop(s)[0] - t
                    tgt = 0.0
                elif not roff:
                    tgt = 1.0
            elif mode == "trim":
                tgt = (1.0 - f) if roff else 1.0
                if roff:
                    scale_to(tgt, d)
            elif mode == "keeptop":
                if roff and len(held) > K:
                    sc = rs2.rs_scores(close, d, 120)
                    if sc is not None:
                        rk = {s: sc.get(s, -9e9) for s in held}
                        keep = sorted(rk, key=rk.get, reverse=True)[:K]
                        for s in list(held):
                            if s not in keep:
                                t = realize(held[s], d)
                                cash += held.pop(s)[0] - t
                    tgt = 0.0 if not held else 1.0
                elif not roff:
                    tgt = 1.0

        # ---- MONTHLY selection ----
        if d in me:
            for s in list(held):                 # stock-level exits
                cs = h[s].dropna(); p1 = px.get(s, np.nan)
                if (len(cs) >= L_sma and
                        cs.iloc[-1] < cs.tail(L_sma).mean()) or \
                   (pd.notna(p1) and held[s][3] > 0 and
                        p1 <= (1 - TRAIL) * held[s][3]):
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
            do_fill = True
            if mode == "allcash" and tgt == 0.0:
                do_fill = False
            if mode == "keeptop":
                b = h[BENCH].dropna()
                if len(b) >= 100 and b.iloc[-1] < b.tail(100).mean():
                    do_fill = False              # stay at K while risk-off
            if do_fill:
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
                        if len(cs) < L_sma or \
                           cs.iloc[-1] < cs.tail(L_sma).mean():
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
                    inv = (tgt if mode in ("trim", "allcash") else 1.0) * tot
                    names = ([s for s in held] +
                             [s for s in elig[:N] if s not in held])[:N]
                    if names:
                        w = inv / len(names)
                        turn = (len(set(held).symmetric_difference(
                            set(names))) / max(1, 2 * N))
                        nh = {}
                        for s in names:
                            nh[s] = (held[s] if s in held
                                     else [0, 0, d, px[s]])
                            nh[s] = [w, (nh[s][1] if s in held else w),
                                     (nh[s][2] if s in held else d),
                                     (nh[s][3] if s in held else px[s])]
                        held = nh
                        cash = tot - inv
                        cash -= inv * RT * turn * 2
        nav.append((d, E() + cash))
        prev = d

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
                py=(py * 100).round(1), nav=n)


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    cfgs = [
        ("BASE SMOOTHEST(allcash)", "allcash", 0.0, 0, 100),
        ("A no-regime",             "none",    0.0, 0, 100),
        ("B trim-25 (hold 75)",     "trim",    0.25, 0, 100),
        ("B trim-50 (hold 50)",     "trim",    0.50, 0, 100),
        ("C keep-top5",             "keeptop", 0.0, 5, 100),
        ("C keep-top8",             "keeptop", 0.0, 8, 100),
        ("D perstock-SMA80",        "allcash", 0.0, 0, 80),
        ("D perstock-SMA60",        "allcash", 0.0, 0, 60),
    ]
    rows, pys = [], {}
    for lbl, m, f, K, L in cfgs:
        g = run(close, tv, m, f, K, L)
        t = run(close, tv, m, f, K, L, stcg=0.20)
        rows.append(dict(config=lbl, cagr=round(g['cagr'], 1),
                         post_tax20=round(t['cagr'], 1),
                         maxdd=round(g['dd'], 1), sharpe=round(g['sh'], 2),
                         calmar=round(g['cal'], 2)))
        pys[lbl] = g['py']
        print(f"  {lbl:26} CAGR={g['cagr']:5.1f}% net20={t['cagr']:5.1f}%"
              f" DD={g['dd']:6.1f}% Sh={g['sh']:.2f} Cal={g['cal']:.2f}",
              flush=True)
    pd.DataFrame(rows).to_csv(RES / "phase22_variants.csv", index=False)
    pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(
        RES / "phase22_variants_peryear.csv")
    print("\n=== SMOOTHEST VARIANTS ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print("\nRef: SMOOTHEST all-at-once weekly (daily engine) ~Cal 1.5 / "
          "DD ~-23. VERDICT: a variant wins only if it beats BASE on "
          "Calmar without material post-tax CAGR loss.")


if __name__ == "__main__":
    main()
