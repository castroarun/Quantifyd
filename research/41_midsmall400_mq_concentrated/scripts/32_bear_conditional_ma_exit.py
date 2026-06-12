"""Phase 32 — REGIME-CONDITIONAL per-stock MA-breach exit (two versions).

Unlike Phase 28 (per-stock MA exit ALWAYS on -> churn/tax-ruinous), here the
per-stock L-MA exit fires ONLY while the market regime is bear (NIFTYBEES < its
100-SMA). Two versions:

  A_keeptop8+bearMA : on bear, keep top-8 by RS (sell weak 7) AND additionally
                      exit any held stock that breaches its own L-MA. Progressive
                      pruning of the survivors as the bear deepens.
  B_nosell+bearMA   : on bear, sell NOTHING wholesale; only exit held stocks that
                      breach their own L-MA. A soft gate that de-risks name-by-name.

Both: re-entry (refill to 15) only at a risk-ON month-end (keep-top8 convention).
Core unchanged: PIT mid band, RS-120, q0.5, ATH<=10% entry, N=15, top-22 buffer,
monthly rotation, per-stock-SMA100 + 12% trail at month-ends, 0.4% RT, 6.5% cash,
2014-2026, daily-marked. L in {100,150,200} = the BEAR-time per-stock MA only.
Refs: keep-top8 and all-cash base via trusted p22.
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
N, Q, ATH_THR, TRAIL, CORE_SMA = 15, 0.50, 0.10, 0.12, 100
BUF = int(N * rs2.BUFFER_MULT)
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
RT = rs2.RT_COST
START = pd.Timestamp("2014-01-01")
own_pf = p22.own_pf


def run(close, tv, mode, L, stcg=0.0):
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last().values)
    cash, held = 1.0, {}
    nav, prev = [], None
    st = dict(tax=0.0, cost=0.0, keep_ev=0, ma_exit=0, mfills=0)

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
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
                if len(cs) < CORE_SMA or cs.iloc[-1] < cs.tail(CORE_SMA).mean():
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
        cash = -c
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

        # ---- WEEKLY: only act while regime is BEAR ----
        if d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if roff:
                if mode == "A" and len(held) > 8:      # keep top-8 by RS
                    sc = rs2.rs_scores(close, d, 120)
                    if sc is not None:
                        rk = {s: sc.get(s, -9e9) for s in held}
                        keep = sorted(rk, key=rk.get, reverse=True)[:8]
                        for s in list(held):
                            if s not in keep:
                                t = realize(held[s], d)
                                cash += held.pop(s)[0] - t
                        st['keep_ev'] += 1
                # bear-time per-stock L-MA breach exit (both A and B)
                for s in list(held):
                    cs = h[s].dropna()
                    if len(cs) >= L and cs.iloc[-1] < cs.tail(L).mean():
                        t = realize(held[s], d)
                        cash += held.pop(s)[0] - t
                        st['ma_exit'] += 1

        # ---- MONTHLY: locked core exits + (gated) refill ----
        if d in me:
            for s in list(held):
                cs = h[s].dropna(); p1 = px.get(s, np.nan)
                if (len(cs) >= CORE_SMA and
                        cs.iloc[-1] < cs.tail(CORE_SMA).mean()) or \
                   (pd.notna(p1) and held[s][3] > 0 and
                        p1 <= (1 - TRAIL) * held[s][3]):
                    t = realize(held[s], d); cash += held.pop(s)[0] - t
            b = h[BENCH].dropna()
            roff_me = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if not roff_me:
                monthly_fill(d, px, h)
        nav.append((d, E() + cash))
        prev = d

    n = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    dr = n.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan, st=st)


def main():
    t0 = time.time()
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  loaded {time.time()-t0:.0f}s", flush=True)
    rows = []

    def emit(lbl, g, t, extra=""):
        row = dict(config=lbl, cagr=round(g['cagr'], 1),
                   post_tax20=round(t['cagr'], 1), maxdd=round(g['dd'], 1),
                   sharpe=round(g['sh'], 2), calmar=round(g['cal'], 2))
        rows.append(row)
        print(f"  {lbl:30} CAGR={row['cagr']:5.1f}% net20={row['post_tax20']:5.1f}%"
              f" DD={row['maxdd']:6.1f}% Sh={row['sharpe']:.2f}"
              f" Cal={row['calmar']:.2f} {extra}", flush=True)
        pd.DataFrame(rows).to_csv(RES / "phase32_bear_ma.csv", index=False)

    emit("REF keep-top8 (locked)",
         p22.run(close, tv, "keeptop", 0.0, 8, 100),
         p22.run(close, tv, "keeptop", 0.0, 8, 100, stcg=0.20), "[ref 1.66]")
    emit("REF all-cash base",
         p22.run(close, tv, "allcash", 0.0, 0, 100),
         p22.run(close, tv, "allcash", 0.0, 0, 100, stcg=0.20), "[ref 1.54]")

    for mode, name in [("A", "A keeptop8+bearMA"), ("B", "B nosell+bearMA")]:
        for L in (100, 150, 200):
            ts = time.time()
            g = run(close, tv, mode, L)
            t = run(close, tv, mode, L, stcg=0.20)
            emit(f"{name} L{L}", g, t,
                 f"maX={g['st']['ma_exit']} keepEv={g['st']['keep_ev']}"
                 f" [{time.time()-ts:.0f}s]")

    print("\n=== BEAR-CONDITIONAL PER-STOCK MA EXIT ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print(f"\nTotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
