"""Phase 33 — MA-overlay exit variants (user: 'don't exit all stocks, add the
MA-overlay exit check on each stock'). Three mechanisms, L in {100,150,200}:

  V1 alwayson_nogate : NO Nifty gate. Every week (bull & bear) any held stock
                       closing < its own L-MA is sold; freed slots refill at
                       MONTH-END only (NOT weekly -> avoids the Phase-28 churn).
  V2 alwayson_gate   : Keep the Nifty gate (keep-top8 trim on bear) AND run the
                       always-on per-stock L-MA exit on top, every week. Month-end
                       refill gated on risk-on.
  V3 bearonly_hold15 : Only in bear, hold all 15 (no trim), exit names breaching
                       L-MA. = Phase 32 Version B, re-run here for a clean table.

Refs: keep-top8 (1.66), all-cash base (1.54), Phase-32 A-L100 (1.70).
Core: PIT mid band, RS-120, q0.5, ATH<=10% & above-own-L-MA entry, N=15, top-22
buffer, +12% trail at month-ends, 0.4% RT, 6.5% cash, 2014-2026, daily-marked.
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
_q = importlib.util.spec_from_file_location(
    "p32", str(SCR / "32_bear_conditional_ma_exit.py"))
p32 = importlib.util.module_from_spec(_q); _q.loader.exec_module(p32)

BENCH = rs2.NIFTY_SYM
N, Q, ATH_THR, TRAIL = 15, 0.50, 0.10, 0.12
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
    st = dict(tax=0.0, cost=0.0, ma_exit=0, keep_ev=0, mfills=0)
    use_gate = mode in ("V2", "V3")

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

        if d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if mode == "V2" and roff and len(held) > 8:     # gate keep-top8 trim
                sc = rs2.rs_scores(close, d, 120)
                if sc is not None:
                    rk = {s: sc.get(s, -9e9) for s in held}
                    keep = sorted(rk, key=rk.get, reverse=True)[:8]
                    for s in list(held):
                        if s not in keep:
                            t = realize(held[s], d); cash += held.pop(s)[0] - t
                    st['keep_ev'] += 1
            # per-stock MA exit: V1/V2 always; V3 only in bear
            if mode in ("V1", "V2") or (mode == "V3" and roff):
                for s in list(held):
                    cs = h[s].dropna()
                    if len(cs) >= L and cs.iloc[-1] < cs.tail(L).mean():
                        t = realize(held[s], d); cash += held.pop(s)[0] - t
                        st['ma_exit'] += 1

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
        print(f"  {lbl:28} CAGR={row['cagr']:5.1f}% net20={row['post_tax20']:5.1f}%"
              f" DD={row['maxdd']:6.1f}% Sh={row['sharpe']:.2f}"
              f" Cal={row['calmar']:.2f} {extra}", flush=True)
        pd.DataFrame(rows).to_csv(RES / "phase33_ma_overlay.csv", index=False)

    emit("REF keep-top8", p22.run(close, tv, "keeptop", 0.0, 8, 100),
         p22.run(close, tv, "keeptop", 0.0, 8, 100, stcg=0.20), "[1.66]")
    emit("REF all-cash base", p22.run(close, tv, "allcash", 0.0, 0, 100),
         p22.run(close, tv, "allcash", 0.0, 0, 100, stcg=0.20), "[1.54]")
    emit("REF Ph32 A-L100", p32.run(close, tv, "A", 100),
         p32.run(close, tv, "A", 100, stcg=0.20), "[1.70]")

    for mode, name in [("V1", "V1 alwayson NOgate"),
                       ("V2", "V2 alwayson +gate"),
                       ("V3", "V3 bearonly hold15")]:
        for L in (100, 150, 200):
            ts = time.time()
            g = run(close, tv, mode, L)
            t = run(close, tv, mode, L, stcg=0.20)
            emit(f"{name} L{L}", g, t,
                 f"maX={g['st']['ma_exit']} [{time.time()-ts:.0f}s]")

    print("\n=== MA-OVERLAY VARIANTS ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print(f"\nTotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
