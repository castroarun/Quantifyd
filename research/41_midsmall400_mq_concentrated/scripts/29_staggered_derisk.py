"""Phase 29 — Gradual de-risk bake-off: staggered chunk-out vs keep-top8 vs 1-shot.

User wants a de-risk that does NOT dump the whole book to debt in one move. Put
every gradual candidate on ONE daily-marked engine:
  - BASE allcash (1-shot to cash)        -> via trusted p22 engine (anchor)
  - keep-top8 (partial: keep 8, cash 7)  -> via trusted p22 engine
  - STAGGER chunk c: each weekly risk-off bar scale the book down by another c of
    total (c=0.25 -> ~4 wks to flat); stop scaling once risk-on; full redeploy at
    the next monthly rebalance (slow re-entry). c=1.0 == allcash (parity sanity).

Core: PIT mid band, RS-120, q0.5, ATH<=10% entry, N=15, top-22 buffer,
per-stock-100SMA +12% trail at month-ends, weekly 100-SMA gate, 0.4% RT, 6.5%
cash, 2014-2026, daily-marked. Only the risk-off ACTION changes.
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
N, Q, ATH_THR, TRAIL, L_SMA = 15, 0.50, 0.10, 0.12, 100
BUF = int(N * rs2.BUFFER_MULT)
DAY_CASH = (1 + rs2.CASH_ANNUAL) ** (1 / 252)
RT = rs2.RT_COST
START = pd.Timestamp("2014-01-01")
own_pf = p22.own_pf


def run_stagger(close, tv, chunk, stcg=0.0):
    """Cumulative weekly scale-out by `chunk` of total while risk-off; slow
    (month-end) re-entry to full when risk-on. chunk=1.0 == allcash."""
    idx = close.index[close.index >= START]
    me = set(rs2.month_ends(close.index))
    wk_last = set(pd.Series(idx, index=idx).groupby(
        [idx.isocalendar().year, idx.isocalendar().week]).last().values)
    cash, held = 1.0, {}
    tgt = 1.0
    nav, prev = [], None
    st = dict(tax=0.0, cost=0.0, scale_ev=0, mfills=0)

    def E():
        return sum(v[0] for v in held.values())

    def realize(stt, d):
        g = stt[0] - stt[1]
        if stcg and g > 0 and (d - stt[2]).days < 365:
            t = g * stcg; st['tax'] += t; return t
        return 0.0

    def scale_to(frac, d):
        nonlocal cash
        eq = E(); tot = eq + cash
        want = frac * tot
        if eq > want + 1e-9 and eq > 0:
            k = want / eq
            for s, stt in list(held.items()):
                sv = stt[0] * (1 - k)
                t = realize([sv, stt[1] * (1 - k), stt[2], stt[3]], d)
                stt[0] *= k; stt[1] *= k
                cash += sv - t
                if stt[0] < 1e-9:
                    held.pop(s)
            cash -= (eq - want) * RT
            st['cost'] += (eq - want) * RT
            st['scale_ev'] += 1

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
                if len(cs) < L_SMA or cs.iloc[-1] < cs.tail(L_SMA).mean():
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
        inv = tgt * tot
        names = ([s for s in held] +
                 [s for s in elig[:N] if s not in held])[:N]
        if not names:
            return
        w = inv / len(names)
        turn = (len(set(held).symmetric_difference(set(names)))
                / max(1, 2 * N))
        nh = {}
        for s in names:
            base = held[s] if s in held else [0, 0, d, px[s]]
            nh[s] = [w, (base[1] if s in held else w),
                     (base[2] if s in held else d),
                     (base[3] if s in held else px[s])]
        held = nh
        c = inv * RT * turn * 2
        cash = tot - inv - c
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

        # ---- WEEKLY: staggered scale-out while risk-off ----
        if d in wk_last:
            b = h[BENCH].dropna()
            roff = len(b) >= 100 and b.iloc[-1] < b.tail(100).mean()
            if roff:
                tgt = max(0.0, tgt - chunk)
                scale_to(tgt, d)
            else:
                tgt = 1.0          # permit full re-entry at month-end

        # ---- MONTHLY: stock exits + (gated) refill to tgt ----
        if d in me:
            for s in list(held):
                cs = h[s].dropna(); p1 = px.get(s, np.nan)
                if (len(cs) >= L_SMA and cs.iloc[-1] < cs.tail(L_SMA).mean()) or \
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
    py = n.groupby(n.index.year).last().pct_change()
    py.iloc[0] = n.groupby(n.index.year).last().iloc[0] / n.iloc[0] - 1
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan,
                py=(py * 100).round(1), st=st)


def main():
    t0 = time.time()
    print("Loading market data ...", flush=True)
    close, tv = rs2.load()
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)
    rows, pys = [], {}

    def emit(lbl, g, t, extra=""):
        row = dict(config=lbl, cagr=round(g['cagr'], 1),
                   post_tax20=round(t['cagr'], 1), maxdd=round(g['dd'], 1),
                   sharpe=round(g['sh'], 2), calmar=round(g['cal'], 2))
        rows.append(row); pys[lbl] = g['py']
        print(f"  {lbl:30} CAGR={row['cagr']:5.1f}% net20={row['post_tax20']:5.1f}%"
              f" DD={row['maxdd']:6.1f}% Sh={row['sharpe']:.2f}"
              f" Cal={row['calmar']:.2f} {extra}", flush=True)
        pd.DataFrame(rows).to_csv(RES / "phase29_staggered.csv", index=False)
        pd.DataFrame(pys).reindex(range(2014, 2027)).to_csv(
            RES / "phase29_staggered_peryear.csv")

    # anchors from trusted p22
    emit("BASE allcash (anchor)",
         p22.run(close, tv, "allcash", 0.0, 0, 100),
         p22.run(close, tv, "allcash", 0.0, 0, 100, stcg=0.20),
         "[SANITY ~34.2/-22.2]")
    emit("keep-top8 (p22)",
         p22.run(close, tv, "keeptop", 0.0, 8, 100),
         p22.run(close, tv, "keeptop", 0.0, 8, 100, stcg=0.20),
         "[ref 1.66]")

    for c, wks in [(1.0, "1wk=allcash sanity"), (0.50, "2wk"),
                   (0.33, "3wk"), (0.25, "4wk"), (0.20, "5wk")]:
        ts = time.time()
        g = run_stagger(close, tv, c)
        t = run_stagger(close, tv, c, stcg=0.20)
        emit(f"stagger c={c:.2f} ({wks})", g, t,
             f"scaleEv={g['st']['scale_ev']} [{time.time()-ts:.0f}s]")

    print("\n=== GRADUAL DE-RISK BAKE-OFF ===\n" +
          pd.DataFrame(rows).to_string(index=False))
    print(f"\nTotal {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
