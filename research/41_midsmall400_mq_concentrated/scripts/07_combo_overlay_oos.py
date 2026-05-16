"""Phase 06 — same drawdown-control + OOS/post-tax pipeline, COMBO (mid+small, rank 101-500) core.

Mid was chosen for the headline because, RS-alone, small had ~same/higher
CAGR but far deeper drawdown (-40% to -66%) and worse Calmar (0.95 vs
mid 1.29). BUT Phase 03's regime gate is a drawdown lever, and small has
*more* bear beta to remove — so the regime filter may help small
disproportionately. This phase tests that, apples-to-apples with mid.

Small core = best risk-adjusted robust small RS-alone config:
`small_blend_6m12m_N15` (39.3% CAGR / -41.5% DD / Calmar 0.95 / robust
ex-top-3). Same overlay axes and same OOS/post-tax tests as Phase 03/04.
Volume-confirm omitted (Phase 03 proved it poison everywhere).

Reuses 02_rs_sweep.py loaders/helpers. NIFTYBEES benchmark.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_spec = importlib.util.spec_from_file_location(
    "rs2", str(Path(__file__).resolve().parent / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rs2)

BAND, L, N = "combo", None, 20          # L=None -> blended 6m+12m RS
BUFFER = int(N * rs2.BUFFER_MULT)
RT_COST = rs2.RT_COST
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)
BENCH = rs2.NIFTY_SYM
SMA_N = 200
Q_POSFRAC = [None, 0.50, 0.58]
Q_MAXDD = [None, -0.50, -0.40]
REGIME = [False, True]


def own_q(h, s):
    ss = h[s].dropna().tail(252)
    if len(ss) < 120:
        return None, None
    blocks = [ss.iloc[i:i + 21] for i in range(0, len(ss) - 1, 21)]
    rets = [b.iloc[-1] / b.iloc[0] - 1 for b in blocks
            if len(b) >= 2 and b.iloc[0] > 0]
    pf = float(np.mean([r > 0 for r in rets])) if rets else 0.0
    dd = ((ss - ss.cummax()) / ss.cummax()).min()
    return pf, dd


def backtest(close, tv, qp, qd, regime, start, end, stcg=0.0):
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]
    cash, equity = 1.0, 0.0
    held, last_px, nav, taxc = {}, {}, [], 0.0
    for dt in me:
        h = close.loc[:dt]; px = h.iloc[-1]
        if held:
            ne = 0.0
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and s in last_px and last_px[s] > 0:
                    st[0] *= (1 + p1 / last_px[s] - 1)
                ne += st[0]
            equity = ne
        total = cash * CASH_M + equity
        if regime and BENCH in h.columns:
            b = h[BENCH].dropna()
            if len(b) >= SMA_N and b.iloc[-1] < b.tail(SMA_N).mean():
                if stcg and held:
                    for s, st in held.items():
                        g = st[0] - st[1]
                        if g > 0 and (dt - st[2]).days < 365:
                            total -= g * stcg; taxc += g * stcg
                cash, equity, held, last_px = total, 0.0, {}, {}
                nav.append((dt, total)); continue
        uni = rs2.pit_universe(tv, close, dt, BAND)
        sc = rs2.rs_scores(close, dt, L)
        if sc is None or not uni:
            nav.append((dt, total)); continue
        cand = [s for s in sc.index if s in uni and s != BENCH]
        sc = sc[cand].sort_values(ascending=False)
        picked = []
        for s in sc.index:
            if qp is not None or qd is not None:
                pf, od = own_q(h, s)
                if pf is None:
                    continue
                if qp is not None and pf < qp:
                    continue
                if qd is not None and (od is None or od < qd):
                    continue
            picked.append(s)
            if len(picked) >= BUFFER:
                break
        if not picked:
            cash, equity, held, last_px = total, 0.0, {}, {}
            nav.append((dt, total)); continue
        top = picked[:N]; buf = set(picked[:BUFFER])
        keep = [s for s in held if s in buf]
        fill = [s for s in top if s not in keep]
        target = [s for s in (keep + fill)[:N]
                  if pd.notna(px.get(s, np.nan))]
        if not target:
            nav.append((dt, total)); continue
        dropped = [s for s in held if s not in set(target)]
        if stcg:
            for s in dropped:
                st = held[s]; g = st[0] - st[1]
                if g > 0 and (dt - st[2]).days < 365:
                    total -= g * stcg; taxc += g * stcg
        turn = len(set(held).symmetric_difference(set(target))) / max(1, 2 * N)
        total *= (1 - RT_COST * turn * 2)
        w = total / len(target)
        nh = {}
        for s in target:
            if s in held and s not in dropped:
                nh[s] = [w, held[s][1], held[s][2]]
            else:
                nh[s] = [w, w, dt]
        held = nh
        last_px = {s: px[s] for s in target}
        cash, equity = 0.0, total
        nav.append((dt, total))
    ndf = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")
    return ndf, taxc


def m(ndf):
    n = ndf["nav"]
    if len(n) < 6:
        return dict(cagr=np.nan, dd=np.nan, sh=np.nan, cal=np.nan)
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    mr = n.pct_change().dropna()
    sh = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0
    return dict(cagr=cg * 100, dd=dd * 100, sh=sh,
                cal=(cg / abs(dd)) if dd < 0 else np.nan)


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  combo core = combo_blend_6m12m_N20, benchmark={BENCH}",
          flush=True)
    FULL = ("2014-01-01", "2026-12-31")

    b0, _ = backtest(close, tv, None, None, False, *FULL)
    M0 = m(b0)
    print(f"\nCOMBO RS-ALONE baseline (combo_blend_N20): "
          f"CAGR={M0['cagr']:.1f}% Sh={M0['sh']:.2f} DD={M0['dd']:.1f}% "
          f"Cal={M0['cal']:.2f}", flush=True)

    print("\n=== COMBO: drawdown-control overlay sweep ===", flush=True)
    rows = [dict(config="COMBO_baseline", q=None, dd=None, regime=False,
                 cagr=round(M0['cagr'], 1), sharpe=round(M0['sh'], 2),
                 maxdd=round(M0['dd'], 1), calmar=round(M0['cal'], 2))]
    best = None
    for qp in Q_POSFRAC:
        for qd in Q_MAXDD:
            for rg in REGIME:
                if qp is None and qd is None and not rg:
                    continue
                nd, _ = backtest(close, tv, qp, qd, rg, *FULL)
                M = m(nd)
                lbl = f"q{qp or '_'}_dd{qd or '_'}_{'REG' if rg else 'nor'}"
                rows.append(dict(config=lbl, q=qp, dd=qd, regime=rg,
                                 cagr=round(M['cagr'], 1),
                                 sharpe=round(M['sh'], 2),
                                 maxdd=round(M['dd'], 1),
                                 calmar=round(M['cal'], 2)))
                print(f"  {lbl:24} CAGR={M['cagr']:5.1f}% DD={M['dd']:6.1f}%"
                      f" Sh={M['sh']:.2f} Cal={M['cal']:.2f}", flush=True)
                # pick best Calmar with CAGR>=30 as the small champion
                if M['cagr'] >= 30 and (best is None or M['cal'] > best[1]):
                    best = (lbl, M['cal'], qp, qd, rg, M)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "phase07_combo_overlay.csv", index=False)

    if best is None:
        print("\nNo combo overlay holds CAGR>=30% — combo result vs mid "
              "to the gated mid config. Verdict: keep mid.")
        return
    lbl, cal, qp, qd, rg, MB = best
    print(f"\n=== COMBO champion: {lbl}  "
          f"CAGR={MB['cagr']:.1f}% DD={MB['dd']:.1f}% "
          f"Sh={MB['sh']:.2f} Cal={MB['cal']:.2f} ===", flush=True)

    print("\n=== COMBO champion — OOS sub-period ===", flush=True)
    for tag, s, e in [("H1 2014-2019", "2014-01-01", "2019-12-31"),
                      ("H2 2020-2026", "2020-01-01", "2026-12-31")]:
        nd, _ = backtest(close, tv, qp, qd, rg, s, e)
        MM = m(nd)
        print(f"  {tag}: CAGR={MM['cagr']:5.1f}% DD={MM['dd']:6.1f}% "
              f"Sh={MM['sh']:.2f}", flush=True)

    print("\n=== COMBO champion — post-tax (STCG) ===", flush=True)
    g, _ = backtest(close, tv, qp, qd, rg, *FULL, stcg=0.0)
    MG = m(g)
    print(f"  GROSS        : CAGR={MG['cagr']:5.1f}% DD={MG['dd']:.1f}%",
          flush=True)
    for rate in (0.15, 0.20):
        nt, tax = backtest(close, tv, qp, qd, rg, *FULL, stcg=rate)
        MT = m(nt)
        print(f"  NET STCG@{int(rate*100)}% : CAGR={MT['cagr']:5.1f}% "
              f"DD={MT['dd']:.1f}% Sh={MT['sh']:.2f} "
              f"(drag {MG['cagr']-MT['cagr']:+.1f}pp)", flush=True)

    print("\n=== MID vs COMBO (gated, post-tax @20% STCG) ===")
    print("  MID  q0.5_dd__v__REG : gross 35.3% / net20 28.9% / DD -24.6% "
          "/ Sh 1.53 / Cal 1.44  (Phase 03/04)")
    print(f"  SMALL {lbl:18}: see above — compare net20 CAGR & DD vs mid")
    print("\nVERDICT RULE: small wins only if gated post-tax CAGR clearly")
    print("beats mid's 28.9% AND drawdown is not materially deeper than")
    print("mid's -24.6%. Otherwise mid remains the recommended system.")


if __name__ == "__main__":
    main()
