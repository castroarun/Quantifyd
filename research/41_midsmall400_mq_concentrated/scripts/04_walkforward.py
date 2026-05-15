"""Phase 04 — out-of-sample validation + post-tax (STCG) for the
recommended config `q0.5_dd__v__REG` on the mid_120d_N15 core.

The strategy is a fixed RULE, so the real overfitting risk is the
SELECTION (we picked band=mid, L=120, N=15, q=0.5, regime=on out of a
75-config + 53-overlay sweep on the full 2014-2026 window). This phase
tests three honest things:

  A. Sub-period stability — run the FIXED chosen config separately on
     2014-2019 and 2020-2026. A real edge must show up in BOTH halves,
     not be concentrated in one regime.
  B. Walk-forward selection of the RS lookback — the main fitted knob.
     Each year Y (from 2019), pick the lookback L in {55,120,126,252}
     with the best trailing-3y Calmar using data up to Dec(Y-1), trade
     it through year Y, chain the NAVs. If the WF chain ≈ the static
     120d pick, the lookback choice is robust, not lucky.
  C. Post-tax — apply Indian STCG (positions held <365d sold at a gain)
     at 20% (current) and 15% (pre-Jul-2024) to the chosen config and
     report the net CAGR drag. LTCG ignored (conservative direction is
     stated; monthly rotation is overwhelmingly short-term anyway).

Reuses the vetted loaders/helpers from 02_rs_sweep.py.
"""
from __future__ import annotations
import importlib.util, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
_spec = importlib.util.spec_from_file_location(
    "rs2", str(Path(__file__).resolve().parent / "02_rs_sweep.py"))
rs2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rs2)

BAND, N, BENCH = "mid", 15, rs2.NIFTY_SYM
BUFFER = int(N * rs2.BUFFER_MULT)
RT_COST = rs2.RT_COST
CASH_M = (1 + rs2.CASH_ANNUAL) ** (1 / 12)
Q_POSFRAC, SMA_N = 0.50, 200


def own_pos_frac(s):
    s = s.dropna().tail(252)
    if len(s) < 120:
        return None
    blocks = [s.iloc[i:i + 21] for i in range(0, len(s) - 1, 21)]
    rets = [b.iloc[-1] / b.iloc[0] - 1 for b in blocks
            if len(b) >= 2 and b.iloc[0] > 0]
    return float(np.mean([r > 0 for r in rets])) if rets else 0.0


def backtest(close, tv, L, start, end, use_overlay, stcg_rate=0.0):
    """Windowed backtest. use_overlay -> add q0.5 quality + SMA200 regime.
    stcg_rate>0 -> deduct tax on short-term (held<365d) realized gains.
    Returns (nav_df, gross_cagr, net_cagr, stcg_paid_frac)."""
    me = rs2.month_ends(close.index)
    me = me[(me >= pd.Timestamp(start)) & (me <= pd.Timestamp(end))]
    cash, equity = 1.0, 0.0
    held = {}            # sym -> [value, cost_basis, buy_date]
    last_px = {}
    nav, tax_cum = [], 0.0
    init = 1.0
    for dt in me:
        h = close.loc[:dt]
        px = h.iloc[-1]
        if held:
            ne = 0.0
            for s, st in held.items():
                p1 = px.get(s, np.nan)
                if pd.notna(p1) and s in last_px and last_px[s] > 0:
                    st[0] *= (1 + p1 / last_px[s] - 1)
                ne += st[0]
            equity = ne
        total = cash * CASH_M + equity

        if use_overlay and BENCH in h.columns:
            b = h[BENCH].dropna()
            if len(b) >= SMA_N and b.iloc[-1] < b.tail(SMA_N).mean():
                # liquidate -> realize STCG on short holds
                if stcg_rate and held:
                    for s, st in held.items():
                        g = st[0] - st[1]
                        if g > 0 and (dt - st[2]).days < 365:
                            t = g * stcg_rate; total -= t; tax_cum += t
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
            if use_overlay:
                pf = own_pos_frac(h[s])
                if pf is None or pf < Q_POSFRAC:
                    continue
            picked.append(s)
            if len(picked) >= BUFFER:
                break
        if not picked:
            if stcg_rate and held:
                for s, st in held.items():
                    g = st[0] - st[1]
                    if g > 0 and (dt - st[2]).days < 365:
                        t = g * stcg_rate; total -= t; tax_cum += t
            cash, equity, held, last_px = total, 0.0, {}, {}
            nav.append((dt, total)); continue
        top = picked[:N]; buf = set(picked[:BUFFER])
        keep = [s for s in held if s in buf]
        fill = [s for s in top if s not in keep]
        target = [s for s in (keep + fill)[:N] if pd.notna(px.get(s, np.nan))]
        if not target:
            nav.append((dt, total)); continue
        # realize STCG on names being dropped
        dropped = [s for s in held if s not in set(target)]
        if stcg_rate:
            for s in dropped:
                st = held[s]; g = st[0] - st[1]
                if g > 0 and (dt - st[2]).days < 365:
                    t = g * stcg_rate; total -= t; tax_cum += t
        turn = len(set(held).symmetric_difference(set(target))) / max(1, 2 * N)
        total *= (1 - RT_COST * turn * 2)
        w = total / len(target)
        new_held = {}
        for s in target:
            if s in held and s not in dropped:
                bd, cb = held[s][2], held[s][1]   # carry basis/buy-date
            else:
                bd, cb = dt, w
            new_held[s] = [w, cb, bd]
        held = new_held
        last_px = {s: px[s] for s in target}
        cash, equity = 0.0, total
        nav.append((dt, total))
    ndf = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date")
    if len(ndf) < 6:
        return ndf, np.nan, np.nan, 0.0
    yrs = (ndf.index[-1] - ndf.index[0]).days / 365.25
    net_cagr = (ndf["nav"].iloc[-1] / init) ** (1 / yrs) - 1
    return ndf, net_cagr, net_cagr, tax_cum / init


def cagr_dd(ndf):
    n = ndf["nav"]
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cg = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    dd = ((n - n.cummax()) / n.cummax()).min()
    mret = n.pct_change().dropna()
    sh = mret.mean() / mret.std() * np.sqrt(12) if mret.std() > 0 else 0
    return cg * 100, dd * 100, sh


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print(f"  benchmark={BENCH}, {close.shape[1]} symbols", flush=True)

    print("\n=== A. SUB-PERIOD STABILITY (fixed config q0.5_dd__v__REG) ===")
    for tag, s, e in [("FULL 2014-2026", "2014-01-01", "2026-12-31"),
                      ("H1   2014-2019", "2014-01-01", "2019-12-31"),
                      ("H2   2020-2026", "2020-01-01", "2026-12-31")]:
        nd, *_ = backtest(close, tv, 120, s, e, True)
        cg, dd, sh = cagr_dd(nd)
        print(f"  {tag}: CAGR={cg:5.1f}%  MaxDD={dd:6.1f}%  Sharpe={sh:.2f}")

    print("\n=== B. WALK-FORWARD LOOKBACK SELECTION (the fitted knob) ===")
    print("  Each year: pick L by best trailing-3y Calmar (RS-alone, no")
    print("  overlay), trade it that year. Chain vs static L=120.")
    grid_L = {"55d": 55, "120d": 120, "126d_6m": 126, "252d_1y": 252}
    wf_nav, picks = [], []
    base = 1.0
    for Y in range(2019, 2027):
        best, bestcal = None, -1e9
        for lk, L in grid_L.items():
            nd, *_ = backtest(close, tv, L, f"{Y-3}-01-01",
                              f"{Y-1}-12-31", False)
            if len(nd) < 12:
                continue
            cg, dd, _ = cagr_dd(nd)
            cal = (cg / abs(dd)) if dd < 0 else 0
            if cal > bestcal:
                bestcal, best = cal, (lk, L)
        if best is None:
            continue
        nd, *_ = backtest(close, tv, best[1], f"{Y}-01-01",
                          f"{Y}-12-31", False)
        if len(nd) < 2:
            continue
        yr_ret = nd["nav"].iloc[-1] / nd["nav"].iloc[0]
        base *= yr_ret
        wf_nav.append((Y, base, best[0], (yr_ret - 1) * 100))
        picks.append(f"{Y}:{best[0]}")
    wf = pd.DataFrame(wf_nav, columns=["year", "cum", "L_picked", "yr_ret%"])
    print("  " + "  ".join(picks))
    if len(wf):
        yrs = len(wf)
        wf_cagr = wf["cum"].iloc[-1] ** (1 / yrs) - 1
        print(wf.to_string(index=False))
        print(f"  WALK-FWD chained CAGR (RS-alone, {yrs}y 2019-2026): "
              f"{wf_cagr*100:.1f}%")
    stat, *_ = backtest(close, tv, 120, "2019-01-01", "2026-12-31", False)
    sc, sd, _ = cagr_dd(stat)
    print(f"  STATIC L=120 (RS-alone, same 2019-2026 window): "
          f"CAGR={sc:.1f}%  DD={sd:.1f}%")

    print("\n=== C. POST-TAX (STCG drag on chosen config, full window) ===")
    g_nd, *_ = backtest(close, tv, 120, "2014-01-01", "2026-12-31", True, 0.0)
    gcg, gdd, gsh = cagr_dd(g_nd)
    print(f"  GROSS            : CAGR={gcg:5.1f}%  DD={gdd:.1f}%  Sh={gsh:.2f}")
    for rate in (0.15, 0.20):
        n_nd, ncg, _, taxfrac = backtest(close, tv, 120, "2014-01-01",
                                         "2026-12-31", True, rate)
        ncg_p, ndd, nsh = cagr_dd(n_nd)
        print(f"  NET STCG@{int(rate*100)}%      : CAGR={ncg_p:5.1f}%  "
              f"DD={ndd:.1f}%  Sh={nsh:.2f}  "
              f"(drag {gcg-ncg_p:+.1f}pp; cum tax ~{taxfrac*100:.0f}% of init)")

    print("\n=== HONEST VERDICT ===")
    print("  Pass test if: edge present in BOTH sub-periods, WF lookback")
    print("  selection ~= static 120d (selection robust), and post-tax CAGR")
    print("  still comfortably > 20% hurdle. LTCG ignored (small vs STCG;")
    print("  understates total tax slightly -- stated, not hidden).")
    g_nd.to_csv(RES / "phase04_chosen_gross_nav.csv")
    wf.to_csv(RES / "phase04_walkforward.csv", index=False)
    print("\nWrote phase04_chosen_gross_nav.csv, phase04_walkforward.csv")


if __name__ == "__main__":
    main()
