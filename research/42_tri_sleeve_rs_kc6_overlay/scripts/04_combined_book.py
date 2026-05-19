"""Combined single-book engine — Sleeve 1 + Sleeve 2 + Sleeve 3 on ONE pool.

Capital model (STATUS-MD §2, user-locked): ONE Rs.1-crore book. Sleeve 1
(locked RS-momentum) OWNS the equity and drives the monthly NAV spine.
Sleeve 2 (KC6 option spreads) and Sleeve 3 (short/hedge) are OVERLAYS
funded as margin against the whole book — capacity is CONSTANT and
REGIME-INDEPENDENT; the only constraint is each overlay's swept
max-defined-risk-% of the *current* (compounding) book. Any month where
required overlay risk exceeds book capacity is flagged INFEASIBLE
(caveat C3) — counted, not silently rescued.

Methodology = MONTHLY NAV, deliberately identical to research/41 so the
combined-vs-base comparison is apples-to-apples (research/41's headline
35.3%/−24.6% is a monthly-NAV metric). Reports GROSS combined vs GROSS
base; the post-tax read uses research/41's measured STCG drag
(−6.4pp CAGR @20%) applied uniformly — the canonical post-tax base
remains research/41's 28.9% (caveat C2; we do not re-derive it).

Grid (STATUS-MD §4): Phase-1 Sleeve-2 (expr × IV × risk%), Phase-2
Sleeve-3 (variant × OTM [× short-risk%]), Phase-3 best-combo + OOS.

Run on VPS (canonical-host rule). Incremental CSV. ~50 cells.
    python research/42_tri_sleeve_rs_kc6_overlay/scripts/04_combined_book.py
"""
from __future__ import annotations
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"; RES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

CAPITAL0 = 1e7  # Rs 1 crore — for rupee<->NAV-unit and risk-cap reconciliation
STCG_DRAG_PP = 6.4  # research/41 Phase-04 measured post-tax drag @20% STCG


def _load(modfile, name):
    sp = importlib.util.spec_from_file_location(
        name, str(HERE / "scripts" / modfile))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


s1 = _load("01_sleeve1_base_replay.py", "s1")
s2 = _load("02_sleeve2_kc6_options.py", "s2")
s3 = _load("03_sleeve3_variants.py", "s3")
rs2 = s1.rs2


def metrics_from_nav(nav: pd.Series):
    n = nav.dropna()
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    cagr = (n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1
    mret = n.pct_change().dropna()
    sharpe = (mret.mean() / mret.std() * np.sqrt(12)) if mret.std() > 0 else 0
    mdd = ((n - n.cummax()) / n.cummax()).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return dict(cagr=cagr * 100, sharpe=sharpe, mdd=mdd * 100,
                calmar=calmar, yrs=round(yrs, 1))


def combine(timeline, s1_nav, s2_pnl_rupees, s2_trades,
            s3_navpnl, s2_risk_pct, s3_short_risk_pct):
    """Compose the three sleeves into ONE monthly combined NAV (units,
    start 1.0). Returns (combined_nav_series, n_infeasible_months).

    - Sleeve 1 monthly return drives the spine (already net of base
      cost/cash/regime).
    - Sleeve 2: KC6 rupee P&L is scaled so concurrent defined risk ≤
      s2_risk_pct × current book; converted to a NAV-unit delta on the
      running book. Months where even the minimum lot exceeds the cap
      are INFEASIBLE (flagged).
    - Sleeve 3: per-month NAV-unit P&L added directly (C already sized
      to s3_short_risk_pct × book inside run_sleeve3).
    """
    me = list(s1_nav.index)
    s1n = s1_nav.values
    # Monthly-bucket Sleeve-2 rupee P&L (booked on exit date).
    s2_m = (s2_pnl_rupees.groupby(pd.Grouper(freq="ME")).sum()
            if len(s2_pnl_rupees) else pd.Series(dtype=float))
    # Per-trade max_risk to size against the book-risk cap.
    avg_risk = (s2_trades["max_risk"].mean()
                if len(s2_trades) else 0.0) or 1.0
    s3_m = (s3_navpnl.groupby(pd.Grouper(freq="ME")).sum()
            if len(s3_navpnl) else pd.Series(dtype=float))

    comb = [1.0]
    infeasible = 0
    for i in range(1, len(me)):
        dt = me[i]
        r1 = s1n[i] / s1n[i - 1] - 1.0
        book_rupees = comb[-1] * CAPITAL0

        # ---- Sleeve 2: scale to risk_pct of current book ----
        s2_navdelta = 0.0
        raw = float(s2_m.get(pd.Timestamp(dt), 0.0))
        if raw != 0.0:
            risk_budget = s2_risk_pct * book_rupees
            # KC6 ran ≤5 concurrent at ~avg_risk each → standalone peak
            # risk ≈ 5*avg_risk. Scale factor to hit the book budget:
            standalone_peak = 5.0 * avg_risk
            if standalone_peak <= 0:
                scale = 0.0
            else:
                scale = risk_budget / standalone_peak
            if risk_budget < avg_risk:        # cannot even fund 1 lot
                infeasible += 1
                scale = 0.0
            s2_navdelta = (raw * scale) / CAPITAL0

        # ---- Sleeve 3: NAV-unit P&L (C pre-sized to short_risk_pct) ----
        s3_navdelta = float(s3_m.get(pd.Timestamp(dt), 0.0))
        # express s3 on the *combined* book (it was sized on the s1 book)
        s3_navdelta *= (comb[-1] / s1n[i - 1]) if s1n[i - 1] else 1.0

        comb.append(comb[-1] * (1 + r1) + s2_navdelta + s3_navdelta)

    return pd.Series(comb, index=me), infeasible


FIELDS = ["phase", "label", "s2_expr", "s2_iv", "s2_risk_pct",
          "s3_variant", "s3_otm", "s3_short_risk",
          "cagr", "sharpe", "maxdd", "calmar",
          "d_cagr_vs_base", "d_maxdd_vs_base", "d_calmar_vs_base",
          "cagr_h1", "cagr_h2", "mdd_h1", "mdd_h2",
          "infeasible_months", "posttax_cagr_approx", "verdict"]


def main():
    out = RES / "combined_sweep.csv"
    print("Combined engine — loading panels (one-time) ...", flush=True)
    close, tv = rs2.load()
    import sqlite3
    con = sqlite3.connect(rs2.DB)
    vol = pd.read_sql("SELECT symbol,date,volume FROM market_data_unified "
                      "WHERE timeframe='day' AND date>='2012-06-01' "
                      "ORDER BY symbol,date", con, parse_dates=["date"]
                      ).pivot_table(index="date", columns="symbol",
                                    values="volume").sort_index()
    con.close()

    s1_nav_df, timeline = s1.run_sleeve1(close, tv, vol)
    s1_nav = s1_nav_df["nav"]
    base = metrics_from_nav(s1_nav)
    split = s1_nav.index[len(s1_nav) // 2]
    print(f"  BASE (Sleeve-1 only): CAGR={base['cagr']:.2f}% "
          f"MDD={base['mdd']:.2f}% Sharpe={base['sharpe']:.2f} "
          f"Calmar={base['calmar']:.2f}", flush=True)

    # Sleeve-2 panel (KC6 native F&O universe)
    uni2 = s2.fno_universe()
    panel2 = s2.load_panel(list(uni2))
    uar = s2.universe_atr_ratio_series(panel2)
    print(f"  Sleeve-2 universe: {len(panel2)} F&O names loaded", flush=True)

    def oos(nav):
        a = metrics_from_nav(nav.loc[:split])
        b = metrics_from_nav(nav.loc[split:])
        return a["cagr"], b["cagr"], a["mdd"], b["mdd"]

    rows = []

    def emit(phase, label, **kw):
        nav = kw.pop("nav")
        m = metrics_from_nav(nav)
        h1, h2, d1, d2 = oos(nav)
        # decision rule (STATUS §8): beat base post-tax Calmar OR cut DD
        # at CAGR≥~25%, in BOTH halves.
        pt = m["cagr"] - STCG_DRAG_PP
        better_calmar = m["calmar"] > base["calmar"] + 0.02
        cut_dd = (m["mdd"] > base["mdd"] + 1.0) and m["cagr"] >= 25.0
        both_oos = (h1 >= 20 and h2 >= 20)
        verdict = ("WORTH-IT" if (better_calmar or cut_dd) and both_oos
                   else "base-alone-wins")
        row = dict(phase=phase, label=label,
                   s2_expr=kw.get("s2_expr", ""), s2_iv=kw.get("s2_iv", ""),
                   s2_risk_pct=kw.get("s2_risk_pct", ""),
                   s3_variant=kw.get("s3_variant", ""),
                   s3_otm=kw.get("s3_otm", ""),
                   s3_short_risk=kw.get("s3_short_risk", ""),
                   cagr=round(m["cagr"], 2), sharpe=round(m["sharpe"], 2),
                   maxdd=round(m["mdd"], 2), calmar=round(m["calmar"], 2),
                   d_cagr_vs_base=round(m["cagr"] - base["cagr"], 2),
                   d_maxdd_vs_base=round(m["mdd"] - base["mdd"], 2),
                   d_calmar_vs_base=round(m["calmar"] - base["calmar"], 2),
                   cagr_h1=round(h1, 1), cagr_h2=round(h2, 1),
                   mdd_h1=round(d1, 1), mdd_h2=round(d2, 1),
                   infeasible_months=kw.get("infeasible", 0),
                   posttax_cagr_approx=round(pt, 2), verdict=verdict)
        rows.append(row)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
            w.writerows(rows)
        print(f"  [{phase}] {label:34} CAGR={m['cagr']:6.2f}% "
              f"MDD={m['mdd']:6.2f}% Cal={m['calmar']:.2f} "
              f"(dCal={m['calmar']-base['calmar']:+.2f}) "
              f"inf={kw.get('infeasible',0)} -> {verdict}", flush=True)

    # Phase-0 baseline row
    emit("0", "BASE_sleeve1_only", nav=s1_nav, infeasible=0)

    # ---- Phase 1: Sleeve-2 over base (expr × IV × risk%) ----
    s2_cache = {}
    for expr in ("credit", "debit"):
        for iv in (0.20, 0.25, 0.30):
            p, tr = s2.run_sleeve2_kc6(panel2, uni2, uar,
                                       expression=expr, iv=iv)
            s2_cache[(expr, iv)] = (p, tr)
            for rp in (0.05, 0.10, 0.15):
                nav, inf = combine(timeline, s1_nav, p, tr,
                                   pd.Series(dtype=float), rp, 0.0)
                emit("1", f"S2_{expr}_iv{int(iv*100)}_rp{int(rp*100)}",
                     nav=nav, infeasible=inf, s2_expr=expr, s2_iv=iv,
                     s2_risk_pct=rp)

    # best Sleeve-2 by combined Calmar
    s1_rows = [r for r in rows if r["phase"] == "1"]
    bestS2 = max(s1_rows, key=lambda r: r["calmar"])
    bp, bt = s2_cache[(bestS2["s2_expr"], bestS2["s2_iv"])]
    brp = bestS2["s2_risk_pct"]
    print(f"\n  Best Sleeve-2: {bestS2['label']} "
          f"(Calmar {bestS2['calmar']}) — carrying into Phase 2\n", flush=True)

    # ---- Phase 2: Sleeve-3 over (base + best Sleeve-2) ----
    for v in ("A", "B", "C"):
        otms = (3.0, 5.0, 7.0)
        srisks = (0.05, 0.10) if v == "C" else (0.0,)
        for otm in otms:
            for sr in srisks:
                sp, _ = s3.run_sleeve3(timeline, close, variant=v,
                                       otm_pct=otm, iv=0.25,
                                       short_risk_pct=(sr or 0.05))
                nav, inf = combine(timeline, s1_nav, bp, bt, sp, brp,
                                   sr or 0.0)
                lbl = f"S2best+S3{v}_otm{int(otm)}" + (
                    f"_sr{int(sr*100)}" if v == "C" else "")
                emit("2", lbl, nav=nav, infeasible=inf,
                     s2_expr=bestS2["s2_expr"], s2_iv=bestS2["s2_iv"],
                     s2_risk_pct=brp, s3_variant=v, s3_otm=otm,
                     s3_short_risk=(sr if v == "C" else ""))
    # OFF (base + best S2 only) already = bestS2 row.

    # ---- Phase 3: best overall + summary ----
    p23 = [r for r in rows if r["phase"] in ("1", "2")]
    best = max(p23, key=lambda r: (r["verdict"] == "WORTH-IT",
                                   r["calmar"]))
    print("\n=== SUMMARY ===")
    print(f"  BASE         : CAGR={base['cagr']:.2f}% "
          f"MDD={base['mdd']:.2f}% Calmar={base['calmar']:.2f} "
          f"(post-tax ~{base['cagr']-STCG_DRAG_PP:.1f}%, "
          f"research/41 canonical 28.9%)")
    print(f"  BEST COMBINED: {best['label']} CAGR={best['cagr']}% "
          f"MDD={best['maxdd']}% Calmar={best['calmar']} "
          f"-> {best['verdict']}")
    print(f"  OOS halves   : H1 {best['cagr_h1']}% / H2 {best['cagr_h2']}%")
    print(f"  Rows written : {len(rows)} -> {out}")
    print("  Honest read in results/RESULTS.md (write after this run).")


if __name__ == "__main__":
    main()
