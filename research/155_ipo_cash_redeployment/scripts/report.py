"""research/155 — final report: re-runs the headline arms with full per-day diagnostics,
evaluates the pre-registered adoption bar, writes the YoY house table, the per-year
redeployment/pull-back diagnostics, the static-tilt null and the factsheet PNG.

Reads `results/headline.json`:
    [{"label": "...", "asset": null|"OA"|"TN"|"MIX"|"NB", "settle":1, "reserve":0,
      "cadence":"daily", "sell":"prorata", "tax":"full", "cost_bps":25,
      "gateN": null|25|50|100, "frictionless": false}, ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
ROOT = Path("/home/arun/quantifyd")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "153_ipo_base" / "scripts"))
sys.path.insert(0, str(HERE))
import ipo_replay as ir          # noqa: E402
import ipo_park as ip            # noqa: E402
import run_sweep as rs           # noqa: E402

SPEC = rs.SPEC
N = rs.N_PATHS
W_OA, W_TN, W_IPO = rs.W_OA, rs.W_TN, rs.W_IPO


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    c = ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
    d = float((s / s.cummax() - 1).min() * 100)
    return c, d


def wdd_full(s, a, b):
    dd = s / s.cummax() - 1.0
    seg = dd[(dd.index >= a) & (dd.index <= b)]
    return float(seg.min() * 100) if len(seg) else np.nan


def wret(s, a, b):
    seg = s[(s.index >= a) & (s.index <= b)]
    return float((seg.iloc[-1] / seg.iloc[0] - 1) * 100) if len(seg) > 1 else np.nan


def main():
    D = rs.load_all()
    ctx, dates, days, used = D["ctx"], D["dates"], D["days"], D["used"]
    oa, tn = D["oa"], D["tn"]
    arms = json.loads((RES / "headline.json").read_text())

    # NIFTYBEES benchmark (total-price, dividends not reinvested -> stated as a caveat)
    nb = ctx.close["NIFTYBEES"].reindex(dates).ffill().loc[used]

    store = {}          # label -> dict(nav=DataFrame, blend=DataFrame, diag=DataFrame)
    for a in arms:
        lab = a["label"]
        t0 = time.time()
        gn = a.get("gateN")
        pa = None
        if gn:
            pa = D.setdefault("_gate", {}).get(gn)
            if pa is None:
                pa = ip.forward_pool_empty(ctx, gn, SPEC["max_age_m"], SPEC["min_bars"])
                D["_gate"][gn] = pa
        navs, blends, diags = {}, {}, []
        for p in range(N):
            oc, tc, seed = f"s{p+1}", f"off{p % 12}", p + 1
            asset = a.get("asset")
            lvl = (None if asset in (None, "cash") else
                   D["park"]["OA"][oc] if asset == "OA" else
                   D["park"]["TN"][tc] if asset == "TN" else
                   D["park"]["NB"]["nb"] if asset == "NB" else
                   rs.mix5050(D["park"]["OA"][oc], D["park"]["TN"][tc]))
            r = ip.simulate_park(
                seed, days, dates, ctx.C, ctx.O, D["piv"], D["lo"], D["sma"], ctx.TVp,
                D["trig"], D["weak"], park_lvl=lvl, park_allowed=pa,
                cost=a.get("cost_bps", 25) / 1e4, stop=SPEC["stop"], slots=SPEC["slots"],
                size_pct=SPEC["size_pct"], target=SPEC["target"],
                settle_days=a.get("settle", 1), reserve_slots=a.get("reserve", 0),
                cadence=a.get("cadence", "daily"), sell_policy=a.get("sell", "prorata"),
                park_tax=a.get("tax", "full"), frictionless=a.get("frictionless", False))
            nav = pd.Series(r["nav"], index=used)
            navs[f"p{p}"] = nav
            idx = nav.index.intersection(oa.index).intersection(tn.index)
            n_ = nav.loc[idx] / nav.loc[idx].iloc[0]
            o_ = oa[oc].loc[idx] / oa[oc].loc[idx].iloc[0]
            t_ = tn[tc].loc[idx] / tn[tc].loc[idx].iloc[0]
            nm = n_.resample("ME").last().pct_change().fillna(0.0)
            om = o_.resample("ME").last().pct_change().fillna(0.0)
            tm = t_.resample("ME").last().pct_change().fillna(0.0)
            blends[f"p{p}"] = (1 + W_OA * om + W_TN * tm + W_IPO * nm).cumprod()
            dd = pd.DataFrame(dict(year=used.year, parked=r["parked"], nav=r["nav"],
                                   pull=r["pull_n_d"], pcost=r["pull_c_d"],
                                   miss=r["miss_d"]))
            g = dd.groupby("year").agg(parked_pct=("parked", lambda x: 0.0),
                                       pull=("pull", "sum"), pcost=("pcost", "sum"),
                                       miss=("miss", "sum"))
            g["parked_pct"] = (dd.groupby("year")
                               .apply(lambda x: 100 * (x.parked / x.nav).mean(),
                                      include_groups=False))
            g["path"] = p
            diags.append(g.reset_index())
        store[lab] = dict(nav=pd.DataFrame(navs), blend=pd.DataFrame(blends),
                          diag=pd.concat(diags))
        log(f"{lab}: {time.time()-t0:.0f}s")

    inc = store["A_incumbent"]["blend"]

    # ───────────────────────────────── adoption bar, paired on the same 30 paths
    lines = []
    def emit(s=""):
        lines.append(s); print(s, flush=True)

    emit("=" * 108)
    emit("PRE-REGISTERED ADOPTION BAR — paired on the same 30 paths, after tax, "
         "vs the 40/40/20 incumbent")
    emit("=" * 108)
    emit(f"{'arm':<26} {'CAGR':>7} {'MaxDD':>8} {'Calmar':>7} | {'dCAGR':>7} "
         f"{'dCalmar':>8} {'dDD':>7} | {'winC':>5} {'winCal':>6} | "
         f"{'DD0809':>7} {'DD1214':>7} | {'corrOA':>7} {'corrTN':>7}")
    base_stats = pd.DataFrame([stats(inc[c]) for c in inc.columns],
                              columns=["cagr", "dd"], index=inc.columns)
    base_stats["calmar"] = base_stats.cagr / base_stats.dd.abs()
    base_w = pd.DataFrame({
        "dd0809": [wdd_full(inc[c], "2008-01-01", "2009-12-31") for c in inc.columns],
        "dd1214": [wdd_full(inc[c], "2012-01-01", "2014-12-31") for c in inc.columns]},
        index=inc.columns)
    rows = []
    for lab, S in store.items():
        b = S["blend"]
        st = pd.DataFrame([stats(b[c]) for c in b.columns],
                          columns=["cagr", "dd"], index=b.columns)
        st["calmar"] = st.cagr / st.dd.abs()
        w = pd.DataFrame({
            "dd0809": [wdd_full(b[c], "2008-01-01", "2009-12-31") for c in b.columns],
            "dd1214": [wdd_full(b[c], "2012-01-01", "2014-12-31") for c in b.columns]},
            index=b.columns)
        nvd = S["nav"].pct_change().fillna(0)
        co = float(np.median([nvd[f"p{p}"].corr(
            (oa[f"s{p+1}"].reindex(used).ffill()).pct_change().fillna(0)) for p in range(N)]))
        ct = float(np.median([nvd[f"p{p}"].corr(
            (tn[f"off{p%12}"].reindex(used).ffill()).pct_change().fillna(0))
            for p in range(N)]))
        d_c = st.cagr - base_stats.cagr
        d_k = st.calmar - base_stats.calmar
        d_d = st.dd - base_stats.dd
        rows.append(dict(arm=lab, cagr=st.cagr.median(), dd=st.dd.median(),
                         calmar=st.calmar.median(), d_cagr=d_c.median(),
                         d_calmar=d_k.median(), d_dd=d_d.median(),
                         win_cagr=int((d_c > 0).sum()), win_calmar=int((d_k > 0).sum()),
                         dd0809=w.dd0809.median(), dd1214=w.dd1214.median(),
                         d_dd0809=(w.dd0809 - base_w.dd0809).median(),
                         d_dd1214=(w.dd1214 - base_w.dd1214).median(),
                         corr_oa=co, corr_tn=ct))
        r = rows[-1]
        emit(f"{lab:<26} {r['cagr']:7.2f} {r['dd']:8.2f} {r['calmar']:7.3f} | "
             f"{r['d_cagr']:+7.2f} {r['d_calmar']:+8.3f} {r['d_dd']:+7.2f} | "
             f"{r['win_cagr']:5d} {r['win_calmar']:6d} | {r['dd0809']:7.2f} "
             f"{r['dd1214']:7.2f} | {r['corr_oa']:7.3f} {r['corr_tn']:7.3f}")
    adopt = pd.DataFrame(rows)
    adopt.round(4).to_csv(RES / "adoption.csv", index=False)

    emit()
    emit("Bar: (1) +0.10 Calmar OR -2pp MaxDD at >= equal CAGR; (2) wins >= 26/30 paired; "
         "(3) drought-window MaxDD not worse by >1.5pp; (6) corr < 0.40 to BOTH legs.")
    for r in rows:
        if r["arm"] == "A_incumbent":
            continue
        c1 = (r["d_calmar"] >= 0.10 or r["d_dd"] >= 2.0) and r["d_cagr"] >= 0.0
        c2 = r["win_calmar"] >= 26
        c3 = r["d_dd0809"] >= -1.5 and r["d_dd1214"] >= -1.5
        c6 = r["corr_oa"] < 0.40 and r["corr_tn"] < 0.40
        emit(f"  {r['arm']:<26} (1){'PASS' if c1 else 'FAIL'} (2){'PASS' if c2 else 'FAIL'}"
             f" (3){'PASS' if c3 else 'FAIL'} (6){'PASS' if c6 else 'FAIL'}"
             f"   -> {'ADOPT-eligible' if all([c1,c2,c3,c6]) else 'REJECT'}")

    # ───────────────────────────────── per-year diagnostics
    emit()
    emit("=" * 108)
    emit("PER-YEAR — blend return / intra-year DD (FULL-curve peak) / % of the IPO sleeve "
         "redeployed / pull-backs + their cost")
    emit("=" * 108)
    yrs = sorted(set(used.year))
    peryear = {}
    for lab, S in store.items():
        b, dg = S["blend"], S["diag"]
        gy = dg.groupby("year").median(numeric_only=True)
        rr = {y: float(np.median([wret(b[c], f"{y}-01-01", f"{y}-12-31")
                                  for c in b.columns])) for y in yrs}
        dd = {y: float(np.median([wdd_full(b[c], f"{y}-01-01", f"{y}-12-31")
                                  for c in b.columns])) for y in yrs}
        peryear[lab] = pd.DataFrame(dict(ret=pd.Series(rr), dd=pd.Series(dd),
                                         parked_pct=gy.parked_pct, pull=gy.pull,
                                         pcost_x=gy.pcost / ip.CAPITAL, miss=gy.miss))
    pd.concat(peryear, axis=1).round(3).to_csv(RES / "peryear.csv")
    hd = [l for l in store if l != "A_incumbent"]
    show = ["A_incumbent"] + hd[:2]
    hdr = f"{'yr':>5}"
    for l in show:
        hdr += f" | {l[:20]:>34}"
    emit(hdr)
    emit(f"{'':>5}" + "".join(f" | {'ret':>7} {'ddF':>7} {'%park':>6} {'pull':>5} {'cost':>5}"
                              for _ in show))
    for y in yrs:
        line = f"{y:>5}"
        for l in show:
            r = peryear[l].loc[y]
            line += (f" | {r['ret']:7.1f} {r['dd']:7.1f} {r['parked_pct']:6.1f} "
                     f"{r['pull']:5.0f} {r['pcost_x']:5.2f}")
        emit(line)

    # ───────────────────────────────── YoY house table
    emit()
    emit("=" * 108)
    emit("YoY HOUSE TABLE (median across the 30 paired paths; return with intra-year max DD "
         "from the FULL curve's peak)")
    emit("=" * 108)
    cols = {}
    cols["TN"] = pd.DataFrame({f"p{p}": tn[f"off{p%12}"].reindex(used).ffill()
                               for p in range(N)}).dropna()
    cols["OA"] = pd.DataFrame({f"p{p}": oa[f"s{p+1}"].reindex(used).ffill()
                               for p in range(N)}).dropna()
    cols["IPO (cash)"] = store["A_incumbent"]["nav"]
    for l in hd[:2]:
        cols[f"IPO ({l})"] = store[l]["nav"]
    cols["Blend 40/40/20 (incumbent)"] = inc
    for l in hd[:2]:
        cols[f"Blend 40/40/20 ({l})"] = store[l]["blend"]
    cols["NIFTYBEES (bench)"] = pd.DataFrame({"b": nb})
    yoy_r, yoy_d = {}, {}
    for tag, df in cols.items():
        yoy_r[tag] = {y: float(np.median([wret(df[c], f"{y}-01-01", f"{y}-12-31")
                                          for c in df.columns])) for y in yrs}
        yoy_d[tag] = {y: float(np.median([wdd_full(df[c], f"{y}-01-01", f"{y}-12-31")
                                          for c in df.columns])) for y in yrs}
    R = pd.DataFrame(yoy_r); DDf = pd.DataFrame(yoy_d)
    R.round(2).to_csv(RES / "yoy_returns.csv")
    DDf.round(2).to_csv(RES / "yoy_intradd.csv")
    picks = [c for c in R.columns if "bench" not in c]
    best_cagr = R[picks].idxmax(axis=1)
    least_dd = DDf[picks].idxmax(axis=1)
    best_all = (R[picks] + DDf[picks]).idxmax(axis=1)
    emit("year | " + " | ".join(f"{c[:22]:>22}" for c in R.columns) +
         " || BEST CAGR / LEAST DD / BEST OVERALL")
    for y in yrs:
        emit(f"{y} | " + " | ".join(f"{R[c][y]:>10.1f} ({DDf[c][y]:>6.1f})"
                                    for c in R.columns) +
             f" || {best_cagr[y][:16]} / {least_dd[y][:16]} / {best_all[y][:16]}")
    summ = {}
    for tag, df in cols.items():
        cs = [stats(df[c])[0] for c in df.columns]
        ds = [stats(df[c])[1] for c in df.columns]
        summ[tag] = (float(np.median(cs)), float(np.median(ds)))
    emit("FULL | " + " | ".join(f"{summ[c][0]:>10.2f} ({summ[c][1]:>6.2f})"
                                for c in R.columns))
    pd.DataFrame(summ, index=["cagr", "maxdd"]).T.round(2).to_csv(RES / "yoy_summary.csv")

    # ───────────────────────────────── static-tilt null
    emit()
    emit("=" * 108)
    emit("STATIC-TILT NULL — can a plain STATIC weight vector (TN / OA / IPO-incumbent) "
         "dominate the dynamic arms?")
    emit("=" * 108)
    inc_nav = store["A_incumbent"]["nav"]
    grid = []
    for wi in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33):
        for wo in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            wt = 1 - wi - wo
            if wt < 0.05:
                continue
            cs, ds = [], []
            for p in range(N):
                idx = inc_nav[f"p{p}"].index.intersection(oa.index).intersection(tn.index)
                n_ = inc_nav[f"p{p}"].loc[idx]; n_ = n_ / n_.iloc[0]
                o_ = oa[f"s{p+1}"].loc[idx]; o_ = o_ / o_.iloc[0]
                t_ = tn[f"off{p%12}"].loc[idx]; t_ = t_ / t_.iloc[0]
                m = lambda x: x.resample("ME").last().pct_change().fillna(0.0)  # noqa: E731
                bl = (1 + wo * m(o_) + wt * m(t_) + wi * m(n_)).cumprod()
                c_, d_ = stats(bl)
                cs.append(c_); ds.append(d_)
            grid.append(dict(w_tn=round(wt, 2), w_oa=wo, w_ipo=wi,
                             cagr=float(np.median(cs)), dd=float(np.median(ds)),
                             calmar=float(np.median(cs)) / abs(float(np.median(ds)))))
    G = pd.DataFrame(grid)
    G.round(3).to_csv(RES / "static_tilt_null.csv", index=False)
    for r in rows:
        if r["arm"] == "A_incumbent":
            continue
        dom = G[(G.cagr >= r["cagr"]) & (G.dd >= r["dd"])]
        emit(f"  {r['arm']:<26} {r['cagr']:6.2f}/{r['dd']:7.2f} — static vectors that "
             f"dominate it on BOTH: {len(dom)}/{len(G)}"
             + (f"  best: TN{dom.iloc[dom.calmar.argmax()].w_tn:.2f}/"
                f"OA{dom.iloc[dom.calmar.argmax()].w_oa:.2f}/"
                f"IPO{dom.iloc[dom.calmar.argmax()].w_ipo:.2f} "
                f"= {dom.calmar.max():.2f} Calmar" if len(dom) else ""))
    emit(f"  best STATIC vector overall: "
         f"{G.iloc[G.calmar.argmax()].to_dict()}")

    (RES / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    # ───────────────────────────────── factsheet
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.4, 1]))
    plotset = [("Blend 40/40/20 — IPO idle to CASH (incumbent)", inc, "#1f77b4"),
               ]
    palette = ["#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for i, l in enumerate(hd[:4]):
        plotset.append((f"Blend 40/40/20 — {l}", store[l]["blend"], palette[i % 4]))
    for tag, df, col in plotset:
        med = df.apply(lambda r: np.median(r), axis=1)
        med = 100 * med / med.iloc[0]
        ax[0].plot(med.index, med, label=f"{tag}", color=col, lw=1.6)
        ax[1].plot(med.index, 100 * (med / med.cummax() - 1), color=col, lw=1.0)
    bm = nb.resample("ME").last().dropna()
    bm = 100 * bm / bm.iloc[0]
    ax[0].plot(bm.index, bm, label="NIFTYBEES (benchmark)", color="#7f7f7f",
               lw=1.2, ls="--")
    ax[1].plot(bm.index, 100 * (bm / bm.cummax() - 1), color="#7f7f7f", lw=0.8, ls="--")
    ax[0].set_yscale("log"); ax[0].set_ylabel("growth of Rs 100 (log)")
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=.25)
    ax[0].set_title("research/155 — redeploying the IPO sleeve's idle cash into OA / TN\n"
                    "median of 30 paired paths, after tax, 25 bps/side, monthly rebalanced",
                    fontsize=11)
    ax[1].set_ylabel("drawdown %"); ax[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(RES / "ipo_cash_redeployment_research155.png", dpi=130)
    log("wrote factsheet PNG")
    log("REPORT DONE")


if __name__ == "__main__":
    main()
