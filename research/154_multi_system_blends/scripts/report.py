"""research/154 P8 - deliverables: gold-only frontier, daily-marked robustness, the YoY
house-format table, and the figures.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from blend_matrix import PANELS, NPATH, NSEED, NOFF, Panel, build_monthly, load_daily, \
    path_stats, RES, ROOT

pd.set_option("display.width", 300)
pd.set_option("display.max_rows", 400)

BOOKS = [
    ("TN+OA 50-50 (deployed)", {"OA": .50, "TN": .50}),
    ("OA45 TN35 IPO10 GOLD10", {"OA": .45, "TN": .35, "IPO": .10, "GOLD": .10}),
    ("OA40 TN25 IPO20 GOLD15", {"OA": .40, "TN": .25, "IPO": .20, "GOLD": .15}),
    ("OA+IPO+GOLD 33/33/33", {"OA": 1 / 3, "IPO": 1 / 3, "GOLD": 1 / 3}),
]
YOY_PANEL = "B"


# --------------------------------------------------------------- daily-marked blend
def blend_daily(navs: dict, w: dict, idx: pd.DatetimeIndex) -> pd.Series:
    """Monthly rebalance to target weights, marked DAILY (honest intra-month drawdown)."""
    sub = {k: navs[k].reindex(idx).ffill() for k in w}
    months = idx.to_period("M")
    out = np.ones(len(idx))
    lvl = 1.0
    for m in months.unique():
        sel = np.where(months == m)[0]
        base = {k: sub[k].iloc[sel[0]] for k in w}
        seg = np.zeros(len(sel))
        for k, wt in w.items():
            seg += wt * (sub[k].iloc[sel].values / base[k])
        out[sel] = lvl * seg
        lvl = out[sel[-1]]
    return pd.Series(out, index=idx)


def stats_of(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    dd = float((nav / nav.cummax() - 1).min() * 100)
    return cagr, dd, (cagr / abs(dd) if dd < 0 else np.nan)


def main():
    d = load_daily()
    mm = build_monthly(d)

    # ---------------------------------------------------------- gold-only frontier
    print("=" * 118)
    print("A. THE ACTIONABLE-TODAY SUBSET - weight vectors that use NO unproven sleeve")
    print("   (IPO has zero live or paper history; gold is a listed ETF we can buy tomorrow)")
    print("=" * 118)
    fr = pd.read_csv(RES / "p7_frontier_OA_TN_IPO_GOLD.csv")
    g = fr[(fr.ADMITTED) & (fr.w_IPO == 0)].sort_values("minCalmar", ascending=False)
    sh = ["weights"] + [f"{k}_{m}" for k in "ABC" for m in ("cagr", "dcagr", "dd", "calmar", "winb", "winc")]
    print(f"{len(g)} of the 197 admitted vectors contain NO IPO:")
    print(g[sh].to_string(index=False))

    # ------------------------------------------------ daily-marked robustness check
    print("\n" + "=" * 118)
    print("B. DAILY-MARKED ROBUSTNESS - the same books re-marked every day instead of at "
          "month end.")
    print("   Month-end marking (the convention of r/146-153, kept above for comparability) "
          "understates\n   drawdown because it cannot see an intra-month trough. Gold before "
          "2015 is the daily\n   reconstruction, whose DAILY correlation to real GOLDBEES is "
          "only 0.39 (COMEX-close vs\n   NSE-close timing), so the 2015+ panel is the "
          "trustworthy one and is marked *.")
    print("=" * 118)
    rows = []
    for pkey, lo in (("B", "2006-04-01"), ("C*", "2015-01-01")):
        idxall = None
        for s in ("OA", "TN", "IPO", "GOLD"):
            i = d[s if s != "GOLD" else "GOLD_full"].index
            idxall = i if idxall is None else idxall.intersection(i)
        idx = idxall[idxall >= pd.Timestamp(lo)]
        for name, w in BOOKS:
            cs, ds, ks = [], [], []
            for si in range(1, NSEED + 1, 3):          # 10 seeds x 12 offsets = 120 paths
                for off in range(NOFF):
                    navs = {"OA": d["OA"].iloc[:, si - 1], "TN": d["TN"].iloc[:, off],
                            "IPO": d["IPO"].iloc[:, si - 1],
                            "GOLD": d["GOLD_full"].iloc[:, 0]}
                    nav = blend_daily(navs, w, idx)
                    c, dd, k = stats_of(nav)
                    cs.append(c); ds.append(dd); ks.append(k)
            rows.append(dict(panel=pkey, book=name, window=f"{idx[0].date()}..{idx[-1].date()}",
                             cagr=round(float(np.median(cs)), 2),
                             dd_daily=round(float(np.median(ds)), 2),
                             dd_worst=round(float(np.min(ds)), 2),
                             calmar=round(float(np.median(ks)), 3), paths=len(cs)))
            print(f"  {pkey:>3} {name:<26} CAGR {rows[-1]['cagr']:6.2f}  "
                  f"DAILY MaxDD {rows[-1]['dd_daily']:7.2f} (worst {rows[-1]['dd_worst']:7.2f})  "
                  f"Calmar {rows[-1]['calmar']:5.3f}   [{rows[-1]['paths']} paths]", flush=True)
    pd.DataFrame(rows).to_csv(RES / "p8_daily_marked.csv", index=False)

    # ------------------------------------------------------------- YoY house table
    print("\n" + "=" * 118)
    print(f"C. YoY HOUSE-FORMAT TABLE - panel {YOY_PANEL} "
          f"({PANELS[YOY_PANEL]['label']}), after tax, net of 25 bps,")
    print("   median across 360 paired paths (30 OA seeds x 12 TN offsets).")
    print("=" * 118)
    pn = Panel(YOY_PANEL, mm)
    bench = d["OA"].index          # placeholder
    import sqlite3
    con = sqlite3.connect(str(ROOT / "backtest_data/market_data.db"))
    nb = pd.read_sql_query("select date, close from market_data_unified where "
                           "symbol='NIFTYBEES' and timeframe='day' order by date", con)
    con.close()
    nb["date"] = pd.to_datetime(nb["date"].str[:10])
    nb = nb.drop_duplicates("date").set_index("date")["close"].astype(float)
    nbm = nb.resample("ME").last()
    nbm.index = nbm.index.to_period("M")
    nbm = nbm.reindex(pn.months).ffill()

    yoy_ret, yoy_dd = {}, {}
    for name, w in BOOKS:
        if not set(w) <= set(pn.cfg["members"]):
            continue
        nav = pn.blend(w)
        med = np.median(nav, axis=1)
        s = pd.Series(med, index=pn.months)
        run = s.cummax()
        yr, dd = {}, {}
        for y, grp in s.groupby(s.index.year):
            i0 = s.index.get_loc(grp.index[0])
            prev = s.iloc[i0 - 1] if i0 > 0 else s.iloc[0]
            yr[y] = (grp.iloc[-1] / prev - 1) * 100
            seg = s.loc[grp.index]
            dd[y] = float((seg / run.loc[grp.index] - 1).min() * 100)
        yoy_ret[name], yoy_dd[name] = yr, dd
    srun = nbm.cummax()
    byr, bdd = {}, {}
    for y, grp in nbm.groupby(nbm.index.year):
        i0 = nbm.index.get_loc(grp.index[0])
        prev = nbm.iloc[i0 - 1] if i0 > 0 else nbm.iloc[0]
        byr[y] = (grp.iloc[-1] / prev - 1) * 100
        bdd[y] = float((grp / srun.loc[grp.index] - 1).min() * 100)
    yoy_ret["NIFTY 50 (NIFTYBEES)"] = byr
    yoy_dd["NIFTY 50 (NIFTYBEES)"] = bdd

    names = [n for n, _ in BOOKS if n in yoy_ret] + ["NIFTY 50 (NIFTYBEES)"]
    strat = [n for n, _ in BOOKS if n in yoy_ret]
    yrs = sorted(yoy_ret[names[0]])
    hdr = f"{'Year':<6}" + "".join(f"{n[:24]:>26}" for n in names) + \
        f"{'BEST CAGR':>26}{'LEAST DD':>26}{'BEST OVERALL':>26}"
    print(hdr)
    lines = []
    for y in yrs:
        cells = "".join(f"{yoy_ret[n][y]:>+15.1f} ({yoy_dd[n][y]:>+6.1f}) " for n in names)
        bc = max(strat, key=lambda n: yoy_ret[n][y])
        bd = max(strat, key=lambda n: yoy_dd[n][y])
        bo = max(strat, key=lambda n: yoy_ret[n][y] + yoy_dd[n][y])
        print(f"{y:<6}{cells}{bc[:24]:>26}{bd[:24]:>26}{bo[:24]:>26}")
        lines.append(dict(year=y, **{f"{n} ret": round(yoy_ret[n][y], 1) for n in names},
                          **{f"{n} dd": round(yoy_dd[n][y], 1) for n in names},
                          best_cagr=bc, least_dd=bd, best_overall=bo))
    summ = {}
    for name, w in BOOKS:
        if name not in yoy_ret:
            continue
        c, dd, k = path_stats(pn.blend(w), pn.years)
        summ[name] = (float(np.median(c)), float(np.median(dd)), float(np.nanmedian(k)))
    byrs = (pn.months[-1].to_timestamp() - pn.months[0].to_timestamp()).days / 365.25
    bc_ = ((nbm.iloc[-1] / nbm.iloc[0]) ** (1 / byrs) - 1) * 100
    bdd_ = float((nbm / nbm.cummax() - 1).min() * 100)
    summ["NIFTY 50 (NIFTYBEES)"] = (bc_, bdd_, bc_ / abs(bdd_))
    print(f"{'FULL':<6}" + "".join(
        f"{summ[n][0]:>+9.2f}/{summ[n][1]:>7.2f}/{summ[n][2]:>5.2f} " for n in names))
    print("      (cells = annual return with intra-year max drawdown from the running peak "
          "beneath; FULL row = CAGR / MaxDD / Calmar)")
    pd.DataFrame(lines).to_csv(RES / "p8_yoy.csv", index=False)
    pd.DataFrame(summ, index=["cagr", "maxdd", "calmar"]).T.to_csv(RES / "p8_yoy_summary.csv")

    # ------------------------------------------------------------------- the figure
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [2.4, 1]})
    colors = {"TN+OA 50-50 (deployed)": "#8892b0", "OA45 TN35 IPO10 GOLD10": "#4cc9f0",
              "OA40 TN25 IPO20 GOLD15": "#f9c74f", "OA+IPO+GOLD 33/33/33": "#90be6d"}
    x = pn.months.to_timestamp()
    for name, w in BOOKS:
        if name not in yoy_ret:
            continue
        med = np.median(pn.blend(w), axis=1) * 100
        ax[0].plot(x, med, label=f"{name}  ({summ[name][0]:.1f}% / {summ[name][1]:.1f}% / "
                                 f"Calmar {summ[name][2]:.2f})", lw=1.9, color=colors[name])
        s = pd.Series(med, index=x)
        ax[1].plot(x, (s / s.cummax() - 1) * 100, lw=1.2, color=colors[name])
    b = (nbm / nbm.iloc[0] * 100).values
    ax[0].plot(x, b, label=f"NIFTY 50 (NIFTYBEES)  ({bc_:.1f}% / {bdd_:.1f}%)",
               lw=1.5, color="#d1495b", ls="--")
    sb = pd.Series(b, index=x)
    ax[1].plot(x, (sb / sb.cummax() - 1) * 100, lw=1.0, color="#d1495b", ls="--")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of Rs 100 (log)")
    ax[0].set_title("research/154 - multi-sleeve blends vs the deployed TN+OA pair\n"
                    f"{pn.months[0]} to {pn.months[-1]}, after tax, net of 25 bps/side, "
                    f"median of 360 paired paths (30 OA seeds x 12 TN offsets), monthly rebalanced")
    ax[0].legend(loc="upper left", fontsize=9); ax[0].grid(alpha=.25, which="both")
    ax[1].set_ylabel("Drawdown %"); ax[1].grid(alpha=.25)
    ax[1].axhline(0, color="k", lw=.6)
    plt.tight_layout()
    plt.savefig(RES / "multi_system_blends_research154.png", dpi=130)
    print("\nfigure written: results/multi_system_blends_research154.png")
    print("REPORT DONE")


if __name__ == "__main__":
    main()
