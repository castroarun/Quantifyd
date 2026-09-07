"""r/156 P1 (G1) — does sector leadership exist at all?

Builds the synthetic industry baskets, validates them against the real sector indices, then
measures the forward information coefficient of every signal spec on both asset sets.

PRE-REGISTERED G1 GATE (STATUS doc §4.4): a family proceeds only if the monthly rank IC vs the
forward 1-month return has |t| >= 2.0 on BOTH asset sets, or |t| >= 2.5 on SECT9, AND the
top-minus-bottom tercile spread is monotone.

Outputs (results/): basket_validation.csv, sect9_close.csv, ind20_nav.csv, p1_ic.csv
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy import stats

import common as C

log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

IND2SECT = {
    "Information Technology": "NIFTYIT",
    "Healthcare": "NIFTYPHARMA",
    "Automobile and Auto Components": "NIFTYAUTO",
    "Fast Moving Consumer Goods": "NIFTYFMCG",
    "Metals & Mining": "NIFTYMETAL",
    "Realty": "NIFTYREALTY",
    "Financial Services": "NIFTYFINSRV",
    "Oil Gas & Consumable Fuels": "NIFTYENERGY",
}


def build_panels():
    log("loading sector indices ...")
    sect, _ = C.load_close(C.SECT9 + C.BENCHES, C.SECT_START, C.END)
    cal = C.trading_calendar(C.SECT_START, C.END)
    sect = sect.reindex(cal).ffill(limit=2)
    sect9 = sect[C.SECT9]
    sect9.to_csv(C.RES / "sect9_close.csv")

    log("loading 500-stock panel (2007+) ...")
    close, tv, imap = C.load_stock_panel(C.IND_START, C.END)
    log(f"  stock panel {close.shape}, {len(set(imap.values()))} industries")
    nav, cnt = C.build_industry_baskets(close, tv, imap)
    nav.to_csv(C.RES / "ind20_nav.csv")
    cnt.to_csv(C.RES / "ind20_counts.csv")
    log(f"  baskets built: {nav.shape[1]} industries, "
        f"{nav.index[nav.notna().any(axis=1)][0].date()} -> {nav.index[-1].date()}")

    log("computing breadth panels ...")
    brd50 = C.breadth(close, imap, 50)
    brd200 = C.breadth(close, imap, 200)
    brd50.to_csv(C.RES / "ind20_breadth50.csv")
    brd200.to_csv(C.RES / "ind20_breadth200.csv")
    return sect, sect9, nav, cnt, brd50, brd200, close, imap


def validate(sect, nav):
    rows = []
    for ind, sym in IND2SECT.items():
        if ind not in nav.columns or sym not in sect.columns:
            continue
        a = nav[ind].dropna()
        b = sect[sym].dropna()
        j = a.index.intersection(b.index)
        j = j[j >= pd.Timestamp("2015-01-01")]
        ra, rb = a.reindex(j).pct_change().dropna(), b.reindex(j).pct_change().dropna()
        k = ra.index.intersection(rb.index)
        ma = a.reindex(j).resample("ME").last().pct_change().dropna()
        mb = b.reindex(j).resample("ME").last().pct_change().dropna()
        mk = ma.index.intersection(mb.index)
        yrs = (j[-1] - j[0]).days / 365.25
        rows.append(dict(industry=ind, sector_index=sym, n_days=len(k),
                         corr_daily=round(float(ra[k].corr(rb[k])), 3),
                         corr_monthly=round(float(ma[mk].corr(mb[mk])), 3),
                         cagr_synth=round(((a.reindex(j).iloc[-1] / a.reindex(j).iloc[0])
                                           ** (1 / yrs) - 1) * 100, 2),
                         cagr_real=round(((b.reindex(j).iloc[-1] / b.reindex(j).iloc[0])
                                          ** (1 / yrs) - 1) * 100, 2)))
    df = pd.DataFrame(rows)
    df["drift_pp"] = (df.cagr_synth - df.cagr_real).round(2)
    df.to_csv(C.RES / "basket_validation.csv", index=False)
    log("basket validation vs real sector indices (2015+ overlap):")
    log("\n" + df.to_string(index=False))
    return df


def ic_table(px, label, brd50=None, brd200=None, windows=None):
    specs = C.signal_specs(with_breadth=brd50 is not None)
    me = C.rebal_dates(px.dropna(how="all").index, "M", 0)
    fwd = {}
    for h, nper in (("1m", 1), ("3m", 3)):
        p = px.reindex(me)
        fwd[h] = p.shift(-nper) / p - 1
    rows = []
    for name, (kind, L, skip) in specs.items():
        try:
            sig = C.compute_signal(px, kind, L, skip, brd50, brd200)
        except Exception as e:
            log(f"  {label} {name}: signal failed {e}")
            continue
        s = sig.reindex(me)
        for h in ("1m", "3m"):
            f = fwd[h]
            for wname, wmask in windows.items():
                ss, ff = s.loc[wmask], f.loc[wmask]
                ics, t1, t3, t2 = [], [], [], []
                for d in ss.index:
                    a, b = ss.loc[d], ff.loc[d]
                    ok = a.notna() & b.notna()
                    if ok.sum() < 5:
                        continue
                    aa, bb = a[ok], b[ok]
                    ics.append(stats.spearmanr(aa, bb).correlation)
                    q = aa.rank(pct=True)
                    t1.append(bb[q > 2 / 3].mean())
                    t2.append(bb[(q > 1 / 3) & (q <= 2 / 3)].mean())
                    t3.append(bb[q <= 1 / 3].mean())
                ics = np.array([x for x in ics if np.isfinite(x)])
                if len(ics) < 12:
                    continue
                t = stats.ttest_1samp(ics, 0.0)
                per = 12 if h == "1m" else 4
                ann = lambda arr: (np.nanmean(arr) * per * 100)
                a1, a2, a3 = ann(t1), ann(t2), ann(t3)
                rows.append(dict(assets=label, signal=name, horizon=h, window=wname,
                                 n_obs=len(ics), ic_mean=round(float(ics.mean()), 4),
                                 ic_t=round(float(t.statistic), 2),
                                 ic_hit=round(float((ics > 0).mean()) * 100, 1),
                                 top_ann=round(a1, 2), mid_ann=round(a2, 2),
                                 bot_ann=round(a3, 2), spread_ann=round(a1 - a3, 2),
                                 monotone=bool(a1 > a2 > a3)))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    sect, sect9, nav, cnt, brd50, brd200, close, imap = build_panels()
    validate(sect, nav)

    out = []
    # --- SECT9 (real indices, 2015+)
    idx = sect9.dropna(how="all").index
    me = C.rebal_dates(idx, "M", 0)
    mid = me[len(me) // 2]
    w = {"full": me, "half1": me[me < mid], "half2": me[me >= mid]}
    out.append(ic_table(sect9, "SECT9", windows=w))
    log(f"SECT9 IC done ({time.time()-t0:.0f}s)")

    # --- IND20 (synthetic, 2007+)
    navc = nav.loc[nav.notna().sum(axis=1) >= 8]
    me2 = C.rebal_dates(navc.index, "M", 0)
    mid2 = me2[len(me2) // 2]
    w2 = {"full": me2, "half1": me2[me2 < mid2], "half2": me2[me2 >= mid2]}
    out.append(ic_table(navc, "IND20", brd50.reindex(navc.index)[navc.columns],
                        brd200.reindex(navc.index)[navc.columns], windows=w2))
    log(f"IND20 IC done ({time.time()-t0:.0f}s)")

    # --- IND20 restricted to the SECT9-comparable window, to separate window from universe
    navw = navc.loc[navc.index >= pd.Timestamp(C.SECT_START)]
    me3 = C.rebal_dates(navw.index, "M", 0)
    out.append(ic_table(navw, "IND20_2015", brd50.reindex(navw.index)[navw.columns],
                        brd200.reindex(navw.index)[navw.columns],
                        windows={"full": me3}))

    df = pd.concat(out, ignore_index=True)
    df.to_csv(C.RES / "p1_ic.csv", index=False)
    log(f"p1_ic.csv written: {len(df)} cells ({time.time()-t0:.0f}s)")

    log("\n=== G1 SCREEN: 1-month horizon, full window ===")
    f = df[(df.horizon == "1m") & (df.window == "full")]
    piv = f.pivot_table(index="signal", columns="assets",
                        values=["ic_t", "spread_ann", "monotone"])
    log("\n" + piv.round(2).to_string())

    log("\n=== PRE-REGISTERED G1 PASSES ===")
    s9 = f[f.assets == "SECT9"].set_index("signal")
    i20 = f[f.assets == "IND20"].set_index("signal")
    passes = []
    for sig in s9.index:
        a = s9.loc[sig]
        b = i20.loc[sig] if sig in i20.index else None
        cond = (abs(a.ic_t) >= 2.5) or (b is not None and abs(a.ic_t) >= 2.0
                                        and abs(b.ic_t) >= 2.0)
        if cond and a.monotone:
            passes.append(sig)
    for sig in i20.index:
        if sig in s9.index:
            continue
        b = i20.loc[sig]
        if abs(b.ic_t) >= 2.5 and b.monotone:
            passes.append(sig + " (IND20-only)")
    log("PASS: " + (", ".join(passes) if passes else "NONE — G1 KILL"))
    pd.Series(passes, name="g1_pass").to_csv(C.RES / "p1_g1_passes.csv", index=False)


if __name__ == "__main__":
    main()
