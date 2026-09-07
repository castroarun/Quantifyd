"""r/156 P1b — is the IND20 momentum IC a sector effect, or survivorship in the construction?

The basket validation showed every synthetic industry basket out-drifting its real sector index
by +4 to +14 pp of CAGR per year. Three controls decide whether the IND20 IC means anything:

  A. SAME-8 HEAD-TO-HEAD. Restrict to the 8 industries that have a real sector index. Compute the
     identical IC on (i) the 8 REAL indices and (ii) the 8 SYNTHETIC baskets, over the identical
     2015+ window. Same sectors, same dates, same signals - only the construction differs.
  B. DRIFT-STRIPPED SYNTHETIC. Remove each basket's own full-sample mean daily return before
     computing forward returns. This is deliberately look-ahead, used only as a diagnostic: if
     the IC collapses, the "edge" is persistent per-basket drift, i.e. survivorship.
  C. RANDOM-INDUSTRY-LABEL NULL. Reshuffle the industry label of every stock (preserving
     industry sizes) and rebuild baskets, 50 draws. Real sector economics should vanish; a
     survivorship/size artefact should survive.

Outputs: p1b_same8.csv, p1b_drift.csv, p1b_shuffle.csv
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from scipy import stats

import common as C
from p1_ic import IND2SECT, ic_table

log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
SIGS = ["ABSMOM63", "ABSMOM126", "ABSMOM252", "MOMSKIP126", "RISKADJ63", "RISKADJ126",
        "MADIST200", "ACCEL"]


def ic_only(px, label, windows):
    df = ic_table(px, label, windows=windows)
    return df[df.signal.isin(SIGS)]


def main():
    t0 = time.time()
    sect, _ = C.load_close(C.SECT9, C.SECT_START, C.END)
    cal = C.trading_calendar(C.SECT_START, C.END)
    sect = sect.reindex(cal).ffill(limit=2)
    nav = pd.read_csv(C.RES / "ind20_nav.csv", index_col=0, parse_dates=True)

    inds = [i for i in IND2SECT if i in nav.columns]
    syms = [IND2SECT[i] for i in inds]
    log(f"same-8 head-to-head on {len(inds)} industries: {inds}")

    real8 = sect[syms].loc[sect.index >= pd.Timestamp("2015-01-01")]
    syn8 = nav[inds].reindex(real8.index).ffill(limit=2)
    me = C.rebal_dates(real8.dropna(how="all").index, "M", 0)
    w = {"full": me}

    a = ic_only(real8, "REAL8", w)
    b = ic_only(syn8, "SYNTH8", w)
    same8 = pd.concat([a, b], ignore_index=True)
    same8.to_csv(C.RES / "p1b_same8.csv", index=False)
    log("\n=== A. SAME 8 SECTORS, SAME WINDOW, REAL INDEX vs SYNTHETIC BASKET ===")
    p = same8[same8.horizon == "1m"].pivot_table(index="signal", columns="assets",
                                                 values=["ic_t", "spread_ann"])
    log("\n" + p.round(2).to_string())
    p3 = same8[same8.horizon == "3m"].pivot_table(index="signal", columns="assets",
                                                  values=["ic_t", "spread_ann"])
    log("\n(3-month horizon)\n" + p3.round(2).to_string())

    # --- B. drift-stripped synthetic (diagnostic, look-ahead by construction)
    log("\n=== B. IND20 WITH EACH BASKET'S OWN FULL-SAMPLE DRIFT REMOVED (diagnostic) ===")
    r = nav.pct_change()
    r_dm = r - r.mean()
    nav_dm = (1 + r_dm.fillna(0)).cumprod().where(nav.notna())
    navc = nav.loc[nav.notna().sum(axis=1) >= 8]
    nav_dm = nav_dm.reindex(navc.index)
    me2 = C.rebal_dates(navc.index, "M", 0)
    raw = ic_only(navc, "IND20_raw", {"full": me2})
    dm = ic_only(nav_dm, "IND20_driftstripped", {"full": me2})
    d = pd.concat([raw, dm], ignore_index=True)
    d.to_csv(C.RES / "p1b_drift.csv", index=False)
    log("\n" + d[d.horizon == "1m"].pivot_table(index="signal", columns="assets",
                                                values=["ic_t", "spread_ann"]).round(2).to_string())

    # --- C. shuffled industry labels
    log("\n=== C. RANDOM INDUSTRY-LABEL NULL (50 draws) ===")
    close, tv, imap = C.load_stock_panel(C.IND_START, C.END)
    syms_all = sorted(imap)
    labels = [imap[s] for s in syms_all]
    rows = []
    rng = np.random.default_rng(20260907)
    for k in range(50):
        perm = rng.permutation(labels)
        fake = dict(zip(syms_all, perm))
        fnav, _ = C.build_industry_baskets(close, tv, fake)
        fc = fnav.loc[fnav.notna().sum(axis=1) >= 8]
        mef = C.rebal_dates(fc.index, "M", 0)
        t = ic_only(fc, f"SHUF{k}", {"full": mef})
        t["draw"] = k
        rows.append(t)
        if k % 10 == 0:
            log(f"  shuffle draw {k}/50 ({time.time()-t0:.0f}s)")
    sh = pd.concat(rows, ignore_index=True)
    sh.to_csv(C.RES / "p1b_shuffle.csv", index=False)
    g = sh[sh.horizon == "1m"].groupby("signal").ic_t.describe(percentiles=[.05, .5, .95])
    log("\nshuffled-label IC t distribution (1m):\n" +
        g[["mean", "5%", "50%", "95%", "min", "max"]].round(2).to_string())
    log(f"\nreal-label IND20 IC t for comparison:\n" +
        raw[raw.horizon == "1m"].set_index("signal").ic_t.round(2).to_string())
    log(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
