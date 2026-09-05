"""research/155 Phase R — replication gate.

(a) Reproduce research/153's published 40/40/20 TN+OA+IPO blend (28.27% CAGR / -12.79% MaxDD
    / Calmar 2.21) from the SAME cached inputs it used: r/146's 10 OA seeds x 3 TN offsets
    (off0/4/8) x the first 10 IPO seeds, monthly rebalanced, unpaired medians.
(b) Re-state the same blend on THIS study's 30 PAIRED paths (30 OA seeds from r/154 x 12 TN
    offsets cycled x IPO seed matched to the OA seed) so the paired baseline is on the record.
(c) Re-measure the sub-window drawdowns with the r/154 FULL-CURVE-PEAK convention and show how
    far the old window-slice convention was out.

If (a) does not reproduce, the study STOPS here and reports the discrepancy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/arun/quantifyd")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
R146 = ROOT / "research" / "146_complementary_third_sleeve" / "results"
R153 = ROOT / "research" / "153_ipo_base" / "results"
R154 = ROOT / "research" / "154_multi_system_blends" / "results"
RES = Path(__file__).resolve().parents[1] / "results"


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    dd = float((nav / nav.cummax() - 1).min() * 100)
    return cagr, dd


def wdd_slice(nav, a, b):
    seg = nav[(nav.index >= a) & (nav.index <= b)]
    return float((seg / seg.cummax() - 1).min() * 100) if len(seg) else np.nan


def wdd_full(nav, a, b):
    d = nav / nav.cummax() - 1.0
    seg = d[(d.index >= a) & (d.index <= b)]
    return float(seg.min() * 100) if len(seg) else np.nan


def monthly(df):
    return df.resample("ME").last().pct_change().fillna(0.0)


ipo = pd.read_csv(R153 / "ipo_equity_seeds.csv", index_col=0, parse_dates=True)
oa10 = pd.read_csv(R146 / "oa_navs.csv", index_col=0, parse_dates=True)
tn3 = pd.concat([pd.read_csv(R146 / f"tn_nav_off{o}.csv", index_col=0, parse_dates=True)
                 .rename(columns={"0": f"off{o}"}) for o in (0, 4, 8)], axis=1)
oa30 = pd.read_csv(R154 / "oa_navs30.csv", index_col=0, parse_dates=True)
tn12 = pd.read_csv(R154 / "tn_navs12.csv", index_col=0, parse_dates=True)

print("=" * 96)
print("(a) REPLICATION of research/153's published 40/40/20 blend "
      "(r/146 inputs, unpaired medians)")
print("=" * 96)
idx = ipo.index.intersection(oa10.index).intersection(tn3.index)
i_, o_, t_ = (x.loc[idx] for x in (ipo, oa10, tn3))
i_, o_, t_ = (x / x.iloc[0] for x in (i_, o_, t_))
im, om, tm = monthly(i_), monthly(o_), monthly(t_)
print(f"common window {idx[0].date()} -> {idx[-1].date()} ({len(idx)} days)")

for w in (0.0, 0.10, 0.20):
    cs, ds, s08, s18, s20, s22, f08 = [], [], [], [], [], [], []
    for oc in om.columns:
        for tc in tm.columns:
            for ic in list(im.columns)[:10]:
                r = (1 - w) / 2 * (om[oc] + tm[tc]) + (w * im[ic] if w else 0.0)
                bl = (1 + r).cumprod()
                c_, d_ = stats(bl)
                cs.append(c_); ds.append(d_)
                s08.append(wdd_slice(bl, "2008-01-01", "2008-12-31"))
                f08.append(wdd_full(bl, "2008-01-01", "2008-12-31"))
                s18.append(wdd_slice(bl, "2018-01-01", "2018-12-31"))
                s20.append(wdd_slice(bl, "2020-01-01", "2020-12-31"))
                s22.append(wdd_slice(bl, "2022-01-01", "2022-06-30"))
                if w == 0.0:
                    break                      # cash weight 0 -> IPO seed irrelevant
    cm, dm = float(np.median(cs)), float(np.median(ds))
    print(f"  w_IPO {w*100:4.0f}%  n={len(cs):4d}  CAGR {cm:6.2f}  MaxDD {dm:7.2f}  "
          f"Calmar {cm/abs(dm):5.2f} | 2008 slice {np.median(s08):6.2f} "
          f"FULL-peak {np.median(f08):6.2f} | 2018 {np.median(s18):6.2f} "
          f"2020 {np.median(s20):6.2f} 22H1 {np.median(s22):6.2f}")

print("\n  research/153 published: w=0%  27.14 / -16.42 / 1.65 ;  "
      "w=10%  27.72 / -14.44 / 1.92 ;  w=20%  28.27 / -12.79 / 2.21")

print()
print("=" * 96)
print("(b) THE PAIRED BASELINE used by research/155 "
      "(30 paths: OA seed s(p+1) x TN off(p mod 12) x IPO seed p+1)")
print("=" * 96)
idx2 = ipo.index.intersection(oa30.index).intersection(tn12.index)
print(f"common window {idx2[0].date()} -> {idx2[-1].date()} ({len(idx2)} days)")
rows = []
for p in range(30):
    oc, tc, ic = f"s{p+1}", f"off{p%12}", f"seed{p+1}"
    n_ = ipo[ic].loc[idx2]; n_ = n_ / n_.iloc[0]
    oo = oa30[oc].loc[idx2]; oo = oo / oo.iloc[0]
    tt = tn12[tc].loc[idx2]; tt = tt / tt.iloc[0]
    nm, oM, tM = monthly(n_), monthly(oo), monthly(tt)
    for tag, w in (("pair", 0.0), ("blend", 0.20)):
        r = (1 - w) / 2 * (oM + tM) + (w * nm if w else 0.0)
        bl = (1 + r).cumprod()
        c_, d_ = stats(bl)
        rows.append(dict(path=p, arm=tag, cagr=c_, dd=d_, calmar=c_ / abs(d_),
                         dd0809=wdd_full(bl, "2008-01-01", "2009-12-31"),
                         dd1214=wdd_full(bl, "2012-01-01", "2014-12-31"),
                         dd1314=wdd_full(bl, "2013-01-01", "2014-12-31"),
                         ret1314=float(bl[(bl.index >= "2013-01-01") &
                                          (bl.index <= "2014-12-31")].iloc[-1] /
                                       bl[(bl.index >= "2013-01-01")].iloc[0] - 1) * 100))
df = pd.DataFrame(rows)
df.to_csv(RES / "r_baseline_paths.csv", index=False)
for tag in ("pair", "blend"):
    s = df[df.arm == tag]
    print(f"  {tag:>6}  CAGR {s.cagr.median():6.2f} [{s.cagr.min():.2f}..{s.cagr.max():.2f}] "
          f" MaxDD {s.dd.median():7.2f} [{s.dd.min():.2f}..{s.dd.max():.2f}]  "
          f"Calmar {s.calmar.median():5.2f} | DD08-09 {s.dd0809.median():6.2f} "
          f"DD12-14 {s.dd1214.median():6.2f} DD13-14 {s.dd1314.median():6.2f}")
a = df[df.arm == "blend"].set_index("path")
b = df[df.arm == "pair"].set_index("path")
print(f"  paired: 40/40/20 beats the 50-50 pair on CAGR {int((a.cagr>b.cagr).sum())}/30, "
      f"on Calmar {int((a.calmar>b.calmar).sum())}/30, on MaxDD "
      f"{int((a.dd>b.dd).sum())}/30")
print("\nREPLICATION DONE")
