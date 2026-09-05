"""research/153 G4 — correlation and blend value against the DEPLOYED book.

Baselines reused (not recomputed) from research/146, all after-tax, cash-yield modelled:
  results/oa_navs.csv        Open Alpha adopted spec, 10 selection seeds
  results/tn_nav_off{0,4,8}  True North incumbent spec, 3 rebalance-day offsets
  results/nav_cashnull_tax1  the cash sleeve, for the cash-null test
IPO sleeve: results/ipo_equity_seeds.csv (30 seeds, this study).

Reported as medians across the sleeve seed/offset grid, never a single path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
R146 = Path("/home/arun/quantifyd/research/146_complementary_third_sleeve/results")

ipo = pd.read_csv(RES / "ipo_equity_seeds.csv", index_col=0, parse_dates=True)
oa = pd.read_csv(R146 / "oa_navs.csv", index_col=0, parse_dates=True)
tn = pd.concat([pd.read_csv(R146 / f"tn_nav_off{o}.csv", index_col=0,
                            parse_dates=True).rename(columns={"0": f"off{o}"})
                for o in (0, 4, 8)], axis=1)
cash = pd.read_csv(R146 / "nav_cashnull_tax1.csv", index_col=0, parse_dates=True)
cash.columns = ["cash"]

idx = ipo.index.intersection(oa.index).intersection(tn.index).intersection(cash.index)
print(f"common window {idx[0].date()} -> {idx[-1].date()} ({len(idx)} days)")
ipo, oa, tn, cash = (x.loc[idx] for x in (ipo, oa, tn, cash))


def norm(df):
    return df / df.iloc[0]


ipo, oa, tn, cash = (norm(x) for x in (ipo, oa, tn, cash))
ipo_m = ipo.resample("ME").last().pct_change().fillna(0.0)
oa_m = oa.resample("ME").last().pct_change().fillna(0.0)
tn_m = tn.resample("ME").last().pct_change().fillna(0.0)
cash_m = cash.resample("ME").last().pct_change().fillna(0.0)
ipo_d, oa_d, tn_d = (x.pct_change().fillna(0.0) for x in (ipo, oa, tn))


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    dd = float((nav / nav.cummax() - 1).min() * 100)
    return cagr, dd


def wdd(nav, a, b):
    seg = nav[(nav.index >= a) & (nav.index <= b)]
    return float((seg / seg.cummax() - 1).min() * 100) if len(seg) else np.nan


def wret(nav, a, b):
    seg = nav[(nav.index >= a) & (nav.index <= b)]
    return float((seg.iloc[-1] / seg.iloc[0] - 1) * 100) if len(seg) > 1 else np.nan


print("\n=== STANDALONE (medians over the sleeve's own seeds/offsets) ===")
for tag, df in (("IPO base (30 seeds)", ipo), ("Open Alpha (10 seeds)", oa),
                ("True North (3 offsets)", tn), ("Cash sleeve", cash)):
    cs = [stats(df[c])[0] for c in df.columns]
    ds = [stats(df[c])[1] for c in df.columns]
    print(f"  {tag:<24} CAGR {np.median(cs):6.2f} [{min(cs):.2f}..{max(cs):.2f}]  "
          f"DD {np.median(ds):7.2f}%  Calmar {np.median(cs)/abs(np.median(ds)):5.2f}")

print("\n=== CORRELATIONS (daily / monthly), median over seed pairs ===")


def corr(a, b, monthly=False):
    A = (a.resample("ME").last().pct_change() if monthly else a.pct_change()).fillna(0)
    B = (b.resample("ME").last().pct_change() if monthly else b.pct_change()).fillna(0)
    vals = [A[i].corr(B[j]) for i in A.columns for j in B.columns]
    return float(np.median(vals))


print(f"  IPO vs Open Alpha : {corr(ipo, oa):.3f} daily / {corr(ipo, oa, True):.3f} monthly")
print(f"  IPO vs True North : {corr(ipo, tn):.3f} daily / {corr(ipo, tn, True):.3f} monthly")
print(f"  OA  vs True North : {corr(oa, tn):.3f} daily / {corr(oa, tn, True):.3f} monthly")

# ── blends, monthly rebalanced ──
print("\n=== 3-SLEEVE BLEND, monthly rebalanced (TN/OA split equally, IPO weight w) ===")
print(f"{'w_IPO':>6} | {'CAGR med [min..max]':>24} | {'DD':>7} | {'Calmar':>6} | "
      f"{'2008':>7} {'2018':>7} {'2020':>7} {'22H1':>7}")
rows = []
for w in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33):
    cs, ds, w08, w18, w20, w22 = [], [], [], [], [], []
    for oc in oa_m.columns:
        for tc in tn_m.columns:
            for ic in list(ipo_m.columns)[:10]:
                r = (1 - w) / 2 * (oa_m[oc] + tn_m[tc]) + w * ipo_m[ic]
                bl = (1 + r).cumprod()
                c_, d_ = stats(bl)
                cs.append(c_); ds.append(d_)
                w08.append(wdd(bl, "2008-01-01", "2008-12-31"))
                w18.append(wdd(bl, "2018-01-01", "2018-12-31"))
                w20.append(wdd(bl, "2020-01-01", "2020-12-31"))
                w22.append(wdd(bl, "2022-01-01", "2022-06-30"))
    cm, dm = float(np.median(cs)), float(np.median(ds))
    rows.append(dict(w=w, cagr=round(cm, 2), cagr_lo=round(min(cs), 2),
                     cagr_hi=round(max(cs), 2), dd=round(dm, 2),
                     calmar=round(cm / abs(dm), 3), dd2008=round(float(np.median(w08)), 2),
                     dd2018=round(float(np.median(w18)), 2),
                     dd2020=round(float(np.median(w20)), 2),
                     dd2022h1=round(float(np.median(w22)), 2)))
    print(f"{w*100:5.0f}% | {cm:7.2f} [{min(cs):6.2f}..{max(cs):6.2f}] | {dm:6.2f}% | "
          f"{cm/abs(dm):6.2f} | {np.median(w08):6.2f}% {np.median(w18):6.2f}% "
          f"{np.median(w20):6.2f}% {np.median(w22):6.2f}%")

print("\n=== CASH-NULL: replace the IPO sleeve with plain cash at the same weight ===")
for w in (0.10, 0.20):
    cs, ds = [], []
    for oc in oa_m.columns:
        for tc in tn_m.columns:
            r = (1 - w) / 2 * (oa_m[oc] + tn_m[tc]) + w * cash_m["cash"]
            bl = (1 + r).cumprod()
            c_, d_ = stats(bl)
            cs.append(c_); ds.append(d_)
    cm, dm = float(np.median(cs)), float(np.median(ds))
    print(f"  cash at {w*100:.0f}%: CAGR {cm:.2f}%  DD {dm:.2f}%  Calmar {cm/abs(dm):.2f}")

pd.DataFrame(rows).to_csv(RES / "g4_blend.csv", index=False)

# ── 50-50 IPO vs OA and IPO vs TN pairs, for completeness ──
print("\n=== 50-50 PAIRS (monthly rebalanced) ===")
for tag, a_m in (("IPO+OA", oa_m), ("IPO+TN", tn_m)):
    cs, ds = [], []
    for ac in a_m.columns:
        for ic in list(ipo_m.columns)[:10]:
            bl = (1 + 0.5 * a_m[ac] + 0.5 * ipo_m[ic]).cumprod()
            c_, d_ = stats(bl)
            cs.append(c_); ds.append(d_)
    cm, dm = float(np.median(cs)), float(np.median(ds))
    print(f"  {tag}: CAGR {cm:.2f} [{min(cs):.2f}..{max(cs):.2f}]  DD {dm:.2f}%  "
          f"Calmar {cm/abs(dm):.2f}")

# ── YoY table data (house format) ──
print("\n=== YoY DATA (median across seeds; return with intra-year max DD) ===")
cols = {}
cols["IPO"] = ipo
cols["OA"] = oa
cols["TN"] = tn
bl5050 = pd.DataFrame({f"{oc}|{tc}": (1 + 0.5 * oa_m[oc] + 0.5 * tn_m[tc]).cumprod()
                       for oc in oa_m.columns for tc in tn_m.columns})
bl3 = pd.DataFrame({f"{oc}|{tc}|{ic}":
                    (1 + 0.45 * oa_m[oc] + 0.45 * tn_m[tc] + 0.10 * ipo_m[ic]).cumprod()
                    for oc in oa_m.columns for tc in tn_m.columns
                    for ic in list(ipo_m.columns)[:10]})
bl4 = pd.DataFrame({f"{oc}|{tc}|{ic}":
                    (1 + 0.40 * oa_m[oc] + 0.40 * tn_m[tc] + 0.20 * ipo_m[ic]).cumprod()
                    for oc in oa_m.columns for tc in tn_m.columns
                    for ic in list(ipo_m.columns)[:10]})
cols["TN+OA 50-50"] = bl5050
cols["TN+OA+IPO 45/45/10"] = bl3
cols["TN+OA+IPO 40/40/20"] = bl4
years = sorted(set(idx.year))
out = {}
for tag, df in cols.items():
    rr, dd_ = {}, {}
    for y in years:
        rr[y] = float(np.median([wret(df[c], f"{y}-01-01", f"{y}-12-31") for c in df.columns]))
        dd_[y] = float(np.median([wdd(df[c], f"{y}-01-01", f"{y}-12-31") for c in df.columns]))
    cs = [stats(df[c])[0] for c in df.columns]
    ds = [stats(df[c])[1] for c in df.columns]
    out[tag] = dict(ret=rr, dd=dd_, cagr=float(np.median(cs)), maxdd=float(np.median(ds)))
yoy = pd.DataFrame({t: pd.Series(v["ret"]) for t, v in out.items()})
yoydd = pd.DataFrame({t: pd.Series(v["dd"]) for t, v in out.items()})
yoy.round(2).to_csv(RES / "g4_yoy_returns.csv")
yoydd.round(2).to_csv(RES / "g4_yoy_intradd.csv")
print(yoy.round(1).to_string())
print("\nintra-year max DD:")
print(yoydd.round(1).to_string())
print("\nsummary:", {t: (round(v["cagr"], 2), round(v["maxdd"], 2)) for t, v in out.items()})
print("\nBLEND DONE")
