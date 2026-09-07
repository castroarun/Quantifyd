"""r/156 P6 - portfolio fit: correlation and blend value against the deployed book.

Baseline: True North 40 / Open Alpha 40 / IPO-base 20, monthly rebalanced, after-tax sleeve NAVs
cached by r/154 and r/153. Candidate is added by scaling the three incumbents down pro rata.

EVERY comparison is computed on the CANDIDATE'S OWN window - the baseline and the cash null are
re-run on the identical index. Mixing windows is the error r/152 was caught on.

Paths are PAIRED: TN offset i, OA seed i, IPO seed i, candidate offset (i mod 4). Reported as
median [min..max] across paths, never a point.

Adoption bar (pre-registered): +0.10 Calmar or -2pp drawdown at >= equal CAGR, beating the cash
null at the same weight, correlation < 0.40 to both TN and OA.

Outputs: p6_corr.csv, p6_blend.csv
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import common as C

log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
R153 = C.ROOT / "research/153_ipo_base/results/ipo_equity_seeds.csv"
R154 = C.ROOT / "research/154_multi_system_blends/results"
WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BASE = dict(TN=0.40, OA=0.40, IPO=0.20)
N = 12


def rebal_blend(navs, weights, freq="ME"):
    idx = None
    for s in navs.values():
        idx = s.dropna().index if idx is None else idx.intersection(s.dropna().index)
    rets = {k: navs[k].reindex(idx).pct_change().fillna(0) for k in navs}
    marks = set(pd.DatetimeIndex(pd.Series(idx, index=idx).resample(freq).last().dropna().values))
    parts = dict(weights)
    out = []
    for d in idx:
        for k in parts:
            parts[k] *= (1 + rets[k].loc[d])
        cur = sum(parts.values())
        out.append(cur)
        if d in marks:
            parts = {k: cur * weights[k] for k in weights}
    return pd.Series(out, index=idx)


def main():
    t0 = time.time()
    oa = pd.read_csv(R154 / "oa_navs30.csv", index_col=0, parse_dates=True)
    tn = pd.read_csv(R154 / "tn_navs12.csv", index_col=0, parse_dates=True)
    ipo = pd.read_csv(R153, index_col=0, parse_dates=True)
    cand = pd.read_csv(C.RES / "p5_navs.csv", index_col=0, parse_dates=True)
    log(f"OA {oa.shape}; TN {tn.shape}; IPO {ipo.shape}; candidates {list(cand.columns)[:4]} ...")
    tags = sorted({c.rsplit("_off", 1)[0] for c in cand.columns})

    rows = []
    for tag in tags:
        cols = [c for c in cand.columns if c.startswith(tag + "_off")]
        cs = cand[cols].median(axis=1).dropna()
        for name, ref in (("TN", tn.median(axis=1)), ("OA", oa.median(axis=1)),
                          ("IPO", ipo.median(axis=1))):
            j = cs.index.intersection(ref.dropna().index)
            if len(j) < 200:
                continue
            a, b = cs.reindex(j).pct_change().dropna(), ref.reindex(j).pct_change().dropna()
            k = a.index.intersection(b.index)
            ma = cs.reindex(j).resample("ME").last().pct_change().dropna()
            mb = ref.reindex(j).resample("ME").last().pct_change().dropna()
            mk = ma.index.intersection(mb.index)
            rows.append(dict(candidate=tag, vs=name, n_days=len(k),
                             start=str(j[0].date()), end=str(j[-1].date()),
                             corr_daily=round(float(a[k].corr(b[k])), 3),
                             corr_monthly=round(float(ma[mk].corr(mb[mk])), 3)))
    corr = pd.DataFrame(rows)
    corr.to_csv(C.RES / "p6_corr.csv", index=False)
    log("\n=== CORRELATION TO THE DEPLOYED SLEEVES (median paths) ===")
    log("\n" + corr.to_string(index=False))

    out = []
    for tag in tags:
        cols_t = [c for c in cand.columns if c.startswith(tag + "_off")]
        cwin = cand[cols_t].dropna(how="all").index
        for w in [0.0] + WEIGHTS:
            for variant in (["BASELINE"] if w == 0 else [tag, "CASHNULL"]):
                vals = []
                for i in range(N):
                    legs = {"TN": tn.iloc[:, i % tn.shape[1]],
                            "OA": oa.iloc[:, i % oa.shape[1]],
                            "IPO": ipo.iloc[:, i % ipo.shape[1]]}
                    legs = {k: v.loc[v.index.isin(cwin)] for k, v in legs.items()}
                    ww = {k: BASE[k] * (1 - w) for k in BASE}
                    if w > 0:
                        if variant == "CASHNULL":
                            idx0 = legs["TN"].dropna().index
                            legs["X"] = pd.Series(
                                (1 + C.CASH_YIELD) ** (np.arange(len(idx0)) / 252.0), index=idx0)
                        else:
                            legs["X"] = cand[cols_t[i % len(cols_t)]]
                        ww["X"] = w
                    nav = rebal_blend(legs, ww)
                    m = C.metrics(nav)
                    m["start"] = nav.index[0]
                    vals.append(m)
                out.append(dict(window_of=tag, candidate=variant, weight=w,
                                start=str(min(v["start"] for v in vals).date()),
                                cagr_med=round(np.median([v["cagr"] for v in vals]), 2),
                                cagr_min=round(min(v["cagr"] for v in vals), 2),
                                dd_med=round(np.median([v["maxdd"] for v in vals]), 2),
                                dd_worst=round(min(v["maxdd"] for v in vals), 2),
                                calmar_med=round(np.median([v["calmar"] for v in vals]), 2),
                                calmar_min=round(min(v["calmar"] for v in vals), 2)))
    bl = pd.DataFrame(out)
    bl.to_csv(C.RES / "p6_blend.csv", index=False)
    log("\n=== BLEND VALUE vs TN40/OA40/IPO20, each block on the CANDIDATE'S OWN window ===")
    for tag in tags:
        log(f"\n--- {tag} ---")
        log("\n" + bl[bl.window_of == tag].drop(columns=["window_of"]).to_string(index=False))
    log(f"P6 done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
