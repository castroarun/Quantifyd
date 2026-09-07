"""r/156 ROSTER - every individual system in the book, side by side (doctrine section 9.1).

Systems: the live pair (Open Alpha, True North), the deployed blend, the tested-but-not-adopted
sleeves (IPO base, gold, VCP, multi-year breakout) and the two systems research/156 just tested
(sector rotation, sector-gated stocks). Each keeps its own start date, stated.

Emits results/roster.json: monthly growth-of-100 curves, drawdown paths, a YoY house table with
per-year return and intra-year drawdown (from the FULL curve peak), and a summary row.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import common as C

R151 = C.ROOT / "research/151_vcp_breakout/results/vcp_equity_seeds.csv"
R152 = C.ROOT / "research/152_multiyear_breakout/results/myb_equity_seeds.csv"
R153 = C.ROOT / "research/153_ipo_base/results/ipo_equity_seeds.csv"
R154 = C.ROOT / "research/154_multi_system_blends/results"

STATUS = {
    "Open Alpha": "LIVE",
    "True North": "LIVE",
    "Deployed blend TN40/OA40/IPO20": "LIVE (the book)",
    "IPO Base": "PAPER",
    "Gold (GOLDBEES)": "CANDIDATE - not adopted",
    "VCP breakout": "NO EDGE",
    "Multi-year breakout": "SIGNAL - not adopted",
    "Sector rotation (best)": "NO EDGE",
    "Sector-gated stocks": "NO ADDED VALUE",
    "NIFTY 500": "BENCHMARK",
    "Midcap 150": "BENCHMARK",
}
BENCH = {"NIFTY 500", "Midcap 150"}


def med(path):
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    return d.median(axis=1).dropna()


def blend(navs, weights):
    idx = None
    for s in navs.values():
        idx = s.dropna().index if idx is None else idx.intersection(s.dropna().index)
    rets = {k: navs[k].reindex(idx).pct_change().fillna(0) for k in navs}
    marks = set(pd.DatetimeIndex(pd.Series(idx, index=idx).resample("ME").last().dropna().values))
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
    oa, tn, ipo = med(R154 / "oa_navs30.csv"), med(R154 / "tn_navs12.csv"), med(R153)
    vcp, myb = med(R151), med(R152)
    gold = pd.read_csv(R154 / "gold_nav.csv", index_col=0, parse_dates=True)["close"].dropna()
    cand = pd.read_csv(C.RES / "p5_navs.csv", index_col=0, parse_dates=True)

    def cmed(tag):
        cols = [c for c in cand.columns if c.startswith(tag + "_off")]
        return cand[cols].median(axis=1).dropna()

    sect, _ = C.load_close(["NIFTY500", "NIFTYMIDCAP150"], C.SECT_START, C.END)
    cal = C.trading_calendar(C.SECT_START, C.END)
    sect = sect.reindex(cal).ffill(limit=2)

    S = {
        "Open Alpha": oa,
        "True North": tn,
        "Deployed blend TN40/OA40/IPO20": blend({"TN": tn, "OA": oa, "IPO": ipo},
                                                {"TN": .40, "OA": .40, "IPO": .20}),
        "IPO Base": ipo,
        "Gold (GOLDBEES)": gold,
        "VCP breakout": vcp,
        "Multi-year breakout": myb,
        "Sector rotation (best)": cmed("BEST_CAGR"),
        "Sector-gated stocks": cmed("BRANCHB"),
        "NIFTY 500": sect["NIFTY500"].dropna(),
        "Midcap 150": sect["NIFTYMIDCAP150"].dropna(),
    }

    curves, dds, summary, spans = {}, {}, {}, {}
    for k, s in S.items():
        s = s.dropna()
        m = s.resample("ME").last().dropna()
        g = 100 * m / m.iloc[0]
        curves[k] = {str(i.date())[:7]: round(float(v), 2) for i, v in g.items()}
        peak = m.cummax()
        dds[k] = {str(i.date())[:7]: round(float(v), 2)
                  for i, v in ((m / peak - 1) * 100).items()}
        mm = C.metrics(s)
        summary[k] = dict(cagr=round(mm["cagr"], 2), maxdd=round(mm["maxdd"], 2),
                          calmar=round(mm["calmar"], 2), status=STATUS[k])
        spans[k] = [str(s.index[0].date()), str(s.index[-1].date())]

    years = sorted({i.year for s in S.values() for i in s.index})
    per = {k: C.yearly(v.dropna()) for k, v in S.items()}
    yoy = []
    for y in years:
        row = {"year": y, "cells": {}}
        for k in S:
            if y in per[k]:
                r, d = per[k][y]
                row["cells"][k] = [round(r, 1), round(d, 1)]
        cands = {k: per[k][y] for k in S if k not in BENCH and y in per[k]}
        if cands:
            row["best_cagr"] = max(cands, key=lambda k: cands[k][0])
            row["least_dd"] = max(cands, key=lambda k: cands[k][1])
            row["best_overall"] = max(cands, key=lambda k: cands[k][0] + cands[k][1])
        yoy.append(row)

    out = dict(systems=list(S), status=STATUS, spans=spans, curves=curves, dd=dds,
               summary=summary, yoy=yoy,
               note=("Month-end marks, growth of Rs 100 from each system's own start. "
                     "After tax (20% STCG / 12.5% LTCG, Indian FY netting), 25 bps per side, "
                     "idle cash 5% p.a. Ensemble medians: 30 seeds for the slot-constrained "
                     "books, 12 rebalance-day offsets for True North, 4 offsets for the "
                     "research/156 books. Every drawdown is measured from the running peak of "
                     "the FULL curve, never from a window's first bar."))
    (C.RES / "roster.json").write_text(json.dumps(out), encoding="utf-8")
    print("roster.json written;", len(S), "systems")
    for k, v in summary.items():
        print(f"  {k:34s} {spans[k][0]} -> {spans[k][1]}  "
              f"{v['cagr']:6.2f}% / {v['maxdd']:7.2f}% / {v['calmar']:.2f}   {v['status']}")


if __name__ == "__main__":
    main()
