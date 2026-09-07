"""r/156 P7 — the report package: YoY house-format table + growth-of-100 chart with DD panel.

House format (project CLAUDE.md, 2026-09-04): one column per system AND per blend, plus the
benchmark; each year-cell = the annual return with the intra-year max drawdown beneath it;
three best-of columns on the right (BEST CAGR / LEAST DD / BEST OVERALL), benchmarks excluded
from the picks; summary row with full-period CAGR / MaxDD / Calmar.

Every drawdown here is measured from the running peak of the FULL curve.

Outputs: p7_yoy.csv, p7_yoy.json, sector_rotation_research156.png
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
R153 = C.ROOT / "research/153_ipo_base/results/ipo_equity_seeds.csv"
R154 = C.ROOT / "research/154_multi_system_blends/results"

LABELS = {
    "BEST_CALMAR": "Sector rotation (best Calmar)",
    "BEST_CAGR": "Sector rotation (best CAGR)",
    "R147_SECROT": "r/147 SECROT cell",
    "EW_ALL": "Equal-weight all 9 sectors",
    "BRANCHB": "Sector-gated stock book",
    "NOSECT": "Same stocks, NO sector filter",
}
BENCH_LABELS = {"NIFTY500": "NIFTY 500", "NIFTY50": "NIFTY 50",
                "NIFTYMIDCAP150": "Midcap 150", "NIFTYSMLCAP250": "Smallcap 250"}


def med_of(df, tag):
    cols = [c for c in df.columns if c.startswith(tag + "_off")]
    return df[cols].median(axis=1).dropna() if cols else None


def blend_navs(navs, weights):
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
    cand = pd.read_csv(C.RES / "p5_navs.csv", index_col=0, parse_dates=True)
    sect, _ = C.load_close(C.SECT9 + C.BENCHES, C.SECT_START, C.END)
    cal = C.trading_calendar(C.SECT_START, C.END)
    sect = sect.reindex(cal).ffill(limit=2)
    oa = pd.read_csv(R154 / "oa_navs30.csv", index_col=0, parse_dates=True).median(axis=1)
    tn = pd.read_csv(R154 / "tn_navs12.csv", index_col=0, parse_dates=True).median(axis=1)
    ipo = pd.read_csv(R153, index_col=0, parse_dates=True).median(axis=1)

    series = {}
    for tag, lab in LABELS.items():
        s = med_of(cand, tag)
        if s is not None and len(s) > 200:
            series[lab] = s
    series["True North (LIVE)"] = tn
    series["Open Alpha (LIVE)"] = oa
    series["Deployed TN40/OA40/IPO20"] = blend_navs(
        {"TN": tn, "OA": oa, "IPO": ipo}, {"TN": .40, "OA": .40, "IPO": .20})
    bench = {BENCH_LABELS[b]: sect[b].dropna() for b in BENCH_LABELS if b in sect.columns}

    # common window = the candidate start
    start = max(s.index[0] for s in series.values() if s is not None)
    start = max(start, min(s.index[0] for lab, s in series.items() if "Sector" in lab))
    cs = {k: v.loc[v.index >= start] for k, v in series.items()}
    cb = {k: v.loc[v.index >= start] for k, v in bench.items()}
    log(f"YoY common window {start.date()} -> {list(cs.values())[0].index[-1].date()}")

    allser = {**cs, **cb}
    years = sorted({y for s in allser.values() for y in s.index.year})
    yoy_rows = []
    peaks = {k: v.cummax() for k, v in allser.items()}
    per = {}
    for k, s in allser.items():
        per[k] = C.yearly(s)
    syscols = list(cs)
    for y in years:
        row = {"Year": y}
        for k in allser:
            if y in per[k]:
                r, d = per[k][y]
                row[k] = f"{r:+.1f} ({d:.1f})"
        cands = {k: per[k][y] for k in syscols if y in per[k]}
        if cands:
            row["BEST CAGR"] = max(cands, key=lambda k: cands[k][0])
            row["LEAST DD"] = max(cands, key=lambda k: cands[k][1])
            row["BEST OVERALL"] = max(cands, key=lambda k: cands[k][0] + cands[k][1])
        yoy_rows.append(row)
    summary = {"Year": "FULL PERIOD"}
    for k, s in allser.items():
        m = C.metrics(s)
        summary[k] = f"{m['cagr']:.1f}% / {m['maxdd']:.1f}% / {m['calmar']:.2f}"
    yoy_rows.append(summary)
    yoy = pd.DataFrame(yoy_rows)
    yoy.to_csv(C.RES / "p7_yoy.csv", index=False)
    log("\n" + yoy.to_string(index=False))

    # ------------------------------------------------------------------- the chart
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1]})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (k, s) in enumerate(cs.items()):
        g = 100 * s / s.iloc[0]
        ax[0].plot(g.index, g, lw=1.9, label=k, color=colors[i % 10])
        dd = (s / s.cummax() - 1) * 100
        ax[1].plot(dd.index, dd, lw=1.2, color=colors[i % 10])
    for k, s in cb.items():
        g = 100 * s / s.iloc[0]
        ax[0].plot(g.index, g, lw=1.1, ls="--", color="#888", alpha=.8, label=k)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Growth of Rs 100 (log)")
    ax[0].legend(fontsize=8, ncol=2, loc="upper left")
    ax[0].grid(alpha=.25)
    ax[0].set_title(f"research/156 — sector trend detection vs the deployed book "
                    f"({start.date()} to {list(cs.values())[0].index[-1].date()}), "
                    f"after tax, 25 bps/side, offset-ensemble medians", fontsize=11)
    ax[1].set_ylabel("Drawdown (%)")
    ax[1].grid(alpha=.25)
    plt.tight_layout()
    out = C.RES / "sector_rotation_research156.png"
    plt.savefig(out, dpi=110)
    log(f"chart written {out}")

    json.dump({"columns": list(yoy.columns), "rows": yoy.astype(str).values.tolist()},
              open(C.RES / "p7_yoy.json", "w"), indent=1)


if __name__ == "__main__":
    main()
