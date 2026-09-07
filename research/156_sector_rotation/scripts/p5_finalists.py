"""r/156 P5 — finalists: NAV export, plateau, two windows, cost ladder, per-year.

Ranking metric fixed in the STATUS doc before any run: offset-ensemble MEDIAN after-tax Calmar,
with median CAGR >= 20% and worst offset >= 18% as hard filters.

Also re-runs the r/147 SECROT cell verbatim (top-2 by 126d momentum, monthly, equal weight, no
gate) so the two studies can be compared like for like.

Outputs: p5_navs.csv, p5_finalists.csv, p5_costladder.csv, p5_windows.csv, p5_yearly.csv
"""
import sys, time, itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import common as C
from p1_ic import IND2SECT
from p2_rotation import build_targets, weights_for
import p4_stocks as P4

log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)
OFFSETS = [0, 1, 2, 3]


def load_sect():
    sect, _ = C.load_close(C.SECT9 + C.BENCHES, C.SECT_START, C.END)
    cal = C.trading_calendar(C.SECT_START, C.END)
    sect = sect.reindex(cal).ffill(limit=2)
    return sect


def run_rotation(px, bench, sname, n, weight, clock, gate, off, cost=C.COST, tax=True):
    idx = px.dropna(how="all").index
    ret = px.pct_change()
    vol = ret.rolling(63, min_periods=30).std() * np.sqrt(252)
    own_mom = px / px.shift(126) - 1
    b = bench.reindex(idx).ffill()
    mkt_ok = (b > b.rolling(200, min_periods=100).mean()).reindex(idx).fillna(True).to_dict()
    kind, L, skip = C.signal_specs()[sname]
    sig = C.compute_signal(px, kind, L, skip)
    rd = C.rebal_dates(idx, clock, off)
    rd = rd[rd >= idx[260]]
    tg = build_targets(px, sig.reindex(rd), vol.reindex(rd), rd, n, weight, gate,
                       own_mom.reindex(rd), mkt_ok)
    return C.Book(idx[idx >= rd[0]], px, cost=cost, tax=tax).run(tg)


def main():
    t0 = time.time()
    sect = load_sect()
    sect9 = sect[C.SECT9]
    bench = sect["NIFTY500"]

    p2 = pd.read_csv(C.RES / "p2_rotation.csv")
    g = (p2.groupby(["assets", "signal", "n", "weight", "clock", "gate"])
           .agg(cagr_med=("cagr", "median"), cagr_min=("cagr", "min"),
                dd_med=("maxdd", "median"), calmar_med=("calmar", "median"),
                calmar_min=("calmar", "min"), turn=("turnover_yr", "median"),
                expo=("exposure", "median")).reset_index())
    g.to_csv(C.RES / "p5_config_ensembles.csv", index=False)
    s9 = g[g.assets == "SECT9"].sort_values("calmar_med", ascending=False)
    log("\n=== SECT9 branch-A: top 15 configs by offset-median after-tax Calmar ===")
    log("\n" + s9.head(15).round(2).to_string(index=False))
    log("\n=== SECT9 branch-A: top 10 by median CAGR ===")
    log("\n" + s9.sort_values("cagr_med", ascending=False).head(10).round(2).to_string(index=False))
    log(f"\nSECT9 configs clearing the pre-registered bar (CAGR med>=20, min>=18, Calmar>=1.0): "
        f"{int(((s9.cagr_med>=20)&(s9.cagr_min>=18)&(s9.calmar_med>=1.0)).sum())} of {len(s9)}")
    i20 = g[g.assets == "IND20"].sort_values("calmar_med", ascending=False)
    if len(i20):
        log("\n=== IND20 (survivorship-inflated) top 10 by Calmar — reference only ===")
        log("\n" + i20.head(10).round(2).to_string(index=False))

    # ---- finalists: best SECT9 by Calmar, best by CAGR, and the r/147 SECROT cell
    picks = []
    top = s9.iloc[0]
    picks.append(("BEST_CALMAR", top.signal, int(top.n), top.weight, top.clock, top.gate))
    topc = s9.sort_values("cagr_med", ascending=False).iloc[0]
    picks.append(("BEST_CAGR", topc.signal, int(topc.n), topc.weight, topc.clock, topc.gate))
    picks.append(("R147_SECROT", "ABSMOM126", 2, "eq", "M", "none"))
    picks.append(("EW_ALL", None, None, None, "M", None))
    picks = list(dict.fromkeys(picks))

    navs, rows = {}, []
    for tag, sname, n, weight, clock, gate in picks:
        for off in OFFSETS:
            if tag == "EW_ALL":
                idx = sect9.dropna(how="all").index
                rd = C.rebal_dates(idx, "M", off)
                rd = rd[rd >= idx[260]]
                tg = {d: {c: 1.0 / sect9.loc[d].notna().sum() for c in sect9.columns
                          if np.isfinite(sect9.loc[d, c])} for d in rd}
                r = C.Book(idx[idx >= rd[0]], sect9).run(tg)
            else:
                r = run_rotation(sect9, bench, sname, n, weight, clock, gate, off)
            navs[f"{tag}_off{off}"] = r["nav"]
            m = C.metrics(r["nav"])
            rows.append(dict(tag=tag, signal=sname, n=n, weight=weight, clock=clock,
                             gate=gate, offset=off, turnover_yr=round(r["turnover_yr"], 2),
                             **{k: round(v, 3) for k, v in m.items()}))
    # ---- branch B finalists
    if (C.RES / "p4_books.csv").exists():
        p4 = pd.read_csv(C.RES / "p4_books.csv")
        gb = (p4.groupby(["src", "K", "slots", "stocksig", "clock"])
                .agg(cagr_med=("cagr", "median"), cagr_min=("cagr", "min"),
                     dd_med=("maxdd", "median"), calmar_med=("calmar", "median"))
                .reset_index().sort_values("calmar_med", ascending=False))
        gb.to_csv(C.RES / "p5_branchb_ensembles.csv", index=False)
        log("\n=== branch B: top 15 by offset-median Calmar ===")
        log("\n" + gb.head(15).round(2).to_string(index=False))
        close, tv, imap, elig, nav20, sectp = P4.load()
        idx = close.index
        start_i = idx[idx >= pd.Timestamp(P4.START)][260]
        sect_sources = {}
        kind, L, skip = P4.SECTSIG
        sect_sources["synth20"] = (C.compute_signal(nav20, kind, L, skip), None)
        real = sectp[[IND2SECT[i] for i in IND2SECT if IND2SECT[i] in sectp.columns]]
        sect_sources["real8"] = (C.compute_signal(real, kind, L, skip),
                                 {IND2SECT[i]: i for i in IND2SECT})
        best = gb.iloc[0]
        sc = P4.stock_scores(close, best.stocksig)
        for off in OFFSETS:
            rd = C.rebal_dates(idx, best.clock, off)
            rd = rd[rd >= start_i]
            srank, i2c = sect_sources[best.src]
            tg = P4.build(rd, srank.reindex(rd), int(best.K), imap, close, elig.reindex(rd),
                          sc.reindex(rd), int(best.slots), ind2col=i2c)
            r = C.Book(idx[idx >= rd[0]], close).run(tg)
            navs[f"BRANCHB_off{off}"] = r["nav"]
            m = C.metrics(r["nav"])
            rows.append(dict(tag="BRANCHB", signal=f"{best.src}/K{best.K}/{best.stocksig}",
                             n=int(best.K), weight="eq", clock=best.clock, gate="none",
                             offset=off, turnover_yr=round(r["turnover_yr"], 2),
                             **{k: round(v, 3) for k, v in m.items()}))
            tg = P4.build(rd, None, int(best.K), imap, close, elig.reindex(rd),
                          sc.reindex(rd), int(best.slots), no_sector=True)
            r = C.Book(idx[idx >= rd[0]], close).run(tg)
            navs[f"NOSECT_off{off}"] = r["nav"]
            m = C.metrics(r["nav"])
            rows.append(dict(tag="NOSECT", signal=f"fulluniv/{best.stocksig}", n=None,
                             weight="eq", clock=best.clock, gate="none", offset=off,
                             turnover_yr=round(r["turnover_yr"], 2),
                             **{k: round(v, 3) for k, v in m.items()}))
        log(f"branch-B finalist rerun done ({time.time()-t0:.0f}s)")

    pd.DataFrame(navs).to_csv(C.RES / "p5_navs.csv")
    fin = pd.DataFrame(rows)
    fin.to_csv(C.RES / "p5_finalists.csv", index=False)
    log("\n=== FINALIST ENSEMBLES (median [min..max] across 4 rebalance-day offsets) ===")
    log("\n" + fin.groupby("tag").agg(cagr_med=("cagr", "median"), cagr_min=("cagr", "min"),
                                      cagr_max=("cagr", "max"), dd_med=("maxdd", "median"),
                                      dd_worst=("maxdd", "min"),
                                      calmar_med=("calmar", "median"),
                                      calmar_min=("calmar", "min"),
                                      turn=("turnover_yr", "median")).round(2).to_string())

    # ---- cost ladder + gross/no-tax on the SECT9 finalists
    lad = []
    for tag, sname, n, weight, clock, gate in picks:
        if tag == "EW_ALL":
            continue
        for cost in (0.0025, 0.0040, 0.0060):
            for tax in (True, False):
                vals = []
                for off in OFFSETS:
                    r = run_rotation(sect9, bench, sname, n, weight, clock, gate, off,
                                     cost=cost, tax=tax)
                    vals.append(C.metrics(r["nav"]))
                lad.append(dict(tag=tag, cost_bps=int(cost * 10000), tax=tax,
                                cagr_med=round(np.median([v["cagr"] for v in vals]), 2),
                                dd_med=round(np.median([v["maxdd"] for v in vals]), 2),
                                calmar_med=round(np.median([v["calmar"] for v in vals]), 2)))
    pd.DataFrame(lad).to_csv(C.RES / "p5_costladder.csv", index=False)
    log("\n=== COST / TAX LADDER (offset medians) ===")
    log("\n" + pd.DataFrame(lad).to_string(index=False))

    # ---- two windows + per-year, drawdown always from the FULL curve peak
    wr, yr = [], []
    for name, s in navs.items():
        s = s.dropna()
        peak = s.cummax()
        for wname, a, b in (("H1", s.index[0], s.index[len(s) // 2]),
                            ("H2", s.index[len(s) // 2], s.index[-1]),
                            ("2018", "2018-01-01", "2018-12-31"),
                            ("2020crash", "2020-01-01", "2020-04-30"),
                            ("2022H1", "2022-01-01", "2022-06-30")):
            seg = s.loc[str(a):str(b)]
            if len(seg) < 20:
                continue
            yrs = (seg.index[-1] - seg.index[0]).days / 365.25
            wr.append(dict(series=name, window=wname,
                           ret=round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 2),
                           cagr=round(((seg.iloc[-1] / seg.iloc[0]) ** (1 / max(yrs, .01)) - 1) * 100, 2),
                           dd_from_full_peak=round((seg / peak.reindex(seg.index) - 1).min() * 100, 2)))
        for y, (rt, dd) in C.yearly(s).items():
            yr.append(dict(series=name, year=y, ret=round(rt, 2), dd=round(dd, 2)))
    pd.DataFrame(wr).to_csv(C.RES / "p5_windows.csv", index=False)
    pd.DataFrame(yr).to_csv(C.RES / "p5_yearly.csv", index=False)
    log(f"\nP5 done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
