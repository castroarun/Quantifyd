"""r/156 P2 (G2) — branch A: rotation ACROSS sectors, full sweep + the four required nulls.

Signal shortlist declared BEFORE any book was run (G1 produced no clean SECT9 survivor, so the
shortlist is the eight most standard / best-behaved specs rather than a data-chosen set):
  ABSMOM63, ABSMOM126, ABSMOM252, MOMSKIP126, RISKADJ63, RISKADJ126, MADIST200, ACCEL

Grid: 8 signals x N{1..5} x weighting{eq,rank,invvol,sigprop} x clock{M,Q,F} x gate{none,absmom,
mkt} = 1,440 configs, each on 4 rebalance-day offsets = 5,760 runs.

Nulls: equal-weight-all-sectors, NIFTY500 buy-and-hold, 500-draw random-N-sector null, cash-null.
All after 25 bps/side, after Indian FY-netted tax, idle cash 5%.

Resume-safe: rows already present in the output CSV are skipped.
"""
import sys, time, itertools, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import common as C

log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

SIGS = ["ABSMOM63", "ABSMOM126", "ABSMOM252", "MOMSKIP126", "RISKADJ63", "RISKADJ126",
        "MADIST200", "ACCEL"]
NS = [1, 2, 3, 4, 5]
WEIGHTS = ["eq", "rank", "invvol", "sigprop"]
CLOCKS = ["M", "Q", "F"]
GATES = ["none", "absmom", "mkt"]
OFFSETS = [0, 1, 2, 3]
OUT = C.RES / "p2_rotation.csv"
FIELDS = ["assets", "signal", "n", "weight", "clock", "gate", "offset", "cagr", "maxdd",
          "calmar", "sharpe", "vol", "turnover_yr", "exposure"]


def weights_for(scores, vol, kind, n):
    top = scores.dropna().sort_values(ascending=False).head(n)
    if len(top) == 0:
        return {}
    if kind == "eq":
        w = pd.Series(1.0, index=top.index)
    elif kind == "rank":
        w = pd.Series(np.arange(len(top), 0, -1), index=top.index, dtype=float)
    elif kind == "invvol":
        v = vol.reindex(top.index).replace(0, np.nan)
        w = (1.0 / v).fillna(v.mean())
    else:                                        # sigprop
        z = top - top.min()
        w = z + z.mean() * 0.1 if z.sum() > 0 else pd.Series(1.0, index=top.index)
    w = w / w.sum()
    return w.to_dict()


def build_targets(px, sig, vol, dates, n, weight, gate, own_mom, mkt_ok):
    tg = {}
    for d in dates:
        if gate == "mkt" and not bool(mkt_ok.get(d, True)):
            tg[d] = {}
            continue
        s = sig.loc[d]
        if gate == "absmom":
            s = s.where(own_mom.loc[d] > 0)
        tg[d] = weights_for(s, vol.loc[d], weight, n)
    return tg


def run_grid(px, label, bench_close, done, writer, fh):
    idx = px.dropna(how="all").index
    ret = px.pct_change()
    vol = ret.rolling(63, min_periods=30).std() * np.sqrt(252)
    own_mom = px / px.shift(126) - 1
    b = bench_close.reindex(idx).ffill()
    sma200 = b.rolling(200, min_periods=100).mean()
    mkt_ok = (b > sma200).reindex(idx).fillna(True).to_dict()

    sigcache = {}
    specs = C.signal_specs()
    n_done = 0
    t0 = time.time()
    for sname in SIGS:
        kind, L, skip = specs[sname]
        sigcache[sname] = C.compute_signal(px, kind, L, skip)
    for sname, clock, off in itertools.product(SIGS, CLOCKS, OFFSETS):
        rd = C.rebal_dates(idx, clock, off)
        rd = rd[rd >= idx[260]]
        sig = sigcache[sname].reindex(rd)
        for n, weight, gate in itertools.product(NS, WEIGHTS, GATES):
            key = (label, sname, n, weight, clock, gate, off)
            if key in done:
                continue
            tg = build_targets(px, sig, vol.reindex(rd), rd, n, weight, gate,
                               own_mom.reindex(rd), mkt_ok)
            bk = C.Book(idx[idx >= rd[0]], px)
            r = bk.run(tg)
            m = C.metrics(r["nav"])
            writer.writerow(dict(assets=label, signal=sname, n=n, weight=weight, clock=clock,
                                 gate=gate, offset=off, turnover_yr=round(r["turnover_yr"], 2),
                                 exposure=round(r["exposure"], 3),
                                 **{k: round(v, 3) for k, v in m.items()}))
            n_done += 1
            if n_done % 200 == 0:
                fh.flush()
                log(f"  {label}: {n_done} cells ({time.time()-t0:.0f}s)")
    fh.flush()
    log(f"{label}: {n_done} new cells in {time.time()-t0:.0f}s")


def nulls(px, label, bench_close, benches):
    idx = px.dropna(how="all").index
    idx = idx[idx >= idx[260]]
    rows = []
    # equal weight all sectors
    for clock in CLOCKS:
        for off in OFFSETS:
            rd = C.rebal_dates(px.dropna(how="all").index, clock, off)
            rd = rd[rd >= idx[0]]
            tg = {d: {c: 1.0 / px.loc[d].notna().sum() for c in px.columns
                      if np.isfinite(px.loc[d, c])} for d in rd}
            r = C.Book(idx, px).run(tg)
            rows.append(dict(kind="EQUALWEIGHT_ALL", detail=f"{clock}off{off}", **C.metrics(r["nav"])))
    # buy and hold benchmarks (no tax until sale -> long-term, modelled untaxed B&H)
    for bname, s in benches.items():
        s = s.reindex(idx).ffill().dropna()
        rows.append(dict(kind="BUYHOLD", detail=bname, **C.metrics(s)))
    # cash null
    cash = pd.Series((1 + C.CASH_YIELD) ** (np.arange(len(idx)) / 252.0), index=idx)
    rows.append(dict(kind="CASH", detail="5pct", **C.metrics(cash)))
    df = pd.DataFrame(rows)
    df["assets"] = label
    return df


def random_null(px, label, n_draws=500):
    idx = px.dropna(how="all").index
    start = idx[260]
    rows = []
    rng = np.random.default_rng(1560907)
    cols = list(px.columns)
    for n in NS:
        rd = C.rebal_dates(idx, "M", 0)
        rd = rd[rd >= start]
        for k in range(n_draws):
            tg = {}
            for d in rd:
                avail = [c for c in cols if np.isfinite(px.loc[d, c])]
                pick = rng.choice(avail, size=min(n, len(avail)), replace=False)
                tg[d] = {c: 1.0 / len(pick) for c in pick}
            r = C.Book(idx[idx >= rd[0]], px).run(tg)
            m = C.metrics(r["nav"])
            rows.append(dict(assets=label, n=n, draw=k, **m))
        log(f"  random null N={n} done ({n_draws} draws)")
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    sect, _ = C.load_close(C.SECT9 + C.BENCHES, C.SECT_START, C.END)
    cal = C.trading_calendar(C.SECT_START, C.END)
    sect = sect.reindex(cal).ffill(limit=2)
    sect9 = sect[C.SECT9]
    benches = {b: sect[b] for b in C.BENCHES if b in sect.columns}

    nav = pd.read_csv(C.RES / "ind20_nav.csv", index_col=0, parse_dates=True)
    ind = nav.loc[nav.notna().sum(axis=1) >= 8]

    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for row in csv.DictReader(f):
                done.add((row["assets"], row["signal"], int(row["n"]), row["weight"],
                          row["clock"], row["gate"], int(row["offset"])))
        log(f"resuming: {len(done)} cells already present")
    new = not OUT.exists()
    fh = open(OUT, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        wr.writeheader()

    run_grid(sect9, "SECT9", sect["NIFTY500"], done, wr, fh)
    run_grid(ind, "IND20", sect["NIFTY500"].reindex(ind.index).ffill(), done, wr, fh)
    fh.close()

    n1 = nulls(sect9, "SECT9", sect["NIFTY500"], benches)
    n2 = nulls(ind, "IND20", sect["NIFTY500"],
               {b: s.reindex(ind.index).ffill() for b, s in benches.items()})
    pd.concat([n1, n2], ignore_index=True).to_csv(C.RES / "p2_nulls.csv", index=False)
    log("nulls written")

    rn = random_null(sect9, "SECT9")
    rn.to_csv(C.RES / "p2_random_null.csv", index=False)
    log(f"all done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
