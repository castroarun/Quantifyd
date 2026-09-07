"""r/156 P4 (G3) — branch B: sector as a UNIVERSE FILTER for a curated stock portfolio.

The book: rank industries by sector momentum, keep the top K, then hold the best stocks WITHIN
those industries, equal-weighted across `slots`, rebalanced monthly or quarterly.

The only question that matters is the decomposition, so every book is run against three PAIRED
controls on the identical offsets, dates and slot count:

  CTRL_NOSECT  same stock rule over the FULL universe, no sector filter
               -> if the book does not beat this, the sector layer adds nothing
  CTRL_RNDSTK  random stocks drawn from the SAME top-K industries (100 draws)
               -> if the book does not beat this, the stock layer is decoration
  CTRL_RNDSEC  same stock rule inside K RANDOM industries (100 draws)
               -> isolates whether picking the RIGHT sectors matters at all

Sector ranking source is an axis: the survivorship-inflated synthetic baskets (what a
practitioner would eyeball) and the real NSE sector indices (tradeable, only 8 industries map).

Outputs: p4_books.csv, p4_controls.csv
"""
import sys, time, itertools, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

import common as C
from p1_ic import IND2SECT

log = lambda *a: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

KS = [2, 3, 4, 5]
SLOTS = [10, 15, 20]
STOCKSIG = {"mom126": ("mom", 126, 0), "momskip126": ("mom", 126, 21),
            "riskadj126": ("riskadj", 126, 0), "disthigh252": ("disthigh", 252, 0)}
CLOCKS = ["M", "Q"]
OFFSETS = [0, 1, 2, 3]
SECTSIG = ("mom", 126, 0)            # ABSMOM126 on the sector series
START = "2015-01-01"

BOOKS = C.RES / "p4_books.csv"
CTRL = C.RES / "p4_controls.csv"
BF = ["src", "K", "slots", "stocksig", "clock", "offset", "cagr", "maxdd", "calmar",
      "sharpe", "vol", "turnover_yr", "exposure"]
CF = ["control", "src", "K", "slots", "stocksig", "clock", "offset", "draw", "cagr",
      "maxdd", "calmar", "sharpe"]


def load():
    close, tv, imap = C.load_stock_panel(START, C.END)
    elig = (tv.shift(1) >= C.TV_FLOOR) & close.notna() & close.shift(1).notna()
    # drop split-scale artefacts from the eligibility mask
    r = close.pct_change()
    elig &= r.abs() < 0.40
    nav = pd.read_csv(C.RES / "ind20_nav.csv", index_col=0, parse_dates=True)
    nav = nav.reindex(close.index).ffill(limit=2)
    sect, _ = C.load_close(C.SECT9, C.SECT_START, C.END)
    sect = sect.reindex(close.index).ffill(limit=2)
    return close, tv, imap, elig, nav, sect


def stock_scores(close, name):
    kind, L, skip = STOCKSIG[name]
    return C.compute_signal(close, kind, L, skip)


def pick(scores_d, elig_d, members, slots):
    s = scores_d[[c for c in members if c in scores_d.index]]
    s = s[elig_d.reindex(s.index).fillna(False)]
    top = s.dropna().sort_values(ascending=False).head(slots)
    return list(top.index)


def build(dates, sect_rank, K, imap, close, elig, scores, slots, ind2col=None,
          rng=None, rnd_stocks=False, rnd_sectors=False, no_sector=False):
    tg = {}
    all_inds = sorted(set(imap.values()))
    for d in dates:
        if no_sector:
            members = [c for c in close.columns]
        else:
            sr = sect_rank.loc[d].dropna()
            if len(sr) < K:
                tg[d] = {}
                continue
            if rnd_sectors:
                chosen = list(rng.choice(sr.index.values, size=K, replace=False))
            else:
                chosen = list(sr.sort_values(ascending=False).head(K).index)
            inds = set(ind2col[c] for c in chosen) if ind2col else set(chosen)
            members = [c for c in close.columns if imap.get(c) in inds]
        if rnd_stocks:
            e = elig.loc[d]
            cand = [c for c in members if e.get(c, False) and np.isfinite(close.loc[d, c])]
            if not cand:
                tg[d] = {}
                continue
            names = list(rng.choice(cand, size=min(slots, len(cand)), replace=False))
        else:
            names = pick(scores.loc[d], elig.loc[d], members, slots)
        tg[d] = {c: 1.0 / len(names) for c in names} if names else {}
    return tg


def main():
    t0 = time.time()
    close, tv, imap, elig, nav, sect = load()
    idx = close.index
    start_i = idx[idx >= pd.Timestamp(START)][260]
    log(f"panel {close.shape}; book start {start_i.date()}")

    # sector ranking series
    sect_sources = {}
    kind, L, skip = SECTSIG
    sect_sources["synth20"] = (C.compute_signal(nav, kind, L, skip), None)
    real = sect[[IND2SECT[i] for i in IND2SECT if IND2SECT[i] in sect.columns]]
    ind2col = {IND2SECT[i]: i for i in IND2SECT}
    sect_sources["real8"] = (C.compute_signal(real, kind, L, skip), ind2col)

    sc = {k: stock_scores(close, k) for k in STOCKSIG}

    done = set()
    if BOOKS.exists():
        with open(BOOKS) as f:
            for row in csv.DictReader(f):
                done.add((row["src"], int(row["K"]), int(row["slots"]), row["stocksig"],
                          row["clock"], int(row["offset"])))
    newf = not BOOKS.exists()
    fh = open(BOOKS, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=BF)
    if newf:
        wr.writeheader()

    n = 0
    for src, (srank, i2c) in sect_sources.items():
        for K, slots, sig, clock, off in itertools.product(KS, SLOTS, STOCKSIG, CLOCKS, OFFSETS):
            key = (src, K, slots, sig, clock, off)
            if key in done:
                continue
            rd = C.rebal_dates(idx, clock, off)
            rd = rd[rd >= start_i]
            tg = build(rd, srank.reindex(rd), K, imap, close, elig.reindex(rd),
                       sc[sig].reindex(rd), slots, ind2col=i2c)
            r = C.Book(idx[idx >= rd[0]], close).run(tg)
            m = C.metrics(r["nav"])
            wr.writerow(dict(src=src, K=K, slots=slots, stocksig=sig, clock=clock, offset=off,
                             turnover_yr=round(r["turnover_yr"], 2),
                             exposure=round(r["exposure"], 3),
                             **{k: round(v, 3) for k, v in m.items()}))
            n += 1
            if n % 40 == 0:
                fh.flush()
                log(f"  books {n} ({time.time()-t0:.0f}s)")
    fh.close()
    log(f"branch-B books done: {n} new cells ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- paired controls
    b = pd.read_csv(BOOKS)
    med = (b.groupby(["src", "K", "slots", "stocksig", "clock"])
             .calmar.median().sort_values(ascending=False))
    finalists = list(med.head(3).index) + [("synth20", 3, 15, "mom126", "M")]
    finalists = list(dict.fromkeys(finalists))
    log(f"control finalists: {finalists}")

    rows = []
    rng = np.random.default_rng(1560907)
    for src, K, slots, sig, clock in finalists:
        srank, i2c = sect_sources[src]
        for off in OFFSETS:
            rd = C.rebal_dates(idx, clock, off)
            rd = rd[rd >= start_i]
            base = dict(src=src, K=K, slots=slots, stocksig=sig, clock=clock, offset=off)
            # CTRL_NOSECT
            tg = build(rd, None, K, imap, close, elig.reindex(rd), sc[sig].reindex(rd),
                       slots, no_sector=True)
            m = C.metrics(C.Book(idx[idx >= rd[0]], close).run(tg)["nav"])
            rows.append(dict(control="CTRL_NOSECT", draw=-1, **base,
                             **{k: round(m[k], 3) for k in ("cagr", "maxdd", "calmar", "sharpe")}))
            for cname, kw in (("CTRL_RNDSTK", dict(rnd_stocks=True)),
                              ("CTRL_RNDSEC", dict(rnd_sectors=True))):
                for k in range(100):
                    tg = build(rd, srank.reindex(rd), K, imap, close, elig.reindex(rd),
                               sc[sig].reindex(rd), slots, ind2col=i2c, rng=rng, **kw)
                    m = C.metrics(C.Book(idx[idx >= rd[0]], close).run(tg)["nav"])
                    rows.append(dict(control=cname, draw=k, **base,
                                     **{q: round(m[q], 3) for q in
                                        ("cagr", "maxdd", "calmar", "sharpe")}))
                log(f"  {cname} {src} K{K} s{slots} {sig} {clock} off{off} done "
                    f"({time.time()-t0:.0f}s)")
    pd.DataFrame(rows).to_csv(CTRL, index=False)
    log(f"controls written ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
