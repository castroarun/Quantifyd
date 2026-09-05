"""research/154 P6 - POSITION-LEVEL overlap, not just correlation.

Two sleeves can show a modest return correlation and still be the same trades. This
measures, for every pair whose trade list can be reconstructed:

  * SIGNAL overlap   - % of sleeve A's (symbol, trigger-date) pairs that are also sleeve
                       B's signals, and the reverse. Computed on the raw screens, before
                       slot competition, so it is seed-free.
  * HOLDING-DAY overlap - % of A's (symbol, day-held) pairs that B also held that day.
                       Seed-dependent; reported across 5 seeds.

OA, VCP and MYB are all decodable from one frame load (they differ only in the pivot
level). IPO needs a vetted listing-date table, so its holding days are read from
research/153's trade list and its trading calendar is reconstructed from the DB and
VALIDATED against the recorded holding period of every trade.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path("/home/arun/quantifyd")
RES = ROOT / "research/154_multi_system_blends/results"
R142 = ROOT / "research/142_bananapatterns_replication/scripts"
sys.path.insert(0, str(R142))
import bluesky_replay as br  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
START = "2010-01-01"          # panel A: the window where all four sleeves coexist


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    w = br.load_frames("2004-06-01", trail_sma=15)
    close, high, open_, athcp, sma, tv20 = (w[k] for k in
                                            ("close", "high", "open", "athcp", "sma50", "tv20"))
    etf = [c for c in close.columns if br.ETF_RE.search(c)]
    tv_prev, prev_close = tv20.shift(1), close.shift(1)
    elig = tv_prev >= br.TV_FLOOR
    elig[etf] = False
    score = (2 * (close / close.shift(63) - 1) + (close / close.shift(126) - 1)
             + (close / close.shift(189) - 1) + (close / close.shift(252) - 1))
    rs = (score.where(elig).rank(axis=1, pct=True) * 100).shift(1)
    dates = close.index
    cols = np.array(close.columns)

    # ---- pivots
    ath = athcp                                     # OA  : prior all-time-high close
    piv30 = close.shift(1).rolling(30).max()        # VCP : highest close of prior 30 days
    piv3y = close.shift(1).rolling(756).max()       # MYB : highest close of prior 3 years

    def trig_of(piv, near, extra=None):
        setup = (prev_close < piv) & (prev_close >= (1 - near) * piv) & elig & (rs >= 70.0)
        if extra is not None:
            setup = setup & extra
        return (setup & (close > piv) & piv.notna()).fillna(False)

    T = {
        "OA": trig_of(ath, 0.20),
        "VCP": trig_of(piv30, 0.20),
        # MYB's distinctive residual: a 3-year high that is NOT an all-time high
        "MYB": trig_of(piv3y, 0.20, extra=(piv3y < ath) & (close < ath)),
    }
    P = {"OA": ath, "VCP": piv30, "MYB": piv3y}

    m = dates >= pd.Timestamp(START)

    def sigset(k):
        v = T[k].loc[m].values
        r, c = np.nonzero(v)
        dd = dates[m]
        ds = dd.strftime("%Y-%m-%d").to_numpy()
        return set(zip(ds[r], cols[c]))

    S = {k: sigset(k) for k in T}
    for k, v in S.items():
        log(f"{k}: {len(v)} raw signals from {START}")

    # IPO signals from its trade list
    con = sqlite3.connect(str(ROOT / "backtest_data/market_data.db"))
    cal = pd.read_sql_query("select distinct date from market_data_unified where "
                            "timeframe='day' and date >= '2005-06-01' order by date", con)
    con.close()
    ipo_dates = pd.DatetimeIndex(pd.to_datetime(cal["date"].str[:10]).drop_duplicates())
    tr = pd.read_csv(ROOT / "research/153_ipo_base/results/g3_trades_adopted.csv")
    ok = (ipo_dates[tr.xi.values] - ipo_dates[tr.ei.values]).days
    match = float((ok == tr.held.values).mean())
    log(f"IPO calendar reconstruction validated on {len(tr)} trades: "
        f"{match*100:.2f}% of recorded holding periods reproduced exactly")
    tr["edate"] = ipo_dates[tr.ei.values]
    tr["xdate"] = ipo_dates[tr.xi.values]
    tr["sym"] = tr.symbol.str.replace("-BE$", "", regex=True)
    S["IPO"] = set(zip(tr[tr.edate >= START].edate.dt.strftime("%Y-%m-%d"),
                       tr[tr.edate >= START].sym))
    log(f"IPO: {len(S['IPO'])} distinct (date, symbol) entries from {START}")

    rows = []
    names = ["OA", "VCP", "MYB", "IPO"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = len(S[a] & S[b])
            rows.append(dict(pair=f"{a}~{b}", kind="signal",
                             a_signals=len(S[a]), b_signals=len(S[b]), both=inter,
                             pct_of_a=round(100 * inter / max(1, len(S[a])), 1),
                             pct_of_b=round(100 * inter / max(1, len(S[b])), 1)))
            log(f"SIGNAL {a}~{b}: {inter} shared -> {rows[-1]['pct_of_a']}% of {a}, "
                f"{rows[-1]['pct_of_b']}% of {b}")

    # ---- holding-day overlap (needs the book simulation: slots compete)
    C, H, O, S50 = close.values, high.values, open_.values, sma.values
    RSv, TVv = rs.values, tv_prev.values
    days = np.array([i for i, dte in enumerate(dates) if str(dte.date()) >= START])
    weak = np.zeros(len(dates), dtype=bool)
    hold = {k: {} for k in ("OA", "VCP", "MYB")}
    for k in ("OA", "VCP", "MYB"):
        trig = T[k].values
        PIV = P[k].values
        for sd in SEEDS:
            _, trades, _ = br.simulate(sd, "random", days, dates, C, H, O, PIV, S50, RSv,
                                       TVv, trig, weak, True, 0.0025, stop=0.08, slots=16,
                                       size_pct=0.0625, stcg=0.20, ltcg=0.125,
                                       cash_yield=0.05)
            hs = set()
            for c, ei, xi, *_ in trades:
                for j in range(ei, xi + 1):
                    hs.add((j, c))
            hold[k][sd] = {(str(dates[j].date()), cols[c]) for j, c in hs}
        log(f"{k}: holding-day sets built for {len(SEEDS)} seeds "
            f"(seed1 = {len(hold[k][1])} symbol-days)")

    hold["IPO"] = {}
    for sd in SEEDS:
        t = tr[(tr.seed == sd) & (tr.edate >= START)]
        hs = set()
        for ei, xi, sym in zip(t.ei.values, t.xi.values, t.sym.values):
            for j in range(ei, xi + 1):
                hs.add((str(ipo_dates[j].date()), sym))
        hold["IPO"][sd] = hs
    log(f"IPO: holding-day sets built (seed1 = {len(hold['IPO'][1])} symbol-days)")

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb, both = [], [], []
            for sd in SEEDS:
                A, B = hold[a][sd], hold[b][sd]
                n = len(A & B)
                both.append(n)
                pa.append(100 * n / max(1, len(A)))
                pb.append(100 * n / max(1, len(B)))
            rows.append(dict(pair=f"{a}~{b}", kind="holding-day",
                             a_signals=int(np.median([len(hold[a][s]) for s in SEEDS])),
                             b_signals=int(np.median([len(hold[b][s]) for s in SEEDS])),
                             both=int(np.median(both)),
                             pct_of_a=round(float(np.median(pa)), 1),
                             pct_of_b=round(float(np.median(pb)), 1)))
            log(f"HOLDING {a}~{b}: {rows[-1]['pct_of_a']}% of {a}'s symbol-days, "
                f"{rows[-1]['pct_of_b']}% of {b}'s")

    pd.DataFrame(rows).to_csv(RES / "p6_overlap.csv", index=False)
    log("P6 written")


if __name__ == "__main__":
    main()
