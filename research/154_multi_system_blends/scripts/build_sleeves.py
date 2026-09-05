"""research/154 P0 - build every sleeve NAV on one host, once, and cache it.

Produces, in results/:
  oa_navs30.csv     Open Alpha, adopted spec, 30 random-selection seeds (daily, after-tax)
  oa_trades_s{1..5}.csv  OA trade lists for 5 seeds (position-level overlap, P6)
  tn_navs12.csv     True North, incumbent spec, 12 rebalance-day offsets (daily, after-tax)
  tn_holdings_off0.csv   TN month-end holdings (position-level overlap, P6)
  gold_nav.csv      GOLDBEES daily 2015+ ; monthly gold-INR reconstruction chained 2004-2014
  coverage.txt      min/max date + row counts for EVERY series used

Nothing here writes to market_data.db. The gold reconstruction is a LABELLED reference
series and lives only in results/.

Resume-safe: each artefact is skipped if it already exists.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path("/home/arun/quantifyd")
STUDY = ROOT / "research" / "154_multi_system_blends"
RES = STUDY / "results"
R142 = ROOT / "research" / "142_bananapatterns_replication" / "scripts"
R144 = ROOT / "research" / "144_truenorth_reassessment" / "scripts"
R147 = ROOT / "research" / "147_third_sleeve_archetypes" / "results"
DB = ROOT / "backtest_data" / "market_data.db"

N_SEEDS = 30
N_OFFSETS = 12
TRADE_SEEDS = [1, 2, 3, 4, 5]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ----------------------------------------------------------------------------- OA
def build_oa():
    navf, trf = RES / "oa_navs30.csv", RES / f"oa_trades_s{TRADE_SEEDS[-1]}.csv"
    if navf.exists() and trf.exists():
        log("OA cached, skip")
        return
    sys.path.insert(0, str(R142))
    import bluesky_replay as br

    log("OA: loading frames (trail_sma=15) ...")
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
    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
    dates = close.index
    C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma.values
    RSv, TVv = rs.values, tv_prev.values
    days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= "2006-01-01"])
    weak = np.zeros(len(dates), dtype=bool)
    cols = list(close.columns)

    navs = []
    for seed in range(1, N_SEEDS + 1):
        t0 = time.time()
        eq, trades, _ = br.simulate(seed, "random", days, dates, C, H, O, ATH, S, RSv, TVv,
                                    trig, weak, True, 0.0025, stop=0.08, slots=16,
                                    size_pct=0.0625, stcg=0.20, ltcg=0.125, cash_yield=0.05)
        navs.append(pd.Series(np.asarray(eq, float), index=dates[days], name=f"s{seed}"))
        if seed in TRADE_SEEDS:
            pd.DataFrame([dict(symbol=cols[c], entry=dates[ei].date(), exit=dates[xi].date(),
                               buy=b, sell=s, reason=r) for c, ei, xi, b, s, r in trades]) \
                .to_csv(RES / f"oa_trades_s{seed}.csv", index=False)
        log(f"OA seed {seed}/{N_SEEDS} ({time.time()-t0:.0f}s) final "
            f"{navs[-1].iloc[-1]/1e6:.2f}x")
    pd.concat(navs, axis=1).to_csv(navf)
    log("OA 30 seeds written")


# ----------------------------------------------------------------------------- TN
def build_tn():
    navf = RES / "tn_navs12.csv"
    if navf.exists():
        log("TN cached, skip")
        return
    spec = importlib.util.spec_from_file_location("tn_sweep", str(R144 / "tn_sweep.py"))
    tn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tn)
    ctx = tn.Ctx()
    cols = {}
    for off in range(N_OFFSETS):
        t0 = time.time()
        r = tn.run(ctx, offset=off, tax=True)
        nav = r["_nav"].dropna()
        cols[f"off{off}"] = nav
        log(f"TN offset {off}/{N_OFFSETS-1} ({time.time()-t0:.0f}s) "
            f"CAGR-ish {r.get('wa_cagr', r.get('cagr'))}")
        if off == 0 and "_holdings" in r:
            r["_holdings"].to_csv(RES / "tn_holdings_off0.csv")
    pd.DataFrame(cols).to_csv(navf)
    log("TN 12 offsets written")


# --------------------------------------------------------------------------- GOLD
def build_gold():
    f = RES / "gold_nav.csv"
    if f.exists():
        log("GOLD cached, skip")
        return
    con = sqlite3.connect(str(DB))
    g = pd.read_sql_query("select date, close from market_data_unified where symbol="
                          "'GOLDBEES' and timeframe='day' order by date", con)
    con.close()
    g["date"] = pd.to_datetime(g["date"].str[:10])
    gb = g.drop_duplicates("date").set_index("date")["close"].astype(float).sort_index()
    log(f"GOLDBEES real daily: {gb.index[0].date()} -> {gb.index[-1].date()} n={len(gb)}")

    ref = pd.read_csv(R147 / "gold_inr_ref.csv", index_col=0, parse_dates=True).iloc[:, 0]
    log(f"gold-INR reconstruction (monthly): {ref.index[0].date()} -> "
        f"{ref.index[-1].date()} n={len(ref)}")

    # validation on the overlap, reported not assumed
    mg = gb.resample("ME").last().pct_change().dropna()
    mr = ref.pct_change().dropna()
    common = mg.index.intersection(mr.index)
    corr = float(mg.loc[common].corr(mr.loc[common]))
    drift = float((mg.loc[common] - mr.loc[common]).mean() * 12 * 100)
    log(f"reconstruction validation on {len(common)} overlapping months: "
        f"monthly-return corr {corr:.3f}, annualised drift {drift:+.2f}pp")

    # chain: reconstruction level scaled to meet GOLDBEES at the first real month
    first_real = gb.index[0]
    anchor = ref.index[ref.index.searchsorted(first_real)]
    scale = float(gb.iloc[0]) / float(ref.loc[anchor])
    pre = (ref[ref.index < first_real] * scale)
    out = pd.concat([pre, gb])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    src = pd.Series(np.where(out.index < first_real, "reconstruction", "GOLDBEES"),
                    index=out.index)
    pd.DataFrame({"close": out, "source": src}).to_csv(f)
    log(f"GOLD chained series written: {out.index[0].date()} -> {out.index[-1].date()} "
        f"n={len(out)} ({int((src=='reconstruction').sum())} reconstructed monthly points)")
    with open(RES / "gold_validation.txt", "w") as fh:
        fh.write(f"overlap months {len(common)}\nmonthly corr {corr:.4f}\n"
                 f"annualised drift {drift:+.3f}pp\nreal from {first_real.date()}\n")


# ------------------------------------------------------------------------ COVERAGE
def coverage():
    lines = []
    def rep(name, s):
        lines.append(f"{name:<28} {str(s.index[0].date()):>12} -> {str(s.index[-1].date()):>12}"
                     f"   n={len(s):>6}")
    for tag, path, kw in [
            ("OA (30 seeds)", RES / "oa_navs30.csv", {}),
            ("TN (12 offsets)", RES / "tn_navs12.csv", {}),
            ("GOLD (chained)", RES / "gold_nav.csv", {}),
            ("VCP (30 seeds)", ROOT / "research/151_vcp_breakout/results/vcp_equity_seeds.csv", {}),
            ("MYB (30 seeds)", ROOT / "research/152_multiyear_breakout/results/myb_equity_seeds.csv", {}),
            ("IPO (30 seeds)", ROOT / "research/153_ipo_base/results/ipo_equity_seeds.csv", {}),
    ]:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        rep(tag, df.iloc[:, 0])
        lines.append(f"{'':28} columns: {len(df.columns)}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    (RES / "coverage.txt").write_text(txt)


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    build_gold()
    build_tn()
    build_oa()
    coverage()
    log("P0 DONE")
