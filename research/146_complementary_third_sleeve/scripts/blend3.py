"""research/146 — 3-sleeve blend stage: TN (deployed, offsets 0/4/8) + OA (adopted spec,
10 seeds) + candidate sleeve(s), all after-tax, monthly rebalanced. Pre-registered rule:
best w3 in {10,15,20,25,33}% (TN=OA split the rest) must beat the TN+OA 50-50 baseline by
+0.10 Calmar (CAGR >= 25.2) OR -2pp DD at CAGR >= 27.2, on the 10-OA-seed median, with
candidate corr < 0.4 vs BOTH legs, surviving TN offsets and the crash-window check.

Usage: blend3.py <candidate_name> [more...]   (reads results/nav_<name>_tax1.csv)
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
R144 = Path("/home/arun/quantifyd/research/144_truenorth_reassessment/scripts")
OA = Path("/home/arun/quantifyd/research/142_bananapatterns_replication/scripts")
sys.path.insert(0, str(OA))
import bluesky_replay as br

W3 = [0.0, 0.10, 0.15, 0.20, 0.25, 0.33]
OFFSETS = [0, 4, 8]
CRASH = {"2008": ("2008-01-01", "2009-03-31"), "2015-16": ("2015-08-01", "2016-02-29"),
         "2018": ("2018-01-01", "2018-10-31"), "2020crash": ("2020-02-01", "2020-04-30"),
         "2022H1": ("2022-01-01", "2022-06-30")}


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = float((nav / nav.cummax() - 1).min())
    return round(cagr * 100, 2), round(dd * 100, 2), \
        (round(cagr / abs(dd), 2) if dd < 0 else np.nan)


def oa_navs_10():
    f = RES / "oa_navs.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        return [df[c] for c in df.columns]
    print("building OA frames (trail_sma=15) ...", flush=True)
    w = br.load_frames("2004-06-01", trail_sma=15)
    close, high, open_, athcp, sma, tv20 = (w[k] for k in
        ("close", "high", "open", "athcp", "sma50", "tv20"))
    etf = [c for c in close.columns if br.ETF_RE.search(c)]
    tv_prev = tv20.shift(1)
    prev_close = close.shift(1)
    elig = tv_prev >= br.TV_FLOOR
    elig[etf] = False
    score = 2 * (close / close.shift(63) - 1) + (close / close.shift(126) - 1) \
        + (close / close.shift(189) - 1) + (close / close.shift(252) - 1)
    rs = (score.where(elig).rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
    dates = close.index
    C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma.values
    RSv, TVv = rs.values, tv_prev.values
    days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= "2006-01-01"])
    weak = np.zeros(len(dates), dtype=bool)
    navs = []
    for seed in range(1, 11):
        eq, _, _ = br.simulate(seed, "random", days, dates, C, H, O, ATH, S, RSv, TVv,
                               trig, weak, True, 0.0025, stop=0.08, slots=16,
                               size_pct=0.0625, stcg=0.20, ltcg=0.125, cash_yield=0.05)
        navs.append(pd.Series(np.asarray(eq, float), index=dates[days], name=f"s{seed}"))
    pd.concat(navs, axis=1).to_csv(f)
    print("OA 10 seeds built + cached", flush=True)
    return navs


def tn_navs():
    out = {}
    missing = [o for o in OFFSETS if not (RES / f"tn_nav_off{o}.csv").exists()]
    if missing:
        _s = importlib.util.spec_from_file_location("tn", str(R144 / "tn_sweep.py"))
        tn = importlib.util.module_from_spec(_s); _s.loader.exec_module(tn)
        ctx = tn.Ctx()
        for o in missing:
            r = tn.run(ctx, offset=o, tax=True)
            r["_nav"].to_csv(RES / f"tn_nav_off{o}.csv")
            print(f"TN offset {o}: waCAGR {r['wa_cagr']}", flush=True)
    for o in OFFSETS:
        out[o] = pd.read_csv(RES / f"tn_nav_off{o}.csv", index_col=0,
                             parse_dates=True).iloc[:, 0]
    return out


def blend(o_nav, t_nav, c_nav, w3):
    idx = o_nav.index.intersection(t_nav.index)
    if c_nav is not None:
        idx = idx.intersection(c_nav.index)
    legs = [o_nav.loc[idx], t_nav.loc[idx]] + ([c_nav.loc[idx]] if c_nav is not None else [])
    m = [x.resample("ME").last().pct_change().fillna(0) for x in legs]
    wl = (1 - w3) / 2
    r = wl * m[0] + wl * m[1] + (w3 * m[2] if c_nav is not None else 0)
    return (1 + r).cumprod()


def main():
    cands = sys.argv[1:]
    assert cands, "pass candidate names (nav_<name>_tax1.csv must exist)"
    oa = oa_navs_10()
    tn = tn_navs()
    rows, crash_rows = [], []
    for name in cands:
        cnav = pd.read_csv(RES / f"nav_{name}_tax1.csv", index_col=0,
                           parse_dates=True).iloc[:, 0]
        # correlations (vs TN offset0, vs each OA seed -> median)
        idx = cnav.index.intersection(tn[0].index)
        corr_tn = float(cnav.loc[idx].pct_change().corr(tn[0].loc[idx].pct_change()))
        corr_oa = float(np.median([
            cnav.loc[cnav.index.intersection(o.index)].pct_change().corr(
                o.loc[cnav.index.intersection(o.index)].pct_change()) for o in oa]))
        for off in OFFSETS:
            for w3 in W3:
                cs, ds, ks = [], [], []
                for o_nav in oa:
                    b = blend(o_nav, tn[off], cnav if w3 > 0 else None, w3)
                    c, d, k = stats(b)
                    cs.append(c); ds.append(d); ks.append(k)
                rows.append(dict(cand=name, offset=off, w3=w3,
                                 cagr_med=round(float(np.median(cs)), 2),
                                 cagr_min=round(min(cs), 2),
                                 dd_med=round(float(np.median(ds)), 2),
                                 dd_worst=round(min(ds), 2),
                                 calmar_med=round(float(np.median(ks)), 2),
                                 calmar_min=round(min(ks), 2),
                                 corr_tn=round(corr_tn, 3), corr_oa=round(corr_oa, 3)))
                if off == 0:
                    print(rows[-1], flush=True)
        # crash windows (offset 0): per-seed blends at w3=0 vs w3=0.25, plus sleeve return
        for wname, (a, b_) in CRASH.items():
            def win_dd(w3):
                vals = []
                for o_nav in oa:
                    bl = blend(o_nav, tn[0], cnav if w3 > 0 else None, w3)
                    s = bl[(bl.index >= a) & (bl.index <= b_)]
                    if len(s) > 2:
                        vals.append(float((s / s.cummax() - 1).min() * 100))
                return round(float(np.median(vals)), 2) if vals else ""
            s = cnav[(cnav.index >= a) & (cnav.index <= b_)]
            sleeve_ret = round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 1) if len(s) > 2 else ""
            crash_rows.append(dict(cand=name, window=wname, sleeve_ret_pct=sleeve_ret,
                                   base_dd=win_dd(0.0), w25_dd=win_dd(0.25)))
            print(crash_rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "blend3.csv", index=False)
    pd.DataFrame(crash_rows).to_csv(RES / "crash_windows.csv", index=False)
    print("BLEND DONE", flush=True)


if __name__ == "__main__":
    main()
