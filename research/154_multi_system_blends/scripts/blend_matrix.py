"""research/154 P1-P5 - correlations, all 57 subsets, weight sweep, cash-null, paired
deltas vs the deployed TN+OA 50-50 pair, and per-window rows.

Path convention: a PATH = (OA seed s in 1..30, TN offset o in 0..11) -> 360 paths.
Within a path the stochastic research sleeves (VCP, MYB, IPO) use the SAME seed index s.
Every A-vs-B number is PAIRED on the path.

Panels (never mixed):
  A  2010-01 -> 2026-08   all 6 sleeves (gold = reconstruction 2010-14 + real 2015+)
  B  2006-04 -> 2026-08   5 sleeves, MYB structurally absent; contains 2008
  C  2015-01 -> 2026-08   all 6, no reconstructed data at all

Outputs (all incremental / resume-safe):
  results/p1_correlations.csv
  results/p2_subsets.csv        equal-weight, every subset, every panel
  results/p3_weights.csv        weight sweep + cash-null + paired deltas
  results/p5_windows.csv        per-window rows for the finalists
"""
from __future__ import annotations

import itertools
import os
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

SLEEVES = ["OA", "TN", "VCP", "MYB", "IPO", "GOLD"]
NSEED, NOFF = 30, 12
NPATH = NSEED * NOFF

PANELS = {
    "A": dict(members=SLEEVES, start="2010-01", label="2010-01 -> 2026-08 (all 6; gold pre-2015 = reconstruction)"),
    "B": dict(members=["OA", "TN", "VCP", "IPO", "GOLD"], start="2006-04", label="2006-04 -> 2026-08 (5 sleeves, contains 2008)"),
    "C": dict(members=SLEEVES, start="2015-01", label="2015-01 -> 2026-08 (all 6, real data only)"),
}
END_MONTH = "2026-08"

WINDOWS = {
    "2008 crash": ("2008-01", "2009-03"),
    "2020 crash": ("2020-02", "2020-04"),
    "2018 grind": ("2018-01", "2018-10"),
    "2022H1 grind": ("2022-01", "2022-06"),
}

PAIR_W = [0.10, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.80, 0.90]
SAT_W = [0.10, 0.20, 0.25, 0.33]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------ load sleeves
def load_daily():
    d = {}
    d["OA"] = pd.read_csv(RES / "oa_navs30.csv", index_col=0, parse_dates=True)
    d["TN"] = pd.read_csv(RES / "tn_navs12.csv", index_col=0, parse_dates=True)
    d["VCP"] = pd.read_csv(ROOT / "research/151_vcp_breakout/results/vcp_equity_seeds.csv",
                           index_col=0, parse_dates=True)
    d["MYB"] = pd.read_csv(ROOT / "research/152_multiyear_breakout/results/myb_equity_seeds.csv",
                           index_col=0, parse_dates=True)
    d["IPO"] = pd.read_csv(ROOT / "research/153_ipo_base/results/ipo_equity_seeds.csv",
                           index_col=0, parse_dates=True)
    g = pd.read_csv(RES / "gold_nav.csv", index_col=0, parse_dates=True)
    d["GOLD_full"] = g[["close"]].rename(columns={"close": "gold"})
    d["GOLD"] = g[g["source"] == "GOLDBEES"][["close"]].rename(columns={"close": "gold"})
    for k in ("OA", "TN", "VCP", "MYB", "IPO"):
        d[k] = d[k].astype(float)
    return d


def monthly_nav(daily: pd.DataFrame) -> pd.DataFrame:
    """Month-end NAV indexed by Period('M'). Only complete months of the source."""
    m = daily.resample("ME").last()
    m.index = m.index.to_period("M")
    return m


def build_monthly(d):
    """Returns dict sleeve -> DataFrame(PeriodIndex 'M', columns)."""
    mm = {}
    for k in ("OA", "TN", "VCP", "MYB", "IPO"):
        mm[k] = monthly_nav(d[k])
    # gold: reconstruction rows are already month-end stamps, GOLDBEES is daily
    mm["GOLD"] = monthly_nav(d["GOLD_full"])
    return mm


# --------------------------------------------------------------------- metrics
def path_stats(nav: np.ndarray, years: float):
    """nav (n_months, n_paths) starting at 1.0. Returns cagr%, dd%, calmar arrays."""
    cagr = (nav[-1] ** (1.0 / years) - 1.0) * 100.0
    run = np.maximum.accumulate(nav, axis=0)
    dd = ((nav / run - 1.0).min(axis=0)) * 100.0
    calmar = np.where(dd < 0, cagr / np.abs(dd), np.nan)
    return cagr, dd, calmar


def cumnav(r: np.ndarray):
    return np.cumprod(1.0 + r, axis=0)


class Panel:
    def __init__(self, key, mm):
        cfg = PANELS[key]
        self.key, self.cfg = key, cfg
        members = cfg["members"]
        idx = None
        for s in members:
            i = mm[s].index
            idx = i if idx is None else idx.intersection(i)
        idx = idx[(idx >= pd.Period(cfg["start"], "M")) & (idx <= pd.Period(END_MONTH, "M"))]
        self.months = idx
        self.years = (len(idx) - 1) / 12.0
        # monthly return matrices, expanded to 360 paths
        self.R = {}
        for s in members:
            nav = mm[s].loc[idx]
            r = nav.pct_change().fillna(0.0).values.astype(np.float64)
            if s == "OA":                       # 30 seeds -> path (s,o) = seed-major
                self.R[s] = np.repeat(r, NOFF, axis=1)      # s1o0..s1o11, s2o0...
            elif s == "TN":                     # 12 offsets
                self.R[s] = np.tile(r, (1, NSEED))
            elif s == "GOLD":
                self.R[s] = np.repeat(r, NPATH, axis=1)
            else:                               # 30 seeds, same pairing as OA
                self.R[s] = np.repeat(r, NOFF, axis=1)
        # cash sleeve at 5% p.a.
        self.R["CASH"] = np.full((len(idx), NPATH), (1.05) ** (1 / 12) - 1.0)
        self.members = members
        # the deployed baseline, per path
        self.base_nav = cumnav(0.5 * self.R["OA"] + 0.5 * self.R["TN"])
        self.base = path_stats(self.base_nav, self.years)

    def blend(self, wmap):
        r = np.zeros_like(self.R["OA"])
        for s, w in wmap.items():
            r += w * self.R[s]
        return cumnav(r)

    def window(self, nav, a, b):
        """Return (window return %, window drawdown % measured from the running peak of the
        FULL curve, not from the window's own first bar).

        Measuring from the window's first bar hides a peak that sits just before it - which
        is exactly what happens in 2008, whose peak is Dec-2007. Prior studies (r/146, r/151)
        started the 2008 window on 2008-01-01 and therefore reported a much shallower 2008
        drawdown than the book actually took."""
        m = self.months
        sel = np.where((m >= pd.Period(a, "M")) & (m <= pd.Period(b, "M")))[0]
        if len(sel) < 2:
            return np.nan, np.nan
        run = np.maximum.accumulate(nav, axis=0)           # running peak of the whole curve
        dd = (nav[sel] / run[sel] - 1).min(axis=0)
        seg = nav[sel] / nav[sel[0]]
        return float(np.median(seg[-1] - 1) * 100), float(np.median(dd) * 100)


def band(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)), float(np.min(x)), float(np.max(x))


def xcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Full cross-correlation matrix between the columns of A and of B."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    A = A / np.where(A.std(0) == 0, np.nan, A.std(0))
    B = B / np.where(B.std(0) == 0, np.nan, B.std(0))
    return (A.T @ B) / A.shape[0]


# ------------------------------------------------------------------ P1 correlations
def p1(mm, d):
    out = []
    for key in PANELS:
        cfg = PANELS[key]
        pn = Panel(key, mm)
        for a, b in itertools.combinations(cfg["members"], 2):
            # monthly, on the panel months
            ra = mm[a].loc[pn.months].pct_change().iloc[1:]
            rb = mm[b].loc[pn.months].pct_change().iloc[1:]
            vals = xcorr(ra.values, rb.values).ravel()
            mmed, mmin, mmax = band(vals)
            # daily, on the panel's daily overlap (gold: real data only, 2015+)
            da = d[a] if a != "GOLD" else d["GOLD"]
            db = d[b] if b != "GOLD" else d["GOLD"]
            di = da.index.intersection(db.index)
            lo = pn.months[0].to_timestamp()
            di = di[di >= lo]
            note = ""
            if "GOLD" in (a, b) and key != "C":
                note = "daily gold = GOLDBEES real 2015+ only"
                di = di[di >= pd.Timestamp("2015-01-01")]
            xa = da.loc[di].pct_change().iloc[1:]
            xb = db.loc[di].pct_change().iloc[1:]
            dv = xcorr(xa.values, xb.values).ravel()
            dmed, dmin, dmax = band(dv)
            out.append(dict(panel=key, window=cfg["label"], pair=f"{a}~{b}",
                            daily_med=round(dmed, 3), daily_min=round(dmin, 3),
                            daily_max=round(dmax, 3), monthly_med=round(mmed, 3),
                            monthly_min=round(mmin, 3), monthly_max=round(mmax, 3),
                            n_combos=len(vals), note=note))
            log(f"P1 {key} {a}~{b}: daily {dmed:.3f} [{dmin:.3f}..{dmax:.3f}]  "
                f"monthly {mmed:.3f} [{mmin:.3f}..{mmax:.3f}] {note}")
    pd.DataFrame(out).to_csv(RES / "p1_correlations.csv", index=False)
    log("P1 written")


# ------------------------------------------------------- weight schemes per subset
def schemes(sub):
    """Yield (label, {sleeve: weight})."""
    n = len(sub)
    yield "EW", {s: 1.0 / n for s in sub}
    core = {"OA", "TN"}
    if n == 2:
        a, b = sub
        for w in PAIR_W:
            if abs(w - 0.5) < 1e-9:
                continue
            yield f"{b}@{int(w*100)}", {a: 1 - w, b: w}
    if core.issubset(set(sub)) and n > 2:
        sats = [s for s in sub if s not in core]
        k = len(sats)
        for w in SAT_W:
            if k * w > 0.60 + 1e-9:
                continue
            wm = {s: w for s in sats}
            wm["OA"] = wm["TN"] = (1 - k * w) / 2
            yield f"sat{int(w*100)}", wm
    if n >= 3 and not core.issubset(set(sub)):
        for tilt in sub:
            wm = {s: 0.5 / (n - 1) for s in sub if s != tilt}
            wm[tilt] = 0.5
            yield f"tilt_{tilt}50", wm


def cash_null(wmap):
    """Same weights, every non-deployed sleeve replaced by cash."""
    out = {}
    for s, w in wmap.items():
        k = s if s in ("OA", "TN") else "CASH"
        out[k] = out.get(k, 0.0) + w
    return out


def run_cells(mm):
    rows, wrows = [], []
    for key in PANELS:
        pn = Panel(key, mm)
        bc, bd, bk = pn.base
        log(f"--- panel {key}: {pn.months[0]} -> {pn.months[-1]} ({pn.years:.1f}y, "
            f"{len(pn.months)} months). baseline TN+OA 50-50: CAGR {np.median(bc):.2f} "
            f"[{bc.min():.2f}..{bc.max():.2f}] DD {np.median(bd):.2f} "
            f"[worst {bd.min():.2f}] Calmar {np.median(bk):.3f}")
        rows.append(dict(panel=key, subset="TN+OA", n=2, scheme="BASELINE 50-50",
                         weights="OA:0.50|TN:0.50",
                         cagr=round(float(np.median(bc)), 2), cagr_lo=round(float(bc.min()), 2),
                         cagr_hi=round(float(bc.max()), 2), dd=round(float(np.median(bd)), 2),
                         dd_worst=round(float(bd.min()), 2),
                         calmar=round(float(np.median(bk)), 3),
                         calmar_worst=round(float(np.nanmin(bk)), 3),
                         d_cagr=0.0, d_dd=0.0, d_calmar=0.0, calmar_wins="-",
                         cn_calmar=np.nan, d_calmar_vs_cash=np.nan, cash_wins="-"))
        mem = pn.cfg["members"]
        subs = [c for r in range(2, len(mem) + 1) for c in itertools.combinations(mem, r)]
        for sub in subs:
            for lab, wm in schemes(list(sub)):
                nav = pn.blend(wm)
                c, dd, k = path_stats(nav, pn.years)
                cn = cash_null(wm)
                has_core = len(set(sub) & {"OA", "TN"}) > 0
                if has_core:
                    cnav = pn.blend(cn)
                    cc, cdd, ck = path_stats(cnav, pn.years)
                    dk_cash = float(np.nanmedian(k - ck))
                    cwins = f"{int(np.nansum(k > ck))}/{NPATH}"
                    cn_calmar = round(float(np.nanmedian(ck)), 3)
                else:
                    dk_cash, cwins, cn_calmar = np.nan, "n/a", np.nan
                row = dict(panel=key, subset="+".join(sub), n=len(sub), scheme=lab,
                           weights="|".join(f"{s}:{w:.2f}" for s, w in sorted(wm.items())),
                           cagr=round(float(np.median(c)), 2),
                           cagr_lo=round(float(c.min()), 2), cagr_hi=round(float(c.max()), 2),
                           dd=round(float(np.median(dd)), 2),
                           dd_worst=round(float(dd.min()), 2),
                           calmar=round(float(np.nanmedian(k)), 3),
                           calmar_worst=round(float(np.nanmin(k)), 3),
                           d_cagr=round(float(np.median(c - bc)), 2),
                           d_dd=round(float(np.median(dd - bd)), 2),
                           d_calmar=round(float(np.nanmedian(k - bk)), 3),
                           calmar_wins=f"{int(np.nansum(k > bk))}/{NPATH}",
                           cn_calmar=cn_calmar,
                           d_calmar_vs_cash=(round(dk_cash, 3) if dk_cash == dk_cash else np.nan),
                           cash_wins=cwins)
                (rows if lab == "EW" else wrows).append(row)
        log(f"panel {key} done: {len([r for r in rows if r['panel']==key])} EW rows, "
            f"{len([r for r in wrows if r['panel']==key])} weighted rows")
    pd.DataFrame(rows).to_csv(RES / "p2_subsets.csv", index=False)
    pd.DataFrame(wrows).to_csv(RES / "p3_weights.csv", index=False)
    log(f"P2/P3 written: {len(rows)} EW cells, {len(wrows)} weighted cells, "
        f"{len(rows)+len(wrows)} total (x{NPATH} paths each)")
    return rows, wrows


# ------------------------------------------------------------------- P5 windows
def p5(mm, finalists):
    out = []
    for key in PANELS:
        pn = Panel(key, mm)
        cands = [("TN+OA 50-50 (deployed)", {"OA": 0.5, "TN": 0.5})]
        for name, wm in finalists:
            if set(wm) <= set(pn.cfg["members"]):
                cands.append((name, wm))
        # single sleeves for reference
        for s in pn.cfg["members"]:
            cands.append((f"{s} alone", {s: 1.0}))
        for name, wm in cands:
            nav = pn.blend(wm)
            row = dict(panel=key, book=name)
            for wn, (a, b) in WINDOWS.items():
                r, dd = pn.window(nav, a, b)
                row[f"{wn} ret"] = None if r != r else round(r, 1)
                row[f"{wn} dd"] = None if dd != dd else round(dd, 1)
            out.append(row)
    df = pd.DataFrame(out)
    df.to_csv(RES / "p5_windows.csv", index=False)
    log("P5 written")
    return df


if __name__ == "__main__":
    d = load_daily()
    mm = build_monthly(d)
    for k, v in mm.items():
        log(f"{k}: {v.index[0]} -> {v.index[-1]} n_months={len(v)} cols={len(v.columns)}")
    p1(mm, d)
    rows, wrows = run_cells(mm)
    log("P1-P3 DONE")
