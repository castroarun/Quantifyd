"""research/155 — phase runner for the IPO idle-cash redeployment study.

Resume-safe: every completed (cell, path) is appended to results/paths.csv immediately and
skipped on a re-run.  One row per (cell, path); report.py does all aggregation and pairing.

Usage:
    venv/bin/python -u research/155_ipo_cash_redeployment/scripts/run_sweep.py [phase ...]
    phases: R 1 2 3 4     (default: all)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RES = STUDY / "results"
RES.mkdir(exist_ok=True)
ROOT = Path("/home/arun/quantifyd")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research" / "153_ipo_base" / "scripts"))
sys.path.insert(0, str(HERE))

import ipo_replay as ir           # noqa: E402
import ipo_park as ip             # noqa: E402

R153 = ROOT / "research" / "153_ipo_base" / "results"
R154 = ROOT / "research" / "154_multi_system_blends" / "results"

SPEC = json.loads((R153 / "ipo_adopted_spec.json").read_text())
N_PATHS = 30
W_TN, W_OA, W_IPO = 0.40, 0.40, 0.20
PATHS_CSV = RES / "paths.csv"

FIELDS = ["cell", "phase", "arm", "asset", "settle", "reserve", "cadence", "sell",
          "tax", "cost_bps", "gateN", "frictionless", "path", "oacol", "tncol", "seed",
          "s_cagr", "s_dd", "s_calmar", "s_invested", "s_parked_pct", "s_trades",
          "s_final_x", "b_cagr", "b_dd", "b_calmar",
          "b_dd_0809", "b_dd_1214", "b_dd_1314", "b_dd_2126",
          "b_ret_0809", "b_ret_1214", "b_ret_1314", "b_ret_2126",
          "n_pull", "pull_val_x", "pull_cost_x", "pull_tax_x", "n_park",
          "park_cost_x", "n_missed", "park_days_pct",
          "corr_d_oa", "corr_d_tn", "corr_m_oa", "corr_m_tn", "secs"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ─────────────────────────────────────────────────────────────────────── data
def load_all():
    ctx = ir.Ctx(verbose=True)
    trig, piv, lo = ir.build_trigger(
        ctx, max_age_m=SPEC["max_age_m"], min_bars=SPEC["min_bars"], L=SPEC["L"],
        max_depth=SPEC["max_depth"], rs_policy=SPEC["rs_policy"], rs_min=SPEC["rs_min"],
        tight_max=SPEC["tight_max"], pivot_mode=SPEC["pivot_mode"])
    sma = ctx.sma(SPEC["trail"])
    dates = ctx.dates
    days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= "2006-01-01"])
    weak = ctx.NOWEAK
    used = dates[days]

    oa = pd.read_csv(R154 / "oa_navs30.csv", index_col=0, parse_dates=True)
    tn = pd.read_csv(R154 / "tn_navs12.csv", index_col=0, parse_dates=True)
    nb = ctx.close.get("NIFTYBEES")

    def align(s):
        return s.reindex(dates).ffill().values.astype(float)

    park_assets = {
        "OA": {c: align(oa[c]) for c in oa.columns},
        "TN": {c: align(tn[c]) for c in tn.columns},
        "NB": {"nb": align(nb)},
    }
    log(f"OA navs {oa.index[0].date()}->{oa.index[-1].date()} {oa.shape}; "
        f"TN navs {tn.index[0].date()}->{tn.index[-1].date()} {tn.shape}")
    return dict(ctx=ctx, trig=trig, piv=piv, lo=lo, sma=sma, dates=dates, days=days,
                weak=weak, used=used, oa=oa, tn=tn, park=park_assets)


def mix5050(a, b):
    """Daily-rebalanced 50/50 of two NAV levels -> a level series."""
    ra = np.zeros_like(a); rb = np.zeros_like(b)
    ra[1:] = a[1:] / a[:-1] - 1.0
    rb[1:] = b[1:] / b[:-1] - 1.0
    r = 0.5 * np.nan_to_num(ra) + 0.5 * np.nan_to_num(rb)
    return np.cumprod(1.0 + r)


# ───────────────────────────────────────────────────────────────────── stats
def curve_stats(nav, idx):
    s = pd.Series(nav, index=idx)
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
    dd = float((s / s.cummax() - 1).min() * 100)
    return cagr, dd


def wdd_full(s, a, b):
    """Sub-window drawdown measured from the running peak of the FULL curve (r/154)."""
    ddser = s / s.cummax() - 1.0
    seg = ddser[(ddser.index >= a) & (ddser.index <= b)]
    return float(seg.min() * 100) if len(seg) else np.nan


def wret(s, a, b):
    seg = s[(s.index >= a) & (s.index <= b)]
    return float((seg.iloc[-1] / seg.iloc[0] - 1) * 100) if len(seg) > 1 else np.nan


WINDOWS = dict(_0809=("2008-01-01", "2009-12-31"), _1214=("2012-01-01", "2014-12-31"),
               _1314=("2013-01-01", "2014-12-31"), _2126=("2021-01-01", "2026-12-31"))


# ────────────────────────────────────────────────────────────────── the runner
def run_cell(D, cell, phase, *, arm, asset=None, settle=1, reserve=0, cadence="daily",
             sell="prorata", tax="full", cost_bps=25, gateN=None, frictionless=False,
             done=None, writer=None, fh=None, save_nav=False):
    ctx, dates, days, used = D["ctx"], D["dates"], D["days"], D["used"]
    oa, tn = D["oa"], D["tn"]
    cost = cost_bps / 10000.0
    key = cell
    if done is not None and done.get(key, 0) >= N_PATHS:
        log(f"skip {cell} (done)")
        return None
    park_allowed = None
    if gateN is not None:
        park_allowed = D.setdefault("_gate", {}).get(gateN)
        if park_allowed is None:
            t0 = time.time()
            park_allowed = ip.forward_pool_empty(ctx, gateN, SPEC["max_age_m"],
                                                 SPEC["min_bars"])
            D["_gate"][gateN] = park_allowed
            log(f"gate N={gateN}: pool empty on {park_allowed[days].mean()*100:.1f}% "
                f"of days ({time.time()-t0:.0f}s)")

    navs = {}
    for p in range(N_PATHS):
        t0 = time.time()
        oacol, tncol, seed = f"s{p+1}", f"off{p % 12}", p + 1
        if asset is None:
            lvl = None
        elif asset == "OA":
            lvl = D["park"]["OA"][oacol]
        elif asset == "TN":
            lvl = D["park"]["TN"][tncol]
        elif asset == "MIX":
            lvl = mix5050(D["park"]["OA"][oacol], D["park"]["TN"][tncol])
        elif asset == "NB":
            lvl = D["park"]["NB"]["nb"]
        else:
            raise ValueError(asset)

        r = ip.simulate_park(
            seed, days, dates, ctx.C, ctx.O, D["piv"], D["lo"], D["sma"], ctx.TVp,
            D["trig"], D["weak"], park_lvl=lvl, park_allowed=park_allowed,
            cost=cost, stop=SPEC["stop"], slots=SPEC["slots"], size_pct=SPEC["size_pct"],
            target=SPEC["target"], settle_days=settle, reserve_slots=reserve,
            cadence=cadence, sell_policy=sell, park_tax=tax, frictionless=frictionless)
        nav = pd.Series(r["nav"], index=used)
        navs[f"seed{seed}"] = nav

        # ── blend, monthly rebalanced, on this exact path ──
        idx = nav.index.intersection(oa.index).intersection(tn.index)
        n_ = (nav.loc[idx] / nav.loc[idx].iloc[0])
        o_ = (oa[oacol].loc[idx] / oa[oacol].loc[idx].iloc[0])
        t_ = (tn[tncol].loc[idx] / tn[tncol].loc[idx].iloc[0])
        nm = n_.resample("ME").last().pct_change().fillna(0.0)
        om = o_.resample("ME").last().pct_change().fillna(0.0)
        tm = t_.resample("ME").last().pct_change().fillna(0.0)
        bl = (1 + W_OA * om + W_TN * tm + W_IPO * nm).cumprod()
        bc, bd = curve_stats(bl.values, bl.index)
        sc, sd = curve_stats(n_.values, n_.index)
        dg = r["diag"]
        row = dict(cell=cell, phase=phase, arm=arm, asset=asset or "cash", settle=settle,
                   reserve=reserve, cadence=cadence, sell=sell, tax=tax,
                   cost_bps=cost_bps, gateN=gateN if gateN is not None else "",
                   frictionless=int(frictionless), path=p, oacol=oacol, tncol=tncol,
                   seed=seed,
                   s_cagr=round(sc, 4), s_dd=round(sd, 4),
                   s_calmar=round(sc / abs(sd), 4) if sd else np.nan,
                   s_invested=round(float(np.mean(r["invested"] / r["nav"]) * 100), 3),
                   s_parked_pct=round(float(np.mean(r["parked"] / r["nav"]) * 100), 3),
                   s_trades=len(r["trades"]),
                   s_final_x=round(float(n_.iloc[-1]), 4),
                   b_cagr=round(bc, 4), b_dd=round(bd, 4),
                   b_calmar=round(bc / abs(bd), 4) if bd else np.nan,
                   n_pull=dg["n_pull"],
                   pull_val_x=round(dg["pull_val"] / ip.CAPITAL, 3),
                   pull_cost_x=round(dg["pull_cost"] / ip.CAPITAL, 5),
                   pull_tax_x=round(dg["pull_tax"] / ip.CAPITAL, 5),
                   n_park=dg["n_park"],
                   park_cost_x=round(dg["park_cost_paid"] / ip.CAPITAL, 5),
                   n_missed=dg["n_missed_settle"],
                   park_days_pct=round(dg["park_days_pct"], 2),
                   secs=round(time.time() - t0, 2))
        for tag, (a, b) in WINDOWS.items():
            row[f"b_dd{tag}"] = round(wdd_full(bl, a, b), 3)
            row[f"b_ret{tag}"] = round(wret(bl, a, b), 3)
        nd = n_.pct_change().fillna(0.0)
        od = o_.pct_change().fillna(0.0)
        td = t_.pct_change().fillna(0.0)
        row["corr_d_oa"] = round(float(nd.corr(od)), 4)
        row["corr_d_tn"] = round(float(nd.corr(td)), 4)
        row["corr_m_oa"] = round(float(nm.corr(om)), 4)
        row["corr_m_tn"] = round(float(nm.corr(tm)), 4)
        writer.writerow({k: row.get(k, "") for k in FIELDS})
        fh.flush()
    if save_nav:
        pd.DataFrame(navs).to_csv(RES / f"nav_{cell}.csv")
    log(f"  cell {cell} done")
    return navs


def main():
    phases = sys.argv[1:] or ["R", "1", "2", "3", "3b", "4", "5"]
    D = load_all()

    done = {}
    if PATHS_CSV.exists():
        df = pd.read_csv(PATHS_CSV)
        done = df.groupby("cell").size().to_dict()
        log(f"resume: {len(done)} cells already in paths.csv")
    fh = open(PATHS_CSV, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if fh.tell() == 0:
        writer.writeheader()

    def cell(*a, **kw):
        return run_cell(D, *a, done=done, writer=writer, fh=fh, **kw)

    # ── Phase R: replication of r/153's arm-A sleeve ──
    if "R" in phases:
        log("PHASE R — replication of the r/153 sleeve (arm A, no parking)")
        cell("A_incumbent", "R", arm="A", asset=None, save_nav=True)

    # ── Phase 1: parking bounds ──
    if "1" in phases:
        log("PHASE 1 — parking asset bounds (T+1, reserve 0, daily, pro-rata, tax=full)")
        for tag, asset in (("B_OA", "OA"), ("C_TN", "TN"), ("D_MIX", "MIX"), ("N_NB", "NB")):
            cell(f"P1_{tag}", "1", arm=tag[0], asset=asset, save_nav=True)
        # frictionless twins -> the headline "what does the mechanism cost" number
        for tag, asset in (("B_OA", "OA"), ("C_TN", "TN"), ("D_MIX", "MIX")):
            cell(f"P1_{tag}_frictionless", "1", arm=tag[0], asset=asset, frictionless=True)

    # ── Phase 2: mechanics sweep on the winning asset ──
    if "2" in phases:
        best = os.environ.get("R155_BEST_ASSET", "OA")
        log(f"PHASE 2 — mechanics sweep on asset={best}")
        for settle in (1, 0):
            for reserve in (0, 1, 2):
                for cadence in ("daily", "weekly", "monthly"):
                    for sell in ("prorata", "lifo", "fifo"):
                        c = f"P2_{best}_s{settle}_r{reserve}_{cadence}_{sell}"
                        cell(c, "2", arm="M", asset=best, settle=settle, reserve=reserve,
                             cadence=cadence, sell=sell)

    # ── Phase 3: arm E, forward-visibility gate ──
    if "3" in phases:
        mech = json.loads(os.environ.get(
            "R155_BEST_MECH", '{"settle":1,"reserve":0,"cadence":"daily","sell":"prorata"}'))
        log(f"PHASE 3 — forward-visibility gate, mech={mech}")
        for asset in ("OA", "TN", "MIX"):
            for gn in (25, 50, 100):
                cell(f"P3_E{gn}_{asset}", "3", arm="E", asset=asset, gateN=gn,
                     save_nav=(asset == "OA"), **mech)

    # ── Phase 3b: the gate is the whole point -- a cash reserve inside it is redundant,
    #    because by construction nothing can trigger while the gate is open ──
    if "3b" in phases:
        log("PHASE 3b — gated arm without the throttling reserve")
        for asset in ("OA", "TN", "MIX", "NB"):
            for res in (0, 1):
                for cad in ("daily", "monthly"):
                    cell(f"P3b_E25_{asset}_r{res}_{cad}", "3b", arm="E", asset=asset,
                         gateN=25, settle=1, reserve=res, cadence=cad, sell="lifo",
                         save_nav=(asset == "OA" and res == 0 and cad == "monthly"))
        # and the T+0 / no-tax sensitivities on the best-shaped one
        for tx in ("full", "txn"):
            for st in (1, 0):
                cell(f"P3b_E25_OA_r0_monthly_s{st}_{tx}", "3b", arm="E", asset="OA",
                     gateN=25, settle=st, reserve=0, cadence="monthly", sell="lifo",
                     tax=tx)
        cell("P3b_E25_OA_r0_monthly_frictionless", "3b", arm="E", asset="OA", gateN=25,
             settle=1, reserve=0, cadence="monthly", sell="lifo", frictionless=True)

    # ── Phase 4: ladders ──
    if "4" in phases:
        best = os.environ.get("R155_BEST_ASSET", "OA")
        mech = json.loads(os.environ.get(
            "R155_BEST_MECH", '{"settle":1,"reserve":0,"cadence":"daily","sell":"prorata"}'))
        gn = os.environ.get("R155_BEST_GATE", "")
        log("PHASE 4 — cost / tax ladders")
        for cb in (25, 40, 60):
            cell(f"P4_A_c{cb}", "4", arm="A", asset=None, cost_bps=cb)
            for tx in ("full", "txn"):
                cell(f"P4_{best}_c{cb}_{tx}", "4", arm="L", asset=best, cost_bps=cb,
                     tax=tx, **mech)
        if gn:
            for cb in (25, 40, 60):
                for tx in ("full", "txn"):
                    cell(f"P4_E{gn}_{best}_c{cb}_{tx}", "4", arm="E", asset=best,
                         gateN=int(gn), cost_bps=cb, tax=tx, **mech)
    # ── Phase 5: cost/tax ladder on the GATED arm as actually specified (no reserve) ──
    if "5" in phases:
        log("PHASE 5 — cost / tax ladder on the gated arm (reserve 0, monthly, LIFO)")
        gm = dict(settle=1, reserve=0, cadence="monthly", sell="lifo")
        for cb in (25, 40, 60):
            for tx in ("full", "txn"):
                cell(f"P5_E25_OA_c{cb}_{tx}", "5", arm="E", asset="OA", gateN=25,
                     cost_bps=cb, tax=tx, **gm)
        cell("P5_E25_OA_frictionless", "5", arm="E", asset="OA", gateN=25,
             frictionless=True, **gm)
    fh.close()
    log("SWEEP DONE")


if __name__ == "__main__":
    main()
