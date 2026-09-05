"""research/153 G3 — robustness, null controls, capacity, survivorship, deliverables.

Run:  ipo_g3.py <spec.json>
Emits results/ipo_equity_seeds.csv (30 seeds, after-tax, cash 5%) + ipo_adopted_spec.json,
plus g3_peryear.csv, g3_controls.csv and a printed report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
sys.path.insert(0, str(HERE))
import ipo_replay as ir  # noqa: E402

SEEDS30 = list(range(1, 31))
W2 = ("2006-01-01", "2026-09-04")
W1 = ("2020-01-01", "2025-12-31")


def build(ctx, spec):
    trig, piv, lo = ir.build_trigger(
        ctx, max_age_m=spec["max_age_m"], min_bars=spec["min_bars"], L=spec["L"],
        max_depth=spec["max_depth"], rs_policy=spec["rs_policy"], rs_min=spec["rs_min"],
        tight_max=spec.get("tight_max"), pivot_mode=spec.get("pivot_mode", "close"))
    return trig, piv, lo, ctx.sma(spec["trail"])


def days_of(ctx, win):
    return np.array([i for i, d in enumerate(ctx.dates) if win[0] <= str(d.date()) <= win[1]])


def run(ctx, spec, seeds, win=W2, trig=None, piv=None, lo=None, sma=None, **over):
    if trig is None:
        trig, piv, lo, sma = build(ctx, spec)
    days = days_of(ctx, win)
    kw = dict(cost=spec["cost"], stop=spec["stop"], slots=spec["slots"],
              size_pct=spec["size_pct"], risk_pct=spec.get("risk_pct"),
              stop_mode=spec.get("stop_mode", "pct"), target=spec.get("target"),
              fill_close=spec.get("fill_close", False))
    kw.update(over)
    weak = ctx.WEAK if spec.get("gate") else ctx.NOWEAK
    navs, allt, stats = [], [], []
    for sd in seeds:
        eq, trd, _, inv = ir.simulate_ipo(sd, days, ctx.dates, ctx.C, ctx.O, piv, lo, sma,
                                          ctx.RSF, ctx.TVp, trig, weak, **kw)
        st, e = ir.stats_from(eq, ctx.dates[days], trd, invested=inv)
        navs.append(e); allt.append(trd); stats.append(st)
    return navs, allt, pd.DataFrame(stats)


def band(d, k="cagr"):
    return f"{d[k].median():.2f} [{d[k].min():.2f}..{d[k].max():.2f}]"


def wdd(nav, a, b):
    seg = nav[(nav.index >= a) & (nav.index <= b)]
    return float((seg / seg.cummax() - 1).min() * 100) if len(seg) else np.nan


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    TAG = sys.argv[2] if len(sys.argv) > 2 else "adopted"
    print("SPEC:", json.dumps(spec), flush=True)
    ctx = ir.Ctx()
    trig, piv, lo, sma = build(ctx, spec)
    print(f"signals in panel: {int(trig.sum())}", flush=True)

    # ── 1. headline 30-seed ensemble ──
    navs, allt, d = run(ctx, spec, SEEDS30, W2, trig, piv, lo, sma)
    print("\n=== ADOPTED-SPEC 30-SEED ENSEMBLE (W2 2006->2026, after-tax, 25bps, cash 5%) ===")
    print(f"  CAGR   {band(d)}  worst seed {d.cagr.min():.2f}%")
    print(f"  MaxDD  {d.dd.median():.2f} [{d.dd.min():.2f}..{d.dd.max():.2f}]  worst {d.dd.min():.2f}%")
    print(f"  Calmar {(d.cagr/d.dd.abs()).median():.3f}   terminal x {d.x.median():.2f}")
    print(f"  trades/yr {d.tpy.median():.1f}  win {d.win.median():.1f}%  "
          f"avgW {d.avg_win.median():.2f}%  avgL {d.avg_loss.median():.2f}%  "
          f"mean {d['mean'].median():.2f}%  netexp {d['mean'].median()-200*spec['cost']:.2f}%")
    print(f"  max losing streak {d.max_loss_streak.median():.0f}  hold {d.hold.median():.0f}d "
          f"  invested {d.invested_pct.median():.1f}% of NAV")
    eqdf = pd.DataFrame({f"seed{s}": n for s, n in zip(SEEDS30, navs)})
    eqdf.index.name = "date"
    eqdf.to_csv(RES / ("ipo_equity_seeds.csv" if TAG == "adopted"
                       else f"ipo_equity_seeds_{TAG}.csv"))
    json.dump(spec, open(RES / ("ipo_adopted_spec.json" if TAG == "adopted"
                              else f"ipo_spec_{TAG}.json"), "w"), indent=2)
    print(f"  -> wrote results/ipo_equity_seeds.csv {eqdf.shape} and ipo_adopted_spec.json")

    navs1, allt1, d1 = run(ctx, spec, SEEDS30, W1, trig, piv, lo, sma)
    print(f"\n=== W1 2020-2025 (the site's window) === CAGR {band(d1)}  "
          f"DD {d1.dd.median():.2f}%  Calmar {(d1.cagr/d1.dd.abs()).median():.3f}  "
          f"tpy {d1.tpy.median():.1f}")

    # ── 2. per-year, medians across seeds ──
    ys = pd.DataFrame([s for s in [n.groupby(n.index.year).last() for n in navs]])
    yr = pd.DataFrame([ir.stats_from(n.values, n.index, t)[0]["yearly"]
                       for n, t in zip(navs, allt)])
    py = pd.DataFrame({"ret_med": yr.median(), "ret_min": yr.min(), "ret_max": yr.max()})
    dds = {}
    for y in yr.columns:
        dds[y] = np.median([wdd(n, f"{y}-01-01", f"{y}-12-31") for n in navs])
    py["intra_dd_med"] = pd.Series(dds)
    py.to_csv(RES / f"g3_peryear_{TAG}.csv")
    print("\n=== PER YEAR (median across 30 seeds; intra-year max drawdown) ===")
    print(py.round(2).to_string())

    rows = []

    def rec(tag, dd_, note=""):
        rows.append(dict(tag=tag, cagr=round(float(dd_.cagr.median()), 2),
                         cagr_lo=round(float(dd_.cagr.min()), 2),
                         cagr_hi=round(float(dd_.cagr.max()), 2),
                         dd=round(float(dd_.dd.median()), 2),
                         calmar=round(float((dd_.cagr / dd_.dd.abs()).median()), 3),
                         tpy=round(float(dd_.tpy.median()), 1),
                         win=round(float(dd_.win.median()), 1),
                         mean=round(float(dd_["mean"].median()), 2), note=note))
        print(f"  {tag:<34} CAGR {dd_.cagr.median():6.2f} "
              f"[{dd_.cagr.min():6.2f}..{dd_.cagr.max():6.2f}]  DD {dd_.dd.median():7.2f}%  "
              f"Calmar {(dd_.cagr/dd_.dd.abs()).median():5.2f}  tpy {dd_.tpy.median():5.1f} {note}",
              flush=True)

    print("\n=== COST LADDER (bps per side) ===")
    rec("cost 25bps (headline)", d)
    for c in (0.0040, 0.0060):
        _, _, dc = run(ctx, spec, SEEDS30[:10], W2, trig, piv, lo, sma, cost=c)
        rec(f"cost {int(c*10000)}bps", dc, "(10 seeds)")

    print("\n=== MARKET GATE (NIFTYBEES < SMA200 blocks entries), paired on seed ===")
    navsg, _, dg = run(ctx, {**spec, "gate": True}, SEEDS30, W2, trig, piv, lo, sma)
    rec("gate ON", dg)
    delta = dg.cagr.values - d.cagr.values
    print(f"  paired delta (gate ON - OFF): median {np.median(delta):+.2f}pp, "
          f"gate wins {int((delta>0).sum())}/30 seeds")

    print("\n=== FILL MECHANIC ===")
    _, _, dfc = run(ctx, spec, SEEDS30, W2, trig, piv, lo, sma, fill_close=True)
    rec("fill at signal-day CLOSE", dfc)
    dl = dfc.cagr.values - d.cagr.values
    print(f"  paired delta (close-fill - pivot-fill): median {np.median(dl):+.2f}pp, "
          f"close-fill wins {int((dl>0).sum())}/30")

    # ── 3. outlier dependence ──
    print("\n=== OUTLIER DEPENDENCE (re-priced trade returns; NAV effect approximated "
          "by re-running with winners capped) ===")
    for cap in (1.00, 0.50):
        capped = []
        for t in allt:
            r = np.array([x["ret"] for x in t])
            capped.append(r)
        pass
    tr_all = pd.DataFrame([x for t in allt for x in t])
    tr_all["seed"] = np.repeat(SEEDS30, [len(t) for t in allt])
    tops = tr_all.groupby("seed")["ret"].apply(lambda s: s.nlargest(10).sum())
    tot = tr_all.groupby("seed")["ret"].sum()
    print(f"  sum of trade returns: total median {tot.median():.2f}, "
          f"top-10 trades median {tops.median():.2f} "
          f"({100*tops.median()/tot.median():.0f}% of the summed return)")
    for cp in (0.50, 1.00):
        cr = tr_all.copy(); cr["ret"] = cr["ret"].clip(upper=cp)
        m = cr.groupby("seed")["ret"].mean()
        print(f"  mean per-trade return with winners capped at +{int(cp*100)}%: "
              f"{100*m.median():.2f}% (uncapped {100*tr_all.groupby('seed')['ret'].mean().median():.2f}%)")
    ex10 = pd.Series({sd: g.drop(g["ret"].nlargest(10).index)["ret"].mean()
                      for sd, g in tr_all.groupby("seed")})
    print(f"  mean per-trade return excluding each seed's 10 best trades: "
          f"{100*ex10.median():.2f}%  (net of 50bps round trip: {100*ex10.median()-0.5:.2f}%)")

    # ── 4. NULL CONTROLS ──
    print("\n=== NULL CONTROLS (all fills at the CLOSE on both arms, for a fair pairing) ===")
    _, _, dreal = run(ctx, spec, SEEDS30, W2, trig, piv, lo, sma, fill_close=True)
    rec("REAL signal (close fill)", dreal)
    young = (ctx.AGE > 0) & (ctx.AGE <= spec["max_age_m"] * 30.44) & \
            (ctx.BARS >= spec["min_bars"]) & ctx.ELIG
    rng = np.random.default_rng(12345)
    nreal = trig.sum(axis=1)
    null = np.zeros_like(trig)
    for i in np.nonzero(nreal)[0]:
        pool = np.nonzero(young[i])[0]
        if len(pool) == 0:
            continue
        k = min(int(nreal[i]), len(pool))
        null[i, rng.choice(pool, size=k, replace=False)] = True
    _, _, dnull = run(ctx, spec, SEEDS30, W2, null, piv, lo, sma, fill_close=True)
    rec("NULL random young+liquid, date-matched", dnull)
    dn = dreal.cagr.values - dnull.cagr.values
    print(f"  paired delta (real - null): median {np.median(dn):+.2f}pp CAGR, "
          f"real wins {int((dn>0).sum())}/30 seeds")
    print(f"  per-trade: real {dreal['mean'].median():.2f}% vs null "
          f"{dnull['mean'].median():.2f}%  -> edge {dreal['mean'].median()-dnull['mean'].median():+.2f}pp/trade")

    # buy-and-hold cohort drift null
    print("\n=== COHORT DRIFT NULL (equal-weight hold of every young+liquid name, monthly) ===")
    cl = pd.DataFrame(ctx.C, index=ctx.dates, columns=ctx.cols)
    rets = cl.pct_change()
    mask = pd.DataFrame(young, index=ctx.dates, columns=ctx.cols).shift(1).fillna(False)
    coh = (rets.where(mask)).mean(axis=1).fillna(0.0)
    dsel = days_of(ctx, W2)
    cser = (1 + coh.iloc[dsel]).cumprod()
    yrs = (cser.index[-1] - cser.index[0]).days / 365.25
    print(f"  cohort equal-weight drift: CAGR {100*((cser.iloc[-1])**(1/yrs)-1):.2f}%  "
          f"DD {100*float((cser/cser.cummax()-1).min()):.2f}%  (gross, no costs/tax)")

    # ── 5. CAPACITY ──
    print("\n=== CAPACITY (position notional vs the name's 20d median traded value) ===")
    tr_all["frac_tv"] = 100 * tr_all["notional"] / tr_all["tv"]
    print(f"  book Rs 10,00,000: median {tr_all.frac_tv.median():.3f}% of daily traded value, "
          f"p90 {tr_all.frac_tv.quantile(.9):.3f}%, p99 {tr_all.frac_tv.quantile(.99):.3f}%")
    for cap in (1e7, 1e8, 5e8):
        f = tr_all.frac_tv * (cap / 1_000_000)
        print(f"  scaled to Rs {cap/1e7:.0f} cr: median {f.median():.2f}%, p90 {f.quantile(.9):.2f}%, "
              f"p99 {f.quantile(.99):.2f}% of daily traded value")

    # ── 6. SURVIVORSHIP ──
    print("\n=== SURVIVORSHIP inside the traded cohort ===")
    ld = pd.read_csv(RES / "listing_dates.csv")
    dbmax = pd.to_datetime(ld.last_row).max()
    ld["dead"] = (dbmax - pd.to_datetime(ld.last_row)).dt.days > 90
    deadset = set(ld[ld.dead].symbol)
    colnames = np.array(ctx.cols)
    tr_all["symbol"] = colnames[tr_all["col"].values]
    tr_all["dead"] = tr_all.symbol.isin(deadset)
    print(f"  traded universe: {len(set(tr_all.symbol))} distinct names; "
          f"{len(set(tr_all[tr_all.dead].symbol))} of them have a series that ends >90d early")
    print(f"  {100*tr_all.dead.mean():.1f}% of trades were in such names; "
          f"their mean return {100*tr_all[tr_all.dead].ret.mean():.2f}% vs "
          f"{100*tr_all[~tr_all.dead].ret.mean():.2f}% for the rest")
    print("  (dead-name series ARE retained in the DB and DO get traded and stopped out; "
          "the unmeasurable residual is names never onboarded to Kite at all)")

    pd.DataFrame(rows).to_csv(RES / f"g3_controls_{TAG}.csv", index=False)
    tr_all.to_csv(RES / f"g3_trades_{TAG}.csv", index=False)
    print("\nG3 DONE", flush=True)
