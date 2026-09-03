"""research/144 — 50-50 monthly-rebalanced blend of True North (both gate actions) with the
Open Alpha ADOPTED spec (r/142): trail-15 SMA, -8% stop, 16 slots @6.25%, NO gate, 25 bps,
cash_yield 5%, calendar-year-netted STCG 20% / LTCG 12.5%. 30 seeds. After-tax on both legs.

Question (Arun): does the softer TN gate action (block-new-only, stays partially invested
through corrections) make TN a better or worse blend partner than liquidate-all?
"""
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OA = Path("/home/arun/quantifyd/research/142_bananapatterns_replication/scripts")
sys.path.insert(0, str(OA))
import bluesky_replay as br

spec = importlib.util.spec_from_file_location("tn", str(HERE / "tn_sweep.py"))
tn = importlib.util.module_from_spec(spec); spec.loader.exec_module(tn)

SEEDS = range(1, 31)

# ── OA leg: adopted spec, 30 seeds, after-tax ──
print("loading OA frames (trail_sma=15) ...", flush=True)
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

oa_navs = []
t0 = time.time()
for seed in SEEDS:
    eq, _, _ = br.simulate(seed, "random", days, dates, C, H, O, ATH, S, RSv, TVv,
                           trig, weak, True, 0.0025, stop=0.08, slots=16,
                           size_pct=0.0625, stcg=0.20, ltcg=0.125, cash_yield=0.05)
    oa_navs.append(pd.Series(np.asarray(eq, float), index=dates[days]))
print(f"OA 30 seeds done [{time.time()-t0:.0f}s]", flush=True)

# ── TN legs (after-tax, offset 0) ──
ctx = tn.Ctx()
tn_navs = {}
for tag, kw in [("TN_cash(incumbent)", dict(action="cash")),
                ("TN_block", dict(action="block"))]:
    r = tn.run(ctx, series="NIFTYBEES", cons="sma100", n=8, exit=("donch", 15),
               tax=True, **kw)
    tn_navs[tag] = r["_nav"]
    print(f"{tag}: waCAGR {r['wa_cagr']} waDD {r['wa_dd']}", flush=True)


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = float((nav / nav.cummax() - 1).min())
    return cagr * 100, dd * 100, (cagr / abs(dd) if dd < 0 else np.nan)


rows = []
for tag, mnav in tn_navs.items():
    cagr_l, dd_l, cal_l, cd_l, cm_l = [], [], [], [], []
    for onav in oa_navs:
        idx = onav.index.intersection(mnav.index)
        o, m = onav.loc[idx], mnav.loc[idx]
        b_m = o.resample("ME").last().pct_change().fillna(0)
        m_m = m.resample("ME").last().pct_change().fillna(0)
        blend = (1 + 0.5 * b_m + 0.5 * m_m).cumprod()
        c, d, k = stats(blend)
        cagr_l.append(c); dd_l.append(d); cal_l.append(k)
        cd_l.append(o.pct_change().corr(m.pct_change()))
        cm_l.append(b_m.corr(m_m))
    row = dict(tn_leg=tag,
               blend_cagr_med=round(float(np.median(cagr_l)), 2),
               blend_cagr_min=round(min(cagr_l), 2), blend_cagr_max=round(max(cagr_l), 2),
               blend_dd_med=round(float(np.median(dd_l)), 2),
               blend_dd_min=round(min(dd_l), 2), blend_dd_max=round(max(dd_l), 2),
               blend_calmar_med=round(float(np.median(cal_l)), 2),
               corr_daily_med=round(float(np.median(cd_l)), 3),
               corr_monthly_med=round(float(np.median(cm_l)), 3))
    tc, td, tk = stats(mnav.loc[mnav.index.intersection(oa_navs[0].index)])
    row.update(tn_solo_cagr=round(tc, 2), tn_solo_dd=round(td, 2))
    rows.append(row)
    print(row, flush=True)

oc = [stats(o)[0] for o in oa_navs]; od = [stats(o)[1] for o in oa_navs]
print(f"OA solo (after-tax): CAGR med {np.median(oc):.2f} [{min(oc):.2f}..{max(oc):.2f}] "
      f"DD med {np.median(od):.2f} [{min(od):.2f}..{max(od):.2f}]", flush=True)
pd.DataFrame(rows).to_csv(RES / "blend_oa.csv", index=False)
print("DONE", flush=True)
