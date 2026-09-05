"""research/153 G4b — sleeve bake-off (narrow / mid / wide) x weight x window, plus a
correctly-computed capacity table and a 4-sleeve check against GOLDBEES (r/147's winner).

Capacity note: the G3 print scaled 'notional / traded value' using the COMPOUNDING NAV,
which inflates late-period positions and understates capacity. The correct question is
"for a book of NAV N, position = size_pct*N; what fraction of the held name's 20-day median
traded value is that?" -- recomputed here from the traded names' actual TV distribution.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
R146 = Path("/home/arun/quantifyd/research/146_complementary_third_sleeve/results")
DB = Path("/home/arun/quantifyd/backtest_data/market_data.db")

oa = pd.read_csv(R146 / "oa_navs.csv", index_col=0, parse_dates=True)
tn = pd.concat([pd.read_csv(R146 / f"tn_nav_off{o}.csv", index_col=0,
                            parse_dates=True).rename(columns={"0": f"off{o}"})
                for o in (0, 4, 8)], axis=1)
cash = pd.read_csv(R146 / "nav_cashnull_tax1.csv", index_col=0, parse_dates=True)
cash.columns = ["cash"]
sleeves = {"narrow(<=3m)": "ipo_equity_seeds.csv",
           "mid(<=6m)": "ipo_equity_seeds_mid.csv",
           "wide(<=24m)": "ipo_equity_seeds_wide.csv"}
sl = {k: pd.read_csv(RES / v, index_col=0, parse_dates=True) for k, v in sleeves.items()}

# GOLDBEES (research/147's adopted third sleeve) for a like-for-like on 2015+
con = sqlite3.connect(str(DB))
g = pd.read_sql_query("select date, close from market_data_unified where symbol='GOLDBEES' "
                      "and timeframe='day' order by date", con)
g["date"] = pd.to_datetime(g["date"].str[:10])
gold = g.set_index("date")["close"].astype(float)
con.close()


def stats(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100, \
        float((nav / nav.cummax() - 1).min() * 100)


def wdd(nav, a, b):
    seg = nav[(nav.index >= a) & (nav.index <= b)]
    return float((seg / seg.cummax() - 1).min() * 100) if len(seg) else np.nan


def monthly(df, idx):
    return df.loc[idx].resample("ME").last().pct_change().fillna(0.0)


for wname, wstart in (("FULL 2006->2026", "2006-01-01"), ("2015+ (gold-comparable)", "2015-01-01")):
    idx = oa.index.intersection(tn.index).intersection(cash.index)
    for s in sl.values():
        idx = idx.intersection(s.index)
    if wname.startswith("2015"):
        idx = idx.intersection(gold.index)
        idx = idx[idx >= wstart]
    else:
        idx = idx[idx >= wstart]
    oam, tnm, cam = monthly(oa, idx), monthly(tn, idx), monthly(cash, idx)
    gm = gold.reindex(idx).ffill().resample("ME").last().pct_change().fillna(0.0)
    print(f"\n{'='*100}\n{wname}   ({idx[0].date()} -> {idx[-1].date()})\n{'='*100}")
    print(f"{'sleeve':>14} {'w':>5} | {'blend CAGR med [min..max]':>28} | {'DD':>8} | "
          f"{'Calmar':>6} | {'2018':>7} {'2020':>7} {'22H1':>7}")

    def blendrow(tag, w, sm_cols, sm):
        cs, ds, w18, w20, w22 = [], [], [], [], []
        for oc in oam.columns:
            for tc in tnm.columns:
                for ic in sm_cols:
                    r = (1 - w) / 2 * (oam[oc] + tnm[tc]) + (w * sm[ic] if w else 0)
                    bl = (1 + r).cumprod()
                    c_, d_ = stats(bl)
                    cs.append(c_); ds.append(d_)
                    w18.append(wdd(bl, "2018-01-01", "2018-12-31"))
                    w20.append(wdd(bl, "2020-01-01", "2020-12-31"))
                    w22.append(wdd(bl, "2022-01-01", "2022-06-30"))
        cm, dm = float(np.median(cs)), float(np.median(ds))
        print(f"{tag:>14} {w*100:4.0f}% | {cm:8.2f} [{min(cs):7.2f}..{max(cs):7.2f}] | "
              f"{dm:7.2f}% | {cm/abs(dm):6.2f} | {np.median(w18):6.2f}% {np.median(w20):6.2f}% "
              f"{np.median(w22):6.2f}%")
        return cm, dm

    blendrow("BASELINE TN+OA", 0.0, ["cash"], cam)
    for tag, s in sl.items():
        sm = monthly(s, idx)
        for w in (0.10, 0.20, 0.33):
            blendrow(tag, w, list(sm.columns)[:10], sm)
    for w in (0.10, 0.20):
        blendrow("CASH-NULL", w, ["cash"], cam)
    if wname.startswith("2015"):
        gmf = pd.DataFrame({"gold": gm})
        for w in (0.10, 0.20):
            blendrow("GOLD (r/147)", w, ["gold"], gmf)
        # 4-sleeve: OA/TN 40/40 + gold 10 + IPO 10
        sm = monthly(sl["narrow(<=3m)"], idx)
        cs, ds = [], []
        for oc in oam.columns:
            for tc in tnm.columns:
                for ic in list(sm.columns)[:10]:
                    r = 0.40 * oam[oc] + 0.40 * tnm[tc] + 0.10 * gm + 0.10 * sm[ic]
                    bl = (1 + r).cumprod()
                    c_, d_ = stats(bl)
                    cs.append(c_); ds.append(d_)
        cm, dm = float(np.median(cs)), float(np.median(ds))
        print(f"{'4-SLEEVE 40/40/10g/10ipo':>24} | {cm:8.2f} [{min(cs):7.2f}..{max(cs):7.2f}] | "
              f"{dm:7.2f}% | Calmar {cm/abs(dm):6.2f}")

# ─────────────────────────────────────────────── capacity, done properly
print(f"\n{'='*100}\nCAPACITY — position as a share of the held name's 20-day median traded "
      f"value\n{'='*100}")
for tag, fn in (("narrow(<=3m)", "g3_trades_adopted.csv"), ("mid(<=6m)", "g3_trades_mid.csv"),
                ("wide(<=24m)", "g3_trades_wide.csv")):
    t = pd.read_csv(RES / fn)
    tv = t["tv"].dropna()
    print(f"\n{tag}: {len(tv)} trade-entries; held-name 20d median traded value: "
          f"p10 Rs {tv.quantile(.1)/1e7:.1f}cr  median Rs {tv.median()/1e7:.1f}cr  "
          f"p90 Rs {tv.quantile(.9)/1e7:.1f}cr")
    size_pct = 0.1875
    print(f"  sleeve sized at {size_pct*100:.2f}% of sleeve NAV per position:")
    print(f"  {'portfolio':>12} {'sleeve@10%':>12} {'position':>12} | "
          f"{'median %TV':>11} {'p90 %TV':>9} {'p99 %TV':>9}")
    for port in (1e6, 1e7, 5e7, 1e8, 5e8):
        sleeve = 0.10 * port
        pos = size_pct * sleeve
        f = 100 * pos / tv
        print(f"  Rs {port/1e7:9.1f}cr Rs {sleeve/1e7:9.2f}cr Rs {pos/1e5:8.2f}L | "
              f"{f.median():10.3f}% {f.quantile(.9):8.2f}% {f.quantile(.99):8.2f}%")
print("\n(rule of thumb: a position above ~10% of a name's daily traded value is not "
      "executable without material impact on a breakout day)")
print("\nDONE")
