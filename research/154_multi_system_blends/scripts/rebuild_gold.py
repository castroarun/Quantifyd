"""research/154 P0b - rebuild the gold-INR reference series at DAILY resolution.

WHY: research/147's cached `gold_inr_ref.csv` is a MONTHLY series that is missing
**40 of its 274 months** (14.6%). Two causes, both in the source JSONs it was built from:
  (a) Yahoo's *monthly* GC=F / INR=X candles themselves drop months (43 and 22 missing);
  (b) their epoch stamps carry a US/UTC offset, so bars like `2004-03-31 23:00` land in
      the WRONG month, collide with the real March bar, and `duplicated(keep='last')`
      deletes one - leaving a hole.
A sparse monthly series makes `pct_change()` silently span two months, which mis-states
every pre-2015 gold return. Fixed here by pulling the **daily** series instead (GC=F from
2000-08, INR=X from 2003-12), stamping months with a +12h offset so the timezone shift can
never move a bar across a month boundary, and aligning onto the Indian trading calendar.

Reference series only. It is written to results/, never into market_data.db, and every
figure derived from it is labelled.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/arun/quantifyd")
RES = ROOT / "research" / "154_multi_system_blends" / "results"
DBQ = ROOT / "backtest_data" / "market_data.db"


def fetch(sym, cache):
    f = RES / cache
    if not f.exists():
        u = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
             "?period1=946684800&period2=1788000000&interval=1d")
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        j = json.load(urllib.request.urlopen(req, timeout=60))
        r = j["chart"]["result"][0]
        rows = [[t, c] for t, c in zip(r["timestamp"], r["indicators"]["quote"][0]["close"])
                if c is not None]
        json.dump(rows, open(f, "w"))
    rows = json.load(open(f))
    s = pd.Series({pd.Timestamp(t, unit="s"): v for t, v in rows}).sort_index()
    # +12h absorbs the +-5h exchange/UTC offset before we date-truncate
    s.index = (s.index + pd.Timedelta(hours=12)).normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s


def main():
    import sqlite3
    xau = fetch("GC=F", "yahoo_gcf_daily.json")
    inr = fetch("INR=X", "yahoo_inrx_daily.json")
    print(f"GC=F  daily {xau.index[0].date()} -> {xau.index[-1].date()} n={len(xau)}")
    print(f"INR=X daily {inr.index[0].date()} -> {inr.index[-1].date()} n={len(inr)}")

    con = sqlite3.connect(str(DBQ))
    g = pd.read_sql_query("select date, close from market_data_unified where symbol="
                          "'GOLDBEES' and timeframe='day' order by date", con)
    cal = pd.read_sql_query("select distinct date from market_data_unified where "
                            "symbol='NIFTYBEES' and timeframe='day' order by date", con)
    con.close()
    g["date"] = pd.to_datetime(g["date"].str[:10])
    gb = g.drop_duplicates("date").set_index("date")["close"].astype(float).sort_index()
    cal = pd.to_datetime(cal["date"].str[:10]).drop_duplicates().sort_values()
    cal = pd.DatetimeIndex(cal)
    print(f"GOLDBEES real {gb.index[0].date()} -> {gb.index[-1].date()} n={len(gb)}")
    print(f"NSE calendar  {cal[0].date()} -> {cal[-1].date()} n={len(cal)}")

    ref = (xau.reindex(cal, method="ffill") * inr.reindex(cal, method="ffill")).dropna()
    ref.name = "gold_inr_ref"
    print(f"reconstruction (daily, NSE calendar) {ref.index[0].date()} -> "
          f"{ref.index[-1].date()} n={len(ref)}")
    pm = ref.index.to_period("M")
    full = pd.period_range(pm.min(), pm.max(), freq="M")
    missing = [str(x) for x in full if x not in set(pm)]
    print(f"missing months after the fix: {len(missing)} {missing}")

    # ---- validation against the real instrument over the overlap
    ov = ref.index.intersection(gb.index)
    dcorr = float(ref.loc[ov].pct_change().corr(gb.loc[ov].pct_change()))
    mr = ref.loc[ov].resample("ME").last().pct_change().dropna()
    mg = gb.loc[ov].resample("ME").last().pct_change().dropna()
    cm = mr.index.intersection(mg.index)
    mcorr = float(mr.loc[cm].corr(mg.loc[cm]))
    yrs = (ov[-1] - ov[0]).days / 365.25
    drift = ((gb.loc[ov].iloc[-1] / gb.loc[ov].iloc[0]) ** (1 / yrs)
             - (ref.loc[ov].iloc[-1] / ref.loc[ov].iloc[0]) ** (1 / yrs)) * 100
    print(f"VALIDATION on {len(ov)} overlapping days / {len(cm)} months: "
          f"daily corr {dcorr:.3f}, monthly corr {mcorr:.3f}, "
          f"annualised drift (GOLDBEES - reconstruction) {drift:+.2f}pp")

    # ---- chain: scale the reconstruction to meet GOLDBEES at its first real day
    first = gb.index[0]
    scale = float(gb.iloc[0]) / float(ref.loc[:first].iloc[-1])
    pre = ref[ref.index < first] * scale
    out = pd.concat([pre, gb.reindex(cal[cal >= first], method="ffill").dropna()])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    src = pd.Series(np.where(out.index < first, "reconstruction", "GOLDBEES"), index=out.index)
    pd.DataFrame({"close": out, "source": src}).to_csv(RES / "gold_nav.csv")
    print(f"GOLD chained written: {out.index[0].date()} -> {out.index[-1].date()} "
          f"n={len(out)} ({int((src=='reconstruction').sum())} reconstructed DAILY points)")
    (RES / "gold_validation.txt").write_text(
        f"source: Yahoo GC=F (COMEX front gold, USD/oz) x INR=X (USDINR), DAILY, aligned to "
        f"the NSE trading calendar\noverlap days {len(ov)} / months {len(cm)}\n"
        f"daily return corr  {dcorr:.4f}\nmonthly return corr {mcorr:.4f}\n"
        f"annualised drift (GOLDBEES minus reconstruction) {drift:+.3f}pp\n"
        f"real GOLDBEES from {first.date()}\n"
        f"months missing after fix: {len(missing)}\n"
        f"NOTE: research/147's cached monthly gold_inr_ref.csv was missing 40 of 274 months; "
        f"this rebuild replaces it for research/154.\n")


if __name__ == "__main__":
    main()
