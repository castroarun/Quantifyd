"""research/72 pass 2 — three questions on the intraday decay / 15:15 IV-spike:
 (a) EXIT-MINUTE sweep: for a short ATM straddle entered 09:16, is buying back at 15:10 vs 15:15
     vs 15:20 vs 15:25 vs 15:30 cheaper? Uses the FIXED entry strike (what a held position feels),
     reports mean P&L + downside (5th pctile) by exit minute — theta vs late-move risk.
 (b) CPR / DAY-TYPE concurrency: does the decay + the 15:15 IV pop differ on narrow-vs-wide CPR days
     and trend-vs-range days? (CPR from prior session H/L/C; day-type from realised OHLC.)
 (c) POP trade: buy the ATM straddle ~15:10, sell ~15:15 — net of theta, by DTE. Tradeable or not?
Read-only on options_data.db. Reuses pass-1 atm_minute_paths.csv for the direction-neutral ATM path."""
import sqlite3, datetime as dt
import numpy as np, pandas as pd
from pathlib import Path

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = Path("/home/arun/quantifyd/research/74_intraday_premium_decay/results")
con = sqlite3.connect(DB)
STEP = 50
days = pd.read_sql("SELECT DISTINCT date(snapshot_time) d FROM option_chain ORDER BY d", con)["d"].tolist()

meta, exitrows = [], []
for d in days:
    exps = pd.read_sql(f"SELECT DISTINCT expiry_date e FROM option_chain "
                       f"WHERE snapshot_time BETWEEN '{d}T09:00' AND '{d}T15:31'", con)["e"].dropna().tolist()
    exps = sorted(exps)
    if not exps:
        continue
    front = next((e for e in exps if e[:10] >= d), exps[0])
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    en = pd.read_sql(f"SELECT strike,instrument_type,ltp,underlying_spot FROM option_chain "
                     f"WHERE snapshot_time BETWEEN '{d}T09:16:00' AND '{d}T09:20:00' AND expiry_date='{front}' "
                     f"AND instrument_type IN ('CE','PE')", con)
    en = en[(en.underlying_spot > 20000) & (en.underlying_spot < 28000)]
    if en.empty:
        continue
    es = int(round(en.underlying_spot.iloc[0] / STEP) * STEP)
    eat = en[en.strike.round().astype(int) == es]
    ce_e, pe_e = eat[eat.instrument_type == 'CE'].ltp.mean(), eat[eat.instrument_type == 'PE'].ltp.mean()
    entry_prem = ce_e + pe_e if (ce_e == ce_e and pe_e == pe_e) else np.nan
    ex = pd.read_sql(f"SELECT snapshot_time,instrument_type,ltp FROM option_chain "
                     f"WHERE snapshot_time BETWEEN '{d}T14:55:00' AND '{d}T15:31:00' AND expiry_date='{front}' "
                     f"AND CAST(ROUND(strike) AS INT)={es} AND instrument_type IN ('CE','PE')", con)
    if not ex.empty:
        ex['min'] = pd.to_datetime(ex.snapshot_time).dt.strftime('%H:%M')
        g = ex.groupby(['min', 'instrument_type']).ltp.mean().unstack()
        if 'CE' in g.columns and 'PE' in g.columns:
            for m, row in g.iterrows():
                if row.get('CE') == row.get('CE') and row.get('PE') == row.get('PE'):
                    exitrows.append(dict(day=d, dte=dte, min=m, exit_prem=row['CE'] + row['PE']))
    sp = pd.read_sql(f"SELECT spot_price FROM underlying_spot WHERE date(snapshot_time)='{d}' "
                     f"AND spot_price BETWEEN 20000 AND 28000 ORDER BY snapshot_time", con)["spot_price"]
    if sp.empty:
        continue
    meta.append(dict(day=d, dte=dte, entry_strike=es, entry_prem=entry_prem,
                     d_open=sp.iloc[0], d_high=sp.max(), d_low=sp.min(), d_close=sp.iloc[-1]))

M = pd.DataFrame(meta).sort_values("day").reset_index(drop=True)
EX = pd.DataFrame(exitrows)
# prior-session H/L/C -> CPR width; realised day-type
M[["pH", "pL", "pC"]] = M[["d_high", "d_low", "d_close"]].shift(1)
piv = (M.pH + M.pL + M.pC) / 3
bc = (M.pH + M.pL) / 2
M["cpr_w"] = (2 * piv - bc - bc).abs() / piv * 100
M["range_pct"] = (M.d_high - M.d_low) / M.d_open * 100
M["trend_ratio"] = (M.d_close - M.d_open).abs() / (M.d_high - M.d_low).clip(lower=1)
M.to_csv(OUT / "pass2_day_meta.csv", index=False)

print("=" * 72)
print("(a) EXIT-MINUTE SWEEP — short ATM straddle (entered 09:16), buyback at minute M")
print("    P&L = entry_prem - exit_prem, per lot=65.  n = clean days")
EX = EX.merge(M[["day", "entry_prem"]], on="day", how="left")
EX["pnl"] = (EX.entry_prem - EX.exit_prem) * 65
for lbl, sub in [("ALL", EX), ("DTE 0-1", EX[EX.dte <= 1]), ("DTE 2+", EX[EX.dte >= 2])]:
    print(f"\n  --- {lbl} ---")
    print(f"    {'exit':6}{'meanP&L':>9}{'medP&L':>9}{'p5(worst)':>10}{'std':>8}{'n':>4}")
    a = sub[sub["min"].isin(["15:00", "15:05", "15:10", "15:15", "15:20", "15:25", "15:29", "15:30"])]
    for m, gg in a.groupby("min"):
        print(f"    {m:6}{gg.pnl.mean():9.0f}{gg.pnl.median():9.0f}{gg.pnl.quantile(.05):10.0f}{gg.pnl.std():8.0f}{len(gg):4}")

print("\n" + "=" * 72)
print("(b) CPR / DAY-TYPE concurrency — does the decay + 15:15 IV pop change by day-type?")
R = pd.read_csv(OUT / "atm_minute_paths.csv", dtype={"min": str})
R = R[(R.spot > 20000) & (R.spot < 28000) & (R.prem > 0)].copy()
ref = R[R["min"] == "14:00"].groupby("day").agg(pr=("prem", "first"), ir=("iv", "first"))
R = R.merge(ref, on="day", how="left")
R["prem_idx"] = R.prem / R.pr * 100
cw_med = M.cpr_w.median()
tags = {"CPR-narrow": set(M[M.cpr_w < cw_med].day), "CPR-wide": set(M[M.cpr_w >= cw_med].day),
        "trend-day(>0.55)": set(M[M.trend_ratio > 0.55].day), "range-day(<0.35)": set(M[M.trend_ratio < 0.35].day)}
KEY = ["14:00", "15:00", "15:10", "15:15", "15:20", "15:30"]
print("  (cpr_w median = %.3f%%)" % cw_med)
for lbl, dset in tags.items():
    s = R[R.day.isin(dset)]
    if s.empty:
        print("  %-18s n=0 (no matching days)" % lbl)
        continue
    a = s.groupby("min").agg(prem_idx=("prem_idx", "mean"), iv=("iv", "mean"), n=("day", "nunique"))
    tw = a[(a.index >= "14:30") & (a.index <= "15:30")]
    iv10 = a.loc["15:10", "iv"] if "15:10" in a.index else np.nan
    iv15 = a.loc["15:15", "iv"] if "15:15" in a.index else np.nan
    row = "  %-18s n=%2d | " % (lbl, s.day.nunique()) + "  ".join(
        "%s:%.0f/%.1fIV" % (m, a.loc[m, "prem_idx"], a.loc[m, "iv"]) for m in KEY if m in a.index)
    print(row)
    if not tw.empty:
        print("      trough prem_idx %.0f @%s | IV 15:10->15:15 = %+.2f" %
              (tw.prem_idx.min(), tw.prem_idx.idxmin(), (iv15 - iv10)))

print("\n" + "=" * 72)
print("(c) POP TRADE — buy ATM straddle @15:10, sell @15:15 (peak); net of theta, by DTE")
piv2 = R.pivot_table(index="day", columns="min", values="prem", aggfunc="mean")
ivp = R.pivot_table(index="day", columns="min", values="iv", aggfunc="mean")
for lbl, dd2 in [("ALL", M.day), ("DTE 0-1", M[M.dte <= 1].day), ("DTE 2+", M[M.dte >= 2].day)]:
    idx = [d for d in dd2 if d in piv2.index and "15:10" in piv2.columns and "15:15" in piv2.columns]
    p = piv2.loc[idx]
    if "15:10" not in p.columns or "15:15" not in p.columns:
        continue
    chg = (p["15:15"] - p["15:10"]).dropna() * 65
    ivc = (ivp.loc[idx, "15:15"] - ivp.loc[idx, "15:10"]).dropna()
    print(f"  {lbl:9} straddle Δ(15:10->15:15) mean Rs{chg.mean():+.0f}  median Rs{chg.median():+.0f}  "
          f"win%%={100*(chg>0).mean():.0f}  IVΔ{ivc.mean():+.2f}  n={len(chg)}")
print("\nwrote pass2_day_meta.csv")
