"""Wide weekly CPR -> actual NIFTY movement, FULL history via 5-min resampled to weekly (2015-2026)."""
import sqlite3, numpy as np, pandas as pd
c = sqlite3.connect("backtest_data/market_data.db")
# confirm the daily gap
dr = c.execute("SELECT MIN(date),MAX(date),COUNT(*) FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='day'").fetchone()
print(f"NIFTY50 'day' coverage: {dr[0]} -> {dr[1]} ({dr[2]} rows)  <- why the first run was short")
df5 = pd.read_sql("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date", c, parse_dates=["date"]).set_index("date")
print(f"NIFTY50 '5minute' coverage: {df5.index[0]} -> {df5.index[-1]} ({len(df5)} bars)\n")

wk = df5.resample("W-FRI").agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last")).dropna()
pH, pL, pC = wk["high"].shift(1), wk["low"].shift(1), wk["close"].shift(1)
wk["cpr_width_pct"] = (2*pC - pH - pL).abs() / 3 / pC * 100
wk["range_pct"] = (wk["high"] - wk["low"]) / pC * 100
wk["excursion_pct"] = np.maximum(wk["high"] - wk["open"], wk["open"] - wk["low"]) / wk["open"] * 100
wk["net_pct"] = (wk["close"] - wk["open"]) / wk["open"] * 100
wk = wk.dropna(subset=["cpr_width_pct"])

n = len(wk)
print(f"weekly bars with a prior-week CPR: {n}  ({wk.index[0].date()} -> {wk.index[-1].date()})")
print(f"corr(CPR width, intra-week range%)      = {wk['cpr_width_pct'].corr(wk['range_pct']):+.3f}")
print(f"corr(CPR width, max excursion from open) = {wk['cpr_width_pct'].corr(wk['excursion_pct']):+.3f}\n")
wk["q"] = pd.qcut(wk["cpr_width_pct"], 5, labels=["Q1 narrow","Q2","Q3","Q4","Q5 WIDE"])
print(f"{'CPR width bucket':14s} {'n':>4} {'wMean%':>7} {'wMax%':>7} | range% mean/med/max | excursion% mean/med/max")
for q, sub in wk.groupby("q", observed=True):
    print(f"{q:14s} {len(sub):>4} {sub['cpr_width_pct'].mean():>7.2f} {sub['cpr_width_pct'].max():>7.2f} | "
          f"{sub['range_pct'].mean():>5.2f}/{sub['range_pct'].median():>5.2f}/{sub['range_pct'].max():>6.2f} | "
          f"{sub['excursion_pct'].mean():>5.2f}/{sub['excursion_pct'].median():>5.2f}/{sub['excursion_pct'].max():>6.2f}")
wide = wk[wk["q"] == "Q5 WIDE"]
print(f"\n=== WIDE weekly CPR (top quintile, n={len(wide)}, width >= {wide['cpr_width_pct'].min():.2f}%) ===")
print(f"intra-week range%:  mean {wide['range_pct'].mean():.2f}  median {wide['range_pct'].median():.2f}  "
      f"p90 {wide['range_pct'].quantile(.9):.2f}  max {wide['range_pct'].max():.2f}  min {wide['range_pct'].min():.2f}")
for p in (2,3,5,7):
    print(f"  weeks with range > {p}%: {(wide['range_pct']>p).sum():>3} / {len(wide)} ({(wide['range_pct']>p).mean()*100:.0f}%)")
print("\nTop 15 widest-CPR weeks -> what NIFTY actually did that week:")
print(f"  {'week-end':10s} {'CPR w%':>7} {'range%':>7} {'net%':>7}")
for d, r in wide.sort_values("cpr_width_pct", ascending=False).head(15).iterrows():
    print(f"  {str(d.date()):10s} {r['cpr_width_pct']:>7.2f} {r['range_pct']:>7.2f} {r['net_pct']:>+7.2f}")
# also: of the biggest actual-movement weeks, how wide was their CPR? (reverse view)
print("\nReverse view — the 12 BIGGEST actual-range weeks, and how wide their CPR was:")
print(f"  {'week-end':10s} {'range%':>7} {'CPR w%':>7} {'CPR pctile':>10}")
wk["cpr_pctile"] = wk["cpr_width_pct"].rank(pct=True)*100
for d, r in wk.sort_values("range_pct", ascending=False).head(12).iterrows():
    print(f"  {str(d.date()):10s} {r['range_pct']:>7.2f} {r['cpr_width_pct']:>7.2f} {r['cpr_pctile']:>9.0f}%")
