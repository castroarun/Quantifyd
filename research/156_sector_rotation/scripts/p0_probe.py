"""r/156 Phase 0 - data reality + integrity probe for sector study. READ ONLY."""
import sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/arun/quantifyd")
DB = ROOT / "backtest_data" / "market_data.db"
con = sqlite3.connect(str(DB))

SECT_IDX = ["NIFTYAUTO","NIFTYIT","NIFTYENERGY","NIFTYFINSRV","NIFTYFMCG","NIFTYMETAL",
            "NIFTYPHARMA","NIFTYPSUBANK","NIFTYREALTY","BANKNIFTY","NIFTYMEDIA",
            "NIFTYINFRA","NIFTYCONSUMPTION","NIFTYCOMMODITIES","NIFTYPVTBANK"]
BENCH = ["NIFTY50","NIFTY500","NIFTYMIDCAP150","NIFTYSMLCAP250","NIFTYNEXT50","NIFTYBEES"]

print("="*80); print("A. SECTOR / BENCHMARK SERIES COVERAGE")
q = """SELECT symbol, COUNT(*) n, MIN(date) d0, MAX(date) d1,
       SUM(CASE WHEN volume=0 OR volume IS NULL THEN 1 ELSE 0 END) zvol,
       SUM(CASE WHEN open=high AND high=low AND low=close THEN 1 ELSE 0 END) flat
       FROM market_data_unified WHERE timeframe='day' AND symbol IN (%s)
       GROUP BY symbol ORDER BY symbol"""
allsym = SECT_IDX + BENCH
df = pd.read_sql(q % ",".join("?"*len(allsym)), con, params=allsym)
print(df.to_string(index=False))
missing = sorted(set(allsym) - set(df.symbol))
print("NOT IN DB:", missing)

print("="*80); print("B. GAP / PHANTOM-ROW CHECK vs NIFTY50 trading calendar (2015+)")
cal = pd.read_sql("SELECT date FROM market_data_unified WHERE timeframe='day' AND symbol='NIFTY50' AND date>='2015-01-01'", con).date
cal = pd.to_datetime(cal).sort_values()
for s in df.symbol:
    d = pd.read_sql("SELECT date,open,high,low,close,volume FROM market_data_unified WHERE timeframe='day' AND symbol=? AND date>='2015-01-01' ORDER BY date", con, params=[s])
    d['date']=pd.to_datetime(d.date)
    extra = len(set(d.date)-set(cal)); miss = len(set(cal)-set(d.date))
    r = d.set_index('date').close.pct_change()
    big = int((r.abs()>0.12).sum())
    # split-scale step detector: single-day move whose magnitude ~ 1/n or n
    step = int(((r<-0.40)|(r>0.60)).sum())
    print(f"{s:18s} n={len(d):5d} extra_days={extra:4d} missing_days={miss:5d} |ret|>12%={big:3d} step={step}")

print("="*80); print("C. INDUSTRY MAP from official CSVs")
BD = ROOT/"backtest_data"
frames=[]
for f in ["nifty200_official.csv","niftymidcap150_official.csv","niftysmallcap250_official.csv"]:
    t = pd.read_csv(BD/f); t['src']=f; frames.append(t)
m = pd.concat(frames).drop_duplicates(subset=['Symbol'])
m = m[m.Industry.notna() & (m.Industry!='Industry')]
print("symbols with industry:", len(m), " industries:", m.Industry.nunique())
print(m.Industry.value_counts().to_string())

print("="*80); print("D. DAILY DATA DEPTH FOR THOSE SYMBOLS")
syms = sorted(m.Symbol.unique())
cov = pd.read_sql("SELECT symbol, COUNT(*) n, MIN(date) d0, MAX(date) d1 FROM market_data_unified WHERE timeframe='day' AND symbol IN (%s) GROUP BY symbol" % ",".join("?"*len(syms)), con, params=syms)
print("in DB:", len(cov), "of", len(syms))
cov['d0']=pd.to_datetime(cov.d0)
for yr in [2005,2008,2011,2015,2018,2021]:
    print(f"  symbols with data from <= {yr}-01-01: {(cov.d0<=pd.Timestamp(f'{yr}-01-01')).sum()}")
print("  max date distribution:", cov.d1.value_counts().head(3).to_dict())
mm = m.set_index('Symbol').Industry
cov2 = cov.set_index('symbol').join(mm)
print("\n  per-industry symbol counts with data from <=2008 / <=2015 / total:")
g = cov2.groupby('Industry').apply(lambda x: pd.Series({
    'tot':len(x), 'from2008':(x.d0<=pd.Timestamp('2008-01-01')).sum(),
    'from2015':(x.d0<=pd.Timestamp('2015-01-01')).sum()}))
print(g.sort_values('tot',ascending=False).to_string())
con.close()
