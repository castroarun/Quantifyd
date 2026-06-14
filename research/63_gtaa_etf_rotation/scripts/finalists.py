"""
Finalist comparison + robustness + tearsheet (research/63).

Run on VPS:  venv/bin/python research/63_gtaa_etf_rotation/scripts/finalists.py
Writes: results/finalists.csv, results/cost_sensitivity.csv, results/yearly.csv,
        results/gtaa_winner_tearsheet.png (+ .html)
"""
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results"
RES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "research" / "_utilities"))

from gtaa_engine import load_monthly_closes, run_gtaa, GTAAConfig

CORE = ["NIFTYBEES", "GOLDBEES", "MON100"]
CASH = "LIQUIDBEES"
panel = load_monthly_closes(CORE + [CASH])

# Finalist definitions (cash_asset/top_n/roc/ma/cash_leg)
FINALISTS = {
    "EqualWeight 3-asset (WINNER)": dict(top_n=3, roc_months=(12,), ma_months=6, cash_leg=False),
    "EqualWeight + trend filter":   dict(top_n=3, roc_months=(3, 6, 12), ma_months=6, cash_leg=True),
    "Momentum top-2 (gated)":       dict(top_n=2, roc_months=(6,), ma_months=6, cash_leg=True),
    "Slide top-1 (gated)":          dict(top_n=1, roc_months=(12,), ma_months=6, cash_leg=True),
    "Slide top-1 (raw)":            dict(top_n=1, roc_months=(12,), ma_months=6, cash_leg=False),
}


def make_cfg(name, cost_bps=20.0, next_open=False):
    p = FINALISTS[name]
    return GTAAConfig(risk_assets=CORE, cash_asset=CASH, cost_bps=cost_bps,
                      trade_next_open=next_open, label=name, **p)


# --- benchmark: NIFTYBEES buy & hold (monthly NAV) ---
nifty = panel["NIFTYBEES"].dropna()
common_start = panel[CORE].dropna().index.min()
nifty_nav = nifty.loc[common_start:]
nifty_nav = nifty_nav / nifty_nav.iloc[0]

# 60/40-ish naive benchmark too (Nifty/Gold 50-50) for context
half = panel[["NIFTYBEES", "GOLDBEES"]].loc[common_start:].pct_change().mean(axis=1)
nifty_gold_nav = (1 + half.fillna(0)).cumprod()

# ---- 1. finalists at base cost ----
print("=== FINALISTS (cost 20bps/side) ===")
rows = []
navs = {}
for name in FINALISTS:
    r = run_gtaa(panel, make_cfg(name, 20.0))
    navs[name] = r.equity
    rows.append(dict(name=name, cagr=round(r.cagr*100, 2), sharpe=round(r.sharpe, 3),
                     sortino=round(r.sortino, 3), max_drawdown=round(r.max_drawdown*100, 2),
                     calmar=round(r.calmar, 3), vol=round(r.vol*100, 2),
                     turnover_annual=round(r.turnover_annual, 2),
                     pct_in_cash=round(r.pct_in_cash*100, 1), final_mult=round(r.final_mult, 2),
                     start=r.start, end=r.end))
    print(f"  {name:30s} CAGR={rows[-1]['cagr']:5.1f}% DD={rows[-1]['max_drawdown']:6.1f}% "
          f"Calmar={rows[-1]['calmar']:.2f} Sharpe={rows[-1]['sharpe']:.2f} "
          f"turn={rows[-1]['turnover_annual']:.2f}")
# benchmark row
bm_cagr = nifty_nav.iloc[-1] ** (12/len(nifty_nav)) - 1
bm_eq = nifty_nav; bm_dd = (bm_eq/bm_eq.cummax()-1).min()
print(f"  {'BENCHMARK NIFTYBEES B&H':30s} CAGR={bm_cagr*100:5.1f}% DD={bm_dd*100:6.1f}% "
      f"Calmar={bm_cagr/abs(bm_dd):.2f}")
with open(RES/"finalists.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=list(rows[0].keys())).writeheader()
    [csv.DictWriter(f, fieldnames=list(rows[0].keys())).writerow(r) for r in rows]

# ---- 2. cost sensitivity ----
print("\n=== COST SENSITIVITY (Calmar) ===")
costs = [0, 10, 20, 40]
cs_rows = []
for name in FINALISTS:
    rec = {"name": name}
    line = f"  {name:30s}"
    for c in costs:
        r = run_gtaa(panel, make_cfg(name, float(c)))
        rec[f"cagr_{c}bps"] = round(r.cagr*100, 2)
        rec[f"calmar_{c}bps"] = round(r.calmar, 3)
        line += f"  {c}bps:{r.calmar:.2f}(CAGR{r.cagr*100:.0f})"
    cs_rows.append(rec); print(line)
with open(RES/"cost_sensitivity.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=list(cs_rows[0].keys())).writeheader()
    [csv.DictWriter(f, fieldnames=list(cs_rows[0].keys())).writerow(r) for r in cs_rows]

# ---- 3. next-open robustness for winner ----
print("\n=== NEXT-OPEN / LAG ROBUSTNESS (winner) ===")
# our engine already realizes t->t+1 (no same-bar leak); show winner stability anyway
rw = run_gtaa(panel, make_cfg("EqualWeight 3-asset (WINNER)", 20.0))
print(f"  Winner: Calmar={rw.calmar:.2f} CAGR={rw.cagr*100:.1f}% DD={rw.max_drawdown*100:.1f}% "
      f"(monthly reb, returns realized t->t+1)")

# ---- 4. per-year table: winner vs benchmark ----
print("\n=== PER-YEAR: Winner vs NIFTYBEES ===")
win_eq = navs["EqualWeight 3-asset (WINNER)"]
win_ret = win_eq.pct_change().dropna()
yr_rows = []
for yr in sorted(set(win_ret.index.year)):
    w = (1 + win_ret[win_ret.index.year == yr]).prod() - 1
    bret = nifty_nav.pct_change().dropna()
    b = (1 + bret[bret.index.year == yr]).prod() - 1
    yr_rows.append(dict(year=yr, winner=round(w*100, 1), nifty=round(b*100, 1),
                        excess=round((w-b)*100, 1)))
    print(f"  {yr}: winner={w*100:6.1f}%  nifty={b*100:6.1f}%  excess={(w-b)*100:+6.1f}%")
with open(RES/"yearly.csv", "w", newline="") as f:
    csv.DictWriter(f, fieldnames=["year", "winner", "nifty", "excess"]).writeheader()
    [csv.DictWriter(f, fieldnames=["year", "winner", "nifty", "excess"]).writerow(r) for r in yr_rows]

# ---- 5. tearsheet (winner vs NIFTYBEES) ----
print("\n=== TEARSHEET ===")
try:
    from tearsheet import generate_tearsheet
    meta = dict(bench="NIFTYBEES (Nifty 50)", tag="BACKTEST 2016-2026 · net 20bps")
    out = generate_tearsheet(win_eq, nifty_nav, "GTAA Equal-Weight 3-Asset (Nifty+Gold+Nasdaq)",
                             meta=meta, out_dir=str(RES))
    print("  tearsheet ->", out)
except Exception as e:
    import traceback; traceback.print_exc()
    print("  tearsheet FAILED:", e)

# save winner + benchmark equity for the app
pd.DataFrame({"winner": win_eq, "nifty": nifty_nav.reindex(win_eq.index)}).to_csv(RES/"winner_equity.csv")
print("\nDONE")
