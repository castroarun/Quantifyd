#!/usr/bin/env python3
"""
research/127 Phase E — G4 portfolio construction on C1.

Slot model: capital C in N_MAX slots. Each monthly cycle, take the liquid C1
entries ranked by ATM volume (known at entry, causal), up to N_MAX. Slot margin
M = C/N_MAX; position notional = M / margin_pct where
margin_pct = 1.25 x max_loss_pct + 2% exposure buffer (modeled SPAN for
defined-risk condor — stated assumption, not exchange-exact).
Rupee P&L per trade = net_pct x notional. Idle slots earn 6.5%/yr (liquid fund).
Monthly equity curve -> CAGR, MaxDD, Calmar, Sharpe, per-year, corr vs NIFTY.
"""
import math, sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "89_short_monthly_straddle" / "scripts"))
import engine as E

RESULTS = HERE.parent / "results"
COST, N_MAX, C0, RF = 0.005, 10, 10_000_000.0, 0.065

tr = pd.read_csv(RESULTS / "phase_b2_trades.csv")
tr = tr[(tr.config == "C1_E45X21W7K25_noSL") & (tr.atm_vol >= 100) & (tr.wing_vol_min >= 10)].copy()
tr["net_pct"] = tr["gross_pct"] - COST * tr["turnover_pct"]
tr["wingdist_pct"] = np.maximum(tr["Kc"] - tr["Ks_ce"], tr["Ks_pe"] - tr["Kp"]) / tr["S0"]
tr["maxloss_pct"] = (tr["wingdist_pct"] - tr["credit_pct"]).clip(lower=0.005)
import os
MARGIN_MULT = float(os.environ.get("MARGIN_MULT", "1.0"))
tr["margin_pct"] = (1.25 * tr["maxloss_pct"] + 0.02) * MARGIN_MULT
tr["entry_date"] = pd.to_datetime(tr["entry_date"]); tr["exit_date"] = pd.to_datetime(tr["exit_date"])
tr = tr.sort_values("entry_date")

# cycle = expiry month; entries cluster on the shared 45-DTE anchor date
tr["cycle"] = pd.to_datetime(tr["expiry"]).dt.to_period("M")

equity = C0
rows = []
for cyc, g in tr.groupby("cycle"):
    g = g.sort_values("atm_vol", ascending=False).head(N_MAX)
    slot_margin = equity / N_MAX
    pnl = 0.0
    for _, t in g.iterrows():
        notional = slot_margin / t["margin_pct"]
        pnl += t["net_pct"] * notional
    idle = N_MAX - len(g)
    hold_frac = g["hold_days"].mean() / 30.0 if len(g) else 0
    pnl += (idle / N_MAX) * equity * RF / 12.0 + (len(g) / N_MAX) * equity * RF / 12.0 * max(0.0, 1 - hold_frac)
    ret = pnl / equity
    equity *= (1 + ret)
    rows.append(dict(cycle=str(cyc), n_pos=len(g), ret=ret, equity=equity,
                     avg_margin_pct=g["margin_pct"].mean() if len(g) else np.nan))
eq = pd.DataFrame(rows)
eq["date"] = pd.PeriodIndex(eq["cycle"], freq="M").to_timestamp("M")

def perf(eqd, label):
    r = eqd["ret"].values
    yrs = len(r) / 12.0
    cagr = (eqd["equity"].iloc[-1] / (eqd["equity"].iloc[0] / (1 + r[0]))) ** (1 / yrs) - 1
    curve = eqd["equity"].values
    peak = np.maximum.accumulate(curve)
    dd = (curve / peak - 1).min()
    sharpe = (r.mean() * 12 - RF) / (r.std(ddof=1) * math.sqrt(12)) if r.std() > 0 else np.nan
    print(f"{label:22s} months={len(r)}  CAGR={cagr*100:.2f}%  MaxDD={dd*100:.1f}%  "
          f"Calmar={cagr/abs(dd):.2f}  Sharpe={sharpe:.2f}  worst-mo={r.min()*100:+.2f}%  "
          f"avg n_pos={eqd['n_pos'].mean():.1f}")
    return cagr, dd

print("="*104)
print(f"PHASE E — C1 portfolio, C0=Rs{C0/1e7:.0f}Cr, {N_MAX} slots, margin=1.25xMaxLoss+2%, idle @6.5%")
print("="*104)
perf(eq, "FULL 2016-2026")
dense = eq[eq.date >= "2021-01-01"]
perf(dense, "DENSE ERA 2021-2026")
print("\nper-year:")
for y, d in eq.groupby(eq.date.dt.year):
    yr_ret = (1 + d['ret']).prod() - 1
    print(f"  {y}: {yr_ret*100:+7.2f}%   (n_pos avg {d['n_pos'].mean():.1f}, months {len(d)})")

print(f"\navg margin_pct per position: {tr['margin_pct'].mean()*100:.1f}% of notional "
      f"(=> leverage ~{1/tr['margin_pct'].mean():.1f}x notional per slot-rupee)")
print(f"avg notional/slot at Rs{C0/N_MAX/1e5:.0f}L slot: Rs{C0/N_MAX/tr['margin_pct'].mean()/1e5:.0f}L")

# benchmark correlation (monthly NIFTY)
conn = sqlite3.connect(E.db_path())
nifty = E.load_daily("NIFTY50", conn)
if nifty.empty: nifty = E.load_daily("NIFTYBEES", conn)
nm = nifty["close"].resample("ME").last().pct_change()
mm = eq.set_index("date")["ret"]
joint = pd.concat([mm, nm], axis=1, keys=["strat", "nifty"]).dropna()
print(f"\ncorr(monthly, NIFTY) = {joint['strat'].corr(joint['nifty']):+.2f}  (n={len(joint)} months)")
neg = joint[joint["nifty"] < -0.03]
print(f"NIFTY down>3% months (n={len(neg)}): strategy avg {neg['strat'].mean()*100:+.2f}%/mo")
eq.to_csv(RESULTS / "phase_e_equity.csv", index=False)
print("\nsaved -> results/phase_e_equity.csv")
