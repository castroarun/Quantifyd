"""G2 — beta-adjusted hedge ratios (>1.0) + hybrid partial-derisk arms, vs the right baselines."""
import importlib.util, csv, sys
import pandas as pd
spec = importlib.util.spec_from_file_location("hs", "/home/arun/quantifyd/research/105_momentum_put_hedge/scripts/run_hedge_sweep.py")
hs = importlib.util.module_from_spec(spec); spec.loader.exec_module(hs)
OUT = "/home/arun/quantifyd/research/105_momentum_put_hedge/results/hedge_g2.csv"
st19 = pd.Timestamp("2019-02-01")
grid = []
# beta-adjusted ratios, monthly (full cycle 2011-2026)
for mny, mtag in ((0.02, "ITM2"), (0.0, "ATM")):
    for ratio in (1.5, 2.0, 2.5):
        grid.append((f"LP_{mtag}_monthly_r{ratio}", dict(mode="hedge", struct="long", moneyness=mny, tenor="monthly", ratio=ratio)))
# beta-adjusted ratios, weekly (2019+)
for ratio in (1.5, 2.0):
    grid.append((f"LP_ATM_weekly_r{ratio}", dict(mode="hedge", struct="long", moneyness=0.0, tenor="weekly", ratio=ratio, start=st19)))
    grid.append((f"SPR_ATM_w10_weekly_r{ratio}", dict(mode="hedge", struct="spread", moneyness=0.0, width=0.10, tenor="weekly", ratio=ratio, start=st19)))
# hybrid: partial de-risk + hedge the remainder (monthly, full cycle)
for df in (0.3, 0.5):
    for ratio in (1.0, 2.0):
        grid.append((f"HYB_d{int(df*100)}_ATM_monthly_r{ratio}", dict(mode="hedge", struct="long", moneyness=0.0, tenor="monthly", ratio=ratio, derisk_frac=df)))
        grid.append((f"HYB_d{int(df*100)}_ATM_weekly_r{ratio}", dict(mode="hedge", struct="long", moneyness=0.0, tenor="weekly", ratio=ratio, derisk_frac=df, start=st19)))
F = ["config","cagr","net_cagr","maxdd","sharpe","calmar","net_calmar","full_exits","stcg","prem","hpnl","hedges"]
w = csv.DictWriter(open(OUT,"w",newline=""), fieldnames=F); w.writeheader()
print(f"{'config':>30} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} {'Calmar':>6} {'netCal':>7} {'prem':>7}", flush=True)
for lbl, kw in grid:
    try: r = hs.run(**kw)
    except Exception as e: print(f"  {lbl} FAILED {e}", flush=True); continue
    w.writerow({k: (lbl if k=="config" else r[k]) for k in F}); 
    print(f"{lbl:>30} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% {r['calmar']:>6.2f} {r['net_calmar']:>7.2f} {r['prem']:>7.1f}", flush=True)
    sys.stdout.flush()
print("done ->", OUT, flush=True)
