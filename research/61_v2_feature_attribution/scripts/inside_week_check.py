"""Is inside-week independent of the daily-CPR skip, and does the compression family
add anything stacked? Reuse feature_screen build then probe. Read-only."""
import io, contextlib, runpy, sys
# re-exec the screen but capture its T dataframe by importing its globals
g = runpy.run_path("/tmp/feature_screen.py", run_name="__feat__") if False else None
# simpler: re-run the heavy build inline via exec of the screen up to T, then probe
import json, numpy as np, pandas as pd
ns = {}
src = open("/tmp/feature_screen.py").read().split("# build labelled table")[0]
exec(src, ns)
feats = ns["feats"]; TR = ns["TR"]; YEARS = ns["YEARS"]
data = []
for d, v, p in TR:
    F = feats(d)
    if F is None: continue
    F.update(_date=d, _vix=v, _pnl=p, _yr=d[:4]); data.append(F)
T = pd.DataFrame(data); B = T[T["_vix"] >= 13].copy()
def met(s):
    s = s.sort_values("_date"); eq=peak=mdd=0
    for p in s["_pnl"]:
        eq+=p; peak=max(peak,eq); mdd=min(mdd,eq-peak)
    yt=s.groupby("_yr")["_pnl"].sum()
    return f"n={len(s):>3} tot={s['_pnl'].sum():>9,.0f} Cal={(s['_pnl'].sum()/YEARS)/abs(mdd) if mdd else 0:>4.2f} DD={mdd:>9,.0f} green={int((yt>0).sum())}/{len(yt)} neg={sorted(yt[yt<0].index)}"

print("BASELINE VIX>=13:                 ", met(B))
print("\n-- inside-week by era --")
print("  inside-week 2019-22:", met(B[(B._yr<='2022')&(B.w_inside==1)]))
print("  inside-week 2023-26:", met(B[(B._yr>='2023')&(B.w_inside==1)]))
print("\n-- overlap: of inside-weeks, how many also have narrow daily CPR (<0.10)? --")
iw = B[B.w_inside==1]
print(f"  inside-weeks n={len(iw)}, of which dCPRw<0.10: {int((iw.dCPRw_pct<0.10).sum())}")
print("\n-- does inside-week still bleed AFTER the CPR survivors filter? (incrementality) --")
surv = B[B.dCPRw_pct>=0.10]
print("  CPR-survivors, NOT inside-week:", met(surv[surv.w_inside==0]))
print("  CPR-survivors, inside-week    :", met(surv[surv.w_inside==1]))
print("\n-- skip rules head to head --")
print("  skip dCPR<0.10            :", met(B[B.dCPRw_pct>=0.10]))
print("  skip inside-week          :", met(B[B.w_inside==0]))
print("  skip dCPR<0.10 OR inside  :", met(B[(B.dCPRw_pct>=0.10)&(B.w_inside==0)]))
print("\n-- negative-week coverage --")
neg = B[B._pnl<0]
print(f"  total negative weeks: {len(neg)}, summing {neg._pnl.sum():,.0f}")
print(f"  caught by dCPR<0.10 : {int((neg.dCPRw_pct<0.10).sum())} weeks, {neg[neg.dCPRw_pct<0.10]._pnl.sum():,.0f}")
print(f"  caught by inside-wk : {int((neg.w_inside==1).sum())} weeks, {neg[neg.w_inside==1]._pnl.sum():,.0f}")
print(f"  caught by EITHER    : {int(((neg.dCPRw_pct<0.10)|(neg.w_inside==1)).sum())} weeks, "
      f"{neg[(neg.dCPRw_pct<0.10)|(neg.w_inside==1)]._pnl.sum():,.0f}")
