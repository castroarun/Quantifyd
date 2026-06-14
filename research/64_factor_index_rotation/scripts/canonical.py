"""SINGLE SOURCE OF TRUTH for every research/64 page number.
One data source (factor_monthly_closes.csv clean factors + Gold/Nasdaq/Nifty from DB),
one method (inverse-vol, lb=12, full 2015-26 window, net 20bps), one set of numbers.
Emits the full sleeve table + factor-only + winner per-year + regenerates the factsheet."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/arun/quantifyd"); RES = ROOT / "research/64_factor_index_rotation/results"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research/63_gtaa_etf_rotation/scripts")); sys.path.insert(0, str(ROOT / "research/_utilities"))
from gtaa_engine import load_monthly_closes, _metrics

fac = pd.read_csv(RES / "factor_monthly_closes.csv", index_col=0, parse_dates=True)  # Momentum/Value/Alpha clean
etf = load_monthly_closes(["NIFTYBEES", "GOLDBEES", "MON100"]).rename(columns={"NIFTYBEES": "Nifty", "GOLDBEES": "Gold", "MON100": "Nasdaq"})
panel = fac.join(etf, how="outer").sort_index()
start = panel[["Value", "Momentum", "Alpha", "Gold", "Nasdaq", "Nifty"]].dropna().index.min()
rets = panel.loc[start:].pct_change()
WIN = f"{rets.dropna().index.min():%Y-%m}..{rets.index.max():%Y-%m}"


def book(assets, mode="invvol", cost=20.0, lb=12):
    R = rets[assets].dropna(); prev = pd.Series(0.0, index=assets); net = []; ix = R.index
    for i in range(len(ix) - 1):
        t1 = ix[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    return pd.Series(dict(net)).sort_index()


def stats(s):
    m = _metrics(s); return dict(cagr=round(m["cagr"]*100, 1), dd=round(m["max_drawdown"]*100, 1),
                                 cal=round(m["calmar"], 2), sh=round(m["sharpe"], 2), tot=round(m["final_mult"], 1))


print(f"WINDOW {WIN}\n--- sleeve + Gold + Nasdaq (inverse-vol) ---")
res = {}
for s in ["Value", "Momentum", "Alpha", "Nifty"]:
    res[s] = stats(book([s, "Gold", "Nasdaq"], "invvol")); print(f"  {s:9s} {res[s]}")
print("--- references ---")
res["Nifty_EW"] = stats(book(["Nifty", "Gold", "Nasdaq"], "equal")); print(f"  Nifty+G+N equal   {res['Nifty_EW']}")
res["factoronly"] = stats(book(["Value", "Momentum", "Alpha"], "invvol")); print(f"  factor-only(V+M+A) {res['factoronly']}")
res["Nifty_solo"] = stats(rets["Nifty"].dropna()); print(f"  Nifty solo         {res['Nifty_solo']}")

# winner = best Calmar among single-factor sleeves
winner = max(["Value", "Momentum", "Alpha", "Nifty"], key=lambda x: res[x]["cal"])
sw = book([winner, "Gold", "Nasdaq"], "invvol"); nav = (1+sw).cumprod()
nb = load_monthly_closes(["NIFTYBEES"])["NIFTYBEES"].dropna()
nifty_nav = (nb.reindex(nb.index.union(nav.index)).ffill().reindex(nav.index)); nifty_nav = nifty_nav/nifty_nav.iloc[0]
nmet = _metrics(nifty_nav.pct_change().dropna())
print(f"\nWINNER = {winner}+Gold+Nasdaq  {res[winner]}  excess={res[winner]['cagr']-round(nmet['cagr']*100,1):+.1f}%/yr  NiftyCAGR={nmet['cagr']*100:.1f}%")
wr = nav.pct_change().dropna(); nr = nifty_nav.pct_change().dropna(); yr = []
for y in sorted(set(wr.index.year)):
    yr.append([y, round((1+wr[wr.index.year == y]).prod()*100-100, 1), round((1+nr[nr.index.year == y]).prod()*100-100, 1)])
print("PERYEAR", json.dumps([[a, b, c, round(b-c, 1)] for a, b, c in yr]))
from tearsheet import generate_tearsheet
generate_tearsheet(nav, nifty_nav, f"Factor GTAA — {winner} + Gold + Nasdaq (inverse-vol), {WIN.replace('..','–')}",
    meta=dict(bench="NIFTYBEES (Nifty 50)", tag=f"BACKTEST {WIN.replace('..','–')} (full window) · net 20bps"), out_dir=str(RES))
import shutil; shutil.copy(RES/"tearsheet.png", ROOT/"frontend/public/factor-gtaa-factsheet.png")
print(f"factsheet regenerated ({winner})")
