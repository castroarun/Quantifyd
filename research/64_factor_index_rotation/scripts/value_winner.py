"""CANONICAL featured-winner computation: Value+Gold+Nasdaq inverse-vol, full 2015-26.
Mirrors replace_nifty_test/master_compare EXACTLY (same panel/start/book) so the number
matches the published master & replace-Nifty tables (Value 16.8% / -9.5% / 1.77). Emits
metrics + per-year + regenerates factor-gtaa-factsheet.png. SINGLE SOURCE OF TRUTH."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/arun/quantifyd")
RES = ROOT / "research/64_factor_index_rotation/results"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research/63_gtaa_etf_rotation/scripts")); sys.path.insert(0, str(ROOT / "research/_utilities"))
from gtaa_engine import load_monthly_closes, _metrics

FACTORS = ["Momentum", "Quality", "Value", "LowVol", "Alpha"]   # same as replace_nifty (sets the start)
fac = pd.read_csv(RES / "factor_monthly_closes.csv", index_col=0, parse_dates=True)
etf = load_monthly_closes(["GOLDBEES", "MON100"]).rename(columns={"GOLDBEES": "Gold", "MON100": "Nasdaq"})
panel = fac.join(etf, how="outer").sort_index()
start = panel[FACTORS + ["Gold", "Nasdaq"]].dropna().index.min()   # identical to replace_nifty_test
rets = panel.loc[start:].pct_change()


def book(assets, mode="invvol", cost=20.0, lb=12):   # identical to replace_nifty_test.book
    R = rets[assets].dropna(); prev = pd.Series(0.0, index=assets); net = []; idx = R.index
    for i in range(len(idx) - 1):
        t1 = idx[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    return pd.Series(dict(net)).sort_index()


s = book(["Value", "Gold", "Nasdaq"], "invvol"); m = _metrics(s); nav = (1+s).cumprod()
nb = load_monthly_closes(["NIFTYBEES"])["NIFTYBEES"].dropna()
nifty = nb.reindex(nb.index.union(nav.index)).ffill().reindex(nav.index); nifty_nav = nifty/nifty.iloc[0]
nm = _metrics(nifty_nav.pct_change().dropna())
print(f"CANONICAL Value+Gold+Nasdaq invvol [{nav.index[0].date()}->{nav.index[-1].date()}, {len(s)} mo]:")
print(f"  CAGR={m['cagr']*100:.1f}% DD={m['max_drawdown']*100:.1f}% Calmar={m['calmar']:.2f} Sharpe={m['sharpe']:.2f} TotRet={m['final_mult']:.1f}x")
print(f"  Nifty CAGR={nm['cagr']*100:.1f}%  excess={(m['cagr']-nm['cagr'])*100:+.1f}%/yr")
wr = nav.pct_change().dropna(); nr = nifty_nav.pct_change().dropna(); yr = []
for y in sorted(set(wr.index.year)):
    w = (1+wr[wr.index.year == y]).prod()-1; n = (1+nr[nr.index.year == y]).prod()-1
    yr.append([y, round(w*100, 1), round(n*100, 1), round((w-n)*100, 1)])
print("PERYEAR", json.dumps(yr))
try:
    from tearsheet import generate_tearsheet
    generate_tearsheet(nav, nifty_nav, "Factor GTAA — Value + Gold + Nasdaq (inverse-vol), 2015-2026",
        meta=dict(bench="NIFTYBEES (Nifty 50)", tag="BACKTEST 2015-2026 (full window) · net 20bps"), out_dir=str(RES))
    import shutil; shutil.copy(RES/"tearsheet.png", ROOT/"frontend/public/factor-gtaa-factsheet.png")
    print("factsheet regenerated (Value, full 2015-26)")
except Exception as e:
    import traceback; traceback.print_exc()
