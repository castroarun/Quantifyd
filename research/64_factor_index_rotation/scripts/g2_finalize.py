"""
research/64 finalize: winner (Momentum+Gold+Nasdaq inverse-vol) NAV, per-year,
cost-sensitivity, tearsheet vs Nifty. Run on VPS.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results"
sys.path.insert(0, str(ROOT / "research" / "63_gtaa_etf_rotation" / "scripts"))
sys.path.insert(0, str(ROOT / "research" / "_utilities"))
from gtaa_engine import load_monthly_closes, _metrics

fac = pd.read_csv(RES / "factor_monthly_closes.csv", index_col=0, parse_dates=True)
etf = load_monthly_closes(["NIFTYBEES", "GOLDBEES", "MON100"]).rename(
    columns={"GOLDBEES": "Gold", "MON100": "Nasdaq", "NIFTYBEES": "NiftyETF"})
panel = fac.join(etf, how="outer").sort_index()
start16 = panel[["Gold", "Nasdaq"]].dropna().index.min()
rets = panel.loc[start16:].pct_change()


def invvol_nav(assets, cost_bps=20.0, lookback=12, mode="invvol"):
    R = rets[assets].dropna()
    idx = R.index
    prev = pd.Series(0.0, index=assets); net = []
    for i in range(lookback, len(idx) - 1):
        t, t1 = idx[i], idx[i + 1]
        if mode == "equal":
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R[assets].iloc[i-lookback+1:i+1].std(); iv = 1/vol.replace(0, np.nan)
            w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost_bps/1e4)))
        prev = w
    s = pd.Series(dict(net)).sort_index()
    return s, (1+s).cumprod()

WIN = ["Momentum", "Gold", "Nasdaq"]
win_ret, win_nav = invvol_nav(WIN, 20.0, mode="invvol")
nifty = panel.loc[win_nav.index[0]:, "NiftyETF"].dropna()
nifty_nav = (nifty/nifty.iloc[0]).reindex(win_nav.index).ffill()

m = _metrics(win_ret)
print(f"WINNER Mom+Gold+Nasdaq invvol: CAGR={m['cagr']*100:.1f}% DD={m['max_drawdown']*100:.1f}% "
      f"Calmar={m['calmar']:.2f} Sharpe={m['sharpe']:.2f} [{win_nav.index[0].date()}->{win_nav.index[-1].date()}]")

print("\nCost sensitivity (Calmar):")
for c in (0, 10, 20, 40):
    s, _ = invvol_nav(WIN, float(c), mode="invvol"); mm = _metrics(s)
    print(f"  {c}bps: Calmar={mm['calmar']:.2f} CAGR={mm['cagr']*100:.1f}%")

print("\nPer-year: winner vs Nifty")
wr = win_nav.pct_change().dropna(); nr = nifty_nav.pct_change().dropna()
yr_rows = []
for y in sorted(set(wr.index.year)):
    w = (1+wr[wr.index.year == y]).prod()-1; n = (1+nr[nr.index.year == y]).prod()-1
    yr_rows.append((y, round(w*100, 1), round(n*100, 1), round((w-n)*100, 1)))
    print(f"  {y}: win={w*100:6.1f}% nifty={n*100:6.1f}% excess={(w-n)*100:+6.1f}%")
pd.DataFrame(yr_rows, columns=["year", "winner", "nifty", "excess"]).to_csv(RES/"g2_yearly.csv", index=False)

try:
    from tearsheet import generate_tearsheet
    out = generate_tearsheet(win_nav, nifty_nav,
        "Factor GTAA — Momentum + Gold + Nasdaq (inverse-vol)",
        meta=dict(bench="NIFTYBEES (Nifty 50)", tag="BACKTEST 2016-2026 · net 20bps"),
        out_dir=str(RES))
    import shutil; shutil.copy(out if isinstance(out, str) else RES/"tearsheet.png", RES/"factor_gtaa.png")
    print("tearsheet ->", out)
except Exception as e:
    import traceback; traceback.print_exc()
pd.DataFrame({"winner": win_nav, "nifty": nifty_nav}).to_csv(RES/"g2_winner_equity.csv")
print("DONE")
