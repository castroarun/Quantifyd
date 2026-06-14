"""
Which Indian equity INDEX is the best 'equity sleeve' in {index + Gold + Nasdaq}?
Tests caps (Nifty50/Midcap/Smallcap/Next50), sectors & thematics, BankNifty, Sensex, BSE500.
Sanity-checks every series (2 of 5 factor indices were corrupt) and rejects bad prints.
Real window 2015-26 (clean Gold/Nasdaq from DB). Monthly, net 20bps.

Run on VPS: venv/bin/python research/64_factor_index_rotation/scripts/indices_sleeve_test.py
"""
import datetime as dt, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "63_gtaa_etf_rotation" / "scripts"))
from gtaa_engine import load_monthly_closes, _metrics
from services.kite_service import get_kite
k = get_kite()

# NSE index tokens (+ BSE Sensex 265, BSE500 276745)
TOK = {
    "Nifty50": 256265, "Next50": 270857, "Nifty500": 268041,
    "Midcap150": 266249, "Midcap100": 256777, "Smallcap250": 267273, "Smallcap100": 267017,
    "BankNifty": 260105, "FinService": 257801, "PvtBank": 271113, "PSUBank": 262921,
    "IT": 259849, "Pharma": 262409, "Healthcare": 288265, "Auto": 263433, "FMCG": 261897,
    "Energy": 261641, "Metal": 263689, "Realty": 261129, "Infra": 261385, "Media": 263945,
    "Consumption": 257545, "Commodities": 257289, "PSE": 262665, "CPSE": 268297,
    "MNC": 262153, "ServSector": 263177,
    "Sensex": 265, "BSE500": 276745,
}


def daily(tok, seg_from=2013):
    out = []; s = dt.datetime(seg_from, 1, 1)
    while s < dt.datetime(2026, 6, 13):
        e = min(s + dt.timedelta(days=1400), dt.datetime(2026, 6, 13))
        try: out += k.historical_data(tok, s, e, "day")
        except: pass
        s = e + dt.timedelta(days=1)
    if not out: return None
    df = pd.DataFrame(out); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.drop_duplicates("date").set_index("date")["close"].astype(float)


monthly = {}
print("Fetching + sanity-checking indices:")
for nm, t in TOK.items():
    s = daily(t)
    if s is None or len(s) < 200:
        print(f"  {nm:12s} NO/short data"); continue
    dr = s.pct_change().dropna()
    vol = dr.std() * np.sqrt(252) * 100; mx = dr.abs().max() * 100
    corrupt = vol > 60 or mx > 35
    print(f"  {nm:12s} {s.index.min().date()} vol={vol:5.0f}% maxmove={mx:5.0f}% {'CORRUPT-skip' if corrupt else 'ok'}")
    if not corrupt:
        monthly[nm] = s.resample("ME").last()

idx = pd.DataFrame(monthly)
etf = load_monthly_closes(["GOLDBEES", "MON100"]).rename(columns={"GOLDBEES": "Gold", "MON100": "Nasdaq"})
panel = idx.join(etf, how="outer").sort_index()
start = panel[["Gold", "Nasdaq", "Nifty50"]].dropna().index.min()
rets = panel.loc[start:].pct_change()
W = f"{panel.loc[start:].index.min():%Y-%m}->{panel.index.max():%Y-%m}"
print(f"\nwindow {W} ({len(rets.dropna())} mo)\n")


def book(assets, mode="invvol", cost=20.0, lb=12):
    R = rets[assets].dropna()
    if len(R) < lb + 6: return None
    prev = pd.Series(0.0, index=assets); net = []; ix = R.index
    for i in range(len(ix) - 1):
        t1 = ix[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    return _metrics(pd.Series(dict(net)).sort_index())


rows = []
for nm in monthly:
    if nm not in rets.columns: continue
    corr = rets[nm].corr(rets["Nifty50"])
    me = book([nm, "Gold", "Nasdaq"], "equal"); mi = book([nm, "Gold", "Nasdaq"], "invvol")
    solo = _metrics(rets[nm].dropna())
    if mi is None: continue
    rows.append((nm, corr, solo["cagr"]*100, solo["max_drawdown"]*100,
                 me["calmar"], mi["calmar"], mi["cagr"]*100, mi["max_drawdown"]*100))

rows.sort(key=lambda x: -x[5])  # by invvol Calmar
print(f"{'sleeve':12s} {'corrNifty':>9s} {'soloCAGR':>8s} {'soloDD':>7s} | {'eqCal':>6s} {'ivCal':>6s} {'ivCAGR':>7s} {'ivDD':>6s}")
print("-"*78)
for nm, c, scg, sdd, ec, ic, icg, idd in rows:
    div = " <-DIVERSIFIER" if c < 0.6 else ""
    print(f"{nm:12s} {c:9.2f} {scg:8.1f} {sdd:7.1f} | {ec:6.2f} {ic:6.2f} {icg:7.1f} {idd:6.1f}{div}")

with open(RES / "indices_sleeve.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["sleeve", "corr_nifty", "solo_cagr", "solo_dd", "equal_calmar", "invvol_calmar", "invvol_cagr", "invvol_dd"])
    for r in rows: w.writerow([r[0], round(r[1], 3)] + [round(x, 2) for x in r[2:]])
print(f"\nwindow {W} ; wrote results/indices_sleeve.csv ; tested {len(rows)} clean indices")
