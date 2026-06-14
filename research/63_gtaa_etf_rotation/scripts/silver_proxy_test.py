"""
Extend SILVERBEES to the full GTAA window via a silver PROXY (no Indian silver ETF
existed pre-2022). Proxy = international silver (SLV, USD) x USDINR -> INR silver.
Validate the proxy vs real SILVERBEES on the 2022+ overlap, then run the 4-asset
equal-weight / inverse-vol book over the same window as the 3-ETF study.

Run on VPS: venv/bin/python research/63_gtaa_etf_rotation/scripts/silver_proxy_test.py
"""
import time
import numpy as np, pandas as pd
from gtaa_engine import load_monthly_closes, _metrics


def yf_monthly(ticker, tries=4):
    import yfinance as yf
    for a in range(tries):
        try:
            df = yf.download(ticker, start="2014-06-01", progress=False, auto_adjust=True)
            if df is not None and len(df):
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s.index = pd.to_datetime(s.index)
                return s.resample("ME").last()
        except Exception as e:
            print(f"  {ticker} try{a+1} err {str(e)[:50]}")
        time.sleep(5)
    return None


print("Fetching silver (SLV) + USDINR (INR=X) from Yahoo...")
slv = yf_monthly("SLV")          # iShares Silver Trust, USD
fx = yf_monthly("INR=X")          # USDINR
if slv is None or fx is None:
    # fallback to silver futures
    slv = slv or yf_monthly("SI=F")
    fx = fx or yf_monthly("USDINR=X")
if slv is None or fx is None:
    raise SystemExit("Could not fetch proxy inputs (Yahoo rate-limited). Retry later.")

silver_inr = (slv * fx).dropna()
silver_inr.name = "SilverProxy"
print(f"Silver proxy: {silver_inr.index.min().date()} -> {silver_inr.index.max().date()} ({len(silver_inr)} months)")

# real SILVERBEES for validation
real = load_monthly_closes(["SILVERBEES"])["SILVERBEES"].dropna()

# ---- validate proxy vs real on overlap ----
ov = pd.concat([silver_inr, real.rename("SILVERBEES")], axis=1).dropna()
rr = ov.pct_change().dropna()
corr = rr["SilverProxy"].corr(rr["SILVERBEES"])
print(f"\nVALIDATION (overlap {ov.index.min().date()}->{ov.index.max().date()}, {len(rr)} mo):")
print(f"  monthly-return corr proxy vs SILVERBEES = {corr:.3f}")
print(f"  proxy CAGR={_metrics(rr['SilverProxy'])['cagr']*100:.1f}% vs SILVERBEES {_metrics(rr['SILVERBEES'])['cagr']*100:.1f}%")

# ---- build 4-asset panel over the 3-ETF window ----
etf = load_monthly_closes(["NIFTYBEES", "GOLDBEES", "MON100"]).rename(
    columns={"GOLDBEES": "Gold", "MON100": "Nasdaq", "NIFTYBEES": "Nifty"})
# splice: use real SILVERBEES where available, else proxy (scaled to align at first real point)
spliced = silver_inr.copy()
if len(ov):
    scale = real.reindex(ov.index).iloc[0] / silver_inr.reindex(ov.index).iloc[0]
    spliced = silver_inr * scale
    spliced.loc[real.index] = real  # prefer real where it exists
panel = etf.join(spliced.rename("Silver"), how="outer").sort_index()
p = panel.loc[panel[["Nifty", "Gold", "Nasdaq", "Silver"]].dropna().index.min():]
r = p.pct_change().dropna()
print(f"\n4-asset window (with silver PROXY pre-2022): {p.index.min().date()} -> {p.index.max().date()} ({len(r)} mo, {len(r)/12:.1f}y)")

print("\nPer-asset:")
for c in ["Nifty", "Gold", "Nasdaq", "Silver"]:
    m = _metrics(r[c]); print(f"  {c:7s} CAGR={m['cagr']*100:6.1f}% vol={r[c].std()*np.sqrt(12)*100:5.1f}% DD={m['max_drawdown']*100:6.1f}%")
print("\nCorrelation:")
print(r[["Nifty", "Gold", "Nasdaq", "Silver"]].corr().round(2).to_string())


def book(rets, assets, mode="equal", cost=20.0, lb=12):
    R = rets[assets]; prev = pd.Series(0.0, index=assets); net = []; idx = R.index
    for i in range(len(idx) - 1):
        t1 = idx[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    return _metrics(pd.Series(dict(net)).sort_index())


print("\n=== GTAA book over full window (silver = proxy pre-2022, real post-2022), net 20bps ===")
for mode in ("equal", "invvol"):
    print(f"-- {mode} --")
    for label, a in [("3-asset (Nifty/Gold/Nasdaq)", ["Nifty", "Gold", "Nasdaq"]),
                     ("4-asset (+Silver)", ["Nifty", "Gold", "Nasdaq", "Silver"])]:
        m = book(r, a, mode)
        print(f"  {label:32s} CAGR={m['cagr']*100:5.1f}% DD={m['max_drawdown']*100:6.1f}% Calmar={m['calmar']:.2f} Sharpe={m['sharpe']:.2f}")
