"""
Extend the GTAA 3-asset book (Nifty+Gold+Nasdaq) back ~20y to stress-test pre-2015
(2008 GFC, 2011, 2013 taper). Factor data unavailable pre-2015 -> use Nifty as the
equity sleeve. Gold/Nasdaq pre-2015 via proxy: GLD x USDINR (gold), QQQ x USDINR (Nasdaq),
validated vs real GOLDBEES/MON100 on the 2015-26 overlap, then chain RETURNS (real ETF
post-2015, proxy pre-2015) into a continuous NAV.

Run on VPS: venv/bin/python research/63_gtaa_etf_rotation/scripts/longhist_test.py
"""
import time, sys
from pathlib import Path
import numpy as np, pandas as pd
from gtaa_engine import load_monthly_closes, _metrics


def yf_m(t, tries=4):
    import yfinance as yf
    for a in range(tries):
        try:
            d = yf.download(t, start="2003-06-01", progress=False, auto_adjust=True)
            if d is not None and len(d):
                s = d["Close"]; s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
                s.index = pd.to_datetime(s.index); return s.resample("ME").last()
        except Exception as e:
            print(f"  {t} try{a+1} {str(e)[:40]}")
        time.sleep(5)
    return None


print("Fetching proxies (GLD, QQQ, INR=X)...")
gld, qqq, inr = yf_m("GLD"), yf_m("QQQ"), yf_m("INR=X")
if any(x is None for x in (gld, qqq, inr)):
    raise SystemExit("Yahoo fetch failed (rate-limited) - retry later.")
gold_px = (gld * inr).dropna()      # gold in INR (proxy)
nas_px = (qqq * inr).dropna()       # nasdaq-100 in INR (proxy)

# real ETFs + NIFTYBEES from DB
real = load_monthly_closes(["NIFTYBEES", "GOLDBEES", "MON100"])
nifty = real["NIFTYBEES"].dropna()

# validate proxies on overlap
for nm, prox, realc in [("Gold", gold_px, real["GOLDBEES"].dropna()), ("Nasdaq", nas_px, real["MON100"].dropna())]:
    ov = pd.concat([prox, realc], axis=1).dropna(); rr = ov.pct_change().dropna()
    print(f"  {nm} proxy vs real: corr={rr.iloc[:,0].corr(rr.iloc[:,1]):.3f}  "
          f"proxyCAGR={_metrics(rr.iloc[:,0])['cagr']*100:.1f}% realCAGR={_metrics(rr.iloc[:,1])['cagr']*100:.1f}%")


def chain(real_s, prox_s):
    """Monthly returns: real where available, else proxy; chained to a NAV from 1.0."""
    rr = real_s.pct_change(); pr = prox_s.pct_change()
    r = rr.combine_first(pr).dropna()
    return (1 + r).cumprod()


gold_nav = chain(real["GOLDBEES"], gold_px)
nas_nav = chain(real["MON100"], nas_px)
nifty_nav = nifty / nifty.iloc[0]
panel = pd.concat([nifty_nav.rename("Nifty"), gold_nav.rename("Gold"), nas_nav.rename("Nasdaq")], axis=1)
panel = panel.loc[panel.dropna().index.min():]
r = panel.pct_change().dropna()
print(f"\nLong-history window: {panel.index.min():%Y-%m} -> {panel.index.max():%Y-%m} ({len(r)} mo, {len(r)/12:.1f}y)\n")

print("Per-asset (proxy-chained, INR):")
for c in panel.columns:
    m = _metrics(r[c]); print(f"  {c:7s} CAGR={m['cagr']*100:6.1f}% vol={r[c].std()*np.sqrt(12)*100:5.1f}% DD={m['max_drawdown']*100:6.1f}%")
print("\nCorrelation (full history):")
print(r.corr().round(2).to_string())


def book(assets, mode="equal", cost=20.0, lb=12):
    R = r[assets]; prev = pd.Series(0.0, index=assets); net = []; idx = R.index
    for i in range(len(idx) - 1):
        t1 = idx[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    s = pd.Series(dict(net)).sort_index(); return s, _metrics(s)


print("\n=== GTAA 3-asset (equal, monthly reb, net 20bps) vs Nifty-only, FULL history ===")
s_ew, m_ew = book(["Nifty", "Gold", "Nasdaq"], "equal")
s_iv, m_iv = book(["Nifty", "Gold", "Nasdaq"], "invvol")
m_nifty = _metrics(r["Nifty"])
print(f"  3-asset equal : CAGR={m_ew['cagr']*100:5.1f}% DD={m_ew['max_drawdown']*100:6.1f}% Calmar={m_ew['calmar']:.2f} Sharpe={m_ew['sharpe']:.2f}")
print(f"  3-asset invvol: CAGR={m_iv['cagr']*100:5.1f}% DD={m_iv['max_drawdown']*100:6.1f}% Calmar={m_iv['calmar']:.2f} Sharpe={m_iv['sharpe']:.2f}")
print(f"  Nifty only    : CAGR={m_nifty['cagr']*100:5.1f}% DD={m_nifty['max_drawdown']*100:6.1f}% Calmar={m_nifty['calmar']:.2f}")

# per-year + worst stress drawdowns
print("\nPer-year (3-asset equal vs Nifty):")
wr = s_ew; nr = r["Nifty"]
for y in sorted(set(wr.index.year)):
    w = (1+wr[wr.index.year == y]).prod()-1; n = (1+nr[nr.index.year == y]).prod()-1
    star = "  <-- stress" if y in (2008, 2011, 2013, 2015, 2020, 2022) else ""
    print(f"  {y}: 3-asset={w*100:7.1f}%  Nifty={n*100:7.1f}%{star}")

# drawdown around 2008
eq = (1+s_ew).cumprod(); dd = eq/eq.cummax()-1
neq = (1+nr).cumprod(); ndd = neq/neq.cummax()-1
print(f"\n2008-09 worst drawdown: 3-asset {dd.loc['2008':'2009'].min()*100:.1f}%  vs Nifty {ndd.loc['2008':'2009'].min()*100:.1f}%")
RES = Path(__file__).resolve().parent.parent / "results"
pd.DataFrame({"three_asset": (1+s_ew).cumprod(), "nifty": (1+nr).cumprod()}).to_csv(RES/"longhist_equity.csv")
print("saved results/longhist_equity.csv")
