"""
research/64 G1 probe: Nifty FACTOR-INDEX coverage + cross-correlation.

Decisive question (kill-cheap): are the factors uncorrelated enough that
equal-weight diversification helps (as in research/63's asset-class trio), or are
they all just Indian-equity beta (corr > 0.8) -> then SELECTION, not diversification,
is the only lever.

Run on VPS: venv/bin/python research/64_factor_index_rotation/scripts/g1_probe.py
"""
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from services.kite_service import get_kite

# index instrument tokens (NSE INDEX series; longer history than the ETFs)
FACTORS = {
    "Momentum":  290057,   # NIFTY200 MOMENTUM 30
    "Quality":   272393,   # NIFTY100 QUALITY 30
    "Value":     267529,   # NIFTY50 VALUE 20
    "LowVol":    272137,   # NIFTY100 LOWVOL 30
    "Alpha":     265993,   # NIFTY ALPHA 50
    "Nifty50":   256265,   # NIFTY 50 (benchmark / beta reference)
}
FROM = dt.datetime(2010, 1, 1)
TO = dt.datetime(2026, 6, 13)


def fetch_daily(k, token):
    """Chunked daily fetch (Kite day limit ~2000 candles/call)."""
    out = []
    start = FROM
    while start < TO:
        end = min(start + dt.timedelta(days=1400), TO)
        try:
            rows = k.historical_data(token, start, end, "day")
            out += rows
        except Exception as e:
            print(f"   chunk {start.date()} ERR {str(e)[:50]}")
        start = end + dt.timedelta(days=1)
    if not out:
        return None
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.drop_duplicates("date").set_index("date")["close"].astype(float)


def main():
    k = get_kite()
    closes = {}
    print("Coverage:")
    for name, tok in FACTORS.items():
        s = fetch_daily(k, tok)
        if s is None:
            print(f"  {name:9s} NO DATA"); continue
        closes[name] = s
        print(f"  {name:9s} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")

    panel = pd.DataFrame(closes).sort_index()
    monthly = panel.resample("ME").last()
    common = monthly.dropna()
    print(f"\nCommon monthly window: {common.index.min().date()} -> {common.index.max().date()} ({len(common)} months)")

    rets = common.pct_change().dropna()
    yrs = len(rets) / 12

    print("\nPer-factor (PRICE-return index, not TR):")
    for c in common.columns:
        mult = (1 + rets[c]).prod()
        eq = (1 + rets[c]).cumprod(); dd = (eq / eq.cummax() - 1).min()
        print(f"  {c:9s} CAGR={mult**(1/yrs)-1:6.2%}  vol={rets[c].std()*np.sqrt(12):5.1%}  DD={dd:6.1%}")

    facs = [c for c in common.columns if c != "Nifty50"]
    print("\nFACTOR CROSS-CORRELATION (monthly returns):")
    print(rets[facs].corr().round(2).to_string())
    cc = rets[facs].corr()
    off = cc.values[np.triu_indices_from(cc.values, 1)]
    print(f"\n  mean off-diagonal corr = {off.mean():.2f}  (range {off.min():.2f}..{off.max():.2f})")
    print(f"  corr of each factor to Nifty50:")
    for c in facs:
        print(f"    {c:9s} {rets[c].corr(rets['Nifty50']):.2f}")

    # quick read: equal-weight factors vs Nifty50 vs best single factor
    ew = rets[facs].mean(axis=1)
    ew_mult = (1 + ew).prod(); ew_eq = (1 + ew).cumprod(); ew_dd = (ew_eq/ew_eq.cummax()-1).min()
    n_mult = (1 + rets["Nifty50"]).prod(); n_eq = (1+rets["Nifty50"]).cumprod(); n_dd=(n_eq/n_eq.cummax()-1).min()
    print(f"\nEqual-weight 5 factors: CAGR={ew_mult**(1/yrs)-1:.2%} DD={ew_dd:.1%} "
          f"Calmar={(ew_mult**(1/yrs)-1)/abs(ew_dd):.2f}")
    print(f"Nifty50 (beta ref):     CAGR={n_mult**(1/yrs)-1:.2%} DD={n_dd:.1%} "
          f"Calmar={(n_mult**(1/yrs)-1)/abs(n_dd):.2f}")

    RES = Path(__file__).resolve().parent.parent / "results"; RES.mkdir(parents=True, exist_ok=True)
    common.to_csv(RES / "factor_monthly_closes.csv")
    rets[facs].corr().round(3).to_csv(RES / "factor_corr.csv")
    print("\nsaved -> results/factor_monthly_closes.csv, factor_corr.csv")


if __name__ == "__main__":
    main()
