"""
research/64 G2 sweep — factor rotation vs equal-weight vs risk-parity vs combined books.

Three families, all monthly, net of cost, vs Nifty 50 + the research/63 EW asset book:
  (a) FACTOR ROTATION  — top-N factor-momentum (reuse research/63 gtaa_engine)
  (b) FACTOR BASKETS   — equal-weight & inverse-vol (risk-parity) of factors
  (c) COMBINED BOOKS   — best factor(s) + Low-Vol + research/63 gold/Nasdaq

Data: factor INDEX monthly closes (2010-26, price-return) from the G1 probe CSV;
ETF monthly closes (Gold/Nasdaq/Nifty/cash) from market_data.db (2015-26).
Synthetic CASH = 6%/yr (fixes the LIQUIDBEES price-return≈0 caveat from research/63).

Run on VPS: venv/bin/python research/64_factor_index_rotation/scripts/g2_sweep.py
"""
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results"
RES.mkdir(exist_ok=True)
# reuse research/63 engine
sys.path.insert(0, str(ROOT / "research" / "63_gtaa_etf_rotation" / "scripts"))
from gtaa_engine import load_monthly_closes, run_gtaa, GTAAConfig, _metrics

COST = 20.0
FACTORS = ["Momentum", "Quality", "Value", "LowVol", "Alpha"]

# ---- build the master monthly panel ----
fac = pd.read_csv(RES / "factor_monthly_closes.csv", index_col=0, parse_dates=True)
etf = load_monthly_closes(["NIFTYBEES", "GOLDBEES", "MON100"])
etf = etf.rename(columns={"GOLDBEES": "Gold", "MON100": "Nasdaq", "NIFTYBEES": "NiftyETF"})
panel = fac.join(etf, how="outer").sort_index()
# synthetic cash compounding at 6%/yr (0.4868%/mo)
cash_m = (1 + 0.06) ** (1/12) - 1
panel["CASH"] = (1 + cash_m) ** np.arange(len(panel))
PANEL_FULL = panel.copy()


def restrict(p, start):
    return p.loc[start:] if start else p


def basket_metrics(rets, assets, mode, cost_bps=COST, lookback=12, label=""):
    """Monthly-rebalanced basket. mode='equal' or 'invvol'. Causal weights (data<=t),
    realize t->t+1. Returns metrics dict."""
    R = rets[assets].dropna()
    if len(R) < lookback + 6:
        return None
    idx = R.index
    prev_w = pd.Series(0.0, index=assets)
    net = []
    for i in range(lookback, len(idx) - 1):
        t, t1 = idx[i], idx[i + 1]
        if mode == "equal":
            w = pd.Series(1.0 / len(assets), index=assets)
        else:  # inverse-vol over trailing lookback (data <= t)
            vol = R[assets].iloc[i - lookback + 1:i + 1].std()
            iv = 1.0 / vol.replace(0, np.nan)
            w = (iv / iv.sum()).fillna(1.0 / len(assets))
        r = float((w * R.loc[t1]).sum())
        cost = float((w - prev_w).abs().sum()) * (cost_bps / 1e4)
        net.append((t1, r - cost))
        prev_w = w
    s = pd.Series(dict(net)).sort_index()
    m = _metrics(s)
    yrs = len(s) / 12
    return dict(label=label, cagr=m["cagr"]*100, sharpe=m["sharpe"], max_drawdown=m["max_drawdown"]*100,
                calmar=m["calmar"], vol=m["vol"]*100, n_months=len(s),
                start=str(s.index[0].date()), end=str(s.index[-1].date()), final_mult=m["final_mult"])


def buyhold(panel, col, start=None):
    p = restrict(panel, start)[col].dropna()
    r = p.pct_change().dropna()
    m = _metrics(r); yrs = len(r)/12
    return dict(label=f"BUYHOLD {col}", cagr=m["cagr"]*100, sharpe=m["sharpe"],
                max_drawdown=m["max_drawdown"]*100, calmar=m["calmar"], vol=m["vol"]*100,
                n_months=len(r), start=str(r.index[0].date()), end=str(r.index[-1].date()),
                final_mult=m["final_mult"])


FIELDS = ["label", "family", "window", "cagr", "sharpe", "max_drawdown", "calmar", "vol",
          "n_months", "start", "end", "final_mult"]
rows = []


def add(d, family, window):
    if d is None:
        return
    d = {**d, "family": family, "window": window}
    for k in ("cagr", "sharpe", "max_drawdown", "calmar", "vol", "final_mult"):
        d[k] = round(d[k], 3)
    rows.append({k: d.get(k, "") for k in FIELDS})
    print(f"  [{family[:4]}|{window}] {d['label'][:42]:42s} CAGR={d['cagr']:5.1f} "
          f"DD={d['max_drawdown']:6.1f} Cal={d['calmar']:.2f} Sh={d['sharpe']:.2f}")


# ============ (a) FACTOR ROTATION (full factor window 2010-26) ============
print("=== (a) FACTOR ROTATION (2010-26) ===")
for n in (1, 2, 3):
    for rk, roc in (("roc6", (6,)), ("roc12", (12,)), ("blend", (3, 6, 12))):
        for mk, ma in (("ma6", 6), ("noma", 0)):
            for cl in (False, True):
                lbl = f"factor_top{n}_{rk}_{mk}_{'cash' if cl else 'nocash'}"
                cfg = GTAAConfig(risk_assets=FACTORS, cash_asset="CASH", top_n=n,
                                 roc_months=roc, ma_months=ma, cash_leg=cl,
                                 require_pos_roc=True, cost_bps=COST, label=lbl)
                try:
                    r = run_gtaa(PANEL_FULL, cfg)
                    add(dict(label=lbl, cagr=r.cagr*100, sharpe=r.sharpe,
                             max_drawdown=r.max_drawdown*100, calmar=r.calmar, vol=r.vol*100,
                             n_months=r.n_months, start=r.start, end=r.end, final_mult=r.final_mult),
                        "rotation", "2010-26")
                except Exception as e:
                    print("   ERR", lbl, str(e)[:50])

# ============ (b) FACTOR BASKETS (2010-26) ============
print("\n=== (b) FACTOR BASKETS (2010-26) ===")
rets_full = PANEL_FULL.pct_change()
for mode in ("equal", "invvol"):
    add(basket_metrics(rets_full, FACTORS, mode, label=f"5factors_{mode}"), "basket", "2010-26")
add(basket_metrics(rets_full, ["Momentum", "LowVol"], "equal", label="Mom+LowVol_equal"), "basket", "2010-26")
add(basket_metrics(rets_full, ["Momentum", "LowVol"], "invvol", label="Mom+LowVol_invvol"), "basket", "2010-26")
add(basket_metrics(rets_full, ["Momentum", "Quality", "LowVol"], "equal", label="Mom+Qual+LowVol_equal"), "basket", "2010-26")
add(basket_metrics(rets_full, ["Momentum", "Quality", "LowVol"], "invvol", label="Mom+Qual+LowVol_invvol"), "basket", "2010-26")

# ============ (c) COMBINED BOOKS (2016-26, common with ETFs) ============
print("\n=== (c) COMBINED factor + asset (2016-26) ===")
start16 = PANEL_FULL[["Gold", "Nasdaq"]].dropna().index.min()
rets16 = restrict(PANEL_FULL, start16).pct_change()
COMBOS = {
    "Mom+LowVol+Gold+Nasdaq": ["Momentum", "LowVol", "Gold", "Nasdaq"],
    "Mom+Gold+Nasdaq": ["Momentum", "Gold", "Nasdaq"],
    "AllFactors+Gold+Nasdaq": FACTORS + ["Gold", "Nasdaq"],
    "Nifty+Gold+Nasdaq (res63)": ["NiftyETF", "Gold", "Nasdaq"],
}
for name, assets in COMBOS.items():
    for mode in ("equal", "invvol"):
        add(basket_metrics(rets16, assets, mode, label=f"{name}_{mode}"), "combined", "2016-26")

# ============ benchmarks + apples-to-apples factor books on 2016-26 ============
print("\n=== benchmarks ===")
add(buyhold(PANEL_FULL, "NiftyETF", start16), "bench", "2016-26")
add(buyhold(PANEL_FULL, "Nifty50"), "bench", "2010-26")
for mode in ("equal", "invvol"):
    add(basket_metrics(rets16, FACTORS, mode, label=f"5factors_{mode}"), "basket", "2016-26")
    add(basket_metrics(rets16, ["Momentum", "LowVol"], mode, label=f"Mom+LowVol_{mode}"), "basket", "2016-26")

# write + summarize
with open(RES / "g2_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); [w.writerow(r) for r in rows]

print("\n=== TOP 15 by Calmar (all families) ===")
for r in sorted(rows, key=lambda x: -float(x["calmar"]))[:15]:
    print(f"  {r['family'][:4]:4s}|{r['window']}  {r['label'][:40]:40s} "
          f"CAGR={r['cagr']:5} DD={r['max_drawdown']:6} Cal={r['calmar']} Sh={r['sharpe']}")
print(f"\nwrote {len(rows)} configs -> results/g2_sweep.csv")
