"""research/64 P3 — premium-aware net edge (the real sweet spot).
We lack historical option premiums, so use VIX as a premium proxy: an ATM straddle's credit scales ~
with IV*sqrt(T)*spot. We hold credit-as-fraction-of-spot ~ VIX/100 * sqrt(H/252) (then the fly keeps a
fraction via the wings). EV per trade = calm*W - (1-calm)*Lstop, where BOTH W and the calm-rate vary by
VIX bucket. Goal: find the VIX (and compression) regime that maximises EV, reconciling the calm-vs-premium
tradeoff that the calm-only screen can't see. Read-only."""
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect
import datetime as dt
pd.set_option("display.width", 200)

ak = os.environ.get("KITE_API_KEY")
try:
    import config; ak = ak or getattr(config, "KITE_API_KEY", None)
except Exception: pass
tj = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
at = tj.get("access_token") if isinstance(tj, dict) else tj
ak = ak or (tj.get("api_key") if isinstance(tj, dict) else None)
kite = KiteConnect(api_key=ak); kite.set_access_token(at)


def daily(token, start=dt.date(2015, 1, 1)):
    rows, s = [], start
    while s < dt.date.today():
        e = min(s + dt.timedelta(days=380), dt.date.today())
        for _try in range(4):
            try:
                rows += kite.historical_data(token, s, e, "day"); break
            except Exception as ex:
                import time; time.sleep(1.0)
        s = e + dt.timedelta(days=1)
    seen = {r["date"].date(): r for r in rows}
    df = pd.DataFrame([seen[k] for k in sorted(seen)])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")[["open", "high", "low", "close"]].astype(float)


CACHE = "/tmp/nifty_vix_cache.pkl"
if os.path.exists(CACHE):
    n, vixs = pd.read_pickle(CACHE); vix = vixs
    print("loaded cached data")
else:
    n = daily(256265); vix = daily(264969)["close"].rename("vix")
    pd.to_pickle((n, vix), CACHE)
H, L, C, O = n.high, n.low, n.close, n.open; prevC = C.shift(1)
rma = lambda x, p: x.ewm(alpha=1/p, adjust=False).mean()
tr = pd.concat([(H-L), (H-prevC).abs(), (L-prevC).abs()], axis=1).max(axis=1); atr14 = rma(tr, 14)
low14, high14 = L.rolling(14).min(), H.rolling(14).max(); piv = (H+L+C)/3; bc = (H+L)/2; tc = 2*piv-bc
f = pd.DataFrame(index=n.index)
f["vix"] = vix.reindex(n.index, method="ffill")
f["atr14_pct"] = atr14/C
f["cpr_width_d"] = (tc-bc).abs()/C
f["stoch_k"] = 100*(C-low14)/(high14-low14)
f["spot"] = C
f = f.shift(1)
clv = C.values; o5 = np.full(len(C), np.nan); maxmove = np.full(len(C), np.nan)
for i in range(1, len(C)-5):
    es = clv[i-1]; mv = np.abs(clv[i:i+5]-es)/es
    o5[i] = 0.0 if mv.max() >= 0.02 else 1.0; maxmove[i] = mv.max()
f["calm"] = pd.Series(o5, index=n.index); f["maxmove"] = pd.Series(maxmove, index=n.index)
d = f.dropna().copy()

# --- premium model (proxy) ---
# ATM straddle credit as fraction of spot over the hold H=5 (~ business week): ~ IV*sqrt(T)
# An iron fly keeps a fraction of that (wings cost ~25-35% of straddle); net credit ~ 0.7 * straddle.
# We express W (rupees, 10 lots, qty 650) = credit_pts * 650 ; credit_pts = spot * straddle_frac * keep
H_yrs = 5/252
KEEP = 0.65          # fly net credit as fraction of the short straddle premium
WIN_CAPTURE = 0.55   # on a calm (won) trade you realise ~55% of the credit (PT/roll, not full decay)
STOP_FRAC = 0.115    # calibrated so Lstop ~= verified Rs34k at current spot (2% MTM stop, not full span)
QTY = 650
d["straddle_frac"] = (d.vix/100)*np.sqrt(H_yrs)*0.8     # ATM straddle ~0.8*IV*sqrt(T) of spot
d["credit_pts"] = d.spot*d.straddle_frac*KEEP
d["W"] = d.credit_pts*WIN_CAPTURE*QTY                   # rupees won if calm
# stop loss in rupees ~ STOP_FRAC of the wing span (~2% of spot) * qty, floored
d["Lstop"] = (0.02*d.spot*STOP_FRAC)*QTY
d["EV"] = d.calm*d.W - (1-d.calm)*d.Lstop
print(f"N={len(d)} | mean credit_pts={d.credit_pts.mean():.0f} | mean W=Rs{d.W.mean():,.0f} | mean Lstop=Rs{d.Lstop.mean():,.0f}")

# 1. EV by VIX bucket (the calm-vs-premium tradeoff made explicit)
print("\n=== 1. EV by VIX bucket (per 10-lot trade) — calm UP as VIX falls, premium DOWN as VIX falls ===")
d["vbin"] = pd.cut(d.vix, [0, 11, 12, 13, 14, 15, 16, 18, 20, 25, 100])
g = d.groupby("vbin").agg(n=("calm", "size"), calm=("calm", "mean"), W=("W", "mean"),
                          Lstop=("Lstop", "mean"), EV=("EV", "mean"))
g["calm"] = (g["calm"]*100).round(0); g[["W", "Lstop", "EV"]] = g[["W", "Lstop", "EV"]].round(0)
print(g.to_string())

# 2. EV: ungated vs compression-gated, within VIX floors (does the gate add EV on top of a VIX floor?)
print("\n=== 2. mean EV/trade: VIX-floor only vs VIX-floor + compression-gate ===")
gate = (d.atr14_pct <= d.atr14_pct.quantile(0.5)) & (d.cpr_width_d <= d.cpr_width_d.quantile(0.5)) & (d.stoch_k >= d.stoch_k.quantile(0.5))
print(f"{'regime':28s} {'n':>5} {'calm':>6} {'EV/trade':>10} {'totalEV(arb)':>12}")
for lab, base_m in [("all days", pd.Series(True, index=d.index)),
                    ("VIX>=13", d.vix >= 13), ("VIX 13-18", (d.vix >= 13) & (d.vix <= 18)),
                    ("VIX<=16", d.vix <= 16)]:
    for gl, gm in [("", base_m), (" + compress-gate", base_m & gate)]:
        s = d[gm]
        if len(s) < 30: continue
        print(f"{(lab+gl):28s} {len(s):5d} {s.calm.mean()*100:5.0f}% {s.EV.mean():10,.0f} {s.EV.sum():12,.0f}")

# 3. sweet-spot search: VIX floor x compression -> EV/trade AND total EV (frequency matters)
print("\n=== 3. sweet-spot grid: per-trade EV vs total EV (coverage) ===")
print(f"{'rule':34s} {'cov%':>5} {'calm':>5} {'EV/tr':>9} {'totalEV':>11}")
rules = {
    "all": pd.Series(True, index=d.index),
    "VIX>=12": d.vix >= 12, "VIX>=13": d.vix >= 13, "VIX>=14": d.vix >= 14,
    "compress-gate": gate,
    "VIX>=13 & compress": (d.vix >= 13) & gate,
    "VIX 13-20 & compress": (d.vix >= 13) & (d.vix <= 20) & gate,
}
N = len(d)
for name, m in rules.items():
    s = d[m]
    print(f"{name:34s} {m.mean()*100:4.0f}% {s.calm.mean()*100:4.0f}% {s.EV.mean():9,.0f} {s.EV.sum():11,.0f}")
print("\n(EV/tr = quality per trade; totalEV = EV/tr x frequency = book-level value. The sweet spot")
print(" maximises totalEV, balancing calm-rate, premium richness and how often you get to trade.)")
print("Model assumptions: KEEP=0.65, WIN_CAPTURE=0.55, STOP_FRAC=0.45 — proxies; AlgoTest gives exact ₹.")
