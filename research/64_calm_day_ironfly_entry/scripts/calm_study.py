"""research/64 — Calm-day entry screen for neutral/iron-fly selling on NIFTY.
Outcome = was the 2% move-stop NOT hit within the next H trading days (calm window = fly wins).
Comprehensive causal-feature univariate screen (no look-ahead: features use data <= prior close).
NIFTY daily + India VIX from Kite (same source as the DB; fuller history). Read-only research."""
import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, "/home/arun/quantifyd")
from kiteconnect import KiteConnect
import datetime as dt

ak = os.environ.get("KITE_API_KEY")
try:
    import config; ak = ak or getattr(config, "KITE_API_KEY", None)
except Exception:
    pass
tj = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))
at = tj.get("access_token") if isinstance(tj, dict) else tj
ak = ak or (tj.get("api_key") if isinstance(tj, dict) else None)
kite = KiteConnect(api_key=ak); kite.set_access_token(at)


def daily(token, start=dt.date(2015, 1, 1)):
    rows, s = [], start
    while s < dt.date.today():
        e = min(s + dt.timedelta(days=380), dt.date.today())
        try:
            rows += kite.historical_data(token, s, e, "day")
        except Exception as ex:
            print("chunk fail", s, ex)
        s = e + dt.timedelta(days=1)
    seen = {r["date"].date(): r for r in rows}
    df = pd.DataFrame([seen[k] for k in sorted(seen)])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")[["open", "high", "low", "close"]].astype(float)


print("loading NIFTY + India VIX daily ...")
n = daily(256265)
vix = daily(264969)["close"].rename("vix")
print(f"NIFTY {len(n)} bars {n.index[0].date()}->{n.index[-1].date()} | VIX {len(vix)} bars")

H, L, C, O = n["high"], n["low"], n["close"], n["open"]
prevC = C.shift(1)


def rma(x, p):  # Wilder
    return x.ewm(alpha=1 / p, adjust=False).mean()


tr = pd.concat([(H - L), (H - prevC).abs(), (L - prevC).abs()], axis=1).max(axis=1)
atr14 = rma(tr, 14)
up = H.diff(); dn = -L.diff()
plus = np.where((up > dn) & (up > 0), up, 0.0); minus = np.where((dn > up) & (dn > 0), dn, 0.0)
pdi = 100 * rma(pd.Series(plus, index=n.index), 14) / atr14
mdi = 100 * rma(pd.Series(minus, index=n.index), 14) / atr14
dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
adx14 = rma(dx, 14)
delta = C.diff(); g = delta.clip(lower=0); ls = (-delta).clip(lower=0)
rsi14 = 100 - 100 / (1 + rma(g, 14) / rma(ls, 14))
ma20, ma50, ma200 = C.rolling(20).mean(), C.rolling(50).mean(), C.rolling(200).mean()
std20 = C.rolling(20).std()
ret = C.pct_change()
lowest14, highest14 = L.rolling(14).min(), H.rolling(14).max()
stochk = 100 * (C - lowest14) / (highest14 - lowest14)
# CPR (prior day)
piv = (H + L + C) / 3; bc = (H + L) / 2; tc = 2 * piv - bc
cpr_w = (tc - bc).abs() / C
# Ichimoku
conv = (H.rolling(9).max() + L.rolling(9).min()) / 2
base = (H.rolling(26).max() + L.rolling(26).min()) / 2
spanA = ((conv + base) / 2)
spanB = (H.rolling(52).max() + L.rolling(52).min()) / 2
cloud = (spanA - spanB).abs() / C
# weekly
wk = n.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
wpiv = (wk.high + wk.low + wk.close) / 3; wbc = (wk.high + wk.low) / 2; wtc = 2 * wpiv - wbc
wcpr = ((wtc - wbc).abs() / wk.close)
w_inside = (wk.high < wk.high.shift(1)) & (wk.low > wk.low.shift(1))
wcpr_d = wcpr.reindex(n.index, method="ffill")
winside_d = w_inside.reindex(n.index, method="ffill").fillna(False)

feat = pd.DataFrame(index=n.index)
feat["vix"] = vix.reindex(n.index, method="ffill")
feat["cpr_width_d"] = cpr_w
feat["cpr_width_w"] = wcpr_d
feat["atr14_pct"] = atr14 / C
feat["rvol_10"] = ret.rolling(10).std()
feat["rvol_20"] = ret.rolling(20).std()
feat["bb_width"] = (4 * std20) / ma20
feat["adx14"] = adx14
feat["rsi_dist"] = (rsi14 - 50).abs()
feat["stoch_k"] = stochk
feat["inside_day"] = ((H < H.shift(1)) & (L > L.shift(1))).astype(int)
feat["inside_week"] = winside_d.astype(int)
feat["dist_20dma"] = (C - ma20).abs() / ma20
feat["dist_50dma"] = (C - ma50).abs() / ma50
feat["dist_200dma"] = (C - ma200).abs() / ma200
feat["ma20_50_compress"] = (ma20 - ma50).abs() / C
feat["ma20_slope5"] = (ma20 - ma20.shift(5)).abs() / C
feat["donch20_width"] = (H.rolling(20).max() - L.rolling(20).min()) / C
feat["ichi_cloud"] = cloud
feat["range_5d"] = (H.rolling(5).max() - L.rolling(5).min()) / C
feat["mom_5d_abs"] = ret.rolling(5).sum().abs()
feat["gap_prev"] = (O - prevC).abs() / prevC
feat["rng_contraction"] = (H - L) / (H - L).rolling(20).mean()  # <1 = today's range below 20d avg
feat["dow"] = n.index.dayofweek
# shift all features by 1 -> known at PRIOR close (entry next morning), no look-ahead
feat = feat.shift(1)

# outcome: 2% move-stop NOT hit within next H trading days (entry_spot = prior close)
clv = C.values
calm = {}
for Hh in (3, 5, 8):
    out = np.full(len(C), np.nan)
    for i in range(1, len(C) - Hh):
        es = clv[i - 1]
        win = clv[i:i + Hh]
        out[i] = 0.0 if np.max(np.abs(win - es) / es) >= 0.02 else 1.0
    calm[Hh] = pd.Series(out, index=n.index)
df = feat.copy()
for Hh in calm:
    df[f"calm_{Hh}"] = calm[Hh]
df = df.dropna(subset=["calm_5", "vix", "atr14_pct", "adx14"])
print(f"\nusable entry days: {len(df)}  base calm-rate: H3={df.calm_3.mean():.1%} H5={df.calm_5.mean():.1%} H8={df.calm_8.mean():.1%}")

# univariate screen on calm_5, with 3-era stability
eras = np.array_split(df.index, 3)
era_of = {}
for k, idx in enumerate(eras):
    for d in idx: era_of[d] = k
df["era"] = [era_of[d] for d in df.index]
BASE = df.calm_5.mean()
numeric = [c for c in feat.columns if c not in ("inside_day", "inside_week", "dow")]
rows = []
for f in numeric:
    s = df[[f, "calm_5", "era"]].dropna()
    if len(s) < 200: continue
    try:
        s["q"] = pd.qcut(s[f], 5, labels=False, duplicates="drop")
    except Exception:
        continue
    g = s.groupby("q").calm_5.mean()
    lo, hi = g.iloc[0], g.iloc[-1]
    direction = "LOW=calm" if lo > hi else "HIGH=calm"
    best, worst = max(lo, hi), min(lo, hi)
    # era stability of the spread sign
    sign_ok = 0
    for er in (0, 1, 2):
        se = s[s.era == er]
        ge = se.groupby("q").calm_5.mean()
        if len(ge) >= 2 and ((ge.iloc[0] - ge.iloc[-1]) * (lo - hi) > 0): sign_ok += 1
    rows.append(dict(feature=f, n=len(s), best_q_calm=round(best, 3), worst_q_calm=round(worst, 3),
                     spread=round(best - worst, 3), direction=direction, eras_consistent=f"{sign_ok}/3"))
res = pd.DataFrame(rows).sort_values("spread", ascending=False)
print(f"\nbase calm_5 = {BASE:.1%}\n=== univariate calm-predictors (sorted by top-vs-bottom-quintile spread) ===")
print(res.to_string(index=False))
# binary features
print("\n=== binary features (calm_5 when =1 vs =0) ===")
for f in ("inside_day", "inside_week"):
    s = df[[f, "calm_5"]].dropna()
    print(f"  {f}: =1 -> {s[s[f]==1].calm_5.mean():.1%} (n={int((s[f]==1).sum())}) | =0 -> {s[s[f]==0].calm_5.mean():.1%}")
print("\n=== day-of-week calm_5 ===")
print(df.groupby("dow").calm_5.mean().round(3).to_dict())
res.to_csv("/tmp/calm_univariate.csv", index=False)
print("\nsaved /tmp/calm_univariate.csv")
