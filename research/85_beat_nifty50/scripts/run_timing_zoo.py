"""research/85 addendum EXP-ZOO (user mandate): per-stock timing-signal zoo on
Nifty 50 sleeves — in-stock when signal ON, liquid fund when OFF.

LOCKED GRID (21 cells, ledger +21; Bonferroni note: best-of-21 haircut):
  ST(7,3) ST(10,3) ST(10,2) ST(14,2) ST(20,3)
  EMAX 9/21, 20/50, 50/100, 50/200 ; P>EMA 20, 50, 100, 200
  DONCH 20/10, 55/20, 100/50 (in on N-high break, out on M-low break)
  RSI14>50, RSI14>60 ; STOCH14-3 K>D ; MACD line>signal
Fair benchmark: SAME-basket B&H. Costs 15bp/side per flip; liquid 5.5%.
Window 2005-2023. Case 1 (all N50); finalists also case 2 (top-10 mcap).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R81 = HERE.parents[1] / "81_swing_edge_discovery"
for p in (str(R81), str(R81.parents[1]), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                     # noqa: E402
from run_beat_n50 import N50                           # noqa: E402
from services.technical_indicators import calc_supertrend  # noqa: E402

START, END = "2005-01-01", "2023-12-31"
LIQ = 0.055 / 252
COST_SIDE = 0.0015


def sig_st(df, period, mult):
    _, direction = calc_supertrend(df, atr_period=period, multiplier=mult)
    return direction > 0


def sig_emax(df, f, s):
    ef = df["close"].ewm(span=f, adjust=False).mean()
    es = df["close"].ewm(span=s, adjust=False).mean()
    return ef > es


def sig_pgtema(df, n):
    return df["close"] > df["close"].ewm(span=n, adjust=False).mean()


def sig_donch(df, n, m):
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(m).min().shift(1)
    state = pd.Series(np.nan, index=df.index)
    state[df["close"] > hi] = 1.0
    state[df["close"] < lo] = 0.0
    return state.ffill().fillna(0.0) > 0.5


def sig_rsi(df, thr):
    d = df["close"].diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return rsi > thr


def sig_stoch(df):
    ll = df["low"].rolling(14).min()
    hh = df["high"].rolling(14).max()
    k = (100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()
    return k > k.rolling(3).mean()


def sig_macd(df):
    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    line = e12 - e26
    return line > line.ewm(span=9, adjust=False).mean()


CELLS = (
    [(f"ST{p}_{m}", (lambda df, p=p, m=m: sig_st(df, p, m))) for p, m in
     ((7, 3.0), (10, 3.0), (10, 2.0), (14, 2.0), (20, 3.0))]
    + [(f"EMAX{f}_{s}", (lambda df, f=f, s=s: sig_emax(df, f, s))) for f, s in
       ((9, 21), (20, 50), (50, 100), (50, 200))]
    + [(f"PgtEMA{n}", (lambda df, n=n: sig_pgtema(df, n))) for n in
       (20, 50, 100, 200)]
    + [(f"DONCH{n}_{m}", (lambda df, n=n, m=m: sig_donch(df, n, m))) for n, m in
       ((20, 10), (55, 20), (100, 50))]
    + [("RSI14gt50", lambda df: sig_rsi(df, 50)),
       ("RSI14gt60", lambda df: sig_rsi(df, 60)),
       ("STOCH_KgtD", sig_stoch),
       ("MACD", sig_macd)]
)


def sleeve(df, sig):
    on = sig.shift(1).fillna(False)
    ret = df["close"].pct_change().fillna(0.0)
    out = np.where(on, ret, LIQ)
    flips = on.astype(int).diff().abs().fillna(0)
    return pd.Series(out - flips.to_numpy() * COST_SIDE, index=df.index), \
        float(flips.sum())


def main():
    bars = {}
    for sym in N50:
        df = loader.load_bars(sym, "day", start="2004-01-01", end=END)
        if len(df) >= 500:
            bars[sym] = df
    print(f"universe: {len(bars)} names, {len(CELLS)} timing cells")
    bh = pd.concat([b["close"].pct_change() for b in bars.values()], axis=1).mean(axis=1)
    bh = bh[(bh.index >= START) & (bh.index <= END)]
    cbh = metrics.curve_stats((1 + bh.fillna(0)).cumprod())
    print(f"SAME-BASKET B&H: CAGR {cbh['cagr']:.1%} MaxDD {cbh['max_dd']:.0%} "
          f"Calmar {cbh['calmar']:.2f}\n")
    rows = []
    for name, fn in CELLS:
        rets, nflips = [], 0.0
        for sym, df in bars.items():
            r, f = sleeve(df, fn(df))
            rets.append(r)
            nflips += f
        port = pd.concat(rets, axis=1).mean(axis=1)
        port = port[(port.index >= START) & (port.index <= END)]
        cs = metrics.curve_stats((1 + port).cumprod())
        rows.append({"cell": name, "CAGR": round(cs["cagr"] * 100, 1),
                     "vs_BH_pp": round((cs["cagr"] - cbh["cagr"]) * 100, 1),
                     "MaxDD": round(cs["max_dd"] * 100, 0),
                     "Calmar": round(cs["calmar"], 2),
                     "Sharpe": round(cs["sharpe"], 2),
                     "flips/stock/yr": round(nflips / len(bars) / 19, 1)})
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows).sort_values("Calmar", ascending=False)
    print("\n=== ZOO ranking (by Calmar; fair bench = same-basket B&H) ===")
    print(tab.to_string(index=False))
    tab.to_csv(HERE.parent / "results" / "zoo_ranking.csv", index=False)
    print("\nZOO COMPLETE", flush=True)


if __name__ == "__main__":
    main()
