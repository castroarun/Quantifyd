"""Super-winner guard + tail concentration for G1 trades (research/73)."""
import os, glob
import numpy as np, pandas as pd

RES = "research/73_weekly_supertrend_investing/results"
for band in ["Nifty50", "Nifty200", "Midcap150", "Smallcap250"]:
    f = os.path.join(RES, f"g1_{band}_trades.csv")
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f)
    net = df["net"].values * 100
    n = len(net)
    total = net.sum()
    mean_all = net.mean()
    # drop top-3 individual trades
    order = np.argsort(net)[::-1]
    drop3 = net[order[3:]].mean()
    # by symbol: contribution
    by_sym = df.groupby("symbol")["net"].sum().sort_values(ascending=False) * 100
    top3_syms = by_sym.head(3)
    share_top3 = 100 * top3_syms.sum() / total if total else 0
    # share of total from top-10% of trades
    k = max(1, n // 10)
    top10pct_share = 100 * net[order[:k]].sum() / total if total else 0
    print(f"\n=== {band} (n={n}) ===")
    print(f"  mean/trade net: all={mean_all:.2f}%  drop-top3-trades={drop3:.2f}%  "
          f"delta={mean_all-drop3:.2f}pp")
    print(f"  top-10% of trades supply {top10pct_share:.0f}% of total return")
    print(f"  top-3 NAMES supply {share_top3:.0f}% of total; they are: "
          f"{', '.join(f'{s}(+{v:.0f}%)' for s,v in top3_syms.items())}")
    # fraction of trades that are net-negative vs tiny
    print(f"  net<0: {100*(net<0).mean():.0f}%   |net|<10%: {100*(abs(net)<10).mean():.0f}%   "
          f"net>50%: {100*(net>50).mean():.0f}%")
