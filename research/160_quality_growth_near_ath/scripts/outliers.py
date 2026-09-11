# -*- coding: utf-8 -*-
"""research/160 G3 - outlier dependence, at trade level.

    outliers.py results/<LABEL>_trades.csv [more...]

The question r/159 asked and answered badly for its own pattern: is the result a broad edge
or a handful of lottery tickets? Three readings, all computed on the engine's own dumped
trade list (net of costs, per trade, not compounded into the book):

  * total  - the product of (1 + net return) over every closed trade, as a crude
             "if you had ridden them one after another" growth factor;
  * ex-top-10 - the same product with the ten best trades deleted;
  * capped +50% / +100% - the same product with every winner truncated at that return.

This is a TRADE-LEVEL proxy, not a re-simulation: the real book holds fifteen to thirty
names at once, so the product overstates compounding. It is reported as a ratio between
arms, never as a return. What matters is the SHAPE - a book whose ex-top-10 product falls
by orders of magnitude was carried by a handful of names.

Also prints the share of total net trade return contributed by the top 1% / 5% / 10% of
trades, which is the reading that does not depend on the compounding proxy at all.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def one(path):
    t = pd.read_csv(path)
    r = pd.to_numeric(t['ret_net_pct'], errors='coerce').dropna().values / 100.0
    n = len(r)
    if not n:
        return None
    order = np.argsort(r)[::-1]
    prod = lambda x: float(np.prod(1.0 + x))                                # noqa: E731
    full = prod(r)
    ex10 = prod(np.delete(r, order[:10]))
    cap50 = prod(np.minimum(r, 0.50))
    cap100 = prod(np.minimum(r, 1.00))
    tot = r.sum()
    share = lambda q: float(r[order[:max(1, int(n * q))]].sum() / tot) if tot else np.nan  # noqa: E731
    return dict(label=Path(path).name.replace('_trades.csv', ''), n_trades=n,
                mean_pct=100 * r.mean(), median_pct=100 * np.median(r),
                win_pct=100 * (r > 0).mean(), best_pct=100 * r.max(), worst_pct=100 * r.min(),
                full=full, ex_top10=ex10, ratio_ex10=full / ex10 if ex10 else np.nan,
                cap50=cap50, cap100=cap100,
                share_top1=share(0.01), share_top5=share(0.05), share_top10=share(0.10))


def main():
    rows = [one(p) for p in sys.argv[1:]]
    rows = [r for r in rows if r]
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    hdr = ('| book | trades | mean % | median % | win % | best % | full x | ex-top-10 x | '
           'full/ex10 | cap+50 x | cap+100 x | top-1% share | top-5% share | top-10% share |')
    print(hdr)
    print('|---|' + '---:|' * 13)
    for _, r in df.iterrows():
        print('| %s | %d | %.2f | %.2f | %.1f | %.0f | %.3g | %.3g | %.1f | %.3g | %.3g | '
              '%.1f%% | %.1f%% | %.1f%% |'
              % (r.label, r.n_trades, r.mean_pct, r.median_pct, r.win_pct, r.best_pct,
                 r.full, r.ex_top10, r.ratio_ex10, r.cap50, r.cap100,
                 100 * r.share_top1, 100 * r.share_top5, 100 * r.share_top10))


if __name__ == '__main__':
    sys.exit(main())
