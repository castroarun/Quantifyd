"""Add the universe-transfer comparison table to the momentum30-subselect study."""
from pathlib import Path
P = Path("/home/arun/quantifyd/frontend/src/data/backtests.ts")
ts = P.read_text()
if "does the edge transfer to mid/small-caps" in ts:
    print("already added"); raise SystemExit

anchor = """            ['Gate + Donchian-15 (winner)', '33.4%', '−17.0%', '~1.7'],
          ],
          highlightRows: [3],
        },
      ],"""
new = """            ['Gate + Donchian-15 (winner)', '33.4%', '−17.0%', '~1.7'],
          ],
          highlightRows: [3],
        },
        {
          title: 'Universe-transfer test — does the edge transfer to mid/small-caps? (net 20bps, 2014–26)',
          caption: 'Same r62 method (top-8 by RS + 100DMA gate + Donchian-15) on different universes. It does NOT transfer: on mid/small-caps the return survives (~34%) but the drawdown blows out and Calmar collapses — the low-DD edge was specific to the Nifty-200 pool. For comparison, research/41 on mid/small-caps uses de-risk-TO-CASH (not per-stock trailing) and holds Calmar 1.44.',
          columns: ['Universe / method', 'CAGR', 'net 20%', 'MaxDD', 'Sharpe', 'Calmar'],
          rows: [
            ['Nifty-200 (r62 method) — original', '34.5%', '30.0%', '−15.6%', '1.83', '2.22'],
            ['MidSmallcap-400 (r62 method)', '34.0%', '28.9%', '−39.2%', '1.81', '0.87'],
            ['mid 100–250 (r62 method)', '28.3%', '24.0%', '−34.5%', '1.59', '0.82'],
            ['MidSmallcap (research/41 own method)', '35.3%', '28.9%', '−24.6%', '1.53', '1.44'],
            ['NIFTYBEES buy & hold', '12.4%', '—', '−36.3%', '0.89', '0.34'],
          ],
          highlightRows: [0, 1],
        },
      ],"""
assert anchor in ts, "anchor missing"
ts = ts.replace(anchor, new, 1)
P.write_text(ts)
print("universe-transfer table added to momentum30-subselect")
