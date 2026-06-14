"""Add the SILVERBEES addendum (comparison table + caveat) to the gtaa-etf-rotation
study in backtests.ts. Run on VPS. Idempotent."""
from pathlib import Path
P = Path("/home/arun/quantifyd/frontend/src/data/backtests.ts")
ts = P.read_text()

if "SILVERBEES add-on" in ts:
    print("addendum already present"); raise SystemExit

# 1) insert a comparison table before the gtaa study's comparisons-array close.
#    anchor = the gtaa per-year table's last row (unique to this study).
anchor = "          ['2026*', '+12.8', '−9.3', '+22.1'],\n        ],\n        highlightRows: [0],\n        heatmap: true,\n      },\n"
table = """      {
        title: 'SILVERBEES add-on (tested per request) — silver HURTS over the full window',
        caption: 'Indian silver ETFs only exist from 2022; pre-2022 silver uses a validated proxy (intl silver × USDINR, monthly-return corr 0.85 to SILVERBEES).',
        columns: ['Book (monthly reb, net 20bps)', 'Window', 'CAGR', 'MaxDD', 'Calmar'],
        rows: [
          ['3-asset (Nifty/Gold/Nasdaq), equal', '2015–26 (11.4y)', '17.6%', '−11.3%', '1.57'],
          ['4-asset (+Silver), equal', '2015–26*', '18.2%', '−12.4%', '1.47'],
          ['4-asset (+Silver), inverse-vol', '2015–26*', '18.0%', '−12.4%', '1.45'],
          ['3-asset, equal', '2022–26 (4.3y, metals bull)', '21.6%', '−9.0%', '2.40'],
          ['4-asset (+Silver), inverse-vol', '2022–26', '24.5%', '−9.4%', '2.61'],
        ],
        highlightRows: [0],
        heatmap: false,
      },
"""
assert anchor in ts, "gtaa per-year anchor not found"
ts = ts.replace(anchor, anchor + table, 1)

# 2) add a caveat to the gtaa study's caveats array (anchor on its unique LIQUIDBEES caveat).
cav_anchor = "      'Backtest, net of 20 bps modelled cost. LIQUIDBEES price-return"
new_cav = ("      'SILVERBEES add-on (user request): Indian silver ETFs only exist from 2022, so pre-2022 "
           "silver uses a validated proxy (intl silver × USDINR, monthly-return corr 0.85 to SILVERBEES). "
           "Over the full 2015–26 window adding silver LOWERS Calmar (1.57→1.47 equal, 1.50→1.45 inv-vol) "
           "— silver is 0.66 correlated to gold (redundant precious-metal) and very volatile (29% vol, −28% DD). "
           "The strong 2022–26 result (Calmar 2.6) was a precious-metals bull, not a durable benefit — a recency-bias trap.',\n")
assert cav_anchor in ts, "gtaa LIQUIDBEES caveat anchor not found"
ts = ts.replace(cav_anchor, new_cav + cav_anchor, 1)

P.write_text(ts)
print("addendum inserted (table + caveat)")
