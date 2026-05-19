"""Live signal generators for the 3 intraday-75WR configs.

Each module exports a single eval_*(df_5m, ctx) -> dict|None function, where
df_5m is the per-stock 5-min OHLCV history including today's session up to
(and including) the just-closed bar, and ctx carries the NIFTY regime tags
+ config params.

The dict (when non-None) has:
    {
        'fired': True,
        'direction': 'LONG' | 'SHORT',
        'entry_price': float,        # next-bar projected open (use last close)
        'sl_price': float,           # entry +/- sl_pct
        'target_price': float,       # entry +/- tp_pct
        'meta': {...}                # signal-specific diagnostics
    }
"""

from . import diamond_short
from . import long_mr
from . import long_tc
from . import multi_bar_bounce

__all__ = ['diamond_short', 'long_mr', 'long_tc', 'multi_bar_bounce']
