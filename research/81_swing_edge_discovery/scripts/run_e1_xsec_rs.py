"""EXP-E1: cross-sectional short-term relative strength, F&O universe, 2-4d hold.

G0: institutional rebalancing chases k-day winners (continuation) — OR
short-term reversal dominates at this horizon (the literature says 1-week
reversal, 3-12m momentum). The screen answers which sign survives costs here.
Signal: symbol ENTERS the top decile (long) / bottom decile (short) of the
universe's 5-day return ranking today (wasn't in it yesterday).
skip1 variant computes the 5-day return ending YESTERDAY (t-6..t-1) to
strip 1-day reversal contamination.
GRID LOCKED: skip {0,1} x dir {L,S} x ts {2,4} = 16 cells. Stop 2.5xATR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from screen_common import (STUDY, WARMUP_START, IS_END, atr14, clip_is,
                           run_screen, loader)

EXP = "e1"
RESULTS = STUDY / "experiments" / "E1_xsec_rs_daily" / "results"
GRID = [(skip, d, ts) for skip in (0, 1) for d in (1, -1) for ts in (2, 4)]
K_ATR = 2.5
LOOKBACK = 5
DECILE = 0.10

_membership: dict[int, dict[int, pd.DataFrame]] = {}


def _build_panel():
    """closes panel -> per-skip {+1: in-top-decile bool DF, -1: bottom}."""
    from services.data_manager import FNO_LOT_SIZES
    closes = {}
    for sym in sorted(FNO_LOT_SIZES.keys()):
        df = loader.load_bars(sym, "day", start=WARMUP_START, end=IS_END)
        if len(df) >= 300:
            closes[sym] = df["close"]
    panel = pd.DataFrame(closes)
    for skip in (0, 1):
        c = panel.shift(skip)
        mom = c / c.shift(LOOKBACK) - 1
        rank = mom.rank(axis=1, pct=True)
        _membership[skip] = {1: rank >= (1 - DECILE), -1: rank <= DECILE}
    print(f"panel built: {panel.shape[1]} symbols x {panel.shape[0]} days",
          flush=True)


def cell_label(cell):
    skip, d, ts = cell
    return f"rs5_skip{skip}_{'L' if d == 1 else 'S'}_ts{ts}"


def build_entries(df, cell, allowed):
    skip, d, ts = cell
    sym = getattr(build_entries, "_sym", None)
    memb = _membership[skip][d]
    if sym not in memb.columns:
        return pd.DataFrame(columns=["direction", "stop", "target"])
    m_sym = memb[sym].reindex(df.index).fillna(False)
    fresh = m_sym & ~m_sym.shift(1).fillna(False)      # ENTERS the decile
    a = atr14(df)
    stop = df["close"] - K_ATR * a if d == 1 else df["close"] + K_ATR * a
    m = fresh & a.notna() & allowed
    sig = clip_is(df.index[m])
    return pd.DataFrame({"direction": d, "stop": stop.loc[sig],
                         "target": np.nan}, index=sig)


def build_entries_wrapper(df, cell, allowed):
    return build_entries(df, cell, allowed)


if __name__ == "__main__":
    _build_panel()
    # run_screen doesn't pass the symbol to build_entries; wrap run to inject it
    import screen_common as sc
    orig_run_symbol = sc.run_symbol

    def patched(bars, entries, cfg, symbol=""):
        return orig_run_symbol(bars, entries, cfg, symbol=symbol)

    # inject current symbol via a tiny shim around build_entries
    def be(df, cell, allowed, _state={}):
        return build_entries(df, cell, allowed)

    # simplest reliable injection: monkey-patch loader.load_bars to remember sym
    orig_load = loader.load_bars

    def load_remember(symbol, *a, **kw):
        build_entries._sym = symbol
        return orig_load(symbol, *a, **kw)

    loader.load_bars = load_remember
    run_screen(EXP, RESULTS, GRID, cell_label, be)
