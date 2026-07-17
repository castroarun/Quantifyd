"""Canonical trade simulator for research/81 — write once, unit-test, reuse.

Non-negotiables implemented here (brief section 3):
  1. No lookahead: a signal on bar t fills at bar t+1 OPEN (slippage-adjusted).
  2. Costs: CostConfig charges both sides + slippage; net AND gross returned.
  3. Execution realism: no fills beyond bar range; stops fill at stop-or-worse
     (gap-through fills at the open); limit targets fill at target-or-better;
     if stop AND target are touched in the same bar → STOP wins (conservative).
  4. Time-stop: forced exit at the CLOSE of the last bar of session
     entry_session + (time_stop_sessions - 1). Hard cap 4 sessions.
  5. One open trade per symbol; signals while in-position are dropped.

Strategy contract: strategies produce a DataFrame of entry intents
(index = SIGNAL bar timestamp, computed causally on that bar):
    direction  +1 long / -1 short
    stop       absolute stop price   (required — R math needs it)
    target     absolute target price (NaN = no target)
The engine does everything else. Position sizing lives at the portfolio
layer; per-trade returns here are in % of entry price with costs applied
on NOTIONAL_RS so flat brokerage is representable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostConfig

# research/81 brief: max holding 3-4 trading days (default). research/82
# (medium-swing, user-mandated 2026-07-17) raises it via env:
#   ENGINE_MAX_TIME_STOP=15
MAX_TIME_STOP = int(os.getenv("ENGINE_MAX_TIME_STOP", "4"))
NOTIONAL_RS = 500_000.0    # notional per trade for cost math (flat fees need one)


@dataclass(frozen=True)
class BTConfig:
    cost: CostConfig = CostConfig()
    time_stop_sessions: int = 4       # 1 = intraday-ish (exit same session close)
    allow_short: bool = True

    def __post_init__(self):
        if not (1 <= self.time_stop_sessions <= MAX_TIME_STOP):
            raise ValueError(f"time_stop_sessions must be 1..{MAX_TIME_STOP}")


def run_symbol(bars: pd.DataFrame, entries: pd.DataFrame,
               cfg: BTConfig, symbol: str = "") -> pd.DataFrame:
    """Simulate all entry intents on one symbol's bars. Returns trades DF."""
    if bars.empty or entries.empty:
        return _empty_trades()

    idx = bars.index
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    sess = idx.normalize()
    sess_codes = pd.factorize(sess)[0]              # 0..S-1 session ordinal
    n = len(idx)
    pos_of = pd.Series(np.arange(n), index=idx)     # timestamp -> bar position

    trades = []
    busy_until = -1                                 # bar position; one trade at a time

    for ts, row in entries.sort_index().iterrows():
        if ts not in pos_of.index:
            continue
        sig_i = int(pos_of[ts])
        ent_i = sig_i + 1                           # fill at NEXT bar open
        if ent_i >= n or ent_i <= busy_until:
            continue
        d = int(row["direction"])
        if d == -1 and not cfg.allow_short:
            continue
        stop = float(row["stop"])
        target = float(row.get("target", np.nan))

        ent_raw = o[ent_i]
        ent_px = cfg.cost.fill_price(ent_raw, is_buy=(d == 1))
        ent_sess = sess_codes[ent_i]
        last_sess = ent_sess + cfg.time_stop_sessions - 1
        # bars of the trade window: ent_i .. last bar of last_sess
        win_end = n - 1
        beyond = np.searchsorted(sess_codes, last_sess + 1)
        if beyond < n:
            win_end = beyond - 1
        elif sess_codes[-1] < last_sess:
            # window runs past data end — cannot verify the time-stop: drop trade
            continue

        exit_i, exit_px, exit_raw, reason = None, None, None, None
        for i in range(ent_i, win_end + 1):
            open_px = o[i]
            if d == 1:
                stop_hit = l[i] <= stop
                gap_stop = open_px <= stop
                tgt_hit = (not np.isnan(target)) and h[i] >= target
                gap_tgt = (not np.isnan(target)) and open_px >= target
            else:
                stop_hit = h[i] >= stop
                gap_stop = open_px >= stop
                tgt_hit = (not np.isnan(target)) and l[i] <= target
                gap_tgt = (not np.isnan(target)) and open_px <= target
            if i == ent_i:
                # entry bar: entry happened AT the open; stop/target act after
                gap_stop = gap_tgt = False
            if stop_hit:                             # stop precedence (conservative)
                exit_raw = open_px if gap_stop else stop
                exit_i, reason = i, "STOP"
                exit_px = cfg.cost.fill_price(exit_raw, is_buy=(d == -1))
                break
            if tgt_hit:
                exit_raw = open_px if gap_tgt else target
                exit_i, reason = i, "TARGET"
                exit_px = cfg.cost.fill_price(exit_raw, is_buy=(d == -1))
                break
        if exit_i is None:                           # time-stop at window close
            exit_i, reason = win_end, "TIME"
            exit_raw = c[win_end]
            exit_px = cfg.cost.fill_price(exit_raw, is_buy=(d == -1))

        qty = NOTIONAL_RS / ent_px
        charges = (cfg.cost.side_cost(ent_px, qty, d == 1)
                   + cfg.cost.side_cost(exit_px, qty, d == -1))
        gross = d * (exit_raw / ent_raw - 1)                 # raw prices: pure gross
        net = d * (exit_px / ent_px - 1) - charges / NOTIONAL_RS  # slippage + charges
        risk = abs(ent_raw - stop) / ent_raw
        trades.append({
            "symbol": symbol, "direction": d,
            "signal_time": ts, "entry_time": idx[ent_i], "exit_time": idx[exit_i],
            "entry_px": ent_px, "exit_px": exit_px, "stop": stop, "target": target,
            "exit_reason": reason,
            "hold_sessions": int(sess_codes[exit_i] - ent_sess + 1),
            "gross_ret": gross, "net_ret": net,
            "r_multiple": net / risk if risk > 0 else np.nan,
        })
        busy_until = exit_i

    out = pd.DataFrame(trades) if trades else _empty_trades()
    if len(out):
        assert out["hold_sessions"].max() <= cfg.time_stop_sessions, \
            "time-stop violated — engine bug"
    return out


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "symbol", "direction", "signal_time", "entry_time", "exit_time",
        "entry_px", "exit_px", "stop", "target", "exit_reason",
        "hold_sessions", "gross_ret", "net_ret", "r_multiple"])
