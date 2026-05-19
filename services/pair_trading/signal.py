"""Pair-Trading (Config D) — Signal generator.

Live port of `simulate_pair` from research/39 to a per-day evaluator:
  - given two daily close-price series for the legs of a pair,
  - apply train-fit alpha/beta to compute spread = log(P_a) - alpha - beta * log(P_b),
  - compute z-score on rolling lookback (window = lookback bars),
  - return today's z-score plus the recommended action given current open state.

Action vocabulary:
  ENTRY_LONG   — open new long-spread (long A, short B). Fires on z <= -entry_z.
  ENTRY_SHORT  — open new short-spread (short A, long B). Fires on z >= +entry_z.
  EXIT_MR      — z mean-reverted past 0 (relative to entry direction).
  EXIT_STOP    — |z| >= stop_z (catastrophic blow-out).
  EXIT_TIME    — hold-cap days reached.
  HOLD         — keep open position, no exit signal.
  NO_ACTION    — flat, no entry trigger.

The evaluator does NOT decide concurrency caps or mode-locks — those belong
to the engine. It only reports the spread mechanics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PairRules:
    """Mirror of research/39's PairRules dataclass."""
    entry_z: float
    stop_z: float
    hold_days: int
    lookback: int


@dataclass
class SignalResult:
    """Output of evaluate_today() for one pair on one day."""
    pair_name: str
    trade_date: str            # ISO date
    priceA: float
    priceB: float
    spread: float
    spread_mu: float
    spread_sd: float
    z: float
    action: str                # see vocabulary above
    block_reason: Optional[str] = None
    notes: Optional[str] = None


def _compute_spread_series(prices_a: pd.Series, prices_b: pd.Series,
                            alpha: float, beta: float) -> pd.DataFrame:
    """Align two daily close-price series and compute spread + log series."""
    df = pd.concat([prices_a, prices_b], axis=1, join='inner').dropna()
    if df.shape[1] != 2:
        return pd.DataFrame()
    df.columns = ['pa', 'pb']
    if df.empty:
        return df
    df['la'] = np.log(df['pa'])
    df['lb'] = np.log(df['pb'])
    df['spread'] = df['la'] - alpha - beta * df['lb']
    return df


def compute_z_today(prices_a: pd.Series, prices_b: pd.Series,
                     alpha: float, beta: float, lookback: int) -> Optional[dict]:
    """Compute today's z-score (last bar). Returns None if insufficient history.

    Output dict: {priceA, priceB, spread, spread_mu, spread_sd, z, trade_date}
    """
    df = _compute_spread_series(prices_a, prices_b, alpha, beta)
    if df.empty or len(df) < lookback + 1:
        return None
    # Rolling stats up to and including today (last index)
    df['mu'] = df['spread'].rolling(lookback, min_periods=lookback).mean()
    df['sd'] = df['spread'].rolling(lookback, min_periods=lookback).std(ddof=1)
    last = df.iloc[-1]
    if not np.isfinite(last['sd']) or last['sd'] == 0:
        return None
    z = (last['spread'] - last['mu']) / last['sd']
    if not np.isfinite(z):
        return None
    idx = df.index[-1]
    trade_date = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
    return {
        'trade_date': trade_date,
        'priceA': float(last['pa']),
        'priceB': float(last['pb']),
        'spread': float(last['spread']),
        'spread_mu': float(last['mu']),
        'spread_sd': float(last['sd']),
        'z': float(z),
    }


def evaluate_today(pair_name: str,
                   prices_a: pd.Series, prices_b: pd.Series,
                   alpha: float, beta: float,
                   rules: PairRules,
                   open_position: Optional[dict] = None) -> Optional[SignalResult]:
    """Evaluate one pair on the latest bar. Returns SignalResult or None.

    open_position: if currently in a position for this pair, pass a dict
      with at least {'direction': +1|-1, 'days_held': int}. The evaluator
      will decide whether to exit (MR / STOP / TIME) or HOLD.

    If open_position is None, the evaluator only checks for entry triggers.
    """
    z_info = compute_z_today(prices_a, prices_b, alpha, beta, rules.lookback)
    if z_info is None:
        # Insufficient history — log NO_ACTION as a BLOCKED-style row
        return SignalResult(
            pair_name=pair_name,
            trade_date='',
            priceA=float('nan'), priceB=float('nan'),
            spread=float('nan'), spread_mu=float('nan'),
            spread_sd=float('nan'), z=float('nan'),
            action='BLOCKED',
            block_reason='insufficient_history',
        )

    z = z_info['z']

    if open_position is not None:
        direction = int(open_position.get('direction', 0))
        days_held = int(open_position.get('days_held', 0))
        # Mean-revert exit
        if direction == 1 and z >= 0:
            action = 'EXIT_MR'
        elif direction == -1 and z <= 0:
            action = 'EXIT_MR'
        # Stop (only fires if non-999)
        elif rules.stop_z < 998 and abs(z) >= rules.stop_z:
            action = 'EXIT_STOP'
        # Hold-cap time exit
        elif days_held >= rules.hold_days:
            action = 'EXIT_TIME'
        else:
            action = 'HOLD'
    else:
        # Flat: check entry triggers
        if z <= -rules.entry_z:
            action = 'ENTRY_LONG'
        elif z >= rules.entry_z:
            action = 'ENTRY_SHORT'
        else:
            action = 'NO_ACTION'

    return SignalResult(
        pair_name=pair_name,
        trade_date=z_info['trade_date'],
        priceA=z_info['priceA'],
        priceB=z_info['priceB'],
        spread=z_info['spread'],
        spread_mu=z_info['spread_mu'],
        spread_sd=z_info['spread_sd'],
        z=z,
        action=action,
    )


def implied_sl_distance_pct(entry_z: float, stop_z: float,
                             spread_sd: float, beta: float) -> float:
    """Return implied per-leg SL distance as % of price for sizing.

    Definition: the spread move from entry-z to stop-z is
        |stop_z - entry_z| * spread_sd  (in log-spread units).
    Translated back to per-leg price: divided across legs by the hedge
    ratio. We approximate "equal rupee weight" as half the move on each
    leg, then add a small floor so a tight pair doesn't size silly.
    """
    if not (np.isfinite(spread_sd) and spread_sd > 0):
        return 0.0
    if stop_z >= 998:
        # No stop — fall back to a notional 5% per leg (rare with 999 stops)
        return 5.0
    log_move = abs(stop_z - entry_z) * spread_sd
    # log-spread move ~ delta(log P_a) - beta * delta(log P_b).
    # Equal-rupee weighting: each leg contributes roughly 1/2 the log_move.
    # So per-leg log return magnitude ~ log_move / 2 (independent of beta direction).
    per_leg_log = log_move / 2.0
    pct = (np.exp(per_leg_log) - 1.0) * 100.0
    return float(max(pct, 0.5))  # floor at 0.5% per leg
