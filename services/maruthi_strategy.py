"""
Maruthi Always-On Strategy Engine
====================================

Dual SuperTrend (Master 7,5 + Child 7,2) regime-based strategy.

Core logic:
- Master ST(7,5) determines regime (BULL/BEAR)
- Child ST(7,2) triggers entries within regime
- Positions accumulate (max 5 futures lots)
- On master reversal: close all except last short option

Signal types:
- MASTER_BULL: Master ST flips bullish → new BULL regime
- MASTER_BEAR: Master ST flips bearish → new BEAR regime
- CHILD_BULL: Child ST flips bullish within regime
- CHILD_BEAR: Child ST flips bearish within regime
- HARD_SL: Price breaches master ST ± buffer → go FLAT

Hard SL is a TRAILING stop: it follows the master SuperTrend line each candle.
- BULL: SL = master_st - buffer, only moves UP (never down)
- BEAR: SL = master_st + buffer, only moves DOWN (never up)

All computation is pure — no API calls, no DB writes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from services.technical_indicators import calc_supertrend

logger = logging.getLogger(__name__)


class MaruthiSignal:
    """Represents a strategy signal."""

    MASTER_BULL = 'MASTER_BULL'
    MASTER_BEAR = 'MASTER_BEAR'
    CHILD_BULL = 'CHILD_BULL'
    CHILD_BEAR = 'CHILD_BEAR'
    HARD_SL = 'HARD_SL'

    def __init__(self, signal_type: str, candle: dict, master_st: float,
                 child_st: float, master_dir: int, child_dir: int):
        self.signal_type = signal_type
        self.candle = candle  # {time, open, high, low, close, volume}
        self.master_st = master_st
        self.child_st = child_st
        self.master_direction = master_dir
        self.child_direction = child_dir

    def __repr__(self):
        return f"<Signal {self.signal_type} @ {self.candle.get('time', '?')} close={self.candle.get('close', '?')}>"


def compute_dual_supertrend(df: pd.DataFrame,
                             master_period: int = 7, master_mult: float = 5.0,
                             child_period: int = 7, child_mult: float = 2.0) -> pd.DataFrame:
    """
    Compute both SuperTrend indicators on 30-min OHLCV data.

    Returns df with columns:
    - master_st, master_dir (1=bull, -1=bear)
    - child_st, child_dir
    - master_flip_bull, master_flip_bear
    - child_flip_bull, child_flip_bear
    """
    df = df.copy()

    # Master SuperTrend (7, 5)
    df['master_st'], df['master_dir'] = calc_supertrend(df, master_period, master_mult)

    # Child SuperTrend (7, 2)
    df['child_st'], df['child_dir'] = calc_supertrend(df, child_period, child_mult)

    # Detect flips
    df['master_flip_bull'] = (df['master_dir'] == 1) & (df['master_dir'].shift(1) == -1)
    df['master_flip_bear'] = (df['master_dir'] == -1) & (df['master_dir'].shift(1) == 1)
    df['child_flip_bull'] = (df['child_dir'] == 1) & (df['child_dir'].shift(1) == -1)
    df['child_flip_bear'] = (df['child_dir'] == -1) & (df['child_dir'].shift(1) == 1)

    return df


def detect_signals(df: pd.DataFrame, current_regime: str,
                   hard_sl_buffer: float = 50.0) -> List[MaruthiSignal]:
    """
    Scan the latest candle(s) for signals.

    Args:
        df: DataFrame with dual SuperTrend computed (use compute_dual_supertrend first)
        current_regime: 'BULL', 'BEAR', or 'FLAT'
        hard_sl_buffer: Points buffer for hard stop loss from master ST line

    Returns:
        List of signals (usually 0 or 1 per candle)
    """
    if len(df) < 2:
        return []

    signals = []
    latest = df.iloc[-1]
    candle = {
        'time': str(latest.get('date', latest.name)),
        'open': float(latest['open']),
        'high': float(latest['high']),
        'low': float(latest['low']),
        'close': float(latest['close']),
        'volume': float(latest.get('volume', 0)),
    }

    master_st = float(latest['master_st'])
    child_st = float(latest['child_st'])
    master_dir = int(latest['master_dir'])
    child_dir = int(latest['child_dir'])

    # Check hard SL first (highest priority)
    if current_regime == 'BULL':
        hard_sl = master_st - hard_sl_buffer
        if latest['low'] <= hard_sl:
            signals.append(MaruthiSignal(
                MaruthiSignal.HARD_SL, candle,
                master_st, child_st, master_dir, child_dir
            ))
            return signals  # Hard SL overrides everything

    elif current_regime == 'BEAR':
        hard_sl = master_st + hard_sl_buffer
        if latest['high'] >= hard_sl:
            signals.append(MaruthiSignal(
                MaruthiSignal.HARD_SL, candle,
                master_st, child_st, master_dir, child_dir
            ))
            return signals

    # Check master flips (regime change)
    if latest['master_flip_bull']:
        signals.append(MaruthiSignal(
            MaruthiSignal.MASTER_BULL, candle,
            master_st, child_st, master_dir, child_dir
        ))
        return signals  # Master flip is a full regime change

    if latest['master_flip_bear']:
        signals.append(MaruthiSignal(
            MaruthiSignal.MASTER_BEAR, candle,
            master_st, child_st, master_dir, child_dir
        ))
        return signals

    # Check child flips (within regime)
    if latest['child_flip_bull']:
        signals.append(MaruthiSignal(
            MaruthiSignal.CHILD_BULL, candle,
            master_st, child_st, master_dir, child_dir
        ))

    if latest['child_flip_bear']:
        signals.append(MaruthiSignal(
            MaruthiSignal.CHILD_BEAR, candle,
            master_st, child_st, master_dir, child_dir
        ))

    return signals


def compute_trigger_price(candle: dict, direction: str, buffer: float = 1.0) -> float:
    """
    Compute SL-L trigger price for futures entry.

    For BUY: trigger = candle high + buffer (enters above the candle)
    For SELL: trigger = candle low - buffer (enters below the candle)
    """
    if direction == 'BUY':
        return round(candle['high'] + buffer, 1)
    else:  # SELL
        return round(candle['low'] - buffer, 1)


def compute_limit_price(trigger_price: float, direction: str, slippage: float = 5.0) -> float:
    """
    Compute limit price for SL-L order (to prevent excessive slippage).

    For BUY: limit = trigger + slippage
    For SELL: limit = trigger - slippage
    """
    if direction == 'BUY':
        return round(trigger_price + slippage, 1)
    else:
        return round(trigger_price - slippage, 1)


def compute_hard_sl(master_st_value: float, regime: str, buffer: float = 50.0,
                    prev_hard_sl: float = 0.0) -> float:
    """
    Compute trailing hard stop loss price.

    The hard SL trails the master SuperTrend line each candle:
    - BULL regime: SL = master_st - buffer, only moves UP (never down)
    - BEAR regime: SL = master_st + buffer, only moves DOWN (never up)

    Args:
        master_st_value: Current master SuperTrend value
        regime: 'BULL' or 'BEAR'
        buffer: Points distance from master ST (default 50)
        prev_hard_sl: Previous hard SL value (for trailing logic)
    """
    if regime == 'BULL':
        new_sl = round(master_st_value - buffer, 1)
        # Trail up only — never move SL down in BULL
        if prev_hard_sl > 0:
            return max(new_sl, prev_hard_sl)
        return new_sl
    elif regime == 'BEAR':
        new_sl = round(master_st_value + buffer, 1)
        # Trail down only — never move SL up in BEAR
        if prev_hard_sl > 0:
            return min(new_sl, prev_hard_sl)
        return new_sl
    return 0.0


def get_otm_strike(spot_price: float, direction: str, strike_interval: int = 100,
                   otm_strikes: int = 1) -> float:
    """
    Get the immediate next OTM strike.

    For CALL (direction='CE'): next strike above spot
    For PUT (direction='PE'): next strike below spot

    Example: spot=7030, strike_interval=100
    - CE: 7100 (1 strike OTM)
    - PE: 7000 (1 strike OTM)
    """
    if direction == 'CE':
        # Round up to next strike
        base_strike = int(np.ceil(spot_price / strike_interval)) * strike_interval
        return float(base_strike + (otm_strikes - 1) * strike_interval)
    else:  # PE
        # Round down to previous strike
        base_strike = int(np.floor(spot_price / strike_interval)) * strike_interval
        return float(base_strike - (otm_strikes - 1) * strike_interval)


def get_protective_strike(spot_price: float, direction: str, otm_pct: float = 0.05,
                          strike_interval: int = 100) -> float:
    """
    Get far OTM strike for protective option (5% OTM).

    For protective PUT (bull regime): 5% below spot
    For protective CALL (bear regime): 5% above spot
    """
    if direction == 'PE':
        target = spot_price * (1 - otm_pct)
        return float(int(np.floor(target / strike_interval)) * strike_interval)
    else:  # CE
        target = spot_price * (1 + otm_pct)
        return float(int(np.ceil(target / strike_interval)) * strike_interval)


def determine_actions(signal: MaruthiSignal, current_regime: str,
                      active_futures_count: int, max_futures: int = 5,
                      config: dict = None) -> List[dict]:
    """
    Determine what actions to take based on a signal.

    Returns list of action dicts like:
    [
        {'action': 'BUY_FUTURES', 'trigger_price': 7050, 'limit_price': 7055, ...},
        {'action': 'SHORT_CALL', 'strike': 7100, ...},
        {'action': 'CLOSE_ALL_FUTURES', ...},
        {'action': 'BUY_PROTECTIVE_PUT', 'strike': 6700, ...},
    ]
    """
    cfg = config or {}
    strike_interval = cfg.get('strike_interval', 100)
    otm_strikes = cfg.get('option_otm_strikes', 1)
    otm_pct = cfg.get('protective_otm_pct', 0.05)
    hard_sl_buffer = cfg.get('hard_sl_buffer', 50)
    actions = []
    spot = signal.candle['close']

    if signal.signal_type == MaruthiSignal.MASTER_BULL:
        # New BULL regime
        # 1. Close all existing positions (handled by executor)
        actions.append({'action': 'REGIME_CHANGE', 'new_regime': 'BULL'})

        # 2. Buy futures with trigger above candle high
        trigger = compute_trigger_price(signal.candle, 'BUY')
        limit = compute_limit_price(trigger, 'BUY')
        hard_sl = compute_hard_sl(signal.master_st, 'BULL', hard_sl_buffer)
        actions.append({
            'action': 'BUY_FUTURES',
            'trigger_price': trigger,
            'limit_price': limit,
            'hard_sl': hard_sl,
            'master_st': signal.master_st,
        })

    elif signal.signal_type == MaruthiSignal.MASTER_BEAR:
        # New BEAR regime
        actions.append({'action': 'REGIME_CHANGE', 'new_regime': 'BEAR'})

        # Short futures with trigger below candle low
        trigger = compute_trigger_price(signal.candle, 'SELL')
        limit = compute_limit_price(trigger, 'SELL')
        hard_sl = compute_hard_sl(signal.master_st, 'BEAR', hard_sl_buffer)
        actions.append({
            'action': 'SHORT_FUTURES',
            'trigger_price': trigger,
            'limit_price': limit,
            'hard_sl': hard_sl,
            'master_st': signal.master_st,
        })

    elif signal.signal_type == MaruthiSignal.CHILD_BULL:
        if current_regime == 'BULL':
            # Add long futures (if under max)
            if active_futures_count < max_futures:
                trigger = compute_trigger_price(signal.candle, 'BUY')
                limit = compute_limit_price(trigger, 'BUY')
                hard_sl = compute_hard_sl(signal.master_st, 'BULL', hard_sl_buffer)
                actions.append({
                    'action': 'BUY_FUTURES',
                    'trigger_price': trigger,
                    'limit_price': limit,
                    'hard_sl': hard_sl,
                    'master_st': signal.master_st,
                })
        elif current_regime == 'BEAR':
            # Short OTM put
            strike = get_otm_strike(spot, 'PE', strike_interval, otm_strikes)
            actions.append({
                'action': 'SHORT_PUT',
                'strike': strike,
            })

    elif signal.signal_type == MaruthiSignal.CHILD_BEAR:
        if current_regime == 'BULL':
            # Short OTM call
            strike = get_otm_strike(spot, 'CE', strike_interval, otm_strikes)
            actions.append({
                'action': 'SHORT_CALL',
                'strike': strike,
            })
        elif current_regime == 'BEAR':
            # Add short futures (if under max)
            if active_futures_count < max_futures:
                trigger = compute_trigger_price(signal.candle, 'SELL')
                limit = compute_limit_price(trigger, 'SELL')
                hard_sl = compute_hard_sl(signal.master_st, 'BEAR', hard_sl_buffer)
                actions.append({
                    'action': 'SHORT_FUTURES',
                    'trigger_price': trigger,
                    'limit_price': limit,
                    'hard_sl': hard_sl,
                    'master_st': signal.master_st,
                })

    elif signal.signal_type == MaruthiSignal.HARD_SL:
        # Go flat — close everything except last short option
        actions.append({'action': 'HARD_SL_EXIT', 'regime': 'FLAT'})

    return actions
