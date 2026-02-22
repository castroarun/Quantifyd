"""
Darvas Box Detection for Momentum + Quality Strategy
=====================================================

Implements Nicolas Darvas's Box Theory for detecting consolidation boxes
and breakout signals. Used for topup decisions on existing portfolio positions.

Darvas Box formation rules:
1. Stock makes a new high (N-day high, configurable)
2. Box TOP confirmed: 3 consecutive days fail to exceed that high
3. Box BOTTOM confirmed: 3 consecutive days fail to break the low
4. If price breaks above the top during bottom formation → restart
5. BREAKOUT: Close > box top with volume > multiplier × average

Key difference from simple consolidation:
- Darvas Boxes only form after NEW HIGHS (uptrend context)
- They use strict 3-day confirmation (not arbitrary windows)
- Boxes stack → each new box creates a higher trailing stop
- Volume confirmation is specifically on the breakout bar

Reference: Nicolas Darvas, "How I Made $2,000,000 in the Stock Market" (1960)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DarvasBox:
    """A confirmed Darvas Box with ceiling and floor."""
    top: float
    bottom: float
    top_date: datetime = None
    bottom_date: datetime = None
    formation_start: datetime = None  # Date of the new high that started this box

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def height_pct(self) -> float:
        return self.height / self.bottom * 100 if self.bottom > 0 else 0

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2

    def __repr__(self):
        return (f"DarvasBox(top={self.top:.2f}, bottom={self.bottom:.2f}, "
                f"height={self.height_pct:.1f}%, "
                f"top_date={self.top_date}, bottom_date={self.bottom_date})")


@dataclass
class DarvasBreakout:
    """A detected breakout above a Darvas Box."""
    symbol: str
    date: datetime
    price: float  # Close price on breakout day
    box: DarvasBox
    volume: float
    avg_volume: float
    volume_ratio: float
    stop_loss: float  # Box bottom

    @property
    def breakout_pct(self) -> float:
        """How far above box top (%)."""
        return (self.price - self.box.top) / self.box.top * 100 if self.box.top > 0 else 0

    @property
    def volume_confirmed(self) -> bool:
        return self.volume_ratio >= 1.5

    def __repr__(self):
        return (f"DarvasBreakout({self.symbol} @ {self.price:.2f}, "
                f"+{self.breakout_pct:.1f}% above box, "
                f"vol={self.volume_ratio:.1f}x, stop={self.stop_loss:.2f})")


@dataclass
class DarvasState:
    """Tracks the Darvas Box state machine for a single stock."""
    symbol: str
    state: str = 'SCANNING'  # SCANNING, FORMING_TOP, FORMING_BOTTOM, BOX_COMPLETE
    candidate_top: float = 0.0
    candidate_bottom: float = float('inf')
    days_since_top: int = 0
    days_since_bottom: int = 0
    current_box: Optional[DarvasBox] = None
    completed_boxes: List[DarvasBox] = field(default_factory=list)
    breakouts: List[DarvasBreakout] = field(default_factory=list)
    formation_start_date: datetime = None


# =============================================================================
# Backward-compatible aliases (used by monitoring agent, etc.)
# =============================================================================

@dataclass
class ConsolidationZone:
    """Backward-compatible wrapper around DarvasBox detection."""
    symbol: str
    is_consolidating: bool
    start_date: Optional[datetime] = None
    days_in_range: int = 0
    range_high: float = 0.0
    range_low: float = 0.0
    range_pct: float = 0.0
    local_highs: int = 0
    local_lows: int = 0
    current_price: float = 0.0
    as_of_date: Optional[datetime] = None
    darvas_box: Optional[DarvasBox] = None

    @property
    def range_midpoint(self) -> float:
        return (self.range_high + self.range_low) / 2 if self.range_high > 0 else 0

    @property
    def breakout_level(self) -> float:
        return self.range_high


@dataclass
class BreakoutSignal:
    """Backward-compatible wrapper around DarvasBreakout."""
    symbol: str
    is_breakout: bool
    breakout_date: Optional[datetime] = None
    breakout_price: float = 0.0
    range_high: float = 0.0
    breakout_pct: float = 0.0
    volume: float = 0.0
    avg_volume_20d: float = 0.0
    volume_ratio: float = 0.0
    consolidation: Optional[ConsolidationZone] = None
    darvas_breakout: Optional[DarvasBreakout] = None

    @property
    def volume_confirmed(self) -> bool:
        return self.volume_ratio >= 1.5


@dataclass
class ConsolidationScreenResult:
    """Result of screening multiple stocks."""
    as_of_date: datetime
    total_scanned: int
    consolidating: List[ConsolidationZone] = field(default_factory=list)
    breakouts: List[BreakoutSignal] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Darvas Box Detector (Core Algorithm)
# =============================================================================

class DarvasBoxDetector:
    """
    Implements the Darvas Box state machine.

    Processes OHLCV bars sequentially to detect boxes and breakouts.

    States:
        SCANNING       → Looking for a new N-day high
        FORMING_TOP    → New high found, waiting 3 days confirmation
        FORMING_BOTTOM → Top confirmed, waiting 3 days for floor
        BOX_COMPLETE   → Box formed, watching for breakout

    Args:
        confirmation_days: Days without new extreme to confirm (default 3)
        new_high_lookback: N-day period for "new high" check (default 20)
        volume_multiplier: Breakout volume vs average (default 1.5)
        volume_lookback: Days for average volume calc (default 50)
        min_box_height_pct: Minimum box height as % (default 1.5)
        max_box_height_pct: Maximum box height as % (default 20.0)
    """

    def __init__(
        self,
        confirmation_days: int = 3,
        new_high_lookback: int = 20,
        volume_multiplier: float = 1.5,
        volume_lookback: int = 50,
        min_box_height_pct: float = 1.5,
        max_box_height_pct: float = 20.0,
    ):
        self.confirmation_days = confirmation_days
        self.new_high_lookback = new_high_lookback
        self.volume_multiplier = volume_multiplier
        self.volume_lookback = volume_lookback
        self.min_box_height_pct = min_box_height_pct
        self.max_box_height_pct = max_box_height_pct

    def detect_boxes(self, df: pd.DataFrame, symbol: str = '') -> DarvasState:
        """
        Process a full OHLCV DataFrame and detect all Darvas Boxes + breakouts.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be DatetimeIndex or have a 'date' column.
            symbol: Stock symbol for logging

        Returns:
            DarvasState with all detected boxes and breakouts
        """
        if len(df) < self.new_high_lookback + self.confirmation_days:
            return DarvasState(symbol=symbol)

        # Normalize index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)

        state = DarvasState(symbol=symbol)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        dates = df.index

        # Rolling volume average
        vol_avg = pd.Series(volumes).rolling(
            window=self.volume_lookback, min_periods=20
        ).mean().values

        for i in range(self.new_high_lookback, len(df)):
            date = dates[i]
            high = highs[i]
            low = lows[i]
            close = closes[i]
            volume = volumes[i]
            avg_vol = vol_avg[i] if not np.isnan(vol_avg[i]) else 0

            # Track the rolling N-day high for "new high" detection
            lookback_start = max(0, i - self.new_high_lookback)
            rolling_high = highs[lookback_start:i].max()

            if state.state == 'SCANNING':
                self._handle_scanning(state, date, high, rolling_high)

            elif state.state == 'FORMING_TOP':
                self._handle_forming_top(state, date, high)

            elif state.state == 'FORMING_BOTTOM':
                self._handle_forming_bottom(state, date, high, low)

            elif state.state == 'BOX_COMPLETE':
                self._handle_box_complete(
                    state, date, high, low, close, volume, avg_vol,
                    rolling_high, symbol
                )

        return state

    def _handle_scanning(self, state: DarvasState, date, high, rolling_high):
        """Look for a new N-day high to start box formation."""
        if high > rolling_high:
            state.candidate_top = high
            state.days_since_top = 0
            state.formation_start_date = date
            state.state = 'FORMING_TOP'

    def _handle_forming_top(self, state: DarvasState, date, high):
        """Wait for confirmation_days without a new high."""
        if high > state.candidate_top:
            # New higher high → reset counter
            state.candidate_top = high
            state.days_since_top = 0
        else:
            state.days_since_top += 1

        if state.days_since_top >= self.confirmation_days:
            # Box TOP confirmed → start looking for bottom
            state.candidate_bottom = float('inf')
            state.days_since_bottom = 0
            state.state = 'FORMING_BOTTOM'

    def _handle_forming_bottom(self, state: DarvasState, date, high, low):
        """Wait for confirmation_days without a new low. Invalidate if top broken."""
        # Invalidation: price breaks above the confirmed top
        if high > state.candidate_top:
            # Top invalidated → new high becomes new candidate
            state.candidate_top = high
            state.days_since_top = 0
            state.candidate_bottom = float('inf')
            state.state = 'FORMING_TOP'
            return

        # Track the lowest low
        if low < state.candidate_bottom:
            state.candidate_bottom = low
            state.days_since_bottom = 0
        else:
            state.days_since_bottom += 1

        if state.days_since_bottom >= self.confirmation_days:
            # Box BOTTOM confirmed → check box validity
            box_height_pct = (
                (state.candidate_top - state.candidate_bottom)
                / state.candidate_bottom * 100
            ) if state.candidate_bottom > 0 else 0

            if (self.min_box_height_pct <= box_height_pct <= self.max_box_height_pct):
                state.current_box = DarvasBox(
                    top=state.candidate_top,
                    bottom=state.candidate_bottom,
                    top_date=date,  # Approximate: confirmation date
                    bottom_date=date,
                    formation_start=state.formation_start_date,
                )
                state.state = 'BOX_COMPLETE'
            else:
                # Box too thin or too wide → restart
                state.state = 'SCANNING'

    def _handle_box_complete(
        self, state: DarvasState, date, high, low, close, volume, avg_vol,
        rolling_high, symbol
    ):
        """Watch for breakout above box top or failure below box bottom."""
        box = state.current_box

        # BREAKOUT: Close above box top
        if close > box.top:
            vol_ratio = volume / avg_vol if avg_vol > 0 else 0

            if vol_ratio >= self.volume_multiplier:
                # Confirmed breakout
                breakout = DarvasBreakout(
                    symbol=symbol,
                    date=date,
                    price=close,
                    box=box,
                    volume=volume,
                    avg_volume=avg_vol,
                    volume_ratio=round(vol_ratio, 2),
                    stop_loss=box.bottom,
                )
                state.breakouts.append(breakout)
                state.completed_boxes.append(box)
                state.current_box = None
                # After breakout, look for the next box
                state.candidate_top = high
                state.days_since_top = 0
                state.formation_start_date = date
                state.state = 'FORMING_TOP'
            else:
                # Price broke out but no volume → still complete, or start new box
                state.completed_boxes.append(box)
                state.current_box = None
                state.candidate_top = high
                state.days_since_top = 0
                state.formation_start_date = date
                state.state = 'FORMING_TOP'
            return

        # BOTTOM BREACH: Close below box bottom
        # Instead of failing the box entirely, re-form the bottom at a lower
        # level. This captures wide consolidation bases (e.g., stock hits 720,
        # drops to 680, then to 650, then consolidates → breakout through 720).
        # The box top stays the same; only the floor widens.
        if close < box.bottom:
            # Check if the box would become too tall (> max_box_height_pct)
            new_height_pct = (box.top - low) / low * 100 if low > 0 else 0
            if new_height_pct > self.max_box_height_pct:
                # Box too wide → truly failed
                state.current_box = None
                state.state = 'SCANNING'
                return
            # Re-enter FORMING_BOTTOM to find the true floor
            state.candidate_bottom = low
            state.days_since_bottom = 0
            state.current_box = None
            state.state = 'FORMING_BOTTOM'
            return

        # Still inside box → keep waiting


# =============================================================================
# Flat-Range Consolidation Detector
# =============================================================================

@dataclass
class FlatConsolidation:
    """A detected flat-range consolidation zone."""
    symbol: str
    start_date: datetime
    end_date: datetime
    days: int
    range_high: float
    range_low: float
    range_pct: float       # (high - low) / low * 100
    std_dev_pct: float     # std dev of closes / mean close * 100
    avg_volume: float

    @property
    def midpoint(self) -> float:
        return (self.range_high + self.range_low) / 2

    def __repr__(self):
        return (f"FlatConsolidation({self.symbol} [{self.range_low:.2f}-{self.range_high:.2f}] "
                f"{self.days}d, range={self.range_pct:.1f}%, stddev={self.std_dev_pct:.1f}%)")


@dataclass
class FlatBreakout:
    """A breakout from a flat consolidation zone."""
    symbol: str
    date: datetime
    price: float           # Close on breakout day
    zone: FlatConsolidation
    volume: float
    avg_volume: float
    volume_ratio: float
    stop_loss: float       # Zone low

    @property
    def breakout_pct(self) -> float:
        return (self.price - self.zone.range_high) / self.zone.range_high * 100 if self.zone.range_high > 0 else 0

    def __repr__(self):
        return (f"FlatBreakout({self.symbol} @ {self.price:.2f}, "
                f"+{self.breakout_pct:.1f}% above range, vol={self.volume_ratio:.1f}x)")


@dataclass
class FlatConsolidationResult:
    """All consolidation zones and breakouts for a stock."""
    symbol: str
    zones: List[FlatConsolidation] = field(default_factory=list)
    breakouts: List[FlatBreakout] = field(default_factory=list)


class FlatRangeDetector:
    """
    Detects flat sideways consolidation zones and breakouts.

    Unlike Darvas (which needs a new high to start), this detector looks for
    periods where price trades in a tight range with low volatility.

    Algorithm:
        1. Slide a window across the price data
        2. At each position, check if the last N days form a tight range
        3. Extend the zone as long as price stays within the range
        4. When close breaks above range high with volume → breakout

    Args:
        min_days: Minimum days in consolidation (default 20)
        max_range_pct: Maximum price range as % (default 12.0)
        max_stddev_pct: Maximum std dev of closes as % of mean (default 3.0)
        volume_multiplier: Breakout volume vs average (default 1.5)
        volume_lookback: Days for average volume (default 50)
    """

    def __init__(
        self,
        min_days: int = 20,
        max_range_pct: float = 12.0,
        max_stddev_pct: float = 3.0,
        volume_multiplier: float = 1.5,
        volume_lookback: int = 50,
    ):
        self.min_days = min_days
        self.max_range_pct = max_range_pct
        self.max_stddev_pct = max_stddev_pct
        self.volume_multiplier = volume_multiplier
        self.volume_lookback = volume_lookback

    def detect(self, df: pd.DataFrame, symbol: str = '') -> FlatConsolidationResult:
        """
        Process OHLCV DataFrame to detect flat consolidation zones and breakouts.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be DatetimeIndex.
            symbol: Stock symbol for logging

        Returns:
            FlatConsolidationResult with zones and breakouts
        """
        if len(df) < self.min_days + 10:
            return FlatConsolidationResult(symbol=symbol)

        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            df.index = pd.to_datetime(df.index)

        result = FlatConsolidationResult(symbol=symbol)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        dates = df.index

        vol_avg = pd.Series(volumes).rolling(
            window=self.volume_lookback, min_periods=20
        ).mean().values

        in_zone = False
        zone_start_idx = 0
        zone_high = 0.0
        zone_low = float('inf')
        cooldown_until = 0  # Skip bars after a breakout

        i = self.min_days
        while i < len(df):
            if i < cooldown_until:
                i += 1
                continue

            if not in_zone:
                # Check if the last min_days form a consolidation
                window_highs = highs[i - self.min_days:i]
                window_lows = lows[i - self.min_days:i]
                window_closes = closes[i - self.min_days:i]

                w_high = window_highs.max()
                w_low = window_lows.min()
                w_range_pct = (w_high - w_low) / w_low * 100 if w_low > 0 else 999
                w_mean = window_closes.mean()
                w_stddev_pct = (window_closes.std() / w_mean * 100) if w_mean > 0 else 999

                if w_range_pct <= self.max_range_pct and w_stddev_pct <= self.max_stddev_pct:
                    in_zone = True
                    zone_start_idx = i - self.min_days
                    zone_high = w_high
                    zone_low = w_low
                else:
                    i += 1
                    continue

            # We're in a zone - check if current bar extends it or breaks out
            close = closes[i]
            high = highs[i]
            low = lows[i]
            volume = volumes[i]
            avg_vol = vol_avg[i] if not np.isnan(vol_avg[i]) else 0

            # Breakout above range high?
            if close > zone_high:
                vol_ratio = volume / avg_vol if avg_vol > 0 else 0
                zone_days = i - zone_start_idx

                zone = FlatConsolidation(
                    symbol=symbol,
                    start_date=dates[zone_start_idx],
                    end_date=dates[i - 1],
                    days=zone_days,
                    range_high=round(zone_high, 2),
                    range_low=round(zone_low, 2),
                    range_pct=round((zone_high - zone_low) / zone_low * 100, 1) if zone_low > 0 else 0,
                    std_dev_pct=round(closes[zone_start_idx:i].std() / closes[zone_start_idx:i].mean() * 100, 1),
                    avg_volume=round(volumes[zone_start_idx:i].mean(), 0),
                )
                result.zones.append(zone)

                if vol_ratio >= self.volume_multiplier:
                    breakout = FlatBreakout(
                        symbol=symbol,
                        date=dates[i],
                        price=round(close, 2),
                        zone=zone,
                        volume=volume,
                        avg_volume=avg_vol,
                        volume_ratio=round(vol_ratio, 2),
                        stop_loss=round(zone_low, 2),
                    )
                    result.breakouts.append(breakout)

                in_zone = False
                cooldown_until = i + 5  # Skip a few bars after breakout
                i += 1
                continue

            # Break below range low → zone failed
            if close < zone_low:
                zone_days = i - zone_start_idx
                zone = FlatConsolidation(
                    symbol=symbol,
                    start_date=dates[zone_start_idx],
                    end_date=dates[i - 1],
                    days=zone_days,
                    range_high=round(zone_high, 2),
                    range_low=round(zone_low, 2),
                    range_pct=round((zone_high - zone_low) / zone_low * 100, 1) if zone_low > 0 else 0,
                    std_dev_pct=round(closes[zone_start_idx:i].std() / closes[zone_start_idx:i].mean() * 100, 1),
                    avg_volume=round(volumes[zone_start_idx:i].mean(), 0),
                )
                result.zones.append(zone)
                in_zone = False
                i += 1
                continue

            # Still inside → zone continues, update high/low if needed
            # (don't expand the range - that would make it not "flat")
            i += 1

        return result


# =============================================================================
# Breakout Entry Filter Presets
# Derived from optimization across 9,739 trades / 457 stocks (2000-2025)
# See docs/BREAKOUT-FILTER-OPTIMIZATION.md for full analysis
# =============================================================================

# V1 Filters (original analysis - simple threshold combos)
BREAKOUT_FILTER_ALPHA = {
    'min_rsi14': 75,              # RSI(14) >= 75 at breakout
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
    # 73.7% win rate | PF 10.37 | Calmar 1.48 | ~1.5 signals/year
}

# V2 Filters (enhanced analysis - 53 indicators + top-down weekly approach)
BREAKOUT_FILTER_TIER1A = {
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 90,      # Within 10% of all-time high
    'require_weekly_ema20_gt_50': True,  # Weekly EMA(20) > EMA(50)
    'min_rsi7': 80,               # RSI(7) >= 80 (short-term overbought)
    # 67.5% win rate | PF 6.54 | Calmar 3.23 | ~6 signals/year
}

BREAKOUT_FILTER_TIER1B = {
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 85,      # Within 15% of all-time high
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
    'require_weekly_ema20_gt_50': True,  # Weekly EMA(20) > EMA(50)
    'min_williams_r': -20,        # Williams %R >= -20 (overbought zone)
    # 65.9% win rate | PF 6.32 | Calmar 1.70 | ~9 signals/year
}

BREAKOUT_FILTER_TIER2 = {
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'min_mom_10d': 5.0,           # 10-day momentum >= 5%
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 85,      # Within 15% of all-time high
    'require_weekly_ema20_gt_50': True,  # Weekly EMA(20) > EMA(50)
    # 64.5% win rate | PF 5.91 | Calmar 1.78 | ~10 signals/year
}

BREAKOUT_FILTER_TIER3 = {
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 85,      # Within 15% of all-time high
    # 58.5% win rate | PF 4.54 | Calmar 1.94 | ~21 signals/year
}

BREAKOUT_FILTER_CALMAR = {
    'min_volume_ratio': 5.0,      # Breakout volume >= 5x 50-day average
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
    'min_rsi7': 80,               # RSI(7) >= 80 (short-term overbought)
    # 64.3% win rate | PF 5.05 | Calmar 4.66 | ~8 signals/year
}

# Legacy alias for backward compatibility
BREAKOUT_FILTER_TIER1 = BREAKOUT_FILTER_TIER1A


# =============================================================================
# V3 Multi-Strategy OR System (Phase 3 deep-dive analysis)
# A trade is taken when ANY component strategy fires (OR logic).
# See docs/BREAKOUT-FILTER-OPTIMIZATION.md sections 14-18 for full analysis.
# =============================================================================

# Component strategies (each is a standalone AND-filter set)
STRATEGY_ALPHA = {
    'min_rsi14': 75,              # RSI(14) >= 75 at breakout
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
    'require_ema20_gt_50': True,  # EMA(20) > EMA(50) at breakout
    # 73.7% win | PF 10.37 | 38 trades
}

STRATEGY_T1A = {
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 90,      # Within 10% of all-time high
    'require_weekly_ema20_gt_50': True,
    'min_rsi7': 80,               # RSI(7) >= 80
    # 67.5% win | PF 6.54 | 151 trades
}

STRATEGY_T1B = {
    'min_breakout_pct': 3.0,      # Close >= 3% above consolidation high
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 85,      # Within 15% of all-time high
    'require_ema20_gt_50': True,
    'require_weekly_ema20_gt_50': True,
    'min_williams_r': -20,        # Williams %R >= -20
    # 65.9% win | PF 6.32 | 220 trades
}

STRATEGY_MOMVOL = {
    'min_mom_60d': 15.0,          # 60-day momentum >= 15%
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 90,      # Within 10% of all-time high
    'min_volume_ratio': 3.0,      # Breakout volume >= 3x 50-day average
    # 67.9% win | PF 7.12 | 134 trades
}

STRATEGY_CALMAR = {
    'min_volume_ratio': 5.0,      # Breakout volume >= 5x 50-day average
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'require_ema20_gt_50': True,
    'min_rsi7': 80,               # RSI(7) >= 80
    # 64.3% win | PF 5.05 | Calmar 4.66 | 210 trades
}

STRATEGY_BB_MOM = {
    'min_bb_pct_b': 1.0,          # Above Bollinger upper band
    'min_mom_60d': 15.0,          # 60-day momentum >= 15%
    'min_vol_trend': 1.2,         # 10d avg vol / 50d avg vol >= 1.2
    'min_ath_proximity': 90,      # Within 10% of all-time high
    # 61.6% win | PF 4.80 | 258 trades
}

# System presets: lists of strategies combined with OR logic
SYSTEM_SNIPER = [STRATEGY_ALPHA, STRATEGY_MOMVOL]
# 169 trades, 69.8% win, PF 8.01, Calmar 1.71, ~7/year

SYSTEM_PRIMARY = [STRATEGY_ALPHA, STRATEGY_T1B, STRATEGY_MOMVOL]
# 332 trades, 66.9% win, PF 6.44, Calmar 1.96, ~13/year (RECOMMENDED)

SYSTEM_BALANCED = [STRATEGY_ALPHA, STRATEGY_T1A, STRATEGY_CALMAR]
# 325 trades, 65.2% win, PF 5.46, Calmar 5.19, ~13/year (best risk-adjusted)

SYSTEM_ACTIVE = [STRATEGY_ALPHA, STRATEGY_T1B, STRATEGY_CALMAR]
# 393 trades, 65.4% win, PF 5.73, Calmar 2.52, ~16/year

SYSTEM_HIGH_VOLUME = [STRATEGY_T1B, STRATEGY_CALMAR, STRATEGY_BB_MOM]
# 535 trades, 62.4% win, PF 5.12, Calmar 3.01, ~21/year


def check_strategy_match(trade_data: dict, strategy: dict) -> bool:
    """Check if a trade matches a single strategy's filter criteria."""
    checks = {
        'min_rsi14': lambda d, v: d.get('rsi14', 0) >= v,
        'min_rsi7': lambda d, v: d.get('rsi7', 0) >= v,
        'min_volume_ratio': lambda d, v: d.get('volume_ratio', 0) >= v,
        'min_vol_trend': lambda d, v: d.get('vol_trend', 0) >= v,
        'min_breakout_pct': lambda d, v: d.get('breakout_pct', 0) >= v,
        'min_ath_proximity': lambda d, v: d.get('ath_proximity', 0) >= v,
        'min_williams_r': lambda d, v: d.get('williams_r', -100) >= v,
        'min_mom_60d': lambda d, v: d.get('mom_60d', 0) >= v,
        'min_mom_10d': lambda d, v: d.get('mom_10d', 0) >= v,
        'min_bb_pct_b': lambda d, v: d.get('bb_pct_b', 0) >= v,
        'require_ema20_gt_50': lambda d, v: d.get('ema20_above_50', 0) == 1 if v else True,
        'require_weekly_ema20_gt_50': lambda d, v: d.get('w_ema20_gt_50', 0) == 1 if v else True,
    }
    for key, value in strategy.items():
        checker = checks.get(key)
        if checker and not checker(trade_data, value):
            return False
    return True


def check_system_match(trade_data: dict, system: list) -> bool:
    """Check if a trade matches ANY strategy in a system (OR logic)."""
    return any(check_strategy_match(trade_data, s) for s in system)


# =============================================================================
# High-level API (backward-compatible with existing callers)
# =============================================================================

_detector_cache = {}


def detect_darvas_boxes(
    symbol: str,
    price_data: pd.DataFrame,
    confirmation_days: int = 3,
    new_high_lookback: int = 20,
    volume_multiplier: float = 1.5,
    min_box_height_pct: float = 1.5,
    max_box_height_pct: float = 20.0,
) -> DarvasState:
    """
    Detect all Darvas Boxes and breakouts for a stock.

    Args:
        symbol: Stock symbol
        price_data: OHLCV DataFrame (DatetimeIndex, columns: open/high/low/close/volume)
        confirmation_days: Days to confirm box top/bottom (default 3)
        new_high_lookback: N-day lookback for "new high" (default 20)
        volume_multiplier: Required volume ratio for breakout (default 1.5)
        min_box_height_pct: Min box height % (default 1.5)
        max_box_height_pct: Max box height % (default 20.0)

    Returns:
        DarvasState with detected boxes and breakouts
    """
    detector = DarvasBoxDetector(
        confirmation_days=confirmation_days,
        new_high_lookback=new_high_lookback,
        volume_multiplier=volume_multiplier,
        min_box_height_pct=min_box_height_pct,
        max_box_height_pct=max_box_height_pct,
    )
    return detector.detect_boxes(price_data, symbol)


def detect_consolidation(
    symbol: str,
    as_of_date: datetime = None,
    lookback_days: int = 60,
    min_consolidation_days: int = 20,
    max_range_pct: float = 0.05,
    min_local_extrema: int = 2,
    db_path=None,
    price_data: pd.DataFrame = None,
) -> ConsolidationZone:
    """
    Backward-compatible API: Detect consolidation using Darvas Box theory.

    If price_data is provided, uses it directly (for backtest engine).
    Otherwise falls back to DB lookup.
    """
    if price_data is None:
        import sqlite3
        from pathlib import Path
        db_path = db_path or Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'
        conn = sqlite3.connect(db_path)

        if as_of_date is None:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(date) FROM market_data_unified "
                "WHERE symbol = ? AND timeframe = 'day'", (symbol,)
            )
            row = cursor.fetchone()
            if not row or not row[0]:
                conn.close()
                return ConsolidationZone(symbol=symbol, is_consolidating=False)
            as_of_date = datetime.strptime(row[0][:10], '%Y-%m-%d')

        from datetime import timedelta
        start = as_of_date - timedelta(days=int(lookback_days * 2))
        query = """
            SELECT date, open, high, low, close, volume
            FROM market_data_unified
            WHERE symbol = ? AND timeframe = 'day'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        price_data = pd.read_sql_query(
            query, conn,
            params=(symbol, start.strftime('%Y-%m-%d'), as_of_date.strftime('%Y-%m-%d'))
        )
        conn.close()

        if len(price_data) < 20:
            return ConsolidationZone(symbol=symbol, is_consolidating=False,
                                     as_of_date=as_of_date)

        price_data['date'] = pd.to_datetime(price_data['date'])
        price_data = price_data.set_index('date')

    darvas = detect_darvas_boxes(symbol, price_data)
    current_price = price_data['close'].iloc[-1] if len(price_data) > 0 else 0

    # A stock is "consolidating" if it has a complete box waiting for breakout
    if darvas.state == 'BOX_COMPLETE' and darvas.current_box is not None:
        box = darvas.current_box
        return ConsolidationZone(
            symbol=symbol,
            is_consolidating=True,
            start_date=box.formation_start,
            days_in_range=0,  # Not directly applicable
            range_high=round(box.top, 2),
            range_low=round(box.bottom, 2),
            range_pct=round(box.height_pct / 100, 4),
            current_price=round(current_price, 2),
            as_of_date=as_of_date,
            darvas_box=box,
        )

    return ConsolidationZone(
        symbol=symbol,
        is_consolidating=False,
        current_price=round(current_price, 2),
        as_of_date=as_of_date,
    )


def detect_breakout(
    symbol: str,
    consolidation: ConsolidationZone = None,
    as_of_date: datetime = None,
    volume_multiplier: float = 1.5,
    breakout_buffer: float = 0.0,  # Darvas uses no buffer (exact box top)
    db_path=None,
    price_data: pd.DataFrame = None,
) -> BreakoutSignal:
    """
    Backward-compatible API: Detect breakout using Darvas Box theory.

    In Darvas method, a breakout is close > box_top with volume confirmation.
    No 2% buffer needed — the box top IS the breakout level.
    """
    if price_data is None:
        import sqlite3
        from pathlib import Path
        db_path = db_path or Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'

        if consolidation is None:
            consolidation = detect_consolidation(symbol, as_of_date, db_path=db_path)

        if not consolidation.is_consolidating:
            return BreakoutSignal(symbol=symbol, is_breakout=False,
                                  consolidation=consolidation)

        conn = sqlite3.connect(db_path)
        from datetime import timedelta
        start = as_of_date - timedelta(days=80)
        query = """
            SELECT date, close, volume
            FROM market_data_unified
            WHERE symbol = ? AND timeframe = 'day'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        df = pd.read_sql_query(
            query, conn,
            params=(symbol, start.strftime('%Y-%m-%d'), as_of_date.strftime('%Y-%m-%d'))
        )
        conn.close()

        if len(df) < 5:
            return BreakoutSignal(symbol=symbol, is_breakout=False,
                                  consolidation=consolidation)

        latest = df.iloc[-1]
        close_price = latest['close']
        volume = latest['volume']
        avg_volume = df['volume'].iloc[-51:-1].mean() if len(df) > 50 else df['volume'].iloc[:-1].mean()
    else:
        # Use provided price data
        darvas = detect_darvas_boxes(symbol, price_data, volume_multiplier=volume_multiplier)

        if not darvas.breakouts:
            return BreakoutSignal(symbol=symbol, is_breakout=False)

        # Return the most recent breakout
        latest_bo = darvas.breakouts[-1]
        box = latest_bo.box

        zone = ConsolidationZone(
            symbol=symbol,
            is_consolidating=True,
            range_high=round(box.top, 2),
            range_low=round(box.bottom, 2),
            range_pct=round(box.height_pct / 100, 4),
            darvas_box=box,
        )

        return BreakoutSignal(
            symbol=symbol,
            is_breakout=True,
            breakout_date=latest_bo.date,
            breakout_price=round(latest_bo.price, 2),
            range_high=round(box.top, 2),
            breakout_pct=round(latest_bo.breakout_pct / 100, 4),
            volume=round(latest_bo.volume, 0),
            avg_volume_20d=round(latest_bo.avg_volume, 0),
            volume_ratio=round(latest_bo.volume_ratio, 2),
            consolidation=zone,
            darvas_breakout=latest_bo,
        )

    # DB-based path
    box_top = consolidation.range_high
    vol_ratio = volume / avg_volume if avg_volume > 0 else 0
    is_price_break = close_price > box_top
    is_vol_ok = vol_ratio >= volume_multiplier
    is_breakout = is_price_break and is_vol_ok

    return BreakoutSignal(
        symbol=symbol,
        is_breakout=is_breakout,
        breakout_date=as_of_date if is_breakout else None,
        breakout_price=round(close_price, 2),
        range_high=round(box_top, 2),
        breakout_pct=round((close_price - box_top) / box_top, 4) if box_top > 0 else 0,
        volume=round(volume, 0),
        avg_volume_20d=round(avg_volume, 0),
        volume_ratio=round(vol_ratio, 2),
        consolidation=consolidation,
    )


def screen_consolidation_breakout(
    symbols: List[str],
    as_of_date: datetime = None,
    min_consolidation_days: int = 20,
    max_range_pct: float = 0.05,
    volume_multiplier: float = 1.5,
    db_path=None,
) -> ConsolidationScreenResult:
    """
    Screen multiple stocks for Darvas Box consolidation and breakouts.
    Backward-compatible with monitoring agent.
    """
    consolidating = []
    breakouts = []
    errors = []

    for symbol in symbols:
        try:
            zone = detect_consolidation(
                symbol, as_of_date, db_path=db_path,
            )
            if zone.is_consolidating:
                consolidating.append(zone)
                signal = detect_breakout(
                    symbol, zone, as_of_date,
                    volume_multiplier=volume_multiplier,
                    db_path=db_path,
                )
                if signal.is_breakout:
                    breakouts.append(signal)
        except Exception as e:
            errors.append(f"{symbol}: {e}")

    eval_date = as_of_date or datetime.now()
    logger.info(
        f"Darvas screen: {len(consolidating)} with boxes, "
        f"{len(breakouts)} breakouts out of {len(symbols)} stocks "
        f"(date={eval_date.date() if isinstance(eval_date, datetime) else eval_date})"
    )

    return ConsolidationScreenResult(
        as_of_date=eval_date,
        total_scanned=len(symbols),
        consolidating=consolidating,
        breakouts=breakouts,
        errors=errors,
    )
