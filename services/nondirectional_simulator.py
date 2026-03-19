"""
Non-Directional Options Strategy Simulator
===========================================

Simulates weekly/bi-weekly short strangles and iron condors on NIFTY/BANKNIFTY
using range-detection signals for entry timing.

Premium Model:
- Uses simplified Black-Scholes approximation for OTM option pricing
- Accounts for theta decay over holding period
- Models loss on zone break as intrinsic value beyond strike

Strategies Simulated:
1. Short Strangle: Sell OTM Call + OTM Put at ±2.5x ATR
2. Iron Condor: Short Strangle + buy wings at ±4x ATR for protection

Capital & Position Sizing:
- Uses margin-based position sizing (SPAN margin approximation)
- BankNifty lot = 15, Nifty lot = 25
"""

import numpy as np
import pandas as pd
import calendar
from datetime import date, timedelta
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

LOT_SIZES = {'NIFTY50': 25, 'BANKNIFTY': 15}
# Approximate annualized IV for estimation (will be overridden by ATR-derived)
BASE_IV = {'NIFTY50': 0.14, 'BANKNIFTY': 0.18}
RISK_FREE_RATE = 0.065  # 6.5% India

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_DAY = 375  # NSE: 9:15 to 15:30


# =============================================================================
# Expiry Date Calculation
# =============================================================================

def get_last_thursday(year: int, month: int) -> date:
    """Get last Thursday of a given month (BankNifty/Nifty monthly expiry)."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:  # 3 = Thursday
        d -= timedelta(days=1)
    return d


def get_next_thursday(entry_date: date) -> date:
    """Get the next Thursday on or after entry_date (Nifty weekly expiry)."""
    days_ahead = (3 - entry_date.weekday()) % 7
    if days_ahead == 0 and entry_date.weekday() == 3:
        # Entry on Thursday itself — use next week's Thursday
        days_ahead = 7
    return entry_date + timedelta(days=days_ahead)


def get_monthly_expiry(entry_date: date) -> date:
    """Get the next monthly expiry (last Thursday) on or after entry_date."""
    expiry = get_last_thursday(entry_date.year, entry_date.month)
    if expiry <= entry_date:
        # Current month's expiry has passed, use next month
        if entry_date.month == 12:
            expiry = get_last_thursday(entry_date.year + 1, 1)
        else:
            expiry = get_last_thursday(entry_date.year, entry_date.month + 1)
    return expiry


def compute_dte(entry_date: date, symbol: str, min_dte_monthly: int = 15,
                nifty_next_week: bool = False) -> tuple:
    """
    Compute actual DTE based on entry date and symbol.

    Returns (dte_days, expiry_date)

    BankNifty (monthly only): If current month DTE < min_dte_monthly, use next month.
    Nifty (weekly available): Use next Thursday, or next-to-next Thursday if nifty_next_week=True.
    """
    if symbol == 'BANKNIFTY':
        expiry = get_monthly_expiry(entry_date)
        dte = (expiry - entry_date).days
        if dte < min_dte_monthly:
            # Option B: roll to next month
            if expiry.month == 12:
                expiry = get_last_thursday(expiry.year + 1, 1)
            else:
                expiry = get_last_thursday(expiry.year, expiry.month + 1)
            dte = (expiry - entry_date).days
        return dte, expiry
    else:
        # NIFTY50 — weekly expiry available
        expiry = get_next_thursday(entry_date)
        if nifty_next_week:
            # Skip to the Thursday after that (next week's expiry)
            expiry = expiry + timedelta(days=7)
        dte = (expiry - entry_date).days
        if dte < 1:
            dte = 1
        return dte, expiry


class StrategyType(Enum):
    SHORT_STRANGLE = 'short_strangle'
    IRON_CONDOR = 'iron_condor'


class TradeOutcome(Enum):
    MAX_PROFIT = 'max_profit'       # price stayed in zone
    PARTIAL_PROFIT = 'partial'      # minor breach, still profitable
    BREAKEVEN = 'breakeven'
    LOSS = 'loss'                   # zone breach, lost money
    STOPPED_OUT = 'stopped_out'     # hit max loss limit


# =============================================================================
# Black-Scholes Pricing
# =============================================================================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_iv_from_atr(atr_value, close_price, timeframe_bars_per_day):
    """
    Estimate implied volatility from ATR.
    ATR is roughly proportional to daily range which relates to realized vol.
    IV typically trades at 1.1-1.3x realized vol for indices.
    """
    # Daily ATR as fraction of price
    if timeframe_bars_per_day > 1:
        # Intraday: scale ATR to daily equivalent
        daily_atr_pct = (atr_value / close_price) * np.sqrt(timeframe_bars_per_day)
    else:
        daily_atr_pct = atr_value / close_price

    # ATR ≈ 1.2 * daily_std (empirical relationship)
    daily_std = daily_atr_pct / 1.2
    # Annualize
    realized_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    # IV premium over RV (typically 10-30% for indices)
    iv = realized_vol * 1.20
    return max(iv, 0.08)  # Floor at 8%


# =============================================================================
# Trade Dataclass
# =============================================================================

@dataclass
class Trade:
    trade_id: int
    entry_date: str
    exit_date: str
    symbol: str
    strategy: str  # 'short_strangle' or 'iron_condor'

    # Entry details
    entry_price: float         # Index level at entry
    call_strike: float         # Short call strike
    put_strike: float          # Short put strike
    call_wing: float = 0       # Long call strike (iron condor)
    put_wing: float = 0        # Long put strike (iron condor)

    # Premium
    call_premium_collected: float = 0
    put_premium_collected: float = 0
    call_wing_paid: float = 0
    put_wing_paid: float = 0
    total_premium: float = 0   # Net credit

    # ATR/IV at entry
    entry_atr: float = 0
    entry_iv: float = 0
    zone_width_pct: float = 0  # zone width as % of entry price

    # Exit details
    exit_price: float = 0      # Index level at exit
    max_price: float = 0       # Highest during trade
    min_price: float = 0       # Lowest during trade
    max_adverse_excursion: float = 0  # Worst breach %

    # P&L
    call_exit_value: float = 0
    put_exit_value: float = 0
    call_wing_exit: float = 0
    put_wing_exit: float = 0
    gross_pnl: float = 0       # Per lot in points
    net_pnl: float = 0         # Per lot after costs
    pnl_rupees: float = 0      # Total P&L in Rs
    lots: int = 1

    # Outcome
    outcome: str = ''
    days_held: int = 0
    zone_held: bool = False

    # DTE info
    dte_at_entry: int = 0
    expiry_date_used: str = ''

    # Signal info
    signals_active: str = ''


# =============================================================================
# Simulation Engine
# =============================================================================

@dataclass
class SimConfig:
    symbol: str = 'BANKNIFTY'
    strategy: StrategyType = StrategyType.SHORT_STRANGLE

    # Zone / Strike parameters
    strike_distance_atr: float = 2.5   # Short strikes at ±N * ATR
    wing_distance_atr: float = 4.0     # Long wings at ±N * ATR (iron condor)
    atr_period: int = 14

    # Holding period
    hold_bars: int = 5                 # 5 daily bars = 1 week

    # Signals required for entry
    require_bb_squeeze: bool = True
    require_atr_contract: bool = True
    bb_period: int = 10
    atr_contract_period: int = 14
    min_consensus: int = 0             # 0 = use specific signals above

    # Trend filter
    use_trend_filter: bool = False
    trend_sma_len: int = 20
    trend_atr_mult: float = 2.0       # Block if price > N ATR from SMA

    # Position sizing
    capital: float = 10_00_000         # Rs 10 Lakh
    max_risk_per_trade_pct: float = 2.0  # Max 2% capital risk per trade
    lots_per_trade: int = 1

    # Risk management
    max_loss_per_trade_pct: float = 3.0  # Stop out at 3% of capital
    min_days_between_trades: int = 5     # No overlapping trades

    # Costs
    brokerage_per_lot: float = 40      # Rs 40 per order (Zerodha)
    stt_pct: float = 0.05              # STT on sell side
    other_charges_pct: float = 0.05    # Exchange, SEBI, GST etc.
    slippage_points: float = 2         # Per leg

    # Timeframe info
    bars_per_day: float = 1            # 1 for daily, 6.25 for 60min, etc.

    # Option expiry modeling
    days_to_expiry_at_entry: int = 7   # Weekly options: 7 DTE (fallback if not dynamic)
    use_dynamic_dte: bool = False      # Compute real DTE from expiry calendar
    min_dte_monthly: int = 15          # BankNifty Option B: roll to next month if DTE < this
    nifty_next_week: bool = False      # Nifty: sell next week's expiry instead of this week's


class NonDirectionalSimulator:
    """
    Simulates non-directional options strategy with range-detection signals.
    """

    def __init__(self, config: SimConfig, df: pd.DataFrame):
        self.config = config
        self.df = df.copy()
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

    def _compute_indicators(self):
        """Pre-compute all needed indicators."""
        df = self.df
        c = self.config

        # ATR
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(c.atr_period).mean()

        # BB Squeeze signal
        bb_period = c.bb_period
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        bw = (2 * 2.0 * std) / sma * 100
        bw_ma = bw.rolling(bb_period).mean()
        df['bb_squeeze'] = (bw < bw_ma).astype(int)

        # ATR Contraction signal
        atr_c = c.atr_contract_period
        atr_vals = tr.rolling(atr_c).mean()
        atr_ma = atr_vals.rolling(atr_c * 2).mean()
        df['atr_contract'] = (atr_vals < atr_ma).astype(int)

        # Choppiness Index
        atr1_sum = tr.rolling(1).mean().rolling(14).sum()
        high_max = high.rolling(14).max()
        low_min = low.rolling(14).min()
        price_range = (high_max - low_min).replace(0, np.nan)
        df['chop'] = 100 * np.log10(atr1_sum / price_range) / np.log10(14)
        df['chop_signal'] = (df['chop'] > 61.8).astype(int)

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        plus_di = 100 * (plus_dm.rolling(14).mean() / df['atr'])
        minus_di = 100 * (minus_dm.rolling(14).mean() / df['atr'])
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.rolling(14).mean()
        df['adx_low'] = (df['adx'] < 25).astype(int)

        # RSI mid zone
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss_s.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df['rsi_mid'] = ((rsi > 40) & (rsi < 60)).astype(int)

        # Consensus score
        df['consensus'] = (df['bb_squeeze'] + df['atr_contract'] +
                          df['chop_signal'] + df['adx_low'] + df['rsi_mid'])

        # Trend filter: price distance from SMA in ATR units
        if c.use_trend_filter:
            trend_sma = close.rolling(c.trend_sma_len).mean()
            price_dist_atr = (close - trend_sma).abs() / df['atr']
            df['is_ranging'] = (price_dist_atr <= c.trend_atr_mult).astype(int)
        else:
            df['is_ranging'] = 1

        # Entry signal
        if c.min_consensus > 0:
            df['entry_signal'] = (df['consensus'] >= c.min_consensus).astype(int)
        else:
            # Use specific signal requirements
            sig = pd.Series(1, index=df.index)
            if c.require_bb_squeeze:
                sig = sig & df['bb_squeeze']
            if c.require_atr_contract:
                sig = sig & df['atr_contract']
            df['entry_signal'] = sig.astype(int)

        # Apply trend filter
        if c.use_trend_filter:
            df['entry_signal'] = df['entry_signal'] & df['is_ranging']

        # Implied volatility estimate
        df['est_iv'] = df.apply(
            lambda row: estimate_iv_from_atr(
                row['atr'], row['close'], c.bars_per_day
            ) if row['atr'] > 0 and not np.isnan(row['atr']) else 0.15,
            axis=1
        )

        self.df = df

    def _price_option(self, S, K, T_years, iv, option_type='call'):
        """Price a single option leg."""
        if option_type == 'call':
            return bs_call_price(S, K, T_years, RISK_FREE_RATE, iv)
        else:
            return bs_put_price(S, K, T_years, RISK_FREE_RATE, iv)

    def _compute_trade_pnl(self, trade: Trade, forward_df: pd.DataFrame):
        """Compute P&L for a trade given forward price data."""
        c = self.config

        entry_price = trade.entry_price
        iv = trade.entry_iv
        lot_size = LOT_SIZES.get(c.symbol, 25)

        # Time to expiry at entry (in years)
        dte = trade.dte_at_entry if trade.dte_at_entry > 0 else c.days_to_expiry_at_entry
        T_entry = dte / 365.0

        # Price options at entry
        call_premium = self._price_option(entry_price, trade.call_strike, T_entry, iv, 'call')
        put_premium = self._price_option(entry_price, trade.put_strike, T_entry, iv, 'put')

        trade.call_premium_collected = round(call_premium, 2)
        trade.put_premium_collected = round(put_premium, 2)

        if c.strategy == StrategyType.IRON_CONDOR:
            call_wing_cost = self._price_option(entry_price, trade.call_wing, T_entry, iv, 'call')
            put_wing_cost = self._price_option(entry_price, trade.put_wing, T_entry, iv, 'put')
            trade.call_wing_paid = round(call_wing_cost, 2)
            trade.put_wing_paid = round(put_wing_cost, 2)

        trade.total_premium = round(
            trade.call_premium_collected + trade.put_premium_collected -
            trade.call_wing_paid - trade.put_wing_paid, 2
        )

        # Track price path
        trade.max_price = forward_df['high'].max()
        trade.min_price = forward_df['low'].min()
        trade.exit_price = forward_df['close'].iloc[-1]
        trade.days_held = len(forward_df)

        # Check zone breach
        call_breached = trade.max_price > trade.call_strike
        put_breached = trade.min_price < trade.put_strike
        trade.zone_held = not (call_breached or put_breached)

        # Max adverse excursion
        upside_breach = max(0, (trade.max_price - trade.call_strike) / entry_price * 100)
        downside_breach = max(0, (trade.put_strike - trade.min_price) / entry_price * 100)
        trade.max_adverse_excursion = round(max(upside_breach, downside_breach), 2)

        # Price options at exit
        # Remaining time (assume we exit at expiry or at hold_bars, whichever is first)
        actual_bars = len(forward_df)
        days_elapsed = actual_bars / c.bars_per_day
        T_exit = max((dte - days_elapsed) / 365.0, 1/365.0)  # min 1 day

        exit_spot = trade.exit_price

        # At exit: option value is max(intrinsic, time_value)
        # If held to expiry, just intrinsic
        if T_exit <= 1/365.0:
            # At expiry
            trade.call_exit_value = max(exit_spot - trade.call_strike, 0)
            trade.put_exit_value = max(trade.put_strike - exit_spot, 0)
            if c.strategy == StrategyType.IRON_CONDOR:
                trade.call_wing_exit = max(exit_spot - trade.call_wing, 0)
                trade.put_wing_exit = max(trade.put_wing - exit_spot, 0)
        else:
            # Before expiry — use BS pricing with reduced IV (post-squeeze IV crush)
            exit_iv = iv * 0.9  # slight IV crush after range period
            trade.call_exit_value = self._price_option(exit_spot, trade.call_strike, T_exit, exit_iv, 'call')
            trade.put_exit_value = self._price_option(exit_spot, trade.put_strike, T_exit, exit_iv, 'put')
            if c.strategy == StrategyType.IRON_CONDOR:
                trade.call_wing_exit = self._price_option(exit_spot, trade.call_wing, T_exit, exit_iv, 'call')
                trade.put_wing_exit = self._price_option(exit_spot, trade.put_wing, T_exit, exit_iv, 'put')

        # BUT: use worst case (max intrinsic during trade) for realistic MTM
        # If at any point the trade was deep in the money, the MTM loss matters
        worst_call_intrinsic = max(trade.max_price - trade.call_strike, 0)
        worst_put_intrinsic = max(trade.put_strike - trade.min_price, 0)

        # For the actual exit P&L, use exit values
        short_legs_exit = trade.call_exit_value + trade.put_exit_value
        long_legs_exit = trade.call_wing_exit + trade.put_wing_exit

        # Gross P&L = Premium collected - (exit value of shorts - exit value of longs)
        trade.gross_pnl = round(trade.total_premium - (short_legs_exit - long_legs_exit), 2)

        # Apply slippage (both entry and exit, 4 legs for IC, 2 for strangle)
        num_legs = 4 if c.strategy == StrategyType.IRON_CONDOR else 2
        slippage = c.slippage_points * num_legs
        trade.net_pnl = round(trade.gross_pnl - slippage, 2)

        # Lots and rupee P&L
        trade.lots = c.lots_per_trade
        # Brokerage: per order per leg
        brokerage = c.brokerage_per_lot * num_legs * 2 * trade.lots  # entry + exit
        point_pnl = trade.net_pnl * lot_size * trade.lots
        trade.pnl_rupees = round(point_pnl - brokerage, 2)

        # Outcome classification
        if trade.zone_held:
            trade.outcome = TradeOutcome.MAX_PROFIT.value
        elif trade.net_pnl > 0:
            trade.outcome = TradeOutcome.PARTIAL_PROFIT.value
        elif abs(trade.net_pnl) < 1:
            trade.outcome = TradeOutcome.BREAKEVEN.value
        else:
            # Check if max loss exceeded
            max_loss_points = c.max_loss_per_trade_pct / 100 * c.capital / (lot_size * trade.lots)
            if abs(trade.net_pnl) > max_loss_points:
                trade.outcome = TradeOutcome.STOPPED_OUT.value
                # Cap loss at max loss
                trade.net_pnl = -max_loss_points
                trade.pnl_rupees = round(-c.max_loss_per_trade_pct / 100 * c.capital, 2)
            else:
                trade.outcome = TradeOutcome.LOSS.value

        return trade

    def run(self) -> Dict:
        """Run the full simulation."""
        self._compute_indicators()

        c = self.config
        df = self.df
        lot_size = LOT_SIZES.get(c.symbol, 25)

        # Find entry signals
        entry_mask = df['entry_signal'] == 1
        valid_atr = df['atr'] > 0
        combined_mask = entry_mask & valid_atr
        entry_indices = df.index[combined_mask]

        trades = []
        trade_id = 0
        last_exit_pos = -1
        capital = c.capital
        equity_curve = [{'date': str(df.index[0]), 'capital': capital, 'trade_id': None}]

        for entry_idx in entry_indices:
            entry_pos = df.index.get_loc(entry_idx)

            # Skip if too close to previous trade
            if entry_pos <= last_exit_pos + c.min_days_between_trades:
                continue

            # Need enough forward data
            exit_pos = entry_pos + c.hold_bars
            if exit_pos >= len(df):
                break

            entry_row = df.iloc[entry_pos]
            forward_df = df.iloc[entry_pos + 1: exit_pos + 1]

            if len(forward_df) < 1:
                continue

            # Compute strikes
            entry_price = entry_row['close']
            atr_val = entry_row['atr']
            iv = entry_row['est_iv']

            call_strike = round(entry_price + c.strike_distance_atr * atr_val)
            put_strike = round(entry_price - c.strike_distance_atr * atr_val)

            # Round to nearest 50 for BankNifty, 50 for Nifty
            strike_step = 100 if c.symbol == 'BANKNIFTY' else 50
            call_strike = round(call_strike / strike_step) * strike_step
            put_strike = round(put_strike / strike_step) * strike_step

            # Compute DTE
            dte_val = c.days_to_expiry_at_entry
            expiry_str = ''
            if c.use_dynamic_dte:
                entry_date_obj = pd.Timestamp(entry_idx).date()
                dte_val, expiry_obj = compute_dte(entry_date_obj, c.symbol, c.min_dte_monthly,
                                                   c.nifty_next_week)
                expiry_str = str(expiry_obj)

            trade = Trade(
                trade_id=trade_id,
                entry_date=str(entry_idx),
                exit_date=str(forward_df.index[-1]),
                symbol=c.symbol,
                strategy=c.strategy.value,
                entry_price=round(entry_price, 2),
                call_strike=call_strike,
                put_strike=put_strike,
                entry_atr=round(atr_val, 2),
                entry_iv=round(iv, 4),
                zone_width_pct=round((call_strike - put_strike) / entry_price * 100, 2),
                dte_at_entry=dte_val,
                expiry_date_used=expiry_str,
                signals_active=self._get_active_signals(entry_row),
            )

            if c.strategy == StrategyType.IRON_CONDOR:
                call_wing = round(entry_price + c.wing_distance_atr * atr_val)
                put_wing = round(entry_price - c.wing_distance_atr * atr_val)
                trade.call_wing = round(call_wing / strike_step) * strike_step
                trade.put_wing = round(put_wing / strike_step) * strike_step

            # Compute P&L
            trade = self._compute_trade_pnl(trade, forward_df)

            # Update capital
            capital += trade.pnl_rupees
            trade_id += 1
            trades.append(trade)
            last_exit_pos = exit_pos

            equity_curve.append({
                'date': trade.exit_date,
                'capital': round(capital, 2),
                'trade_id': trade.trade_id,
            })

        self.trades = trades
        self.equity_curve = equity_curve

        return self._compute_summary()

    def _get_active_signals(self, row):
        """Get comma-separated list of active signals."""
        active = []
        if row.get('bb_squeeze', 0) == 1:
            active.append('BB_SQ')
        if row.get('atr_contract', 0) == 1:
            active.append('ATR_C')
        if row.get('chop_signal', 0) == 1:
            active.append('CHOP')
        if row.get('adx_low', 0) == 1:
            active.append('ADX_L')
        if row.get('rsi_mid', 0) == 1:
            active.append('RSI_M')
        return ','.join(active)

    def _compute_summary(self) -> Dict:
        """Compute summary statistics."""
        trades = self.trades
        if not trades:
            return {'error': 'No trades generated'}

        c = self.config
        lot_size = LOT_SIZES.get(c.symbol, 25)

        pnls = [t.pnl_rupees for t in trades]
        wins = [t for t in trades if t.pnl_rupees > 0]
        losses = [t for t in trades if t.pnl_rupees <= 0]

        # Equity curve for drawdown
        eq = [e['capital'] for e in self.equity_curve]
        peak = eq[0]
        max_dd = 0
        max_dd_pct = 0
        for val in eq:
            if val > peak:
                peak = val
            dd = peak - val
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd = dd

        total_pnl = sum(pnls)

        # Outcomes breakdown
        outcomes = {}
        for t in trades:
            outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1

        # Monthly returns
        monthly_pnl = {}
        for t in trades:
            month = t.entry_date[:7]
            monthly_pnl[month] = monthly_pnl.get(month, 0) + t.pnl_rupees

        # Zone analysis
        zone_widths = [t.zone_width_pct for t in trades]
        premiums_collected = [t.total_premium for t in trades]

        # Consecutive wins/losses
        max_consec_wins = max_consec_losses = 0
        curr_wins = curr_losses = 0
        for t in trades:
            if t.pnl_rupees > 0:
                curr_wins += 1
                curr_losses = 0
            else:
                curr_losses += 1
                curr_wins = 0
            max_consec_wins = max(max_consec_wins, curr_wins)
            max_consec_losses = max(max_consec_losses, curr_losses)

        # CAGR
        if len(self.equity_curve) > 1:
            start_cap = self.equity_curve[0]['capital']
            end_cap = self.equity_curve[-1]['capital']
            start_date = pd.Timestamp(self.equity_curve[0]['date'])
            end_date = pd.Timestamp(self.equity_curve[-1]['date'])
            years = (end_date - start_date).days / 365.25
            if years > 0 and end_cap > 0:
                cagr = (end_cap / start_cap) ** (1 / years) - 1
            else:
                cagr = 0
        else:
            cagr = 0

        return {
            'symbol': c.symbol,
            'strategy': c.strategy.value,
            'strike_distance': f'{c.strike_distance_atr}x ATR',
            'hold_period': f'{c.hold_bars} bars',
            'initial_capital': c.capital,
            'final_capital': round(self.equity_curve[-1]['capital'], 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_pnl / c.capital * 100, 2),
            'cagr_pct': round(cagr * 100, 2),
            'total_trades': len(trades),
            'winners': len(wins),
            'losers': len(losses),
            'win_rate': round(len(wins) / len(trades) * 100, 2),
            'avg_win': round(np.mean([t.pnl_rupees for t in wins]), 2) if wins else 0,
            'avg_loss': round(np.mean([t.pnl_rupees for t in losses]), 2) if losses else 0,
            'max_win': round(max(pnls), 2),
            'max_loss': round(min(pnls), 2),
            'profit_factor': round(
                sum(t.pnl_rupees for t in wins) / abs(sum(t.pnl_rupees for t in losses)), 2
            ) if losses and sum(t.pnl_rupees for t in losses) != 0 else float('inf'),
            'max_drawdown': round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'max_consecutive_wins': max_consec_wins,
            'max_consecutive_losses': max_consec_losses,
            'avg_premium_collected': round(np.mean(premiums_collected), 2),
            'avg_zone_width_pct': round(np.mean(zone_widths), 2),
            'zone_hold_rate': round(sum(1 for t in trades if t.zone_held) / len(trades) * 100, 2),
            'outcomes': outcomes,
            'avg_pnl_per_trade': round(np.mean(pnls), 2),
            'monthly_pnl': monthly_pnl,
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        rows = []
        for t in self.trades:
            rows.append({
                'id': t.trade_id,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'call_strike': t.call_strike,
                'put_strike': t.put_strike,
                'zone_width_pct': t.zone_width_pct,
                'premium': t.total_premium,
                'exit_price': t.exit_price,
                'max_price': round(t.max_price, 2),
                'min_price': round(t.min_price, 2),
                'max_breach_pct': t.max_adverse_excursion,
                'gross_pnl': t.gross_pnl,
                'net_pnl': t.net_pnl,
                'pnl_rs': t.pnl_rupees,
                'outcome': t.outcome,
                'zone_held': t.zone_held,
                'signals': t.signals_active,
                'iv': t.entry_iv,
                'atr': t.entry_atr,
                'dte': t.dte_at_entry,
                'expiry': t.expiry_date_used,
            })
        return pd.DataFrame(rows)

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve)
