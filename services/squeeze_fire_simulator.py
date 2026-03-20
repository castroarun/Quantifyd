"""
BNF Squeeze & Fire — Unified Options System
=============================================

One signal engine, two trade modes based on volatility regime:

SQUEEZE MODE (Non-Directional):
  - Condition: BB width < BB width MA (squeezing) + price near SMA (ranging)
  - Trade: Short Strangle — sell OTM CE + OTM PE
  - Goal: Collect premium in low-vol range-bound markets
  - Exit: Hold to expiry / time-based, or stop on zone breach

FIRE MODE (Directional):
  - Condition: BB expanding AFTER squeeze (squeeze-fire) + price trending away from SMA
  - Trade: Debit Spread — Bull Call Spread (long) or Bear Put Spread (short)
  - Direction: Close > SMA → long, Close < SMA → short
  - Goal: Ride breakout with defined risk
  - Exit: Profit target, stop loss, or hold period

Shared infrastructure:
  - Same BB/ATR/SMA indicators
  - Same BankNifty monthly expiry logic
  - Same Black-Scholes pricing
  - Combined equity curve and trade log
"""

import numpy as np
import pandas as pd
import calendar
from datetime import date, timedelta
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum


# =============================================================================
# Constants
# =============================================================================

LOT_SIZE = 15  # BankNifty
RISK_FREE_RATE = 0.065
TRADING_DAYS_PER_YEAR = 252
STRIKE_STEP = 100  # BankNifty strike intervals


# =============================================================================
# Expiry Helpers
# =============================================================================

def get_last_thursday(year: int, month: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d


def get_monthly_expiry(entry_date: date, min_dte: int = 15) -> Tuple[int, date]:
    """BankNifty monthly expiry with min DTE roll."""
    expiry = get_last_thursday(entry_date.year, entry_date.month)
    if expiry <= entry_date:
        if entry_date.month == 12:
            expiry = get_last_thursday(entry_date.year + 1, 1)
        else:
            expiry = get_last_thursday(entry_date.year, entry_date.month + 1)
    dte = (expiry - entry_date).days
    if dte < min_dte:
        if expiry.month == 12:
            expiry = get_last_thursday(expiry.year + 1, 1)
        else:
            expiry = get_last_thursday(expiry.year, expiry.month + 1)
        dte = (expiry - entry_date).days
    return dte, expiry


def round_strike(price, step=STRIKE_STEP):
    return round(price / step) * step


# =============================================================================
# Black-Scholes
# =============================================================================

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_iv(atr_value, close_price):
    daily_atr_pct = atr_value / close_price
    daily_std = daily_atr_pct / 1.2
    rv = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return max(rv * 1.20, 0.08)


# =============================================================================
# Trade Mode Enum
# =============================================================================

class TradeMode(Enum):
    SQUEEZE = 'squeeze'  # Short strangle
    FIRE = 'fire'        # Debit spread


class FireDirection(Enum):
    LONG = 'long'
    SHORT = 'short'


# =============================================================================
# Config
# =============================================================================

@dataclass
class SqueezeFireConfig:
    # --- Shared indicator params ---
    bb_period: int = 10
    atr_period: int = 14
    sma_period: int = 20

    # --- Squeeze mode (strangle) params ---
    squeeze_enabled: bool = True
    squeeze_strike_atr: float = 1.5       # Strangle strikes at +/- N ATR
    squeeze_hold_bars: int = 10           # Biweekly hold
    squeeze_trend_filter: bool = True
    squeeze_trend_atr_max: float = 2.0    # Block if price > N ATR from SMA (too trendy)
    squeeze_max_loss_rupees: float = 30_000  # Hard cap on max loss per trade (Rs)

    # --- Fire mode (debit spread) params ---
    fire_enabled: bool = True
    fire_squeeze_min_bars: int = 3        # Min squeeze bars before fire
    fire_spread_width_atr: float = 1.5    # Spread width
    fire_long_offset_atr: float = 0.0     # 0 = ATM
    fire_hold_bars: int = 7
    fire_trend_atr_min: float = 1.0       # Price must be > N ATR from SMA (confirm trend)
    fire_profit_target_pct: float = 0.50  # 50% profit on debit
    fire_stop_loss_pct: float = 0.40      # 40% loss on debit
    fire_use_di_filter: bool = False
    fire_use_rsi_filter: bool = False
    fire_rsi_period: int = 14

    # --- Position sizing ---
    capital: float = 10_00_000            # Rs 10L
    squeeze_lots: int = 5
    fire_lots: int = 5
    max_risk_pct: float = 3.0            # Max risk per trade

    # --- Expiry ---
    min_dte: int = 15

    # --- Costs ---
    brokerage_per_lot: float = 40
    slippage_points: float = 2

    # --- Debounce ---
    min_days_between_trades: int = 5      # Per mode


# =============================================================================
# Unified Trade Record
# =============================================================================

@dataclass
class UnifiedTrade:
    trade_id: int
    mode: str              # 'squeeze' or 'fire'
    direction: str         # 'neutral' (squeeze), 'long'/'short' (fire)
    entry_date: str
    exit_date: str = ''

    # Spot
    entry_spot: float = 0
    exit_spot: float = 0

    # Strikes — squeeze uses call_strike/put_strike, fire uses long_strike/short_strike
    strike_1: float = 0    # Squeeze: call strike / Fire: long strike
    strike_2: float = 0    # Squeeze: put strike / Fire: short strike

    # Premium / Debit
    premium_collected: float = 0   # Squeeze: net credit received
    debit_paid: float = 0          # Fire: net debit paid
    exit_value: float = 0          # Value at exit

    # P&L
    pnl_points: float = 0         # Per lot
    pnl_rupees: float = 0         # Total
    pnl_pct: float = 0            # % return on premium/debit
    lots: int = 1

    # Meta
    entry_atr: float = 0
    entry_iv: float = 0
    dte_at_entry: int = 0
    expiry_date: str = ''
    days_held: int = 0
    exit_reason: str = ''
    squeeze_bars: int = 0
    bias_signals: str = ''


# =============================================================================
# Unified Simulator
# =============================================================================

class SqueezeFireSimulator:
    """
    Unified BNF options system:
    - Squeeze mode: short strangles during range-bound BB squeeze
    - Fire mode: debit spreads on BB expansion breakouts
    """

    def __init__(self, config: SqueezeFireConfig, df: pd.DataFrame):
        self.config = config
        self.df = df.copy()
        self.trades: List[UnifiedTrade] = []
        self.equity_curve: List[Dict] = []

    def _compute_indicators(self):
        df = self.df
        c = self.config

        high, low, close = df['high'], df['low'], df['close']

        # --- ATR ---
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(c.atr_period).mean()

        # --- Bollinger Band Width ---
        sma_bb = close.rolling(c.bb_period).mean()
        std_bb = close.rolling(c.bb_period).std()
        bb_width = (2 * 2.0 * std_bb) / sma_bb * 100
        bb_width_ma = bb_width.rolling(c.bb_period).mean()
        df['bb_squeeze'] = (bb_width < bb_width_ma).astype(int)
        df['bb_expanding'] = (bb_width > bb_width_ma).astype(int)

        # --- Squeeze fire detection ---
        # Count consecutive squeeze bars
        squeeze_runs = []
        run = 0
        for val in df['bb_squeeze']:
            if val == 1:
                run += 1
            else:
                squeeze_runs.append(run)
                run = 0
                continue
            squeeze_runs.append(0)
        df['squeeze_run_at_fire'] = squeeze_runs

        # Fire = expanding now + was squeezing for N bars
        squeeze_sum = df['bb_squeeze'].rolling(c.fire_squeeze_min_bars).sum()
        df['squeeze_fire'] = (
            (df['bb_expanding'] == 1) &
            (squeeze_sum.shift(1) >= c.fire_squeeze_min_bars)
        ).astype(int)

        # --- SMA for direction ---
        df['sma'] = close.rolling(c.sma_period).mean()
        df['price_dist_atr'] = (close - df['sma']).abs() / df['atr']
        df['signed_dist_atr'] = (close - df['sma']) / df['atr']

        # --- Ranging vs Trending ---
        df['is_ranging'] = (df['price_dist_atr'] <= c.squeeze_trend_atr_max).astype(int)
        df['is_trending'] = (df['price_dist_atr'] >= c.fire_trend_atr_min).astype(int)

        # --- DI+/DI- ---
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / df['atr'])
        df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / df['atr'])

        # --- RSI ---
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(c.fire_rsi_period).mean()
        loss_s = (-delta.where(delta < 0, 0.0)).rolling(c.fire_rsi_period).mean()
        rs = gain / loss_s.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # --- IV estimate ---
        df['est_iv'] = df.apply(
            lambda row: estimate_iv(row['atr'], row['close'])
            if row['atr'] > 0 and not np.isnan(row['atr']) else 0.15,
            axis=1
        )

        # ===== ENTRY SIGNALS =====

        # SQUEEZE entry: BB squeezing + ranging
        squeeze_sig = (df['bb_squeeze'] == 1)
        if c.squeeze_trend_filter:
            squeeze_sig = squeeze_sig & (df['is_ranging'] == 1)
        df['squeeze_signal'] = squeeze_sig.astype(int)

        # FIRE entry: squeeze fire + trending + directional bias
        fire_long = (
            (df['squeeze_fire'] == 1) &
            (df['is_trending'] == 1) &
            (close > df['sma'])
        )
        fire_short = (
            (df['squeeze_fire'] == 1) &
            (df['is_trending'] == 1) &
            (close < df['sma'])
        )

        # Optional DI filter
        if c.fire_use_di_filter:
            fire_long = fire_long & (df['plus_di'] > df['minus_di'])
            fire_short = fire_short & (df['minus_di'] > df['plus_di'])

        # Optional RSI filter
        if c.fire_use_rsi_filter:
            fire_long = fire_long & (df['rsi'] > 50)
            fire_short = fire_short & (df['rsi'] < 50)

        df['fire_long_signal'] = fire_long.astype(int)
        df['fire_short_signal'] = fire_short.astype(int)

        self.df = df

    # ----- Strangle pricing (squeeze mode) -----

    def _price_strangle_entry(self, spot, call_strike, put_strike, dte, iv):
        T = max(dte, 1) / 365.0
        call_prem = bs_call(spot, call_strike, T, RISK_FREE_RATE, iv)
        put_prem = bs_put(spot, put_strike, T, RISK_FREE_RATE, iv)
        return call_prem, put_prem

    def _price_strangle_exit(self, spot, call_strike, put_strike, remaining_dte, iv):
        T = max(remaining_dte, 0.5) / 365.0
        call_val = bs_call(spot, call_strike, T, RISK_FREE_RATE, iv)
        put_val = bs_put(spot, put_strike, T, RISK_FREE_RATE, iv)
        return call_val, put_val

    # ----- Spread pricing (fire mode) -----

    def _price_spread(self, spot, long_strike, short_strike, dte, iv, direction):
        T = max(dte, 0.5) / 365.0
        if direction == 'long':
            long_val = bs_call(spot, long_strike, T, RISK_FREE_RATE, iv)
            short_val = bs_call(spot, short_strike, T, RISK_FREE_RATE, iv)
        else:
            long_val = bs_put(spot, long_strike, T, RISK_FREE_RATE, iv)
            short_val = bs_put(spot, short_strike, T, RISK_FREE_RATE, iv)
        return long_val, short_val, long_val - short_val

    # ----- Simulate squeeze trade -----

    def _sim_squeeze_trade(self, i, trade_id, row, dte, expiry):
        c = self.config
        spot = row['close']
        atr = row['atr']
        iv = row['est_iv']

        call_strike = round_strike(spot + c.squeeze_strike_atr * atr)
        put_strike = round_strike(spot - c.squeeze_strike_atr * atr)

        call_prem, put_prem = self._price_strangle_entry(spot, call_strike, put_strike, dte, iv)
        total_credit = call_prem + put_prem

        if total_credit <= 0:
            return None

        lots = c.squeeze_lots
        hold_end = min(i + c.squeeze_hold_bars, len(self.df) - 1)

        # Check if expiry within hold
        for j in range(i + 1, hold_end + 1):
            fwd_date = str(self.df.iloc[j]['date'])[:10]
            if fwd_date >= str(expiry):
                hold_end = j
                break

        # Track P&L through hold period
        exit_idx = hold_end
        exit_reason = 'hold_complete'
        # Max loss in points per lot = max_loss_rupees / (lots * lot_size)
        max_loss_pts = c.squeeze_max_loss_rupees / (lots * LOT_SIZE)

        for j in range(i + 1, hold_end + 1):
            fwd = self.df.iloc[j]
            rem_dte = max(dte - (j - i), 0)
            c_val, p_val = self._price_strangle_exit(fwd['close'], call_strike, put_strike, rem_dte, iv)
            current_cost = c_val + p_val
            unrealized_pnl = total_credit - current_cost

            if unrealized_pnl <= -max_loss_pts:
                exit_idx = j
                exit_reason = 'stop_loss'
                break

        exit_row = self.df.iloc[exit_idx]
        rem_dte_exit = max(dte - (exit_idx - i), 0)
        c_exit, p_exit = self._price_strangle_exit(exit_row['close'], call_strike, put_strike, rem_dte_exit, iv)
        exit_cost = c_exit + p_exit

        pnl_points = total_credit - exit_cost
        cost = c.brokerage_per_lot * lots * 4 + c.slippage_points * 4 * lots * LOT_SIZE
        pnl_rupees = pnl_points * lots * LOT_SIZE - cost
        # Cap loss at configured max
        pnl_rupees = max(pnl_rupees, -c.squeeze_max_loss_rupees)

        return UnifiedTrade(
            trade_id=trade_id,
            mode='squeeze',
            direction='neutral',
            entry_date=str(row['date'])[:10],
            exit_date=str(exit_row['date'])[:10],
            entry_spot=round(spot, 2),
            exit_spot=round(exit_row['close'], 2),
            strike_1=call_strike,
            strike_2=put_strike,
            premium_collected=round(total_credit, 2),
            exit_value=round(exit_cost, 2),
            pnl_points=round(pnl_points, 2),
            pnl_rupees=round(pnl_rupees, 2),
            pnl_pct=round(pnl_points / total_credit * 100 if total_credit > 0 else 0, 2),
            lots=lots,
            entry_atr=round(atr, 2),
            entry_iv=round(iv, 4),
            dte_at_entry=dte,
            expiry_date=str(expiry),
            days_held=exit_idx - i,
            exit_reason=exit_reason,
        ), exit_idx

    # ----- Simulate fire trade -----

    def _sim_fire_trade(self, i, trade_id, row, dte, expiry, direction):
        c = self.config
        spot = row['close']
        atr = row['atr']
        iv = row['est_iv']

        if direction == 'long':
            long_strike = round_strike(spot + c.fire_long_offset_atr * atr)
            short_strike = round_strike(long_strike + c.fire_spread_width_atr * atr)
        else:
            long_strike = round_strike(spot - c.fire_long_offset_atr * atr)
            short_strike = round_strike(long_strike - c.fire_spread_width_atr * atr)

        _, _, net_debit = self._price_spread(spot, long_strike, short_strike, dte, iv, direction)

        if net_debit <= 0:
            return None

        lots = c.fire_lots
        total_risk = net_debit * lots * LOT_SIZE
        if total_risk > self._current_equity * c.max_risk_pct / 100:
            lots = int((self._current_equity * c.max_risk_pct / 100) / (net_debit * LOT_SIZE))
            if lots < 1:
                return None

        hold_end = min(i + c.fire_hold_bars, len(self.df) - 1)

        # Check expiry within hold
        for j in range(i + 1, hold_end + 1):
            fwd_date = str(self.df.iloc[j]['date'])[:10]
            if fwd_date >= str(expiry):
                hold_end = j
                break

        profit_target_val = net_debit * (1 + c.fire_profit_target_pct)
        stop_loss_val = net_debit * (1 - c.fire_stop_loss_pct)

        exit_idx = hold_end
        exit_reason = 'hold_complete'

        for j in range(i + 1, hold_end + 1):
            fwd = self.df.iloc[j]
            rem_dte = max(dte - (j - i), 0)
            _, _, spread_val = self._price_spread(fwd['close'], long_strike, short_strike, rem_dte, iv, direction)

            if spread_val >= profit_target_val:
                exit_idx = j
                exit_reason = 'profit_target'
                break
            elif spread_val <= stop_loss_val:
                exit_idx = j
                exit_reason = 'stop_loss'
                break

        exit_row = self.df.iloc[exit_idx]
        rem_dte_exit = max(dte - (exit_idx - i), 0)
        _, _, exit_spread_val = self._price_spread(exit_row['close'], long_strike, short_strike, rem_dte_exit, iv, direction)

        pnl_points = exit_spread_val - net_debit
        cost = c.brokerage_per_lot * lots * 4 + c.slippage_points * 2 * lots * LOT_SIZE
        pnl_rupees = pnl_points * lots * LOT_SIZE - cost

        bias_parts = ['SMA']
        if c.fire_use_di_filter:
            bias_parts.append('DI')
        if c.fire_use_rsi_filter:
            bias_parts.append('RSI')

        return UnifiedTrade(
            trade_id=trade_id,
            mode='fire',
            direction=direction,
            entry_date=str(row['date'])[:10],
            exit_date=str(exit_row['date'])[:10],
            entry_spot=round(spot, 2),
            exit_spot=round(exit_row['close'], 2),
            strike_1=long_strike,
            strike_2=short_strike,
            debit_paid=round(net_debit, 2),
            exit_value=round(exit_spread_val, 2),
            pnl_points=round(pnl_points, 2),
            pnl_rupees=round(pnl_rupees, 2),
            pnl_pct=round(pnl_points / net_debit * 100 if net_debit > 0 else 0, 2),
            lots=lots,
            entry_atr=round(atr, 2),
            entry_iv=round(iv, 4),
            dte_at_entry=dte,
            expiry_date=str(expiry),
            days_held=exit_idx - i,
            exit_reason=exit_reason,
            squeeze_bars=int(row.get('squeeze_run_at_fire', 0)),
            bias_signals='+'.join(bias_parts),
        ), exit_idx

    # ----- Main run -----

    def run(self) -> Dict:
        self._compute_indicators()

        df = self.df
        c = self.config

        self._current_equity = c.capital
        equity = c.capital
        trade_id = 0

        # Separate debounce tracking per mode
        last_squeeze_idx = -c.min_days_between_trades
        last_fire_idx = -c.min_days_between_trades
        # Track active trade end to prevent overlaps
        active_until = -1

        equity_curve = []

        for i in range(50, len(df)):
            row = df.iloc[i]
            current_date = str(row['date'])[:10]

            if i <= active_until:
                equity_curve.append({'date': current_date, 'equity': equity})
                continue

            try:
                entry_date_obj = pd.Timestamp(row['date']).date()
            except:
                entry_date_obj = date.fromisoformat(current_date)

            dte, expiry = get_monthly_expiry(entry_date_obj, c.min_dte)

            trade_result = None

            # Priority: Fire signals first (they're rarer and higher conviction)
            if c.fire_enabled:
                if row['fire_long_signal'] == 1 and (i - last_fire_idx) >= c.min_days_between_trades:
                    result = self._sim_fire_trade(i, trade_id, row, dte, expiry, 'long')
                    if result:
                        trade_result, end_idx = result
                        last_fire_idx = end_idx
                elif row['fire_short_signal'] == 1 and (i - last_fire_idx) >= c.min_days_between_trades:
                    result = self._sim_fire_trade(i, trade_id, row, dte, expiry, 'short')
                    if result:
                        trade_result, end_idx = result
                        last_fire_idx = end_idx

            # Squeeze signals (only if no fire trade taken)
            if trade_result is None and c.squeeze_enabled:
                if row['squeeze_signal'] == 1 and (i - last_squeeze_idx) >= c.min_days_between_trades:
                    result = self._sim_squeeze_trade(i, trade_id, row, dte, expiry)
                    if result:
                        trade_result, end_idx = result
                        last_squeeze_idx = end_idx

            if trade_result:
                self.trades.append(trade_result)
                equity += trade_result.pnl_rupees
                self._current_equity = equity
                active_until = end_idx
                trade_id += 1

            equity_curve.append({'date': current_date, 'equity': equity})

        self.equity_curve = equity_curve
        return self._compute_summary()

    def _compute_summary(self) -> Dict:
        c = self.config

        if not self.trades:
            return {
                'total_trades': 0, 'squeeze_trades': 0, 'fire_trades': 0,
                'cagr': 0, 'sharpe': 0, 'max_drawdown': 0, 'calmar': 0,
                'win_rate': 0, 'profit_factor': 0, 'final_equity': c.capital,
            }

        squeeze_trades = [t for t in self.trades if t.mode == 'squeeze']
        fire_trades = [t for t in self.trades if t.mode == 'fire']
        fire_long = [t for t in fire_trades if t.direction == 'long']
        fire_short = [t for t in fire_trades if t.direction == 'short']

        winners = [t for t in self.trades if t.pnl_rupees > 0]
        losers = [t for t in self.trades if t.pnl_rupees <= 0]
        total = len(self.trades)

        win_rate = len(winners) / total * 100

        gross_profit = sum(t.pnl_rupees for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_rupees for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        final_equity = c.capital + sum(t.pnl_rupees for t in self.trades)
        total_return = final_equity / c.capital

        first_date = pd.Timestamp(self.trades[0].entry_date)
        last_date = pd.Timestamp(self.trades[-1].exit_date)
        years = max((last_date - first_date).days / 365.25, 0.1)
        cagr = (total_return ** (1 / years) - 1) * 100

        # Sharpe
        returns = [t.pnl_pct / 100 for t in self.trades]
        if len(returns) > 1:
            avg_r = np.mean(returns)
            std_r = np.std(returns, ddof=1)
            tpy = total / years
            sharpe = (avg_r * tpy) / (std_r * np.sqrt(tpy)) if std_r > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        equities = [e['equity'] for e in self.equity_curve]
        peak = equities[0]
        max_dd = 0
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)

        calmar = cagr / max_dd if max_dd > 0 else 0

        # Per-mode stats
        def mode_stats(trades_list):
            if not trades_list:
                return {'count': 0, 'win_rate': 0, 'avg_pnl_pct': 0, 'total_pnl': 0, 'pf': 0}
            w = [t for t in trades_list if t.pnl_rupees > 0]
            l = [t for t in trades_list if t.pnl_rupees <= 0]
            gp = sum(t.pnl_rupees for t in w) if w else 0
            gl = abs(sum(t.pnl_rupees for t in l)) if l else 1
            return {
                'count': len(trades_list),
                'win_rate': round(len(w) / len(trades_list) * 100, 1),
                'avg_pnl_pct': round(np.mean([t.pnl_pct for t in trades_list]), 2),
                'total_pnl': round(sum(t.pnl_rupees for t in trades_list), 0),
                'pf': round(gp / gl if gl > 0 else 0, 2),
            }

        # Exit reasons
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            'total_trades': total,
            'squeeze_trades': len(squeeze_trades),
            'fire_trades': len(fire_trades),
            'fire_long': len(fire_long),
            'fire_short': len(fire_short),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(sum(t.pnl_rupees for t in self.trades), 0),
            'avg_pnl_pct': round(np.mean([t.pnl_pct for t in self.trades]), 2),
            'cagr': round(cagr, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'calmar': round(calmar, 2),
            'final_equity': round(final_equity, 0),
            'total_return_pct': round((total_return - 1) * 100, 2),
            'avg_days_held': round(np.mean([t.days_held for t in self.trades]), 1),
            'avg_dte': round(np.mean([t.dte_at_entry for t in self.trades]), 1),
            'squeeze_stats': mode_stats(squeeze_trades),
            'fire_stats': mode_stats(fire_trades),
            'fire_long_stats': mode_stats(fire_long),
            'fire_short_stats': mode_stats(fire_short),
            'exit_reasons': exit_reasons,
        }

    def get_trade_log(self) -> List[Dict]:
        return [
            {
                'trade_id': t.trade_id,
                'mode': t.mode,
                'direction': t.direction,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_spot': t.entry_spot,
                'exit_spot': t.exit_spot,
                'strike_1': t.strike_1,
                'strike_2': t.strike_2,
                'premium_or_debit': t.premium_collected if t.mode == 'squeeze' else t.debit_paid,
                'exit_value': t.exit_value,
                'pnl_points': t.pnl_points,
                'pnl_rupees': t.pnl_rupees,
                'pnl_pct': t.pnl_pct,
                'lots': t.lots,
                'days_held': t.days_held,
                'dte_at_entry': t.dte_at_entry,
                'exit_reason': t.exit_reason,
                'squeeze_bars': t.squeeze_bars,
                'bias_signals': t.bias_signals,
                'entry_atr': t.entry_atr,
                'entry_iv': t.entry_iv,
            }
            for t in self.trades
        ]
