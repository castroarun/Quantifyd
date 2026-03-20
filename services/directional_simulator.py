"""
Directional Options Strategy Simulator — BankNifty Debit Spreads
================================================================

Uses the INVERSE of non-directional range signals:
- BB Squeeze Fire: BB was squeezing, now expanding (breakout)
- Directional Bias: Close vs SMA for long/short
- Trend confirmation: price distance from SMA in ATR units > threshold

Trade Structure:
- Bullish: Bull Call Spread (Buy ATM CE + Sell OTM CE)
- Bearish: Bear Put Spread (Buy ATM PE + Sell OTM PE)

Entry: BB squeeze-fire + directional bias
Exit: Hold period, profit target %, or stop loss % on premium
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

LOT_SIZE = 15  # BankNifty lot size
RISK_FREE_RATE = 0.065
TRADING_DAYS_PER_YEAR = 252


# =============================================================================
# Expiry Helpers (reused from non-directional)
# =============================================================================

def get_last_thursday(year: int, month: int) -> date:
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:
        d -= timedelta(days=1)
    return d


def get_monthly_expiry(entry_date: date, min_dte: int = 15) -> Tuple[int, date]:
    """Get BankNifty monthly expiry with min DTE roll logic."""
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


# =============================================================================
# Black-Scholes
# =============================================================================

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_iv_from_atr(atr_value, close_price):
    """Estimate IV from ATR (daily bars)."""
    daily_atr_pct = atr_value / close_price
    daily_std = daily_atr_pct / 1.2
    realized_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    iv = realized_vol * 1.20  # IV premium over RV
    return max(iv, 0.08)


# =============================================================================
# Enums & Config
# =============================================================================

class Direction(Enum):
    LONG = 'long'   # Bull Call Spread
    SHORT = 'short' # Bear Put Spread


class ExitReason(Enum):
    HOLD_COMPLETE = 'hold_complete'
    PROFIT_TARGET = 'profit_target'
    STOP_LOSS = 'stop_loss'
    EXPIRY = 'expiry'


@dataclass
class DirectionalConfig:
    # Signal parameters
    bb_period: int = 10
    atr_period: int = 14
    sma_period: int = 20           # For directional bias
    squeeze_min_bars: int = 3      # Min bars of squeeze before fire

    # Trend confirmation
    use_trend_confirm: bool = True
    trend_atr_min: float = 1.0     # Price must be > N ATR from SMA (trending)

    # Additional bias filters
    use_di_filter: bool = False    # DI+ vs DI- confirmation
    use_rsi_filter: bool = False   # RSI > 50 for long, < 50 for short
    rsi_period: int = 14

    # Spread construction
    long_strike_offset_atr: float = 0.0   # 0 = ATM, 0.5 = slightly OTM
    spread_width_atr: float = 1.5         # Width between long and short strikes in ATR

    # Holding & exits
    hold_bars: int = 7
    profit_target_pct: float = 0.50   # Exit at 50% profit on debit
    stop_loss_pct: float = 0.40       # Exit at 40% loss on debit (60% of debit remaining)

    # Position sizing
    capital: float = 10_00_000
    lots_per_trade: int = 5
    max_risk_pct: float = 3.0       # Max % capital at risk per trade

    # Option expiry
    min_dte: int = 15               # Roll to next month if DTE < this

    # Costs
    brokerage_per_lot: float = 40   # Per order (4 legs = 4 orders)
    slippage_points: float = 3      # Per leg

    # Debounce
    min_days_between_trades: int = 5


@dataclass
class DirectionalTrade:
    trade_id: int
    direction: str             # 'long' or 'short'
    entry_date: str
    exit_date: str = ''

    # Price levels
    entry_spot: float = 0
    exit_spot: float = 0

    # Spread strikes
    long_strike: float = 0     # Bought option strike
    short_strike: float = 0    # Sold option strike

    # Premiums
    long_premium: float = 0    # Paid for long leg
    short_premium: float = 0   # Received for short leg
    net_debit: float = 0       # Per lot in points

    # Exit values
    long_exit_value: float = 0
    short_exit_value: float = 0
    net_credit_at_exit: float = 0

    # P&L
    pnl_per_lot: float = 0    # Points
    pnl_rupees: float = 0     # Total Rs
    pnl_pct: float = 0        # % return on debit
    lots: int = 1

    # Max spread value during trade
    max_spread_value: float = 0
    min_spread_value: float = 0

    # Meta
    entry_atr: float = 0
    entry_iv: float = 0
    dte_at_entry: int = 0
    expiry_date: str = ''
    days_held: int = 0
    exit_reason: str = ''

    # Signal info
    squeeze_bars: int = 0      # How many bars of squeeze before fire
    bias_signal: str = ''      # What confirmed direction


# =============================================================================
# Simulator Engine
# =============================================================================

class DirectionalSimulator:
    """
    Simulates directional debit spread strategy on BankNifty.
    Entry: BB squeeze fire + directional bias
    Trade: Bull Call Spread (long) or Bear Put Spread (short)
    """

    def __init__(self, config: DirectionalConfig, df: pd.DataFrame):
        self.config = config
        self.df = df.copy()
        self.trades: List[DirectionalTrade] = []
        self.equity_curve: List[Dict] = []

    def _compute_indicators(self):
        """Compute squeeze, trend, and directional indicators."""
        df = self.df
        c = self.config

        high, low, close = df['high'], df['low'], df['close']

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(c.atr_period).mean()

        # Bollinger Band width
        sma_bb = close.rolling(c.bb_period).mean()
        std_bb = close.rolling(c.bb_period).std()
        bb_width = (2 * 2.0 * std_bb) / sma_bb * 100
        bb_width_ma = bb_width.rolling(c.bb_period).mean()
        df['bb_squeeze'] = (bb_width < bb_width_ma).astype(int)
        df['bb_expanding'] = (bb_width > bb_width_ma).astype(int)

        # Squeeze fire: was squeezing for N bars, now expanding
        squeeze_count = df['bb_squeeze'].rolling(c.squeeze_min_bars).sum()
        df['squeeze_fire'] = (
            (df['bb_expanding'] == 1) &
            (squeeze_count.shift(1) >= c.squeeze_min_bars)  # Previous N bars were squeeze
        ).astype(int)

        # Count consecutive squeeze bars before fire (for logging)
        squeeze_run = df['bb_squeeze'].copy()
        df['squeeze_run_len'] = 0
        run = 0
        for i in range(len(df)):
            if squeeze_run.iloc[i] == 1:
                run += 1
            else:
                df.iloc[i, df.columns.get_loc('squeeze_run_len')] = run
                run = 0

        # SMA for directional bias
        df['sma'] = close.rolling(c.sma_period).mean()
        df['bias_long'] = (close > df['sma']).astype(int)
        df['bias_short'] = (close < df['sma']).astype(int)

        # Trend confirmation: price distance from SMA in ATR units
        price_dist_atr = (close - df['sma']).abs() / df['atr']
        df['price_dist_atr'] = price_dist_atr
        if c.use_trend_confirm:
            df['is_trending'] = (price_dist_atr >= c.trend_atr_min).astype(int)
        else:
            df['is_trending'] = 1

        # Directional momentum from signed distance (not abs)
        df['signed_dist_atr'] = (close - df['sma']) / df['atr']

        # DI+/DI- for optional filter
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / df['atr'])
        df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / df['atr'])
        df['di_long'] = (df['plus_di'] > df['minus_di']).astype(int)
        df['di_short'] = (df['minus_di'] > df['plus_di']).astype(int)

        # RSI for optional filter
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(c.rsi_period).mean()
        loss_s = (-delta.where(delta < 0, 0.0)).rolling(c.rsi_period).mean()
        rs = gain / loss_s.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_long'] = (df['rsi'] > 50).astype(int)
        df['rsi_short'] = (df['rsi'] < 50).astype(int)

        # IV estimate
        df['est_iv'] = df.apply(
            lambda row: estimate_iv_from_atr(row['atr'], row['close'])
            if row['atr'] > 0 and not np.isnan(row['atr']) else 0.15,
            axis=1
        )

        # Build entry signals
        # Long entry: squeeze fire + close > SMA + trending + optional filters
        long_sig = (df['squeeze_fire'] == 1) & (df['bias_long'] == 1) & (df['is_trending'] == 1)
        if c.use_di_filter:
            long_sig = long_sig & (df['di_long'] == 1)
        if c.use_rsi_filter:
            long_sig = long_sig & (df['rsi_long'] == 1)
        df['long_signal'] = long_sig.astype(int)

        # Short entry: squeeze fire + close < SMA + trending + optional filters
        short_sig = (df['squeeze_fire'] == 1) & (df['bias_short'] == 1) & (df['is_trending'] == 1)
        if c.use_di_filter:
            short_sig = short_sig & (df['di_short'] == 1)
        if c.use_rsi_filter:
            short_sig = short_sig & (df['rsi_short'] == 1)
        df['short_signal'] = short_sig.astype(int)

        self.df = df

    def _price_spread_at(self, spot, long_strike, short_strike, dte_days, iv, direction):
        """
        Price a debit spread at a given point.
        Returns (long_leg_value, short_leg_value, net_spread_value).
        Net spread value > 0 means we'd receive that much closing the spread.
        """
        T = max(dte_days, 0.5) / 365.0  # Floor at half day

        if direction == 'long':
            # Bull Call Spread: long lower CE, short higher CE
            long_val = bs_call_price(spot, long_strike, T, RISK_FREE_RATE, iv)
            short_val = bs_call_price(spot, short_strike, T, RISK_FREE_RATE, iv)
        else:
            # Bear Put Spread: long higher PE, short lower PE
            long_val = bs_put_price(spot, long_strike, T, RISK_FREE_RATE, iv)
            short_val = bs_put_price(spot, short_strike, T, RISK_FREE_RATE, iv)

        net_value = long_val - short_val  # What we'd get closing the spread
        return long_val, short_val, net_value

    def _round_strike(self, price, step=100):
        """Round to nearest BankNifty strike (100-point intervals)."""
        return round(price / step) * step

    def run(self) -> Dict:
        """Run the backtest."""
        self._compute_indicators()

        df = self.df
        c = self.config

        capital = c.capital
        equity = capital
        last_trade_idx = -c.min_days_between_trades
        trade_id = 0

        equity_curve = []

        for i in range(50, len(df)):  # Skip warmup
            row = df.iloc[i]
            current_date = str(row['date'])[:10] if hasattr(row['date'], 'strftime') else str(row['date'])[:10]

            # Check for entry signal
            direction = None
            if row['long_signal'] == 1 and (i - last_trade_idx) >= c.min_days_between_trades:
                direction = 'long'
            elif row['short_signal'] == 1 and (i - last_trade_idx) >= c.min_days_between_trades:
                direction = 'short'

            if direction is None:
                equity_curve.append({'date': current_date, 'equity': equity})
                continue

            # Compute DTE
            try:
                entry_date_obj = pd.Timestamp(row['date']).date()
            except:
                entry_date_obj = date.fromisoformat(current_date)

            dte, expiry = get_monthly_expiry(entry_date_obj, c.min_dte)

            spot = row['close']
            atr = row['atr']
            iv = row['est_iv']

            # Construct spread strikes
            if direction == 'long':
                # Bull Call Spread
                long_strike_raw = spot + c.long_strike_offset_atr * atr
                long_strike = self._round_strike(long_strike_raw)
                short_strike = self._round_strike(long_strike + c.spread_width_atr * atr)
            else:
                # Bear Put Spread
                long_strike_raw = spot - c.long_strike_offset_atr * atr
                long_strike = self._round_strike(long_strike_raw)
                short_strike = self._round_strike(long_strike - c.spread_width_atr * atr)

            # Price spread at entry
            _, _, net_debit = self._price_spread_at(spot, long_strike, short_strike, dte, iv, direction)

            if net_debit <= 0:
                equity_curve.append({'date': current_date, 'equity': equity})
                continue

            # Risk check: max loss = net_debit * lots * lot_size
            total_risk = net_debit * c.lots_per_trade * LOT_SIZE
            if total_risk > equity * c.max_risk_pct / 100:
                # Reduce lots to fit risk budget
                max_lots = int((equity * c.max_risk_pct / 100) / (net_debit * LOT_SIZE))
                if max_lots < 1:
                    equity_curve.append({'date': current_date, 'equity': equity})
                    continue
                lots = max_lots
            else:
                lots = c.lots_per_trade

            # Cost at entry: brokerage for 4 legs (buy long, sell short)
            entry_cost = c.brokerage_per_lot * lots * 2  # 2 legs
            slippage_cost = c.slippage_points * 2 * lots * LOT_SIZE  # 2 legs

            # Max spread value (profit target) and min (stop loss)
            spread_width_pts = abs(short_strike - long_strike)
            max_possible_value = spread_width_pts  # At expiry, max value = width if fully ITM

            profit_target_value = net_debit * (1 + c.profit_target_pct)
            stop_loss_value = net_debit * (1 - c.stop_loss_pct)

            # Simulate forward
            hold_end = min(i + c.hold_bars, len(df) - 1)

            # Check if expiry falls within hold period
            expiry_idx = None
            for j in range(i + 1, hold_end + 1):
                fwd_date = str(df.iloc[j]['date'])[:10]
                if fwd_date >= str(expiry):
                    expiry_idx = j
                    break

            if expiry_idx:
                hold_end = expiry_idx

            exit_reason = ExitReason.HOLD_COMPLETE
            exit_idx = hold_end
            max_spread_val = net_debit
            min_spread_val = net_debit

            # Check each bar for profit target / stop loss
            for j in range(i + 1, hold_end + 1):
                fwd_row = df.iloc[j]
                fwd_spot = fwd_row['close']
                days_elapsed = j - i
                remaining_dte = max(dte - days_elapsed, 0)

                # Re-estimate IV (use entry IV — simplification)
                _, _, spread_val = self._price_spread_at(
                    fwd_spot, long_strike, short_strike, remaining_dte, iv, direction
                )

                max_spread_val = max(max_spread_val, spread_val)
                min_spread_val = min(min_spread_val, spread_val)

                if spread_val >= profit_target_value:
                    exit_reason = ExitReason.PROFIT_TARGET
                    exit_idx = j
                    break
                elif spread_val <= stop_loss_value:
                    exit_reason = ExitReason.STOP_LOSS
                    exit_idx = j
                    break

            if exit_idx == hold_end and expiry_idx and exit_idx == expiry_idx:
                exit_reason = ExitReason.EXPIRY

            # Compute exit P&L
            exit_row = df.iloc[exit_idx]
            exit_spot = exit_row['close']
            exit_date_str = str(exit_row['date'])[:10]
            remaining_dte_exit = max(dte - (exit_idx - i), 0)

            long_exit, short_exit, exit_spread_val = self._price_spread_at(
                exit_spot, long_strike, short_strike, remaining_dte_exit, iv, direction
            )

            # P&L per lot in points
            pnl_per_lot = exit_spread_val - net_debit

            # Total P&L in Rs
            exit_cost = c.brokerage_per_lot * lots * 2
            total_costs = entry_cost + exit_cost + slippage_cost
            pnl_rupees = (pnl_per_lot * lots * LOT_SIZE) - total_costs

            # Build bias signal description
            bias_parts = ['SMA']
            if c.use_di_filter:
                bias_parts.append('DI')
            if c.use_rsi_filter:
                bias_parts.append('RSI')

            trade = DirectionalTrade(
                trade_id=trade_id,
                direction=direction,
                entry_date=current_date,
                exit_date=exit_date_str,
                entry_spot=round(spot, 2),
                exit_spot=round(exit_spot, 2),
                long_strike=long_strike,
                short_strike=short_strike,
                long_premium=round(self._price_spread_at(spot, long_strike, short_strike, dte, iv, direction)[0], 2),
                short_premium=round(self._price_spread_at(spot, long_strike, short_strike, dte, iv, direction)[1], 2),
                net_debit=round(net_debit, 2),
                long_exit_value=round(long_exit, 2),
                short_exit_value=round(short_exit, 2),
                net_credit_at_exit=round(exit_spread_val, 2),
                pnl_per_lot=round(pnl_per_lot, 2),
                pnl_rupees=round(pnl_rupees, 2),
                pnl_pct=round((pnl_per_lot / net_debit * 100) if net_debit > 0 else 0, 2),
                lots=lots,
                max_spread_value=round(max_spread_val, 2),
                min_spread_value=round(min_spread_val, 2),
                entry_atr=round(atr, 2),
                entry_iv=round(iv, 4),
                dte_at_entry=dte,
                expiry_date=str(expiry),
                days_held=exit_idx - i,
                exit_reason=exit_reason.value,
                squeeze_bars=int(row.get('squeeze_run_len', 0)),
                bias_signal='+'.join(bias_parts),
            )

            self.trades.append(trade)
            equity += pnl_rupees
            last_trade_idx = exit_idx
            trade_id += 1

            equity_curve.append({'date': current_date, 'equity': equity})

        self.equity_curve = equity_curve
        return self._compute_summary(equity)

    def _compute_summary(self, final_equity) -> Dict:
        """Compute backtest summary statistics."""
        c = self.config

        if not self.trades:
            return {'total_trades': 0, 'cagr': 0, 'sharpe': 0, 'max_drawdown': 0}

        total_trades = len(self.trades)
        long_trades = [t for t in self.trades if t.direction == 'long']
        short_trades = [t for t in self.trades if t.direction == 'short']

        winners = [t for t in self.trades if t.pnl_rupees > 0]
        losers = [t for t in self.trades if t.pnl_rupees <= 0]

        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        total_pnl = sum(t.pnl_rupees for t in self.trades)
        avg_win = np.mean([t.pnl_rupees for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl_rupees for t in losers]) if losers else 0

        # Profit factor
        gross_profit = sum(t.pnl_rupees for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_rupees for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # CAGR
        total_return = final_equity / c.capital
        # Estimate years from first to last trade
        first_date = pd.Timestamp(self.trades[0].entry_date)
        last_date = pd.Timestamp(self.trades[-1].exit_date)
        years = max((last_date - first_date).days / 365.25, 0.1)
        cagr = (total_return ** (1 / years) - 1) * 100

        # Sharpe from trade returns
        trade_returns = [t.pnl_pct / 100 for t in self.trades]
        if len(trade_returns) > 1:
            avg_ret = np.mean(trade_returns)
            std_ret = np.std(trade_returns, ddof=1)
            # Annualize: assume ~24 trades/year
            trades_per_year = total_trades / years
            sharpe = (avg_ret * trades_per_year) / (std_ret * np.sqrt(trades_per_year)) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Max drawdown from equity curve
        if self.equity_curve:
            equities = [e['equity'] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                peak = max(peak, eq)
                dd = (peak - eq) / peak * 100
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0

        calmar = cagr / max_dd if max_dd > 0 else 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Avg P&L by direction
        long_pnl = np.mean([t.pnl_pct for t in long_trades]) if long_trades else 0
        short_pnl = np.mean([t.pnl_pct for t in short_trades]) if short_trades else 0

        return {
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 0),
            'avg_win': round(avg_win, 0),
            'avg_loss': round(avg_loss, 0),
            'avg_pnl_pct': round(np.mean([t.pnl_pct for t in self.trades]), 2),
            'long_avg_pnl_pct': round(long_pnl, 2),
            'short_avg_pnl_pct': round(short_pnl, 2),
            'cagr': round(cagr, 2),
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2),
            'calmar': round(calmar, 2),
            'final_equity': round(final_equity, 0),
            'total_return_pct': round((final_equity / c.capital - 1) * 100, 2),
            'exit_reasons': exit_reasons,
            'avg_days_held': round(np.mean([t.days_held for t in self.trades]), 1),
            'avg_dte': round(np.mean([t.dte_at_entry for t in self.trades]), 1),
            'avg_debit_pts': round(np.mean([t.net_debit for t in self.trades]), 1),
            'avg_squeeze_bars': round(np.mean([t.squeeze_bars for t in self.trades]), 1),
        }

    def get_trade_log(self) -> List[Dict]:
        """Get all trades as list of dicts for CSV export."""
        return [
            {
                'trade_id': t.trade_id,
                'direction': t.direction,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_spot': t.entry_spot,
                'exit_spot': t.exit_spot,
                'long_strike': t.long_strike,
                'short_strike': t.short_strike,
                'net_debit': t.net_debit,
                'net_credit_at_exit': t.net_credit_at_exit,
                'pnl_per_lot': t.pnl_per_lot,
                'pnl_rupees': t.pnl_rupees,
                'pnl_pct': t.pnl_pct,
                'lots': t.lots,
                'days_held': t.days_held,
                'dte_at_entry': t.dte_at_entry,
                'exit_reason': t.exit_reason,
                'squeeze_bars': t.squeeze_bars,
                'bias_signal': t.bias_signal,
                'entry_atr': t.entry_atr,
                'entry_iv': t.entry_iv,
            }
            for t in self.trades
        ]
