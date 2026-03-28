"""
NAS ATM Straddle Backtest Engine
=================================
Backtests the ATR Squeeze → ATM Straddle → 30% SL per-leg → Re-entry cycle.

Strategy:
1. Entry: ATR(14) < ATR_MA(50) on 5-min candles → sell ATM straddle
2. SL: 30% per leg (entry × 1.30)
3. On SL hit: close stopped leg, trail surviving leg SL to cost, re-enter new ATM straddle
4. EOD: close all at 15:15
5. Max re-entries per day: configurable (default 5)

Premium simulation:
- Uses Black-Scholes to estimate ATM option premiums from spot + IV + DTE
- IV estimated from daily ATR (ATR/spot as annualized vol proxy)

Usage:
    from services.nas_atm_backtest_engine import NasAtmBacktestEngine, NasAtmBacktestConfig

    config = NasAtmBacktestConfig(symbol='NIFTY50')
    engine = NasAtmBacktestEngine(config)
    result = engine.run()
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from scipy.stats import norm

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'

# Index configurations
INDEX_CONFIG = {
    'NIFTY50': {
        'lot_size': 75,
        'strike_interval': 50,
        'symbol_db': 'NIFTY50',
    },
    'BANKNIFTY': {
        'lot_size': 30,
        'strike_interval': 100,
        'symbol_db': 'BANKNIFTY',
    },
}


@dataclass
class NasAtmBacktestConfig:
    """Configuration for ATM straddle backtest."""
    symbol: str = 'NIFTY50'
    start_date: str = '2024-04-01'
    end_date: str = '2026-03-27'

    # ATR Squeeze parameters
    atr_period: int = 14
    atr_ma_period: int = 50
    min_squeeze_bars: int = 1

    # Straddle parameters
    lots_per_leg: int = 5
    leg_sl_pct: float = 0.30          # 30% SL per leg
    max_reentries: int = 5            # Max re-entries per day
    trail_to_cost_on_sl: bool = True  # Trail surviving leg SL to cost

    # Time windows
    entry_start_time: str = '09:20'   # Earliest entry
    entry_end_time: str = '14:50'     # Latest entry
    eod_exit_time: str = '15:15'      # EOD squareoff

    # Squeeze occurrence filter
    squeeze_occurrence: int = 1       # Enter on Nth squeeze within the day
                                      # 1 = first time ATR drops below MA
                                      # 2 = second time (after ATR went above then below again)
                                      # 3 = third occurrence, etc.

    # IV estimation
    iv_lookback: int = 20             # Days for IV estimation
    default_iv: float = 0.15          # Fallback IV
    dte_for_premium: float = 0.02     # ~0.02 years ≈ weekly option (~5 trading days)


@dataclass
class BtLeg:
    """A single option leg in the backtest."""
    leg_type: str          # 'CE' or 'PE'
    strike: float
    entry_price: float     # Premium at entry
    sl_price: float        # Stop loss price
    entry_time: str
    strangle_id: int
    is_active: bool = True
    exit_price: float = 0.0
    exit_time: str = ''
    exit_reason: str = ''


@dataclass
class BtDayResult:
    """Results for a single trading day."""
    date: str
    spot_open: float
    spot_close: float
    num_entries: int = 0
    num_reentries: int = 0
    num_sl_hits: int = 0
    legs_traded: int = 0
    gross_collected: float = 0.0
    gross_paid: float = 0.0
    net_pnl: float = 0.0
    is_squeeze_day: bool = False
    trades: List[dict] = field(default_factory=list)


@dataclass
class NasAtmBacktestResult:
    """Complete backtest results."""
    symbol: str = ''
    config: dict = field(default_factory=dict)
    total_trading_days: int = 0
    squeeze_days: int = 0
    entry_days: int = 0

    # P&L
    total_pnl: float = 0.0
    total_collected: float = 0.0
    total_paid: float = 0.0
    avg_daily_pnl: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0

    # Win/Loss
    winning_days: int = 0
    losing_days: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade stats
    total_strangles: int = 0
    total_reentries: int = 0
    total_sl_hits: int = 0
    avg_reentries_per_day: float = 0.0
    avg_premium_collected: float = 0.0

    # Curves
    daily_results: List[BtDayResult] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)


# ── Black-Scholes Premium Estimation ─────────────────────

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


def estimate_atm_premium(spot, iv, dte_years=0.02, r=0.07):
    """
    Estimate ATM straddle premium (CE + PE) using Black-Scholes.
    For ATM: strike ≈ spot, so CE ≈ PE ≈ spot * sigma * sqrt(T) * 0.4
    Returns (ce_premium, pe_premium)
    """
    strike = spot  # ATM
    ce = bs_call_price(spot, strike, dte_years, r, iv)
    pe = bs_put_price(spot, strike, dte_years, r, iv)
    return ce, pe


def estimate_premium_at_spot(spot, strike, leg_type, iv, dte_years=0.02, r=0.07):
    """Estimate option premium at a given spot for a specific strike."""
    if leg_type == 'CE':
        return bs_call_price(spot, strike, dte_years, r, iv)
    else:
        return bs_put_price(spot, strike, dte_years, r, iv)


# ── Main Engine ──────────────────────────────────────────

class NasAtmBacktestEngine:
    """Backtests the ATM straddle system on 5-min index data."""

    def __init__(self, config: NasAtmBacktestConfig = None):
        self.config = config or NasAtmBacktestConfig()
        idx = INDEX_CONFIG.get(self.config.symbol, INDEX_CONFIG['NIFTY50'])
        self.lot_size = idx['lot_size']
        self.strike_interval = idx['strike_interval']
        self.symbol_db = idx['symbol_db']

    def _load_data(self) -> pd.DataFrame:
        """Load 5-min OHLCV data from centralized DB."""
        conn = sqlite3.connect(str(DB_PATH))
        query = """
            SELECT date, open, high, low, close, volume
            FROM market_data_unified
            WHERE symbol = ? AND timeframe = '5minute'
              AND date >= ? AND date <= ?
            ORDER BY date
        """
        df = pd.read_sql_query(
            query, conn,
            params=(self.symbol_db, self.config.start_date, self.config.end_date + ' 23:59:59')
        )
        conn.close()

        df['date'] = pd.to_datetime(df['date'])
        df['time'] = df['date'].dt.time
        df['day'] = df['date'].dt.date.astype(str)
        logger.info(f"Loaded {len(df)} 5-min candles for {self.symbol_db}, "
                     f"{df['day'].nunique()} trading days")
        return df

    def _compute_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ATR(14) and ATR_MA(50) using Wilder's RMA.

        Handles overnight gaps: for the first candle of each day,
        use only high-low (not prev close) to avoid gap inflation.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        days = df['day'].values
        n = len(df)

        # True Range — skip gap candles
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            if days[i] != days[i-1]:
                # First candle of new day — use only intraday range
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i],
                            abs(high[i] - close[i-1]),
                            abs(low[i] - close[i-1]))

        # Wilder's RMA for ATR
        period = self.config.atr_period
        atr = np.zeros(n)
        atr[:period] = np.nan
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

        # SMA of ATR
        ma_period = self.config.atr_ma_period
        atr_ma = np.zeros(n)
        atr_ma[:] = np.nan
        for i in range(ma_period - 1, n):
            atr_ma[i] = np.mean(atr[i - ma_period + 1:i + 1])

        df = df.copy()
        df['atr'] = atr
        df['atr_ma'] = atr_ma
        df['is_squeeze'] = df['atr'] < df['atr_ma']

        return df

    def _estimate_iv(self, daily_returns: pd.Series) -> float:
        """Estimate annualized IV from recent daily returns."""
        if len(daily_returns) < 5:
            return self.config.default_iv
        vol = daily_returns.std() * np.sqrt(252)
        return max(vol, 0.08)  # Floor at 8%

    def _get_atm_strike(self, spot: float) -> float:
        """Round spot to nearest strike interval."""
        return round(spot / self.strike_interval) * self.strike_interval

    def _time_in_range(self, t: dtime, start_str: str, end_str: str) -> bool:
        """Check if time is within entry window."""
        start = dtime(*[int(x) for x in start_str.split(':')])
        end = dtime(*[int(x) for x in end_str.split(':')])
        return start <= t <= end

    def _compute_daily_atr_squeeze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ATR squeeze using a ROLLING window across days.
        ATR(14) and ATR_MA(50) are computed on the continuous 5-min series,
        but squeeze detection only happens after the ATR_MA has warmed up
        (first ~50 candles of the dataset, NOT per-day).
        """
        return self._compute_atr(df)

    def run(self) -> NasAtmBacktestResult:
        """Run the full backtest."""
        cfg = self.config
        df = self._load_data()
        if df.empty:
            logger.error(f"No data found for {self.symbol_db}")
            return NasAtmBacktestResult(symbol=cfg.symbol)

        df = self._compute_daily_atr_squeeze(df)

        # Group by trading day
        days = df.groupby('day')
        trading_days = sorted(days.groups.keys())

        # Build daily close series for IV estimation
        daily_closes = df.groupby('day')['close'].last()
        daily_returns = daily_closes.pct_change().dropna()

        result = NasAtmBacktestResult(
            symbol=cfg.symbol,
            config={
                'atr_period': cfg.atr_period,
                'atr_ma_period': cfg.atr_ma_period,
                'leg_sl_pct': cfg.leg_sl_pct,
                'max_reentries': cfg.max_reentries,
                'lots_per_leg': cfg.lots_per_leg,
                'lot_size': self.lot_size,
                'strike_interval': self.strike_interval,
            },
            total_trading_days=len(trading_days),
        )

        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_dd = 0.0

        for day_str in trading_days:
            day_df = days.get_group(day_str).copy()
            day_df = day_df.sort_values('date').reset_index(drop=True)

            # Estimate IV for the day using recent returns
            day_idx = daily_returns.index.get_loc(day_str) if day_str in daily_returns.index else -1
            if day_idx >= cfg.iv_lookback:
                iv = self._estimate_iv(daily_returns.iloc[day_idx - cfg.iv_lookback:day_idx])
            else:
                iv = cfg.default_iv

            day_result = self._simulate_day(day_df, iv, day_str)
            result.daily_results.append(day_result)

            if day_result.is_squeeze_day:
                result.squeeze_days += 1
            if day_result.num_entries > 0:
                result.entry_days += 1

            result.total_strangles += day_result.num_entries
            result.total_reentries += day_result.num_reentries
            result.total_sl_hits += day_result.num_sl_hits
            result.total_collected += day_result.gross_collected
            result.total_paid += day_result.gross_paid

            cumulative_pnl += day_result.net_pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            dd = peak_pnl - cumulative_pnl
            max_dd = max(max_dd, dd)

            result.equity_curve.append({
                'date': day_str,
                'pnl': day_result.net_pnl,
                'cumulative_pnl': cumulative_pnl,
            })

            # Monthly returns
            month_key = day_str[:7]
            result.monthly_returns[month_key] = result.monthly_returns.get(month_key, 0) + day_result.net_pnl

        # Compute summary stats
        result.total_pnl = cumulative_pnl
        result.max_drawdown = max_dd

        daily_pnls = [d.net_pnl for d in result.daily_results if d.num_entries > 0]
        if daily_pnls:
            result.avg_daily_pnl = np.mean(daily_pnls)
            result.best_day = max(daily_pnls)
            result.worst_day = min(daily_pnls)
            result.winning_days = sum(1 for p in daily_pnls if p > 0)
            result.losing_days = sum(1 for p in daily_pnls if p <= 0)
            result.win_rate = result.winning_days / len(daily_pnls) * 100 if daily_pnls else 0

            gross_profit = sum(p for p in daily_pnls if p > 0)
            gross_loss = abs(sum(p for p in daily_pnls if p < 0))
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Sharpe (daily, annualized)
            if len(daily_pnls) > 1:
                daily_std = np.std(daily_pnls, ddof=1)
                if daily_std > 0:
                    result.sharpe_ratio = (np.mean(daily_pnls) / daily_std) * np.sqrt(252)
                    downside = [p for p in daily_pnls if p < 0]
                    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else daily_std
                    if downside_std > 0:
                        result.sortino_ratio = (np.mean(daily_pnls) / downside_std) * np.sqrt(252)

            # Max DD %
            initial_capital = cfg.lots_per_leg * self.lot_size * 2 * 100  # rough margin estimate
            if initial_capital > 0:
                result.max_drawdown_pct = max_dd / initial_capital * 100

        if result.entry_days > 0:
            result.avg_reentries_per_day = result.total_reentries / result.entry_days

        if result.total_strangles > 0:
            qty = cfg.lots_per_leg * self.lot_size
            result.avg_premium_collected = result.total_collected / (result.total_strangles * qty * 2)

        return result

    def _simulate_day(self, day_df: pd.DataFrame, iv: float, day_str: str) -> BtDayResult:
        """Simulate one trading day."""
        cfg = self.config
        qty = cfg.lots_per_leg * self.lot_size  # Total qty per leg

        day_result = BtDayResult(
            date=day_str,
            spot_open=day_df.iloc[0]['open'],
            spot_close=day_df.iloc[-1]['close'],
        )

        active_legs: List[BtLeg] = []
        closed_legs: List[BtLeg] = []
        strangle_id = 0
        reentries = 0
        squeeze_active = False
        prev_squeeze = False          # Track squeeze transitions
        squeeze_occurrence_count = 0  # How many times squeeze turned ON today
        entry_allowed = False         # Only True when Nth occurrence reached

        eod_time = dtime(*[int(x) for x in cfg.eod_exit_time.split(':')])

        for i in range(len(day_df)):
            row = day_df.iloc[i]
            t = row['time']
            spot = row['close']
            candle_high = row['high']
            candle_low = row['low']

            # Check squeeze state and track occurrences
            if not np.isnan(row.get('atr', np.nan)) and not np.isnan(row.get('atr_ma', np.nan)):
                squeeze_active = row['is_squeeze']

            # Detect squeeze transition: OFF → ON
            if squeeze_active and not prev_squeeze:
                squeeze_occurrence_count += 1
                if squeeze_occurrence_count >= cfg.squeeze_occurrence:
                    entry_allowed = True
            prev_squeeze = squeeze_active

            if squeeze_active:
                day_result.is_squeeze_day = True

            # ── EOD Exit ──
            if t >= eod_time:
                for leg in active_legs:
                    if leg.is_active:
                        exit_prem = estimate_premium_at_spot(
                            spot, leg.strike, leg.leg_type, iv, cfg.dte_for_premium
                        )
                        leg.exit_price = exit_prem
                        leg.exit_time = str(row['date'])
                        leg.exit_reason = 'EOD'
                        leg.is_active = False
                        closed_legs.append(leg)
                break  # Stop processing after EOD

            # ── Check SL on active legs ──
            legs_to_close = []
            for leg in active_legs:
                if not leg.is_active:
                    continue

                # Estimate current premium from candle extremes
                # For short positions: premium increases when spot moves against us
                if leg.leg_type == 'CE':
                    # CE premium highest when spot is highest
                    worst_prem = estimate_premium_at_spot(
                        candle_high, leg.strike, 'CE', iv, cfg.dte_for_premium
                    )
                else:
                    # PE premium highest when spot is lowest
                    worst_prem = estimate_premium_at_spot(
                        candle_low, leg.strike, 'PE', iv, cfg.dte_for_premium
                    )

                if worst_prem >= leg.sl_price:
                    # SL hit
                    leg.exit_price = leg.sl_price  # Assume exit at SL level
                    leg.exit_time = str(row['date'])
                    leg.exit_reason = 'SL_HIT'
                    leg.is_active = False
                    closed_legs.append(leg)
                    legs_to_close.append(leg)
                    day_result.num_sl_hits += 1

            # Handle SL aftermath: trail surviving leg + re-enter
            if legs_to_close:
                # Trail surviving legs in the same strangle to cost
                for stopped_leg in legs_to_close:
                    sid = stopped_leg.strangle_id
                    for leg in active_legs:
                        if leg.is_active and leg.strangle_id == sid:
                            if cfg.trail_to_cost_on_sl:
                                leg.sl_price = leg.entry_price  # Trail to cost

                # Re-enter if within limits (re-entries don't need fresh squeeze occurrence)
                if (reentries < cfg.max_reentries and
                    self._time_in_range(t, cfg.entry_start_time, cfg.entry_end_time) and
                    entry_allowed and squeeze_active):

                    strike = self._get_atm_strike(spot)
                    ce_prem, pe_prem = estimate_atm_premium(spot, iv, cfg.dte_for_premium)

                    if ce_prem > 0 and pe_prem > 0:
                        strangle_id += 1
                        reentries += 1
                        day_result.num_reentries += 1

                        ce_leg = BtLeg(
                            leg_type='CE', strike=strike,
                            entry_price=ce_prem,
                            sl_price=ce_prem * (1 + cfg.leg_sl_pct),
                            entry_time=str(row['date']),
                            strangle_id=strangle_id,
                        )
                        pe_leg = BtLeg(
                            leg_type='PE', strike=strike,
                            entry_price=pe_prem,
                            sl_price=pe_prem * (1 + cfg.leg_sl_pct),
                            entry_time=str(row['date']),
                            strangle_id=strangle_id,
                        )
                        active_legs.extend([ce_leg, pe_leg])
                        day_result.gross_collected += (ce_prem + pe_prem) * qty

            # ── Entry: squeeze active, no current straddle, within time window ──
            has_active_straddle = any(
                l.is_active for l in active_legs
                if l.strangle_id == strangle_id  # Check latest strangle
            )

            if (entry_allowed and squeeze_active and
                strangle_id == 0 and  # No entry yet today
                self._time_in_range(t, cfg.entry_start_time, cfg.entry_end_time)):

                strike = self._get_atm_strike(spot)
                ce_prem, pe_prem = estimate_atm_premium(spot, iv, cfg.dte_for_premium)

                if ce_prem > 0 and pe_prem > 0:
                    strangle_id += 1
                    day_result.num_entries += 1

                    ce_leg = BtLeg(
                        leg_type='CE', strike=strike,
                        entry_price=ce_prem,
                        sl_price=ce_prem * (1 + cfg.leg_sl_pct),
                        entry_time=str(row['date']),
                        strangle_id=strangle_id,
                    )
                    pe_leg = BtLeg(
                        leg_type='PE', strike=strike,
                        entry_price=pe_prem,
                        sl_price=pe_prem * (1 + cfg.leg_sl_pct),
                        entry_time=str(row['date']),
                        strangle_id=strangle_id,
                    )
                    active_legs.extend([ce_leg, pe_leg])
                    day_result.gross_collected += (ce_prem + pe_prem) * qty

        # Close any remaining active legs at EOD (if not already closed)
        for leg in active_legs:
            if leg.is_active:
                spot = day_df.iloc[-1]['close']
                exit_prem = estimate_premium_at_spot(
                    spot, leg.strike, leg.leg_type, iv, cfg.dte_for_premium
                )
                leg.exit_price = exit_prem
                leg.exit_time = str(day_df.iloc[-1]['date'])
                leg.exit_reason = 'EOD'
                leg.is_active = False
                closed_legs.append(leg)

        # Compute day P&L: collected - paid (short straddle: we sell premium)
        day_result.gross_paid = sum(l.exit_price for l in closed_legs) * qty
        day_result.gross_collected = sum(l.entry_price for l in closed_legs) * qty
        day_result.net_pnl = day_result.gross_collected - day_result.gross_paid
        day_result.legs_traded = len(closed_legs)

        # Build trade records
        for leg in closed_legs:
            day_result.trades.append({
                'strangle_id': leg.strangle_id,
                'leg': leg.leg_type,
                'strike': leg.strike,
                'entry_price': round(leg.entry_price, 2),
                'exit_price': round(leg.exit_price, 2),
                'entry_time': leg.entry_time,
                'exit_time': leg.exit_time,
                'exit_reason': leg.exit_reason,
                'pnl': round((leg.entry_price - leg.exit_price) * qty, 2),
            })

        return day_result


def run_backtest(symbol='NIFTY50', **kwargs) -> NasAtmBacktestResult:
    """Convenience function to run a backtest."""
    config = NasAtmBacktestConfig(symbol=symbol, **kwargs)
    engine = NasAtmBacktestEngine(config)
    return engine.run()


def print_results(result: NasAtmBacktestResult):
    """Pretty-print backtest results."""
    r = result
    lot_val = r.config.get('lots_per_leg', 5) * r.config.get('lot_size', 75)

    print(f"\n{'='*60}")
    print(f"NAS ATM Straddle Backtest: {r.symbol}")
    print(f"{'='*60}")
    print(f"Period: {r.daily_results[0].date if r.daily_results else '?'} to "
          f"{r.daily_results[-1].date if r.daily_results else '?'}")
    print(f"Trading Days: {r.total_trading_days} | Squeeze Days: {r.squeeze_days} | Entry Days: {r.entry_days}")
    print(f"Lot size: {r.config.get('lot_size')} × {r.config.get('lots_per_leg')} lots = {lot_val} qty/leg")
    print()
    print(f"-- P&L ---------------------------------")
    print(f"Total P&L:       Rs {r.total_pnl:>12,.0f}")
    print(f"Avg Daily P&L:   Rs {r.avg_daily_pnl:>12,.0f}")
    print(f"Best Day:        Rs {r.best_day:>12,.0f}")
    print(f"Worst Day:       Rs {r.worst_day:>12,.0f}")
    print()
    print(f"-- Risk --------------------------------")
    print(f"Win Rate:        {r.win_rate:>8.1f}%")
    print(f"Profit Factor:   {r.profit_factor:>8.2f}")
    print(f"Sharpe Ratio:    {r.sharpe_ratio:>8.2f}")
    print(f"Sortino Ratio:   {r.sortino_ratio:>8.2f}")
    print(f"Max Drawdown:    Rs {r.max_drawdown:>12,.0f}")
    print()
    print(f"-- Trades ------------------------------")
    print(f"Total Strangles: {r.total_strangles:>8}")
    print(f"Total Re-entries:{r.total_reentries:>8}")
    print(f"Total SL Hits:   {r.total_sl_hits:>8}")
    print(f"Avg Re-entries/day:{r.avg_reentries_per_day:>6.1f}")
    print(f"Avg Premium (per lot): Rs {r.avg_premium_collected:>8.1f}")
    print()

    # Monthly returns table
    if r.monthly_returns:
        print(f"-- Monthly Returns ---------------------")
        for month, pnl in sorted(r.monthly_returns.items()):
            bar = '#' * max(1, int(abs(pnl) / 5000))
            marker = 'W' if pnl >= 0 else 'L'
            print(f"  {month}: [{marker}] Rs {pnl:>10,.0f}  {bar}")
