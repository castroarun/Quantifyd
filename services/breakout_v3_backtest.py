"""
Breakout V3 Multi-Strategy Backtest Service
============================================

Loads the 9,739-trade enhanced dataset and applies V3 OR-combination
strategy filters. Computes equity curves, metrics, and year-by-year
breakdown for all 5 system presets and 6 individual strategies.

See docs/BREAKOUT-FILTER-OPTIMIZATION.md sections 14-18 for research.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from services.consolidation_breakout import (
    SYSTEM_SNIPER, SYSTEM_PRIMARY, SYSTEM_BALANCED,
    SYSTEM_ACTIVE, SYSTEM_HIGH_VOLUME,
    STRATEGY_ALPHA, STRATEGY_T1A, STRATEGY_T1B,
    STRATEGY_MOMVOL, STRATEGY_CALMAR, STRATEGY_BB_MOM,
    check_strategy_match, check_system_match,
)

logger = logging.getLogger(__name__)

CSV_PATH = Path(__file__).parent.parent / 'breakout_analysis_enhanced.csv'
CACHE_PATH = Path(__file__).parent.parent / 'backtest_data' / 'breakout_v3_results.json'
CAPITAL_PER_TRADE = 100_000  # ₹1 Lakh equal allocation per trade

SYSTEMS = {
    'SNIPER': SYSTEM_SNIPER,
    'PRIMARY': SYSTEM_PRIMARY,
    'BALANCED': SYSTEM_BALANCED,
    'ACTIVE': SYSTEM_ACTIVE,
    'HIGH_VOLUME': SYSTEM_HIGH_VOLUME,
}

STRATEGIES = {
    'ALPHA': STRATEGY_ALPHA,
    'T1A': STRATEGY_T1A,
    'T1B': STRATEGY_T1B,
    'MOMVOL': STRATEGY_MOMVOL,
    'CALMAR': STRATEGY_CALMAR,
    'BB_MOM': STRATEGY_BB_MOM,
}


def _row_to_dict(row):
    """Convert a DataFrame row to dict for check_strategy_match."""
    return {
        'rsi14': row.get('rsi14', 0) or 0,
        'rsi7': row.get('rsi7', 0) or 0,
        'volume_ratio': row.get('volume_ratio', 0) or 0,
        'vol_trend': row.get('vol_trend', 0) or 0,
        'breakout_pct': row.get('breakout_pct', 0) or 0,
        'ath_proximity': row.get('ath_proximity', 0) or 0,
        'williams_r': row.get('williams_r', -100) if pd.notna(row.get('williams_r')) else -100,
        'mom_60d': row.get('mom_60d', 0) or 0,
        'mom_10d': row.get('mom_10d', 0) or 0,
        'bb_pct_b': row.get('bb_pct_b', 0) or 0,
        'ema20_above_50': int(row.get('ema20_above_50', 0) or 0),
        'w_ema20_gt_50': int(row.get('w_ema20_gt_50', 0) or 0),
    }


def _apply_system_mask(df, system):
    """Vectorized: return boolean mask for trades matching ANY strategy in system."""
    mask = pd.Series(False, index=df.index)
    for strategy in system:
        strat_mask = pd.Series(True, index=df.index)
        for key, value in strategy.items():
            if key == 'min_rsi14':
                strat_mask &= df['rsi14'].fillna(0) >= value
            elif key == 'min_rsi7':
                strat_mask &= df['rsi7'].fillna(0) >= value
            elif key == 'min_volume_ratio':
                strat_mask &= df['volume_ratio'].fillna(0) >= value
            elif key == 'min_vol_trend':
                strat_mask &= df['vol_trend'].fillna(0) >= value
            elif key == 'min_breakout_pct':
                strat_mask &= df['breakout_pct'].fillna(0) >= value
            elif key == 'min_ath_proximity':
                strat_mask &= df['ath_proximity'].fillna(0) >= value
            elif key == 'min_williams_r':
                strat_mask &= df['williams_r'].fillna(-100) >= value
            elif key == 'min_mom_60d':
                strat_mask &= df['mom_60d'].fillna(0) >= value
            elif key == 'min_mom_10d':
                strat_mask &= df['mom_10d'].fillna(0) >= value
            elif key == 'min_bb_pct_b':
                strat_mask &= df['bb_pct_b'].fillna(0) >= value
            elif key == 'require_ema20_gt_50' and value:
                strat_mask &= df['ema20_above_50'].fillna(0) == 1
            elif key == 'require_weekly_ema20_gt_50' and value:
                strat_mask &= df['w_ema20_gt_50'].fillna(0) == 1
        mask |= strat_mask
    return mask


def _compute_metrics(trades_df):
    """Compute performance metrics for a set of trades."""
    n = len(trades_df)
    if n == 0:
        return {'trades': 0, 'win_pct': 0, 'profit_factor': 0,
                'avg_return': 0, 'cagr': 0, 'max_drawdown': 0, 'calmar': 0}

    returns = trades_df['trade_return']
    winners = returns > 0
    win_pct = round(winners.mean() * 100, 1)

    avg_gain = returns[winners].mean() if winners.any() else 0
    avg_loss = abs(returns[~winners].mean()) if (~winners).any() else 0.001
    profit_factor = round(avg_gain / avg_loss, 2) if avg_loss > 0 else 999

    # Equity curve for CAGR / drawdown
    sorted_trades = trades_df.sort_values('date')
    equity = CAPITAL_PER_TRADE
    equity_series = [equity]
    for ret in sorted_trades['trade_return']:
        equity += CAPITAL_PER_TRADE * (ret / 100)
        equity_series.append(equity)

    equity_arr = np.array(equity_series)
    total_invested = CAPITAL_PER_TRADE * n
    total_pnl = equity_arr[-1] - CAPITAL_PER_TRADE

    # Date range for CAGR
    dates = pd.to_datetime(sorted_trades['date'])
    if len(dates) >= 2:
        years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
        years = max(years, 0.5)
    else:
        years = 1.0

    final_ratio = equity_arr[-1] / CAPITAL_PER_TRADE
    cagr = round((final_ratio ** (1 / years) - 1) * 100, 2) if final_ratio > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak * 100
    max_dd = round(drawdown.max(), 2)

    calmar = round(cagr / max_dd, 2) if max_dd > 0 else 999

    return {
        'trades': n,
        'win_pct': win_pct,
        'profit_factor': profit_factor,
        'avg_return': round(returns.mean(), 2),
        'avg_win': round(avg_gain, 2),
        'avg_loss': round(-abs(avg_loss), 2),
        'cagr': cagr,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'total_pnl': round(total_pnl, 0),
        'signals_per_year': round(n / max(years, 0.5), 1),
    }


def _compute_equity_curve(trades_df):
    """Build cumulative equity curve from chronological trades."""
    if len(trades_df) == 0:
        return []
    sorted_trades = trades_df.sort_values('date')
    equity = CAPITAL_PER_TRADE
    curve = []
    for _, row in sorted_trades.iterrows():
        equity += CAPITAL_PER_TRADE * (row['trade_return'] / 100)
        curve.append({
            'date': str(row['date'])[:10],
            'equity': round(equity, 0),
        })
    return curve


def _year_breakdown(trades_df):
    """Per-year metrics breakdown."""
    if len(trades_df) == 0:
        return []
    df = trades_df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    breakdown = []
    for year in sorted(df['year'].unique()):
        year_trades = df[df['year'] == year]
        returns = year_trades['trade_return']
        n = len(year_trades)
        wins = (returns > 0).sum()
        win_pct = round(wins / n * 100, 1) if n > 0 else 0
        avg_gain = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns <= 0].mean()) if (returns <= 0).any() else 0.001
        pf = round(avg_gain / avg_loss, 2) if avg_loss > 0 else 999
        breakdown.append({
            'year': int(year),
            'trades': n,
            'wins': wins,
            'losses': n - wins,
            'win_pct': win_pct,
            'profit_factor': pf,
            'avg_return': round(returns.mean(), 2),
            'total_return': round(returns.sum(), 2),
        })
    return breakdown


def _recent_signals(trades_df, limit=50):
    """Get most recent qualifying signals."""
    if len(trades_df) == 0:
        return []
    recent = trades_df.sort_values('date', ascending=False).head(limit)
    signals = []
    for _, row in recent.iterrows():
        signals.append({
            'date': str(row['date'])[:10],
            'symbol': row['symbol'],
            'detector': row.get('detector', ''),
            'return_pct': round(row['trade_return'], 1),
            'outcome': 'WIN' if row['trade_return'] > 0 else 'LOSS',
            'exit_reason': row.get('exit_reason', ''),
            'rsi14': round(row.get('rsi14', 0) or 0, 1),
            'volume_ratio': round(row.get('volume_ratio', 0) or 0, 1),
            'breakout_pct': round(row.get('breakout_pct', 0) or 0, 1),
            'ath_proximity': round(row.get('ath_proximity', 0) or 0, 1),
        })
    return signals


def _identify_matched_strategies(row_dict):
    """Return list of strategy names that match this trade."""
    matched = []
    for name, strategy in STRATEGIES.items():
        if check_strategy_match(row_dict, strategy):
            matched.append(name)
    return matched


def run_v3_backtest(progress_callback=None):
    """Run the full V3 backtest analysis and return results dict."""
    if progress_callback:
        progress_callback(5, 'Loading enhanced trade data...')

    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'])
    total = len(df)
    date_min = str(df['date'].min())[:10]
    date_max = str(df['date'].max())[:10]
    n_symbols = df['symbol'].nunique()

    logger.info(f"Loaded {total} trades, {n_symbols} symbols, {date_min} to {date_max}")

    if progress_callback:
        progress_callback(15, f'Loaded {total} trades across {n_symbols} stocks')

    # Baseline metrics (all trades, no filter)
    baseline = _compute_metrics(df)

    # Per-system analysis
    systems_results = {}
    system_names = list(SYSTEMS.keys())
    for i, (name, system) in enumerate(SYSTEMS.items()):
        if progress_callback:
            pct = 20 + int(50 * i / len(SYSTEMS))
            progress_callback(pct, f'Analyzing system: {name}...')

        mask = _apply_system_mask(df, system)
        filtered = df[mask].copy()

        # Identify which strategy matched each trade
        strategy_counts = {}
        for sname in STRATEGIES:
            s_mask = _apply_system_mask(filtered, [STRATEGIES[sname]])
            strategy_counts[sname] = int(s_mask.sum())

        systems_results[name] = {
            'metrics': _compute_metrics(filtered),
            'equity_curve': _compute_equity_curve(filtered),
            'year_breakdown': _year_breakdown(filtered),
            'strategy_breakdown': strategy_counts,
            'recent_signals': _recent_signals(filtered),
            'sector_distribution': filtered['symbol'].value_counts().head(20).to_dict() if len(filtered) > 0 else {},
        }

    if progress_callback:
        progress_callback(75, 'Analyzing individual strategies...')

    # Per-strategy analysis
    strategies_results = {}
    for name, strategy in STRATEGIES.items():
        mask = _apply_system_mask(df, [strategy])
        filtered = df[mask]
        strategies_results[name] = {
            'metrics': _compute_metrics(filtered),
            'description': _strategy_description(name),
        }

    if progress_callback:
        progress_callback(90, 'Building final results...')

    results = {
        'generated_at': datetime.now().isoformat(),
        'total_trades': total,
        'total_symbols': n_symbols,
        'date_range': f'{date_min} to {date_max}',
        'baseline': baseline,
        'systems': systems_results,
        'strategies': strategies_results,
    }

    # Cache to JSON
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(results, f, default=str)
    logger.info(f"V3 backtest results cached to {CACHE_PATH}")

    if progress_callback:
        progress_callback(100, 'Complete!')

    return results


def get_cached_results():
    """Load cached results if available."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return None


def _strategy_description(name):
    """Human-readable strategy description."""
    descs = {
        'ALPHA': 'RSI(14)>=75 + Volume>=3x + EMA20>EMA50',
        'T1A': 'BO>=3% + VolTrend>=1.2 + ATH>=90% + Weekly EMA20>50 + RSI7>=80',
        'T1B': 'BO>=3% + VolTrend>=1.2 + ATH>=85% + EMA20>50 + Weekly EMA20>50 + WillR>=-20',
        'MOMVOL': 'Mom60>=15% + VolTrend>=1.2 + ATH>=90% + Volume>=3x',
        'CALMAR': 'Volume>=5x + VolTrend>=1.2 + EMA20>50 + RSI7>=80',
        'BB_MOM': 'BB>Upper + Mom60>=15% + VolTrend>=1.2 + ATH>=90%',
    }
    return descs.get(name, name)
