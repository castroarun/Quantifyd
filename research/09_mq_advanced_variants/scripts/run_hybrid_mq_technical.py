"""
Hybrid MQ Fundamentals + Technical Entry/Exit Backtest
======================================================

Tests whether combining MQ's fundamental stock selection (quality+momentum)
with the best technical entry/exit rules outperforms either approach alone.

Three universe modes:
  A. MQ_HELD: Only stocks currently in MQ portfolio (~30 at any time)
  B. MQ_CANDIDATES: All stocks passing MQ's ATH proximity filter (~50-150)
  C. ALL_STOCKS: Full Nifty 500 (375 stocks) - pure technical baseline

Technical configs (top 4 from strategy exploration):
  1. MACD+SMA200 entry + Trail 20% exit (best Calmar)
  2. MACD+SMA200 entry + SuperTrend exit (best combined)
  3. ADX25 entry + ADX Weak exit (best trend following)
  4. SMA200 entry + ATH 20% exit + Composite Momentum rank (highest CAGR)

Also runs pure MQ baseline for comparison.

Output: hybrid_mq_technical_results.csv (incremental)
"""

import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from services.mq_backtest_engine import MQBacktestEngine, _run_screening
from services.mq_portfolio import MQBacktestConfig
from services.strategy_backtest import (
    preload_exploration_data, enrich_with_indicators,
    load_enriched_cache, save_enriched_cache,
    StrategyConfig, StrategyExplorer,
    # Entry functions
    entry_macd_crossover, entry_adx_trending,
    entry_price_above_sma, entry_supertrend_bullish,
    # Rank functions
    rank_momentum_12m, rank_composite_momentum, rank_rsi_strength,
    # Exit functions
    exit_trailing_stop, exit_ath_drawdown, exit_time_based,
    exit_indicator_reversal, exit_fixed_stop_loss,
    exit_adx_weakening, make_combined_exit,
)
from services.kc6_backtest_engine import extract_mq_daily_holdings

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'hybrid_mq_technical_results.csv')

FIELDNAMES = [
    'label', 'universe_mode', 'tech_config',
    'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
    'profit_factor', 'total_trades', 'win_rate',
    'avg_win_pct', 'avg_loss_pct',
    'final_value', 'total_return_pct',
    'top3_pnl_pct', 'top3_symbols', 'cagr_ex_top3',
    'exit_reason_counts',
]

START_DATE = '2015-01-01'
END_DATE = '2025-12-31'
INITIAL_CAPITAL = 10_000_000


# =============================================================================
# Build MQ universes (daily holdings + periodic candidate lists)
# =============================================================================

def build_mq_held_universe(mq_result):
    """Build date -> set(symbols) from MQ daily holdings."""
    return extract_mq_daily_holdings(mq_result)


def build_mq_candidate_universe(universe, price_data, config):
    """
    Run MQ screening at each semi-annual rebalance point to get the full
    candidate list (all stocks passing ATH proximity filter).
    Returns date -> set(symbols) map, where each rebalance date's candidates
    persist until the next rebalance.
    """
    # Get all trading days
    all_dates = set()
    for sym, df in price_data.items():
        start_dt = pd.Timestamp(START_DATE)
        end_dt = pd.Timestamp(END_DATE)
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        all_dates.update(df.index[mask].tolist())
    trading_days = sorted(all_dates)

    # Find rebalance dates (1st trading day of Jan and Jul)
    rebalance_dates = []
    for d in trading_days:
        if d.month in [1, 7] and d.day <= 7:
            if not rebalance_dates or rebalance_dates[-1].month != d.month or rebalance_dates[-1].year != d.year:
                rebalance_dates.append(d)

    # Screen at each rebalance point
    rebalance_candidates = {}
    for rd in rebalance_dates:
        candidates = _run_screening(universe, price_data, rd, config)
        syms = {c['symbol'] for c in candidates}
        rebalance_candidates[rd] = syms

    # Build daily map: each day uses the most recent rebalance's candidates
    daily_candidates = {}
    current_syms = set()
    rb_idx = 0

    for d in trading_days:
        # Check if there's a new rebalance
        while rb_idx < len(rebalance_dates) and rebalance_dates[rb_idx] <= d:
            current_syms = rebalance_candidates[rebalance_dates[rb_idx]]
            rb_idx += 1
        date_str = d.strftime('%Y-%m-%d')
        daily_candidates[date_str] = current_syms

    return daily_candidates


def make_allowed_fn(daily_map):
    """Convert date -> set(symbols) map to a function for StrategyExplorer."""
    def fn(date):
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime('%Y-%m-%d')
        return daily_map.get(date_str, set())
    return fn


# =============================================================================
# Technical strategy configs
# =============================================================================

def get_tech_configs():
    """Define the 4 best technical configs from exploration."""
    configs = []

    # 1. MACD+SMA200 + Trail 20% (best Calmar from exploration)
    def entry_macd_sma200(df, idx):
        return entry_macd_crossover(df, idx) and entry_price_above_sma(df, idx, period=200)

    configs.append({
        'name': 'MACD_SMA200_Trail20',
        'entry_fn': entry_macd_sma200,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_trailing_stop, {'pct': 20}),
            (exit_time_based, {'max_days': 252}),
        ]),
        'config': StrategyConfig(
            name='MACD_SMA200_Trail20',
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            portfolio_size=20, rebalance_freq='monthly',
        ),
    })

    # 2. MACD+SMA200 + SuperTrend exit (best combined CAGR+Calmar)
    configs.append({
        'name': 'MACD_SMA200_STExit',
        'entry_fn': entry_macd_sma200,
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_indicator_reversal, {'indicator': 'supertrend_10_3'}),
            (exit_fixed_stop_loss, {'pct': 12}),
            (exit_time_based, {'max_days': 180}),
        ]),
        'config': StrategyConfig(
            name='MACD_SMA200_STExit',
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            portfolio_size=20, rebalance_freq='monthly',
        ),
    })

    # 3. ADX25 + ADX Weak exit (best trend following)
    configs.append({
        'name': 'ADX25_ADXWeakExit',
        'entry_fn': lambda df, idx: entry_adx_trending(df, idx, threshold=25),
        'rank_fn': rank_momentum_12m,
        'exit_fn': make_combined_exit([
            (exit_adx_weakening, {'threshold': 20}),
            (exit_trailing_stop, {'pct': 15}),
            (exit_time_based, {'max_days': 252}),
        ]),
        'config': StrategyConfig(
            name='ADX25_ADXWeakExit',
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            portfolio_size=20, rebalance_freq='monthly',
        ),
    })

    # 4. SMA200 + ATH 20% + Composite Momentum (highest CAGR)
    configs.append({
        'name': 'SMA200_CompMom_ATH20',
        'entry_fn': lambda df, idx: entry_price_above_sma(df, idx, period=200),
        'rank_fn': rank_composite_momentum,
        'exit_fn': make_combined_exit([
            (exit_ath_drawdown, {'pct': 20}),
            (exit_time_based, {'max_days': 365}),
        ]),
        'config': StrategyConfig(
            name='SMA200_CompMom_ATH20',
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            portfolio_size=20, rebalance_freq='monthly',
        ),
    })

    return configs


# =============================================================================
# Compute MQ baseline metrics from daily equity
# =============================================================================

def compute_mq_metrics(mq_result):
    """Extract metrics from MQ backtest result for comparison."""
    eq = pd.Series(mq_result.daily_equity, dtype=float)
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    initial = INITIAL_CAPITAL
    final = float(eq.iloc[-1])
    years = (eq.index[-1] - eq.index[0]).days / 365.25

    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
    total_ret = (final / initial - 1) * 100

    daily_ret = eq.pct_change().dropna()
    rf_daily = 0.07 / 252
    excess = daily_ret - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
    downside = daily_ret[daily_ret < 0]
    sortino = float(excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

    peak = eq.expanding().max()
    dd = ((eq - peak) / peak * 100)
    max_dd = abs(float(dd.min()))
    calmar = cagr / max_dd if max_dd > 0 else 0

    # Trade stats
    trades = mq_result.trade_log
    winners = [t for t in trades if t.return_pct > 0]
    losers = [t for t in trades if t.return_pct <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.return_pct * 100 for t in winners]) if winners else 0
    avg_loss = np.mean([t.return_pct * 100 for t in losers]) if losers else 0

    total_wins = sum(t.net_pnl for t in winners)
    total_losses = abs(sum(t.net_pnl for t in losers))
    pf = total_wins / total_losses if total_losses > 0 else float('inf')

    return {
        'cagr': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_drawdown': round(max_dd, 1),
        'calmar': round(calmar, 2),
        'profit_factor': round(pf, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'final_value': round(final, 0),
        'total_return_pct': round(total_ret, 1),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Skip already done
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        print(f'Skipping {len(done)} already-completed configs')
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Build all configs: 3 universe modes × 4 tech configs + 1 MQ baseline = 13
    all_configs = []

    # MQ baseline
    all_configs.append({
        'label': 'MQ_BASELINE',
        'universe_mode': 'MQ_PURE',
        'tech_config': 'none',
    })

    tech_configs = get_tech_configs()
    for mode in ['MQ_HELD', 'MQ_CANDIDATES', 'ALL_STOCKS']:
        for tc in tech_configs:
            label = f'{mode}_{tc["name"]}'
            all_configs.append({
                'label': label,
                'universe_mode': mode,
                'tech_config': tc['name'],
                'tech': tc,
            })

    todo = [c for c in all_configs if c['label'] not in done]
    total = len(todo)
    if total == 0:
        print('All configs already done!')
        sys.exit(0)

    print(f'\nRunning {total} configs ({len(done)} already done)...\n')

    # ── Step 1: Preload MQ data ──
    print('Preloading MQ data...')
    t0 = time.time()
    mq_config = MQBacktestConfig(start_date=START_DATE, end_date=END_DATE,
                                  initial_capital=INITIAL_CAPITAL)
    mq_universe, mq_price_data = MQBacktestEngine.preload_data(mq_config)
    print(f'MQ data loaded in {time.time()-t0:.0f}s ({len(mq_price_data)} stocks)')

    # ── Step 2: Run MQ backtest (needed for all hybrid modes) ──
    needs_mq = any(c['universe_mode'] in ('MQ_PURE', 'MQ_HELD', 'MQ_CANDIDATES')
                   for c in todo)
    mq_result = None
    mq_held_map = None
    mq_candidate_map = None

    if needs_mq:
        print('\nRunning MQ baseline backtest...')
        t1 = time.time()
        engine = MQBacktestEngine(mq_config,
                                   preloaded_universe=mq_universe,
                                   preloaded_price_data=mq_price_data)
        mq_result = engine.run()
        elapsed = time.time() - t1
        print(f'MQ done in {elapsed:.0f}s | CAGR={mq_result.cagr:.2f}% | '
              f'Final=Rs {mq_result.final_value:,.0f}')

        # Build MQ_HELD universe map
        if any(c['universe_mode'] == 'MQ_HELD' for c in todo):
            print('Building MQ_HELD daily universe...')
            mq_held_map = build_mq_held_universe(mq_result)
            # Stats
            avg_held = np.mean([len(v) for v in mq_held_map.values()])
            all_held = set()
            for v in mq_held_map.values():
                all_held.update(v)
            print(f'  Avg {avg_held:.1f} stocks/day, {len(all_held)} unique stocks total')

        # Build MQ_CANDIDATES universe map
        if any(c['universe_mode'] == 'MQ_CANDIDATES' for c in todo):
            print('Building MQ_CANDIDATES semi-annual universe...')
            mq_candidate_map = build_mq_candidate_universe(
                mq_universe, mq_price_data, mq_config)
            avg_cand = np.mean([len(v) for v in mq_candidate_map.values()])
            all_cand = set()
            for v in mq_candidate_map.values():
                all_cand.update(v)
            print(f'  Avg {avg_cand:.1f} candidates/day, {len(all_cand)} unique stocks total')

    # ── Step 3: Load/compute enriched indicators for technical strategies ──
    needs_tech = any(c['universe_mode'] != 'MQ_PURE' for c in todo)
    enriched = None

    if needs_tech:
        print('\nLoading enriched indicator data...')
        enriched = load_enriched_cache(START_DATE, END_DATE)
        if enriched:
            print(f'Loaded from cache ({len(enriched)} stocks)')
        else:
            print('Computing indicators (this takes a few minutes)...')
            # Use the MQ price data (same universe)
            t2 = time.time()
            enriched = enrich_with_indicators(mq_price_data)
            save_enriched_cache(enriched, START_DATE, END_DATE)
            print(f'Indicators computed in {time.time()-t2:.0f}s ({len(enriched)} stocks)')

    # ── Step 4: Run all configs ──
    for i, cfg in enumerate(todo):
        label = cfg['label']
        mode = cfg['universe_mode']

        print(f'\n[{i+1}/{total}] {label} ...', end='', flush=True)
        t1 = time.time()

        if mode == 'MQ_PURE':
            # MQ baseline - just extract metrics from already-run MQ result
            metrics = compute_mq_metrics(mq_result)
            row = {
                'label': label,
                'universe_mode': mode,
                'tech_config': 'none',
                **metrics,
                'top3_pnl_pct': '',
                'top3_symbols': '',
                'cagr_ex_top3': '',
                'exit_reason_counts': str(mq_result.exit_reason_counts),
            }
        else:
            # Technical strategy with optional universe restriction
            tc = cfg['tech']
            explorer = StrategyExplorer(mq_universe, enriched, tc['config'])

            # Determine allowed_symbols_fn based on mode
            allowed_fn = None
            if mode == 'MQ_HELD':
                allowed_fn = make_allowed_fn(mq_held_map)
            elif mode == 'MQ_CANDIDATES':
                allowed_fn = make_allowed_fn(mq_candidate_map)
            # ALL_STOCKS: no restriction

            result = explorer.run(
                tc['entry_fn'], tc['rank_fn'], tc['exit_fn'],
                allowed_symbols_fn=allowed_fn,
            )

            row = {
                'label': label,
                'universe_mode': mode,
                'tech_config': cfg['tech_config'],
                'cagr': result.cagr,
                'sharpe': result.sharpe,
                'sortino': result.sortino,
                'max_drawdown': result.max_drawdown,
                'calmar': result.calmar,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'avg_win_pct': result.avg_win_pct,
                'avg_loss_pct': result.avg_loss_pct,
                'final_value': result.final_value,
                'total_return_pct': result.total_return_pct,
                'top3_pnl_pct': result.top3_pnl_pct,
                'top3_symbols': str(result.top3_symbols),
                'cagr_ex_top3': result.cagr_ex_top3,
                'exit_reason_counts': str(result.exit_reason_counts),
            }

        elapsed = time.time() - t1

        # Write immediately
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        print(f' {elapsed:.0f}s | CAGR={row["cagr"]}% MaxDD={row["max_drawdown"]}% '
              f'Calmar={row["calmar"]} Trades={row["total_trades"]}')
        sys.stdout.flush()

    print(f'\nDone! Results in {OUTPUT_CSV}')
