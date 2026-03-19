"""
MQ+ / KC6 Combined System Sweep
=================================

Tests 3 cases:
  Case 1: MQ+ (X%) + KC6 on MQ holdings only (Y%)
  Case 2: MQ+ (X%) + KC6 on all Nifty 500 (Y%)
  Case 3: KC6 standalone on Nifty 500 (100%)

Sweeps capital splits: 90/10, 80/20, 70/30, 60/40
Also varies KC6 max_positions and position_size_pct.

Output: mq_kc6_sweep_results.csv (incremental)
"""

import csv
import logging
import os
import sys
import time
from datetime import datetime

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig
from services.kc6_backtest_engine import (
    KC6BacktestEngine, KC6BacktestConfig,
    extract_mq_daily_holdings,
)

import numpy as np
import pandas as pd

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mq_kc6_sweep_results.csv')

FIELDNAMES = [
    'label', 'case', 'mq_pct', 'kc_pct',
    'kc_max_pos', 'kc_pos_size_pct',
    'combined_cagr', 'combined_sharpe', 'combined_sortino',
    'combined_max_dd', 'combined_calmar',
    'combined_final_value', 'combined_total_return_pct',
    'mq_cagr', 'mq_final_value',
    'kc_cagr', 'kc_final_value', 'kc_trades', 'kc_win_rate',
    'kc_profit_factor', 'kc_avg_win', 'kc_avg_loss',
    'kc_crash_filter_days', 'kc_avg_positions', 'kc_max_concurrent',
    'kc_exit_reasons',
]

START_DATE = '2010-01-01'
END_DATE = '2026-02-16'
INITIAL_CAPITAL = 10_000_000


def compute_combined_metrics(mq_equity, kc_equity, mq_pct, kc_pct, start_date_str, end_date_str):
    """Combine two equity curves and compute metrics."""
    # Get all dates
    all_dates = sorted(set(list(mq_equity.keys()) + list(kc_equity.keys())))
    if not all_dates:
        return {}

    # Filter to simulation period
    start = start_date_str
    end = end_date_str
    dates = [d for d in all_dates if start <= d <= end]
    if len(dates) < 30:
        return {}

    # Combine: mq contributes mq_pct of total, kc contributes kc_pct
    # MQ equity is scaled to mq_pct of INITIAL_CAPITAL
    # KC equity is scaled to kc_pct of INITIAL_CAPITAL
    combined = {}
    for d in dates:
        mq_val = mq_equity.get(d, None)
        kc_val = kc_equity.get(d, None)
        if mq_val is not None and kc_val is not None:
            combined[d] = mq_val + kc_val
        elif mq_val is not None:
            combined[d] = mq_val + (INITIAL_CAPITAL * kc_pct / 100)
        elif kc_val is not None:
            combined[d] = (INITIAL_CAPITAL * mq_pct / 100) + kc_val

    if len(combined) < 30:
        return {}

    eq = pd.Series(combined, dtype=float)
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    total_capital = INITIAL_CAPITAL
    final = float(eq.iloc[-1])
    years = (eq.index[-1] - eq.index[0]).days / 365.25

    cagr = ((final / total_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    total_ret = (final / total_capital - 1) * 100

    daily_ret = eq.pct_change().dropna()
    sharpe = 0.0
    sortino = 0.0
    if len(daily_ret) > 0 and daily_ret.std() > 0:
        rf_daily = 0.07 / 252
        excess = daily_ret - rf_daily
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252))
        downside = daily_ret[daily_ret < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = float(excess.mean() / downside.std() * np.sqrt(252))

    peak = eq.expanding().max()
    dd = ((eq - peak) / peak * 100)
    max_dd = abs(float(dd.min()))
    calmar = cagr / max_dd if max_dd > 0 else 0

    return {
        'cagr': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_dd': round(max_dd, 1),
        'calmar': round(calmar, 2),
        'final_value': round(final, 0),
        'total_return_pct': round(total_ret, 1),
    }


def run_mq_backtest(capital, universe, price_data):
    """Run MQ+ backtest with given capital."""
    config = MQBacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=capital,
        portfolio_size=30,
        equity_allocation_pct=0.95,
        debt_reserve_pct=0.05,
        hard_stop_loss=0.50,
        rebalance_ath_drawdown=0.20,
        daily_ath_drawdown_exit=True,
        immediate_replacement=True,
        idle_cash_to_nifty_etf=True,
        idle_cash_to_debt=True,
    )
    engine = MQBacktestEngine(config, preloaded_universe=universe, preloaded_price_data=price_data)
    return engine.run()


def run_kc6_backtest(capital, price_data, symbols, max_pos, pos_size_pct,
                     allowed_symbols_fn=None, precomputed=None):
    """Run KC6 backtest with given capital and params."""
    config = KC6BacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=capital,
        max_positions=max_pos,
        position_size_pct=pos_size_pct,
    )
    engine = KC6BacktestEngine(config, price_data, symbols, precomputed=precomputed)
    return engine.run(allowed_symbols_fn)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

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

    # ── Define all configs ──
    configs = []

    # Case 3: KC6 standalone (100% to KC6)
    for max_pos, pos_size in [(3, 0.20), (5, 0.10), (5, 0.15), (7, 0.10), (10, 0.05)]:
        label = f'KC6_STANDALONE_MP{max_pos}_PS{int(pos_size*100)}'
        configs.append({
            'label': label, 'case': 3,
            'mq_pct': 0, 'kc_pct': 100,
            'kc_max_pos': max_pos, 'kc_pos_size_pct': pos_size,
        })

    # Case 2: MQ+ + KC6 on all Nifty 500
    for mq_pct, kc_pct in [(90, 10), (80, 20), (70, 30)]:
        for max_pos, pos_size in [(3, 0.20), (5, 0.10)]:
            label = f'MQ{mq_pct}_KC{kc_pct}_ALL_MP{max_pos}_PS{int(pos_size*100)}'
            configs.append({
                'label': label, 'case': 2,
                'mq_pct': mq_pct, 'kc_pct': kc_pct,
                'kc_max_pos': max_pos, 'kc_pos_size_pct': pos_size,
            })

    # Case 1: MQ+ + KC6 on MQ holdings only
    for mq_pct, kc_pct in [(90, 10), (80, 20), (70, 30)]:
        for max_pos, pos_size in [(3, 0.20), (5, 0.10)]:
            label = f'MQ{mq_pct}_KC{kc_pct}_MQONLY_MP{max_pos}_PS{int(pos_size*100)}'
            configs.append({
                'label': label, 'case': 1,
                'mq_pct': mq_pct, 'kc_pct': kc_pct,
                'kc_max_pos': max_pos, 'kc_pos_size_pct': pos_size,
            })

    # Filter out already done
    todo = [c for c in configs if c['label'] not in done]
    total = len(todo)
    if total == 0:
        print('All configs already done!')
        sys.exit(0)

    print(f'\nRunning {total} configs ({len(done)} already done)...\n')

    # ── Preload data once ──
    print('Preloading data...')
    t0 = time.time()
    from services.mq_backtest_engine import MQBacktestEngine
    from services.mq_portfolio import MQBacktestConfig as _MQCfg
    _cfg = _MQCfg(start_date=START_DATE, end_date=END_DATE)
    universe, price_data = MQBacktestEngine.preload_data(_cfg)
    universe_symbols = universe.symbols
    print(f'Data loaded in {time.time()-t0:.0f}s ({len(price_data)} stocks)')

    # ── Precompute KC6 indicators once ──
    print('Precomputing KC6 indicators...')
    t_kc = time.time()
    kc6_enriched = KC6BacktestEngine.precompute_indicators(
        price_data, universe_symbols, KC6BacktestConfig()
    )
    print(f'KC6 indicators done in {time.time()-t_kc:.0f}s ({len(kc6_enriched)} stocks enriched)')

    # ── Pre-run MQ+ backtests at different capital levels ──
    # We cache MQ results for each unique mq_pct
    mq_cache = {}
    mq_pcts_needed = set(c['mq_pct'] for c in todo if c['mq_pct'] > 0)

    for mq_pct in sorted(mq_pcts_needed):
        mq_capital = INITIAL_CAPITAL * mq_pct / 100
        print(f'\nPre-running MQ+ at {mq_pct}% (Rs {mq_capital:,.0f})...')
        t1 = time.time()
        mq_result = run_mq_backtest(mq_capital, universe, price_data)
        elapsed = time.time() - t1
        print(f'  MQ+ done in {elapsed:.0f}s | CAGR={mq_result.cagr:.2f}% | Final=Rs {mq_result.final_value:,.0f}')

        # Also extract daily holdings for Case 1
        daily_holdings = extract_mq_daily_holdings(mq_result)

        mq_cache[mq_pct] = {
            'result': mq_result,
            'daily_holdings': daily_holdings,
        }
    sys.stdout.flush()

    # ── Run KC6 configs ──
    for i, cfg in enumerate(todo):
        label = cfg['label']
        case = cfg['case']
        mq_pct = cfg['mq_pct']
        kc_pct = cfg['kc_pct']
        max_pos = cfg['kc_max_pos']
        pos_size = cfg['kc_pos_size_pct']

        print(f'\n[{i+1}/{total}] {label} ...', end='', flush=True)
        t1 = time.time()

        kc_capital = INITIAL_CAPITAL * kc_pct / 100

        # Determine allowed symbols function
        allowed_fn = None
        if case == 1:
            # KC6 only on MQ holdings
            mq_holdings = mq_cache[mq_pct]['daily_holdings']

            def make_fn(holdings_map):
                def fn(date):
                    date_str = date.strftime('%Y-%m-%d')
                    return holdings_map.get(date_str, set())
                return fn

            allowed_fn = make_fn(mq_holdings)

        # Run KC6 (with precomputed indicators)
        kc_result = run_kc6_backtest(
            kc_capital, price_data, universe_symbols,
            max_pos, pos_size, allowed_fn,
            precomputed=kc6_enriched,
        )

        # Compute combined metrics
        if mq_pct > 0:
            mq_result = mq_cache[mq_pct]['result']
            combined = compute_combined_metrics(
                mq_result.daily_equity, kc_result.daily_equity,
                mq_pct, kc_pct, START_DATE, END_DATE,
            )
            mq_cagr = mq_result.cagr
            mq_final = mq_result.final_value
        else:
            # Standalone KC6
            combined = {
                'cagr': kc_result.cagr,
                'sharpe': kc_result.sharpe,
                'sortino': kc_result.sortino,
                'max_dd': kc_result.max_drawdown,
                'calmar': kc_result.calmar,
                'final_value': kc_result.final_value,
                'total_return_pct': kc_result.total_return_pct,
            }
            mq_cagr = 0
            mq_final = 0

        elapsed = time.time() - t1

        # Build row
        row = {
            'label': label,
            'case': case,
            'mq_pct': mq_pct,
            'kc_pct': kc_pct,
            'kc_max_pos': max_pos,
            'kc_pos_size_pct': pos_size,
            'combined_cagr': combined.get('cagr', 0),
            'combined_sharpe': combined.get('sharpe', 0),
            'combined_sortino': combined.get('sortino', 0),
            'combined_max_dd': combined.get('max_dd', 0),
            'combined_calmar': combined.get('calmar', 0),
            'combined_final_value': combined.get('final_value', 0),
            'combined_total_return_pct': combined.get('total_return_pct', 0),
            'mq_cagr': round(mq_cagr, 2),
            'mq_final_value': round(mq_final, 0),
            'kc_cagr': round(kc_result.cagr, 2),
            'kc_final_value': round(kc_result.final_value, 0),
            'kc_trades': kc_result.total_trades,
            'kc_win_rate': round(kc_result.win_rate, 1),
            'kc_profit_factor': round(kc_result.profit_factor, 2),
            'kc_avg_win': round(kc_result.avg_win_pct, 2),
            'kc_avg_loss': round(kc_result.avg_loss_pct, 2),
            'kc_crash_filter_days': kc_result.crash_filter_days,
            'kc_avg_positions': kc_result.avg_positions,
            'kc_max_concurrent': kc_result.max_concurrent,
            'kc_exit_reasons': str(kc_result.exit_reason_counts),
        }

        # Write immediately
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        case_label = {1: 'MQ+KC(MQ)', 2: 'MQ+KC(ALL)', 3: 'KC6 Solo'}[case]
        print(f' {elapsed:.0f}s | {case_label} | Combined CAGR={combined.get("cagr", 0):.2f}% '
              f'MaxDD={combined.get("max_dd", 0):.1f}% KC trades={kc_result.total_trades} '
              f'WR={kc_result.win_rate:.0f}%')
        sys.stdout.flush()

    print(f'\nDone! Results in {OUTPUT_CSV}')
