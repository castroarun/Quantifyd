"""
MQ+ / KC6 Combined System - Model Comparison
==============================================

Tests 4 models:
  Model A: MQ+ standalone (100%) with NIFTYBEES parking (baseline)
  Model B: MQ+(raw) + KC6 shared pool (KC6 borrows MQ+ idle cash)
  Model C: MQ+(50%) + KC6 standalone(50%) with debt parking
  Model D: KC6 standalone (100%)

Sweeps KC6 configs: MP3_PS33, MP5_PS20, MP7_PS15, MP10_PS10

Output: mq_kc6_models_results.csv (incremental)
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

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mq_kc6_models_results.csv')

FIELDNAMES = [
    'label', 'model', 'mq_pct', 'kc_pct',
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


def compute_metrics(equity_dict, initial_capital):
    """Compute CAGR, Sharpe, Sortino, MaxDD, Calmar from equity dict."""
    if len(equity_dict) < 30:
        return {}
    eq = pd.Series(equity_dict, dtype=float)
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    final = float(eq.iloc[-1])
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((final / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    total_ret = (final / initial_capital - 1) * 100

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


def build_daily_idle_cash(mq_result, price_data):
    """
    Reconstruct MQ+'s daily idle cash from trade log + final positions.
    Returns dict: date_str -> idle_cash_amount
    """
    intervals = []
    for t in mq_result.trade_log:
        entry = t.entry_date
        exit_d = t.exit_date
        if isinstance(entry, str):
            entry = datetime.strptime(entry, '%Y-%m-%d')
        if isinstance(exit_d, str):
            exit_d = datetime.strptime(exit_d, '%Y-%m-%d')
        intervals.append((t.symbol, entry, exit_d, t.shares_entered))

    for p in mq_result.final_positions:
        entry = p['entry_date']
        if isinstance(entry, str):
            entry = datetime.strptime(entry, '%Y-%m-%d')
        intervals.append((p['symbol'], entry, datetime(2099, 12, 31), p['shares']))

    idle_cash = {}
    for date_str, total_equity in mq_result.daily_equity.items():
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        debt_fund = mq_result.daily_debt_fund.get(date_str, 0)

        invested = 0
        for sym, entry, exit_d, shares in intervals:
            if entry <= dt < exit_d:
                ts = pd.Timestamp(date_str)
                if sym in price_data and ts in price_data[sym].index:
                    close = float(price_data[sym].loc[ts, 'close'])
                    invested += close * shares

        idle = total_equity - invested - max(debt_fund, 0)
        idle_cash[date_str] = max(idle, 0)  # Floor at 0

    return idle_cash


def write_row(row):
    """Append a single row to CSV immediately."""
    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def build_row(label, model, mq_pct, kc_pct, max_pos, pos_size,
              combined_metrics, mq_cagr, mq_final, kc_result):
    return {
        'label': label,
        'model': model,
        'mq_pct': mq_pct,
        'kc_pct': kc_pct,
        'kc_max_pos': max_pos,
        'kc_pos_size_pct': pos_size,
        'combined_cagr': combined_metrics.get('cagr', 0),
        'combined_sharpe': combined_metrics.get('sharpe', 0),
        'combined_sortino': combined_metrics.get('sortino', 0),
        'combined_max_dd': combined_metrics.get('max_dd', 0),
        'combined_calmar': combined_metrics.get('calmar', 0),
        'combined_final_value': combined_metrics.get('final_value', 0),
        'combined_total_return_pct': combined_metrics.get('total_return_pct', 0),
        'mq_cagr': round(mq_cagr, 2),
        'mq_final_value': round(mq_final, 0),
        'kc_cagr': round(kc_result.cagr, 2) if kc_result else 0,
        'kc_final_value': round(kc_result.final_value, 0) if kc_result else 0,
        'kc_trades': kc_result.total_trades if kc_result else 0,
        'kc_win_rate': round(kc_result.win_rate, 1) if kc_result else 0,
        'kc_profit_factor': round(kc_result.profit_factor, 2) if kc_result else 0,
        'kc_avg_win': round(kc_result.avg_win_pct, 2) if kc_result else 0,
        'kc_avg_loss': round(kc_result.avg_loss_pct, 2) if kc_result else 0,
        'kc_crash_filter_days': kc_result.crash_filter_days if kc_result else 0,
        'kc_avg_positions': kc_result.avg_positions if kc_result else 0,
        'kc_max_concurrent': kc_result.max_concurrent if kc_result else 0,
        'kc_exit_reasons': str(kc_result.exit_reason_counts) if kc_result else '',
    }


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

    # KC6 configs to sweep
    KC6_CONFIGS = [
        (3, 0.33, 'MP3_PS33'),
        (5, 0.20, 'MP5_PS20'),
        (7, 0.15, 'MP7_PS15'),
        (10, 0.10, 'MP10_PS10'),
    ]

    # ── Preload data ──
    print('Preloading data...')
    t0 = time.time()
    _cfg = MQBacktestConfig(start_date=START_DATE, end_date=END_DATE)
    universe, price_data = MQBacktestEngine.preload_data(_cfg)
    universe_symbols = universe.symbols
    print(f'Data loaded in {time.time()-t0:.0f}s ({len(price_data)} stocks)')

    # ── Precompute KC6 indicators ──
    print('Precomputing KC6 indicators...')
    t_kc = time.time()
    kc6_enriched = KC6BacktestEngine.precompute_indicators(
        price_data, universe_symbols, KC6BacktestConfig()
    )
    print(f'KC6 indicators done in {time.time()-t_kc:.0f}s ({len(kc6_enriched)} stocks)')

    # ═══════════════════════════════════════════════════════════════
    # MODEL A: MQ+ baseline (100% with NIFTYBEES parking)
    # ═══════════════════════════════════════════════════════════════
    label_a = 'MODEL_A_MQ_BASELINE'
    if label_a not in done:
        print('\n[Model A] MQ+ baseline (100%, NIFTYBEES parking)...')
        t1 = time.time()
        cfg_a = MQBacktestConfig(
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL, portfolio_size=30,
            equity_allocation_pct=0.95, debt_reserve_pct=0.05,
            hard_stop_loss=0.50, rebalance_ath_drawdown=0.20,
            daily_ath_drawdown_exit=True, immediate_replacement=True,
            idle_cash_to_nifty_etf=True, idle_cash_to_debt=True,
        )
        eng_a = MQBacktestEngine(cfg_a, preloaded_universe=universe, preloaded_price_data=price_data)
        r_a = eng_a.run()
        metrics_a = compute_metrics(r_a.daily_equity, INITIAL_CAPITAL)
        row = build_row(label_a, 'A_BASELINE', 100, 0, 0, 0, metrics_a, r_a.cagr, r_a.final_value, None)
        write_row(row)
        print(f'  {time.time()-t1:.0f}s | CAGR={r_a.cagr:.2f}% MaxDD={r_a.max_drawdown:.1f}% Final=Rs {r_a.final_value:,.0f}')
    else:
        # Need MQ+ result for Model B, re-run
        print('\n[Model A] Already done, re-running MQ+ for Model B...')
        cfg_a = MQBacktestConfig(
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL, portfolio_size=30,
            equity_allocation_pct=0.95, debt_reserve_pct=0.05,
            hard_stop_loss=0.50, rebalance_ath_drawdown=0.20,
            daily_ath_drawdown_exit=True, immediate_replacement=True,
            idle_cash_to_nifty_etf=True, idle_cash_to_debt=True,
        )
        eng_a = MQBacktestEngine(cfg_a, preloaded_universe=universe, preloaded_price_data=price_data)
        r_a = eng_a.run()

    sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════
    # MODEL B: MQ+(raw, no parking) + KC6 shared pool
    # ═══════════════════════════════════════════════════════════════
    # Need MQ+ without parking for idle cash computation
    print('\n[Model B prep] Running MQ+ WITHOUT parking...')
    t1 = time.time()
    cfg_raw = MQBacktestConfig(
        start_date=START_DATE, end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL, portfolio_size=30,
        equity_allocation_pct=0.95, debt_reserve_pct=0.05,
        hard_stop_loss=0.50, rebalance_ath_drawdown=0.20,
        daily_ath_drawdown_exit=True, immediate_replacement=True,
        idle_cash_to_nifty_etf=False, idle_cash_to_debt=False,
    )
    eng_raw = MQBacktestEngine(cfg_raw, preloaded_universe=universe, preloaded_price_data=price_data)
    r_raw = eng_raw.run()
    print(f'  MQ+(raw) done in {time.time()-t1:.0f}s | CAGR={r_raw.cagr:.2f}% Final=Rs {r_raw.final_value:,.0f}')

    # Build daily idle cash map
    print('  Building daily idle cash map...')
    t2 = time.time()
    daily_idle = build_daily_idle_cash(r_raw, price_data)
    avg_idle = np.mean(list(daily_idle.values()))
    print(f'  Idle cash map built in {time.time()-t2:.0f}s | Avg idle: Rs {avg_idle:,.0f}')

    # Build pool capital function (returns idle cash for each date)
    def make_pool_fn(idle_map):
        def fn(date):
            date_str = date.strftime('%Y-%m-%d')
            return idle_map.get(date_str, 0)
        return fn

    pool_fn = make_pool_fn(daily_idle)

    # Run Model B for each KC6 config
    for max_pos, pos_size, suffix in KC6_CONFIGS:
        label = f'MODEL_B_SHARED_POOL_{suffix}'
        if label in done:
            print(f'  [Model B] {suffix} already done, skipping')
            continue

        print(f'  [Model B] KC6 shared pool {suffix}...', end='', flush=True)
        t3 = time.time()

        kc_cfg = KC6BacktestConfig(
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=0,  # KC6 starts with no capital, borrows from pool
            max_positions=max_pos, position_size_pct=pos_size,
        )
        kc_engine = KC6BacktestEngine(kc_cfg, price_data, universe_symbols, precomputed=kc6_enriched)
        kc_result = kc_engine.run(pool_capital_fn=pool_fn)

        # Combined equity: MQ+(raw) + KC6 net value
        combined_eq = {}
        for date_str in r_raw.daily_equity:
            mq_val = r_raw.daily_equity[date_str]
            kc_val = kc_result.daily_equity.get(date_str, 0)
            combined_eq[date_str] = mq_val + kc_val

        combined_metrics = compute_metrics(combined_eq, INITIAL_CAPITAL)
        row = build_row(label, 'B_SHARED_POOL', 100, 0, max_pos, pos_size,
                        combined_metrics, r_raw.cagr, r_raw.final_value, kc_result)
        write_row(row)

        elapsed = time.time() - t3
        print(f' {elapsed:.0f}s | Combined CAGR={combined_metrics.get("cagr", 0):.2f}% '
              f'MaxDD={combined_metrics.get("max_dd", 0):.1f}% '
              f'KC trades={kc_result.total_trades} WR={kc_result.win_rate:.0f}%')
        sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════
    # MODEL C: MQ+(50%) + KC6(50%) with debt parking
    # ═══════════════════════════════════════════════════════════════
    print('\n[Model C prep] Running MQ+ at 50%...')
    t1 = time.time()
    cfg_50 = MQBacktestConfig(
        start_date=START_DATE, end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL // 2, portfolio_size=30,
        equity_allocation_pct=0.95, debt_reserve_pct=0.05,
        hard_stop_loss=0.50, rebalance_ath_drawdown=0.20,
        daily_ath_drawdown_exit=True, immediate_replacement=True,
        idle_cash_to_nifty_etf=True, idle_cash_to_debt=True,
    )
    eng_50 = MQBacktestEngine(cfg_50, preloaded_universe=universe, preloaded_price_data=price_data)
    r_50 = eng_50.run()
    print(f'  MQ+(50%) done in {time.time()-t1:.0f}s | CAGR={r_50.cagr:.2f}% Final=Rs {r_50.final_value:,.0f}')
    sys.stdout.flush()

    for max_pos, pos_size, suffix in KC6_CONFIGS:
        label = f'MODEL_C_5050_{suffix}'
        if label in done:
            print(f'  [Model C] {suffix} already done, skipping')
            continue

        print(f'  [Model C] 50/50 MQ+KC6 {suffix}...', end='', flush=True)
        t3 = time.time()

        kc_cfg = KC6BacktestConfig(
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL // 2,
            max_positions=max_pos, position_size_pct=pos_size,
        )
        kc_engine = KC6BacktestEngine(kc_cfg, price_data, universe_symbols, precomputed=kc6_enriched)
        kc_result = kc_engine.run()

        # Combined equity: MQ+(50%) + KC6(50%)
        combined_eq = {}
        all_dates = sorted(set(list(r_50.daily_equity.keys()) + list(kc_result.daily_equity.keys())))
        for d in all_dates:
            mq_val = r_50.daily_equity.get(d, INITIAL_CAPITAL // 2)
            kc_val = kc_result.daily_equity.get(d, INITIAL_CAPITAL // 2)
            combined_eq[d] = mq_val + kc_val

        combined_metrics = compute_metrics(combined_eq, INITIAL_CAPITAL)
        row = build_row(label, 'C_5050', 50, 50, max_pos, pos_size,
                        combined_metrics, r_50.cagr, r_50.final_value, kc_result)
        write_row(row)

        elapsed = time.time() - t3
        print(f' {elapsed:.0f}s | Combined CAGR={combined_metrics.get("cagr", 0):.2f}% '
              f'MaxDD={combined_metrics.get("max_dd", 0):.1f}% '
              f'KC trades={kc_result.total_trades} WR={kc_result.win_rate:.0f}%')
        sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════
    # MODEL D: KC6 standalone (100%) - best configs
    # ═══════════════════════════════════════════════════════════════
    for max_pos, pos_size, suffix in KC6_CONFIGS:
        label = f'MODEL_D_KC6_SOLO_{suffix}'
        if label in done:
            print(f'  [Model D] {suffix} already done, skipping')
            continue

        print(f'  [Model D] KC6 standalone {suffix}...', end='', flush=True)
        t3 = time.time()

        kc_cfg = KC6BacktestConfig(
            start_date=START_DATE, end_date=END_DATE,
            initial_capital=INITIAL_CAPITAL,
            max_positions=max_pos, position_size_pct=pos_size,
        )
        kc_engine = KC6BacktestEngine(kc_cfg, price_data, universe_symbols, precomputed=kc6_enriched)
        kc_result = kc_engine.run()

        combined_metrics = compute_metrics(kc_result.daily_equity, INITIAL_CAPITAL)
        row = build_row(label, 'D_KC6_SOLO', 0, 100, max_pos, pos_size,
                        combined_metrics, 0, 0, kc_result)
        write_row(row)

        elapsed = time.time() - t3
        print(f' {elapsed:.0f}s | CAGR={kc_result.cagr:.2f}% MaxDD={kc_result.max_drawdown:.1f}% '
              f'Trades={kc_result.total_trades} WR={kc_result.win_rate:.0f}%')
        sys.stdout.flush()

    print(f'\nDone! Results in {OUTPUT_CSV}')
