#!/usr/bin/env python3
"""CPR Regime Filter Test + MQ Correlation Analysis
====================================================
1. Test regime filters on best CPR config (CPR0.5_PROX2.0_WICK25_ST7_M3.5)
2. Run MQ backtest for same period and compute daily return correlation

Usage:
  python run_cpr_regime_and_mq_overlay.py            # Run all regime tests
  python run_cpr_regime_and_mq_overlay.py --mq-only  # Only MQ correlation
"""
import csv, os, sys, time, json, logging, io, argparse
import numpy as np
from contextlib import redirect_stdout

logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.cpr_intraday_engine import (
    CPRIntradayEngine, CPRIntradayConfig, INTRADAY_SYMBOLS,
)

# ── Output ────────────────────────────────────────────────────────────────
REGIME_CSV = os.path.join(PROJECT_ROOT, 'cpr_v3_regime_filter.csv')
MQ_CORR_CSV = os.path.join(PROJECT_ROOT, 'cpr_v3_mq_correlation.csv')

FIELDNAMES = [
    'label', 'regime_filter', 'period', 'n_symbols',
    'total_trades', 'win_rate', 'total_pnl', 'pnl_pct',
    'profit_factor', 'max_drawdown', 'sharpe', 'sortino',
    'avg_trades_per_day', 'days_with_trades', 'total_trading_days',
    'exit_reasons',
]

# Best config from V3 sweep
BEST_CONFIG = dict(
    narrow_cpr_threshold=0.5,
    cpr_proximity_pct=2.0,
    max_wick_pct=25.0,
    st_period=7,
    st_multiplier=3.5,
)

SYMBOLS = INTRADAY_SYMBOLS
START_DATE = '2024-01-01'
END_DATE = '2025-10-27'


def build_regime_configs():
    """Test various regime filters on the best CPR config."""
    configs = [
        # Baselines
        ('NONE', 'none', {}),
        # SMA50 filters
        ('ABOVE_SMA50', 'above_sma50', {}),
        ('BELOW_SMA50', 'below_sma50', {}),
        # SMA200 filters
        ('ABOVE_SMA200', 'above_sma200', {}),
        ('BELOW_SMA200', 'below_sma200', {}),
        # RVol filters
        ('RVOL_HIGH_10', 'rvol_high', {'regime_rvol_threshold': 10.0}),
        ('RVOL_HIGH_12', 'rvol_high', {'regime_rvol_threshold': 12.0}),
        ('RVOL_HIGH_15', 'rvol_high', {'regime_rvol_threshold': 15.0}),
        ('RVOL_LOW_10', 'rvol_low', {'regime_rvol_threshold': 10.0}),
        ('RVOL_LOW_12', 'rvol_low', {'regime_rvol_threshold': 12.0}),
    ]
    return configs


def load_done(csv_path):
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            done = {row['label'] for row in csv.DictReader(f)}
    return done


def ensure_header(csv_path, fields):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def run_regime_tests():
    """Run best CPR config with different regime filters."""
    ensure_header(REGIME_CSV, FIELDNAMES)
    done = load_done(REGIME_CSV)

    configs = build_regime_configs()
    pending = [(l, rf, extra) for l, rf, extra in configs if l not in done]

    print(f'=== CPR Regime Filter Test ===')
    print(f'Base config: CPR0.5_PROX2.0_WICK25_ST7_M3.5')
    print(f'Total: {len(configs)} | Done: {len(done)} | Pending: {len(pending)}')

    if not pending:
        print('All regime tests completed!')
        return

    print(f'Universe: {len(SYMBOLS)} F&O stocks')
    print('Preloading data...', flush=True)
    t0 = time.time()
    daily_data, five_min_data, niftybees = CPRIntradayEngine.preload_data(
        SYMBOLS, START_DATE, END_DATE)
    print(f'Data loaded in {time.time()-t0:.1f}s', flush=True)

    for i, (label, regime_filter, extra_params) in enumerate(pending, 1):
        print(f'\n[{i}/{len(pending)}] Regime: {label}', flush=True)
        t1 = time.time()

        try:
            config = CPRIntradayConfig(
                symbols=SYMBOLS,
                start_date=START_DATE,
                end_date=END_DATE,
                initial_capital=1_000_000,
                regime_filter=regime_filter,
                **BEST_CONFIG,
                **extra_params,
            )

            engine = CPRIntradayEngine(config,
                preloaded_daily=daily_data,
                preloaded_5min=five_min_data,
                preloaded_niftybees=niftybees)

            with redirect_stdout(io.StringIO()):
                result = engine.run()

            row = {
                'label': label,
                'regime_filter': regime_filter,
                'period': f'{START_DATE} to {END_DATE}',
                'n_symbols': len(SYMBOLS),
                'total_trades': result.total_trades,
                'win_rate': round(result.win_rate, 2),
                'total_pnl': round(result.total_pnl, 0),
                'pnl_pct': round(result.total_pnl_pct, 2),
                'profit_factor': round(result.profit_factor, 4),
                'max_drawdown': round(result.max_drawdown, 4),
                'sharpe': round(result.sharpe_ratio, 4),
                'sortino': round(result.sortino_ratio, 4),
                'avg_trades_per_day': round(result.avg_trades_per_day, 4),
                'days_with_trades': result.days_with_trades,
                'total_trading_days': result.total_trading_days,
                'exit_reasons': json.dumps(result.exit_reason_counts),
            }

            with open(REGIME_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            elapsed = time.time() - t1
            print(f'  {elapsed:.0f}s | Trades={result.total_trades} '
                  f'WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} '
                  f'PnL={result.total_pnl:+,.0f}',
                  flush=True)

        except Exception as e:
            import traceback
            print(f'  ERROR: {e}', flush=True)
            traceback.print_exc()

    print(f'\nRegime results: {REGIME_CSV}')


def run_mq_correlation():
    """Run MQ backtest for same period and compute correlation with CPR daily P&L."""
    from services.mq_backtest_engine import MQBacktestEngine
    from services.mq_portfolio import MQBacktestConfig

    print('\n=== MQ + CPR Correlation Analysis ===')

    # --- MQ Backtest ---
    print('Running MQ backtest (2024-2025)...', flush=True)
    mq_config = MQBacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=10_000_000,
        portfolio_size=30,
        equity_allocation_pct=0.95,
        hard_stop_loss=0.50,
        rebalance_ath_drawdown=0.20,
    )
    t0 = time.time()
    universe, price_data = MQBacktestEngine.preload_data(mq_config)
    mq_engine = MQBacktestEngine(mq_config,
        preloaded_universe=universe,
        preloaded_price_data=price_data)
    mq_result = mq_engine.run()
    print(f'MQ done in {time.time()-t0:.0f}s | CAGR={mq_result.cagr:.1f}% '
          f'MaxDD={mq_result.max_drawdown:.1f}%', flush=True)

    # --- CPR Backtest (best config, no regime filter) ---
    print('Running CPR backtest (best config, 79 stocks)...', flush=True)
    t0 = time.time()
    daily_data, five_min_data, niftybees = CPRIntradayEngine.preload_data(
        SYMBOLS, START_DATE, END_DATE)

    cpr_config = CPRIntradayConfig(
        symbols=SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=1_000_000,
        **BEST_CONFIG,
    )
    cpr_engine = CPRIntradayEngine(cpr_config,
        preloaded_daily=daily_data,
        preloaded_5min=five_min_data,
        preloaded_niftybees=niftybees)

    with redirect_stdout(io.StringIO()):
        cpr_result = cpr_engine.run()
    print(f'CPR done in {time.time()-t0:.0f}s | Trades={cpr_result.total_trades} '
          f'PF={cpr_result.profit_factor:.2f}', flush=True)

    # --- Compute Correlation ---
    import pandas as pd

    # MQ equity curve (daily)
    mq_eq = pd.Series(mq_result.daily_equity, name='mq_equity')
    mq_eq.index = pd.to_datetime(mq_eq.index)
    mq_eq = mq_eq.sort_index()
    mq_returns = mq_eq.pct_change().dropna()

    # CPR equity curve (daily)
    cpr_eq = pd.Series(cpr_result.equity_curve, name='cpr_equity')
    cpr_eq.index = pd.to_datetime(cpr_eq.index)
    cpr_eq = cpr_eq.sort_index()
    cpr_returns = cpr_eq.pct_change().dropna()

    # Align dates
    common = mq_returns.index.intersection(cpr_returns.index)
    if len(common) < 10:
        print(f'WARNING: Only {len(common)} common dates, not enough for correlation')
        return

    mq_r = mq_returns.loc[common]
    cpr_r = cpr_returns.loc[common]

    corr = mq_r.corr(cpr_r)

    # Monthly correlation
    print(f'\n--- Overall ---')
    print(f'Daily return correlation (MQ vs CPR): {corr:.4f}')
    print(f'Common trading days: {len(common)}')

    # Combined equity (hypothetical: 90% MQ + 10% CPR allocation)
    mq_norm = mq_eq / mq_eq.iloc[0]
    cpr_norm = cpr_eq / cpr_eq.iloc[0]

    # Align for combined
    all_dates = mq_norm.index.union(cpr_norm.index).sort_values()
    mq_filled = mq_norm.reindex(all_dates, method='ffill')
    cpr_filled = cpr_norm.reindex(all_dates, method='ffill')

    combined_90_10 = 0.90 * mq_filled + 0.10 * cpr_filled
    combined_80_20 = 0.80 * mq_filled + 0.20 * cpr_filled

    # Compute combined stats
    for name, combined in [('90/10', combined_90_10), ('80/20', combined_80_20)]:
        c_ret = combined.pct_change().dropna()
        c_ann_ret = (combined.iloc[-1] ** (252 / len(c_ret))) - 1
        c_vol = c_ret.std() * np.sqrt(252)
        c_sharpe = c_ann_ret / c_vol if c_vol > 0 else 0
        c_dd = (combined / combined.cummax() - 1).min()
        print(f'\n--- Combined {name} (MQ/CPR) ---')
        print(f'Ann. Return: {c_ann_ret*100:.2f}% | Vol: {c_vol*100:.2f}% '
              f'| Sharpe: {c_sharpe:.2f} | MaxDD: {c_dd*100:.2f}%')

    # MQ standalone
    mq_ann = (mq_norm.iloc[-1] ** (252 / len(mq_returns))) - 1
    mq_vol = mq_returns.std() * np.sqrt(252)
    mq_sharpe = mq_ann / mq_vol if mq_vol > 0 else 0
    mq_dd = (mq_norm / mq_norm.cummax() - 1).min()
    print(f'\n--- MQ Standalone ---')
    print(f'Ann. Return: {mq_ann*100:.2f}% | Vol: {mq_vol*100:.2f}% '
          f'| Sharpe: {mq_sharpe:.2f} | MaxDD: {mq_dd*100:.2f}%')

    # Save to CSV
    with open(MQ_CORR_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['correlation', f'{corr:.4f}'])
        w.writerow(['common_days', len(common)])
        w.writerow(['mq_cagr', f'{mq_result.cagr:.2f}'])
        w.writerow(['mq_maxdd', f'{mq_result.max_drawdown:.2f}'])
        w.writerow(['mq_sharpe', f'{mq_sharpe:.4f}'])
        w.writerow(['cpr_trades', cpr_result.total_trades])
        w.writerow(['cpr_pf', f'{cpr_result.profit_factor:.4f}'])
        w.writerow(['cpr_pnl_pct', f'{cpr_result.total_pnl_pct:.2f}'])

    print(f'\nCorrelation results: {MQ_CORR_CSV}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mq-only', action='store_true', help='Only run MQ correlation')
    parser.add_argument('--regime-only', action='store_true', help='Only run regime tests')
    args = parser.parse_args()

    if args.mq_only:
        run_mq_correlation()
    elif args.regime_only:
        run_regime_tests()
    else:
        run_regime_tests()
        run_mq_correlation()


if __name__ == '__main__':
    main()
