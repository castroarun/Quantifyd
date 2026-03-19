"""
Run Non-Directional Options Strategy Simulation
================================================

Tests multiple configurations:
1. BankNifty Short Strangle (weekly, bi-weekly)
2. BankNifty Iron Condor (weekly, bi-weekly)
3. Nifty Short Strangle (weekly, bi-weekly)
4. Nifty Iron Condor (weekly, bi-weekly)

Across signal combos:
- BB_Squeeze + ATR_Contract (primary)
- Consensus 3/5 (secondary)
- BB_Squeeze only (comparison)

Output: CSV trade logs + summary report
"""

import os, sys, time, json
import pandas as pd
import numpy as np
import logging
import csv

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.range_detection_engine import load_index_data
from services.nondirectional_simulator import (
    SimConfig, StrategyType, NonDirectionalSimulator, LOT_SIZES
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'nondirectional_summary.csv')
TRADES_CSV = os.path.join(OUTPUT_DIR, 'nondirectional_trades.csv')


def build_configs():
    """Build all simulation configurations."""
    configs = []

    for symbol in ['BANKNIFTY', 'NIFTY50']:
        for strategy in [StrategyType.SHORT_STRANGLE, StrategyType.IRON_CONDOR]:
            for hold, dte, label in [(5, 7, 'weekly'), (10, 14, 'biweekly')]:
                for signal_name, bb, atr_c, consensus in [
                    ('BB+ATR', True, True, 0),
                    ('Consensus3', False, False, 3),
                    ('Consensus4', False, False, 4),
                    ('BB_only', True, False, 0),
                ]:
                    for strike_dist in [2.0, 2.5, 3.0]:
                        configs.append({
                            'label': f'{symbol}_{strategy.value}_{label}_{signal_name}_SD{strike_dist}',
                            'config': SimConfig(
                                symbol=symbol,
                                strategy=strategy,
                                strike_distance_atr=strike_dist,
                                wing_distance_atr=strike_dist + 1.5,
                                hold_bars=hold,
                                days_to_expiry_at_entry=dte,
                                require_bb_squeeze=bb,
                                require_atr_contract=atr_c,
                                min_consensus=consensus,
                                capital=10_00_000,
                                lots_per_trade=1,
                                bars_per_day=1,
                            ),
                        })

    return configs


def main():
    start_time = time.time()

    # Load data
    print("Loading data...")
    data = {}
    for symbol in ['BANKNIFTY', 'NIFTY50']:
        df = load_index_data(symbol, 'day')
        if df.empty:
            print(f"  ERROR: No daily data for {symbol}!")
            continue
        print(f"  {symbol}: {len(df)} bars ({df.index.min().date()} to {df.index.max().date()})")
        data[symbol] = df

    configs = build_configs()
    print(f"\nRunning {len(configs)} configurations...")

    # Summary CSV header
    summary_fields = [
        'label', 'symbol', 'strategy', 'strike_dist', 'hold_period', 'signal',
        'total_trades', 'win_rate', 'total_pnl', 'total_return_pct', 'cagr_pct',
        'profit_factor', 'max_drawdown_pct', 'avg_premium', 'avg_zone_width_pct',
        'zone_hold_rate', 'max_consecutive_losses', 'avg_pnl_per_trade',
        'avg_win', 'avg_loss', 'max_win', 'max_loss',
    ]

    with open(SUMMARY_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    all_trades = []
    all_summaries = []

    for i, cfg in enumerate(configs):
        label = cfg['label']
        config = cfg['config']
        symbol = config.symbol

        if symbol not in data:
            continue

        df = data[symbol].copy()

        sim = NonDirectionalSimulator(config, df)
        summary = sim.run()

        if 'error' in summary:
            print(f"  [{i+1}/{len(configs)}] {label}: {summary['error']}")
            continue

        # Extract signal name from label
        parts = label.split('_')
        # Find signal part
        signal_part = 'unknown'
        for p in ['BB+ATR', 'Consensus3', 'Consensus4', 'BB_only']:
            if p in label:
                signal_part = p
                break

        row = {
            'label': label,
            'symbol': symbol,
            'strategy': config.strategy.value,
            'strike_dist': config.strike_distance_atr,
            'hold_period': 'weekly' if config.hold_bars == 5 else 'biweekly',
            'signal': signal_part,
            'total_trades': summary['total_trades'],
            'win_rate': summary['win_rate'],
            'total_pnl': summary['total_pnl'],
            'total_return_pct': summary['total_return_pct'],
            'cagr_pct': summary['cagr_pct'],
            'profit_factor': summary['profit_factor'],
            'max_drawdown_pct': summary['max_drawdown_pct'],
            'avg_premium': summary['avg_premium_collected'],
            'avg_zone_width_pct': summary['avg_zone_width_pct'],
            'zone_hold_rate': summary['zone_hold_rate'],
            'max_consecutive_losses': summary['max_consecutive_losses'],
            'avg_pnl_per_trade': summary['avg_pnl_per_trade'],
            'avg_win': summary['avg_win'],
            'avg_loss': summary['avg_loss'],
            'max_win': summary['max_win'],
            'max_loss': summary['max_loss'],
        }

        with open(SUMMARY_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=summary_fields).writerow(row)

        all_summaries.append(row)

        # Save trade log
        trade_df = sim.get_trade_log()
        if not trade_df.empty:
            trade_df['config'] = label
            all_trades.append(trade_df)

        status = (f"W:{summary['win_rate']:.0f}% PF:{summary['profit_factor']:.1f} "
                  f"Ret:{summary['total_return_pct']:.1f}% DD:{summary['max_drawdown_pct']:.1f}%")
        print(f"  [{i+1}/{len(configs)}] {label}: {summary['total_trades']} trades | {status}")

    # Save all trades
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(TRADES_CSV, index=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETED {len(all_summaries)} configs in {elapsed:.0f}s")
    print(f"Summary: {SUMMARY_CSV}")
    print(f"Trades:  {TRADES_CSV}")

    # =========================================================================
    # ANALYSIS
    # =========================================================================

    df_s = pd.DataFrame(all_summaries)
    if df_s.empty:
        print("No results!")
        return

    # Filter viable strategies (win rate > 50%, profit factor > 1)
    viable = df_s[(df_s['win_rate'] > 50) & (df_s['profit_factor'] > 1)].copy()

    print(f"\n{'='*80}")
    print(f"VIABLE STRATEGIES (WR > 50%, PF > 1): {len(viable)} of {len(df_s)}")
    print(f"{'='*80}")

    if viable.empty:
        print("No viable strategies found! Showing all results:")
        viable = df_s.copy()

    # Sort by total return
    viable = viable.sort_values('total_return_pct', ascending=False)

    print(f"\n{'Label':<55} {'Trades':<7} {'WR%':<6} {'PF':<5} "
          f"{'Ret%':<8} {'DD%':<6} {'Prem':<7} {'ZnHld%':<7}")
    print("-" * 110)

    for _, row in viable.head(30).iterrows():
        pf_str = f"{row['profit_factor']:.1f}" if row['profit_factor'] < 100 else 'inf'
        print(f"{row['label']:<55} {row['total_trades']:<7} {row['win_rate']:<6.1f} "
              f"{pf_str:<5} {row['total_return_pct']:<8.1f} {row['max_drawdown_pct']:<6.1f} "
              f"{row['avg_premium']:<7.1f} {row['zone_hold_rate']:<7.1f}")

    # Best per category
    print(f"\n{'='*80}")
    print("BEST CONFIG PER CATEGORY")
    print(f"{'='*80}")

    for symbol in ['BANKNIFTY', 'NIFTY50']:
        for strategy in ['short_strangle', 'iron_condor']:
            for hold in ['weekly', 'biweekly']:
                mask = ((viable['symbol'] == symbol) &
                        (viable['strategy'] == strategy) &
                        (viable['hold_period'] == hold))
                subset = viable[mask]
                if subset.empty:
                    continue
                best = subset.nlargest(1, 'total_return_pct').iloc[0]
                print(f"\n  {symbol} {strategy} {hold}:")
                print(f"    Best: {best['label']}")
                print(f"    Trades={best['total_trades']}, WR={best['win_rate']:.1f}%, "
                      f"PF={best['profit_factor']:.1f}")
                print(f"    Return={best['total_return_pct']:.1f}%, DD={best['max_drawdown_pct']:.1f}%, "
                      f"Zone Hold={best['zone_hold_rate']:.1f}%")
                print(f"    Avg Premium={best['avg_premium']:.1f} pts, "
                      f"Avg P&L/trade=Rs {best['avg_pnl_per_trade']:.0f}")

    # Strike distance comparison
    print(f"\n{'='*80}")
    print("STRIKE DISTANCE COMPARISON (avg across configs)")
    print(f"{'='*80}")

    sd_comp = df_s.groupby(['symbol', 'strike_dist']).agg({
        'win_rate': 'mean',
        'total_return_pct': 'mean',
        'max_drawdown_pct': 'mean',
        'profit_factor': 'mean',
        'zone_hold_rate': 'mean',
    }).round(1)
    print(sd_comp.to_string())

    # Signal comparison
    print(f"\n{'='*80}")
    print("SIGNAL COMPARISON (avg across configs)")
    print(f"{'='*80}")

    sig_comp = df_s.groupby(['symbol', 'signal']).agg({
        'win_rate': 'mean',
        'total_return_pct': 'mean',
        'profit_factor': 'mean',
        'total_trades': 'mean',
        'zone_hold_rate': 'mean',
    }).round(1)
    print(sig_comp.to_string())


if __name__ == '__main__':
    main()
