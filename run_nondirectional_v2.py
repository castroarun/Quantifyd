"""
Non-Directional Simulation V2
==============================
- 3 lots per 10L capital (realistic sizing)
- Added 1.5x ATR strikes (tighter, more premium)
- Test lot scaling: 3, 5 lots
- Better premium model with IV premium bump
"""

import os, sys, time, csv
import pandas as pd
import numpy as np
import logging

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.range_detection_engine import load_index_data
from services.nondirectional_simulator import (
    SimConfig, StrategyType, NonDirectionalSimulator, LOT_SIZES
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'nondirectional_v2_summary.csv')
TRADES_CSV = os.path.join(OUTPUT_DIR, 'nondirectional_v2_trades.csv')


def build_configs():
    configs = []

    for symbol in ['BANKNIFTY', 'NIFTY50']:
        for strategy in [StrategyType.SHORT_STRANGLE, StrategyType.IRON_CONDOR]:
            for hold, dte, label in [(5, 7, 'weekly'), (10, 14, 'biweekly')]:
                for signal_name, bb, atr_c, consensus in [
                    ('BB+ATR', True, True, 0),
                    ('BB_only', True, False, 0),
                    ('Consensus3', False, False, 3),
                ]:
                    for strike_dist in [1.5, 2.0, 2.5, 3.0]:
                        for lots in [3, 5]:
                            configs.append({
                                'label': (f'{symbol}_{strategy.value}_{label}_{signal_name}'
                                          f'_SD{strike_dist}_L{lots}'),
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
                                    lots_per_trade=lots,
                                    bars_per_day=1,
                                    min_days_between_trades=hold,  # no overlap
                                ),
                            })

    return configs


def main():
    start_time = time.time()

    print("Loading data...")
    data = {}
    for symbol in ['BANKNIFTY', 'NIFTY50']:
        df = load_index_data(symbol, 'day')
        if not df.empty:
            print(f"  {symbol}: {len(df)} bars ({df.index.min().date()} to {df.index.max().date()})")
            data[symbol] = df

    configs = build_configs()
    print(f"\nRunning {len(configs)} configurations...")

    summary_fields = [
        'label', 'symbol', 'strategy', 'strike_dist', 'hold_period', 'signal', 'lots',
        'total_trades', 'win_rate', 'total_pnl', 'total_return_pct', 'cagr_pct',
        'profit_factor', 'max_drawdown_pct', 'avg_premium', 'avg_zone_width_pct',
        'zone_hold_rate', 'max_consecutive_losses', 'avg_pnl_per_trade',
        'avg_win', 'avg_loss', 'max_win', 'max_loss', 'final_capital',
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

        sim = NonDirectionalSimulator(config, data[symbol].copy())
        summary = sim.run()

        if 'error' in summary:
            continue

        signal_part = 'unknown'
        for p in ['BB+ATR', 'BB_only', 'Consensus3']:
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
            'lots': config.lots_per_trade,
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
            'final_capital': summary['final_capital'],
        }

        with open(SUMMARY_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=summary_fields).writerow(row)
        all_summaries.append(row)

        trade_df = sim.get_trade_log()
        if not trade_df.empty:
            trade_df['config'] = label
            trade_df['lots'] = config.lots_per_trade
            all_trades.append(trade_df)

        ret = summary['total_return_pct']
        pf = summary['profit_factor']
        wr = summary['win_rate']
        dd = summary['max_drawdown_pct']
        status = f"W:{wr:.0f}% PF:{pf:.1f} Ret:{ret:+.1f}% DD:{dd:.1f}% CAGR:{summary['cagr_pct']:.1f}%"
        print(f"  [{i+1}/{len(configs)}] {label}: {summary['total_trades']}t | {status}")

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(TRADES_CSV, index=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*90}")
    print(f"COMPLETED {len(all_summaries)} configs in {elapsed:.0f}s")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    df_s = pd.DataFrame(all_summaries)
    if df_s.empty:
        return

    # Top 20 by CAGR
    viable = df_s[(df_s['win_rate'] > 50) & (df_s['profit_factor'] >= 1.0)].copy()
    viable = viable.sort_values('cagr_pct', ascending=False)

    print(f"\n{'='*90}")
    print(f"TOP 25 BY CAGR (WR>50%, PF>=1) — {len(viable)} viable of {len(df_s)}")
    print(f"{'='*90}")

    print(f"\n{'Label':<60} {'T':<4} {'WR%':<6} {'PF':<5} {'CAGR%':<7} "
          f"{'TotRet%':<8} {'DD%':<6} {'Prem':<6} {'AvgPnL':<8} {'Final':<10}")
    print("-" * 130)

    for _, r in viable.head(25).iterrows():
        pf = f"{r['profit_factor']:.1f}" if r['profit_factor'] < 100 else 'inf'
        print(f"{r['label']:<60} {r['total_trades']:<4} {r['win_rate']:<6.0f} "
              f"{pf:<5} {r['cagr_pct']:<7.1f} {r['total_return_pct']:<8.1f} "
              f"{r['max_drawdown_pct']:<6.1f} {r['avg_premium']:<6.0f} "
              f"Rs{r['avg_pnl_per_trade']:<7,.0f} {r['final_capital']:>10,.0f}")

    # Lot size comparison
    print(f"\n{'='*90}")
    print("LOT SIZE COMPARISON (avg across all BankNifty configs)")
    print(f"{'='*90}")
    bn_mask = df_s['symbol'] == 'BANKNIFTY'
    lot_comp = df_s[bn_mask].groupby('lots').agg({
        'cagr_pct': 'mean',
        'total_return_pct': 'mean',
        'max_drawdown_pct': 'mean',
        'win_rate': 'mean',
        'profit_factor': 'mean',
    }).round(1)
    print(lot_comp.to_string())

    # Strike distance comparison
    print(f"\n{'='*90}")
    print("STRIKE DISTANCE COMPARISON (BankNifty, 3 lots)")
    print(f"{'='*90}")
    mask = (df_s['symbol'] == 'BANKNIFTY') & (df_s['lots'] == 3)
    sd_comp = df_s[mask].groupby('strike_dist').agg({
        'cagr_pct': 'mean',
        'total_return_pct': 'mean',
        'win_rate': 'mean',
        'max_drawdown_pct': 'mean',
        'profit_factor': 'mean',
        'avg_premium': 'mean',
    }).round(1)
    print(sd_comp.to_string())

    # Best per category
    print(f"\n{'='*90}")
    print("BEST CONFIG PER CATEGORY (by CAGR)")
    print(f"{'='*90}")

    for symbol in ['BANKNIFTY', 'NIFTY50']:
        for strategy in ['short_strangle', 'iron_condor']:
            for hold in ['weekly', 'biweekly']:
                mask = ((viable['symbol'] == symbol) &
                        (viable['strategy'] == strategy) &
                        (viable['hold_period'] == hold) &
                        (viable['lots'] == 3))
                subset = viable[mask]
                if subset.empty:
                    continue
                best = subset.nlargest(1, 'cagr_pct').iloc[0]
                print(f"\n  {symbol} {strategy} {hold} (3 lots):")
                print(f"    Config: {best['label']}")
                print(f"    CAGR={best['cagr_pct']:.1f}%, Total={best['total_return_pct']:.1f}%, "
                      f"DD={best['max_drawdown_pct']:.1f}%")
                print(f"    Trades={best['total_trades']}, WR={best['win_rate']:.0f}%, "
                      f"PF={best['profit_factor']:.1f}")
                print(f"    Final Capital: Rs {best['final_capital']:,.0f}")


if __name__ == '__main__':
    main()
