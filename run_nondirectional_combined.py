"""
Combined Non-Directional Strategy: BankNifty Monthly + Nifty Weekly
====================================================================
- BankNifty: BB Squeeze + Trend Filter, bi-weekly hold, monthly options (Option B: min 15 DTE)
- Nifty: BB Squeeze + Trend Filter, weekly hold, weekly Thursday expiry
- Realistic DTE from actual expiry calendar
- Tests capital splits: 6L/4L, 7L/3L, 5L/5L
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
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'nondirectional_combined_summary.csv')
TRADES_CSV = os.path.join(OUTPUT_DIR, 'nondirectional_combined_trades.csv')


def build_configs():
    """Build all config pairs (BNF leg + Nifty leg) with different capital splits."""
    configs = []

    # Capital splits to test: (bnf_capital, bnf_lots, nifty_capital, nifty_lots, label)
    splits = [
        (6_00_000, 3, 4_00_000, 2, '6L_4L'),
        (7_00_000, 4, 3_00_000, 2, '7L_3L'),
        (5_00_000, 3, 5_00_000, 3, '5L_5L'),
        (8_00_000, 4, 2_00_000, 1, '8L_2L'),
    ]

    # Signal modes to test
    signal_configs = [
        ('BB_only', True, False, 0),
        ('BB+ATR', True, True, 0),
    ]

    # Strike distances
    strike_dists = [1.5, 2.0]

    # Trend filter settings
    trend_configs = [
        (True, 2.0, 'TF2.0'),
        (True, 1.5, 'TF1.5'),
        (False, 2.0, 'noTF'),
    ]

    for bnf_cap, bnf_lots, nft_cap, nft_lots, split_label in splits:
        for sig_name, bb, atr_c, consensus in signal_configs:
            for sd in strike_dists:
                for tf_on, tf_mult, tf_label in trend_configs:
                    label = f'COMBINED_{split_label}_{sig_name}_SD{sd}_{tf_label}'

                    bnf_config = SimConfig(
                        symbol='BANKNIFTY',
                        strategy=StrategyType.SHORT_STRANGLE,
                        strike_distance_atr=sd,
                        wing_distance_atr=sd + 1.5,
                        hold_bars=10,
                        use_dynamic_dte=True,
                        min_dte_monthly=15,
                        require_bb_squeeze=bb,
                        require_atr_contract=atr_c,
                        min_consensus=consensus,
                        use_trend_filter=tf_on,
                        trend_sma_len=20,
                        trend_atr_mult=tf_mult,
                        capital=bnf_cap,
                        lots_per_trade=bnf_lots,
                        bars_per_day=1,
                        min_days_between_trades=10,
                    )

                    nft_config = SimConfig(
                        symbol='NIFTY50',
                        strategy=StrategyType.SHORT_STRANGLE,
                        strike_distance_atr=sd,
                        wing_distance_atr=sd + 1.5,
                        hold_bars=5,
                        use_dynamic_dte=True,
                        min_dte_monthly=15,
                        nifty_next_week=True,    # Sell next week's expiry
                        require_bb_squeeze=bb,
                        require_atr_contract=atr_c,
                        min_consensus=consensus,
                        use_trend_filter=tf_on,
                        trend_sma_len=20,
                        trend_atr_mult=tf_mult,
                        capital=nft_cap,
                        lots_per_trade=nft_lots,
                        bars_per_day=1,
                        min_days_between_trades=5,
                    )

                    configs.append({
                        'label': label,
                        'split': split_label,
                        'bnf_config': bnf_config,
                        'nft_config': nft_config,
                        'bnf_cap': bnf_cap,
                        'nft_cap': nft_cap,
                    })

    return configs


def merge_equity_curves(bnf_eq, nft_eq, bnf_start_cap, nft_start_cap):
    """Merge two equity curves into a combined daily curve."""
    # Convert to DataFrames indexed by date
    df_bnf = pd.DataFrame(bnf_eq).set_index('date')['capital']
    df_nft = pd.DataFrame(nft_eq).set_index('date')['capital']

    # Forward-fill to get capital at every date
    all_dates = sorted(set(df_bnf.index) | set(df_nft.index))
    combined = []
    bnf_cap = bnf_start_cap
    nft_cap = nft_start_cap
    for d in all_dates:
        if d in df_bnf.index:
            bnf_cap = df_bnf[d]
        if d in df_nft.index:
            nft_cap = df_nft[d]
        combined.append({'date': d, 'capital': bnf_cap + nft_cap})

    return combined


def compute_combined_stats(equity_curve, total_capital):
    """Compute summary stats from a combined equity curve."""
    if len(equity_curve) < 2:
        return {}

    eq = [e['capital'] for e in equity_curve]
    peak = eq[0]
    max_dd_pct = 0
    for val in eq:
        if val > peak:
            peak = val
        dd_pct = (peak - val) / peak * 100 if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    start_cap = equity_curve[0]['capital']
    end_cap = equity_curve[-1]['capital']
    start_date = pd.Timestamp(equity_curve[0]['date'])
    end_date = pd.Timestamp(equity_curve[-1]['date'])
    years = (end_date - start_date).days / 365.25

    total_return = (end_cap - start_cap) / start_cap * 100
    cagr = ((end_cap / start_cap) ** (1 / years) - 1) * 100 if years > 0 and end_cap > 0 else 0

    return {
        'total_return_pct': round(total_return, 2),
        'cagr_pct': round(cagr, 2),
        'max_drawdown_pct': round(max_dd_pct, 2),
        'final_capital': round(end_cap, 2),
    }


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
    print(f"\nRunning {len(configs)} combined configurations...")

    summary_fields = [
        'label', 'split', 'signal', 'strike_dist', 'trend_filter',
        # BNF leg
        'bnf_trades', 'bnf_win_rate', 'bnf_pf', 'bnf_cagr', 'bnf_return', 'bnf_dd',
        'bnf_avg_dte', 'bnf_final_cap',
        # Nifty leg
        'nft_trades', 'nft_win_rate', 'nft_pf', 'nft_cagr', 'nft_return', 'nft_dd',
        'nft_avg_dte', 'nft_final_cap',
        # Combined
        'combined_return', 'combined_cagr', 'combined_dd', 'combined_final',
        'total_trades', 'total_capital',
    ]

    with open(SUMMARY_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    all_trades = []
    all_summaries = []

    for i, cfg in enumerate(configs):
        label = cfg['label']

        # Run BankNifty leg
        bnf_sim = NonDirectionalSimulator(cfg['bnf_config'], data['BANKNIFTY'].copy())
        bnf_summary = bnf_sim.run()

        # Run Nifty leg
        nft_sim = NonDirectionalSimulator(cfg['nft_config'], data['NIFTY50'].copy())
        nft_summary = nft_sim.run()

        bnf_ok = 'error' not in bnf_summary
        nft_ok = 'error' not in nft_summary

        if not bnf_ok and not nft_ok:
            continue

        # Average DTE from trade logs
        bnf_trades_df = bnf_sim.get_trade_log()
        nft_trades_df = nft_sim.get_trade_log()
        bnf_avg_dte = bnf_trades_df['dte'].mean() if not bnf_trades_df.empty else 0
        nft_avg_dte = nft_trades_df['dte'].mean() if not nft_trades_df.empty else 0

        # Merge equity curves
        combined_eq = merge_equity_curves(
            bnf_sim.equity_curve, nft_sim.equity_curve,
            cfg['bnf_cap'], cfg['nft_cap']
        )
        combined_stats = compute_combined_stats(combined_eq, cfg['bnf_cap'] + cfg['nft_cap'])

        # Extract signal/trend from label
        sig_part = 'BB_only' if 'BB_only' in label else 'BB+ATR'
        tf_part = 'noTF'
        for t in ['TF1.5', 'TF2.0', 'noTF']:
            if t in label:
                tf_part = t
                break

        row = {
            'label': label,
            'split': cfg['split'],
            'signal': sig_part,
            'strike_dist': cfg['bnf_config'].strike_distance_atr,
            'trend_filter': tf_part,
            # BNF
            'bnf_trades': bnf_summary.get('total_trades', 0),
            'bnf_win_rate': bnf_summary.get('win_rate', 0),
            'bnf_pf': bnf_summary.get('profit_factor', 0),
            'bnf_cagr': bnf_summary.get('cagr_pct', 0),
            'bnf_return': bnf_summary.get('total_return_pct', 0),
            'bnf_dd': bnf_summary.get('max_drawdown_pct', 0),
            'bnf_avg_dte': round(bnf_avg_dte, 1),
            'bnf_final_cap': bnf_summary.get('final_capital', cfg['bnf_cap']),
            # Nifty
            'nft_trades': nft_summary.get('total_trades', 0),
            'nft_win_rate': nft_summary.get('win_rate', 0),
            'nft_pf': nft_summary.get('profit_factor', 0),
            'nft_cagr': nft_summary.get('cagr_pct', 0),
            'nft_return': nft_summary.get('total_return_pct', 0),
            'nft_dd': nft_summary.get('max_drawdown_pct', 0),
            'nft_avg_dte': round(nft_avg_dte, 1),
            'nft_final_cap': nft_summary.get('final_capital', cfg['nft_cap']),
            # Combined
            'combined_return': combined_stats.get('total_return_pct', 0),
            'combined_cagr': combined_stats.get('cagr_pct', 0),
            'combined_dd': combined_stats.get('max_drawdown_pct', 0),
            'combined_final': combined_stats.get('final_capital', 0),
            'total_trades': bnf_summary.get('total_trades', 0) + nft_summary.get('total_trades', 0),
            'total_capital': cfg['bnf_cap'] + cfg['nft_cap'],
        }

        with open(SUMMARY_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=summary_fields).writerow(row)
        all_summaries.append(row)

        # Save trade logs
        if not bnf_trades_df.empty:
            bnf_trades_df['config'] = label
            bnf_trades_df['leg'] = 'BNF'
            all_trades.append(bnf_trades_df)
        if not nft_trades_df.empty:
            nft_trades_df['config'] = label
            nft_trades_df['leg'] = 'NIFTY'
            all_trades.append(nft_trades_df)

        # Progress
        bnf_status = f"BNF:{bnf_summary.get('total_trades',0)}t WR:{bnf_summary.get('win_rate',0):.0f}% DTE:{bnf_avg_dte:.0f}"
        nft_status = f"NFT:{nft_summary.get('total_trades',0)}t WR:{nft_summary.get('win_rate',0):.0f}% DTE:{nft_avg_dte:.0f}"
        comb_status = f"Combined: CAGR:{combined_stats.get('cagr_pct',0):.1f}% Ret:{combined_stats.get('total_return_pct',0):.1f}% DD:{combined_stats.get('max_drawdown_pct',0):.1f}%"
        print(f"  [{i+1}/{len(configs)}] {label}")
        print(f"    {bnf_status} | {nft_status} | {comb_status}")

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(TRADES_CSV, index=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*100}")
    print(f"COMPLETED {len(all_summaries)} configs in {elapsed:.0f}s")

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    df_s = pd.DataFrame(all_summaries)
    if df_s.empty:
        return

    df_s = df_s.sort_values('combined_cagr', ascending=False)

    print(f"\n{'='*100}")
    print(f"TOP 15 COMBINED CONFIGS BY CAGR")
    print(f"{'='*100}")

    print(f"\n{'Label':<50} {'Split':<7} {'BNF':<20} {'NFT':<20} "
          f"{'Comb CAGR':<10} {'Comb Ret':<10} {'Comb DD':<8} {'Final':<12}")
    print("-" * 140)

    for _, r in df_s.head(15).iterrows():
        bnf_info = f"{r['bnf_trades']}t WR{r['bnf_win_rate']:.0f}% DTE{r['bnf_avg_dte']:.0f}"
        nft_info = f"{r['nft_trades']}t WR{r['nft_win_rate']:.0f}% DTE{r['nft_avg_dte']:.0f}"
        print(f"{r['label']:<50} {r['split']:<7} {bnf_info:<20} {nft_info:<20} "
              f"{r['combined_cagr']:<10.1f} {r['combined_return']:<10.1f} "
              f"{r['combined_dd']:<8.1f} Rs{r['combined_final']:>10,.0f}")

    # Capital split comparison (BB_only, SD1.5, TF2.0)
    print(f"\n{'='*100}")
    print("CAPITAL SPLIT COMPARISON (BB_only, SD1.5, TF2.0)")
    print(f"{'='*100}")
    mask = ((df_s['signal'] == 'BB_only') & (df_s['strike_dist'] == 1.5) &
            (df_s['trend_filter'] == 'TF2.0'))
    split_comp = df_s[mask][['split', 'bnf_trades', 'bnf_cagr', 'bnf_avg_dte',
                              'nft_trades', 'nft_cagr', 'nft_avg_dte',
                              'combined_cagr', 'combined_return', 'combined_dd', 'combined_final']]
    if not split_comp.empty:
        print(split_comp.to_string(index=False))

    # Trend filter comparison
    print(f"\n{'='*100}")
    print("TREND FILTER IMPACT (6L_4L split, BB_only, SD1.5)")
    print(f"{'='*100}")
    mask = ((df_s['split'] == '6L_4L') & (df_s['signal'] == 'BB_only') &
            (df_s['strike_dist'] == 1.5))
    tf_comp = df_s[mask][['trend_filter', 'bnf_trades', 'bnf_win_rate', 'bnf_cagr',
                           'nft_trades', 'nft_win_rate', 'nft_cagr',
                           'combined_cagr', 'combined_return', 'combined_dd']]
    if not tf_comp.empty:
        print(tf_comp.to_string(index=False))

    # DTE analysis
    print(f"\n{'='*100}")
    print("AVERAGE DTE AT ENTRY (best config)")
    print(f"{'='*100}")
    if not df_s.empty:
        best = df_s.iloc[0]
        print(f"  BankNifty avg DTE: {best['bnf_avg_dte']:.1f} days (monthly options, Option B: min 15 DTE)")
        print(f"  Nifty avg DTE: {best['nft_avg_dte']:.1f} days (weekly Thursday expiry)")


if __name__ == '__main__':
    main()
