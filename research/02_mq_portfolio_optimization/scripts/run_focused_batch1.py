"""
Focused Batch 1: High-impact configs for PS30+ targeting 30%+ CAGR
PRELOADS data once, then reuses across all configs for speed.
Writes results incrementally to CSV.
"""
import logging
logging.disable(logging.WARNING)

import sys, os, time, csv, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization_focused_results.csv")

FIELDNAMES = ['rank', 'label', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
              'total_trades', 'win_rate', 'final_value', 'total_return_pct', 'topups']

configs = []

# ── PRIORITY 1: Push best PS30 wider ──
configs.append(('HSL50_ATH20_EQ95', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.50, 'rebalance_ath_drawdown': 0.20}))
configs.append(('HSL60_ATH20_EQ95', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.60, 'rebalance_ath_drawdown': 0.20}))
configs.append(('HSL100_ATH20_EQ95', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 1.00, 'rebalance_ath_drawdown': 0.20}))
configs.append(('HSL40_ATH30_EQ95', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.30}))
configs.append(('HSL40_ATH50_EQ95', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.50}))
configs.append(('HSL50_ATH30_EQ95', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.50, 'rebalance_ath_drawdown': 0.30}))
configs.append(('HSL40_ATH20_EQ98', {'portfolio_size': 30, 'equity_allocation_pct': 0.98, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'debt_reserve_pct': 0.02}))
configs.append(('HSL40_ATH20_EQ100', {'portfolio_size': 30, 'equity_allocation_pct': 1.00, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'debt_reserve_pct': 0.0}))

# ── PRIORITY 2: Quality weight changes ──
configs.append(('WREV70_OPMG20', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'weight_revenue': 0.70, 'weight_opm_growth': 0.20, 'weight_opm': 0.05, 'weight_debt': 0.05}))
configs.append(('WREV50_OPMG30', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'weight_revenue': 0.50, 'weight_opm_growth': 0.30, 'weight_opm': 0.10, 'weight_debt': 0.10}))
configs.append(('WOPMG50_REV30', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'weight_opm_growth': 0.50, 'weight_revenue': 0.30, 'weight_opm': 0.10, 'weight_debt': 0.10}))

# ── PRIORITY 3: Fundamental filters ──
configs.append(('RELAXED_FUND', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'min_roe': 0.08, 'min_opm_3y': 0.10, 'min_revenue_growth_3y_cagr': 0.10, 'max_debt_to_equity': 0.50}))
configs.append(('TIGHT_FUND', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'min_roe': 0.18, 'min_opm_3y': 0.20, 'min_revenue_growth_3y_cagr': 0.20, 'max_debt_to_equity': 0.10}))
configs.append(('NO_OPM_DECLINE', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'require_opm_no_decline': False}))

# ── PRIORITY 4: Topup variations ──
configs.append(('TOPUP_30PCT', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'topup_pct_of_initial': 0.30}))
configs.append(('TOPUP_50PCT', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'topup_pct_of_initial': 0.50}))
configs.append(('TOPUP_30_SL8', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'topup_pct_of_initial': 0.30, 'topup_stop_loss_pct': 0.08}))

# ── PRIORITY 5: Rebalance frequency ──
configs.append(('QUARTERLY_REBAL', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'rebalance_months': [1, 4, 7, 10]}))
configs.append(('BIMONTHLY_REBAL', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20, 'rebalance_months': [1, 3, 5, 7, 9, 11]}))

# ── PRIORITY 6: COMBO configs ──
configs.append(('COMBO_WIDE_MOMENTUM', {'portfolio_size': 30, 'equity_allocation_pct': 0.98, 'hard_stop_loss': 0.60, 'rebalance_ath_drawdown': 0.30, 'weight_revenue': 0.60, 'weight_opm_growth': 0.25, 'weight_opm': 0.10, 'weight_debt': 0.05, 'topup_pct_of_initial': 0.30}))
configs.append(('COMBO_QUARTERLY_WIDE', {'portfolio_size': 30, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.50, 'rebalance_ath_drawdown': 0.25, 'rebalance_months': [1, 4, 7, 10], 'min_roe': 0.08, 'max_debt_to_equity': 0.50}))
configs.append(('COMBO_NOSTOP', {'portfolio_size': 30, 'equity_allocation_pct': 0.98, 'hard_stop_loss': 1.00, 'rebalance_ath_drawdown': 1.00, 'weight_revenue': 0.60, 'weight_opm_growth': 0.25}))
configs.append(('COMBO_PS40_AGG', {'portfolio_size': 40, 'equity_allocation_pct': 0.95, 'hard_stop_loss': 0.50, 'rebalance_ath_drawdown': 0.25, 'topup_pct_of_initial': 0.30, 'max_position_size': 0.08}))
configs.append(('COMBO_BEST_ALL', {'portfolio_size': 30, 'equity_allocation_pct': 0.98, 'hard_stop_loss': 0.50, 'rebalance_ath_drawdown': 0.25, 'weight_revenue': 0.50, 'weight_opm_growth': 0.30, 'weight_opm': 0.10, 'weight_debt': 0.10, 'topup_pct_of_initial': 0.30, 'min_roe': 0.08, 'max_debt_to_equity': 0.50}))


def main():
    total = len(configs)
    print(f"{'='*80}")
    print(f"  FOCUSED BATCH 1: {total} configs (PS30+) | PRELOADING DATA")
    print(f"{'='*80}")

    # PRELOAD data once
    t_load = time.time()
    print("Loading universe + price data...", flush=True)
    universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig())
    print(f"Data loaded in {time.time()-t_load:.0f}s | {len(price_data)} symbols", flush=True)

    # Write CSV header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    results = []
    t0 = time.time()
    best_cagr = 0

    for i, (label, params) in enumerate(configs, 1):
        t1 = time.time()
        print(f"[{i:2d}/{total}] {label} ...", end="", flush=True)
        try:
            cfg = MQBacktestConfig()
            for k, v in params.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            engine = MQBacktestEngine(cfg, universe, price_data)
            result = engine.run()
            row = {
                'rank': 0, 'label': label, 'cagr': result.cagr,
                'sharpe': result.sharpe_ratio, 'sortino': result.sortino_ratio,
                'max_drawdown': result.max_drawdown, 'calmar': result.calmar_ratio,
                'total_trades': result.total_trades, 'win_rate': result.win_rate,
                'final_value': result.final_value, 'total_return_pct': result.total_return_pct,
                'topups': result.total_topups,
            }
            results.append(row)
            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
            is_best = row['cagr'] > best_cagr
            if is_best:
                best_cagr = row['cagr']
            mark = " ***BEST***" if is_best else ""
            elapsed = time.time() - t1
            print(f" {elapsed:.0f}s | CAGR={row['cagr']:.2f}% Sharpe={row['sharpe']:.2f} MaxDD={row['max_drawdown']:.2f}%{mark}", flush=True)
        except Exception as e:
            elapsed = time.time() - t1
            print(f" FAILED {elapsed:.0f}s | {e}", flush=True)

    total_time = time.time() - t0
    results.sort(key=lambda r: r['cagr'], reverse=True)

    print(f"\n{'='*100}")
    print(f"  TOP RESULTS (by CAGR) | {len(results)} done | {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"{'='*100}")
    for rank, r in enumerate(results[:25], 1):
        print(f"{rank:2d}. {r['label']:<35} CAGR={r['cagr']:6.2f}%  Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_drawdown']:5.2f}%  Calmar={r['calmar']:5.2f}  Final={r['final_value']:>14,.0f}")

    # Rewrite CSV with ranks
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
            w.writerow(r)
    print(f"\nSaved to: {OUTPUT_CSV}")
    above30 = sum(1 for r in results if r['cagr'] >= 30)
    print(f"Configs >= 30% CAGR: {above30} | Best: {results[0]['cagr']:.2f}% ({results[0]['label']})")


if __name__ == '__main__':
    main()
