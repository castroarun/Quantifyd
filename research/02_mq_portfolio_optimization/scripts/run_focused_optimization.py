"""
Focused Optimization: High-Impact Untested Levers for PS30+
Targets 30%+ CAGR by exploring quality weights, fundamentals, topups,
very loose stops, and high equity allocation.
Writes results INCREMENTALLY after each config.
"""
import logging
logging.disable(logging.WARNING)

import sys, os, time, csv, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization_focused_results.csv")

FIELDNAMES = ['rank', 'label', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
              'total_trades', 'win_rate', 'avg_win_pct', 'avg_loss_pct',
              'final_value', 'total_return_pct', 'topups']

# ── Build high-impact configs ─────────────────────────────────────────
configs = []

# ── TIER 1: Best PS30 baseline from previous optimization + variations ──
# Previous best PS30: HSL=0.40, ATH=0.20, EQ=0.95 → 24.08%

# T1a: Push stops even wider (let winners run much longer)
for hsl in [0.50, 0.60, 0.70, 1.00]:  # 1.00 = effectively no hard stop
    configs.append((f'WIDE_STOP_HSL{int(hsl*100)}_ATH20', {
        'portfolio_size': 30, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': hsl, 'rebalance_ath_drawdown': 0.20,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
    }))

# T1b: Higher equity allocation (less debt reserve = more in stocks)
for eq in [0.97, 0.99, 1.00]:
    configs.append((f'HIGH_EQ{int(eq*100)}_HSL40_ATH20', {
        'portfolio_size': 30, 'equity_allocation_pct': eq,
        'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        'debt_reserve_pct': 1.0 - eq if eq < 1.0 else 0.0,
    }))

# T1c: Wider ATH drawdown (allow stocks to fall further before rebalance exit)
for ath in [0.25, 0.30, 0.40, 0.50, 1.00]:
    configs.append((f'WIDE_ATH{int(ath*100)}_HSL40_EQ95', {
        'portfolio_size': 30, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': ath,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
    }))

# ── TIER 2: Topup parameter sweep ──
# Topup = adding to winning positions on Darvas breakout
for topup_pct in [0.10, 0.20, 0.30, 0.50]:
    for topup_sl in [0.0, 0.05, 0.08]:
        for cooldown in [3, 5, 10]:
            configs.append((f'TOPUP_p{int(topup_pct*100)}_sl{int(topup_sl*100)}_cd{cooldown}', {
                'portfolio_size': 30, 'equity_allocation_pct': 0.95,
                'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20,
                'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
                'topup_pct_of_initial': topup_pct,
                'topup_stop_loss_pct': topup_sl,
                'topup_cooldown_days': cooldown,
            }))

# ── TIER 3: Quality weight variations (changes stock selection ranking) ──
weight_combos = [
    # Favor revenue growth heavily (more momentum)
    {'weight_revenue': 0.50, 'weight_opm': 0.20, 'weight_opm_growth': 0.20, 'weight_debt': 0.10},
    # Favor OPM growth (improving profitability)
    {'weight_revenue': 0.20, 'weight_opm': 0.20, 'weight_opm_growth': 0.50, 'weight_debt': 0.10},
    # Revenue + OPM growth combo
    {'weight_revenue': 0.40, 'weight_opm': 0.10, 'weight_opm_growth': 0.40, 'weight_debt': 0.10},
    # Equal weights
    {'weight_revenue': 0.25, 'weight_opm': 0.25, 'weight_opm_growth': 0.25, 'weight_debt': 0.25},
    # Pure momentum (revenue growth only)
    {'weight_revenue': 0.70, 'weight_opm': 0.10, 'weight_opm_growth': 0.10, 'weight_debt': 0.10},
]
for i, wc in enumerate(weight_combos):
    lbl = f"WEIGHTS_rev{int(wc['weight_revenue']*100)}_opmg{int(wc['weight_opm_growth']*100)}"
    configs.append((lbl, {
        'portfolio_size': 30, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        **wc,
    }))

# ── TIER 4: Fundamental filter relaxation (wider universe) ──
fund_combos = [
    # Relaxed (more stocks pass filters → potentially better momentum picks)
    {'min_roe': 0.08, 'min_opm_3y': 0.10, 'min_revenue_growth_3y_cagr': 0.10, 'max_debt_to_equity': 0.50},
    # Very relaxed
    {'min_roe': 0.05, 'min_opm_3y': 0.05, 'min_revenue_growth_3y_cagr': 0.05, 'max_debt_to_equity': 1.00},
    # Tighter (only highest quality)
    {'min_roe': 0.18, 'min_opm_3y': 0.20, 'min_revenue_growth_3y_cagr': 0.20, 'max_debt_to_equity': 0.10},
    # Remove revenue positive requirement
    {'min_roe': 0.12, 'min_opm_3y': 0.15, 'min_revenue_growth_3y_cagr': 0.15, 'require_revenue_positive_each_year': False},
    # Remove OPM decline requirement
    {'min_roe': 0.12, 'min_opm_3y': 0.15, 'min_revenue_growth_3y_cagr': 0.15, 'require_opm_no_decline': False},
]
for i, fc in enumerate(fund_combos):
    lbl = f"FUND{i+1}_roe{int(fc.get('min_roe',0.12)*100)}_de{int(fc.get('max_debt_to_equity',0.2)*100)}"
    configs.append((lbl, {
        'portfolio_size': 30, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        **fc,
    }))

# ── TIER 5: Consolidation/breakout params (how topups work) ──
for consol_days in [10, 15, 20, 30, 40]:
    for vol_mult in [1.0, 1.5, 2.0]:
        configs.append((f'CONSOL_d{consol_days}_vol{int(vol_mult*10)}x', {
            'portfolio_size': 30, 'equity_allocation_pct': 0.95,
            'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20,
            'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
            'consolidation_days': consol_days,
            'breakout_volume_multiplier': vol_mult,
        }))

# ── TIER 6: Position sizing & sector limits ──
for max_pos in [0.08, 0.10, 0.15, 0.20]:
    for max_sector in [0.20, 0.25, 0.35, 0.50]:
        configs.append((f'SIZE_pos{int(max_pos*100)}_sec{int(max_sector*100)}', {
            'portfolio_size': 30, 'equity_allocation_pct': 0.95,
            'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20,
            'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
            'max_position_size': max_pos,
            'max_sector_weight': max_sector,
        }))

# ── TIER 7: Rebalance frequency variations ──
for months in [[1, 7], [1, 4, 7, 10], [3, 9], [1, 3, 5, 7, 9, 11], [6, 12]]:
    lbl = f"REBAL_{'_'.join(str(m) for m in months)}"
    configs.append((lbl, {
        'portfolio_size': 30, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.20,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        'rebalance_months': months,
    }))

# ── TIER 8: COMBO CONFIGS (combine best ideas) ──
combo_configs = [
    # Aggressive: wide stops + high equity + aggressive topups
    ('COMBO_AGG_V1', {
        'portfolio_size': 30, 'equity_allocation_pct': 0.99,
        'hard_stop_loss': 0.60, 'rebalance_ath_drawdown': 0.30,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        'topup_pct_of_initial': 0.30,
        'weight_revenue': 0.50, 'weight_opm_growth': 0.30,
        'weight_opm': 0.10, 'weight_debt': 0.10,
    }),
    # No stops at all + pure momentum selection
    ('COMBO_NOSTOP_MOMENTUM', {
        'portfolio_size': 30, 'equity_allocation_pct': 0.98,
        'hard_stop_loss': 1.00, 'rebalance_ath_drawdown': 1.00,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        'weight_revenue': 0.70, 'weight_opm_growth': 0.20,
        'weight_opm': 0.05, 'weight_debt': 0.05,
    }),
    # Quarterly rebalance + wide stops + relaxed fundamentals
    ('COMBO_QUARTERLY_WIDE', {
        'portfolio_size': 30, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': 0.50, 'rebalance_ath_drawdown': 0.25,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        'rebalance_months': [1, 4, 7, 10],
        'min_roe': 0.08, 'max_debt_to_equity': 0.50,
    }),
    # Best from prev optimization + wider stops
    ('COMBO_PREV_BEST_WIDER', {
        'portfolio_size': 30, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': 0.50, 'rebalance_ath_drawdown': 0.20,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
    }),
    # PS40 with aggressive config
    ('COMBO_PS40_AGG', {
        'portfolio_size': 40, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': 0.50, 'rebalance_ath_drawdown': 0.25,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        'topup_pct_of_initial': 0.30,
        'max_position_size': 0.08, 'max_sector_weight': 0.20,
    }),
    # PS50 more diversified
    ('COMBO_PS50_WIDE', {
        'portfolio_size': 50, 'equity_allocation_pct': 0.95,
        'hard_stop_loss': 0.40, 'rebalance_ath_drawdown': 0.25,
        'trailing_stop_loss': False, 'daily_ath_drawdown_exit': False,
        'max_position_size': 0.05, 'max_sector_weight': 0.25,
    }),
]
configs.extend(combo_configs)


def run_single(label, params):
    try:
        cfg = MQBacktestConfig()
        for k, v in params.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                print(f"  [WARN] {label}: unknown param '{k}'")
        engine = MQBacktestEngine(cfg)
        result = engine.run()
        return {
            'label': label,
            'cagr': result.cagr,
            'sharpe': result.sharpe_ratio,
            'sortino': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar': result.calmar_ratio,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'avg_win_pct': result.avg_win_pct,
            'avg_loss_pct': result.avg_loss_pct,
            'final_value': result.final_value,
            'total_return_pct': result.total_return_pct,
            'topups': result.total_topups,
        }
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        traceback.print_exc()
        return None


def main():
    total = len(configs)
    print(f"{'='*80}")
    print(f"  FOCUSED OPTIMIZATION: {total} configs targeting 30%+ CAGR (PS30+)")
    print(f"  Results written incrementally to: {OUTPUT_CSV}")
    print(f"{'='*80}")

    # Write CSV header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

    results = []
    t0 = time.time()
    best_cagr = 0

    for i, (label, params) in enumerate(configs, 1):
        t1 = time.time()
        print(f"\n[{i:3d}/{total}] Running: {label} ...", end="", flush=True)
        row = run_single(label, params)
        elapsed = time.time() - t1

        if row:
            row['rank'] = 0  # Will be assigned at the end
            results.append(row)

            # Append to CSV incrementally
            with open(OUTPUT_CSV, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writerow(row)

            is_best = row['cagr'] > best_cagr
            if is_best:
                best_cagr = row['cagr']
            marker = " *** NEW BEST ***" if is_best else ""
            print(f"  {elapsed:.0f}s | CAGR={row['cagr']:.2f}% Sharpe={row['sharpe']:.2f} MaxDD={row['max_drawdown']:.2f}%{marker}", flush=True)
        else:
            print(f"  FAILED in {elapsed:.0f}s", flush=True)

    total_time = time.time() - t0

    # Final sorted results
    results.sort(key=lambda r: r['cagr'], reverse=True)
    print(f"\n{'='*120}")
    print(f"  TOP 20 RESULTS (sorted by CAGR) | Total: {len(results)} | Time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"{'='*120}")
    for rank, r in enumerate(results[:20], 1):
        r['rank'] = rank
        print(f"{rank:3d}. {r['label']:<40} CAGR={r['cagr']:6.2f}%  Sharpe={r['sharpe']:5.2f}  MaxDD={r['max_drawdown']:5.2f}%  Calmar={r['calmar']:5.2f}  Trades={r['total_trades']:3d}  Final={r['final_value']:>14,.0f}")

    # Rewrite CSV with ranks
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
            writer.writerow(r)
    print(f"\nFinal ranked results saved to: {OUTPUT_CSV}")

    # Summary stats
    above_30 = [r for r in results if r['cagr'] >= 30]
    above_25 = [r for r in results if r['cagr'] >= 25]
    print(f"\nConfigs with CAGR >= 30%: {len(above_30)}")
    print(f"Configs with CAGR >= 25%: {len(above_25)}")
    print(f"Best CAGR: {results[0]['cagr']:.2f}% ({results[0]['label']})")


if __name__ == '__main__':
    main()
