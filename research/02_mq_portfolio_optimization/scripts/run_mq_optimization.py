"""
MQ Strategy Parameter Optimizer
================================

Systematically tests parameter combinations to find the
best CAGR, Sharpe, and lowest drawdown configurations.

Uses multiprocessing to run backtests in parallel across CPU cores.
Each worker process loads data once and reuses it for all its runs.

Runs in rounds:
  Round 1: Broad sweep of exit thresholds + portfolio construction
  Round 2: Refine top configs with entry/topup/rebalance variations
  Round 3: Fine-tune consolidation/breakout params on winners

Results saved to backtest_data/optimization_results.json after each round.
Summary written to docs/MQ-OPTIMIZATION-REPORT.md at the end.
"""

import json
import time
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from itertools import product
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))

from services.mq_portfolio import MQBacktestConfig
from services.mq_backtest_engine import MQBacktestEngine

# Suppress verbose logging in workers
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESULTS_FILE = Path('backtest_data') / 'optimization_results.json'
REPORT_FILE = Path('docs') / 'MQ-OPTIMIZATION-REPORT.md'
RESULTS_FILE.parent.mkdir(exist_ok=True)
REPORT_FILE.parent.mkdir(exist_ok=True)

NUM_WORKERS = max(1, cpu_count() - 1)  # Leave 1 core free

# Per-worker globals (initialized once per process)
_worker_universe = None
_worker_price_data = None


def _worker_init():
    """Initialize each worker process with preloaded data."""
    global _worker_universe, _worker_price_data
    # Suppress logging in workers
    logging.getLogger('services.mq_backtest_engine').setLevel(logging.ERROR)
    logging.getLogger('services.mq_portfolio').setLevel(logging.ERROR)
    logging.getLogger('services.nifty500_universe').setLevel(logging.ERROR)
    logging.getLogger('services.momentum_filter').setLevel(logging.ERROR)
    logging.getLogger('services.consolidation_breakout').setLevel(logging.ERROR)

    _worker_universe, _worker_price_data = MQBacktestEngine.preload_data(MQBacktestConfig())


def _run_task(args):
    """Run a single backtest in a worker process."""
    config_dict, label = args
    t0 = time.time()
    try:
        config = MQBacktestConfig(**config_dict)
        engine = MQBacktestEngine(config, _worker_universe, _worker_price_data)
        result = engine.run(quality_scores={}, progress_callback=None)
        dur = time.time() - t0

        exit_pnl = {}
        for reason, data in result.exit_reason_pnl.items():
            exit_pnl[reason] = {
                'count': data['count'],
                'total_pnl': data['total_pnl'],
                'avg_return_pct': data['avg_return_pct'],
                'win_rate': data['win_rate'],
            }

        return {
            'label': label, 'status': 'ok', 'duration_s': round(dur, 1),
            'cagr': result.cagr, 'sharpe': result.sharpe_ratio,
            'sortino': result.sortino_ratio, 'max_drawdown': result.max_drawdown,
            'calmar': result.calmar_ratio, 'total_return_pct': result.total_return_pct,
            'final_value': result.final_value, 'total_trades': result.total_trades,
            'win_rate': result.win_rate, 'avg_win_pct': result.avg_win_pct,
            'avg_loss_pct': result.avg_loss_pct, 'total_topups': result.total_topups,
            'exit_reason_counts': result.exit_reason_counts,
            'exit_reason_pnl': exit_pnl,
            'consolidation_detections': len(result.consolidation_log),
            'config': {
                'portfolio_size': config.portfolio_size,
                'equity_allocation_pct': config.equity_allocation_pct,
                'hard_stop_loss': config.hard_stop_loss,
                'rebalance_ath_drawdown': config.rebalance_ath_drawdown,
                'ath_proximity_threshold': config.ath_proximity_threshold,
                'topup_pct_of_initial': config.topup_pct_of_initial,
                'consolidation_days': config.consolidation_days,
                'consolidation_range_pct': config.consolidation_range_pct,
                'breakout_volume_multiplier': config.breakout_volume_multiplier,
                'topup_cooldown_days': config.topup_cooldown_days,
                'max_sector_weight': config.max_sector_weight,
                'rebalance_months': config.rebalance_months,
                'quarterly_decline_count': config.quarterly_decline_count,
                'trailing_stop_loss': config.trailing_stop_loss,
                'daily_ath_drawdown_exit': config.daily_ath_drawdown_exit,
            },
        }
    except Exception as e:
        return {'label': label, 'status': 'error', 'error': str(e), 'duration_s': round(time.time() - t0, 1)}


def run_parallel(tasks, pool, round_name, total_label=""):
    """Run tasks in parallel with progress tracking."""
    total = len(tasks)
    logger.info(f"  {total_label or round_name}: {total} runs across {NUM_WORKERS} workers...")
    results = []
    start = time.time()

    for i, result in enumerate(pool.imap_unordered(_run_task, tasks)):
        results.append(result)
        done = len(results)
        if done % 10 == 0 or done == total:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            status = result.get('status', '?')
            cagr = f"{result.get('cagr', 0):.1f}%" if status == 'ok' else 'ERR'
            logger.info(f"  [{done}/{total}] {cagr} ({rate:.1f}/s, ETA {eta:.0f}s)")

    return results


def save_round(results: list, round_name: str):
    """Save round results to JSON."""
    existing = {}
    if RESULTS_FILE.exists():
        try:
            existing = json.loads(RESULTS_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass

    existing[round_name] = {
        'timestamp': datetime.now().isoformat(),
        'count': len(results),
        'results': results,
    }
    RESULTS_FILE.write_text(json.dumps(existing, indent=2, default=str), encoding='utf-8')


def top_n(results, n=5, key='cagr'):
    valid = [r for r in results if r.get('status') == 'ok']
    return sorted(valid, key=lambda x: x.get(key, 0), reverse=True)[:n]


def print_board(results, title, n=10):
    valid = [r for r in results if r.get('status') == 'ok']
    if not valid:
        return
    by_cagr = sorted(valid, key=lambda x: x['cagr'], reverse=True)
    print(f"\n{'='*110}")
    print(f"  {title} - TOP {min(n, len(by_cagr))}")
    print(f"{'='*110}")
    print(f"{'#':>3} {'Label':<45} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} {'WinR':>6} {'Trades':>6}")
    print("-" * 110)
    for i, r in enumerate(by_cagr[:n]):
        print(f"{i+1:>3} {r['label']:<45} {r['cagr']:>6.1f}% {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['max_drawdown']:>6.1f}% {r['calmar']:>7.2f} {r['win_rate']:>5.1f}% {r['total_trades']:>6}")


def make_config_dict(**overrides):
    """Build a config kwargs dict with defaults + overrides."""
    cfg = {}
    for k, v in overrides.items():
        cfg[k] = v
    if 'equity_allocation_pct' in cfg and 'debt_reserve_pct' not in cfg:
        cfg['debt_reserve_pct'] = 1 - cfg['equity_allocation_pct']
    return cfg


# =============================================================================
# ROUND 1: Exit Thresholds + Portfolio Construction
# =============================================================================
def round_1(pool):
    """
    Sweep:
    - hard_stop_loss: 20%, 25%, 30%, 40%, 50%
    - rebalance_ath_drawdown: 10%, 15%, 20%, 25%, 30%
    - portfolio_size: 15, 20, 25, 30
    - equity_allocation_pct: 0.80, 0.90, 0.95
    = 5 * 5 * 4 * 3 = 300 combos
    """
    tasks = []
    for hsl, ath_dd, ps, eq in product(
        [0.20, 0.25, 0.30, 0.40, 0.50],
        [0.10, 0.15, 0.20, 0.25, 0.30],
        [15, 20, 25, 30],
        [0.80, 0.90, 0.95],
    ):
        label = f"HSL{int(hsl*100)}_ATH{int(ath_dd*100)}_PS{ps}_EQ{int(eq*100)}"
        cfg = make_config_dict(
            hard_stop_loss=hsl, rebalance_ath_drawdown=ath_dd,
            portfolio_size=ps, equity_allocation_pct=eq,
        )
        tasks.append((cfg, label))

    results = run_parallel(tasks, pool, 'round_1', 'ROUND 1: Exit + Portfolio')
    save_round(results, 'round_1')
    print_board(results, "ROUND 1: Exit + Portfolio")
    return results


# =============================================================================
# ROUND 2: Entry + Topup + Rebalance Frequency
# =============================================================================
def round_2(r1_results, pool):
    """
    Top 5 from R1, vary:
    - ath_proximity_threshold: 0.05, 0.10, 0.15, 0.20
    - topup_pct_of_initial: 0, 0.10, 0.20, 0.30
    - rebalance_months: [1,7] vs [1,4,7,10] vs [3,9]
    = 5 * 4 * 4 * 3 = 240 combos
    """
    top5 = top_n(r1_results, 5)
    tasks = []

    for base in top5:
        bc = base['config']
        for ath_prox, topup, reb in product(
            [0.05, 0.10, 0.15, 0.20],
            [0.0, 0.10, 0.20, 0.30],
            [[1, 7], [1, 4, 7, 10], [3, 9]],
        ):
            reb_lbl = f"R{''.join(str(m) for m in reb)}"
            tu_lbl = f"TU{int(topup*100)}" if topup > 0 else "NOTU"
            label = f"R2_{base['label'][:18]}_AP{int(ath_prox*100)}_{tu_lbl}_{reb_lbl}"

            cfg = make_config_dict(
                hard_stop_loss=bc['hard_stop_loss'],
                rebalance_ath_drawdown=bc['rebalance_ath_drawdown'],
                portfolio_size=bc['portfolio_size'],
                equity_allocation_pct=bc['equity_allocation_pct'],
                ath_proximity_threshold=ath_prox,
                topup_pct_of_initial=topup,
                rebalance_months=reb,
            )
            tasks.append((cfg, label))

    results = run_parallel(tasks, pool, 'round_2', 'ROUND 2: Entry + Topup + Rebalance')
    save_round(results, 'round_2')
    print_board(results, "ROUND 2: Entry + Topup + Rebalance")
    return results


# =============================================================================
# ROUND 3: Consolidation/Breakout Fine-tuning
# =============================================================================
def round_3(r2_results, pool):
    """
    Top 3 from R2 (only those WITH topups), vary:
    - consolidation_days: 10, 15, 20, 30
    - consolidation_range_pct: 0.03, 0.05, 0.08
    - breakout_volume_multiplier: 1.0, 1.3, 1.5, 2.0
    - topup_cooldown_days: 3, 5, 10
    = 3 * 4 * 3 * 4 * 3 = 432 combos
    """
    valid_with_topups = [r for r in r2_results if r.get('status') == 'ok' and r['config'].get('topup_pct_of_initial', 0) > 0]
    top3 = top_n(valid_with_topups, 3)

    if not top3:
        logger.warning("  No R2 results with topups - skipping R3")
        return []

    tasks = []
    for base in top3:
        bc = base['config']
        for cd, cr, bv, cl in product(
            [10, 15, 20, 30],
            [0.03, 0.05, 0.08],
            [1.0, 1.3, 1.5, 2.0],
            [3, 5, 10],
        ):
            label = f"R3_{base['label'][:22]}_CD{cd}_CR{int(cr*100)}_BV{bv}_CL{cl}"
            cfg = make_config_dict(
                hard_stop_loss=bc['hard_stop_loss'],
                rebalance_ath_drawdown=bc['rebalance_ath_drawdown'],
                portfolio_size=bc['portfolio_size'],
                equity_allocation_pct=bc['equity_allocation_pct'],
                ath_proximity_threshold=bc['ath_proximity_threshold'],
                topup_pct_of_initial=bc['topup_pct_of_initial'],
                rebalance_months=bc['rebalance_months'],
                consolidation_days=cd,
                consolidation_range_pct=cr,
                breakout_volume_multiplier=bv,
                topup_cooldown_days=cl,
            )
            tasks.append((cfg, label))

    results = run_parallel(tasks, pool, 'round_3', 'ROUND 3: Consolidation + Breakout')
    save_round(results, 'round_3')
    print_board(results, "ROUND 3: Consolidation + Breakout")
    return results


# =============================================================================
# ROUND 4: Trailing Stop Loss + Daily ATH Drawdown
# =============================================================================
def round_4(pool):
    """
    Fresh sweep with trailing stop + daily ATH exit combos.
    Fixed: PS=15, EQ=0.95 (R1 winners)

    Variations:
    - trailing_stop_loss: True, False
    - daily_ath_drawdown_exit: True, False
    - hard_stop_loss (trailing threshold): 0.12, 0.15, 0.18, 0.20, 0.25
    - rebalance_ath_drawdown: 0.08, 0.10, 0.12, 0.15, 0.18, 0.20
    = 2 * 2 * 5 * 6 = 120 combos
    """
    tasks = []
    for tsl, dath, hsl, ath_dd in product(
        [True, False],
        [True, False],
        [0.12, 0.15, 0.18, 0.20, 0.25],
        [0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
    ):
        # Skip: no trailing + no daily = already tested in R1
        if not tsl and not dath:
            continue
        # Skip: trailing_stop at same or tighter level than ath_drawdown makes ath redundant
        # (keep these to see the interaction)

        tsl_lbl = "TSL" if tsl else "FSL"
        dath_lbl = "DATH" if dath else "RATH"
        label = f"R4_{tsl_lbl}{int(hsl*100)}_{dath_lbl}{int(ath_dd*100)}_PS15_EQ95"
        cfg = make_config_dict(
            portfolio_size=15,
            equity_allocation_pct=0.95,
            hard_stop_loss=hsl,
            rebalance_ath_drawdown=ath_dd,
            trailing_stop_loss=tsl,
            daily_ath_drawdown_exit=dath,
        )
        tasks.append((cfg, label))

    results = run_parallel(tasks, pool, 'round_4', 'ROUND 4: Trailing SL + Daily ATH')
    save_round(results, 'round_4')
    print_board(results, "ROUND 4: Trailing SL + Daily ATH")
    return results


# =============================================================================
# Consolidation Diagnostic (single process, with logging)
# =============================================================================
def run_consolidation_diagnostic(universe, price_data):
    """
    Run a single backtest with verbose consolidation logging
    to identify actual consolidation zones for manual chart verification.
    """
    from services.mq_portfolio import MQBacktestConfig
    from services.mq_backtest_engine import MQBacktestEngine

    config = MQBacktestConfig(
        portfolio_size=15,
        equity_allocation_pct=0.95,
        consolidation_range_pct=0.08,  # Wider range to catch more
        consolidation_days=15,         # Shorter window to catch more
        breakout_volume_multiplier=1.2, # Lower bar for volume
        topup_pct_of_initial=0.20,
    )

    engine = MQBacktestEngine(config, universe, price_data)
    result = engine.run(quality_scores={})

    # Collect unique consolidation detections
    consol_log = result.consolidation_log
    logger.info(f"\n  CONSOLIDATION DIAGNOSTIC: {len(consol_log)} detections found")

    if consol_log:
        # Show unique stock+date combos
        seen = set()
        unique = []
        for entry in consol_log:
            key = f"{entry['symbol']}_{entry['consol_start']}"
            if key not in seen:
                seen.add(key)
                unique.append(entry)

        unique.sort(key=lambda x: x['date'])

        print(f"\n{'='*120}")
        print(f"  CONSOLIDATION ZONES DETECTED (for manual chart verification)")
        print(f"  Total detections: {len(consol_log)} | Unique zones: {len(unique)}")
        print(f"{'='*120}")
        print(f"{'Symbol':<15} {'Consol Start':<14} {'Check Date':<14} {'Days':<6} {'Range Low':<12} {'Range High':<12} {'Range%':<8} {'Status':<20} {'Breakout':<10} {'VolRatio':<10}")
        print("-" * 120)
        for e in unique[:50]:  # Show top 50
            bo_price = e.get('breakout_price', '-')
            vol_ratio = e.get('vol_ratio', '-')
            print(f"{e['symbol']:<15} {e['consol_start']:<14} {e['date']:<14} {e['days']:<6} {e['range_low']:<12} {e['range_high']:<12} {e['range_pct']:<8} {e['status']:<20} {bo_price:<10} {vol_ratio:<10}")

        return unique
    else:
        print("\n  NO consolidation zones detected in any portfolio stock.")
        return []


# =============================================================================
# Generate Markdown Report
# =============================================================================
def generate_report(baseline, r1, r2, r3, total_time):
    """Write comprehensive optimization report to docs/."""
    all_valid = [r for r in (r1 + r2 + r3) if r.get('status') == 'ok']
    by_cagr = sorted(all_valid, key=lambda x: x['cagr'], reverse=True)
    by_sharpe = sorted(all_valid, key=lambda x: x['sharpe'], reverse=True)
    by_calmar = sorted(all_valid, key=lambda x: x['calmar'], reverse=True)
    decent = [r for r in all_valid if r['cagr'] > 10]
    by_dd = sorted(decent, key=lambda x: x['max_drawdown']) if decent else []

    def cfg_table(config):
        lines = []
        for k, v in config.items():
            lines.append(f"| `{k}` | `{v}` |")
        return "\n".join(lines)

    def exit_table(r):
        reason_labels = {
            'hard_stop_loss': 'Hard Stop',
            'ath_drawdown_rebalance': 'ATH Drawdown',
            'fundamental_3q_decline': 'Fundamental 3Q',
            'rebalance_replaced': 'Rebalance',
            'fundamental_2y_decline': 'Fundamental 2Y',
            'manual': 'Manual',
        }
        lines = []
        for reason, data in r.get('exit_reason_pnl', {}).items():
            pnl = data.get('total_pnl', 0)
            sign = '+' if pnl >= 0 else ''
            name = reason_labels.get(reason, reason)
            lines.append(f"| {name} | {data.get('count',0)} | {sign}{pnl:,.0f} | {data.get('avg_return_pct',0):+.1f}% | {data.get('win_rate',0):.0f}% |")
        return "\n".join(lines) if lines else "| -- | -- | -- | -- | -- |"

    md = f"""# MQ Strategy Optimization Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Total runs:** {len(r1) + len(r2) + len(r3)} ({len(all_valid)} successful)
**Duration:** {total_time/60:.1f} minutes
**Workers:** {NUM_WORKERS} parallel processes

---

## Baseline (Current Defaults)

| Metric | Value |
|--------|-------|
| CAGR | {baseline.get('cagr', 0):.1f}% |
| Sharpe | {baseline.get('sharpe', 0):.2f} |
| Sortino | {baseline.get('sortino', 0):.2f} |
| Max Drawdown | {baseline.get('max_drawdown', 0):.1f}% |
| Calmar | {baseline.get('calmar', 0):.2f} |
| Win Rate | {baseline.get('win_rate', 0):.1f}% |
| Total Trades | {baseline.get('total_trades', 0)} |
| Total Topups | {baseline.get('total_topups', 0)} |
| Final Value | {baseline.get('final_value', 0):,.0f} |

---

## Top 10 Configurations by CAGR

| # | Config | CAGR | Sharpe | Sortino | Max DD | Calmar | Win Rate | Trades |
|---|--------|------|--------|---------|--------|--------|----------|--------|
"""
    for i, r in enumerate(by_cagr[:10]):
        md += f"| {i+1} | {r['label']} | {r['cagr']:.1f}% | {r['sharpe']:.2f} | {r['sortino']:.2f} | {r['max_drawdown']:.1f}% | {r['calmar']:.2f} | {r['win_rate']:.1f}% | {r['total_trades']} |\n"

    md += f"""
## Top 5 by Sharpe Ratio (Risk-Adjusted)

| # | Config | Sharpe | CAGR | Max DD | Calmar |
|---|--------|--------|------|--------|--------|
"""
    for i, r in enumerate(by_sharpe[:5]):
        md += f"| {i+1} | {r['label']} | {r['sharpe']:.2f} | {r['cagr']:.1f}% | {r['max_drawdown']:.1f}% | {r['calmar']:.2f} |\n"

    md += f"""
## Top 5 by Calmar Ratio (Return/Risk)

| # | Config | Calmar | CAGR | Max DD | Sharpe |
|---|--------|--------|------|--------|--------|
"""
    for i, r in enumerate(by_calmar[:5]):
        md += f"| {i+1} | {r['label']} | {r['calmar']:.2f} | {r['cagr']:.1f}% | {r['max_drawdown']:.1f}% | {r['sharpe']:.2f} |\n"

    if by_dd:
        md += f"""
## Lowest Drawdown (CAGR > 10%)

| # | Config | Max DD | CAGR | Sharpe | Calmar |
|---|--------|--------|------|--------|--------|
"""
        for i, r in enumerate(by_dd[:5]):
            md += f"| {i+1} | {r['label']} | {r['max_drawdown']:.1f}% | {r['cagr']:.1f}% | {r['sharpe']:.2f} | {r['calmar']:.2f} |\n"

    # Best overall config details
    best = by_cagr[0] if by_cagr else None
    if best:
        md += f"""
---

## Best Configuration: `{best['label']}`

### Performance
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| CAGR | {best['cagr']:.1f}% | {best['cagr'] - baseline.get('cagr', 0):+.1f}% |
| Sharpe | {best['sharpe']:.2f} | {best['sharpe'] - baseline.get('sharpe', 0):+.2f} |
| Sortino | {best['sortino']:.2f} | {best['sortino'] - baseline.get('sortino', 0):+.2f} |
| Max Drawdown | {best['max_drawdown']:.1f}% | {best['max_drawdown'] - baseline.get('max_drawdown', 0):+.1f}% |
| Calmar | {best['calmar']:.2f} | {best['calmar'] - baseline.get('calmar', 0):+.2f} |
| Win Rate | {best['win_rate']:.1f}% | {best['win_rate'] - baseline.get('win_rate', 0):+.1f}% |
| Final Value | {best['final_value']:,.0f} | |

### Parameters
| Parameter | Value |
|-----------|-------|
{cfg_table(best['config'])}

### Exit Reason P&L
| Reason | Count | Total P&L | Avg Return | Win Rate |
|--------|-------|-----------|------------|----------|
{exit_table(best)}
"""

    # Best risk-adjusted config
    best_ra = by_sharpe[0] if by_sharpe else None
    if best and best_ra and best_ra['label'] != best['label']:
        md += f"""
---

## Best Risk-Adjusted: `{best_ra['label']}`

### Performance
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| CAGR | {best_ra['cagr']:.1f}% | {best_ra['cagr'] - baseline.get('cagr', 0):+.1f}% |
| Sharpe | {best_ra['sharpe']:.2f} | {best_ra['sharpe'] - baseline.get('sharpe', 0):+.2f} |
| Max Drawdown | {best_ra['max_drawdown']:.1f}% | {best_ra['max_drawdown'] - baseline.get('max_drawdown', 0):+.1f}% |
| Calmar | {best_ra['calmar']:.2f} | {best_ra['calmar'] - baseline.get('calmar', 0):+.2f} |

### Parameters
| Parameter | Value |
|-----------|-------|
{cfg_table(best_ra['config'])}

### Exit Reason P&L
| Reason | Count | Total P&L | Avg Return | Win Rate |
|--------|-------|-----------|------------|----------|
{exit_table(best_ra)}
"""

    # Key findings - sensitivity analysis
    md += """
---

## Key Findings

### Parameter Sensitivity Analysis
"""
    if all_valid:
        for param_name, param_key, fmt_fn in [
            ('Hard Stop Loss', 'hard_stop_loss', lambda v: f"{int(v*100)}%"),
            ('ATH Drawdown Exit', 'rebalance_ath_drawdown', lambda v: f"{int(v*100)}%"),
            ('Portfolio Size', 'portfolio_size', lambda v: str(int(v))),
            ('Equity Allocation', 'equity_allocation_pct', lambda v: f"{int(v*100)}%"),
        ]:
            groups = {}
            for r in all_valid:
                val = r['config'].get(param_key)
                if val is not None:
                    groups.setdefault(val, []).append(r)
            if len(groups) > 1:
                md += f"\n**{param_name} Impact:**\n\n| Value | Avg CAGR | Best CAGR | Avg Sharpe | Avg MaxDD | Count |\n|-------|----------|-----------|------------|-----------|-------|\n"
                for val in sorted(groups.keys()):
                    runs = groups[val]
                    cagrs = [r['cagr'] for r in runs]
                    sharpes = [r['sharpe'] for r in runs]
                    dds = [r['max_drawdown'] for r in runs]
                    md += f"| {fmt_fn(val)} | {sum(cagrs)/len(cagrs):.1f}% | {max(cagrs):.1f}% | {sum(sharpes)/len(sharpes):.2f} | {sum(dds)/len(dds):.1f}% | {len(runs)} |\n"

        # Topup impact (R2+R3 only)
        r2r3 = [r for r in all_valid if r['label'].startswith('R2_') or r['label'].startswith('R3_')]
        if r2r3:
            tu_groups = {}
            for r in r2r3:
                tu = r['config'].get('topup_pct_of_initial', 0.2)
                tu_groups.setdefault(tu, []).append(r)
            if len(tu_groups) > 1:
                md += f"\n**Topup % Impact (R2+R3):**\n\n| Topup | Avg CAGR | Best CAGR | Avg Topups | Count |\n|-------|----------|-----------|------------|-------|\n"
                for tu in sorted(tu_groups.keys()):
                    runs = tu_groups[tu]
                    cagrs = [r['cagr'] for r in runs]
                    topups = [r.get('total_topups', 0) for r in runs]
                    lbl = "None" if tu == 0 else f"{int(tu*100)}%"
                    md += f"| {lbl} | {sum(cagrs)/len(cagrs):.1f}% | {max(cagrs):.1f}% | {sum(topups)/len(topups):.0f} | {len(runs)} |\n"

            # Rebalance frequency
            reb_groups = {}
            for r in r2r3:
                reb = str(r['config'].get('rebalance_months', [1, 7]))
                reb_groups.setdefault(reb, []).append(r)
            if len(reb_groups) > 1:
                md += f"\n**Rebalance Frequency Impact (R2+R3):**\n\n| Months | Avg CAGR | Best CAGR | Count |\n|--------|----------|-----------|-------|\n"
                for reb in sorted(reb_groups.keys()):
                    runs = reb_groups[reb]
                    cagrs = [r['cagr'] for r in runs]
                    md += f"| {reb} | {sum(cagrs)/len(cagrs):.1f}% | {max(cagrs):.1f}% | {len(runs)} |\n"

    # Recommendations
    if by_cagr:
        best = by_cagr[0]
        best_ra = by_sharpe[0]
        md += f"""
---

## Recommendations

### For Maximum Growth
Use config `{best['label']}`:
- CAGR: **{best['cagr']:.1f}%**, Sharpe: {best['sharpe']:.2f}, MaxDD: {best['max_drawdown']:.1f}%
- Risk: Higher drawdown tolerance needed

### For Risk-Adjusted Returns
Use config `{best_ra['label']}`:
- Sharpe: **{best_ra['sharpe']:.2f}**, CAGR: {best_ra['cagr']:.1f}%, MaxDD: {best_ra['max_drawdown']:.1f}%
- Better risk/reward tradeoff
"""
        # Balanced recommendation
        balanced = sorted(all_valid, key=lambda x: x['cagr'] * 0.4 + x['sharpe'] * 20 + x['calmar'] * 5 - abs(x['max_drawdown']) * 0.3, reverse=True)
        if balanced:
            bal = balanced[0]
            md += f"""
### Balanced (Recommended)
Config `{bal['label']}`:
- CAGR: **{bal['cagr']:.1f}%**, Sharpe: **{bal['sharpe']:.2f}**, MaxDD: **{bal['max_drawdown']:.1f}%**, Calmar: **{bal['calmar']:.2f}**

**Suggested changes from current defaults:**
"""
            default_cfg = {
                'hard_stop_loss': 0.30, 'rebalance_ath_drawdown': 0.20,
                'portfolio_size': 30, 'equity_allocation_pct': 0.80,
                'ath_proximity_threshold': 0.10, 'topup_pct_of_initial': 0.20,
                'consolidation_days': 20, 'consolidation_range_pct': 0.05,
                'breakout_volume_multiplier': 1.5, 'topup_cooldown_days': 5,
                'rebalance_months': [1, 7],
            }
            changes = []
            for k, default_v in default_cfg.items():
                new_v = bal['config'].get(k, default_v)
                if new_v != default_v:
                    changes.append(f"- `{k}`: {default_v} -> **{new_v}**")
            if changes:
                md += "\n".join(changes)
            else:
                md += "- No changes needed (defaults are optimal!)"

    md += f"\n\n---\n\n*Report generated by MQ Optimization Agent on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n"

    REPORT_FILE.write_text(md, encoding='utf-8')
    logger.info(f"Report saved to {REPORT_FILE}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--r4-only', action='store_true',
                        help='Run only Round 4 (trailing SL + daily ATH) + consolidation diagnostic')
    parser.add_argument('--full', action='store_true',
                        help='Run all rounds (R1-R4)')
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 70)
    print("  MQ STRATEGY PARAMETER OPTIMIZER")
    print(f"  Workers: {NUM_WORKERS} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Run baseline first (single process, with logging)
    logger.info("Running BASELINE...")
    logging.getLogger('services.mq_backtest_engine').setLevel(logging.INFO)
    logging.getLogger('services.mq_portfolio').setLevel(logging.WARNING)
    logging.getLogger('services.nifty500_universe').setLevel(logging.INFO)

    universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig())
    engine = MQBacktestEngine(MQBacktestConfig(), universe, price_data)
    result = engine.run(quality_scores={})
    baseline = {
        'label': 'BASELINE', 'status': 'ok',
        'cagr': result.cagr, 'sharpe': result.sharpe_ratio,
        'sortino': result.sortino_ratio, 'max_drawdown': result.max_drawdown,
        'calmar': result.calmar_ratio, 'total_return_pct': result.total_return_pct,
        'final_value': result.final_value, 'total_trades': result.total_trades,
        'win_rate': result.win_rate, 'total_topups': result.total_topups,
    }
    print(f"  BASELINE: CAGR={baseline['cagr']:.1f}%, Sharpe={baseline['sharpe']:.2f}, MaxDD={baseline['max_drawdown']:.1f}%, Calmar={baseline['calmar']:.2f}")
    print()

    r1, r2, r3, r4 = [], [], [], []

    if not args.r4_only:
        # Create worker pool
        logger.info(f"Spawning {NUM_WORKERS} worker processes...")
        pool = Pool(processes=NUM_WORKERS, initializer=_worker_init)

        try:
            # Round 1
            t0 = time.time()
            r1 = round_1(pool)
            logger.info(f"  Round 1 done in {(time.time()-t0)/60:.1f}m")

            # Round 2
            t0 = time.time()
            r2 = round_2(r1, pool)
            logger.info(f"  Round 2 done in {(time.time()-t0)/60:.1f}m")

            # Round 3
            t0 = time.time()
            r3 = round_3(r2, pool)
            logger.info(f"  Round 3 done in {(time.time()-t0)/60:.1f}m")

        finally:
            pool.close()
            pool.join()

    # Round 4: Trailing SL + Daily ATH (always runs)
    logger.info(f"Spawning {NUM_WORKERS} worker processes for R4...")
    pool4 = Pool(processes=NUM_WORKERS, initializer=_worker_init)
    try:
        t0 = time.time()
        r4 = round_4(pool4)
        logger.info(f"  Round 4 done in {(time.time()-t0)/60:.1f}m")
    finally:
        pool4.close()
        pool4.join()

    # Consolidation diagnostic (single process, verbose)
    logger.info("Running consolidation diagnostic...")
    consol_zones = run_consolidation_diagnostic(universe, price_data)

    # Save consolidation zones
    if consol_zones:
        consol_file = Path('backtest_data') / 'consolidation_zones.json'
        consol_file.write_text(json.dumps(consol_zones, indent=2, default=str), encoding='utf-8')
        logger.info(f"  Consolidation zones saved to {consol_file}")

    # Generate report
    total_time = time.time() - start_time
    generate_report(baseline, r1, r2, r3, total_time)

    # Final summary
    all_ok = [r for r in (r1 + r2 + r3 + r4) if r.get('status') == 'ok']
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  Runs: {len(r1)+len(r2)+len(r3)+len(r4)} | Success: {len(all_ok)} | Time: {total_time/60:.1f} min")
    if all_ok:
        best = sorted(all_ok, key=lambda x: x['cagr'], reverse=True)[0]
        print(f"  BEST: {best['label']} -> CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.2f}, MaxDD={best['max_drawdown']:.1f}%")

    # R4 specific summary
    r4_ok = [r for r in r4 if r.get('status') == 'ok']
    if r4_ok:
        r4_best = sorted(r4_ok, key=lambda x: x['cagr'], reverse=True)[0]
        print(f"  R4 BEST: {r4_best['label']} -> CAGR={r4_best['cagr']:.1f}%, Sharpe={r4_best['sharpe']:.2f}, MaxDD={r4_best['max_drawdown']:.1f}%")

    if consol_zones:
        print(f"  Consolidation zones found: {len(consol_zones)} (saved to backtest_data/consolidation_zones.json)")
    else:
        print(f"  Consolidation zones found: 0 (feature not triggering)")

    print(f"  Report: {REPORT_FILE}")
    print(f"  Data:   {RESULTS_FILE}")
    print(f"{'='*70}")
