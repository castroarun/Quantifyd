"""
Technical Indicator Optimization Runner
========================================

Runs optimization rounds R5-R9 testing various technical indicator combinations.

R5: EMA Crossovers - entry/exit based on EMA combinations
R6: RSI + Stochastics - momentum filters and overbought/oversold
R7: Ichimoku Cloud - above/below cloud entry/exit
R8: Supertrend - trend following entry/exit
R9: Top-down + Combined - weekly filters + best indicator combinations

Each round runs ~50-200 combinations in parallel.
Status updates every 5 minutes to observations MD file.
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from itertools import product
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'backtest_data'
DOCS_DIR = BASE_DIR / 'docs'
RESULTS_FILE = DATA_DIR / 'technical_optimization_results.json'
OBSERVATIONS_FILE = DOCS_DIR / 'TECHNICAL-OPTIMIZATION-OBSERVATIONS.md'


# =============================================================================
# Result Tracking
# =============================================================================

@dataclass
class OptimizationRun:
    """Single optimization run result."""
    round: str
    config_name: str
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    total_trades: int
    win_rate: float
    final_value: float
    config_params: Dict


def save_results(results: List[OptimizationRun], append: bool = True):
    """Save results to JSON file."""
    data = []
    if append and RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)

    for r in results:
        data.append(asdict(r))

    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def load_results() -> List[Dict]:
    """Load existing results."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []


# =============================================================================
# Status Updates (every 5 mins)
# =============================================================================

def update_observations(
    round_name: str,
    status: str,
    completed: int,
    total: int,
    best_so_far: Optional[Dict] = None,
    elapsed_mins: float = 0,
    findings: List[str] = None,
):
    """Update the observations MD file with current status."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Load existing content
    if OBSERVATIONS_FILE.exists():
        with open(OBSERVATIONS_FILE, 'r') as f:
            content = f.read()
    else:
        content = """# Technical Indicators Optimization - Live Observations

**Started:** {date}
**Status:** INITIALIZING

---

## Progress Tracker

| Round | Focus | Combos | Status | Best CAGR | Best Calmar | Duration |
|-------|-------|--------|--------|-----------|-------------|----------|

---

## Timeline

---

## Key Findings (Updated Live)

*Will be populated as optimization progresses...*

---

## Best Configurations Found

| Rank | Config | CAGR | MaxDD | Calmar | Sharpe | Round |
|------|--------|------|-------|--------|--------|-------|

---

*Last updated: INITIALIZING*
""".format(date=datetime.now().strftime('%Y-%m-%d'))

    # Update progress tracker row for this round
    pct_done = completed / total * 100 if total > 0 else 0
    best_cagr = f"{best_so_far['cagr']:.1f}%" if best_so_far else "-"
    best_calmar = f"{best_so_far['calmar']:.2f}" if best_so_far else "-"
    duration = f"{elapsed_mins:.0f}m" if elapsed_mins > 0 else "-"

    # Find and update or add the progress row
    lines = content.split('\n')
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"| {round_name} "):
            lines[i] = f"| {round_name} | Various | {total} | {status} ({pct_done:.0f}%) | {best_cagr} | {best_calmar} | {duration} |"
            updated = True
            break

    if not updated:
        # Find the progress tracker table and add row
        for i, line in enumerate(lines):
            if line.startswith("|-------|-------|"):
                lines.insert(i + 1, f"| {round_name} | Various | {total} | {status} ({pct_done:.0f}%) | {best_cagr} | {best_calmar} | {duration} |")
                break

    # Add timeline entry
    timeline_entry = f"\n### Update - {now}\n"
    timeline_entry += f"- **Round:** {round_name}\n"
    timeline_entry += f"- **Status:** {status} - {completed}/{total} ({pct_done:.1f}%)\n"
    if best_so_far:
        timeline_entry += f"- **Best so far:** {best_so_far['config_name']} - CAGR={best_cagr}, Calmar={best_calmar}\n"
    if findings:
        timeline_entry += f"- **Observations:**\n"
        for f in findings:
            timeline_entry += f"  - {f}\n"

    # Insert timeline entry
    for i, line in enumerate(lines):
        if line.strip() == "## Timeline":
            lines.insert(i + 2, timeline_entry)
            break

    # Update last updated timestamp
    for i, line in enumerate(lines):
        if line.startswith("*Last updated:"):
            lines[i] = f"*Last updated: {now}*"
            break

    content = '\n'.join(lines)

    with open(OBSERVATIONS_FILE, 'w') as f:
        f.write(content)


# =============================================================================
# Worker Function (runs in separate process)
# =============================================================================

def _run_task(config_dict: Dict) -> Dict:
    """Run a single backtest with given config (in subprocess)."""
    try:
        # Import inside subprocess to avoid pickling issues
        from services.mq_backtest_engine import MQBacktestEngine
        from services.mq_portfolio import MQBacktestConfig

        # Strip metadata keys before creating config
        config_name = config_dict.pop('_config_name', 'unknown')
        round_name = config_dict.pop('_round', 'unknown')
        clean_config = {k: v for k, v in config_dict.items() if not k.startswith('_')}
        config = MQBacktestConfig(**clean_config)
        engine = MQBacktestEngine(config)
        result = engine.run()

        return {
            'success': True,
            'config_name': config_name,
            'round': round_name,
            'cagr': result.cagr,
            'sharpe': result.sharpe_ratio,
            'sortino': result.sortino_ratio,
            'max_drawdown': result.max_drawdown,
            'calmar': result.calmar_ratio,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'final_value': result.final_value,
            'config_params': clean_config,
        }
    except Exception as e:
        # If we haven't extracted metadata yet, try to get it
        cfg_name = config_dict.get('_config_name', 'unknown') if '_config_name' in config_dict else 'unknown'
        rnd_name = config_dict.get('_round', 'unknown') if '_round' in config_dict else 'unknown'
        return {
            'success': False,
            'config_name': cfg_name,
            'round': rnd_name,
            'error': str(e),
        }


# =============================================================================
# Round Definitions
# =============================================================================

def generate_r5_configs() -> List[Dict]:
    """R5: EMA Crossovers - 120 combinations."""
    base = {
        'start_date': '2023-01-01',
        'end_date': '2025-12-31',
        'use_technical_filter': True,
        '_round': 'R5',
    }

    configs = []

    # EMA entry variations
    ema_combos = [
        (9, 21), (12, 26), (5, 20), (10, 50), (20, 50), (50, 200)
    ]
    entry_types = ['crossover', 'price_above', 'both']

    # With and without exit
    for (fast, slow), entry_type in product(ema_combos, entry_types):
        for use_exit in [False, True]:
            for exit_type in (['crossover', 'price_below'] if use_exit else [None]):
                cfg = base.copy()
                cfg.update({
                    'use_ema_entry': True,
                    'ema_fast': fast,
                    'ema_slow': slow,
                    'ema_entry_type': entry_type,
                    'use_ema_exit': use_exit,
                })
                if exit_type:
                    cfg['ema_exit_type'] = exit_type

                name = f"EMA{fast}_{slow}_{entry_type[:3]}"
                if use_exit:
                    name += f"_exit{exit_type[:3]}"
                cfg['_config_name'] = name
                configs.append(cfg)

    return configs


def generate_r6_configs() -> List[Dict]:
    """R6: RSI + Stochastics - 100 combinations."""
    base = {
        'start_date': '2023-01-01',
        'end_date': '2025-12-31',
        'use_technical_filter': True,
        '_round': 'R6',
    }

    configs = []

    # RSI variations
    rsi_periods = [7, 14, 21]
    rsi_ranges = [(30, 70), (40, 70), (40, 80), (50, 75)]

    for period, (rsi_min, rsi_max) in product(rsi_periods, rsi_ranges):
        for use_exit in [False, True]:
            cfg = base.copy()
            cfg.update({
                'use_rsi_filter': True,
                'rsi_period': period,
                'rsi_min_entry': rsi_min,
                'rsi_max_entry': rsi_max,
                'use_rsi_exit': use_exit,
                'rsi_exit_overbought': 80 if use_exit else 100,
            })
            name = f"RSI{period}_{rsi_min}_{rsi_max}"
            if use_exit:
                name += "_exit80"
            cfg['_config_name'] = name
            configs.append(cfg)

    # Stochastics variations
    stoch_periods = [(14, 3), (5, 3), (21, 5)]
    stoch_levels = [(80, 20), (70, 30)]

    for (k, d), (ob, os) in product(stoch_periods, stoch_levels):
        cfg = base.copy()
        cfg.update({
            'use_stoch_filter': True,
            'stoch_k': k,
            'stoch_d': d,
            'stoch_overbought': ob,
            'stoch_oversold': os,
        })
        cfg['_config_name'] = f"STOCH{k}_{d}_{ob}_{os}"
        configs.append(cfg)

    # Combined RSI + Stochastics
    for (period, (rsi_min, rsi_max)), ((k, d), (ob, os)) in product(
        list(zip(rsi_periods, rsi_ranges[:3])),
        list(zip(stoch_periods[:2], stoch_levels))
    ):
        cfg = base.copy()
        cfg.update({
            'use_rsi_filter': True,
            'rsi_period': period,
            'rsi_min_entry': rsi_min,
            'rsi_max_entry': rsi_max,
            'use_stoch_filter': True,
            'stoch_k': k,
            'stoch_d': d,
            'stoch_overbought': ob,
            'stoch_oversold': os,
        })
        cfg['_config_name'] = f"RSI{period}_STOCH{k}"
        configs.append(cfg)

    return configs


def generate_r7_configs() -> List[Dict]:
    """R7: Ichimoku Cloud - 48 combinations."""
    base = {
        'start_date': '2023-01-01',
        'end_date': '2025-12-31',
        'use_technical_filter': True,
        '_round': 'R7',
    }

    configs = []

    tenkan_vals = [9, 20]
    kijun_vals = [26, 52]
    require_cloud = [True, False]
    exit_cloud = [True, False]

    for tenkan, kijun, req_cloud, exit_c in product(tenkan_vals, kijun_vals, require_cloud, exit_cloud):
        cfg = base.copy()
        cfg.update({
            'use_ichimoku': True,
            'ichimoku_tenkan': tenkan,
            'ichimoku_kijun': kijun,
            'require_above_cloud': req_cloud,
            'exit_below_cloud': exit_c,
        })
        name = f"ICHI{tenkan}_{kijun}"
        if req_cloud:
            name += "_aboveCloud"
        if exit_c:
            name += "_exitCloud"
        cfg['_config_name'] = name
        configs.append(cfg)

    # Ichimoku + EMA combination
    for tenkan, (fast, slow) in product([9, 20], [(9, 21), (20, 50)]):
        cfg = base.copy()
        cfg.update({
            'use_ichimoku': True,
            'ichimoku_tenkan': tenkan,
            'ichimoku_kijun': 26,
            'require_above_cloud': True,
            'use_ema_entry': True,
            'ema_fast': fast,
            'ema_slow': slow,
            'ema_entry_type': 'crossover',
        })
        cfg['_config_name'] = f"ICHI{tenkan}_EMA{fast}_{slow}"
        configs.append(cfg)

    return configs


def generate_r8_configs() -> List[Dict]:
    """R8: Supertrend - 72 combinations."""
    base = {
        'start_date': '2023-01-01',
        'end_date': '2025-12-31',
        'use_technical_filter': True,
        '_round': 'R8',
    }

    configs = []

    atr_periods = [7, 10, 14]
    multipliers = [2.0, 2.5, 3.0, 3.5]
    entry_bullish = [True]
    exit_flip = [True, False]

    for atr, mult, entry_b, exit_f in product(atr_periods, multipliers, entry_bullish, exit_flip):
        cfg = base.copy()
        cfg.update({
            'use_supertrend': True,
            'supertrend_atr': atr,
            'supertrend_mult': mult,
            'supertrend_entry_bullish': entry_b,
            'supertrend_exit_flip': exit_f,
        })
        name = f"ST{atr}_{mult}"
        if exit_f:
            name += "_exitFlip"
        cfg['_config_name'] = name
        configs.append(cfg)

    # Supertrend + MACD
    for atr, mult in product([10, 14], [2.5, 3.0]):
        cfg = base.copy()
        cfg.update({
            'use_supertrend': True,
            'supertrend_atr': atr,
            'supertrend_mult': mult,
            'supertrend_entry_bullish': True,
            'use_macd': True,
            'require_macd_positive': True,
            'require_macd_above_signal': True,
        })
        cfg['_config_name'] = f"ST{atr}_{mult}_MACD"
        configs.append(cfg)

    # Supertrend + ADX
    for atr, mult, adx_min in product([10, 14], [2.5, 3.0], [20, 25]):
        cfg = base.copy()
        cfg.update({
            'use_supertrend': True,
            'supertrend_atr': atr,
            'supertrend_mult': mult,
            'supertrend_entry_bullish': True,
            'use_adx': True,
            'adx_min_trend': adx_min,
            'require_plus_di_above': True,
        })
        cfg['_config_name'] = f"ST{atr}_{mult}_ADX{adx_min}"
        configs.append(cfg)

    return configs


def generate_r9_configs() -> List[Dict]:
    """R9: Top-down + Combined Best - 80 combinations."""
    base = {
        'start_date': '2023-01-01',
        'end_date': '2025-12-31',
        'use_technical_filter': True,
        '_round': 'R9',
    }

    configs = []

    # Weekly filter with various indicators
    weekly_ema_periods = [10, 20, 50]

    # Weekly + EMA
    for weekly_ema, (fast, slow) in product(weekly_ema_periods, [(9, 21), (20, 50)]):
        cfg = base.copy()
        cfg.update({
            'use_weekly_filter': True,
            'weekly_ema_period': weekly_ema,
            'require_weekly_above_ema': True,
            'use_ema_entry': True,
            'ema_fast': fast,
            'ema_slow': slow,
            'ema_entry_type': 'crossover',
        })
        cfg['_config_name'] = f"W{weekly_ema}_EMA{fast}_{slow}"
        configs.append(cfg)

    # Weekly + Supertrend
    for weekly_ema, (atr, mult) in product(weekly_ema_periods, [(10, 3.0), (14, 2.5)]):
        cfg = base.copy()
        cfg.update({
            'use_weekly_filter': True,
            'weekly_ema_period': weekly_ema,
            'require_weekly_above_ema': True,
            'use_supertrend': True,
            'supertrend_atr': atr,
            'supertrend_mult': mult,
            'supertrend_entry_bullish': True,
        })
        cfg['_config_name'] = f"W{weekly_ema}_ST{atr}"
        configs.append(cfg)

    # Multi-indicator combinations (best from each category)
    # EMA + RSI
    for (fast, slow), period in product([(9, 21), (20, 50)], [14]):
        cfg = base.copy()
        cfg.update({
            'use_ema_entry': True,
            'ema_fast': fast,
            'ema_slow': slow,
            'ema_entry_type': 'crossover',
            'use_rsi_filter': True,
            'rsi_period': period,
            'rsi_min_entry': 40,
            'rsi_max_entry': 70,
        })
        cfg['_config_name'] = f"EMA{fast}_{slow}_RSI{period}"
        configs.append(cfg)

    # EMA + MACD
    for fast, slow in [(9, 21), (12, 26)]:
        cfg = base.copy()
        cfg.update({
            'use_ema_entry': True,
            'ema_fast': fast,
            'ema_slow': slow,
            'ema_entry_type': 'crossover',
            'use_macd': True,
            'require_macd_positive': True,
            'require_macd_above_signal': True,
        })
        cfg['_config_name'] = f"EMA{fast}_{slow}_MACD"
        configs.append(cfg)

    # EMA + Bollinger squeeze
    for fast, slow in [(9, 21), (20, 50)]:
        cfg = base.copy()
        cfg.update({
            'use_ema_entry': True,
            'ema_fast': fast,
            'ema_slow': slow,
            'ema_entry_type': 'crossover',
            'use_bollinger': True,
            'bb_entry_squeeze': True,
        })
        cfg['_config_name'] = f"EMA{fast}_{slow}_BBsqueeze"
        configs.append(cfg)

    # Triple confirmation: Weekly + EMA + RSI
    for weekly_ema in [20]:
        for (fast, slow), rsi_period in product([(9, 21), (20, 50)], [14]):
            cfg = base.copy()
            cfg.update({
                'use_weekly_filter': True,
                'weekly_ema_period': weekly_ema,
                'require_weekly_above_ema': True,
                'use_ema_entry': True,
                'ema_fast': fast,
                'ema_slow': slow,
                'ema_entry_type': 'crossover',
                'use_rsi_filter': True,
                'rsi_period': rsi_period,
                'rsi_min_entry': 40,
                'rsi_max_entry': 70,
            })
            cfg['_config_name'] = f"W{weekly_ema}_EMA{fast}_{slow}_RSI{rsi_period}"
            configs.append(cfg)

    # Full system: Weekly + EMA + RSI + Supertrend
    for weekly_ema in [20]:
        cfg = base.copy()
        cfg.update({
            'use_weekly_filter': True,
            'weekly_ema_period': weekly_ema,
            'require_weekly_above_ema': True,
            'use_ema_entry': True,
            'ema_fast': 9,
            'ema_slow': 21,
            'ema_entry_type': 'crossover',
            'use_rsi_filter': True,
            'rsi_period': 14,
            'rsi_min_entry': 40,
            'rsi_max_entry': 70,
            'use_supertrend': True,
            'supertrend_atr': 10,
            'supertrend_mult': 3.0,
            'supertrend_entry_bullish': True,
        })
        cfg['_config_name'] = f"FULL_W{weekly_ema}_EMA_RSI_ST"
        configs.append(cfg)

    # Baseline comparison (no technical filter)
    cfg = base.copy()
    cfg['use_technical_filter'] = False
    cfg['_config_name'] = 'BASELINE_NO_TECH'
    configs.append(cfg)

    return configs


# =============================================================================
# Main Runner
# =============================================================================

def run_round(
    round_name: str,
    configs: List[Dict],
    workers: int = 3,
    update_interval: int = 300,  # 5 minutes
) -> List[OptimizationRun]:
    """Run a full round of optimization."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {round_name}: {len(configs)} configurations")
    logger.info(f"Workers: {workers} | Update interval: {update_interval}s")
    logger.info(f"{'='*60}\n")

    results = []
    best_so_far = None
    start_time = time.time()
    last_update = start_time

    # Initial status update
    update_observations(round_name, 'RUNNING', 0, len(configs))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_task, cfg): cfg for cfg in configs}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            result = future.result()

            if result['success']:
                run = OptimizationRun(
                    round=result['round'],
                    config_name=result['config_name'],
                    cagr=result['cagr'],
                    sharpe=result['sharpe'],
                    sortino=result['sortino'],
                    max_drawdown=result['max_drawdown'],
                    calmar=result['calmar'],
                    total_trades=result['total_trades'],
                    win_rate=result['win_rate'],
                    final_value=result['final_value'],
                    config_params=result['config_params'],
                )
                results.append(run)

                # Update best
                if best_so_far is None or result['calmar'] > best_so_far['calmar']:
                    best_so_far = result

                logger.info(
                    f"[{completed}/{len(configs)}] {result['config_name']}: "
                    f"CAGR={result['cagr']:.1f}% | Calmar={result['calmar']:.2f} | "
                    f"MaxDD={result['max_drawdown']:.1f}%"
                )
            else:
                logger.error(f"[{completed}/{len(configs)}] {result['config_name']}: FAILED - {result.get('error', 'Unknown')}")

            # Periodic status update
            now = time.time()
            if now - last_update >= update_interval:
                elapsed_mins = (now - start_time) / 60
                update_observations(
                    round_name, 'RUNNING', completed, len(configs),
                    best_so_far=best_so_far, elapsed_mins=elapsed_mins
                )
                last_update = now

    # Final update
    elapsed_mins = (time.time() - start_time) / 60
    update_observations(
        round_name, 'COMPLETED', len(configs), len(configs),
        best_so_far=best_so_far, elapsed_mins=elapsed_mins,
        findings=[
            f"Total runs: {len(results)}",
            f"Best CAGR: {best_so_far['cagr']:.1f}%" if best_so_far else "No successful runs",
            f"Best Calmar: {best_so_far['calmar']:.2f}" if best_so_far else "",
        ]
    )

    # Sort by Calmar (risk-adjusted)
    results.sort(key=lambda r: r.calmar, reverse=True)

    logger.info(f"\n{round_name} completed in {elapsed_mins:.1f} minutes")
    logger.info(f"Best by Calmar: {results[0].config_name} - CAGR={results[0].cagr:.1f}%, Calmar={results[0].calmar:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Technical Indicator Optimization')
    parser.add_argument('--round', type=str, choices=['R5', 'R6', 'R7', 'R8', 'R9', 'all'],
                        default='all', help='Which round to run')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')
    parser.add_argument('--update-interval', type=int, default=300, help='Status update interval in seconds')
    args = parser.parse_args()

    logger.info("Technical Indicator Optimization")
    logger.info(f"Round: {args.round} | Workers: {args.workers}")

    all_results = []

    round_generators = {
        'R5': generate_r5_configs,
        'R6': generate_r6_configs,
        'R7': generate_r7_configs,
        'R8': generate_r8_configs,
        'R9': generate_r9_configs,
    }

    rounds_to_run = list(round_generators.keys()) if args.round == 'all' else [args.round]

    for round_name in rounds_to_run:
        configs = round_generators[round_name]()
        results = run_round(round_name, configs, args.workers, args.update_interval)
        all_results.extend(results)
        save_results(results, append=True)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*60)

    # Top 10 overall by Calmar
    all_results.sort(key=lambda r: r.calmar, reverse=True)
    logger.info("\nTop 10 Configurations (by Calmar):")
    for i, r in enumerate(all_results[:10], 1):
        logger.info(f"  {i}. [{r.round}] {r.config_name}: CAGR={r.cagr:.1f}%, MaxDD={r.max_drawdown:.1f}%, Calmar={r.calmar:.2f}")


if __name__ == '__main__':
    main()
