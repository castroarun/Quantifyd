"""ORB universe optimization — find top F&O stocks for live ORB expansion.

Runs the live ORB rules (V0 baseline = OR15 + VWAP + RSI60 + CPR-dir + CPR-w 0.65
+ gap filter + 1.5R target) on ALL ~80 F&O stocks individually for 2024-2026,
ranks by per-stock Sharpe / PF / WR. Identifies top 25-30 candidates to add
to the live ORB universe.

Train/test split:
  Train: 2024-03-18 to 2025-03-15 (~1 year)
  Test:  2025-03-16 to 2026-03-12 (~1 year)

Pass criteria for inclusion:
  Train PF >= 1.0 AND Test PF >= 1.0 AND Test Sharpe >= 0.5
  (i.e., works in both periods, not just one — avoid hindsight bias)
"""
from __future__ import annotations

import csv
import logging
import sqlite3
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.disable(logging.WARNING); warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from services.data_manager import FNO_LOT_SIZES
from services.orb_backtest_engine import ORBConfig, ORBBacktestEngine

OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(parents=True, exist_ok=True)

# Live ORB universe (15 stocks) — exclude these from candidates (already in)
LIVE_ORB_UNIVERSE = {
    'ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BPCL', 'M&M', 'BAJFINANCE',
    'TRENT', 'HAL', 'IRCTC', 'GRASIM', 'GODREJPROP', 'RELIANCE',
    'AXISBANK', 'APOLLOHOSP',
}

# Periods
TRAIN_START = '2024-03-18'
TRAIN_END   = '2025-03-15'
TEST_START  = '2025-03-16'
TEST_END    = '2026-03-12'

# Pass criteria
TRAIN_MIN_PF      = 1.0
TEST_MIN_PF       = 1.0
TEST_MIN_SHARPE   = 0.5
MIN_TRADES        = 20    # need at least 20 trades for stat power


def make_orb_config() -> ORBConfig:
    """Live ORB V0 baseline config — matches config.py ORB_DEFAULTS."""
    return ORBConfig(
        or_minutes=15,
        last_entry_time='14:00',
        eod_exit_time='15:18',
        max_trades_per_day=1,
        allow_longs=True,
        allow_shorts=True,
        sl_type='or_opposite',
        target_type='r_multiple',
        r_multiple=1.5,
        use_vwap_filter=True,
        use_rsi_filter=True,
        rsi_long_threshold=60.0,
        rsi_short_threshold=40.0,
        use_cpr_dir_filter=True,
        use_cpr_width_filter=True,
        cpr_width_threshold_pct=0.65,
    )


def filter_data_by_period(data: dict, start_date: str, end_date: str) -> dict:
    """Trim 5-min and daily data to a period."""
    out = {}
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    for tf in ('5min', 'day'):
        df = data[tf]
        mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
        out[tf] = df[mask].reset_index(drop=True).copy()
    return out


def metrics_dict(result):
    """BacktestResult → flat dict for CSV."""
    return {
        'total_trades': result.total_trades,
        'winners': result.winners,
        'losers': result.losers,
        'win_rate': round(result.win_rate, 2),
        'total_pnl_pts': round(result.total_pnl_pts, 2),
        'avg_win_pts': round(result.avg_win_pts, 2),
        'avg_loss_pts': round(result.avg_loss_pts, 2),
        'profit_factor': round(result.profit_factor, 3),
        'max_drawdown_pts': round(result.max_drawdown_pts, 2),
        'avg_trade_pts': round(result.avg_trade_pts, 2),
        'sharpe': round(result.sharpe, 2),
        'long_trades': result.long_trades,
        'short_trades': result.short_trades,
        'long_win_rate': round(result.long_win_rate, 2),
        'short_win_rate': round(result.short_win_rate, 2),
        'target_exits': result.target_exits,
        'sl_exits': result.sl_exits,
    }


def main():
    t_start = time.time()
    cfg = make_orb_config()
    print(f'ORB config: {cfg.label()}')

    # Identify candidate F&O universe (those NOT already in live ORB)
    fno = list(FNO_LOT_SIZES.keys())
    candidates = [s for s in fno if s not in LIVE_ORB_UNIVERSE]
    print(f'F&O total: {len(fno)} | Already in live ORB: {len(LIVE_ORB_UNIVERSE)} | Candidates: {len(candidates)}')

    # Pre-filter: ensure 5-min data availability
    conn = sqlite3.connect(str(ROOT / 'backtest_data' / 'market_data.db'))
    valid_candidates = []
    for s in candidates:
        r = conn.execute(
            "SELECT COUNT(*) FROM market_data_unified WHERE symbol=? AND timeframe='5minute' AND date>=?",
            (s, TRAIN_START)
        ).fetchone()
        if r[0] >= 25000:
            valid_candidates.append(s)
    conn.close()
    print(f'Candidates with sufficient 5-min data: {len(valid_candidates)}')

    # Preload all data
    print('Preloading 5-min + daily data for all candidates...')
    data = ORBBacktestEngine.preload_data(valid_candidates)
    print(f'Preload done in {time.time()-t_start:.1f}s')

    # Run train + test for each candidate
    rows = []
    engine = ORBBacktestEngine(cfg)

    print()
    print(f'{"Stock":>14} {"TR-Trades":>9} {"TR-WR":>6} {"TR-PF":>7} {"TR-Sharpe":>10} | {"TS-Trades":>9} {"TS-WR":>6} {"TS-PF":>7} {"TS-Sharpe":>10} {"Pass?":>6}')
    print('-' * 110)

    for i, sym in enumerate(valid_candidates, 1):
        try:
            train_data = filter_data_by_period(data[sym], TRAIN_START, TRAIN_END)
            test_data  = filter_data_by_period(data[sym], TEST_START,  TEST_END)

            train_result = engine.run(sym, train_data)
            test_result  = engine.run(sym, test_data)

            tm = metrics_dict(train_result)
            te = metrics_dict(test_result)

            passes = (
                tm['total_trades'] >= MIN_TRADES
                and te['total_trades'] >= MIN_TRADES
                and tm['profit_factor'] >= TRAIN_MIN_PF
                and te['profit_factor'] >= TEST_MIN_PF
                and te['sharpe'] >= TEST_MIN_SHARPE
            )

            row = {'symbol': sym, **{f'train_{k}': v for k, v in tm.items()},
                   **{f'test_{k}': v for k, v in te.items()},
                   'passes_gate': 1 if passes else 0}
            rows.append(row)

            flag = 'YES' if passes else 'no'
            print(f'{sym:>14} {tm["total_trades"]:>9} {tm["win_rate"]:>6.1f} {tm["profit_factor"]:>7.2f} {tm["sharpe"]:>10.2f} '
                  f'| {te["total_trades"]:>9} {te["win_rate"]:>6.1f} {te["profit_factor"]:>7.2f} {te["sharpe"]:>10.2f} {flag:>6}', flush=True)
        except Exception as e:
            print(f'{sym:>14}  ERROR: {e}', flush=True)
            continue

    # Save full per-stock results
    if rows:
        keys = list(rows[0].keys())
        with (OUT / 'per_stock.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

        # Sort by test_sharpe (most rigorous discriminator)
        passing = [r for r in rows if r['passes_gate'] == 1]
        passing.sort(key=lambda r: -r['test_sharpe'])

        print()
        print('=' * 110)
        print(f'PASSING UNIVERSE (gate: trades>={MIN_TRADES} both periods, train PF>=1, test PF>=1, test Sharpe>=0.5)')
        print('=' * 110)
        print(f'{len(passing)} of {len(rows)} stocks passed.')
        print()
        if passing:
            print(f'{"Rank":>4} {"Stock":>14} {"Train PF":>9} {"Test PF":>8} {"Test Sharpe":>11} {"Test WR":>8} {"Test Trades":>11}')
            print('-' * 80)
            for rank, r in enumerate(passing, 1):
                print(f'{rank:>4} {r["symbol"]:>14} {r["train_profit_factor"]:>9.2f} {r["test_profit_factor"]:>8.2f} '
                      f'{r["test_sharpe"]:>11.2f} {r["test_win_rate"]:>8.1f} {r["test_total_trades"]:>11}')

            # Top-N suggested for live expansion
            top_n = min(15, len(passing))
            print()
            print(f'TOP {top_n} suggested for live ORB universe expansion (alongside existing 15):')
            print(f'  {[r["symbol"] for r in passing[:top_n]]}')

            # Save shortlist
            with (OUT / 'shortlist.txt').open('w') as f:
                f.write('# Top candidates by test Sharpe — passing both train and test gates\n')
                for r in passing:
                    f.write(f'{r["symbol"]}\t{r["test_sharpe"]:.2f}\t{r["test_profit_factor"]:.2f}\t{r["test_win_rate"]:.1f}\n')

    print(f'\nTotal runtime: {time.time()-t_start:.1f}s')
    print(f'Artifacts: {OUT}')


if __name__ == '__main__':
    sys.exit(main())
