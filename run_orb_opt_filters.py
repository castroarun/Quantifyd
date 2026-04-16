"""
ORB Filter Combinations Optimization
=====================================
Sweep 16 filter configs across 8 stocks = 128 runs.
Base: OR15, SL=or_opposite, Target=R1.5
"""

import sys, os, csv, logging, time
logging.disable(logging.WARNING)
sys.path.insert(0, r'c:\Users\Castro\Documents\Projects\Covered_Calls')
os.chdir(r'c:\Users\Castro\Documents\Projects\Covered_Calls')

from services.orb_backtest_engine import ORBConfig, ORBBacktestEngine

SYMBOLS = ['ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BANKBARODA', 'BPCL', 'M&M', 'BAJFINANCE']

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_orb_filters.csv')

FIELDNAMES = [
    'label', 'symbol', 'trades', 'win_rate', 'profit_factor', 'total_pnl',
    'avg_win', 'avg_loss', 'max_dd', 'sharpe', 'target_exits', 'sl_exits',
    'eod_exits', 'long_wr', 'short_wr'
]

# Base config params (shared by all)
BASE = dict(
    or_minutes=15,
    sl_type='or_opposite',
    target_type='r_multiple',
    r_multiple=1.5,
)

# Define 16 filter configs: (label, extra_params)
CONFIGS = [
    ('F01_VWAP', dict(use_vwap_filter=True)),
    ('F02_RSI60', dict(use_rsi_filter=True, rsi_long_threshold=60.0, rsi_short_threshold=40.0)),
    ('F03_RSI55', dict(use_rsi_filter=True, rsi_long_threshold=55.0, rsi_short_threshold=45.0)),
    ('F04_RSI50', dict(use_rsi_filter=True, rsi_long_threshold=50.0, rsi_short_threshold=50.0)),
    ('F05_CPRdir', dict(use_cpr_dir_filter=True)),
    ('F06_CPRw03', dict(use_cpr_width_filter=True, cpr_width_threshold_pct=0.3)),
    ('F07_CPRw05', dict(use_cpr_width_filter=True, cpr_width_threshold_pct=0.5)),
    ('F08_CPRw08', dict(use_cpr_width_filter=True, cpr_width_threshold_pct=0.8)),
    ('F09_VWAP_RSI60', dict(use_vwap_filter=True, use_rsi_filter=True, rsi_long_threshold=60.0, rsi_short_threshold=40.0)),
    ('F10_VWAP_CPRdir', dict(use_vwap_filter=True, use_cpr_dir_filter=True)),
    ('F11_RSI60_CPRdir', dict(use_rsi_filter=True, rsi_long_threshold=60.0, rsi_short_threshold=40.0, use_cpr_dir_filter=True)),
    ('F12_VWAP_RSI60_CPRdir', dict(use_vwap_filter=True, use_rsi_filter=True, rsi_long_threshold=60.0, rsi_short_threshold=40.0, use_cpr_dir_filter=True)),
    ('F13_VWAP_RSI60_CPRdir_CPRw05', dict(use_vwap_filter=True, use_rsi_filter=True, rsi_long_threshold=60.0, rsi_short_threshold=40.0, use_cpr_dir_filter=True, use_cpr_width_filter=True, cpr_width_threshold_pct=0.5)),
    ('F14_VWAP_RSI55_CPRdir', dict(use_vwap_filter=True, use_rsi_filter=True, rsi_long_threshold=55.0, rsi_short_threshold=45.0, use_cpr_dir_filter=True)),
    ('F15_VWAP_CPRdir_CPRw05', dict(use_vwap_filter=True, use_cpr_dir_filter=True, use_cpr_width_filter=True, cpr_width_threshold_pct=0.5)),
    ('F16_ALL', dict(use_vwap_filter=True, use_rsi_filter=True, rsi_long_threshold=60.0, rsi_short_threshold=40.0, use_cpr_dir_filter=True, use_cpr_width_filter=True, cpr_width_threshold_pct=0.5)),
]

def main():
    # Load done set
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add(f"{row['label']}_{row['symbol']}")
        print(f'Skipping {len(done)} already-completed runs')
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    total = len(CONFIGS) * len(SYMBOLS)
    remaining = total - len(done)
    print(f'Total: {total} runs, Remaining: {remaining}')

    # Preload data once
    t0 = time.time()
    print(f'Preloading data for {len(SYMBOLS)} symbols...', end='', flush=True)
    data = ORBBacktestEngine.preload_data(SYMBOLS)
    print(f' {time.time()-t0:.1f}s')

    run_idx = 0
    for cfg_label, cfg_params in CONFIGS:
        # Build config
        params = {**BASE, **cfg_params}
        config = ORBConfig(**params)

        engine = ORBBacktestEngine(config)

        for sym in SYMBOLS:
            key = f'{cfg_label}_{sym}'
            if key in done:
                run_idx += 1
                continue

            run_idx += 1
            t1 = time.time()
            print(f'[{run_idx}/{total}] {cfg_label} | {sym} ...', end='', flush=True)

            result = engine.run(sym, data[sym])
            elapsed = time.time() - t1

            row = {
                'label': cfg_label,
                'symbol': sym,
                'trades': result.total_trades,
                'win_rate': round(result.win_rate, 2),
                'profit_factor': round(result.profit_factor, 2),
                'total_pnl': round(result.total_pnl_pts, 2),
                'avg_win': round(result.avg_win_pts, 2),
                'avg_loss': round(result.avg_loss_pts, 2),
                'max_dd': round(result.max_drawdown_pts, 2),
                'sharpe': round(result.sharpe, 2),
                'target_exits': result.target_exits,
                'sl_exits': result.sl_exits,
                'eod_exits': result.eod_exits,
                'long_wr': round(result.long_win_rate, 2),
                'short_wr': round(result.short_win_rate, 2),
            }

            # Write immediately
            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

            print(f' {elapsed:.1f}s | Trades={result.total_trades} WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} PnL={result.total_pnl_pts:.1f}')
            sys.stdout.flush()

    print(f'\nDone! Results in {OUTPUT_CSV}')

if __name__ == '__main__':
    main()
