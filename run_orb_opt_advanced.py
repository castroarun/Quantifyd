"""
ORB Advanced Filters Optimization Sweep
18 configs x 8 stocks = 144 runs
"""
import sys, os, csv, logging, time
logging.disable(logging.WARNING)
sys.path.insert(0, r'c:\Users\Castro\Documents\Projects\Covered_Calls')
os.chdir(r'c:\Users\Castro\Documents\Projects\Covered_Calls')

from services.orb_backtest_engine import ORBConfig, ORBBacktestEngine

STOCKS = ['ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BANKBARODA', 'BPCL', 'M&M', 'BAJFINANCE']

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_orb_advanced.csv')
FIELDNAMES = ['label', 'symbol', 'trades', 'win_rate', 'profit_factor', 'total_pnl',
              'avg_win', 'avg_loss', 'max_dd', 'sharpe', 'target_exits', 'sl_exits',
              'eod_exits', 'long_wr', 'short_wr']

# Base config kwargs
BASE = dict(or_minutes=15, sl_type='or_opposite', target_type='r_multiple', r_multiple=1.5)

# 18 filter configs: (label, extra_kwargs)
CONFIGS = [
    ('VCPR', dict(use_virgin_cpr_filter=True)),
    ('PCPR', dict(use_prev_cpr_filter=True)),
    ('PHL', dict(use_prev_hl_filter=True)),
    ('PVT_0.1', dict(use_pivot_sr_filter=True, pivot_sr_buffer_pct=0.1)),
    ('PVT_0.2', dict(use_pivot_sr_filter=True, pivot_sr_buffer_pct=0.2)),
    ('PVT_0.3', dict(use_pivot_sr_filter=True, pivot_sr_buffer_pct=0.3)),
    ('ID', dict(use_inside_day_filter=True)),
    ('NR4', dict(use_narrow_range_filter=True, nr_lookback=4)),
    ('NR7', dict(use_narrow_range_filter=True, nr_lookback=7)),
    ('VCPR_PCPR', dict(use_virgin_cpr_filter=True, use_prev_cpr_filter=True)),
    ('ID_VWAP', dict(use_inside_day_filter=True, use_vwap_filter=True)),
    ('ID_NR4', dict(use_inside_day_filter=True, use_narrow_range_filter=True, nr_lookback=4)),
    ('PHL_CPRd', dict(use_prev_hl_filter=True, use_cpr_dir_filter=True)),
    ('PHL_VWAP_RSI', dict(use_prev_hl_filter=True, use_vwap_filter=True,
                          use_rsi_filter=True, rsi_long_threshold=60.0, rsi_short_threshold=40.0)),
    ('VCPR_VWAP_CPRd', dict(use_virgin_cpr_filter=True, use_vwap_filter=True, use_cpr_dir_filter=True)),
    ('ID_VWAP_RSI_CPRd', dict(use_inside_day_filter=True, use_vwap_filter=True,
                              use_rsi_filter=True, rsi_long_threshold=60.0, rsi_short_threshold=40.0,
                              use_cpr_dir_filter=True)),
    ('NR4_VWAP_CPRd', dict(use_narrow_range_filter=True, nr_lookback=4,
                           use_vwap_filter=True, use_cpr_dir_filter=True)),
    ('PCPR_PHL_VWAP', dict(use_prev_cpr_filter=True, use_prev_hl_filter=True, use_vwap_filter=True)),
]

def main():
    # Load done set
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add(f"{row['label']}|{row['symbol']}")
        print(f'Skipping {len(done)} already-completed runs')
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    total = len(CONFIGS) * len(STOCKS)
    completed = len(done)

    # Preload data for all stocks
    print(f'Preloading data for {len(STOCKS)} stocks...', flush=True)
    t0 = time.time()
    all_data = ORBBacktestEngine.preload_data(STOCKS)
    print(f'Data loaded in {time.time()-t0:.1f}s', flush=True)

    for cfg_label, cfg_kwargs in CONFIGS:
        params = {**BASE, **cfg_kwargs}
        config = ORBConfig(**params)

        for sym in STOCKS:
            key = f'{cfg_label}|{sym}'
            if key in done:
                continue
            completed += 1

            print(f'[{completed}/{total}] {cfg_label} / {sym} ...', end='', flush=True)
            t1 = time.time()

            try:
                engine = ORBBacktestEngine(config)
                result = engine.run(sym, all_data[sym])

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

                with open(OUTPUT_CSV, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                elapsed = time.time() - t1
                print(f' {elapsed:.1f}s | T={result.total_trades} WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} PnL={result.total_pnl_pts:.1f}', flush=True)

            except Exception as e:
                elapsed = time.time() - t1
                print(f' ERROR {elapsed:.1f}s | {e}', flush=True)
                row = {
                    'label': cfg_label, 'symbol': sym, 'trades': 0,
                    'win_rate': 0, 'profit_factor': 0, 'total_pnl': 0,
                    'avg_win': 0, 'avg_loss': 0, 'max_dd': 0, 'sharpe': 0,
                    'target_exits': 0, 'sl_exits': 0, 'eod_exits': 0,
                    'long_wr': 0, 'short_wr': 0,
                }
                with open(OUTPUT_CSV, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f'\nDone! Results in {OUTPUT_CSV}')

if __name__ == '__main__':
    main()
