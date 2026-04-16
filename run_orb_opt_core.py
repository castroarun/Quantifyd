"""
ORB Core Parameter Optimization
Sweeps: OR window, max trades/day, SL type, target type across 8 stocks.
48 configs x 8 stocks = 384 runs.
"""

import sys, os, csv, logging, time
logging.disable(logging.WARNING)
sys.path.insert(0, r'c:\Users\Castro\Documents\Projects\Covered_Calls')
os.chdir(r'c:\Users\Castro\Documents\Projects\Covered_Calls')

from services.orb_backtest_engine import ORBConfig, ORBBacktestEngine

SYMBOLS = ['ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BANKBARODA', 'BPCL', 'M&M', 'BAJFINANCE']

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_orb_core.csv')
FIELDNAMES = [
    'label', 'symbol', 'trades', 'win_rate', 'profit_factor', 'total_pnl',
    'avg_win', 'avg_loss', 'max_dd', 'sharpe', 'target_exits', 'sl_exits',
    'eod_exits', 'long_wr', 'short_wr'
]

# --- Build all 48 configs ---
def build_configs():
    """Generate all (label, ORBConfig) tuples."""
    sl_target_combos = [
        # (sl_type, fixed_sl_pct, atr_sl_multiple, target_type, r_multiple, or_range_multiple, suffix)
        ('or_opposite', 1.0, 1.5, 'r_multiple',        1.5, 1.0, 'SLoro_TGTr1.5'),
        ('or_opposite', 1.0, 1.5, 'r_multiple',        2.0, 1.0, 'SLoro_TGTr2.0'),
        ('or_opposite', 1.0, 1.5, 'r_multiple',        1.0, 1.0, 'SLoro_TGTr1.0'),
        ('fixed_pct',   0.5, 1.5, 'r_multiple',        1.5, 1.0, 'SLfp0.5_TGTr1.5'),
        ('fixed_pct',   1.0, 1.5, 'r_multiple',        1.5, 1.0, 'SLfp1.0_TGTr1.5'),
        ('fixed_pct',   1.5, 1.5, 'r_multiple',        2.0, 1.0, 'SLfp1.5_TGTr2.0'),
        ('atr_multiple', 1.0, 1.0, 'r_multiple',       1.5, 1.0, 'SLatr1.0_TGTr1.5'),
        ('atr_multiple', 1.0, 1.5, 'r_multiple',       2.0, 1.0, 'SLatr1.5_TGTr2.0'),
        ('atr_multiple', 1.0, 2.0, 'r_multiple',       2.5, 1.0, 'SLatr2.0_TGTr2.5'),
        ('or_opposite', 1.0, 1.5, 'or_range_multiple', 1.5, 1.0, 'SLoro_TGTorm1.0'),
        ('or_opposite', 1.0, 1.5, 'or_range_multiple', 1.5, 1.5, 'SLoro_TGTorm1.5'),
        ('or_opposite', 1.0, 1.5, 'or_range_multiple', 1.5, 2.0, 'SLoro_TGTorm2.0'),
    ]

    configs = []
    for or_min in [15, 30]:
        for max_t in [1, 2]:
            for sl_type, fp, atr_m, tgt_type, r_m, orm, suffix in sl_target_combos:
                label = f'OR{or_min}_MT{max_t}_{suffix}'
                cfg = ORBConfig(
                    or_minutes=or_min,
                    max_trades_per_day=max_t,
                    sl_type=sl_type,
                    fixed_sl_pct=fp,
                    atr_sl_multiple=atr_m,
                    target_type=tgt_type,
                    r_multiple=r_m,
                    or_range_multiple=orm,
                )
                configs.append((label, cfg))
    return configs


def load_done():
    """Load already-completed (label, symbol) pairs."""
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r') as f:
            for row in csv.DictReader(f):
                done.add((row['label'], row['symbol']))
        print(f'Found {len(done)} already-completed runs, skipping them.')
    return done


def main():
    configs = build_configs()
    print(f'Total configs: {len(configs)}, symbols: {len(SYMBOLS)}, total runs: {len(configs) * len(SYMBOLS)}')

    done = load_done()

    # Write header if file doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Preload data once
    t0 = time.time()
    print('Preloading data...', flush=True)
    data = ORBBacktestEngine.preload_data(SYMBOLS)
    print(f'Data loaded in {time.time() - t0:.1f}s for {len(data)} symbols', flush=True)

    total = len(configs) * len(SYMBOLS)
    i = 0
    for label, cfg in configs:
        engine = ORBBacktestEngine(cfg)
        for sym in SYMBOLS:
            i += 1
            if (label, sym) in done:
                print(f'[{i}/{total}] {label} {sym} ... SKIP (already done)')
                sys.stdout.flush()
                continue

            t1 = time.time()
            try:
                result = engine.run(sym, data[sym])
                elapsed = time.time() - t1

                row = {
                    'label': label,
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

                print(f'[{i}/{total}] {label} {sym} ... {elapsed:.1f}s | WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} PnL={result.total_pnl_pts:.1f}')
                sys.stdout.flush()

            except Exception as e:
                elapsed = time.time() - t1
                print(f'[{i}/{total}] {label} {sym} ... ERROR {elapsed:.1f}s: {e}')
                sys.stdout.flush()

    print(f'\nDone! Results in {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
