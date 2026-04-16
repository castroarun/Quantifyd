"""
ORB Optimization: Bollinger Bands, Multi-Timeframe, and Combined Best configs
Sweeps 20 configs x 8 stocks = 160 runs
"""
import sys, os, csv, logging, time
logging.disable(logging.WARNING)
sys.path.insert(0, r'c:\Users\Castro\Documents\Projects\Covered_Calls')
os.chdir(r'c:\Users\Castro\Documents\Projects\Covered_Calls')

from services.orb_backtest_engine import ORBConfig, ORBBacktestEngine

STOCKS = ['ADANIENT','TATASTEEL','BEL','VEDL','BANKBARODA','BPCL','M&M','BAJFINANCE']

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_orb_combined.csv')
FIELDNAMES = ['label','symbol','trades','win_rate','profit_factor','total_pnl',
              'avg_win','avg_loss','max_dd','sharpe','target_exits','sl_exits',
              'eod_exits','long_wr','short_wr']

# Base params shared by all configs (unless overridden)
BASE = dict(or_minutes=15, sl_type='or_opposite', target_type='r_multiple', r_multiple=1.5)

def make_configs():
    """Return list of (label, config_kwargs) tuples."""
    cfgs = []

    # --- Bollinger Band configs ---
    cfgs.append(('BB20_2', {**BASE, 'use_bb_filter': True, 'bb_period': 20, 'bb_std': 2.0}))
    cfgs.append(('BB20_1.5', {**BASE, 'use_bb_filter': True, 'bb_period': 20, 'bb_std': 1.5}))
    cfgs.append(('BB10_2', {**BASE, 'use_bb_filter': True, 'bb_period': 10, 'bb_std': 2.0}))
    cfgs.append(('BB20_2_VWAP', {**BASE, 'use_bb_filter': True, 'bb_period': 20, 'bb_std': 2.0, 'use_vwap_filter': True}))

    # --- Multi-Timeframe (daily EMA) configs ---
    cfgs.append(('MTF_EMA20', {**BASE, 'use_mtf_filter': True, 'mtf_ema_period': 20}))
    cfgs.append(('MTF_EMA50', {**BASE, 'use_mtf_filter': True, 'mtf_ema_period': 50}))
    cfgs.append(('MTF_EMA10', {**BASE, 'use_mtf_filter': True, 'mtf_ema_period': 10}))
    cfgs.append(('MTF_EMA20_VWAP', {**BASE, 'use_mtf_filter': True, 'mtf_ema_period': 20, 'use_vwap_filter': True}))
    cfgs.append(('MTF_EMA20_CPRd', {**BASE, 'use_mtf_filter': True, 'mtf_ema_period': 20, 'use_cpr_dir_filter': True}))

    # --- Combined "best of" configs ---
    cfgs.append(('VWAP_RSI6040_MTF20', {**BASE,
        'use_vwap_filter': True, 'use_rsi_filter': True, 'rsi_long_threshold': 60.0, 'rsi_short_threshold': 40.0,
        'use_mtf_filter': True, 'mtf_ema_period': 20}))
    cfgs.append(('VWAP_CPRd_MTF20_CPRw05', {**BASE,
        'use_vwap_filter': True, 'use_cpr_dir_filter': True, 'use_mtf_filter': True, 'mtf_ema_period': 20,
        'use_cpr_width_filter': True, 'cpr_width_threshold_pct': 0.5}))
    cfgs.append(('VWAP_MTF20_BB20', {**BASE,
        'use_vwap_filter': True, 'use_mtf_filter': True, 'mtf_ema_period': 20,
        'use_bb_filter': True, 'bb_period': 20, 'bb_std': 2.0}))
    cfgs.append(('VWAP_RSI5545_CPRd_MTF20', {**BASE,
        'use_vwap_filter': True, 'use_rsi_filter': True, 'rsi_long_threshold': 55.0, 'rsi_short_threshold': 45.0,
        'use_cpr_dir_filter': True, 'use_mtf_filter': True, 'mtf_ema_period': 20}))
    cfgs.append(('ID_VWAP_MTF20_CPRd', {**BASE,
        'use_inside_day_filter': True, 'use_vwap_filter': True, 'use_mtf_filter': True, 'mtf_ema_period': 20,
        'use_cpr_dir_filter': True}))
    cfgs.append(('NR4_VWAP_BB20_MTF20', {**BASE,
        'use_narrow_range_filter': True, 'nr_lookback': 4,
        'use_vwap_filter': True, 'use_bb_filter': True, 'bb_period': 20, 'bb_std': 2.0,
        'use_mtf_filter': True, 'mtf_ema_period': 20}))
    cfgs.append(('VWAP_CPRd_PHL_MTF20', {**BASE,
        'use_vwap_filter': True, 'use_cpr_dir_filter': True, 'use_prev_hl_filter': True,
        'use_mtf_filter': True, 'mtf_ema_period': 20}))

    # --- OR=30 configs ---
    cfgs.append(('OR30_VWAP_RSI6040_CPRd', {**BASE, 'or_minutes': 30,
        'use_vwap_filter': True, 'use_rsi_filter': True, 'rsi_long_threshold': 60.0, 'rsi_short_threshold': 40.0,
        'use_cpr_dir_filter': True}))
    cfgs.append(('OR30_VWAP_MTF20_CPRw05', {**BASE, 'or_minutes': 30,
        'use_vwap_filter': True, 'use_mtf_filter': True, 'mtf_ema_period': 20,
        'use_cpr_width_filter': True, 'cpr_width_threshold_pct': 0.5}))

    # --- Better SL/Target combos ---
    cfgs.append(('ATR_SL1.5_R2_VWAP_CPRd', {
        'or_minutes': 15, 'sl_type': 'atr_multiple', 'atr_sl_multiple': 1.5,
        'target_type': 'r_multiple', 'r_multiple': 2.0,
        'use_vwap_filter': True, 'use_cpr_dir_filter': True}))
    cfgs.append(('ATR_SL1_R1.5_VWAP_RSI_MTF20', {
        'or_minutes': 15, 'sl_type': 'atr_multiple', 'atr_sl_multiple': 1.0,
        'target_type': 'r_multiple', 'r_multiple': 1.5,
        'use_vwap_filter': True, 'use_rsi_filter': True, 'rsi_long_threshold': 60.0, 'rsi_short_threshold': 40.0,
        'use_mtf_filter': True, 'mtf_ema_period': 20}))

    return cfgs

def run_batch(configs, data, done_set):
    """Run a batch of configs across all stocks, append to CSV."""
    total = len(configs) * len(STOCKS)
    i = 0
    for label, kwargs in configs:
        config = ORBConfig(**kwargs)
        engine = ORBBacktestEngine(config)
        for sym in STOCKS:
            i += 1
            key = f"{label}|{sym}"
            if key in done_set:
                print(f'[{i}/{total}] SKIP {key}')
                continue
            if sym not in data:
                print(f'[{i}/{total}] NO DATA {sym}')
                continue
            t0 = time.time()
            print(f'[{i}/{total}] {label} | {sym} ...', end='', flush=True)
            try:
                result = engine.run(sym, data[sym])
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
                elapsed = time.time() - t0
                print(f' {elapsed:.1f}s | Trades={result.total_trades} WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} PnL={result.total_pnl_pts:.1f}')
                sys.stdout.flush()
            except Exception as e:
                elapsed = time.time() - t0
                print(f' ERROR {elapsed:.1f}s: {e}')
                sys.stdout.flush()

def main():
    # Load done set
    done_set = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done_set = {f"{r['label']}|{r['symbol']}" for r in csv.DictReader(f)}
        print(f'Loaded {len(done_set)} already-done runs')
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        print('Created new CSV with header')

    all_configs = make_configs()
    print(f'Total configs: {len(all_configs)} x {len(STOCKS)} stocks = {len(all_configs)*len(STOCKS)} runs')

    # Preload data
    print('Preloading data for all stocks...', flush=True)
    t0 = time.time()
    data = ORBBacktestEngine.preload_data(STOCKS)
    print(f'Data loaded in {time.time()-t0:.1f}s')
    for sym, d in data.items():
        print(f'  {sym}: {len(d["5min"])} 5-min bars, {len(d["day"])} daily bars')

    # Run in batches of 8 configs (8 configs x 8 stocks = 64 runs per batch)
    BATCH_SIZE = 8
    for batch_start in range(0, len(all_configs), BATCH_SIZE):
        batch = all_configs[batch_start:batch_start+BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(all_configs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f'\n=== Batch {batch_num}/{total_batches} ({len(batch)} configs) ===')
        run_batch(batch, data, done_set)

    print('\nDone! Results in:', OUTPUT_CSV)

if __name__ == '__main__':
    main()
