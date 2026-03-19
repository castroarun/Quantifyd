
import pathlib

target = pathlib.Path('run_agent3_technical_optimization.py')

script = """
import sys, time, logging, csv, pathlib
logging.disable(logging.WARNING)
from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT = str(pathlib.Path(__file__).parent / 'optimization_agent3_technical.csv')

base = dict(portfolio_size=30, equity_allocation_pct=0.95, hard_stop_loss=0.20,
            rebalance_ath_drawdown=0.15, trailing_stop_loss=True,
            daily_ath_drawdown_exit=True, use_technical_filter=True)

configs = []
configs.append(('BASELINE_NO_TECH', {**base, 'use_technical_filter': False}))

for f, s in [(9,21),(10,30),(20,50)]:
    for et in ['crossover','price_below']:
        configs.append((f'EMA_{f}_{s}_exit_{et}', {
            **base, 'use_ema_entry': True, 'use_ema_exit': True,
            'ema_fast': f, 'ema_slow': s, 'ema_exit_type': et}))

for p in [7,14]:
    for ob in [75,80,85]:
        configs.append((f'RSI_{p}_ob{ob}', {
            **base, 'use_rsi_filter': True, 'use_rsi_exit': True,
            'rsi_period': p, 'rsi_exit_overbought': ob,
            'rsi_min_entry': 40, 'rsi_max_entry': 70}))

for atr in [7, 10, 14]:
    for mult in [2.0, 3.0]:
        configs.append((f'STREND_atr{atr}_m{mult}', {
            **base, 'use_supertrend': True, 'supertrend_atr': atr,
            'supertrend_mult': mult, 'supertrend_entry_bullish': True,
            'supertrend_exit_flip': True}))

configs.append(('MACD_12_26_9', {**base, 'use_macd': True, 'macd_fast': 12,
    'macd_slow': 26, 'macd_signal': 9, 'require_macd_positive': True,
    'require_macd_above_signal': True}))

for mt in [20,25,30]:
    configs.append((f'ADX_min{mt}', {**base, 'use_adx': True, 'adx_period': 14,
        'adx_min_trend': mt, 'require_plus_di_above': True}))

configs.append(('WEEKLY_EMA20', {**base, 'use_weekly_filter': True,
    'weekly_ema_period': 20, 'require_weekly_above_ema': True}))

configs.append(('COMBO_EMA20_50_RSI14', {**base, 'use_ema_entry': True,
    'use_ema_exit': True, 'ema_fast': 20, 'ema_slow': 50,
    'ema_exit_type': 'price_below', 'use_rsi_filter': True, 'rsi_period': 14,
    'rsi_min_entry': 40, 'rsi_max_entry': 70}))

configs.append(('COMBO_EMA9_21_STREND', {**base, 'use_ema_entry': True,
    'use_ema_exit': True, 'ema_fast': 9, 'ema_slow': 21,
    'use_supertrend': True, 'supertrend_atr': 10, 'supertrend_mult': 3.0,
    'supertrend_exit_flip': True}))

configs.append(('COMBO_MACD_RSI_WEEKLY', {**base, 'use_macd': True,
    'use_rsi_filter': True, 'use_weekly_filter': True, 'rsi_period': 14,
    'rsi_min_entry': 40, 'rsi_max_entry': 70}))


def run_one(label, params):
    try:
        cfg = MQBacktestConfig()
        for k, v in params.items():
            if hasattr(cfg, k): setattr(cfg, k, v)
        r = MQBacktestEngine(cfg).run()
        return dict(label=label, cagr=r.cagr, sharpe=r.sharpe_ratio,
            sortino=r.sortino_ratio, max_drawdown=r.max_drawdown,
            calmar=r.calmar_ratio, total_trades=r.total_trades,
            win_rate=r.win_rate, avg_win_pct=r.avg_win_pct,
            avg_loss_pct=r.avg_loss_pct, final_value=r.final_value,
            total_return_pct=r.total_return_pct, topups=r.total_topups)
    except Exception as e:
        print(f' ERROR: {e}')
        return None


if __name__ == '__main__':
    total = len(configs)
    print(f'Testing {total} configs')
    print('=' * 80)
    sys.stdout.flush()
    results = []
    t0 = time.time()
    for i, (label, params) in enumerate(configs, 1):
        t1 = time.time()
        print(f'[{i:2d}/{total}] {label} ...', end='', flush=True)
        row = run_one(label, params)
        el = time.time() - t1
        if row:
            results.append(row)
            print(f' {el:.0f}s | CAGR={row["cagr"]:.2f}% Sharpe={row["sharpe"]:.2f} DD={row["max_drawdown"]:.2f}%')
        else:
            print(f' FAILED {el:.1f}s')
        sys.stdout.flush()

    tt = time.time() - t0
    results.sort(key=lambda x: x['cagr'], reverse=True)

    print()
    print('=' * 130)
    print(f'RESULTS sorted by CAGR | Time: {tt:.0f}s ({tt/60:.1f}min)')
    print('=' * 130)
    hdr = f"{'Rank':>4} {'Label':<35} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} {'Trades':>7} {'WinR%':>6} {'FinalVal':>14} {'Topups':>6}"
    print(hdr)
    print('-' * 130)
    for rank, r in enumerate(results, 1):
        l = r['label']
        print(f'{rank:4d} {l:<35} {r["cagr"]:7.2f} {r["sharpe"]:7.2f} {r["sortino"]:8.2f} {r["max_drawdown"]:7.2f} {r["calmar"]:7.2f} {r["total_trades"]:7d} {r["win_rate"]:6.1f} {r["final_value"]:14,.0f} {r["topups"]:6d}')

    fns = ['rank','label','cagr','sharpe','sortino','max_drawdown','calmar',
           'total_trades','win_rate','avg_win_pct','avg_loss_pct',
           'final_value','total_return_pct','topups']
    with open(OUTPUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for rank, r in enumerate(results, 1):
            w.writerow({'rank': rank, **r})
    print(f'
Saved to: {OUTPUT}')
    print(f'Total: {total} | OK: {len(results)} | Failed: {total-len(results)} | Time: {tt:.0f}s')
"""

target.write_text(script.strip())
print(f'Written {len(script)} bytes to {target}')
