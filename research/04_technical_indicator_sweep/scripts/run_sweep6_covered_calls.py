"""
Sweep 6: Covered Call Overlay Optimization
Test different technical signals for timing covered call sells on MQ portfolio stocks.
"""
import sys, os, csv, time, logging

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_sweep6_covered_calls.csv')
FIELDNAMES = ['label', 'cagr', 'sharpe', 'sortino', 'max_drawdown', 'calmar',
              'total_trades', 'win_rate', 'final_value', 'total_return_pct', 'topups',
              'cc_calls_sold', 'cc_premium', 'cc_buyback', 'cc_net_income',
              'cc_otm', 'cc_itm', 'cc_buybacks', 'cc_income_pct']

# Skip already-done configs
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['label'] for row in csv.DictReader(f)}
    print(f'Skipping {len(done)} already-completed configs')
else:
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Base params
base = dict(
    portfolio_size=30,
    equity_allocation_pct=0.95,
    hard_stop_loss=0.50,
    rebalance_ath_drawdown=0.20,
)

configs = []

# 1. Baseline (no CC)
configs.append(('BASELINE_NO_CC', {**base}))

# 2. Always sell CC (monthly on every stock)
for otm in [0.02, 0.05, 0.10]:
    configs.append((f'ALWAYS_{int(otm*100)}OTM', {**base, 'cc_enabled': True, 'cc_signal_type': 'always', 'cc_strike_otm_pct': otm}))

# 3. EMA cross below (bearish crossover)
for fast, slow in [(5, 20), (10, 20), (10, 30), (20, 50)]:
    for otm in [0.02, 0.05, 0.10]:
        configs.append((f'EMA_CROSS_{fast}_{slow}_{int(otm*100)}OTM', {
            **base, 'cc_enabled': True, 'cc_signal_type': 'ema_cross',
            'cc_signal_fast': fast, 'cc_signal_slow': slow, 'cc_strike_otm_pct': otm}))

# 4. EMA below (state: fast EMA below slow EMA)
for fast, slow in [(10, 20), (20, 50)]:
    for otm in [0.02, 0.05, 0.10]:
        configs.append((f'EMA_BELOW_{fast}_{slow}_{int(otm*100)}OTM', {
            **base, 'cc_enabled': True, 'cc_signal_type': 'ema_below',
            'cc_signal_fast': fast, 'cc_signal_slow': slow, 'cc_strike_otm_pct': otm}))

# 5. RSI overbought
for period, threshold in [(14, 70), (14, 65), (7, 70)]:
    for otm in [0.02, 0.05, 0.10]:
        configs.append((f'RSI_{period}_OB{int(threshold)}_{int(otm*100)}OTM', {
            **base, 'cc_enabled': True, 'cc_signal_type': 'rsi_ob',
            'cc_signal_fast': period, 'cc_signal_threshold': threshold, 'cc_strike_otm_pct': otm}))

# 6. Stochastic overbought
for otm in [0.02, 0.05, 0.10]:
    configs.append((f'STOCH_14_OB80_{int(otm*100)}OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': 'stoch_ob',
        'cc_signal_fast': 14, 'cc_signal_threshold': 80, 'cc_strike_otm_pct': otm}))

# 7. Bollinger Band upper touch
for otm in [0.02, 0.05, 0.10]:
    configs.append((f'BB_UPPER_20_2.0_{int(otm*100)}OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': 'bb_upper',
        'cc_signal_fast': 20, 'cc_signal_threshold': 2.0, 'cc_strike_otm_pct': otm}))

# 8. Keltner Channel upper touch
for otm in [0.02, 0.05, 0.10]:
    configs.append((f'KC_UPPER_20_2.0_{int(otm*100)}OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': 'kc_upper',
        'cc_signal_fast': 20, 'cc_signal_threshold': 2.0, 'cc_strike_otm_pct': otm}))

# 9. Ichimoku bearish cross (Tenkan < Kijun)
for otm in [0.02, 0.05, 0.10]:
    configs.append((f'ICHIMOKU_BEAR_9_26_{int(otm*100)}OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': 'ichimoku_bear',
        'cc_signal_fast': 9, 'cc_signal_slow': 26, 'cc_strike_otm_pct': otm}))

# 10. MACD bearish cross
for otm in [0.02, 0.05, 0.10]:
    configs.append((f'MACD_BEAR_12_26_{int(otm*100)}OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': 'macd_bear',
        'cc_signal_fast': 12, 'cc_signal_slow': 26, 'cc_strike_otm_pct': otm}))

# 11. ADX low (sideways market — good for CC)
for threshold in [20, 25]:
    for otm in [0.02, 0.05, 0.10]:
        configs.append((f'ADX_LOW_{int(threshold)}_{int(otm*100)}OTM', {
            **base, 'cc_enabled': True, 'cc_signal_type': 'adx_low',
            'cc_signal_fast': 14, 'cc_signal_threshold': threshold, 'cc_strike_otm_pct': otm}))

# ============================================================
# 12. CC Position Management Strategies (using best signal types at 5% OTM)
# ============================================================

# Roll Up: when stock approaches strike, roll to higher strike
for sig_type, fast, slow in [('ema_cross', 10, 20), ('rsi_ob', 14, 70), ('always', 10, 20)]:
    lbl = f'{sig_type.upper()}_ROLLUP' if sig_type != 'always' else 'ALWAYS_ROLLUP'
    configs.append((f'{lbl}_5OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': sig_type,
        'cc_signal_fast': fast, 'cc_signal_slow': slow, 'cc_signal_threshold': 70.0,
        'cc_strike_otm_pct': 0.05, 'cc_mgmt': 'roll_up',
        'cc_roll_up_trigger': 0.02, 'cc_roll_up_distance': 0.05}))

# Roll Out: near expiry + near ITM → extend to next month
for sig_type, fast, slow in [('ema_cross', 10, 20), ('rsi_ob', 14, 70), ('always', 10, 20)]:
    lbl = f'{sig_type.upper()}_ROLLOUT' if sig_type != 'always' else 'ALWAYS_ROLLOUT'
    configs.append((f'{lbl}_5OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': sig_type,
        'cc_signal_fast': fast, 'cc_signal_slow': slow, 'cc_signal_threshold': 70.0,
        'cc_strike_otm_pct': 0.05, 'cc_mgmt': 'roll_out', 'cc_roll_out_days': 5}))

# Stop Loss: buy back CC if it doubles in value (stock rallied hard)
for sig_type, fast, slow in [('ema_cross', 10, 20), ('rsi_ob', 14, 70), ('always', 10, 20)]:
    lbl = f'{sig_type.upper()}_SL2X' if sig_type != 'always' else 'ALWAYS_SL2X'
    configs.append((f'{lbl}_5OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': sig_type,
        'cc_signal_fast': fast, 'cc_signal_slow': slow, 'cc_signal_threshold': 70.0,
        'cc_strike_otm_pct': 0.05, 'cc_mgmt': 'stop_loss', 'cc_stop_loss_mult': 2.0}))

# Roll Defend: combine roll up + defend on drops
for sig_type, fast, slow in [('ema_cross', 10, 20), ('always', 10, 20)]:
    lbl = f'{sig_type.upper()}_DEFEND' if sig_type != 'always' else 'ALWAYS_DEFEND'
    configs.append((f'{lbl}_5OTM', {
        **base, 'cc_enabled': True, 'cc_signal_type': sig_type,
        'cc_signal_fast': fast, 'cc_signal_slow': slow, 'cc_signal_threshold': 70.0,
        'cc_strike_otm_pct': 0.05, 'cc_mgmt': 'roll_defend',
        'cc_roll_up_trigger': 0.02, 'cc_roll_up_distance': 0.05, 'cc_defend_drop_pct': 0.05}))

print(f'=== Sweep 6: Covered Call Overlay | {len(configs)} configs ===')
print(f'Base: PS30, HSL50, EQ95, ATH20')
print(f'Output: {OUTPUT_CSV}')

# Preload data once
print('Preloading universe + price data...', end='', flush=True)
t0 = time.time()
universe, price_data = MQBacktestEngine.preload_data(MQBacktestConfig())
print(f' {time.time()-t0:.0f}s')

total = len(configs)
for i, (label, params) in enumerate(configs, 1):
    if label in done:
        print(f'[{i}/{total}] {label} ... SKIPPED')
        continue

    print(f'[{i}/{total}] {label} ...', end='', flush=True)
    t1 = time.time()

    config = MQBacktestConfig(**params)
    engine = MQBacktestEngine(config, preloaded_universe=universe, preloaded_price_data=price_data)
    result = engine.run()

    elapsed = time.time() - t1

    # CC income as % of initial capital per year
    years = 3.0  # 2023-2025
    cc_income_pct = (result.cc_net_income / 10_000_000 / years * 100) if result.cc_net_income else 0

    row = {
        'label': label,
        'cagr': result.cagr,
        'sharpe': result.sharpe_ratio,
        'sortino': result.sortino_ratio,
        'max_drawdown': result.max_drawdown,
        'calmar': result.calmar_ratio,
        'total_trades': result.total_trades,
        'win_rate': result.win_rate,
        'final_value': round(result.final_value, 2),
        'total_return_pct': result.total_return_pct,
        'topups': result.total_topups,
        'cc_calls_sold': result.cc_total_calls_sold,
        'cc_premium': round(result.cc_total_premium, 0),
        'cc_buyback': round(result.cc_total_buyback, 0),
        'cc_net_income': round(result.cc_net_income, 0),
        'cc_otm': result.cc_expired_otm,
        'cc_itm': result.cc_expired_itm,
        'cc_buybacks': result.cc_buybacks,
        'cc_income_pct': round(cc_income_pct, 2),
    }

    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f' {elapsed:.0f}s | CAGR={row["cagr"]:.2f}% Sharpe={row["sharpe"]:.2f} CC_sold={row["cc_calls_sold"]} OTM={row["cc_otm"]} ITM={row["cc_itm"]} Net={row["cc_net_income"]:,.0f} Income%={row["cc_income_pct"]:.2f}%')
    sys.stdout.flush()

print(f'\n=== DONE: {total} configs ===')
print(f'Results: {OUTPUT_CSV}')
