"""
Phase 5 IPO Sweep: Push past 65% WR ceiling.

Key insights from Phase 4 (318 configs):
- Best WR = 61.7% (ListGain + BRK3% + T30/SL15, 60 trades)
- ListGain + BRK3% is the strongest filter combo
- T8/SL7 gives 56.7% WR on 233 trades
- RSI/EMA/ADX/MFI are redundant (don't change WR)

Strategy for 65%+:
1. Wider SL ratios (T30/SL20, T30/SL25, T40/SL20, T40/SL25)
2. Tight targets with wide SL (T5-T8 / SL10-SL20)
3. Best filters (ListGain, BRK3-5%) with wider SL
4. Higher breakout thresholds (BRK7%, BRK10%)
5. Wider ATH drawdown tolerance (ATH15, ATH20)
6. Time-capped exits with loose SL
"""

import sys, os, csv, time, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.disable(logging.WARNING)

from services.ipo_strategy import IPOConfig, IPOStrategyBacktester

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ipo_phase5_sweep.csv')
FIELDNAMES = ['label', 'trades', 'win_rate', 'profit_factor', 'expectancy',
              'total_return', 'median_return', 'avg_hold_days', 'filters']

configs = []

# ============================================================
# PHASE E: Wider SL ratios on base signal (no filters)
# ============================================================
for tgt in [30, 40, 50]:
    for sl in [20, 25, 30]:
        if sl >= tgt:
            continue
        label = f'E_BASE_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
        ), f'T{tgt}/SL{sl}'))

# Tight targets with wide SL
for tgt in [5, 6, 8, 10]:
    for sl in [10, 15, 20]:
        label = f'E_BASE_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
        ), f'T{tgt}/SL{sl}'))

# Ultra-tight: T3-T4 with moderate SL
for tgt in [3, 4]:
    for sl in [5, 7, 10]:
        label = f'E_BASE_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
        ), f'T{tgt}/SL{sl}'))

# ============================================================
# PHASE F: Best filters (LG + BRK3%) with wider SL
# ============================================================
# ListGain + BRK3% with wider SL
for tgt in [30, 40, 50]:
    for sl in [20, 25, 30]:
        if sl >= tgt:
            continue
        label = f'F_LG_BRK3_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
            require_listing_gain=True, min_breakout_pct=3.0,
        ), f'LG+BRK3+T{tgt}/SL{sl}'))

# LG + BRK3 + tight targets + wide SL
for tgt in [5, 6, 8, 10]:
    for sl in [10, 15, 20]:
        label = f'F_LG_BRK3_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
            require_listing_gain=True, min_breakout_pct=3.0,
        ), f'LG+BRK3+T{tgt}/SL{sl}'))

# LG + BRK3 + ultra-tight
for tgt in [3, 4]:
    for sl in [5, 7, 10]:
        label = f'F_LG_BRK3_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
            require_listing_gain=True, min_breakout_pct=3.0,
        ), f'LG+BRK3+T{tgt}/SL{sl}'))

# ============================================================
# PHASE G: Higher breakout thresholds + ListGain
# ============================================================
for brk in [5, 7, 10]:
    for tgt, sl in [(30, 15), (30, 20), (40, 20), (8, 10), (10, 15), (5, 7)]:
        label = f'G_LG_BRK{brk}_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
            require_listing_gain=True, min_breakout_pct=float(brk),
        ), f'LG+BRK{brk}+T{tgt}/SL{sl}'))

# BRK without LG (higher thresholds may be enough alone)
for brk in [5, 7, 10]:
    for tgt, sl in [(30, 20), (40, 20), (8, 10), (5, 7)]:
        label = f'G_BRK{brk}_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
            min_breakout_pct=float(brk),
        ), f'BRK{brk}+T{tgt}/SL{sl}'))

# ============================================================
# PHASE H: Time-capped exits with loose SL
# ============================================================
for days in [10, 15, 20, 30]:
    for sl in [10, 15, 20]:
        label = f'H_TIME{days}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='time_exit', max_hold_days=days, stop_loss_pct=sl,
        ), f'TIME{days}d/SL{sl}'))

# Time cap + ListGain + BRK3%
for days in [10, 15, 20, 30]:
    for sl in [10, 15, 20]:
        label = f'H_LG_BRK3_TIME{days}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='time_exit', max_hold_days=days, stop_loss_pct=sl,
            require_listing_gain=True, min_breakout_pct=3.0,
        ), f'LG+BRK3+TIME{days}d/SL{sl}'))

# ============================================================
# PHASE I: Wider ATH lookback + trail-based exits
# ============================================================
# Wider trailing SL (very loose)
for trail in [20, 25, 30]:
    label = f'I_TRAIL_{trail}PCT'
    configs.append((label, dict(
        exit_strategy='trailing_sl', trail_pct=trail,
    ), f'TRAIL{trail}%'))

# Trail + ListGain + BRK3%
for trail in [20, 25, 30]:
    label = f'I_LG_BRK3_TRAIL_{trail}PCT'
    configs.append((label, dict(
        exit_strategy='trailing_sl', trail_pct=trail,
        require_listing_gain=True, min_breakout_pct=3.0,
    ), f'LG+BRK3+TRAIL{trail}%'))

# ============================================================
# PHASE J: Narrow IPO age window + best combos
# ============================================================
# Earlier breakouts only (max 20-30 days)
for max_age in [15, 20, 25]:
    for tgt, sl in [(30, 20), (8, 10), (5, 7)]:
        label = f'J_AGE{max_age}_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
            max_ipo_age_days=max_age,
        ), f'AGE<={max_age}d+T{tgt}/SL{sl}'))

# Narrow age + LG + BRK3%
for max_age in [20, 25, 30]:
    for tgt, sl in [(30, 20), (30, 25), (40, 20), (8, 10), (5, 7)]:
        label = f'J_AGE{max_age}_LG_BRK3_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
            max_ipo_age_days=max_age,
            require_listing_gain=True, min_breakout_pct=3.0,
        ), f'AGE<={max_age}d+LG+BRK3+T{tgt}/SL{sl}'))

# ============================================================
# PHASE K: LG + BRK5%+ with wider SL (strongest filter stack)
# ============================================================
for brk in [5, 7]:
    for tgt, sl in [(30, 20), (30, 25), (40, 20), (40, 25), (20, 15), (15, 10), (10, 10), (8, 10), (5, 7), (5, 10)]:
        label = f'K_LG_BRK{brk}_T{tgt}_SL{sl}'
        configs.append((label, dict(
            exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
            require_listing_gain=True, min_breakout_pct=float(brk),
        ), f'LG+BRK{brk}+T{tgt}/SL{sl}'))

# ============================================================
# PHASE L: GapUp entry filter combos
# ============================================================
for tgt, sl in [(30, 20), (30, 25), (8, 10), (5, 7), (5, 10)]:
    label = f'L_GAPUP_LG_T{tgt}_SL{sl}'
    configs.append((label, dict(
        exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
        require_gap_up=True, require_listing_gain=True,
    ), f'GAPUP+LG+T{tgt}/SL{sl}'))

for tgt, sl in [(30, 20), (30, 25), (8, 10), (5, 7)]:
    label = f'L_GAPUP_LG_BRK3_T{tgt}_SL{sl}'
    configs.append((label, dict(
        exit_strategy='fixed_target', target_pct=tgt, stop_loss_pct=sl,
        require_gap_up=True, require_listing_gain=True, min_breakout_pct=3.0,
    ), f'GAPUP+LG+BRK3+T{tgt}/SL{sl}'))


# ============================================================
# RUN SWEEP
# ============================================================
print(f'\n=== IPO Phase 5 Sweep: {len(configs)} configs ===')
print('Targeting 65%+ WR with wider SL ratios + best filter combos\n')

# Skip already-done
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, encoding='utf-8') as f:
        done = {row['label'] for row in csv.DictReader(f)}
    print(f'Skipping {len(done)} already-completed configs')
else:
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Preload data once
print('Preloading IPO stock data...', end='', flush=True)
t0 = time.time()
preloaded = IPOStrategyBacktester.preload_data(ipo_start_year=2015, min_bars=10)
print(f' {len(preloaded)} stocks loaded in {time.time()-t0:.1f}s')

total = len(configs)
done_count = 0
hit_65 = 0

for idx, (label, params, filter_desc) in enumerate(configs):
    if label in done:
        done_count += 1
        continue

    if (idx + 1) % 25 == 0:
        print(f'\n--- Progress: {idx+1}/{total} ---')

    t1 = time.time()
    cfg = IPOConfig(**params)
    bt = IPOStrategyBacktester(cfg, preloaded_data=preloaded)
    result = bt.run()
    elapsed = time.time() - t1

    trades = result.total_trades
    if trades == 0:
        print(f'  [{label}] {elapsed:.1f}s | NO TRADES')
        row = dict(label=label, trades=0, win_rate=0, profit_factor=0,
                   expectancy=0, total_return=0, median_return=0,
                   avg_hold_days=0, filters=filter_desc)
    else:
        wr = result.win_rate
        returns = [t.return_pct for t in result.trades]
        pf = result.profit_factor
        exp = result.expectancy
        total_ret = sum(returns)
        sorted_r = sorted(returns)
        median_r = sorted_r[len(sorted_r)//2]
        avg_hold = result.avg_hold_days

        marker = ''
        if wr >= 65:
            marker = ' *** 65%+ ***'
            hit_65 += 1
        elif wr >= 60:
            marker = ' ** 60%+ **'

        print(f'  [{label}] {elapsed:.1f}s | T={trades} W={wr:.1f}% PF={pf:.2f} '
              f'Exp={exp:.2f} Med={median_r:.1f}% AvgHold={avg_hold:.1f}d{marker}')

        row = dict(label=label, trades=trades, win_rate=round(wr, 2),
                   profit_factor=round(pf, 2), expectancy=round(exp, 2),
                   total_return=round(total_ret, 2), median_return=round(median_r, 2),
                   avg_hold_days=round(avg_hold, 1), filters=filter_desc)

    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    done_count += 1
    sys.stdout.flush()

print(f'\n=== COMPLETE: {done_count}/{total} configs ===')
print(f'Configs hitting 65%+ WR: {hit_65}')
print(f'Results saved to: {OUTPUT_CSV}')
