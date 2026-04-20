"""Re-run the ORB backtest numbers as if each trade were taken as FUTURES with
the instrument's F&O lot size. Prints aggregate stats + drawdown profile.

Reads from backtest_data/orb_backtest.db (already populated by the daily
backfill). No recomputation of trade logic — same entries, exits, exit
reasons; only P&L is re-sized to lot_size instead of cash-qty."""
import os, sys
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
import sqlite3
from datetime import date
from services.data_manager import FNO_LOT_SIZES

c = sqlite3.connect('backtest_data/orb_backtest.db')
c.row_factory = sqlite3.Row

# ORB universe
UNIVERSE = ['ADANIENT','TATASTEEL','BEL','VEDL','BPCL','M&M','BAJFINANCE','TRENT','HAL','IRCTC','GRASIM','GODREJPROP','RELIANCE','AXISBANK','APOLLOHOSP']

# Confirm lot sizes for all — M&M is missing from the dict, use 350 (typical)
LOT = {s: FNO_LOT_SIZES.get(s, 350) for s in UNIVERSE}

print('Universe lot sizes:')
for s in UNIVERSE:
    print(f'  {s:12s}  {LOT[s]}')
print()

trades = c.execute('''SELECT run_date, instrument, direction, entry_time, entry_price,
                             exit_time, exit_price, exit_reason, or_high, or_low
                      FROM orb_backtest_signals
                      WHERE signal_type=\'TAKEN\'
                      ORDER BY run_date, entry_time, instrument''').fetchall()

if not trades:
    print('No taken trades in DB — run backfill first.')
    sys.exit(0)

# Compute futures P&L per trade
annotated = []
for t in trades:
    sym = t['instrument']
    lot = LOT.get(sym, 350)
    entry = t['entry_price']
    exit_ = t['exit_price']
    direction = t['direction']
    if entry is None or exit_ is None:
        continue
    if direction == 'LONG':
        pnl = (exit_ - entry) * lot
    else:
        pnl = (entry - exit_) * lot
    annotated.append({
        'date': t['run_date'], 'sym': sym, 'dir': direction, 'lot': lot,
        'entry': entry, 'exit': exit_, 'reason': t['exit_reason'],
        'entry_t': t['entry_time'], 'exit_t': t['exit_time'],
        'pnl': pnl,
    })

total_pnl = sum(a['pnl'] for a in annotated)
wins = [a for a in annotated if a['pnl'] > 0]
losses = [a for a in annotated if a['pnl'] <= 0]
gross_win = sum(a['pnl'] for a in wins)
gross_loss = abs(sum(a['pnl'] for a in losses))
pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

# Per-day aggregation
from collections import defaultdict
day_pnl = defaultdict(float)
day_count = defaultdict(int)
for a in annotated:
    day_pnl[a['date']] += a['pnl']
    day_count[a['date']] += 1

days = sorted(day_pnl.keys())
win_days = sum(1 for d in days if day_pnl[d] > 0)
loss_days = sum(1 for d in days if day_pnl[d] < 0)
flat_days = sum(1 for d in days if day_pnl[d] == 0)
best_day = max(days, key=lambda d: day_pnl[d])
worst_day = min(days, key=lambda d: day_pnl[d])

# Equity curve + drawdown
equity = 0.0
peak = 0.0
max_dd_abs = 0.0
max_dd_day = None
max_dd_peak_day = None
current_peak_day = None
curve = []
for d in days:
    equity += day_pnl[d]
    if equity > peak:
        peak = equity
        current_peak_day = d
    dd = peak - equity
    if dd > max_dd_abs:
        max_dd_abs = dd
        max_dd_day = d
        max_dd_peak_day = current_peak_day
    curve.append({'date': d, 'equity': equity, 'peak': peak, 'dd': dd})

# Longest losing / winning streak
cur_streak = 0
best_win_streak = 0
cur_loss_streak = 0
worst_loss_streak = 0
for d in days:
    if day_pnl[d] > 0:
        cur_streak += 1
        best_win_streak = max(best_win_streak, cur_streak)
        cur_loss_streak = 0
    elif day_pnl[d] < 0:
        cur_loss_streak += 1
        worst_loss_streak = max(worst_loss_streak, cur_loss_streak)
        cur_streak = 0
    else:
        cur_streak = 0
        cur_loss_streak = 0

print('=' * 70)
print(f'FUTURES BACKTEST  ({len(days)} trading days · {days[0]} to {days[-1]})')
print('=' * 70)
print(f'Trades taken:      {len(annotated)}')
print(f'Win rate:          {len(wins)/len(annotated)*100:.1f}%  ({len(wins)} wins / {len(losses)} losses)')
print(f'Profit factor:     {pf:.2f}')
print(f'Net P&L:           Rs {total_pnl:+,.0f}')
print(f'Avg per trade:     Rs {total_pnl/len(annotated):+,.0f}')
print(f'Avg win:           Rs {gross_win/len(wins):+,.0f}')
print(f'Avg loss:          Rs {-gross_loss/max(len(losses),1):+,.0f}')
_max_pnl = max(a['pnl'] for a in annotated)
_min_pnl = min(a['pnl'] for a in annotated)
print(f'Largest win:       Rs {_max_pnl:+,.0f}')
print(f'Largest loss:      Rs {_min_pnl:+,.0f}')
print()
print(f'Win/loss/flat days:  {win_days}/{loss_days}/{flat_days}')
print(f'Best day:          {best_day}  Rs {day_pnl[best_day]:+,.0f}  ({day_count[best_day]} trades)')
print(f'Worst day:         {worst_day}  Rs {day_pnl[worst_day]:+,.0f}  ({day_count[worst_day]} trades)')
print(f'Longest win streak:  {best_win_streak} days')
print(f'Longest loss streak: {worst_loss_streak} days')

print()
print('=' * 70)
print('DRAWDOWN')
print('=' * 70)
print(f'Max drawdown:      Rs {max_dd_abs:,.0f}')
_peak_equity = next(x['peak'] for x in curve if x['date'] == max_dd_peak_day)
_trough_equity = next(x['equity'] for x in curve if x['date'] == max_dd_day)
print(f'  - from peak on:  {max_dd_peak_day}  (equity Rs {_peak_equity:,.0f})')
print(f'  - to trough on:  {max_dd_day}  (equity Rs {_trough_equity:,.0f})')
print(f'  - as % of peak equity: {max_dd_abs/peak*100:.1f}%')
print(f'Final equity:      Rs {equity:+,.0f}')
print(f'Peak equity:       Rs {peak:+,.0f}')
print()
print('EQUITY CURVE (every 5th day):')
for i in range(0, len(curve), 5):
    x = curve[i]
    bar = '#' * max(0, int(x['equity'] / max(1, peak) * 40))
    dd_bar = '.' * max(0, int(x['dd'] / max(1, max_dd_abs) * 10))
    print(f"  {x['date']}  equity Rs {x['equity']:+10,.0f}  dd Rs {x['dd']:>8,.0f} {dd_bar}")

# Per-symbol breakdown
print()
print('=' * 70)
print('PER-SYMBOL')
print('=' * 70)
sym_pnl = defaultdict(float)
sym_trades = defaultdict(int)
sym_wins = defaultdict(int)
for a in annotated:
    sym_pnl[a['sym']] += a['pnl']
    sym_trades[a['sym']] += 1
    if a['pnl'] > 0: sym_wins[a['sym']] += 1
print('Symbol        Lot  Trades   Win%   Net P&L (Rs)')
for s in sorted(UNIVERSE, key=lambda s: sym_pnl[s], reverse=True):
    if sym_trades[s] == 0:
        continue
    wr = sym_wins[s] / sym_trades[s] * 100
    print(f'{s:12s} {LOT[s]:>5d} {sym_trades[s]:>6d}  {wr:>4.1f}%  {sym_pnl[s]:>+13,.0f}')
