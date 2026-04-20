"""ORB backtest side-by-side: Futures (lot-sized) vs Intraday MIS equity (leveraged).

Assumptions:
  - Futures intraday margin ~20% (SPAN+Exposure avg) -> 5x leverage
  - MIS equity intraday leverage 5x for large caps
  - Capital needed = max single-day total futures margin deployed (sum of all
    that day's trade margins -- conservative upper bound assuming concurrent)
  - MIS mode uses the SAME capital
  - Max 5 concurrent trades (per ORB config) -> allocation per trade = capital / 5
  - MIS qty per trade = (capital / 5) * 5 / entry = capital / entry
    i.e. full allocation leverages to the stock price at entry
"""
import os, sys
sys.path.insert(0, '/home/arun/quantifyd')
os.chdir('/home/arun/quantifyd')
import sqlite3
from collections import defaultdict
from services.data_manager import FNO_LOT_SIZES

FUT_MARGIN_PCT = 0.20  # Futures intraday margin ~20%
MIS_LEVERAGE = 5.0
MAX_CONCURRENT = 5

UNIVERSE = ['ADANIENT','TATASTEEL','BEL','VEDL','BPCL','M&M','BAJFINANCE',
            'TRENT','HAL','IRCTC','GRASIM','GODREJPROP','RELIANCE','AXISBANK','APOLLOHOSP']
LOT = {s: FNO_LOT_SIZES.get(s, 350) for s in UNIVERSE}

c = sqlite3.connect('backtest_data/orb_backtest.db')
c.row_factory = sqlite3.Row
trades = c.execute('''SELECT run_date, instrument, direction, entry_time, entry_price,
                             exit_time, exit_price, exit_reason
                      FROM orb_backtest_signals
                      WHERE signal_type='TAKEN'
                      ORDER BY run_date, entry_time, instrument''').fetchall()

# Compute per-trade futures margin + peak daily margin
daily_margin = defaultdict(float)
by_trade = []
for t in trades:
    sym = t['instrument']
    lot = LOT.get(sym, 350)
    entry = t['entry_price']
    if entry is None or t['exit_price'] is None:
        continue
    notional = entry * lot
    fut_margin = FUT_MARGIN_PCT * notional
    daily_margin[t['run_date']] += fut_margin
    by_trade.append({
        'date': t['run_date'], 'sym': sym, 'dir': t['direction'], 'lot': lot,
        'entry': entry, 'exit': t['exit_price'], 'reason': t['exit_reason'],
        'entry_t': t['entry_time'], 'exit_t': t['exit_time'],
        'notional': notional, 'fut_margin': fut_margin,
    })

# Peak capital (max day's total margin if all trades concurrent)
peak_daily_margin = max(daily_margin.values())
peak_day = max(daily_margin, key=daily_margin.get)
avg_daily_margin = sum(daily_margin.values()) / len(daily_margin)

# Round up to a clean number
import math
CAPITAL = math.ceil(peak_daily_margin / 10000) * 10000  # round to nearest 10k

print('CAPITAL ASSUMPTION')
print(f'  Peak single-day futures margin:  Rs {peak_daily_margin:,.0f}  (on {peak_day})')
print(f'  Avg daily futures margin:        Rs {avg_daily_margin:,.0f}')
print(f'  Working capital used for both:   Rs {CAPITAL:,.0f}')
print()

# Compute futures P&L (unchanged) and MIS P&L (leveraged qty)
PER_TRADE_CAPITAL = CAPITAL / MAX_CONCURRENT
MIS_BUYING_POWER_PER_TRADE = PER_TRADE_CAPITAL * MIS_LEVERAGE  # = CAPITAL
# i.e. each trade can deploy the full capital as MIS buying power

for t in by_trade:
    entry = t['entry']
    exit_ = t['exit']
    sign = 1 if t['dir'] == 'LONG' else -1
    # Futures
    t['fut_pnl'] = sign * (exit_ - entry) * t['lot']
    # MIS equity qty = buying_power / entry (integer), take floor
    mis_qty = int(MIS_BUYING_POWER_PER_TRADE / entry)
    t['mis_qty'] = mis_qty
    t['mis_pnl'] = sign * (exit_ - entry) * mis_qty

# Aggregate
def summary(key_pnl):
    pnls = [t[key_pnl] for t in by_trade]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gw = sum(wins)
    gl = abs(sum(losses))
    total = sum(pnls)
    # Equity curve by day for drawdown
    day_pnl = defaultdict(float)
    for t in by_trade:
        day_pnl[t['date']] += t[key_pnl]
    days = sorted(day_pnl.keys())
    eq = 0.0; peak = 0.0; max_dd = 0.0
    for d in days:
        eq += day_pnl[d]
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    return {
        'trades': len(pnls),
        'wins': len(wins), 'losses': len(losses),
        'win_rate': len(wins)/len(pnls)*100 if pnls else 0,
        'pf': gw/gl if gl > 0 else float('inf'),
        'net': total,
        'avg_trade': total/len(pnls) if pnls else 0,
        'avg_win': gw/len(wins) if wins else 0,
        'avg_loss': -gl/len(losses) if losses else 0,
        'max_win': max(pnls) if pnls else 0,
        'max_loss': min(pnls) if pnls else 0,
        'win_days': sum(1 for d in days if day_pnl[d] > 0),
        'loss_days': sum(1 for d in days if day_pnl[d] < 0),
        'best_day': max(days, key=lambda d: day_pnl[d]) if days else None,
        'best_day_pnl': max(day_pnl.values()) if day_pnl else 0,
        'worst_day': min(days, key=lambda d: day_pnl[d]) if days else None,
        'worst_day_pnl': min(day_pnl.values()) if day_pnl else 0,
        'max_dd': max_dd,
        'peak': peak,
        'final': eq,
    }

fut = summary('fut_pnl')
mis = summary('mis_pnl')

# Side-by-side report
print('COMPARISON TABLE  (same trades, same capital)')
print()
w_label = 26; w_fut = 18; w_mis = 18
print(f'{"":<{w_label}} {"Futures":>{w_fut}} {"Intraday MIS":>{w_mis}}')
print('-' * (w_label + w_fut + w_mis + 2))

def row(label, fut_val, mis_val, fmt='{:+,.0f}'):
    f_s = fmt.format(fut_val) if isinstance(fut_val, (int, float)) else str(fut_val)
    m_s = fmt.format(mis_val) if isinstance(mis_val, (int, float)) else str(mis_val)
    print(f'{label:<{w_label}} {f_s:>{w_fut}} {m_s:>{w_mis}}')

row('Trades taken', fut['trades'], mis['trades'], '{:d}')
row('Win rate (%)', fut['win_rate'], mis['win_rate'], '{:.1f}')
row('Profit factor', fut['pf'], mis['pf'], '{:.2f}')
row('Net P&L (Rs)', fut['net'], mis['net'])
row('Avg per trade (Rs)', fut['avg_trade'], mis['avg_trade'])
row('Avg win (Rs)', fut['avg_win'], mis['avg_win'])
row('Avg loss (Rs)', fut['avg_loss'], mis['avg_loss'])
row('Largest win (Rs)', fut['max_win'], mis['max_win'])
row('Largest loss (Rs)', fut['max_loss'], mis['max_loss'])
row('Winning days', fut['win_days'], mis['win_days'], '{:d}')
row('Losing days', fut['loss_days'], mis['loss_days'], '{:d}')
row('Best day (Rs)', fut['best_day_pnl'], mis['best_day_pnl'])
row('Worst day (Rs)', fut['worst_day_pnl'], mis['worst_day_pnl'])
row('Max drawdown (Rs)', fut['max_dd'], mis['max_dd'])
row('Peak equity (Rs)', fut['peak'], mis['peak'])
row('Final equity (Rs)', fut['final'], mis['final'])
row('Return on capital (%)', fut['final']/CAPITAL*100, mis['final']/CAPITAL*100, '{:.1f}')
row('MaxDD on capital (%)', fut['max_dd']/CAPITAL*100, mis['max_dd']/CAPITAL*100, '{:.1f}')
row('Calmar (ret/DD)', fut['final']/fut['max_dd'] if fut['max_dd'] else 0,
    mis['final']/mis['max_dd'] if mis['max_dd'] else 0, '{:.2f}')

# Position-size comparison (sample)
print()
print('POSITION SIZE SAMPLE (5 random trades)')
print(f'Working capital: Rs {CAPITAL:,.0f} · per-trade allocation Rs {PER_TRADE_CAPITAL:,.0f} at {MIS_LEVERAGE:.0f}x -> Rs {MIS_BUYING_POWER_PER_TRADE:,.0f} buying power')
print()
print(f'{"Stock":12s} {"Lot":>5s} {"Entry":>10s} {"Fut notional":>14s} {"Fut margin":>12s} {"MIS qty":>9s} {"MIS notional":>14s}')
import random
sample = random.sample(by_trade, min(8, len(by_trade)))
sample.sort(key=lambda t: t['entry'])
for t in sample:
    mis_not = t['mis_qty'] * t['entry']
    print(f'{t["sym"]:12s} {t["lot"]:>5d} {t["entry"]:>10.2f} {t["notional"]:>14,.0f} {t["fut_margin"]:>12,.0f} {t["mis_qty"]:>9d} {mis_not:>14,.0f}')
