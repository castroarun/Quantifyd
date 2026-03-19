"""
COVID Crash Recovery Table - Detailed Stock-Level Analysis
===========================================================

Runs PS20 baseline over 2019-2021 and tracks:
1. Pre-crash holdings (Feb 2020)
2. Each exit during crash with stock name, date, reason
3. NIFTYBEES deployment events
4. Each new entry during recovery with stock name, date
5. Weekly snapshot of holdings until full 20 stocks restored

Output: Printed table + covid_recovery_detail.csv
"""

import csv
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mq_backtest_engine import MQBacktestEngine
from services.mq_portfolio import MQBacktestConfig

import pandas as pd

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'covid_recovery_detail.csv')


def run_and_extract():
    """Run PS20 backtest and extract COVID crash details."""
    print('Loading data...')
    t0 = time.time()
    config = MQBacktestConfig(
        start_date='2019-01-01',
        end_date='2021-12-31',
        initial_capital=10_000_000,
        portfolio_size=20,
        equity_allocation_pct=0.95,
        debt_reserve_pct=0.05,
        hard_stop_loss=0.50,
        rebalance_ath_drawdown=0.20,
        daily_ath_drawdown_exit=True,
        immediate_replacement=True,
        idle_cash_to_nifty_etf=True,
        idle_cash_to_debt=True,
    )

    universe, price_data = MQBacktestEngine.preload_data(config)
    print(f'Data loaded in {time.time()-t0:.0f}s ({len(price_data)} stocks)\n')

    print('Running backtest...')
    t1 = time.time()
    engine = MQBacktestEngine(config,
                               preloaded_universe=universe,
                               preloaded_price_data=price_data)
    result = engine.run()
    print(f'Backtest done in {time.time()-t1:.0f}s\n')

    return result, engine


def build_events_timeline(result):
    """Build a timeline of all events (entries and exits) from trade log."""
    events = []

    for t in result.trade_log:
        # Entry event
        events.append({
            'date': t.entry_date,
            'type': 'ENTRY',
            'symbol': t.symbol,
            'price': t.entry_price,
            'reason': '',
            'return_pct': 0,
        })
        # Exit event
        events.append({
            'date': t.exit_date,
            'type': 'EXIT',
            'symbol': t.symbol,
            'price': t.exit_price,
            'reason': t.exit_reason,
            'return_pct': t.return_pct * 100,
        })

    events.sort(key=lambda x: x['date'])
    return events


def build_holdings_snapshots(events, start_date='2020-01-01', end_date='2021-06-30'):
    """Build weekly snapshots of which stocks are held."""
    holdings = set()
    event_idx = 0

    # Fast-forward to populate holdings before start_date
    while event_idx < len(events) and events[event_idx]['date'] < datetime.strptime(start_date, '%Y-%m-%d'):
        e = events[event_idx]
        if e['type'] == 'ENTRY':
            holdings.add(e['symbol'])
        elif e['type'] == 'EXIT':
            holdings.discard(e['symbol'])
        event_idx += 1

    # Now build weekly snapshots
    snapshots = []
    dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')

    for snap_date in dates:
        snap_dt = snap_date.to_pydatetime().replace(hour=0, minute=0, second=0)
        # Process events up to this date
        while event_idx < len(events) and events[event_idx]['date'] <= snap_dt:
            e = events[event_idx]
            if e['type'] == 'ENTRY':
                holdings.add(e['symbol'])
            elif e['type'] == 'EXIT':
                holdings.discard(e['symbol'])
            event_idx += 1

        snapshots.append({
            'date': snap_date.strftime('%Y-%m-%d'),
            'count': len(holdings),
            'stocks': sorted(holdings),
        })

    return snapshots


def print_crash_events(events):
    """Print detailed crash and recovery events."""

    # COVID crash period: Feb 2020 - Jun 2020
    crash_start = datetime(2020, 2, 1)
    recovery_end = datetime(2021, 6, 30)

    print('=' * 100)
    print('COVID CRASH TIMELINE - Exits')
    print('=' * 100)
    print(f'{"Date":<14} {"Type":<6} {"Symbol":<16} {"Price":>10} {"Reason":<30} {"Return%":>10}')
    print('-' * 100)

    exits_during_crash = [e for e in events
                          if e['type'] == 'EXIT'
                          and datetime(2020, 2, 15) <= e['date'] <= datetime(2020, 4, 30)]
    for e in exits_during_crash:
        print(f'{e["date"].strftime("%Y-%m-%d"):<14} {e["type"]:<6} {e["symbol"]:<16} '
              f'{e["price"]:>10.2f} {str(e["reason"]):<30} {e["return_pct"]:>+10.1f}%')

    print(f'\nTotal exits during crash: {len(exits_during_crash)}')

    print('\n' + '=' * 100)
    print('COVID RECOVERY TIMELINE - New Entries')
    print('=' * 100)
    print(f'{"Date":<14} {"Type":<6} {"Symbol":<16} {"Price":>10}')
    print('-' * 100)

    entries_during_recovery = [e for e in events
                                if e['type'] == 'ENTRY'
                                and datetime(2020, 3, 1) <= e['date'] <= datetime(2021, 6, 30)]
    for e in entries_during_recovery:
        print(f'{e["date"].strftime("%Y-%m-%d"):<14} {e["type"]:<6} {e["symbol"]:<16} '
              f'{e["price"]:>10.2f}')

    print(f'\nTotal entries during recovery: {len(entries_during_recovery)}')


def print_holdings_snapshots(snapshots):
    """Print weekly holdings snapshots from crash through recovery."""
    print('\n' + '=' * 130)
    print('WEEKLY HOLDINGS SNAPSHOTS - COVID Crash & Recovery')
    print('=' * 130)

    # Find the minimum holdings count (trough)
    min_count = 20
    min_idx = 0
    for i, snap in enumerate(snapshots):
        if snap['count'] < min_count:
            min_count = snap['count']
            min_idx = i

    # Start printing 2 weeks before first drop below 20
    crash_started = False
    hit_trough = False
    recovered_to_18 = False
    weeks_after_recovery = 0

    for i, snap in enumerate(snapshots):
        count = snap['count']

        # Start 1 week before crash
        if count < 20 and not crash_started:
            crash_started = True
        if not crash_started:
            continue

        stocks = ', '.join(snap['stocks'])
        marker = ''

        if count == min_count and not hit_trough:
            hit_trough = True
            marker = ' <<< TROUGH'

        if hit_trough and count >= 18 and not recovered_to_18:
            recovered_to_18 = True
            marker = f' <<< RECOVERED TO {count}/20'

        if recovered_to_18:
            weeks_after_recovery += 1

        print(f'{snap["date"]}  [{count:>2}/20]  {stocks}{marker}')

        # Show 2 more weeks after full recovery, or continue to end
        if recovered_to_18 and weeks_after_recovery > 2:
            break

    if not recovered_to_18:
        print(f'\n*** Never recovered to 18+ stocks by {snapshots[-1]["date"]} ***')
        final = snapshots[-1]
        print(f'Final: [{final["count"]}/20] {", ".join(final["stocks"])}')


def save_csv(events, snapshots):
    """Save events and snapshots to CSV."""
    fieldnames = ['date', 'type', 'symbol', 'price', 'reason', 'return_pct', 'holdings_count', 'holdings']

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write crash events
        crash_events = [e for e in events
                        if datetime(2020, 2, 1) <= e['date'] <= datetime(2021, 6, 30)]
        for e in crash_events:
            writer.writerow({
                'date': e['date'].strftime('%Y-%m-%d'),
                'type': e['type'],
                'symbol': e['symbol'],
                'price': round(e['price'], 2),
                'reason': str(e['reason']),
                'return_pct': round(e['return_pct'], 2),
                'holdings_count': '',
                'holdings': '',
            })

        # Write snapshots
        for snap in snapshots:
            writer.writerow({
                'date': snap['date'],
                'type': 'SNAPSHOT',
                'symbol': '',
                'price': '',
                'reason': '',
                'return_pct': '',
                'holdings_count': snap['count'],
                'holdings': '|'.join(snap['stocks']),
            })


if __name__ == '__main__':
    result, engine = run_and_extract()

    events = build_events_timeline(result)
    print_crash_events(events)

    snapshots = build_holdings_snapshots(events, '2020-01-01', '2021-06-30')
    print_holdings_snapshots(snapshots)

    # Also print equity curve around crash
    eq = pd.Series(result.daily_equity, dtype=float)
    eq.index = pd.to_datetime(eq.index)
    eq = eq.sort_index()

    crash_eq = eq['2020-01':'2020-12']
    peak = crash_eq.max()
    trough = crash_eq.min()
    peak_date = crash_eq.idxmax()
    trough_date = crash_eq.idxmin()

    print(f'\n{"=" * 80}')
    print('EQUITY CURVE SUMMARY')
    print(f'{"=" * 80}')
    print(f'Pre-crash peak: Rs {peak:,.0f} on {peak_date.strftime("%Y-%m-%d")}')
    print(f'Crash trough:   Rs {trough:,.0f} on {trough_date.strftime("%Y-%m-%d")}')
    print(f'Drawdown:       {(trough - peak) / peak * 100:.1f}%')

    # Find recovery date
    recovery_eq = eq[trough_date:]
    recovered = recovery_eq[recovery_eq >= peak]
    if len(recovered) > 0:
        recovery_date = recovered.index[0]
        days_to_recover = (recovery_date - trough_date).days
        print(f'Recovery date:  {recovery_date.strftime("%Y-%m-%d")} ({days_to_recover} days)')
    else:
        print('Recovery:       Did not recover within backtest period')

    save_csv(events, snapshots)
    print(f'\nDetailed data saved to {OUTPUT_CSV}')
