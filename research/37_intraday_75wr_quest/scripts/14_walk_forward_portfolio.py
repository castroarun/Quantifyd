"""Phase 1b — Combined-portfolio walk-forward.

Splits the Stage-13 trade history at 2025-09-30. Computes equity-curve
stats for the train half and the test half independently. Goal: confirm
the combined-portfolio behaviour is stable across the train/test split,
not just per-system.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))


def stats(trades: pd.DataFrame, daily: pd.DataFrame, initial_capital: float) -> dict:
    if trades.empty or daily.empty:
        return {}
    total = len(trades)
    wins = (trades['net_ret_pct'] > 0).sum()
    wr = wins / total
    gp = trades.loc[trades['net_ret_pct'] > 0, 'net_ret_pct'].sum()
    gl = -trades.loc[trades['net_ret_pct'] <= 0, 'net_ret_pct'].sum()
    pf = gp / gl if gl > 0 else float('inf')
    final_nav = daily['nav'].iloc[-1]
    total_return = (final_nav / initial_capital - 1) * 100
    n_days = len(daily)
    n_years = n_days / 252
    cagr = ((final_nav / initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100
    daily_rets = daily['daily_return_pct'].dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    nav_series = daily['nav']
    peak = nav_series.cummax()
    dd = (peak - nav_series) / peak * 100
    max_dd = dd.max() if len(dd) else 0
    calmar = cagr / max_dd if max_dd > 0 else float('inf')
    return dict(
        n=total, wr=wr, pf=pf,
        total_return=total_return, cagr=cagr, sharpe=sharpe,
        max_dd=max_dd, calmar=calmar,
        final_nav=final_nav, n_days=n_days,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='rs3k', help='Output suffix to read')
    ap.add_argument('--split-date', default='2025-09-30')
    ap.add_argument('--initial-capital', type=float, default=10_00_000)
    args = ap.parse_args()

    suffix = f'_{args.label}' if args.label else ''
    trades = pd.read_csv(os.path.join(RESULTS_DIR, f'13_portfolio_trades{suffix}.csv'))
    daily = pd.read_csv(os.path.join(RESULTS_DIR, f'13_portfolio_daily_nav{suffix}.csv'))
    print(f'Loaded {len(trades)} trades, {len(daily)} daily NAV rows')

    trades['entry_dt'] = pd.to_datetime(trades['entry_time'])
    daily['date_dt'] = pd.to_datetime(daily['date'])

    split = pd.Timestamp(args.split_date)
    train_t = trades[trades['entry_dt'] <= split].copy()
    test_t = trades[trades['entry_dt'] > split].copy()

    # rebuild daily NAV for each half independently
    train_d = daily[daily['date_dt'] <= split].copy().reset_index(drop=True)
    test_d_raw = daily[daily['date_dt'] > split].copy().reset_index(drop=True)
    if not test_d_raw.empty:
        # restart NAV from initial_capital for test half (treat as fresh deployment)
        starting_nav = args.initial_capital
        test_d_raw['pnl'] = test_d_raw['pnl'].astype(float)
        test_d_raw['nav'] = starting_nav + test_d_raw['pnl'].cumsum()
        test_d_raw['daily_return_pct'] = test_d_raw['pnl'] / test_d_raw['nav'].shift(1).fillna(starting_nav) * 100

    train_stats = stats(train_t, train_d, args.initial_capital)
    test_stats = stats(test_t, test_d_raw, args.initial_capital)

    print()
    print(f'=== Combined-portfolio walk-forward (label={args.label}) ===')
    print(f'Split: {args.split_date}  | Initial capital: Rs.{args.initial_capital:,.0f}')
    print()
    print(f'{"":<20} {"TRAIN":>15} {"TEST":>15}  drift')
    fmt = lambda k, train_v, test_v: print(f'{k:<20} {train_v:>15} {test_v:>15}')
    fmt('Period sessions', f'{train_stats.get("n_days",0)}', f'{test_stats.get("n_days",0)}')
    fmt('Trades', f'{train_stats.get("n",0)}', f'{test_stats.get("n",0)}')
    fmt('Win rate', f'{train_stats.get("wr",0):.1%}',
        f'{test_stats.get("wr",0):.1%}')
    drift_wr = (train_stats.get('wr', 0) - test_stats.get('wr', 0)) * 100
    print(f'{"  WR drift":<20} {"":>31}  {drift_wr:+.2f} pp')
    fmt('Profit factor', f'{train_stats.get("pf",0):.2f}', f'{test_stats.get("pf",0):.2f}')
    fmt('Total return', f'{train_stats.get("total_return",0):+.2f}%', f'{test_stats.get("total_return",0):+.2f}%')
    fmt('CAGR', f'{train_stats.get("cagr",0):+.2f}%', f'{test_stats.get("cagr",0):+.2f}%')
    fmt('Sharpe', f'{train_stats.get("sharpe",0):.2f}', f'{test_stats.get("sharpe",0):.2f}')
    fmt('Max drawdown', f'{train_stats.get("max_dd",0):.2f}%', f'{test_stats.get("max_dd",0):.2f}%')
    fmt('Calmar', f'{train_stats.get("calmar",0):.2f}', f'{test_stats.get("calmar",0):.2f}')

    # write to file
    out = os.path.join(RESULTS_DIR, f'14_walk_forward_portfolio{suffix}.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'Combined-portfolio walk-forward (label={args.label})\n')
        f.write(f'Split: {args.split_date}\n\n')
        for k in ('n', 'wr', 'pf', 'total_return', 'cagr', 'sharpe', 'max_dd', 'calmar'):
            f.write(f'  {k}: TRAIN={train_stats.get(k):.4f} | TEST={test_stats.get(k):.4f}\n')
    print(f'\nWrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
