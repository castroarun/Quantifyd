"""Stage 13 — combined 3-system portfolio backtest.

Realistic account-NAV simulation of all three walk-forward-validated systems
running together on the same 2-year history. Models capital allocation,
concurrency limits, and per-trade sizing.

Inputs:
    Initial capital: Rs.10,00,000 (10L)
    Risk per trade: 0.5% of current NAV (= Rs.5,000 at start)
    Max concurrent positions: configurable (default 5)
    Costs: 0.05% per side (round-trip 0.10%) — covers brokerage + STT + slippage

Three systems run in parallel:
    1. SHORT — Diamond Short Variant B (Stage 8/9): RSI<40 + NIFTY first-30m<0
    2. LONG-MR — Late-day mean-reversion top10 (Stage 11c): -2% drop + NIFTY not crashing
    3. LONG-TC — Trend-continuation pullback rank4 (Stage 11b): gap-up + pullback + NIFTY strong

Output:
    results/13_portfolio_daily_nav.csv         # one row per session
    results/13_portfolio_trades.csv            # all executed trades
    results/13_portfolio_summary.txt           # headline stats

Process:
    - Generate signals per stock per system (vectorised via existing scripts).
    - Merge into one chronologically-ordered trade-event stream.
    - Walk session-by-session, bar-by-bar, applying concurrency cap and
      sizing rules. Skip signals when capacity is full.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _engine import load_5min, enrich  # type: ignore

mod_8 = importlib.import_module('08_diamond_short_with_nifty')
mod_10 = importlib.import_module('10_diamond_long_with_nifty')

# Load the LONG-MR signal builder from Stage 11c
import importlib.util
SCRIPT_11C = os.path.join(os.path.dirname(os.path.abspath(__file__)), '11c_long_late_reversal.py')
spec = importlib.util.spec_from_file_location('mod_11c', SCRIPT_11C)
mod_11c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_11c)

# Load the LONG-TC signal builder from Stage 11b
SCRIPT_11B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '11b_long_trend_pullback.py')
spec = importlib.util.spec_from_file_location('mod_11b', SCRIPT_11B)
mod_11b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_11b)


RESULTS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results',
))


# -------------------------------------------------------------------------
# 1. signal generation per system
# -------------------------------------------------------------------------

def build_signals_short(stocks: list[str], cache: dict, nifty: pd.DataFrame,
                        variant: str = 'volume',
                        tp_pct: float = 0.5, sl_pct: float = 1.5,
                        hold_bars: int = 60) -> pd.DataFrame:
    """Stage 8 short signals. Returns trade-event DataFrame with columns:
       symbol, system, entry_time, sl_pct, tp_pct, hold_bars, direction.
    """
    if variant == 'volume':
        rsi_threshold = 40
        nifty_filter = 'b3_change_neg'
    else:  # strict
        rsi_threshold = 35
        nifty_filter = 'b3_change_neg_strong'

    diamond_short = mod_8.diamond_short
    filter_by_nifty = mod_8.filter_by_nifty

    rows = []
    for sym in stocks:
        if sym not in cache:
            continue
        df = cache[sym]
        sig = diamond_short(df, rsi_threshold=rsi_threshold, entry_bar=6)
        sig = filter_by_nifty(sig, df, nifty, nifty_filter)
        sig_idx = np.where(sig.values)[0]
        for i in sig_idx:
            entry_i = i + 1
            if entry_i >= len(df):
                continue
            rows.append(dict(
                symbol=sym, system='SHORT',
                entry_time=df.index[entry_i],
                entry_bar_idx=entry_i,
                tp_pct=tp_pct, sl_pct=sl_pct, hold_bars=hold_bars,
                direction='short',
            ))
    return pd.DataFrame(rows)


def build_signals_long_mr(stocks: list[str], cache: dict, nifty: pd.DataFrame,
                          tp_pct: float = 0.5, sl_pct: float = 1.5,
                          hold_bars: int = 60) -> pd.DataFrame:
    """Stage 11c top10 signals — drop=-2.0, window 24-48, rsi 28→35, NIFTY 'b3_not_crashing'."""
    rows = []
    for sym in stocks:
        if sym not in cache:
            continue
        df = cache[sym]
        sig = mod_11c.late_reversal_signal(
            df,
            drop_pct=-2.0,
            entry_window_start=24,
            entry_window_end=48,
            rsi_oversold=28,
            rsi_lift=35,
        )
        sig = mod_11c.filter_by_nifty_for_reversal(sig, df, nifty, 'b3_not_crashing')
        sig_idx = np.where(sig.values)[0]
        for i in sig_idx:
            entry_i = i + 1
            if entry_i >= len(df):
                continue
            rows.append(dict(
                symbol=sym, system='LONG_MR',
                entry_time=df.index[entry_i],
                entry_bar_idx=entry_i,
                tp_pct=tp_pct, sl_pct=sl_pct, hold_bars=hold_bars,
                direction='long',
            ))
    return pd.DataFrame(rows)


def build_signals_long_tc(stocks: list[str], cache: dict, nifty: pd.DataFrame,
                          tp_pct: float = 0.5, sl_pct: float = 1.5,
                          hold_bars: int = 60) -> pd.DataFrame:
    """Stage 11b rank4 signals — vwap_within_0p3, rsi 45, gap 0.5, NIFTY 'nifty_strong_both'."""
    rows = []
    for sym in stocks:
        if sym not in cache:
            continue
        df = cache[sym]
        sig = mod_11b.trend_pullback_signal(
            df,
            gap_min_pct=0.5,
            first_hour_strength_pct=0.5,
            pullback_mode='vwap_within_0p3',
            rsi_floor=45,
            bar_min=7,
            bar_max=15,
        )
        sig = mod_11b.filter_by_nifty_long(sig, df, nifty, 'nifty_strong_both')
        sig_idx = np.where(sig.values)[0]
        for i in sig_idx:
            entry_i = i + 1
            if entry_i >= len(df):
                continue
            rows.append(dict(
                symbol=sym, system='LONG_TC',
                entry_time=df.index[entry_i],
                entry_bar_idx=entry_i,
                tp_pct=tp_pct, sl_pct=sl_pct, hold_bars=hold_bars,
                direction='long',
            ))
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------
# 2. per-trade simulator (uses preloaded df bars to compute exit + PnL)
# -------------------------------------------------------------------------

def simulate_trade(df: pd.DataFrame, entry_i: int, tp_pct: float, sl_pct: float,
                   hold_bars: int, direction: str, cost_per_side_pct: float = 0.05) -> dict:
    """Compute the exit time + return for a single trade. Returns dict with
    entry/exit time, prices, and net return after costs."""
    n = len(df)
    if entry_i >= n:
        return None
    open_a = df['open'].to_numpy()
    high_a = df['high'].to_numpy()
    low_a = df['low'].to_numpy()
    close_a = df['close'].to_numpy()
    sess_a = df['session'].to_numpy()

    entry_price = open_a[entry_i]
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    sess = sess_a[entry_i]
    sess_end_pos = pd.Series(np.arange(n), index=df['session'].values).groupby(level=0).max()
    try:
        sep = int(sess_end_pos.loc[sess])
    except KeyError:
        sep = n - 1
    max_exit_i = min(entry_i + hold_bars, sep, n - 1)

    is_long = direction == 'long'
    tp_mult = (1 + tp_pct / 100) if is_long else (1 - tp_pct / 100)
    sl_mult = (1 - sl_pct / 100) if is_long else (1 + sl_pct / 100)
    tp_price = entry_price * tp_mult
    sl_price = entry_price * sl_mult

    hi = high_a[entry_i:max_exit_i + 1]
    lo = low_a[entry_i:max_exit_i + 1]
    if is_long:
        sl_hit = lo <= sl_price
        tp_hit = hi >= tp_price
    else:
        sl_hit = hi >= sl_price
        tp_hit = lo <= tp_price

    sl_first = int(np.argmax(sl_hit)) if sl_hit.any() else -1
    tp_first = int(np.argmax(tp_hit)) if tp_hit.any() else -1

    if sl_first == -1 and tp_first == -1:
        offset = max_exit_i - entry_i
        exit_reason = 'EOD'
        exit_price = close_a[max_exit_i]
    elif sl_first == -1:
        offset = tp_first
        exit_reason = 'TP'
        exit_price = tp_price
    elif tp_first == -1:
        offset = sl_first
        exit_reason = 'SL'
        exit_price = sl_price
    elif sl_first <= tp_first:
        offset = sl_first
        exit_reason = 'SL'
        exit_price = sl_price
    else:
        offset = tp_first
        exit_reason = 'TP'
        exit_price = tp_price

    exit_i = entry_i + offset
    gross_ret = (exit_price - entry_price) / entry_price * 100
    if not is_long:
        gross_ret = -gross_ret
    net_ret = gross_ret - 2 * cost_per_side_pct  # round-trip cost

    return dict(
        entry_time=df.index[entry_i],
        exit_time=df.index[exit_i],
        entry_price=entry_price,
        exit_price=exit_price,
        bars_held=offset,
        gross_ret_pct=gross_ret,
        net_ret_pct=net_ret,
        exit_reason=exit_reason,
    )


# -------------------------------------------------------------------------
# 3. portfolio walker
# -------------------------------------------------------------------------

def run_portfolio(events: pd.DataFrame, caches: dict, initial_capital: float,
                  risk_per_trade_pct: float, max_concurrent: int,
                  cost_per_side_pct: float = 0.05,
                  risk_per_trade_rs: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk the trade-event stream chronologically, apply concurrency cap and
    sizing. Return (trades_df, daily_nav_df)."""
    events = events.sort_values('entry_time').reset_index(drop=True)
    nav = initial_capital
    open_trades = []  # list of dicts: symbol, entry_time, exit_time, capital_used, return
    closed_trades = []

    daily_records = []
    last_session = None

    for _, ev in events.iterrows():
        # Close any open trades that have exited before this event's entry time
        still_open = []
        for t in open_trades:
            if t['exit_time'] <= ev['entry_time']:
                pnl = t['capital_used'] * t['net_ret_pct'] / 100
                nav += pnl
                t['nav_after'] = nav
                closed_trades.append(t)
            else:
                still_open.append(t)
        open_trades = still_open

        # capacity check
        if len(open_trades) >= max_concurrent:
            continue

        # simulate the trade
        sym = ev['symbol']
        df = caches.get(sym)
        if df is None:
            continue
        result = simulate_trade(df, ev['entry_bar_idx'], ev['tp_pct'], ev['sl_pct'],
                                ev['hold_bars'], ev['direction'], cost_per_side_pct)
        if result is None:
            continue

        # sizing: fixed rupee risk if specified, else % of current NAV
        if risk_per_trade_rs is not None:
            risk_rs = risk_per_trade_rs
        else:
            risk_rs = nav * risk_per_trade_pct / 100
        capital_used = risk_rs / (ev['sl_pct'] / 100)  # capital where SL hit = risk_rs

        trade = dict(
            symbol=sym,
            system=ev['system'],
            direction=ev['direction'],
            entry_time=result['entry_time'],
            exit_time=result['exit_time'],
            entry_price=result['entry_price'],
            exit_price=result['exit_price'],
            bars_held=result['bars_held'],
            gross_ret_pct=result['gross_ret_pct'],
            net_ret_pct=result['net_ret_pct'],
            exit_reason=result['exit_reason'],
            capital_used=capital_used,
            risk_rs=risk_rs,
            nav_at_entry=nav,
        )
        open_trades.append(trade)

    # close any still-open at the end
    for t in open_trades:
        pnl = t['capital_used'] * t['net_ret_pct'] / 100
        nav += pnl
        t['nav_after'] = nav
        closed_trades.append(t)

    trades_df = pd.DataFrame(closed_trades)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    # daily NAV
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time']).dt.date
    daily_pnl = trades_df.groupby('exit_date').apply(
        lambda g: (g['capital_used'] * g['net_ret_pct'] / 100).sum()
    ).reset_index()
    daily_pnl.columns = ['date', 'pnl']
    daily_pnl['nav'] = initial_capital + daily_pnl['pnl'].cumsum()
    daily_pnl['cum_return_pct'] = (daily_pnl['nav'] / initial_capital - 1) * 100
    daily_pnl['daily_return_pct'] = daily_pnl['pnl'] / daily_pnl['nav'].shift(1).fillna(initial_capital) * 100

    return trades_df, daily_pnl


def summarise(trades_df: pd.DataFrame, daily_nav: pd.DataFrame, initial_capital: float) -> dict:
    if trades_df.empty:
        return {}
    total = len(trades_df)
    wins = (trades_df['net_ret_pct'] > 0).sum()
    wr = wins / total
    gp = trades_df.loc[trades_df['net_ret_pct'] > 0, 'net_ret_pct'].sum()
    gl = -trades_df.loc[trades_df['net_ret_pct'] <= 0, 'net_ret_pct'].sum()
    pf = gp / gl if gl > 0 else float('inf')
    final_nav = daily_nav['nav'].iloc[-1] if len(daily_nav) else initial_capital
    total_return = (final_nav / initial_capital - 1) * 100
    n_days = len(daily_nav)
    n_years = n_days / 252
    cagr = ((final_nav / initial_capital) ** (1 / max(n_years, 0.1)) - 1) * 100
    daily_rets = daily_nav['daily_return_pct'].dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
    nav_series = daily_nav['nav']
    peak = nav_series.cummax()
    dd = (peak - nav_series) / peak * 100
    max_dd = dd.max() if len(dd) else 0
    calmar = cagr / max_dd if max_dd > 0 else float('inf')

    by_system = trades_df.groupby('system').agg(
        n=('symbol', 'count'),
        wr=('net_ret_pct', lambda x: (x > 0).mean()),
        avg_ret=('net_ret_pct', 'mean'),
    ).reset_index()

    return dict(
        total_trades=total,
        win_rate=wr,
        profit_factor=pf,
        total_return_pct=total_return,
        cagr_pct=cagr,
        sharpe=sharpe,
        max_dd_pct=max_dd,
        calmar=calmar,
        final_nav=final_nav,
        n_trading_days=n_days,
        by_system=by_system,
    )


# -------------------------------------------------------------------------
# 4. main
# -------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--initial-capital', type=float, default=10_00_000)
    ap.add_argument('--risk-per-trade-pct', type=float, default=0.5)
    ap.add_argument('--risk-per-trade-rs', type=float, default=None,
                    help='Fixed rupee risk cap per trade (overrides --risk-per-trade-pct, matches ORB convention)')
    ap.add_argument('--max-concurrent', type=int, default=5)
    ap.add_argument('--cost-per-side-pct', type=float, default=0.05)
    ap.add_argument('--start', default='2024-03-18')
    ap.add_argument('--end', default='2026-03-25')
    ap.add_argument('--label', default='', help='Suffix for output files')
    ap.add_argument('--tp-pct', type=float, default=0.5)
    ap.add_argument('--sl-pct', type=float, default=1.5)
    ap.add_argument('--hold-bars', type=int, default=60)
    args = ap.parse_args()

    print('=== Stage 13 — combined 3-system portfolio backtest ===')
    print(f'Initial capital: Rs.{args.initial_capital:,.0f}')
    print(f'Risk per trade:  {args.risk_per_trade_pct}%')
    print(f'Max concurrent:  {args.max_concurrent} positions')
    print(f'Cost per side:   {args.cost_per_side_pct}% (round-trip {2*args.cost_per_side_pct}%)')
    print(f'Period:          {args.start} to {args.end}')
    print()

    # 1. load cohorts
    short_cohort = open(os.path.join(RESULTS_DIR, '07_short_diamonds.txt')).read().split()
    long_mr_cohort = open(os.path.join(RESULTS_DIR, '11c_long_reversal_diamonds.txt')).read().split()
    long_tc_cohort = open(os.path.join(RESULTS_DIR, '11b_trend_pullback_diamonds.txt')).read().split()
    all_stocks = sorted(set(short_cohort) | set(long_mr_cohort) | set(long_tc_cohort))
    print(f'Cohorts: SHORT {len(short_cohort)}, LONG_MR {len(long_mr_cohort)}, LONG_TC {len(long_tc_cohort)}; '
          f'unique total {len(all_stocks)}')

    # 2. load + enrich data
    print('Loading + enriching data ...')
    t0 = time.time()
    cache = {}
    for sym in all_stocks:
        df = load_5min(sym, start=args.start, end=args.end)
        if df.empty or len(df) < 500:
            continue
        try:
            cache[sym] = enrich(df, or_minutes=15)
        except Exception:
            continue
    print(f'  loaded {len(cache)}/{len(all_stocks)} stocks in {time.time()-t0:.0f}s')

    # 3. NIFTY regime
    print('Building NIFTY regime ...', end='', flush=True)
    nifty = mod_8.build_nifty_regime()
    print(f' {len(nifty)} sessions')

    # 4. signals
    print('Building signals ...')
    t0 = time.time()
    sig_short = build_signals_short(short_cohort, cache, nifty, variant='volume',
                                    tp_pct=args.tp_pct, sl_pct=args.sl_pct, hold_bars=args.hold_bars)
    sig_long_mr = build_signals_long_mr(long_mr_cohort, cache, nifty,
                                        tp_pct=args.tp_pct, sl_pct=args.sl_pct, hold_bars=args.hold_bars)
    sig_long_tc = build_signals_long_tc(long_tc_cohort, cache, nifty,
                                        tp_pct=args.tp_pct, sl_pct=args.sl_pct, hold_bars=args.hold_bars)
    print(f'  SHORT: {len(sig_short)} signals')
    print(f'  LONG_MR: {len(sig_long_mr)} signals')
    print(f'  LONG_TC: {len(sig_long_tc)} signals')
    events = pd.concat([sig_short, sig_long_mr, sig_long_tc], ignore_index=True)
    print(f'  TOTAL: {len(events)} signals in {time.time()-t0:.0f}s')

    # 5. run portfolio
    print('Running portfolio walker ...')
    t0 = time.time()
    trades_df, daily_nav = run_portfolio(
        events, cache,
        initial_capital=args.initial_capital,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_concurrent=args.max_concurrent,
        cost_per_side_pct=args.cost_per_side_pct,
        risk_per_trade_rs=args.risk_per_trade_rs,
    )
    print(f'  done in {time.time()-t0:.0f}s; {len(trades_df)} trades executed')

    # 6. write outputs
    suffix = f'_{args.label}' if args.label else ''
    out_trades = os.path.join(RESULTS_DIR, f'13_portfolio_trades{suffix}.csv')
    out_nav = os.path.join(RESULTS_DIR, f'13_portfolio_daily_nav{suffix}.csv')
    trades_df.to_csv(out_trades, index=False)
    daily_nav.to_csv(out_nav, index=False)

    # 7. summary
    summary = summarise(trades_df, daily_nav, args.initial_capital)
    summary_path = os.path.join(RESULTS_DIR, f'13_portfolio_summary{suffix}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('=== Combined 3-System Portfolio Summary ===\n\n')
        f.write(f'Initial capital:     Rs.{args.initial_capital:,.0f}\n')
        f.write(f'Final NAV:           Rs.{summary["final_nav"]:,.0f}\n')
        f.write(f'Total return:        {summary["total_return_pct"]:.2f}%\n')
        f.write(f'CAGR:                {summary["cagr_pct"]:.2f}%\n')
        f.write(f'Sharpe (daily):      {summary["sharpe"]:.2f}\n')
        f.write(f'Max drawdown:        {summary["max_dd_pct"]:.2f}%\n')
        f.write(f'Calmar:              {summary["calmar"]:.2f}\n\n')
        f.write(f'Total trades:        {summary["total_trades"]}\n')
        f.write(f'Win rate:            {summary["win_rate"]:.1%}\n')
        f.write(f'Profit factor:       {summary["profit_factor"]:.2f}\n')
        f.write(f'Trading days:        {summary["n_trading_days"]}\n')
        f.write(f'Trades per day avg:  {summary["total_trades"]/max(summary["n_trading_days"],1):.2f}\n\n')
        f.write('=== By system ===\n')
        f.write(summary['by_system'].to_string(index=False))
        f.write('\n')

    print()
    print(open(summary_path).read())
    print(f'Outputs: {out_trades}, {out_nav}, {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
