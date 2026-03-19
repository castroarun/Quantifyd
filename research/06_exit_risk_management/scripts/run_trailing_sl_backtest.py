"""
Trailing Stop Loss Backtest - Breakout V3 PRIMARY System
Compares fixed SL (consolidation low) vs trailing SL across all 332 PRIMARY trades.

Trailing SL methods:
1. FIXED: SL = consolidation low (current system)
2. TRAIL-X%: SL trails at X% below highest high since entry
3. RATCHET: SL starts at consol low, ratchets up to trail once price gains > trail%

Uses actual daily OHLCV data from market_data.db for accurate simulation.
"""

import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from services.consolidation_breakout import SYSTEM_PRIMARY, check_system_match

DB_PATH = Path(__file__).parent / "backtest_data" / "market_data.db"
CSV_PATH = Path(__file__).parent / "breakout_analysis_enhanced.csv"
HOLD_DAYS = 125


def load_primary_trades():
    """Load all PRIMARY system trades from the enhanced CSV."""
    df = pd.read_csv(CSV_PATH)
    mask = df.apply(lambda r: check_system_match(r.to_dict(), SYSTEM_PRIMARY), axis=1)
    primary = df[mask].copy()
    primary['date'] = pd.to_datetime(primary['date'])
    primary = primary.sort_values(['date', 'symbol']).reset_index(drop=True)
    return primary


def get_daily_prices(symbol, entry_date, conn, extra_before=5):
    """Get daily OHLCV for a symbol from entry_date onwards (+ buffer before for consol low calc)."""
    query = """
        SELECT date, open, high, low, close, volume
        FROM market_data_unified
        WHERE symbol = ? AND timeframe = 'day' AND date >= date(?, '-' || ? || ' days')
        ORDER BY date ASC
        LIMIT ?
    """
    # Get extra days before entry for safety, then 125 after
    buffer_days = extra_before * 2  # calendar days buffer
    df = pd.read_sql_query(query, conn, params=[symbol, entry_date.strftime('%Y-%m-%d'), str(buffer_days), HOLD_DAYS + buffer_days + 50])
    if df.empty:
        return None
    df['date'] = pd.to_datetime(df['date'])
    return df


def simulate_trade(prices_df, entry_date, entry_price, fixed_sl, trail_pcts):
    """
    Simulate a single trade with fixed SL and multiple trailing SL percentages.

    Returns dict with results for each method.
    """
    # Filter to trading days from entry onwards
    trade_days = prices_df[prices_df['date'] >= entry_date].head(HOLD_DAYS)
    if len(trade_days) < 2:
        return None

    entry_close = trade_days.iloc[0]['close']
    # Use actual close on entry date as entry price (may differ slightly from CSV due to data)

    results = {}

    # ---- FIXED SL ----
    fixed_exit_price, fixed_exit_day, fixed_exit_reason = _run_sl_sim(
        trade_days, entry_close, fixed_sl, trail_pct=None
    )
    fixed_return = (fixed_exit_price - entry_close) / entry_close * 100
    results['FIXED'] = {
        'exit_price': fixed_exit_price,
        'exit_day': fixed_exit_day,
        'exit_reason': fixed_exit_reason,
        'return_pct': fixed_return,
    }

    # ---- TRAILING SL at various percentages ----
    for pct in trail_pcts:
        trail_exit_price, trail_exit_day, trail_exit_reason = _run_sl_sim(
            trade_days, entry_close, fixed_sl, trail_pct=pct
        )
        trail_return = (trail_exit_price - entry_close) / entry_close * 100
        results[f'TRAIL-{pct}%'] = {
            'exit_price': trail_exit_price,
            'exit_day': trail_exit_day,
            'exit_reason': trail_exit_reason,
            'return_pct': trail_return,
        }

    # ---- RATCHET: starts at fixed SL, switches to trailing once price gains > trail% ----
    for pct in trail_pcts:
        ratch_exit_price, ratch_exit_day, ratch_exit_reason = _run_sl_sim(
            trade_days, entry_close, fixed_sl, trail_pct=pct, ratchet=True
        )
        ratch_return = (ratch_exit_price - entry_close) / entry_close * 100
        results[f'RATCHET-{pct}%'] = {
            'exit_price': ratch_exit_price,
            'exit_day': ratch_exit_day,
            'exit_reason': ratch_exit_reason,
            'return_pct': ratch_return,
        }

    # Track max gain for this trade
    highs = trade_days['high'].values
    peak = np.maximum.accumulate(highs)
    max_gain_pct = (peak.max() - entry_close) / entry_close * 100
    results['_max_gain'] = max_gain_pct
    results['_entry_price'] = entry_close

    return results


def _run_sl_sim(trade_days, entry_price, fixed_sl, trail_pct=None, ratchet=False):
    """
    Run stop loss simulation on daily bars.

    trail_pct: if set, trailing SL = highest_high * (1 - trail_pct/100)
    ratchet: if True, SL starts at fixed_sl and only switches to trailing
             once trailing SL > fixed SL (i.e., price has risen enough)
    """
    highest_high = entry_price
    current_sl = fixed_sl

    for i, (_, row) in enumerate(trade_days.iterrows()):
        if i == 0:
            # Entry bar - update highest but don't check SL
            highest_high = max(highest_high, row['high'])
            continue

        # Update highest high
        highest_high = max(highest_high, row['high'])

        # Calculate trailing SL level
        if trail_pct is not None:
            trail_sl = highest_high * (1 - trail_pct / 100)

            if ratchet:
                # Ratchet: use whichever is HIGHER (more protective)
                current_sl = max(fixed_sl, trail_sl)
            else:
                # Pure trailing: always use trail SL
                current_sl = trail_sl
        else:
            current_sl = fixed_sl

        # Check if SL hit (low touches or breaches SL)
        if row['low'] <= current_sl:
            return current_sl, i, 'STOP'

    # Time exit after HOLD_DAYS
    exit_price = trade_days.iloc[-1]['close']
    return exit_price, len(trade_days) - 1, 'OPEN'


def compute_metrics(returns, exit_reasons):
    """Compute summary metrics for a list of trade returns."""
    returns = np.array(returns)
    reasons = np.array(exit_reasons)
    n = len(returns)
    if n == 0:
        return {}

    wins = returns >= 0
    losses = returns < 0

    win_pct = wins.sum() / n * 100
    avg_return = returns.mean()
    avg_win = returns[wins].mean() if wins.any() else 0
    avg_loss = returns[losses].mean() if losses.any() else 0

    # Profit factor
    gross_profit = returns[wins].sum() if wins.any() else 0
    gross_loss = abs(returns[losses].sum()) if losses.any() else 0.01
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Stopped out stats
    stopped = reasons == 'STOP'
    stop_pct = stopped.sum() / n * 100

    # Big winner preservation (trades with >30% max gain)
    median_return = np.median(returns)

    # Severe losses (> -20%)
    severe = (returns < -20).sum()

    return {
        'trades': n,
        'win_pct': round(win_pct, 1),
        'avg_return': round(avg_return, 1),
        'avg_win': round(avg_win, 1),
        'avg_loss': round(avg_loss, 1),
        'median_return': round(median_return, 1),
        'profit_factor': round(pf, 2),
        'stop_pct': round(stop_pct, 1),
        'stopped_out': int(stopped.sum()),
        'severe_losses': severe,
        'total_return': round(returns.sum(), 1),
    }


def main():
    print("=" * 100)
    print("TRAILING STOP LOSS BACKTEST - Breakout V3 PRIMARY System")
    print("=" * 100)
    print(f"System: PRIMARY = ALPHA OR T1B OR MOMVOL")
    print(f"Hold period: {HOLD_DAYS} trading days | Fixed SL: Consolidation Low")
    print()

    # Load trades
    primary = load_primary_trades()
    print(f"Loaded {len(primary)} PRIMARY trades")

    # Trail percentages to test
    trail_pcts = [10, 15, 20, 25, 30]

    # Connect to DB
    conn = sqlite3.connect(str(DB_PATH))

    # Storage for all results
    all_methods = ['FIXED'] + [f'TRAIL-{p}%' for p in trail_pcts] + [f'RATCHET-{p}%' for p in trail_pcts]
    method_returns = {m: [] for m in all_methods}
    method_exits = {m: [] for m in all_methods}
    method_days = {m: [] for m in all_methods}

    # Per-trade detail for analysis
    trade_details = []

    skipped = 0
    processed = 0

    for idx, row in primary.iterrows():
        symbol = row['symbol']
        entry_date = row['date']
        risk_pct = row['risk_pct']

        # Get daily prices
        prices = get_daily_prices(symbol, entry_date, conn)
        if prices is None or len(prices) < 10:
            skipped += 1
            continue

        # Entry price = close on entry date
        entry_bar = prices[prices['date'] >= entry_date]
        if entry_bar.empty:
            skipped += 1
            continue

        entry_price = entry_bar.iloc[0]['close']
        fixed_sl = entry_price * (1 - risk_pct / 100)

        # Simulate all methods
        results = simulate_trade(prices, entry_date, entry_price, fixed_sl, trail_pcts)
        if results is None:
            skipped += 1
            continue

        processed += 1

        detail = {
            'symbol': symbol,
            'date': entry_date,
            'entry_price': results['_entry_price'],
            'max_gain': results['_max_gain'],
            'fixed_sl': fixed_sl,
        }

        for method in all_methods:
            if method in results:
                method_returns[method].append(results[method]['return_pct'])
                method_exits[method].append(results[method]['exit_reason'])
                method_days[method].append(results[method]['exit_day'])
                detail[f'{method}_return'] = results[method]['return_pct']
                detail[f'{method}_exit'] = results[method]['exit_reason']
                detail[f'{method}_days'] = results[method]['exit_day']

        trade_details.append(detail)

        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(primary)} trades...")

    conn.close()

    print(f"\nProcessed: {processed} | Skipped (no data): {skipped}")
    print()

    # ===== RESULTS TABLE =====
    print("=" * 100)
    print("SECTION 1: OVERALL COMPARISON")
    print("=" * 100)
    print(f"  {'Method':<18} {'Trades':>6} {'Win%':>6} {'AvgRet':>7} {'AvgWin':>7} {'AvgLoss':>8} {'Median':>7} {'PF':>6} {'Stop%':>6} {'Severe':>7} {'TotalRet':>9}")
    print(f"  {'-'*92}")

    for method in all_methods:
        if not method_returns[method]:
            continue
        m = compute_metrics(method_returns[method], method_exits[method])
        marker = " <-- CURRENT" if method == 'FIXED' else ""
        print(f"  {method:<18} {m['trades']:>6} {m['win_pct']:>5.1f}% {m['avg_return']:>+6.1f}% {m['avg_win']:>+6.1f}% {m['avg_loss']:>+7.1f}% {m['median_return']:>+6.1f}% {m['profit_factor']:>5.2f} {m['stop_pct']:>5.1f}% {m['severe_losses']:>6} {m['total_return']:>+8.1f}%{marker}")

    # ===== TRADES THAT "GOT AWAY" =====
    # Trades where max gain > 30% but fixed SL ended in loss
    print()
    print("=" * 100)
    print("SECTION 2: RESCUED TRADES - Had >30% gain but fixed SL ended in loss")
    print("=" * 100)

    df_details = pd.DataFrame(trade_details)
    if 'FIXED_return' in df_details.columns:
        got_away = df_details[(df_details['max_gain'] > 30) & (df_details['FIXED_return'] < 0)]
        print(f"  {len(got_away)} trades had >30% unrealized gain but ended as fixed-SL losses")
        print()

        if len(got_away) > 0:
            # Show how each trailing method performed on these trades
            print(f"  {'Method':<18} {'Rescued':>8} {'AvgRet':>8} {'StillLoss':>10}")
            print(f"  {'-'*50}")
            for method in all_methods:
                col = f'{method}_return'
                if col in got_away.columns:
                    rescued = (got_away[col] >= 0).sum()
                    avg_ret = got_away[col].mean()
                    still_loss = (got_away[col] < 0).sum()
                    print(f"  {method:<18} {rescued:>7}/{len(got_away)} {avg_ret:>+7.1f}% {still_loss:>9}")

            print()
            print("  Top 10 worst 'got away' trades (sorted by max unrealized gain):")
            print(f"  {'Symbol':<12} {'Date':<12} {'MaxGain':>8} {'FIXED':>8} {'TRAIL-15%':>10} {'TRAIL-20%':>10} {'RATCH-15%':>10} {'RATCH-20%':>10}")
            print(f"  {'-'*82}")
            worst = got_away.nlargest(10, 'max_gain')
            for _, r in worst.iterrows():
                t15 = r.get('TRAIL-15%_return', 0)
                t20 = r.get('TRAIL-20%_return', 0)
                r15 = r.get('RATCHET-15%_return', 0)
                r20 = r.get('RATCHET-20%_return', 0)
                print(f"  {r['symbol']:<12} {str(r['date'])[:10]:<12} {r['max_gain']:>+7.1f}% {r['FIXED_return']:>+7.1f}% {t15:>+9.1f}% {t20:>+9.1f}% {r15:>+9.1f}% {r20:>+9.1f}%")

    # ===== HOLDING PERIOD ANALYSIS =====
    print()
    print("=" * 100)
    print("SECTION 3: AVERAGE HOLDING PERIOD (trading days)")
    print("=" * 100)
    print(f"  {'Method':<18} {'AvgDays':>8} {'MedianDays':>11} {'Avg(Wins)':>10} {'Avg(Losses)':>12}")
    print(f"  {'-'*62}")

    for method in all_methods:
        if not method_days[method]:
            continue
        days = np.array(method_days[method])
        rets = np.array(method_returns[method])
        wins = rets >= 0
        losses = rets < 0
        avg_d = days.mean()
        med_d = np.median(days)
        avg_w = days[wins].mean() if wins.any() else 0
        avg_l = days[losses].mean() if losses.any() else 0
        print(f"  {method:<18} {avg_d:>7.0f} {med_d:>10.0f} {avg_w:>9.0f} {avg_l:>11.0f}")

    # ===== BEST METHOD SUMMARY =====
    print()
    print("=" * 100)
    print("SECTION 4: RECOMMENDATION")
    print("=" * 100)

    # Find best by various criteria
    best_wr = max(all_methods, key=lambda m: compute_metrics(method_returns[m], method_exits[m])['win_pct'] if method_returns[m] else 0)
    best_avg = max(all_methods, key=lambda m: compute_metrics(method_returns[m], method_exits[m])['avg_return'] if method_returns[m] else -999)
    best_pf = max(all_methods, key=lambda m: compute_metrics(method_returns[m], method_exits[m])['profit_factor'] if method_returns[m] else 0)
    best_total = max(all_methods, key=lambda m: compute_metrics(method_returns[m], method_exits[m])['total_return'] if method_returns[m] else -99999)

    print(f"  Best Win Rate:       {best_wr} ({compute_metrics(method_returns[best_wr], method_exits[best_wr])['win_pct']}%)")
    print(f"  Best Avg Return:     {best_avg} ({compute_metrics(method_returns[best_avg], method_exits[best_avg])['avg_return']}%)")
    print(f"  Best Profit Factor:  {best_pf} ({compute_metrics(method_returns[best_pf], method_exits[best_pf])['profit_factor']})")
    print(f"  Best Total Return:   {best_total} ({compute_metrics(method_returns[best_total], method_exits[best_total])['total_return']}%)")

    # Save detailed results
    output_path = Path(__file__).parent / "trailing_sl_results.csv"
    df_details.to_csv(output_path, index=False)
    print(f"\n  Detailed per-trade results saved to: {output_path}")

    # Save summary
    summary_path = Path(__file__).parent / "trailing_sl_summary.txt"
    with open(summary_path, 'w') as f:
        for method in all_methods:
            if not method_returns[method]:
                continue
            m = compute_metrics(method_returns[method], method_exits[method])
            f.write(f"{method}: Win%={m['win_pct']}, AvgRet={m['avg_return']}%, PF={m['profit_factor']}, Total={m['total_return']}%, Severe={m['severe_losses']}\n")
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
