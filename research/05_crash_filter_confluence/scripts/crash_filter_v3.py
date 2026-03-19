"""
Crash Filter V3: VIX-like & Volatility Expansion Filters
=========================================================
Build synthetic market-level volatility measures from the universe:

1. Synthetic VIX: Cross-sectional realized vol (stdev of daily returns across stocks)
2. Universe ATR Surge: Average ATR ratio across all stocks (vol expansion)
3. Volatility Expansion Speed: Rate of change of synthetic VIX
4. NIFTYBEES drawdown: Market-level drawdown from ETF
5. Market Return Regime: Rolling N-day return of NIFTYBEES/universe

Then combine the best with breadth + SL counter from V2 to find
the optimal multi-layer crash protection.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'


def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(window=p).mean()

def atr_series(high, low, close, p=14):
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def keltner(df, ep=6, ap=6, m=1.3):
    mid = ema(df['close'], ep)
    a = atr_series(df['high'], df['low'], df['close'], ap)
    return mid, mid + m * a, mid - m * a


def build_market_indicators(all_symbol_data, conn):
    """
    Build daily market-level indicators from the full universe.
    Returns a DataFrame indexed by date with columns for each indicator.
    """
    # Collect daily returns and ATR ratios for all symbols
    daily_returns = {}
    daily_atr_ratios = {}
    daily_above200 = {}

    for sym, df in all_symbol_data.items():
        ret = df['close'].pct_change()
        daily_returns[sym] = ret

        atr14 = atr_series(df['high'], df['low'], df['close'], 14)
        atr50avg = atr14.rolling(50).mean()
        daily_atr_ratios[sym] = atr14 / atr50avg

        if 'above200' in df.columns:
            daily_above200[sym] = df['above200'].astype(int)

    ret_df = pd.DataFrame(daily_returns)
    atr_df = pd.DataFrame(daily_atr_ratios)
    above_df = pd.DataFrame(daily_above200)

    mkt = pd.DataFrame(index=ret_df.index)

    # 1. Synthetic VIX: Cross-sectional stdev of daily returns (annualized)
    mkt['synth_vix'] = ret_df.std(axis=1) * np.sqrt(252) * 100  # as %
    mkt['synth_vix_20avg'] = mkt['synth_vix'].rolling(20).mean()
    mkt['vix_ratio'] = mkt['synth_vix'] / mkt['synth_vix_20avg']
    mkt['vix_5d_chg'] = mkt['synth_vix'] - mkt['synth_vix'].shift(5)

    # 2. Universe ATR surge: median ATR ratio across all stocks
    mkt['univ_atr_ratio'] = atr_df.median(axis=1)
    mkt['univ_atr_20avg'] = mkt['univ_atr_ratio'].rolling(20).mean()
    mkt['atr_surge'] = mkt['univ_atr_ratio'] / mkt['univ_atr_20avg']
    mkt['atr_5d_chg'] = mkt['univ_atr_ratio'] - mkt['univ_atr_ratio'].shift(5)

    # 3. Market breadth
    mkt['breadth'] = above_df.mean(axis=1) * 100
    mkt['breadth_chg5'] = mkt['breadth'] - mkt['breadth'].shift(5)
    mkt['breadth_chg10'] = mkt['breadth'] - mkt['breadth'].shift(10)

    # 4. Universe average return (market momentum)
    mkt['univ_ret_5d'] = ret_df.mean(axis=1).rolling(5).sum() * 100
    mkt['univ_ret_10d'] = ret_df.mean(axis=1).rolling(10).sum() * 100

    # 5. NIFTYBEES as market proxy (if available)
    try:
        nifty = pd.read_sql_query(
            "SELECT date,open,high,low,close FROM market_data_unified "
            "WHERE symbol='NIFTYBEES' AND timeframe='day' ORDER BY date", conn)
        if len(nifty) > 100:
            nifty['date'] = pd.to_datetime(nifty['date'])
            nifty.set_index('date', inplace=True)
            nifty = nifty.astype(float)
            nifty_high20 = nifty['high'].rolling(20).max()
            mkt['nifty_dd20'] = (nifty['close'] / nifty_high20 - 1) * 100
            mkt['nifty_ret5'] = nifty['close'].pct_change(5) * 100
            mkt['nifty_ret10'] = nifty['close'].pct_change(10) * 100
            nifty_atr = atr_series(nifty['high'], nifty['low'], nifty['close'], 14)
            nifty_atr_avg = nifty_atr.rolling(50).mean()
            mkt['nifty_atr_ratio'] = nifty_atr / nifty_atr_avg
    except Exception:
        pass

    # 6. % of stocks hitting new 20-day lows (panic breadth)
    pct_new_lows = {}
    for sym, df in all_symbol_data.items():
        low20 = df['low'].rolling(20).min()
        pct_new_lows[sym] = (df['low'] <= low20).astype(int)
    lows_df = pd.DataFrame(pct_new_lows)
    mkt['pct_new_lows'] = lows_df.mean(axis=1) * 100

    # 7. Volatility expansion speed: 5-day rate of change of synth VIX
    mkt['vol_expansion'] = (mkt['synth_vix'] / mkt['synth_vix'].shift(5) - 1) * 100

    return mkt


def run_backtest_with_full_context(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15):
    """Run V2 backtest with all market-level context."""
    print("  Phase 1: Loading data...")
    all_symbol_data = {}
    entry_signals = []

    for sym in symbols:
        df = pd.read_sql_query(
            "SELECT date,open,high,low,close,volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date",
            conn, params=[sym])
        if len(df) < 250:
            continue
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.astype(float)

        c = df['close']
        df['sma_200'] = sma(c, 200)
        df['kc6_m'], df['kc6_u'], df['kc6_l'] = keltner(df, 6, 6, 1.3)
        df['above200'] = c > df['sma_200']

        all_symbol_data[sym] = df

        for i in range(200, len(df)):
            row = df.iloc[i]
            if row['close'] < row['kc6_l'] and row['above200']:
                entry_signals.append((df.index[i], sym, i))

    entry_signals.sort(key=lambda x: x[0])

    print("  Phase 2: Building market indicators...")
    mkt = build_market_indicators(all_symbol_data, conn)

    # Signals per day
    signals_per_day = defaultdict(int)
    for dt, sym, idx in entry_signals:
        signals_per_day[dt] += 1

    print("  Phase 3: Running chronological backtest...")
    all_trades = []
    sl_history = []
    in_trade = set()

    for entry_date, sym, entry_i in entry_signals:
        if sym in in_trade:
            continue

        df = all_symbol_data[sym]
        row = df.iloc[entry_i]
        entry_price = row['close']

        # Market context
        m = mkt.loc[entry_date] if entry_date in mkt.index else pd.Series()

        ctx = {
            'signals_today': signals_per_day[entry_date],
            'synth_vix': m.get('synth_vix', np.nan),
            'vix_ratio': m.get('vix_ratio', np.nan),
            'vix_5d_chg': m.get('vix_5d_chg', np.nan),
            'vol_expansion': m.get('vol_expansion', np.nan),
            'univ_atr_ratio': m.get('univ_atr_ratio', np.nan),
            'atr_surge': m.get('atr_surge', np.nan),
            'atr_5d_chg': m.get('atr_5d_chg', np.nan),
            'breadth': m.get('breadth', np.nan),
            'breadth_chg5': m.get('breadth_chg5', np.nan),
            'breadth_chg10': m.get('breadth_chg10', np.nan),
            'univ_ret_5d': m.get('univ_ret_5d', np.nan),
            'univ_ret_10d': m.get('univ_ret_10d', np.nan),
            'nifty_dd20': m.get('nifty_dd20', np.nan),
            'nifty_ret5': m.get('nifty_ret5', np.nan),
            'nifty_atr_ratio': m.get('nifty_atr_ratio', np.nan),
            'pct_new_lows': m.get('pct_new_lows', np.nan),
        }

        # Rolling SL count
        sl_5d = sum(1 for d, s in sl_history if d >= entry_date - timedelta(days=7))
        ctx['sl_last_5d'] = sl_5d

        # Execute trade
        in_trade.add(sym)
        n = len(df)
        j = entry_i + 1
        while j < n:
            cur = df.iloc[j]
            hold = j - entry_i
            low_pnl = (cur['low'] / entry_price - 1) * 100
            high_pnl = (cur['high'] / entry_price - 1) * 100

            exit_reason = exit_price = None

            if low_pnl <= -sl_pct:
                exit_reason = 'STOP_LOSS'
                exit_price = round(entry_price * (1 - sl_pct / 100), 2)
            elif high_pnl >= tp_pct:
                exit_reason = 'TAKE_PROFIT'
                exit_price = round(entry_price * (1 + tp_pct / 100), 2)
            elif hold >= max_hold:
                exit_reason = 'MAX_HOLD'
                exit_price = round(cur['close'], 2)
            elif cur['high'] > cur['kc6_m']:
                exit_reason = 'SIGNAL_KC6_MID'
                exit_price = round(cur['kc6_m'], 2)

            if exit_reason:
                pnl = (exit_price / entry_price - 1) * 100
                if exit_reason == 'STOP_LOSS':
                    sl_history.append((df.index[j], sym))

                trade = {
                    'symbol': sym, 'entry_date': entry_date,
                    'pnl_pct': round(pnl, 2), 'hold_days': hold,
                    'exit_reason': exit_reason, 'win': pnl > 0,
                }
                trade.update(ctx)
                all_trades.append(trade)
                in_trade.discard(sym)
                break
            j += 1
        else:
            in_trade.discard(sym)

    return pd.DataFrame(all_trades)


def fstats(df, mask, label, baseline_pnl):
    """Filter stats."""
    allowed = df[~mask]
    blocked = df[mask]
    if len(allowed) == 0:
        return None

    al_wr = allowed['win'].sum() / len(allowed) * 100
    al_avg = allowed['pnl_pct'].mean()
    al_total = allowed['pnl_pct'].sum()
    gp = allowed[allowed['pnl_pct'] > 0]['pnl_pct'].sum()
    gl = abs(allowed[allowed['pnl_pct'] < 0]['pnl_pct'].sum())
    al_pf = gp / gl if gl > 0 else float('inf')

    bl_sl = len(blocked[blocked['exit_reason'] == 'STOP_LOSS'])
    total_sl = len(df[df['exit_reason'] == 'STOP_LOSS'])
    bl_wins = blocked['win'].sum()
    total_wins = df['win'].sum()

    return {
        'label': label, 'blocked': len(blocked),
        'blocked_pct': round(len(blocked)/len(df)*100, 1),
        'sl_caught': bl_sl,
        'sl_catch_pct': round(bl_sl/total_sl*100, 1) if total_sl > 0 else 0,
        'wins_lost': bl_wins,
        'wins_lost_pct': round(bl_wins/total_wins*100, 1) if total_wins > 0 else 0,
        'allowed': len(allowed),
        'wr': round(al_wr, 1), 'avg_pnl': round(al_avg, 2),
        'pf': round(al_pf, 2), 'total_pnl': round(al_total, 1),
        'delta': round(al_total - baseline_pnl, 1),
    }


def dist_table(df, col, buckets, label):
    """Print distribution table for a column."""
    print(f"\n  {label}:")
    print(f"  {'Range':>18s} {'Total':>7s} {'Wins':>6s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'SL':>5s}")
    print(f"  {'-'*60}")

    for lo, hi, name in buckets:
        mask = (df[col] >= lo) & (df[col] < hi)
        sub = df[mask].dropna(subset=[col])
        if len(sub) > 5:
            wr = sub['win'].sum() / len(sub) * 100
            avg = sub['pnl_pct'].mean()
            sl = len(sub[sub['exit_reason'] == 'STOP_LOSS'])
            gpx = sub[sub['pnl_pct'] > 0]['pnl_pct'].sum()
            glx = abs(sub[sub['pnl_pct'] < 0]['pnl_pct'].sum())
            pf = gpx / glx if glx > 0 else float('inf')
            print(f"  {name:>18s} {len(sub):>7d} {sub['win'].sum():>6d} "
                  f"{wr:>5.1f}% {avg:>+7.2f}% {pf:>5.2f} {sl:>5d}")


def main():
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()

    print("=" * 130)
    print("CRASH FILTER V3: VIX, Volatility Expansion & Combined Circuit Breakers")
    print("=" * 130)

    df = run_backtest_with_full_context(symbols, conn, sl_pct=5, tp_pct=15, max_hold=15)
    conn.close()

    total = len(df)
    baseline_wr = df['win'].sum() / total * 100
    baseline_pnl = df['pnl_pct'].sum()
    baseline_avg = df['pnl_pct'].mean()
    gp = df[df['pnl_pct'] > 0]['pnl_pct'].sum()
    gl = abs(df[df['pnl_pct'] < 0]['pnl_pct'].sum())
    baseline_pf = gp / gl if gl > 0 else float('inf')

    print(f"\nBaseline: {total} trades | WR: {baseline_wr:.1f}% | "
          f"Avg P/L: {baseline_avg:+.2f}% | PF: {baseline_pf:.2f} | "
          f"Total P/L: {baseline_pnl:+.1f}%")

    # =====================================================================
    # PART 1: Distribution analysis of new indicators
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 1: TRADE QUALITY BY VOLATILITY INDICATORS")
    print(f"{'='*130}")

    dist_table(df, 'synth_vix', [
        (0, 15, '<15% (calm)'), (15, 25, '15-25% (normal)'),
        (25, 35, '25-35% (elevated)'), (35, 50, '35-50% (high)'),
        (50, 75, '50-75% (panic)'), (75, 999, '75%+ (extreme)'),
    ], "A. Synthetic VIX (annualized cross-sectional vol)")

    dist_table(df, 'vix_ratio', [
        (0, 0.8, '<0.8x (low)'), (0.8, 1.0, '0.8-1.0x'),
        (1.0, 1.2, '1.0-1.2x'), (1.2, 1.5, '1.2-1.5x (elevated)'),
        (1.5, 2.0, '1.5-2.0x (high)'), (2.0, 99, '2.0x+ (extreme)'),
    ], "B. VIX Ratio (current / 20-day avg)")

    dist_table(df, 'vol_expansion', [
        (-99, -20, '<-20% (vol shrinking)'), (-20, 0, '-20 to 0%'),
        (0, 20, '0 to +20%'), (20, 50, '+20 to +50%'),
        (50, 100, '+50 to +100%'), (100, 9999, '+100%+ (vol exploding)'),
    ], "C. Volatility Expansion Speed (5d % change in synth VIX)")

    dist_table(df, 'univ_atr_ratio', [
        (0, 0.8, '<0.8x'), (0.8, 1.0, '0.8-1.0x'),
        (1.0, 1.2, '1.0-1.2x'), (1.2, 1.5, '1.2-1.5x'),
        (1.5, 2.0, '1.5-2.0x'), (2.0, 99, '2.0x+'),
    ], "D. Universe Median ATR Ratio (vol expansion breadth)")

    dist_table(df, 'atr_surge', [
        (0, 0.9, '<0.9x'), (0.9, 1.0, '0.9-1.0x'),
        (1.0, 1.1, '1.0-1.1x'), (1.1, 1.2, '1.1-1.2x'),
        (1.2, 1.5, '1.2-1.5x'), (1.5, 99, '1.5x+'),
    ], "E. ATR Surge (universe ATR ratio / 20d avg)")

    dist_table(df, 'pct_new_lows', [
        (0, 5, '<5%'), (5, 10, '5-10%'), (10, 20, '10-20%'),
        (20, 35, '20-35%'), (35, 50, '35-50%'), (50, 100, '50%+'),
    ], "F. % of Stocks at 20-Day Lows")

    dist_table(df, 'univ_ret_5d', [
        (-99, -3, '<-3%'), (-3, -1.5, '-3 to -1.5%'),
        (-1.5, 0, '-1.5 to 0%'), (0, 1, '0 to +1%'),
        (1, 2, '+1 to +2%'), (2, 99, '>+2%'),
    ], "G. Universe Average 5-Day Return")

    if 'nifty_dd20' in df.columns:
        dist_table(df, 'nifty_dd20', [
            (0.01, 99, '>0% (new high)'), (-3, 0.01, '0 to -3%'),
            (-6, -3, '-3 to -6%'), (-10, -6, '-6 to -10%'),
            (-15, -10, '-10 to -15%'), (-99, -15, '<-15%'),
        ], "H. NIFTYBEES 20-Day Drawdown")

    if 'nifty_atr_ratio' in df.columns:
        dist_table(df, 'nifty_atr_ratio', [
            (0, 0.8, '<0.8x'), (0.8, 1.0, '0.8-1.0x'),
            (1.0, 1.3, '1.0-1.3x'), (1.3, 1.7, '1.3-1.7x'),
            (1.7, 2.5, '1.7-2.5x'), (2.5, 99, '2.5x+'),
        ], "I. NIFTYBEES ATR Ratio (Nifty-level vol expansion)")

    # =====================================================================
    # PART 2: Test all individual filters
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 2: INDIVIDUAL VIX/VOL FILTER TESTS")
    print(f"{'='*130}")
    print(f"  Baseline: {total} trades | WR: {baseline_wr:.1f}% | PF: {baseline_pf:.2f} | Total P/L: {baseline_pnl:+.1f}%\n")

    filters = []

    # Synthetic VIX level
    for t in [35, 40, 50, 60]:
        m = df['synth_vix'].fillna(0) >= t
        f = fstats(df, m, f"Synth VIX >= {t}%", baseline_pnl)
        if f: filters.append(f)

    # VIX ratio (current vs avg)
    for t in [1.3, 1.5, 1.8, 2.0]:
        m = df['vix_ratio'].fillna(1) >= t
        f = fstats(df, m, f"VIX ratio >= {t}x", baseline_pnl)
        if f: filters.append(f)

    # Volatility expansion speed
    for t in [30, 50, 75, 100]:
        m = df['vol_expansion'].fillna(0) >= t
        f = fstats(df, m, f"Vol expand >= +{t}%", baseline_pnl)
        if f: filters.append(f)

    # Universe ATR ratio
    for t in [1.2, 1.3, 1.5]:
        m = df['univ_atr_ratio'].fillna(1) >= t
        f = fstats(df, m, f"Univ ATR ratio >= {t}x", baseline_pnl)
        if f: filters.append(f)

    # ATR surge
    for t in [1.1, 1.2, 1.3]:
        m = df['atr_surge'].fillna(1) >= t
        f = fstats(df, m, f"ATR surge >= {t}x", baseline_pnl)
        if f: filters.append(f)

    # % new lows
    for t in [20, 30, 40, 50]:
        m = df['pct_new_lows'].fillna(0) >= t
        f = fstats(df, m, f"New lows >= {t}%", baseline_pnl)
        if f: filters.append(f)

    # Universe return
    for t in [-1.5, -2.0, -3.0]:
        m = df['univ_ret_5d'].fillna(0) <= t
        f = fstats(df, m, f"Univ 5d ret <= {t}%", baseline_pnl)
        if f: filters.append(f)

    # Nifty drawdown
    if 'nifty_dd20' in df.columns:
        for t in [-5, -8, -10]:
            m = df['nifty_dd20'].fillna(0) <= t
            f = fstats(df, m, f"Nifty DD20 <= {t}%", baseline_pnl)
            if f: filters.append(f)

    # Nifty ATR ratio
    if 'nifty_atr_ratio' in df.columns:
        for t in [1.3, 1.5, 2.0]:
            m = df['nifty_atr_ratio'].fillna(1) >= t
            f = fstats(df, m, f"Nifty ATR >= {t}x", baseline_pnl)
            if f: filters.append(f)

    # V2 best for reference
    m_breadth10 = df['breadth_chg10'].fillna(0) < -20
    f = fstats(df, m_breadth10, "Breadth 10d chg < -20pp", baseline_pnl)
    if f: filters.append(f)
    m_sl5d = df['sl_last_5d'] >= 3
    f = fstats(df, m_sl5d, "SL(5d) >= 3 [from V2]", baseline_pnl)
    if f: filters.append(f)

    # Sort by total P/L (best first)
    filters.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"  {'Filter':<30s} {'Blk':>5s} {'Blk%':>5s} {'SLcatch':>9s} {'WinLost':>9s} "
          f"{'Alwd':>5s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'TotP/L':>10s} {'Delta':>8s}")
    print(f"  {'-'*110}")

    for f in filters:
        print(f"  {f['label']:<30s} {f['blocked']:>5d} {f['blocked_pct']:>4.1f}% "
              f"{f['sl_caught']:>4d}({f['sl_catch_pct']:>4.1f}%) "
              f"{f['wins_lost']:>4d}({f['wins_lost_pct']:>4.1f}%) "
              f"{f['allowed']:>5d} {f['wr']:>5.1f}% {f['avg_pnl']:>+7.2f}% "
              f"{f['pf']:>5.2f} {f['total_pnl']:>+9.1f}% {f['delta']:>+7.1f}%")

    # =====================================================================
    # PART 3: COMBO FILTERS (best vol + best from V2)
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 3: COMBINATION FILTERS (VIX/VOL + BREADTH + SL COUNTER)")
    print(f"{'='*130}\n")

    combos = []

    # VIX + SL counter
    for vt, st in [(40, 3), (50, 3), (35, 2), (40, 2)]:
        m = (df['synth_vix'].fillna(0) >= vt) & (df['sl_last_5d'] >= st)
        f = fstats(df, m, f"VIX>={vt} AND SL5d>={st}", baseline_pnl)
        if f: combos.append(f)

    # VIX ratio + SL counter
    for vr, st in [(1.3, 3), (1.5, 2), (1.3, 2)]:
        m = (df['vix_ratio'].fillna(1) >= vr) & (df['sl_last_5d'] >= st)
        f = fstats(df, m, f"VIXr>={vr} AND SL5d>={st}", baseline_pnl)
        if f: combos.append(f)

    # Vol expansion + breadth
    for ve, bc in [(50, -10), (50, -15), (75, -10)]:
        m = (df['vol_expansion'].fillna(0) >= ve) & (df['breadth_chg10'].fillna(0) < bc)
        f = fstats(df, m, f"VolExp>={ve} AND BChg10<{bc}", baseline_pnl)
        if f: combos.append(f)

    # New lows + VIX
    for nl, vt in [(30, 35), (20, 40), (30, 40)]:
        m = (df['pct_new_lows'].fillna(0) >= nl) & (df['synth_vix'].fillna(0) >= vt)
        f = fstats(df, m, f"NewLows>={nl} AND VIX>={vt}", baseline_pnl)
        if f: combos.append(f)

    # OR combos (any one triggers block)
    for label, mask in [
        ("VIX>=50 OR SL5d>=3",
         (df['synth_vix'].fillna(0) >= 50) | (df['sl_last_5d'] >= 3)),
        ("VIXr>=1.5 OR SL5d>=3",
         (df['vix_ratio'].fillna(1) >= 1.5) | (df['sl_last_5d'] >= 3)),
        ("VIX>=40 OR BChg10<-20",
         (df['synth_vix'].fillna(0) >= 40) | (df['breadth_chg10'].fillna(0) < -20)),
        ("NewLows>=30 OR SL5d>=3",
         (df['pct_new_lows'].fillna(0) >= 30) | (df['sl_last_5d'] >= 3)),
        ("VolExp>=50 OR SL5d>=3",
         (df['vol_expansion'].fillna(0) >= 50) | (df['sl_last_5d'] >= 3)),
    ]:
        f = fstats(df, mask, label, baseline_pnl)
        if f: combos.append(f)

    # Triple combos (any two of three triggers)
    for label, mask in [
        ("2of3: VIX>=40|SL5d>=3|BChg10<-15",
         ((df['synth_vix'].fillna(0) >= 40).astype(int) +
          (df['sl_last_5d'] >= 3).astype(int) +
          (df['breadth_chg10'].fillna(0) < -15).astype(int)) >= 2),
        ("2of3: VIXr>=1.3|SL5d>=2|NewLow>=20",
         ((df['vix_ratio'].fillna(1) >= 1.3).astype(int) +
          (df['sl_last_5d'] >= 2).astype(int) +
          (df['pct_new_lows'].fillna(0) >= 20).astype(int)) >= 2),
        ("2of3: VIX>=35|SL5d>=2|BChg10<-20",
         ((df['synth_vix'].fillna(0) >= 35).astype(int) +
          (df['sl_last_5d'] >= 2).astype(int) +
          (df['breadth_chg10'].fillna(0) < -20).astype(int)) >= 2),
    ]:
        f = fstats(df, mask, label, baseline_pnl)
        if f: combos.append(f)

    combos.sort(key=lambda x: x['total_pnl'], reverse=True)

    print(f"  {'Filter':<40s} {'Blk':>5s} {'Blk%':>5s} {'SLcatch':>9s} {'WinLost':>9s} "
          f"{'Alwd':>5s} {'WR%':>6s} {'AvgP/L':>8s} {'PF':>6s} {'TotP/L':>10s} {'Delta':>8s}")
    print(f"  {'-'*120}")

    for f in combos:
        print(f"  {f['label']:<40s} {f['blocked']:>5d} {f['blocked_pct']:>4.1f}% "
              f"{f['sl_caught']:>4d}({f['sl_catch_pct']:>4.1f}%) "
              f"{f['wins_lost']:>4d}({f['wins_lost_pct']:>4.1f}%) "
              f"{f['allowed']:>5d} {f['wr']:>5.1f}% {f['avg_pnl']:>+7.2f}% "
              f"{f['pf']:>5.2f} {f['total_pnl']:>+9.1f}% {f['delta']:>+7.1f}%")

    # =====================================================================
    # PART 4: Top filters across crash periods
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 4: TOP FILTERS vs CRASH PERIODS")
    print(f"{'='*130}")

    crashes = [
        ('2008 GFC', '2008-01-01', '2008-12-31'),
        ('2011 Euro', '2011-07-01', '2012-01-31'),
        ('2015-16', '2015-08-01', '2016-03-31'),
        ('2018 NBFC', '2018-08-01', '2018-12-31'),
        ('2020 COVID', '2020-02-01', '2020-04-30'),
        ('2022 War', '2022-01-01', '2022-07-31'),
    ]

    top_filters = [
        ("No Filter", pd.Series(False, index=df.index)),
        ("Synth VIX >= 50%", df['synth_vix'].fillna(0) >= 50),
        ("VIX ratio >= 1.5x", df['vix_ratio'].fillna(1) >= 1.5),
        ("Vol expand >= +50%", df['vol_expansion'].fillna(0) >= 50),
        ("New lows >= 30%", df['pct_new_lows'].fillna(0) >= 30),
        ("Breadth 10d < -20pp", df['breadth_chg10'].fillna(0) < -20),
        ("SL(5d) >= 3", df['sl_last_5d'] >= 3),
        ("VIX>=40 AND SL5d>=3", (df['synth_vix'].fillna(0) >= 40) & (df['sl_last_5d'] >= 3)),
        ("2of3: VIX>=40|SL>=3|BChg<-15",
         ((df['synth_vix'].fillna(0) >= 40).astype(int) +
          (df['sl_last_5d'] >= 3).astype(int) +
          (df['breadth_chg10'].fillna(0) < -15).astype(int)) >= 2),
    ]

    for pname, start, end in crashes:
        cm = (df['entry_date'] >= start) & (df['entry_date'] <= end)
        ct = df[cm]
        if len(ct) == 0:
            continue

        ct_pnl = ct['pnl_pct'].sum()
        ct_sl = len(ct[ct['exit_reason'] == 'STOP_LOSS'])

        print(f"\n  {pname}: {len(ct)} trades | P/L: {ct_pnl:+.1f}% | SL: {ct_sl}")
        print(f"  {'Filter':<40s} {'Kept':>5s} {'Blkd':>5s} {'WR%':>6s} {'P/L':>10s} {'SL blkd':>8s}")
        print(f"  {'-'*80}")

        for fname, fmask in top_filters:
            fc = fmask[cm]
            al = ct[~fc]
            bl = ct[fc]
            al_wr = al['win'].sum()/len(al)*100 if len(al) > 0 else 0
            al_pnl = al['pnl_pct'].sum() if len(al) > 0 else 0
            bl_sl = len(bl[bl['exit_reason']=='STOP_LOSS']) if len(bl) > 0 else 0
            mk = ' <<<' if al_pnl > ct_pnl + 30 else ''
            print(f"  {fname:<40s} {len(al):>5d} {len(bl):>5d} "
                  f"{al_wr:>5.1f}% {al_pnl:>+9.1f}% {bl_sl:>7d}{mk}")

    # =====================================================================
    # PART 5: Final recommendation with year-by-year
    # =====================================================================
    print(f"\n\n{'='*130}")
    print("PART 5: YEAR-BY-YEAR -- RECOMMENDED FILTERS")
    print(f"{'='*130}")

    df['year'] = df['entry_date'].dt.year

    recommended = [
        ("Synth VIX>=50", df['synth_vix'].fillna(0) >= 50),
        ("BChg10<-20pp", df['breadth_chg10'].fillna(0) < -20),
        ("2of3: VIX|SL|BChg",
         ((df['synth_vix'].fillna(0) >= 40).astype(int) +
          (df['sl_last_5d'] >= 3).astype(int) +
          (df['breadth_chg10'].fillna(0) < -15).astype(int)) >= 2),
    ]

    print(f"  {'Year':<6s} | {'BASELINE':^18s} | ", end='')
    for fn, _ in recommended:
        print(f"{fn:^18s} | ", end='')
    print()
    print(f"  {'':6s} | {'Trd':>4s} {'WR':>4s} {'P/L':>8s} | ", end='')
    for _ in recommended:
        print(f"{'Trd':>4s} {'WR':>4s} {'P/L':>8s} | ", end='')
    print()
    print(f"  {'-'*90}")

    for yr in sorted(df['year'].unique()):
        base = df[df['year'] == yr]
        if len(base) < 3:
            continue
        b_wr = base['win'].sum()/len(base)*100
        b_pnl = base['pnl_pct'].sum()
        print(f"  {yr:<6d} | {len(base):>4d} {b_wr:>3.0f}% {b_pnl:>+7.1f}% | ", end='')

        for fn, fm in recommended:
            fyr = base[~fm[base.index]]
            if len(fyr) > 0:
                f_wr = fyr['win'].sum()/len(fyr)*100
                f_pnl = fyr['pnl_pct'].sum()
                print(f"{len(fyr):>4d} {f_wr:>3.0f}% {f_pnl:>+7.1f}% | ", end='')
            else:
                print(f"{'--':>4s} {'--':>4s} {'--':>8s} | ", end='')
        print()

    print(f"  {'-'*90}")
    print(f"  {'TOTAL':<6s} | {len(df):>4d} {baseline_wr:>3.0f}% {baseline_pnl:>+7.1f}% | ", end='')
    for fn, fm in recommended:
        filt = df[~fm]
        f_wr = filt['win'].sum()/len(filt)*100
        f_pnl = filt['pnl_pct'].sum()
        print(f"{len(filt):>4d} {f_wr:>3.0f}% {f_pnl:>+7.1f}% | ", end='')
    print()


if __name__ == '__main__':
    main()
