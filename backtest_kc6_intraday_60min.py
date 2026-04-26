"""
KC6 Intraday/Swing Backtest - Phase 2 (60-min)
==============================================
95 stocks, 2018-2026 (~8 years), longs + shorts.

Tier 1 sweep (6 configs):
  Mode    : EOD squareoff vs overnight hold (MaxHold 15 days / 105 bars)
  KC mult : 1.0 / 1.3 / 1.6

Fixed: SL=2%, TP=4%, KC EMA=6, KC ATR=6, SMA=200

Costs:
  EOD mode (MIS long+short): 0.20% round-trip
  Overnight mode (CNC long + F&O futures short): 0.15% round-trip

Signal:
  Long:  close < KC_lower AND close > SMA200
  Short: close > KC_upper AND close < SMA200

Exits (both sides):
  SL / TP / KC_mid touch / EOD (if eod_mode) / MaxHold
"""

import sqlite3
import time as _t

import numpy as np
import pandas as pd

DB = 'backtest_data/market_data.db'
TIMEFRAME = '60minute'

SL_PCT = 0.02       # 2% stop
TP_PCT = 0.04       # 4% target
KC_EMA = 6
KC_ATR_P = 6
SMA_PERIOD = 200
MAX_HOLD_BARS = 105  # ~15 trading days x 7 bars/session

COST_EOD = 0.0020
COST_OVN = 0.0015

CONFIGS = []
for mode in ['EOD', 'OVERNIGHT']:
    for mult in [1.0, 1.3, 1.6]:
        CONFIGS.append({'mode': mode, 'kc_mult': mult})


def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def sma(s, p):
    return s.rolling(p).mean()


def atr(h, l, c, p):
    hl = h - l
    hc = (h - c.shift(1)).abs()
    lc = (l - c.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def add_indicators(df, kc_mult):
    df = df.copy()
    df['kc_mid'] = ema(df['close'], KC_EMA)
    a = atr(df['high'], df['low'], df['close'], KC_ATR_P)
    df['kc_upper'] = df['kc_mid'] + kc_mult * a
    df['kc_lower'] = df['kc_mid'] - kc_mult * a
    df['sma200'] = sma(df['close'], SMA_PERIOD)
    return df


def load_all(conn):
    """Load all 60-min symbols once."""
    t0 = _t.time()
    syms = [r[0] for r in conn.execute(
        f"SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='{TIMEFRAME}' ORDER BY symbol"
    ).fetchall()]
    # Skip indices
    syms = [s for s in syms if s not in ('NIFTY50', 'BANKNIFTY', 'NIFTY', 'FINNIFTY')]
    print(f"Loading {len(syms)} symbols ({TIMEFRAME})...", flush=True)

    all_data = {}
    for sym in syms:
        q = (
            "SELECT date, open, high, low, close, volume FROM market_data_unified "
            f"WHERE symbol=? AND timeframe='{TIMEFRAME}' ORDER BY date"
        )
        df = pd.read_sql(q, conn, params=[sym])
        if len(df) < 300:
            continue
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        all_data[sym] = df
    print(f"  Loaded {len(all_data)} symbols in {_t.time()-t0:.1f}s", flush=True)
    return all_data


def backtest_symbol(df, sym, mode, cost_rt):
    """df already has indicators. Returns list of trade dicts."""
    trades = []
    position = None

    days = df.index.date
    n = len(df)
    last_bar_mask = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if days[i] != days[i + 1]:
            last_bar_mask[i] = True
    last_bar_mask[-1] = True

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    kc_lowers = df['kc_lower'].values
    kc_uppers = df['kc_upper'].values
    kc_mids = df['kc_mid'].values
    sma200s = df['sma200'].values
    ts = df.index

    for i in range(n):
        if np.isnan(sma200s[i]) or np.isnan(kc_lowers[i]):
            continue

        if position is not None:
            entry = position['entry']
            side = position['side']
            bars_held = i - position['entry_idx']
            exit_price = None
            reason = None

            if side == 'LONG':
                sl = entry * (1 - SL_PCT)
                tp = entry * (1 + TP_PCT)
                mid = kc_mids[i]
                if lows[i] <= sl:
                    exit_price, reason = sl, 'SL'
                elif highs[i] >= tp:
                    exit_price, reason = tp, 'TP'
                elif not np.isnan(mid) and highs[i] >= mid and mid > entry:
                    exit_price, reason = mid, 'MID'
                elif mode == 'EOD' and last_bar_mask[i]:
                    exit_price, reason = closes[i], 'EOD'
                elif bars_held >= MAX_HOLD_BARS:
                    exit_price, reason = closes[i], 'MAXHOLD'
            else:  # SHORT
                sl = entry * (1 + SL_PCT)
                tp = entry * (1 - TP_PCT)
                mid = kc_mids[i]
                if highs[i] >= sl:
                    exit_price, reason = sl, 'SL'
                elif lows[i] <= tp:
                    exit_price, reason = tp, 'TP'
                elif not np.isnan(mid) and lows[i] <= mid and mid < entry:
                    exit_price, reason = mid, 'MID'
                elif mode == 'EOD' and last_bar_mask[i]:
                    exit_price, reason = closes[i], 'EOD'
                elif bars_held >= MAX_HOLD_BARS:
                    exit_price, reason = closes[i], 'MAXHOLD'

            if exit_price is not None:
                if side == 'LONG':
                    gross = exit_price / entry - 1
                else:
                    gross = entry / exit_price - 1
                net = gross - cost_rt
                trades.append({
                    'symbol': sym, 'side': side,
                    'entry_time': position['entry_time'], 'exit_time': ts[i],
                    'entry': round(entry, 2), 'exit': round(exit_price, 2),
                    'bars_held': bars_held,
                    'gross_pct': gross, 'net_pct': net, 'reason': reason,
                })
                position = None

        if position is None:
            # Don't enter on last bar of day in EOD mode
            if mode == 'EOD' and last_bar_mask[i]:
                continue
            c = closes[i]
            lo = kc_lowers[i]
            up = kc_uppers[i]
            s200 = sma200s[i]
            if c < lo and c > s200:
                position = {'side': 'LONG', 'entry': c, 'entry_time': ts[i], 'entry_idx': i}
            elif c > up and c < s200:
                position = {'side': 'SHORT', 'entry': c, 'entry_time': ts[i], 'entry_idx': i}

    return trades


def stats_line(label, df):
    n = len(df)
    if n == 0:
        return f"{label:32s} N=      0"
    wins = df[df['net_pct'] > 0]
    losses = df[df['net_pct'] <= 0]
    wr = len(wins) / n
    avg_w = wins['net_pct'].mean() if len(wins) else 0
    avg_l = losses['net_pct'].mean() if len(losses) else 0
    net = df['net_pct'].sum()
    gsum_w = wins['net_pct'].sum()
    gsum_l = losses['net_pct'].sum()
    pf = -gsum_w / gsum_l if gsum_l < 0 else (float('inf') if gsum_w > 0 else 0)
    return (f"{label:32s} N={n:6d} WR={wr*100:5.1f}% "
            f"AvgW={avg_w*100:+.2f}% AvgL={avg_l*100:+.2f}% "
            f"Net={net*100:+8.1f}% PF={pf:5.2f}")


def run_config(cfg, all_data):
    mode = cfg['mode']
    mult = cfg['kc_mult']
    cost_rt = COST_EOD if mode == 'EOD' else COST_OVN
    label = f"{mode}_mult{mult}"

    t0 = _t.time()
    all_trades = []
    for sym, raw_df in all_data.items():
        df = add_indicators(raw_df, mult)
        trades = backtest_symbol(df, sym, mode, cost_rt)
        all_trades.extend(trades)

    tdf = pd.DataFrame(all_trades)
    elapsed = _t.time() - t0
    print(f"\n=== CONFIG: {label}  ({elapsed:.1f}s, {len(tdf)} trades) ===")
    if len(tdf) == 0:
        return tdf, {'config': label, 'trades': 0}

    print(stats_line('ALL', tdf))
    print(stats_line('LONG', tdf[tdf['side'] == 'LONG']))
    print(stats_line('SHORT', tdf[tdf['side'] == 'SHORT']))
    print("By exit reason:")
    for r in ['MID', 'SL', 'TP', 'EOD', 'MAXHOLD']:
        sub = tdf[tdf['reason'] == r]
        if len(sub):
            print('  ' + stats_line(r, sub))

    # Save per-config CSV
    out = f'backtest_kc6_60min_{label}.csv'
    tdf.to_csv(out, index=False)

    # Summary row
    n = len(tdf)
    wins = tdf[tdf['net_pct'] > 0]
    losses = tdf[tdf['net_pct'] <= 0]
    wr = len(wins) / n if n else 0
    net = tdf['net_pct'].sum()
    gsum_w = wins['net_pct'].sum()
    gsum_l = losses['net_pct'].sum()
    pf = -gsum_w / gsum_l if gsum_l < 0 else (float('inf') if gsum_w > 0 else 0)
    return tdf, {
        'config': label, 'mode': mode, 'kc_mult': mult,
        'trades': n, 'win_rate_pct': round(wr*100, 1),
        'net_pct': round(net*100, 1), 'pf': round(pf, 2),
    }


def main():
    conn = sqlite3.connect(DB)
    all_data = load_all(conn)
    conn.close()

    summaries = []
    for cfg in CONFIGS:
        _, s = run_config(cfg, all_data)
        summaries.append(s)

    print("\n" + "="*90)
    print("PHASE 2 CONFIG COMPARISON")
    print("="*90)
    sdf = pd.DataFrame(summaries).sort_values('pf', ascending=False)
    print(sdf.to_string(index=False))
    sdf.to_csv('backtest_kc6_60min_summary.csv', index=False)
    print("\nSaved summary: backtest_kc6_60min_summary.csv")


if __name__ == '__main__':
    main()
