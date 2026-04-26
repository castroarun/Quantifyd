"""
KC6 60-min Tier 2 sweep
=======================
Base: EOD mode, KC mult=1.6, TP 4%, 93 stocks, 2018-2026.

Dimensions:
  entry_confirm : False | True   (True = signal bar must reclaim back inside band)
  sl_type       : 'fixed_2pct' | 'atr_1.0' | 'atr_1.5' | 'atr_2.0'
  crash_filter  : False | True   (daily universe ATR ratio < 1.3)

16 configs total.
"""

import sqlite3
import time as _t

import numpy as np
import pandas as pd

DB = 'backtest_data/market_data.db'

KC_EMA = 6
KC_ATR_P = 6
KC_MULT = 1.6
SMA_PERIOD = 200
TP_PCT = 0.04
FIXED_SL_PCT = 0.02
COST_EOD = 0.0020
CRASH_THRESHOLD = 1.3
DAILY_ATR_LOOKBACK = 14
DAILY_ATR_AVG_WINDOW = 50


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


def add_indicators(df):
    df = df.copy()
    df['kc_mid'] = ema(df['close'], KC_EMA)
    a = atr(df['high'], df['low'], df['close'], KC_ATR_P)
    df['kc_upper'] = df['kc_mid'] + KC_MULT * a
    df['kc_lower'] = df['kc_mid'] - KC_MULT * a
    df['sma200'] = sma(df['close'], SMA_PERIOD)
    df['atr'] = a  # ATR6 for entry-time SL calc
    return df


def load_60min(conn):
    t0 = _t.time()
    syms = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='60minute' ORDER BY symbol"
    ).fetchall()]
    syms = [s for s in syms if s not in ('NIFTY50', 'BANKNIFTY', 'NIFTY', 'FINNIFTY')]
    print(f"Loading {len(syms)} symbols (60min)...", flush=True)
    data = {}
    for sym in syms:
        q = ("SELECT date, open, high, low, close, volume FROM market_data_unified "
             "WHERE symbol=? AND timeframe='60minute' ORDER BY date")
        df = pd.read_sql(q, conn, params=[sym])
        if len(df) < 300:
            continue
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
        data[sym] = add_indicators(df)
    print(f"  60min: {len(data)} symbols in {_t.time()-t0:.1f}s", flush=True)
    return data


def compute_daily_crash_series(conn, symbols):
    """For each trading day, compute median ATR-ratio across universe -> dict[date]=ratio."""
    t0 = _t.time()
    # Load daily data for the same symbols
    daily = {}
    for sym in symbols:
        q = ("SELECT date, open, high, low, close, volume FROM market_data_unified "
             "WHERE symbol=? AND timeframe='day' ORDER BY date")
        df = pd.read_sql(q, conn, params=[sym])
        if len(df) < DAILY_ATR_AVG_WINDOW + DAILY_ATR_LOOKBACK + 10:
            continue
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.set_index('date', inplace=True)
        for c in ['open', 'high', 'low', 'close']:
            df[c] = df[c].astype(float)
        a14 = atr(df['high'], df['low'], df['close'], DAILY_ATR_LOOKBACK)
        a50avg = a14.rolling(DAILY_ATR_AVG_WINDOW).mean()
        ratio = a14 / a50avg
        daily[sym] = ratio

    # Align on union of dates
    combined = pd.concat(daily.values(), axis=1)
    combined.columns = list(daily.keys())
    # Per-day median across valid stocks
    daily_median = combined.median(axis=1)
    ratio_map = {d: float(v) for d, v in daily_median.items() if pd.notna(v)}
    active_days = sum(1 for v in ratio_map.values() if v >= CRASH_THRESHOLD)
    print(f"  Daily crash series: {len(ratio_map)} days in {_t.time()-t0:.1f}s "
          f"({active_days} days crash-active, {active_days/len(ratio_map)*100:.1f}%)", flush=True)
    return ratio_map


def backtest_symbol(df, sym, cfg, ratio_map):
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
    atrs = df['atr'].values
    ts = df.index

    entry_confirm = cfg['entry_confirm']
    sl_type = cfg['sl_type']
    crash_on = cfg['crash_filter']

    for i in range(n):
        if np.isnan(sma200s[i]) or np.isnan(kc_lowers[i]):
            continue

        # Exit logic
        if position is not None:
            entry = position['entry']
            side = position['side']
            sl_price = position['sl_price']
            exit_price = None
            reason = None

            if side == 'LONG':
                tp = entry * (1 + TP_PCT)
                mid = kc_mids[i]
                if lows[i] <= sl_price:
                    exit_price, reason = sl_price, 'SL'
                elif highs[i] >= tp:
                    exit_price, reason = tp, 'TP'
                elif not np.isnan(mid) and highs[i] >= mid and mid > entry:
                    exit_price, reason = mid, 'MID'
                elif last_bar_mask[i]:
                    exit_price, reason = closes[i], 'EOD'
            else:
                tp = entry * (1 - TP_PCT)
                mid = kc_mids[i]
                if highs[i] >= sl_price:
                    exit_price, reason = sl_price, 'SL'
                elif lows[i] <= tp:
                    exit_price, reason = tp, 'TP'
                elif not np.isnan(mid) and lows[i] <= mid and mid < entry:
                    exit_price, reason = mid, 'MID'
                elif last_bar_mask[i]:
                    exit_price, reason = closes[i], 'EOD'

            if exit_price is not None:
                if side == 'LONG':
                    gross = exit_price / entry - 1
                else:
                    gross = entry / exit_price - 1
                net = gross - COST_EOD
                trades.append({
                    'symbol': sym, 'side': side,
                    'entry_time': position['entry_time'], 'exit_time': ts[i],
                    'entry': round(entry, 2), 'exit': round(exit_price, 2),
                    'gross_pct': gross, 'net_pct': net, 'reason': reason,
                })
                position = None

        # Entry logic
        if position is None and not last_bar_mask[i]:
            # Crash filter
            if crash_on:
                r = ratio_map.get(days[i])
                if r is not None and r >= CRASH_THRESHOLD:
                    continue

            c = closes[i]
            lo = kc_lowers[i]
            up = kc_uppers[i]
            s200 = sma200s[i]

            want_long = False
            want_short = False

            if entry_confirm:
                # Prior bar closed outside band, current bar reclaimed
                if i == 0:
                    continue
                pc = closes[i - 1]
                plo = kc_lowers[i - 1]
                pup = kc_uppers[i - 1]
                # Long: prev close < prev lower AND curr close >= curr lower AND curr close > sma
                if pc < plo and c >= lo and c > s200:
                    want_long = True
                elif pc > pup and c <= up and c < s200:
                    want_short = True
            else:
                if c < lo and c > s200:
                    want_long = True
                elif c > up and c < s200:
                    want_short = True

            if want_long or want_short:
                # Compute SL price
                if sl_type == 'fixed_2pct':
                    sl_price = c * (1 - FIXED_SL_PCT) if want_long else c * (1 + FIXED_SL_PCT)
                else:
                    mult = float(sl_type.split('_')[1])
                    a = atrs[i]
                    if np.isnan(a) or a <= 0:
                        continue
                    sl_price = c - mult * a if want_long else c + mult * a

                position = {
                    'side': 'LONG' if want_long else 'SHORT',
                    'entry': c, 'entry_time': ts[i], 'sl_price': sl_price,
                }

    return trades


def summarize_cfg(cfg_label, all_trades):
    n = len(all_trades)
    if n == 0:
        return {'config': cfg_label, 'trades': 0, 'net_pct': 0, 'pf': 0, 'wr': 0,
                'mid_pf': 0, 'sl_hit_pct': 0}
    df = pd.DataFrame(all_trades)
    wins = df[df['net_pct'] > 0]
    losses = df[df['net_pct'] <= 0]
    wr = len(wins) / n
    net = df['net_pct'].sum()
    gsum_w = wins['net_pct'].sum()
    gsum_l = losses['net_pct'].sum()
    pf = -gsum_w / gsum_l if gsum_l < 0 else (float('inf') if gsum_w > 0 else 0)

    mid = df[df['reason'] == 'MID']
    mid_pf = 0
    if len(mid):
        mw = mid[mid['net_pct'] > 0]['net_pct'].sum()
        ml = mid[mid['net_pct'] <= 0]['net_pct'].sum()
        mid_pf = -mw / ml if ml < 0 else (float('inf') if mw > 0 else 0)

    sl_count = (df['reason'] == 'SL').sum()
    sl_hit_pct = sl_count / n * 100

    return {
        'config': cfg_label,
        'trades': n,
        'wr_pct': round(wr*100, 1),
        'net_pct': round(net*100, 1),
        'pf': round(pf, 2),
        'mid_pf': round(mid_pf, 1) if mid_pf != float('inf') else 999.0,
        'sl_hit_pct': round(sl_hit_pct, 1),
    }


def main():
    conn = sqlite3.connect(DB)
    data = load_60min(conn)
    ratio_map = compute_daily_crash_series(conn, list(data.keys()))
    conn.close()

    # Build 16 configs
    configs = []
    for ec in [False, True]:
        for sl in ['fixed_2pct', 'atr_1.0', 'atr_1.5', 'atr_2.0']:
            for cf in [False, True]:
                label = f"EC{'Y' if ec else 'N'}_SL{sl}_CF{'Y' if cf else 'N'}"
                configs.append({
                    'label': label,
                    'entry_confirm': ec, 'sl_type': sl, 'crash_filter': cf,
                })

    summaries = []
    for cfg in configs:
        t0 = _t.time()
        all_trades = []
        for sym, df in data.items():
            all_trades.extend(backtest_symbol(df, sym, cfg, ratio_map))
        s = summarize_cfg(cfg['label'], all_trades)
        s['elapsed'] = round(_t.time() - t0, 1)
        summaries.append(s)
        print(f"{cfg['label']:42s} N={s['trades']:5d} "
              f"WR={s['wr_pct']:5.1f}% Net={s['net_pct']:+7.1f}% "
              f"PF={s['pf']:5.2f} MidPF={s['mid_pf']:6.1f} SL%={s['sl_hit_pct']:4.1f} "
              f"[{s['elapsed']:.1f}s]", flush=True)
        # Save per-config CSV only if interesting
        if s['pf'] >= 1.0 and s['trades'] > 0:
            pd.DataFrame(all_trades).to_csv(f"backtest_kc6_60min_tier2_{cfg['label']}.csv", index=False)

    print("\n" + "=" * 100)
    print("TIER 2 RANKING BY PF")
    print("=" * 100)
    sdf = pd.DataFrame(summaries).sort_values('pf', ascending=False)
    print(sdf.to_string(index=False))
    sdf.to_csv('backtest_kc6_60min_tier2_summary.csv', index=False)
    print("\nSaved: backtest_kc6_60min_tier2_summary.csv")

    # Highlight winners
    winners = sdf[sdf['pf'] >= 1.3]
    print(f"\n{'-'*100}")
    if len(winners):
        print(f"*** GATE CLEARED: {len(winners)} config(s) with PF >= 1.3 ***")
        print(winners.to_string(index=False))
    else:
        best = sdf.iloc[0]
        print(f"Gate NOT cleared. Best config: {best['config']} @ PF={best['pf']}")


if __name__ == '__main__':
    main()
