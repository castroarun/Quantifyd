"""v4 — First-move commit entry + ST trail variants.

Per analyze_followthrough_st.py finding: of all squeeze fires that move 1xATR
in either direction within ~60 min, ~62% follow through to 2xATR same direction.
The breakout flag itself is meaningless (48.8% aligned with first move). The
edge is conditional on the first 1xATR move.

This script tests the entry rule that follows from that finding:
  1. Squeeze fires on 5-min bars (existing detection)
  2. Watch up to N bars for first 1xATR move in either direction
  3. Enter that direction at next bar's open
  4. Exit per variant: fixed target/stop OR ST trail

Variants (10 total):
  Bar-TF   Exit            Notes
  5min     fixed_1.5R      1xATR stop, 1.5xATR target
  5min     fixed_2.0R      1xATR stop, 2.0xATR target
  5min     ST(7,2)         original — known too tight
  5min     ST(10,3)
  5min     ST(14,3)
  5min     ST(21,3)
  10min    fixed_1.5R
  10min    ST(7,2)
  10min    ST(14,3)
  10min    ST(21,3)

Universe: 15 non-ORB stocks + NIFTY50, 2024-03-18 to 2026-03-25.
"""
from __future__ import annotations

import csv
import logging
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

logging.disable(logging.WARNING); warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = Path(__file__).resolve().parents[1] / 'results' / 'v4_first_move'
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = ['NIFTY50',
            'INDUSINDBK','PNB','BANKBARODA','WIPRO','HCLTECH','MARUTI','HEROMOTOCO',
            'EICHERMOT','TITAN','ASIANPAINT','JSWSTEEL','HINDALCO','JINDALSTEL','LT','ADANIPORTS']

START_DATE = '2024-03-18'
END_DATE   = '2026-03-25'

# Squeeze detection (always 5-min)
BB_PERIOD, BB_K       = 20, 2.0
KC_PERIOD, KC_K       = 20, 1.5
ATR_PERIOD            = 14
MIN_SQZ_BARS          = 6

# First-move trigger
FIRST_MOVE_ATR        = 1.0

# Trade rules
ENTRY_WINDOW_START    = dtime(9, 45)
ENTRY_WINDOW_END      = dtime(14, 0)
EOD_EXIT              = dtime(15, 15)
INITIAL_STOP_ATR_MULT = 1.0

# Sizing + costs
RISK_PER_TRADE        = 2_500
MAX_NOTIONAL          = 300_000
COST_PCT              = 0.0015

# Variant configs — (bar_tf, exit_method, params)
# bar_tf:    '5min' or '10min' — the TF for first-move watch and (if applicable) ST trail
# watch_bars: how many bars after fire to wait for 1xATR commit (40 min on each TF)
# time_stop_bars: how long to hold after entry before forcing exit
VARIANTS = {
    '5m_fixed_1.5R':  {'bar_tf': '5min',  'watch_bars': 8,  'time_stop': 16, 'exit': 'fixed', 'r_mult': 1.5},
    '5m_fixed_2.0R':  {'bar_tf': '5min',  'watch_bars': 8,  'time_stop': 16, 'exit': 'fixed', 'r_mult': 2.0},
    '5m_st_7_2':      {'bar_tf': '5min',  'watch_bars': 8,  'time_stop': 16, 'exit': 'st', 'st_p': 7,  'st_k': 2.0},
    '5m_st_10_3':     {'bar_tf': '5min',  'watch_bars': 8,  'time_stop': 16, 'exit': 'st', 'st_p': 10, 'st_k': 3.0},
    '5m_st_14_3':     {'bar_tf': '5min',  'watch_bars': 8,  'time_stop': 16, 'exit': 'st', 'st_p': 14, 'st_k': 3.0},
    '5m_st_21_3':     {'bar_tf': '5min',  'watch_bars': 8,  'time_stop': 16, 'exit': 'st', 'st_p': 21, 'st_k': 3.0},
    '10m_fixed_1.5R': {'bar_tf': '10min', 'watch_bars': 4,  'time_stop': 8,  'exit': 'fixed', 'r_mult': 1.5},
    '10m_st_7_2':     {'bar_tf': '10min', 'watch_bars': 4,  'time_stop': 8,  'exit': 'st', 'st_p': 7,  'st_k': 2.0},
    '10m_st_14_3':    {'bar_tf': '10min', 'watch_bars': 4,  'time_stop': 8,  'exit': 'st', 'st_p': 14, 'st_k': 3.0},
    '10m_st_21_3':    {'bar_tf': '10min', 'watch_bars': 4,  'time_stop': 8,  'exit': 'st', 'st_p': 21, 'st_k': 3.0},
}


@dataclass
class Trade:
    variant: str; symbol: str; date: str; direction: str
    fire_time: str; entry_time: str; entry: float; stop: float
    qty: int; atr_at_entry: float; first_move_dir: str
    exit_time: str = ''; exit_price: float = 0.0; exit_reason: str = ''
    bars_to_commit: int = 0
    gross_pnl: float = 0.0; net_pnl: float = 0.0

    def close(self, exit_time, exit_price, reason):
        self.exit_time, self.exit_price, self.exit_reason = exit_time, exit_price, reason
        sign = 1 if self.direction == 'LONG' else -1
        self.gross_pnl = sign * (exit_price - self.entry) * self.qty
        cost = COST_PCT * (self.entry + exit_price) * self.qty / 2.0
        self.net_pnl = self.gross_pnl - cost


def load_5min(symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
            WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=?
         ORDER BY date""",
        conn, params=(symbol, START_DATE, END_DATE + ' 23:59:59'),
    )
    conn.close()
    if df.empty: return df
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['day'] = df.index.date
    return df


def add_5min_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Squeeze detection + ATR — always on 5-min."""
    df['bb_mid']   = df['close'].rolling(BB_PERIOD).mean()
    bb_std         = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_K * bb_std
    df['bb_lower'] = df['bb_mid'] - BB_K * bb_std
    pc = df['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    df['kc_mid'] = df['close'].ewm(span=KC_PERIOD, adjust=False).mean()
    atr_kc = tr.rolling(KC_PERIOD).mean()
    df['kc_upper'] = df['kc_mid'] + KC_K * atr_kc
    df['kc_lower'] = df['kc_mid'] - KC_K * atr_kc
    df['in_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    sq = df['in_squeeze'].astype(int)
    df['sqz_bars'] = sq * (sq.groupby((sq != sq.shift()).cumsum()).cumcount() + 1)
    df['fire'] = (~df['in_squeeze']) & (df['in_squeeze'].shift(1).fillna(False)) \
                 & (df['sqz_bars'].shift(1).fillna(0) >= MIN_SQZ_BARS)
    return df


def resample_to_tf(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 5-min OHLCV to target freq. Carries day column over."""
    if freq == '5min':
        return df.copy()
    out = df[['open','high','low','close','volume']].resample(freq, label='right', closed='right').agg(
        {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
    ).dropna()
    out['day'] = out.index.date
    return out


def add_st(df: pd.DataFrame, st_period: int, st_k: float) -> pd.DataFrame:
    """Compute Supertrend on the dataframe (any TF). Adds st_dir and st_line columns."""
    pc = df['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/st_period, adjust=False).mean()
    hl2 = (df['high'] + df['low']) / 2.0
    upper_basic = (hl2 + st_k * atr).values
    lower_basic = (hl2 - st_k * atr).values

    n = len(df)
    final_upper = np.zeros(n); final_lower = np.zeros(n)
    st_dir = np.ones(n, dtype=int); st_line = np.zeros(n)
    closes = df['close'].values

    for i in range(n):
        if i == 0:
            final_upper[i] = upper_basic[i]; final_lower[i] = lower_basic[i]
            st_dir[i] = 1; st_line[i] = lower_basic[i]; continue
        if (upper_basic[i] < final_upper[i-1]) or (closes[i-1] > final_upper[i-1]):
            final_upper[i] = upper_basic[i]
        else:
            final_upper[i] = final_upper[i-1]
        if (lower_basic[i] > final_lower[i-1]) or (closes[i-1] < final_lower[i-1]):
            final_lower[i] = lower_basic[i]
        else:
            final_lower[i] = final_lower[i-1]
        if st_dir[i-1] == 1 and closes[i] < final_lower[i]:
            st_dir[i] = -1
        elif st_dir[i-1] == -1 and closes[i] > final_upper[i]:
            st_dir[i] = 1
        else:
            st_dir[i] = st_dir[i-1]
        st_line[i] = final_lower[i] if st_dir[i] == 1 else final_upper[i]

    df['st_dir']  = st_dir
    df['st_line'] = st_line
    return df


def find_fire_idx_in_tf(df_tf: pd.DataFrame, fire_times: list[pd.Timestamp]) -> list[int]:
    """For each 5-min fire timestamp, find the first index in df_tf at or after it."""
    idx_arr = df_tf.index
    out = []
    for ft in fire_times:
        # Find first bar that closes AT OR AFTER fire time
        pos = idx_arr.searchsorted(ft)
        if pos >= len(idx_arr):
            out.append(None)
        else:
            out.append(pos)
    return out


def run_variant(df5: pd.DataFrame, symbol: str, variant: str, cfg: dict) -> tuple[list[Trade], dict]:
    if df5.empty: return [], {}

    # Resample if needed
    df_tf = resample_to_tf(df5, cfg['bar_tf'])

    # ATR on the trade TF (used for stops/targets and first-move threshold)
    pc = df_tf['close'].shift(1)
    tr_tf = pd.concat([df_tf['high']-df_tf['low'], (df_tf['high']-pc).abs(), (df_tf['low']-pc).abs()], axis=1).max(axis=1)
    df_tf['atr'] = tr_tf.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    # ST if needed
    if cfg['exit'] == 'st':
        df_tf = add_st(df_tf, cfg['st_p'], cfg['st_k'])
        df_tf['st_flipped_up']   = (df_tf['st_dir'] == 1)  & (df_tf['st_dir'].shift(1).fillna(1) == -1)
        df_tf['st_flipped_down'] = (df_tf['st_dir'] == -1) & (df_tf['st_dir'].shift(1).fillna(1) == 1)

    # Identify fire times on 5-min
    in_window = (df5.index.time >= ENTRY_WINDOW_START) & (df5.index.time <= ENTRY_WINDOW_END)
    fire_times = df5.index[df5['fire'].values & in_window].tolist()

    # Map to df_tf positions
    fire_positions = find_fire_idx_in_tf(df_tf, fire_times)

    trades: list[Trade] = []
    daily: dict = {}

    arr_ts = df_tf.index
    arr_open = df_tf['open'].values
    arr_high = df_tf['high'].values
    arr_low  = df_tf['low'].values
    arr_close = df_tf['close'].values
    arr_atr = df_tf['atr'].values
    arr_day = df_tf['day'].values
    arr_st_dir = df_tf['st_dir'].values if cfg['exit'] == 'st' else None
    arr_st_line = df_tf['st_line'].values if cfg['exit'] == 'st' else None
    arr_flip_up = df_tf['st_flipped_up'].values if cfg['exit'] == 'st' else None
    arr_flip_down = df_tf['st_flipped_down'].values if cfg['exit'] == 'st' else None

    n = len(df_tf)

    # Track which days have already had a trade triggered (1 trade/stock/day)
    days_with_trade = set()

    for fp_idx, fpos in enumerate(fire_positions):
        if fpos is None: continue
        fire_day = arr_day[fpos] if fpos < n else None
        if fire_day in days_with_trade: continue
        if fpos + 1 >= n: continue  # Need at least one more bar to detect first-move

        fire_ts = fire_times[fp_idx]
        fire_close = arr_close[fpos]
        fire_atr = arr_atr[fpos]
        if not (fire_atr > 0): continue
        first_move_thresh = FIRST_MOVE_ATR * fire_atr

        # Watch next watch_bars for first-move commit
        commit_pos = None
        commit_dir = None
        commit_price = None  # price at which threshold first hit
        watch_end = min(n, fpos + 1 + cfg['watch_bars'])
        for j in range(fpos + 1, watch_end):
            high_dist = arr_high[j] - fire_close
            low_dist  = fire_close - arr_low[j]
            if high_dist >= first_move_thresh:
                commit_pos = j; commit_dir = 'LONG'
                commit_price = fire_close + first_move_thresh
                break
            if low_dist >= first_move_thresh:
                commit_pos = j; commit_dir = 'SHORT'
                commit_price = fire_close - first_move_thresh
                break

        if commit_pos is None: continue
        # Entry at NEXT bar's open (after commit bar). If commit was last in window, skip.
        entry_pos = commit_pos + 1
        if entry_pos >= n: continue
        if arr_day[entry_pos] != fire_day: continue  # don't carry across days

        entry_px = arr_open[entry_pos]
        if not (entry_px > 0): continue

        # Set up stop, target
        atr_e = arr_atr[entry_pos] if arr_atr[entry_pos] > 0 else fire_atr
        if commit_dir == 'LONG':
            stop = entry_px - INITIAL_STOP_ATR_MULT * atr_e
            target = entry_px + cfg.get('r_mult', 1.5) * INITIAL_STOP_ATR_MULT * atr_e if cfg['exit'] == 'fixed' else None
            rps = entry_px - stop
        else:
            stop = entry_px + INITIAL_STOP_ATR_MULT * atr_e
            target = entry_px - cfg.get('r_mult', 1.5) * INITIAL_STOP_ATR_MULT * atr_e if cfg['exit'] == 'fixed' else None
            rps = stop - entry_px

        if rps <= 0: continue
        qty = int(RISK_PER_TRADE // rps)
        if qty * entry_px > MAX_NOTIONAL: qty = int(MAX_NOTIONAL // entry_px)
        if qty <= 0: continue

        active = Trade(variant, symbol, str(fire_day), commit_dir,
                       fire_ts.isoformat(), arr_ts[entry_pos].isoformat(),
                       entry_px, stop, qty, atr_e, commit_dir,
                       bars_to_commit=commit_pos - fpos)
        days_with_trade.add(fire_day)

        # Walk forward bars to manage trade
        bars_held = 0
        end_bound = min(n, entry_pos + cfg['time_stop'])
        exited = False
        for j in range(entry_pos + 1, end_bound + 1):
            if j >= n: break
            bars_held += 1
            ts = arr_ts[j]; t = ts.time()
            day_j = arr_day[j]
            if day_j != fire_day:
                # End-of-day forced
                active.close(arr_ts[j-1].isoformat(), arr_close[j-1], 'EOD_FORCED')
                exited = True
                break

            # Update ST trail (long: keep stop = max(stop, st_line if st_dir==1))
            if cfg['exit'] == 'st':
                sd = arr_st_dir[j]; sl = arr_st_line[j]
                if active.direction == 'LONG' and sd == 1 and sl > active.stop:
                    active.stop = float(sl)
                elif active.direction == 'SHORT' and sd == -1 and sl < active.stop:
                    active.stop = float(sl)

            hit_stop = (active.direction == 'LONG'  and arr_low[j]  <= active.stop) \
                    or (active.direction == 'SHORT' and arr_high[j] >= active.stop)

            if cfg['exit'] == 'fixed':
                hit_tgt  = (active.direction == 'LONG'  and arr_high[j] >= target) \
                        or (active.direction == 'SHORT' and arr_low[j]  <= target)
                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP'); exited = True; break
                if hit_tgt:
                    active.close(ts.isoformat(), target, 'TARGET'); exited = True; break
            else:  # ST trail
                st_flip = (active.direction == 'LONG'  and bool(arr_flip_down[j])) \
                       or (active.direction == 'SHORT' and bool(arr_flip_up[j]))
                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP'); exited = True; break
                if st_flip:
                    active.close(ts.isoformat(), arr_close[j], 'ST_FLIP'); exited = True; break

            if t >= EOD_EXIT:
                active.close(ts.isoformat(), arr_close[j], 'EOD'); exited = True; break

        if not exited:
            # Time-stop or run-off end of data
            j_last = min(end_bound, n - 1)
            active.close(arr_ts[j_last].isoformat(), arr_close[j_last], 'TIME_STOP')

        trades.append(active)
        daily[active.date] = daily.get(active.date, 0) + active.net_pnl

    return trades, daily


def compute_metrics(trades, daily):
    if not trades:
        return {k: 0 for k in ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                               'profit_factor','net_pnl','sharpe','max_dd_pct','calmar']}
    net = [t.net_pnl for t in trades]
    wins = [p for p in net if p > 0]; losses = [p for p in net if p < 0]
    sd = sorted(daily.keys())
    series = [daily[d] for d in sd]
    r = peak = mdd = 0.0
    for p in series:
        r += p
        if r > peak: peak = r
        mdd = max(mdd, peak - r)
    n = len(series); sharpe = 0.0
    if n > 1:
        m = sum(series)/n; s = (sum((x-m)**2 for x in series)/(n-1))**0.5
        if s > 0: sharpe = (m/s) * (252**0.5)
    cap = 300_000
    yrs = n / 252 if n else 1
    end = cap + sum(net)
    cagr = ((end/cap)**(1/yrs) - 1) * 100 if yrs > 0 and end > 0 else 0.0
    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100*len(wins)/len(trades), 2),
        'avg_win': round(sum(wins)/len(wins), 0) if wins else 0,
        'avg_loss': round(sum(losses)/len(losses), 0) if losses else 0,
        'profit_factor': round(sum(wins)/abs(sum(losses)), 2) if losses else 0,
        'net_pnl': round(sum(net), 0),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(100*mdd/cap, 2),
        'cagr_pct': round(cagr, 2),
        'calmar': round(cagr / (100*mdd/cap), 2) if mdd > 0 else 0,
    }


def main():
    t_start = time.time()
    bt_trades = {v: [] for v in VARIANTS}
    bt_daily  = {v: {} for v in VARIANTS}

    for i, sym in enumerate(UNIVERSE, 1):
        t0 = time.time()
        df5 = load_5min(sym)
        if df5.empty: continue
        df5 = add_5min_indicators(df5)
        n_fires = int(df5['fire'].sum())

        parts = [f'[{i:2d}/{len(UNIVERSE)}] {sym:12s} fires={n_fires:>3}']
        for vname, vcfg in VARIANTS.items():
            trades, daily = run_variant(df5, sym, vname, vcfg)
            bt_trades[vname].extend(trades)
            for d, p in daily.items(): bt_daily[vname][d] = bt_daily[vname].get(d, 0) + p
            net = sum(t.net_pnl for t in trades)
            parts.append(f'{vname[:10]:<10}={len(trades):>3}/Rs{net:>+7,.0f}')
        elapsed = time.time() - t0
        print('  '.join(parts) + f'  ({elapsed:.1f}s)', flush=True)

    # Trade log
    with (OUT / 'trades.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','date','direction','fire_time','entry_time','entry','stop',
                    'qty','atr_at_entry','first_move_dir','bars_to_commit',
                    'exit_time','exit_price','exit_reason','gross_pnl','net_pnl'])
        for v, tr in bt_trades.items():
            for t in tr:
                w.writerow([t.variant, t.symbol, t.date, t.direction, t.fire_time, t.entry_time,
                            f'{t.entry:.2f}', f'{t.stop:.2f}', t.qty, f'{t.atr_at_entry:.2f}',
                            t.first_move_dir, t.bars_to_commit,
                            t.exit_time, f'{t.exit_price:.2f}', t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss','profit_factor',
            'net_pnl','sharpe','max_dd_pct','cagr_pct','calmar']
    with (OUT / 'summary.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant'] + keys)
        for vname in VARIANTS:
            m = compute_metrics(bt_trades[vname], bt_daily[vname])
            w.writerow([vname] + [m.get(k, '') for k in keys])

    # Print summary
    print()
    print('=' * 110)
    print(f'V4 FIRST-MOVE COMMIT ENTRY — {len(UNIVERSE)} symbols, 5-min squeeze, 2024-03-18 to 2026-03-25')
    print('=' * 110)
    print(f'{"Variant":17s} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgWin":>8} {"AvgLoss":>9} {"Net P&L":>13} {"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}')
    print('-' * 110)
    for vname in VARIANTS:
        m = compute_metrics(bt_trades[vname], bt_daily[vname])
        print(f'{vname:17s} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'Rs{m["avg_win"]:>+6,.0f} Rs{m["avg_loss"]:>+7,.0f} Rs{m["net_pnl"]:>+11,.0f} '
              f'{m["cagr_pct"]:>+6.2f} {m["sharpe"]:>7.2f} {m["max_dd_pct"]:>7.2f} {m["calmar"]:>7.2f}')
    print(f'\nRuntime: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
