"""Two analyses on BB squeeze fires:

1) FOLLOW-THROUGH stat — establish directionality of explosion
   For each fire, classify the first 1xATR move (which direction price hit
   first) and then check: does that direction extend to 2xATR same, or
   reverse 2xATR opposite, or neither?

2) ST(7,2) TRAIL backtest — three variants
   v1 'direct_trail':       enter in breakout direction, trail with ST(7,2),
                            exit on ST flip / EOD / time stop
   v2 'reverse_on_flip':    same as v1 but on ST flip, immediately enter the
                            opposite direction (max 1 reversal per day)
   v3 'wait_for_st':        skip the breakout direction, enter only when ST
                            establishes direction post-fire (within K bars)

Universe: 15 non-ORB stocks + NIFTY50, 5-min bars 2024-03-18 to 2026-03-25.
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
OUT = Path(__file__).resolve().parents[1] / 'results' / 'st_trail'
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = ['NIFTY50',
            'INDUSINDBK','PNB','BANKBARODA','WIPRO','HCLTECH','MARUTI','HEROMOTOCO',
            'EICHERMOT','TITAN','ASIANPAINT','JSWSTEEL','HINDALCO','JINDALSTEL','LT','ADANIPORTS']

START_DATE = '2024-03-18'
END_DATE   = '2026-03-25'

# Indicator periods
BB_PERIOD, BB_K       = 20, 2.0
KC_PERIOD, KC_K       = 20, 1.5
ATR_PERIOD            = 14
MIN_SQZ_BARS          = 6
ST_PERIOD, ST_K       = 7, 2.0  # Supertrend(7, 2)

# Trade rules
ENTRY_WINDOW_START    = dtime(9, 45)
ENTRY_WINDOW_END      = dtime(14, 0)
EOD_EXIT              = dtime(15, 15)
TIME_STOP_BARS        = 30      # let trends run if ST hasn't flipped (150 min)
INITIAL_STOP_ATR_MULT = 1.0     # initial protective stop (in case ST hasn't formed)
ST_CONFIRM_BARS       = 6       # wait_for_st variant: wait up to 6 bars for ST to point

# Sizing + costs
RISK_PER_TRADE        = 2_500
MAX_NOTIONAL          = 300_000
COST_PCT              = 0.0015

# Follow-through analysis params
FT_FIRST_MOVE_ATR     = 1.0     # threshold for "first move" detection
FT_FOLLOW_ATR         = 2.0     # continuation target
FT_LOOKAHEAD_BARS     = 18


@dataclass
class Trade:
    variant: str; symbol: str; date: str; direction: str
    entry_time: str; entry: float; stop: float; qty: int
    atr_at_entry: float
    exit_time: str = ''; exit_price: float = 0.0; exit_reason: str = ''
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df

    # BB
    df['bb_mid']   = df['close'].rolling(BB_PERIOD).mean()
    bb_std         = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_K * bb_std
    df['bb_lower'] = df['bb_mid'] - BB_K * bb_std

    # ATR (Wilder)
    pc = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    # KC
    df['kc_mid']   = df['close'].ewm(span=KC_PERIOD, adjust=False).mean()
    atr_kc         = tr.rolling(KC_PERIOD).mean()
    df['kc_upper'] = df['kc_mid'] + KC_K * atr_kc
    df['kc_lower'] = df['kc_mid'] - KC_K * atr_kc

    # Squeeze + fire
    df['in_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    sq = df['in_squeeze'].astype(int)
    df['sqz_bars'] = sq * (sq.groupby((sq != sq.shift()).cumsum()).cumcount() + 1)
    df['fire'] = (~df['in_squeeze']) & (df['in_squeeze'].shift(1).fillna(False)) \
                 & (df['sqz_bars'].shift(1).fillna(0) >= MIN_SQZ_BARS)
    df['break_up']   = df['fire'] & (df['close'] > df['bb_upper'])
    df['break_down'] = df['fire'] & (df['close'] < df['bb_lower'])

    # Supertrend(ST_PERIOD, ST_K)
    atr_st = tr.ewm(alpha=1/ST_PERIOD, adjust=False).mean()
    hl2 = (df['high'] + df['low']) / 2.0
    upper_basic = hl2 + ST_K * atr_st
    lower_basic = hl2 - ST_K * atr_st

    n = len(df)
    final_upper = np.zeros(n); final_lower = np.zeros(n)
    st_dir = np.ones(n, dtype=int)  # 1 = uptrend, -1 = downtrend
    st_line = np.zeros(n)
    closes = df['close'].values
    upper_basic_v = upper_basic.values; lower_basic_v = lower_basic.values

    for i in range(n):
        if i == 0:
            final_upper[i] = upper_basic_v[i]
            final_lower[i] = lower_basic_v[i]
            st_dir[i] = 1
            st_line[i] = lower_basic_v[i]
            continue
        # Final upper: lowest path
        if (upper_basic_v[i] < final_upper[i-1]) or (closes[i-1] > final_upper[i-1]):
            final_upper[i] = upper_basic_v[i]
        else:
            final_upper[i] = final_upper[i-1]
        # Final lower: highest path
        if (lower_basic_v[i] > final_lower[i-1]) or (closes[i-1] < final_lower[i-1]):
            final_lower[i] = lower_basic_v[i]
        else:
            final_lower[i] = final_lower[i-1]
        # Direction
        if st_dir[i-1] == 1 and closes[i] < final_lower[i]:
            st_dir[i] = -1
        elif st_dir[i-1] == -1 and closes[i] > final_upper[i]:
            st_dir[i] = 1
        else:
            st_dir[i] = st_dir[i-1]
        st_line[i] = final_lower[i] if st_dir[i] == 1 else final_upper[i]
    df['st_dir'] = st_dir
    df['st_line'] = st_line
    df['st_flipped_up']   = (df['st_dir'] == 1)  & (df['st_dir'].shift(1).fillna(1) == -1)
    df['st_flipped_down'] = (df['st_dir'] == -1) & (df['st_dir'].shift(1).fillna(1) == 1)

    return df


# ---------- ANALYSIS 1: follow-through ----------

def followthrough_stats(df: pd.DataFrame, symbol: str) -> list[dict]:
    """For each fire, classify first 1xATR move and follow-through outcome."""
    in_window = (df.index.time >= ENTRY_WINDOW_START) & (df.index.time <= ENTRY_WINDOW_END)
    fire_mask = df['fire'].values & in_window
    fire_pos = np.where(fire_mask)[0]

    arr_close = df['close'].values
    arr_high  = df['high'].values
    arr_low   = df['low'].values
    arr_atr   = df['atr'].values

    rows = []
    for i in fire_pos:
        if i + FT_LOOKAHEAD_BARS >= len(df): continue
        atr = arr_atr[i]
        if not (atr > 0): continue
        entry = arr_close[i]
        is_up_fire = bool(df['break_up'].iloc[i])
        is_down_fire = bool(df['break_down'].iloc[i])
        if not (is_up_fire or is_down_fire): continue
        fire_dir = 'UP' if is_up_fire else 'DOWN'

        # Walk forward — find which direction first reaches +/-1xATR
        first_move_dir = None
        first_move_bar = None
        for j in range(1, FT_LOOKAHEAD_BARS + 1):
            up_d = (arr_high[i+j] - entry) / atr
            dn_d = (entry - arr_low[i+j]) / atr
            if up_d >= FT_FIRST_MOVE_ATR and first_move_dir is None:
                first_move_dir = 'UP'; first_move_bar = j; break
            if dn_d >= FT_FIRST_MOVE_ATR and first_move_dir is None:
                first_move_dir = 'DOWN'; first_move_bar = j; break

        if first_move_dir is None:
            outcome = 'NEITHER'
            continues = False; reverses = False
        else:
            # From the first-move bar onwards, does price reach 2xATR in same dir?
            # OR does it reverse and reach 2xATR opposite from entry?
            continues = False; reverses = False
            for j in range(first_move_bar, FT_LOOKAHEAD_BARS + 1):
                up_d = (arr_high[i+j] - entry) / atr
                dn_d = (entry - arr_low[i+j]) / atr
                if first_move_dir == 'UP':
                    if up_d >= FT_FOLLOW_ATR: continues = True; break
                    if dn_d >= FT_FOLLOW_ATR: reverses = True; break
                else:
                    if dn_d >= FT_FOLLOW_ATR: continues = True; break
                    if up_d >= FT_FOLLOW_ATR: reverses = True; break
            if continues:
                outcome = 'FOLLOW_THROUGH'
            elif reverses:
                outcome = 'REVERSE'
            else:
                outcome = 'STALL'

        rows.append({
            'symbol': symbol,
            'fire_dir': fire_dir,
            'first_move_dir': first_move_dir,
            'first_move_aligned': (first_move_dir == fire_dir) if first_move_dir else False,
            'outcome': outcome,
            'continues': continues,
            'reverses': reverses,
        })
    return rows


# ---------- ANALYSIS 2: ST trail backtest ----------

def run_st_trail(df: pd.DataFrame, symbol: str, variant: str) -> tuple[list[Trade], dict]:
    """Three variants:
       direct_trail:    take breakout direction, exit on ST flip
       reverse_on_flip: take breakout direction; on ST flip enter opposite once
       wait_for_st:     skip breakout direction, enter when ST aligns post-fire
    """
    trades = []
    daily = {}
    in_window = (df.index.time >= ENTRY_WINDOW_START) & (df.index.time <= ENTRY_WINDOW_END)

    for day, day_df in df.groupby('day'):
        active = None; bars_held = 0; reversed_today = False
        pending = None  # ('LONG'/'SHORT', atr, fire_dir)
        watching_st = None  # for wait_for_st: ('UP'/'DOWN', expiry_bar)
        rows_iter = list(day_df.itertuples())

        for idx, row in enumerate(rows_iter):
            ts = row.Index; t = ts.time()

            # Fill pending entry on this bar's open
            if pending is not None and active is None:
                direction, atr_e, fire_dir = pending
                entry_px = row.open
                if pd.notna(entry_px) and atr_e > 0:
                    if direction == 'LONG':
                        stop = entry_px - INITIAL_STOP_ATR_MULT * atr_e
                        rps = entry_px - stop
                    else:
                        stop = entry_px + INITIAL_STOP_ATR_MULT * atr_e
                        rps = stop - entry_px
                    if rps > 0:
                        qty = int(RISK_PER_TRADE // rps)
                        if qty * entry_px > MAX_NOTIONAL: qty = int(MAX_NOTIONAL // entry_px)
                        if qty > 0:
                            active = Trade(variant, symbol, day.isoformat(), direction, ts.isoformat(),
                                           entry_px, stop, qty, atr_e)
                            bars_held = 0
                pending = None

            # Exit logic
            if active is not None:
                bars_held += 1
                # Hard initial stop (in case ST hasn't formed yet or bad gap)
                hit_stop = (active.direction == 'LONG'  and row.low  <= active.stop) \
                        or (active.direction == 'SHORT' and row.high >= active.stop)
                # ST flip exit
                st_flip = (active.direction == 'LONG'  and bool(row.st_flipped_down)) \
                       or (active.direction == 'SHORT' and bool(row.st_flipped_up))
                # Trailing stop based on ST line — tighten only
                if active.direction == 'LONG' and row.st_dir == 1 and row.st_line > active.stop:
                    active.stop = float(row.st_line)
                elif active.direction == 'SHORT' and row.st_dir == -1 and row.st_line < active.stop:
                    active.stop = float(row.st_line)

                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP')
                    trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl
                    flipped_dir = active.direction
                    active = None
                    # reverse_on_flip variant: enter opposite if ST flipped (not just normal stop)
                    if variant == 'reverse_on_flip' and st_flip and not reversed_today:
                        atr_now = row.atr if pd.notna(row.atr) else 0
                        if atr_now > 0:
                            new_dir = 'SHORT' if flipped_dir == 'LONG' else 'LONG'
                            pending = (new_dir, atr_now, 'REV')
                            reversed_today = True
                elif st_flip:
                    active.close(ts.isoformat(), row.close, 'ST_FLIP')
                    trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl
                    flipped_dir = active.direction
                    active = None
                    if variant == 'reverse_on_flip' and not reversed_today:
                        atr_now = row.atr if pd.notna(row.atr) else 0
                        if atr_now > 0:
                            new_dir = 'SHORT' if flipped_dir == 'LONG' else 'LONG'
                            pending = (new_dir, atr_now, 'REV')
                            reversed_today = True
                elif bars_held >= TIME_STOP_BARS:
                    active.close(ts.isoformat(), row.close, 'TIME_STOP')
                    trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), row.close, 'EOD')
                    trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl
                    active = None

            # Watch-for-ST entry (variant 3)
            if watching_st is not None and active is None and pending is None:
                fire_dir, expire = watching_st
                if idx > expire:
                    watching_st = None
                else:
                    if int(row.st_dir) == 1:
                        atr_now = row.atr if pd.notna(row.atr) else 0
                        if atr_now > 0:
                            pending = ('LONG', atr_now, fire_dir)
                            watching_st = None
                    elif int(row.st_dir) == -1:
                        atr_now = row.atr if pd.notna(row.atr) else 0
                        if atr_now > 0:
                            pending = ('SHORT', atr_now, fire_dir)
                            watching_st = None

            # Detect new squeeze fire
            if (active is None and pending is None and watching_st is None
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END
                    and pd.notna(row.atr) and row.atr > 0):
                if bool(row.break_up):
                    if variant == 'wait_for_st':
                        watching_st = ('UP', idx + ST_CONFIRM_BARS)
                    else:
                        pending = ('LONG', row.atr, 'UP')
                elif bool(row.break_down):
                    if variant == 'wait_for_st':
                        watching_st = ('DOWN', idx + ST_CONFIRM_BARS)
                    else:
                        pending = ('SHORT', row.atr, 'DOWN')

        if active is not None and rows_iter:
            last = rows_iter[-1]
            active.close(last.Index.isoformat(), last.close, 'EOD_FORCED')
            trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl

    return trades, daily


def compute_metrics(trades, daily):
    if not trades:
        return {k: 0 for k in ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                               'profit_factor','net_pnl','sharpe','max_dd_pct']}
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
    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100*len(wins)/len(trades), 2),
        'avg_win': round(sum(wins)/len(wins), 0) if wins else 0,
        'avg_loss': round(sum(losses)/len(losses), 0) if losses else 0,
        'profit_factor': round(sum(wins)/abs(sum(losses)), 2) if losses else 0,
        'net_pnl': round(sum(net), 0),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(100*mdd/cap, 2),
    }


def main():
    t_start = time.time()
    ft_rows = []
    bt_trades = {'direct_trail': [], 'reverse_on_flip': [], 'wait_for_st': []}
    bt_daily  = {'direct_trail': {}, 'reverse_on_flip': {}, 'wait_for_st': {}}

    for i, sym in enumerate(UNIVERSE, 1):
        t0 = time.time()
        df = load_5min(sym)
        if df.empty: continue
        df = add_indicators(df)

        # Analysis 1: follow-through
        ft_rows.extend(followthrough_stats(df, sym))

        # Analysis 2: backtests
        for variant in bt_trades.keys():
            trades, daily = run_st_trail(df, sym, variant)
            bt_trades[variant].extend(trades)
            for d, p in daily.items():
                bt_daily[variant][d] = bt_daily[variant].get(d, 0) + p

        elapsed = time.time() - t0
        print(f'[{i:2d}/{len(UNIVERSE)}] {sym:12s} fires={int(df["fire"].sum()):>3}  '
              f'direct={len([t for t in bt_trades["direct_trail"] if t.symbol==sym]):>3}/Rs{sum(t.net_pnl for t in bt_trades["direct_trail"] if t.symbol==sym):>+8,.0f}  '
              f'rev={len([t for t in bt_trades["reverse_on_flip"] if t.symbol==sym]):>3}/Rs{sum(t.net_pnl for t in bt_trades["reverse_on_flip"] if t.symbol==sym):>+8,.0f}  '
              f'wait={len([t for t in bt_trades["wait_for_st"] if t.symbol==sym]):>3}/Rs{sum(t.net_pnl for t in bt_trades["wait_for_st"] if t.symbol==sym):>+8,.0f}  ({elapsed:.1f}s)', flush=True)

    # Save follow-through
    ft_df = pd.DataFrame(ft_rows)
    ft_df.to_csv(OUT / 'followthrough.csv', index=False)

    # ----- ANALYSIS 1: Follow-through summary
    print()
    print('=' * 90)
    print('ANALYSIS 1: FOLLOW-THROUGH (1xATR first move -> 2xATR continuation vs reverse)')
    print('=' * 90)
    total = len(ft_df)
    aligned = (ft_df['first_move_aligned']).sum()
    print(f'Total fires: {total}')
    print(f'  First-move aligned with breakout direction: {aligned} ({100*aligned/total:.1f}%)')
    print(f'  First-move opposite to breakout direction:  {total-aligned} ({100*(total-aligned)/total:.1f}%)')
    print()
    print('Outcome categories (after first-move detection):')
    for outcome in ['FOLLOW_THROUGH', 'REVERSE', 'STALL', 'NEITHER']:
        n = (ft_df['outcome']==outcome).sum()
        print(f'  {outcome:18s}: {n:>5} ({100*n/total:>5.1f}%)')
    print()
    print('Critical: of fires where first move WAS in breakout direction,')
    print('what % continue to 2xATR same direction vs reverse 2xATR opposite?')
    print()
    aligned_df = ft_df[ft_df['first_move_aligned']==True]
    if len(aligned_df) > 0:
        cont = (aligned_df['outcome']=='FOLLOW_THROUGH').sum()
        rev  = (aligned_df['outcome']=='REVERSE').sum()
        stall= (aligned_df['outcome']=='STALL').sum()
        print(f'  Of {len(aligned_df)} fires with first-move aligned:')
        print(f'    Continues to 2x same dir:    {cont:>4} ({100*cont/len(aligned_df):.1f}%)')
        print(f'    Reverses to 2x opposite dir: {rev:>4}  ({100*rev/len(aligned_df):.1f}%)')
        print(f'    Stalls (neither):            {stall:>4}  ({100*stall/len(aligned_df):.1f}%)')
    print()
    opposite_df = ft_df[ft_df['first_move_aligned']==False]
    opposite_df = opposite_df[opposite_df['first_move_dir'].notna()]
    if len(opposite_df) > 0:
        cont = (opposite_df['outcome']=='FOLLOW_THROUGH').sum()
        rev  = (opposite_df['outcome']=='REVERSE').sum()
        stall= (opposite_df['outcome']=='STALL').sum()
        print(f'  Of {len(opposite_df)} fires with first-move OPPOSITE to breakout:')
        print(f'    Continues to 2x same as first move: {cont:>4} ({100*cont/len(opposite_df):.1f}%)')
        print(f'    Reverses (back to breakout dir):    {rev:>4}  ({100*rev/len(opposite_df):.1f}%)')
        print(f'    Stalls:                             {stall:>4}  ({100*stall/len(opposite_df):.1f}%)')

    # ----- ANALYSIS 2: ST trail backtest summary
    print()
    print('=' * 90)
    print('ANALYSIS 2: ST(7,2) TRAIL BACKTESTS')
    print('=' * 90)
    print(f'{"Variant":18s} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgWin":>9} {"AvgLoss":>9} {"Net P&L":>13} {"Sharpe":>7} {"MaxDD%":>7}')
    print('-' * 90)
    rows = []
    for v in ['direct_trail', 'reverse_on_flip', 'wait_for_st']:
        m = compute_metrics(bt_trades[v], bt_daily[v])
        print(f'{v:18s} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'Rs{m["avg_win"]:>+7,.0f} Rs{m["avg_loss"]:>+7,.0f} Rs{m["net_pnl"]:>+11,.0f} {m["sharpe"]:>7.2f} {m["max_dd_pct"]:>7.2f}')
        rows.append({'variant': v, **m})

    pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)

    # Trade log
    with (OUT / 'trades.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','date','direction','entry_time','entry','stop',
                    'qty','atr_at_entry','exit_time','exit_price','exit_reason','gross_pnl','net_pnl'])
        for v, tr in bt_trades.items():
            for t in tr:
                w.writerow([t.variant, t.symbol, t.date, t.direction, t.entry_time,
                            f'{t.entry:.2f}', f'{t.stop:.2f}', t.qty,
                            f'{t.atr_at_entry:.2f}',
                            t.exit_time, f'{t.exit_price:.2f}', t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

    print(f'\nRuntime: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
