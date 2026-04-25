"""EOD swing momentum breakout — Nifty 500 portfolio backtest.

Long-only, daily bars, multi-day positional holds. Event-driven entries
at next-day open after EOD signal close. Tests breakout type, volume
filter, regime filter, and exit method as orthogonal dimensions around
a baseline configuration.

Baseline:  50-day Donchian high + vol >= 2x + close > 200-SMA + Donch10 exit
           initial 12% protective stop (cap downside on first day)

Variants (11 total) — each changes ONE dimension vs baseline:

  Block A — breakout window
    A1  20-day high       (vs baseline 50-day)
    A2  252-day high      (~52-week high / ATH proxy)

  Block B — volume filter
    B1  no volume filter
    B2  vol >= 1.5x
    B3  vol >= 3.0x

  Block C — regime filter
    C1  no regime filter (no 200-SMA gate)

  Block D — exit method
    D1  fixed 25% target / 8% stop
    D2  ATR trail 3x
    D3  Chandelier 3xATR (highest close since entry − 3xATR)
    D4  Donch20 exit (vs baseline Donch10)

Universe: ~373 Nifty 500 stocks with >=1500 daily bars since 2018-01-01.
Period: 2018-01-01 to 2025-12-31 (~8 years).
Capital: Rs 10,00,000.
Risk per trade: 1% of capital.
Max concurrent positions: 10.
Costs: 0.20% round-trip (delivery STT + brokerage + slippage estimate).
"""
from __future__ import annotations

import csv
import logging
import sqlite3
import sys
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.disable(logging.WARNING); warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
NIFTY500 = ROOT / 'data' / 'nifty500_list.csv'
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(parents=True, exist_ok=True)

START_DATE = '2018-01-01'
END_DATE   = '2025-12-31'
MIN_BARS   = 1500     # universe filter — exclude IPOs / short history

# Capital and sizing
CAPITAL                = 10_00_000
RISK_PER_TRADE_PCT     = 0.01
MAX_CONCURRENT         = 10
COST_PCT               = 0.0020   # 0.20% round-trip
INITIAL_HARD_STOP_PCT  = 0.08     # backstop on every entry — cap day-1 risk

# Indicator params (constants used across variants)
ATR_PERIOD             = 14
SMA_REGIME_PERIOD      = 200
VOL_AVG_PERIOD         = 50

# Variant configs
def make_variant(name, breakout_n=50, vol_mult=2.0, regime=True,
                 exit_kind='donch', exit_donch_n=10,
                 exit_atr_mult=None, exit_target_pct=None, exit_stop_pct=None):
    return {
        'name': name,
        'breakout_n': breakout_n,
        'vol_mult': vol_mult,
        'regime_filter': regime,
        'exit_kind': exit_kind,           # 'donch' | 'atr_trail' | 'chandelier' | 'fixed'
        'exit_donch_n': exit_donch_n,
        'exit_atr_mult': exit_atr_mult,
        'exit_target_pct': exit_target_pct,
        'exit_stop_pct': exit_stop_pct,
    }

VARIANTS = [
    # Baseline first — others sweep one dimension vs this
    make_variant('baseline_50_2x_200_d10'),
    # Block A — breakout window
    make_variant('A1_breakout_20',  breakout_n=20),
    make_variant('A2_breakout_252', breakout_n=252),
    # Block B — volume filter
    make_variant('B1_no_vol',  vol_mult=0.0),
    make_variant('B2_vol_15x', vol_mult=1.5),
    make_variant('B3_vol_30x', vol_mult=3.0),
    # Block C — regime filter
    make_variant('C1_no_regime', regime=False),
    # Block D — exit method
    make_variant('D1_fixed_25_8',  exit_kind='fixed',      exit_target_pct=0.25, exit_stop_pct=0.08),
    make_variant('D2_atr_trail_3', exit_kind='atr_trail',  exit_atr_mult=3.0),
    make_variant('D3_chandelier_3', exit_kind='chandelier', exit_atr_mult=3.0),
    make_variant('D4_donch_20',    exit_kind='donch',       exit_donch_n=20),
]


@dataclass
class Position:
    symbol: str
    entry_date: str
    entry_price: float
    qty: int
    initial_stop: float
    atr_at_entry: float
    breakout_n: int
    highest_close: float = field(init=False)

    def __post_init__(self):
        self.highest_close = self.entry_price


@dataclass
class Trade:
    variant: str; symbol: str
    entry_date: str; exit_date: str
    entry_price: float; exit_price: float
    qty: int; days_held: int
    exit_reason: str
    gross_pnl: float; net_pnl: float


def load_universe() -> list[str]:
    syms = []
    with NIFTY500.open() as f:
        for r in csv.DictReader(f): syms.append(r['Symbol'])
    # Filter to those with sufficient daily bars
    conn = sqlite3.connect(DB)
    ok = []
    for s in syms:
        r = conn.execute("""SELECT COUNT(*) FROM market_data_unified
                            WHERE symbol=? AND timeframe='day' AND date>=?""",
                         (s, START_DATE)).fetchone()
        if r[0] >= MIN_BARS:
            ok.append(s)
    conn.close()
    return ok


def load_all_bars(universe: list[str]) -> dict[str, pd.DataFrame]:
    """Returns {symbol: DataFrame indexed by date with OHLCV + indicators}."""
    conn = sqlite3.connect(DB)
    out = {}
    for sym in universe:
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close, volume FROM market_data_unified
                WHERE symbol=? AND timeframe='day' AND date>=? AND date<=?
             ORDER BY date""",
            conn, params=(sym, START_DATE, END_DATE + ' 23:59:59'),
        )
        if df.empty: continue
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.set_index('date', inplace=True)

        # Indicators (vectorized)
        # Rolling N-day high (excluding today) for breakout detection
        for n in [20, 50, 252]:
            df[f'high_{n}d'] = df['high'].shift(1).rolling(n).max()

        # Rolling N-day low (excluding today) for exit
        for n in [10, 20]:
            df[f'low_{n}d'] = df['low'].shift(1).rolling(n).min()

        # ATR (Wilder)
        prev_close = df['close'].shift(1)
        tr = pd.concat([df['high']-df['low'], (df['high']-prev_close).abs(),
                        (df['low']-prev_close).abs()], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

        # Volume avg (excluding today)
        df['vol_avg'] = df['volume'].shift(1).rolling(VOL_AVG_PERIOD).mean()

        # 200-SMA regime filter
        df['sma_200'] = df['close'].rolling(SMA_REGIME_PERIOD).mean()

        out[sym] = df
    conn.close()
    return out


def get_signal(row: pd.Series, cfg: dict) -> bool:
    """Long signal at this row's close — entry would happen NEXT day's open."""
    high_n = row.get(f'high_{cfg["breakout_n"]}d')
    if pd.isna(high_n): return False
    if not (row['close'] > high_n): return False
    # Volume filter
    if cfg['vol_mult'] > 0:
        if pd.isna(row['vol_avg']) or row['vol_avg'] <= 0: return False
        if not (row['volume'] >= cfg['vol_mult'] * row['vol_avg']): return False
    # Regime filter
    if cfg['regime_filter']:
        if pd.isna(row['sma_200']): return False
        if not (row['close'] > row['sma_200']): return False
    return True


def check_exit(pos: Position, today: pd.Series, today_date, cfg: dict) -> tuple[bool, float, str]:
    """Returns (exit?, exit_price, reason)."""
    # Initial hard stop (first-line defense, always active)
    if today['low'] <= pos.initial_stop:
        return True, pos.initial_stop, 'INITIAL_STOP'

    if cfg['exit_kind'] == 'fixed':
        target = pos.entry_price * (1 + cfg['exit_target_pct'])
        if today['high'] >= target:
            return True, target, 'TARGET'
        # initial stop already covers the % stop level

    elif cfg['exit_kind'] == 'donch':
        n = cfg['exit_donch_n']
        low_n = today.get(f'low_{n}d')
        if not pd.isna(low_n) and today['low'] <= low_n:
            return True, low_n, f'DONCH_{n}_LOW'

    elif cfg['exit_kind'] == 'atr_trail':
        # Trailing stop = highest_close_since_entry - mult * ATR
        trail_stop = pos.highest_close - cfg['exit_atr_mult'] * pos.atr_at_entry
        if today['low'] <= trail_stop:
            return True, max(trail_stop, today['low']), f'ATR_TRAIL_{cfg["exit_atr_mult"]}x'

    elif cfg['exit_kind'] == 'chandelier':
        # Chandelier = highest_close - mult * current ATR
        atr_today = today.get('atr', pos.atr_at_entry)
        if pd.isna(atr_today): atr_today = pos.atr_at_entry
        ch_stop = pos.highest_close - cfg['exit_atr_mult'] * atr_today
        if today['low'] <= ch_stop:
            return True, max(ch_stop, today['low']), f'CHANDELIER_{cfg["exit_atr_mult"]}x'

    return False, 0.0, ''


def run_variant(bars: dict[str, pd.DataFrame], cfg: dict, start_date: str = START_DATE,
                end_date: str = END_DATE) -> tuple[list[Trade], pd.Series]:
    """Run portfolio backtest for one variant. Returns (trades, daily_equity_series)."""
    # Build a master timeline (union of all symbols' trading days)
    all_dates = sorted({d for df in bars.values() for d in df.index
                        if str(d) >= start_date and str(d) <= end_date})
    open_positions: dict[str, Position] = {}
    trades: list[Trade] = []
    equity = CAPITAL
    daily_equity = {}
    pending_entries: list[tuple[str, str, dict]] = []   # (signal_date, symbol, cfg) → enter at next day's open

    # Pre-compute prev-day close for quick lookup (for trailing highs)
    # Actually highest_close is updated daily inside the loop

    for i, today_date in enumerate(all_dates):
        today_date_str = str(today_date)

        # 1) Process pending entries from yesterday — fill at TODAY's open
        new_entries = []
        for sig_date, sym, var_cfg in pending_entries:
            df = bars[sym]
            if today_date not in df.index: continue   # symbol not trading this day
            row = df.loc[today_date]
            entry_px = row['open']
            if not (entry_px > 0): continue
            atr_e = df.loc[sig_date, 'atr'] if sig_date in df.index else row['atr']
            if pd.isna(atr_e) or atr_e <= 0: continue
            # Initial stop = max(entry - 2*ATR, entry * (1 - INITIAL_HARD_STOP_PCT))
            stop_atr = entry_px - 2 * atr_e
            stop_pct = entry_px * (1 - INITIAL_HARD_STOP_PCT)
            initial_stop = max(stop_atr, stop_pct)   # whichever is closer to entry (smaller risk)
            risk_per_share = entry_px - initial_stop
            if risk_per_share <= 0: continue
            risk_rs = equity * RISK_PER_TRADE_PCT
            qty = int(risk_rs // risk_per_share)
            # Notional cap: capital / max_concurrent
            cap_per_pos = equity / MAX_CONCURRENT
            if qty * entry_px > cap_per_pos: qty = int(cap_per_pos // entry_px)
            if qty <= 0: continue
            # Slot check
            if len(open_positions) >= MAX_CONCURRENT: continue
            if sym in open_positions: continue

            pos = Position(
                symbol=sym, entry_date=today_date_str, entry_price=entry_px, qty=qty,
                initial_stop=initial_stop, atr_at_entry=atr_e,
                breakout_n=var_cfg['breakout_n'],
            )
            open_positions[sym] = pos
            new_entries.append(sym)
        pending_entries = []   # cleared

        # 2) Process exits on existing positions
        to_close = []
        for sym, pos in list(open_positions.items()):
            df = bars[sym]
            if today_date not in df.index: continue
            row = df.loc[today_date]
            # Update highest_close (for trail/chandelier)
            if row['close'] > pos.highest_close:
                pos.highest_close = row['close']
            # Skip exit on entry day (gives at least 1 bar)
            if pos.entry_date == today_date_str: continue
            should_exit, exit_px, reason = check_exit(pos, row, today_date, cfg)
            if should_exit:
                to_close.append((sym, exit_px, reason))

        for sym, exit_px, reason in to_close:
            pos = open_positions.pop(sym)
            entry_value = pos.entry_price * pos.qty
            exit_value  = exit_px * pos.qty
            gross_pnl = (exit_px - pos.entry_price) * pos.qty
            cost = COST_PCT * (entry_value + exit_value) / 2.0
            net_pnl = gross_pnl - cost
            equity += net_pnl
            days = (today_date - pd.to_datetime(pos.entry_date).date()).days
            trades.append(Trade(
                variant=cfg['name'], symbol=sym,
                entry_date=pos.entry_date, exit_date=today_date_str,
                entry_price=pos.entry_price, exit_price=exit_px,
                qty=pos.qty, days_held=days, exit_reason=reason,
                gross_pnl=gross_pnl, net_pnl=net_pnl,
            ))

        # 3) Scan for new entry signals (only if slots available)
        if len(open_positions) < MAX_CONCURRENT:
            candidates = []
            for sym, df in bars.items():
                if sym in open_positions: continue
                if today_date not in df.index: continue
                row = df.loc[today_date]
                if get_signal(row, cfg):
                    # Rank by volume spike (stronger conviction first)
                    vspike = (row['volume'] / row['vol_avg']) if (row['vol_avg'] and row['vol_avg'] > 0) else 1.0
                    candidates.append((vspike, sym))
            candidates.sort(reverse=True)
            slots_left = MAX_CONCURRENT - len(open_positions) - len(pending_entries)
            for _, sym in candidates[:slots_left]:
                pending_entries.append((today_date_str, sym, cfg))

        # 4) Mark-to-market: equity = cash + sum(open positions' current value - entry_value)
        unrealized = 0.0
        for sym, pos in open_positions.items():
            df = bars[sym]
            if today_date not in df.index: continue
            close_today = df.loc[today_date, 'close']
            unrealized += (close_today - pos.entry_price) * pos.qty
        daily_equity[today_date_str] = equity + unrealized

    # Close any still-open positions at last available bar
    for sym, pos in list(open_positions.items()):
        df = bars[sym]
        last_date = df.index[-1]
        if last_date < pd.to_datetime(pos.entry_date).date(): continue
        last_close = df.loc[last_date, 'close']
        gross_pnl = (last_close - pos.entry_price) * pos.qty
        cost = COST_PCT * (pos.entry_price * pos.qty + last_close * pos.qty) / 2.0
        net_pnl = gross_pnl - cost
        equity += net_pnl
        days = (last_date - pd.to_datetime(pos.entry_date).date()).days
        trades.append(Trade(
            variant=cfg['name'], symbol=sym,
            entry_date=pos.entry_date, exit_date=str(last_date),
            entry_price=pos.entry_price, exit_price=last_close,
            qty=pos.qty, days_held=days, exit_reason='END_OF_BACKTEST',
            gross_pnl=gross_pnl, net_pnl=net_pnl,
        ))

    eq_series = pd.Series(daily_equity).sort_index()
    return trades, eq_series


def compute_metrics(trades: list[Trade], eq: pd.Series) -> dict:
    if not trades or eq.empty:
        return {k: 0 for k in ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                               'avg_days_held','profit_factor','net_pnl','cagr_pct',
                               'sharpe','max_dd_pct','calmar','final_equity']}
    net_list = [t.net_pnl for t in trades]
    wins = [p for p in net_list if p > 0]; losses = [p for p in net_list if p < 0]
    total_net = sum(net_list)

    # Daily returns from equity series
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0

    # Drawdown
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd_pct = abs(dd.min()) * 100 if not dd.empty else 0

    # CAGR
    n_days = (pd.to_datetime(eq.index[-1]) - pd.to_datetime(eq.index[0])).days
    yrs = n_days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1) * 100 if yrs > 0 and eq.iloc[0] > 0 else 0

    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100*len(wins)/len(trades), 2),
        'avg_win': round(sum(wins)/len(wins), 0) if wins else 0,
        'avg_loss': round(sum(losses)/len(losses), 0) if losses else 0,
        'avg_days_held': round(sum(t.days_held for t in trades)/len(trades), 1),
        'profit_factor': round(sum(wins)/abs(sum(losses)), 2) if losses else 0,
        'net_pnl': round(total_net, 0),
        'cagr_pct': round(cagr, 2),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(max_dd_pct, 2),
        'calmar': round(cagr / max_dd_pct, 2) if max_dd_pct > 0 else 0,
        'final_equity': round(eq.iloc[-1], 0),
    }


def main():
    t_start = time.time()
    print('Loading universe...')
    uni = load_universe()
    print(f'Universe: {len(uni)} stocks')
    print('Loading bars + computing indicators (one-shot)...')
    t0 = time.time()
    bars = load_all_bars(uni)
    print(f'  loaded {len(bars)} stocks in {time.time()-t0:.1f}s')

    summary_keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                    'avg_days_held','profit_factor','net_pnl','cagr_pct','sharpe',
                    'max_dd_pct','calmar','final_equity']

    summary_rows = []
    print()
    print(f'{"Variant":>30} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgDays":>8} {"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7} {"FinalEq":>10}')
    print('-' * 105)
    for cfg in VARIANTS:
        t0 = time.time()
        trades, eq = run_variant(bars, cfg)
        m = compute_metrics(trades, eq)
        summary_rows.append([cfg['name']] + [m.get(k, '') for k in summary_keys])
        print(f'{cfg["name"]:>30} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'{m["avg_days_held"]:>8.1f} {m["cagr_pct"]:>+6.2f} {m["sharpe"]:>7.2f} '
              f'{m["max_dd_pct"]:>7.2f} {m["calmar"]:>7.2f} Rs{m["final_equity"]:>+8,.0f}  ({time.time()-t0:.1f}s)', flush=True)

        # Save trade log + equity per variant
        with (OUT / f'trades_{cfg["name"]}.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['variant','symbol','entry_date','exit_date','entry_price','exit_price',
                        'qty','days_held','exit_reason','gross_pnl','net_pnl'])
            for t in trades:
                w.writerow([t.variant, t.symbol, t.entry_date, t.exit_date,
                            f'{t.entry_price:.2f}', f'{t.exit_price:.2f}',
                            t.qty, t.days_held, t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])
        eq.to_csv(OUT / f'equity_{cfg["name"]}.csv', header=['equity'])

    # Master summary
    with (OUT / 'summary.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant'] + summary_keys)
        w.writerows(summary_rows)

    print(f'\nTotal runtime: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
