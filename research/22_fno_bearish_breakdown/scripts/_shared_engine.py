"""Shared multi-signal F&O backtest engine.

Used by research/22 .. research/27 to avoid duplicating engine code.

Core abstraction:
  - Caller provides a `signal_fn(df, idx, cfg) -> bool` that returns True when
    the bar at `idx` triggers an entry.
  - Caller specifies `direction` ('LONG' or 'SHORT') per variant.
  - Caller provides indicator-precompute hooks (added to `df` once at load time).

Universe: F&O 76 symbols, mirror of research/21.
Period:   2018-01-01 to 2025-12-31.
Capital:  Rs 10,00,000.
Risk/trade: 1%. Max concurrent: 10. Cost: 0.20% round-trip.

Stop / target geometry mirrors research/21:
  LONG:  stop  = max(entry - 2*ATR, entry * (1 - hard_stop_pct))
         target = entry * (1 + target_pct)
  SHORT: stop  = min(entry + 2*ATR, entry * (1 + hard_stop_pct))
         target = entry * (1 - target_pct)
  Max hold: 60 days (safety).

Indicators precomputed and cached on each symbol's DataFrame:
  high_n (rolling N-day high excluding today, configurable),
  low_n  (rolling N-day low  excluding today, configurable),
  atr (Wilder, 14),
  vol_avg (50-day, excluding today),
  sma_50, sma_200, ema_20, ema_50,
  rsi_14, adx_14 (+DI / -DI),
  macd_hist (EMA12 - EMA26, signal EMA9 of that),
  stoch_k, stoch_d (14, 3),
  rs_30d, rs_60d  (stock 60d return MINUS Nifty50 60d return; same for 30d),
  rs_30d_z, rs_60d_z (z-score of those over a 252-bar rolling window),
"""
from __future__ import annotations

import csv
import logging
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'

# Make services/ importable for FNO_LOT_SIZES
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants (mirror of research/21)
# ---------------------------------------------------------------------------
START_DATE = '2018-01-01'
END_DATE   = '2025-12-31'
MIN_BARS   = 1500

CAPITAL                = 10_00_000
RISK_PER_TRADE_PCT     = 0.01
MAX_CONCURRENT         = 10
DEFAULT_COST_PCT       = 0.0020
INITIAL_HARD_STOP_PCT  = 0.08
MAX_HOLD_DAYS          = 60

ATR_PERIOD             = 14
SMA_REGIME_PERIOD      = 200
VOL_AVG_PERIOD         = 50
DEFAULT_BREAKOUT_N     = 252

NIFTY_PROXY = 'NIFTYBEES'   # full-history Nifty50 ETF (NIFTY50 only has 2023+)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Position:
    symbol: str
    direction: str            # 'LONG' or 'SHORT'
    entry_date: str
    entry_price: float
    qty: int
    initial_stop: float
    atr_at_entry: float
    target_price: float
    highest_close: float = field(init=False)
    lowest_close:  float = field(init=False)

    def __post_init__(self):
        self.highest_close = self.entry_price
        self.lowest_close  = self.entry_price


@dataclass
class Trade:
    variant: str
    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    qty: int
    days_held: int
    exit_reason: str
    gross_pnl: float
    net_pnl: float


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def build_universe_csv(out_path: Path) -> list[str]:
    """Reuse research/21 universe (76 F&O names with >=1500 bars)."""
    from services.data_manager import FNO_LOT_SIZES
    candidates = sorted(FNO_LOT_SIZES.keys())

    conn = sqlite3.connect(DB)
    rows, keep = [], []
    for sym in candidates:
        r = conn.execute(
            """SELECT COUNT(*), MIN(date), MAX(date)
                 FROM market_data_unified
                WHERE symbol=? AND timeframe='day' AND date>=?""",
            (sym, START_DATE),
        ).fetchone()
        n, dmin, dmax = (r[0], r[1], r[2]) if r else (0, None, None)
        rows.append((sym, n, dmin, dmax))
        if n >= MIN_BARS:
            keep.append(sym)
    conn.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['symbol', 'bars', 'date_min', 'date_max', 'kept'])
        for sym, n, dmin, dmax in rows:
            w.writerow([sym, n, dmin or '', dmax or '', 'Y' if n >= MIN_BARS else 'N'])
    return keep


def load_universe(universe_csv: Path) -> list[str]:
    if not universe_csv.exists():
        return build_universe_csv(universe_csv)
    keep = []
    with universe_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get('kept', 'N') == 'Y':
                keep.append(r['symbol'])
    if not keep:
        return build_universe_csv(universe_csv)
    return keep


# ---------------------------------------------------------------------------
# Indicator pre-compute
# ---------------------------------------------------------------------------
def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = _wilder_ema(up, period)
    roll_dn = _wilder_ema(down, period)
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(high, low, close, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = _wilder_ema(tr, period)
    plus_di = 100 * _wilder_ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_ema(minus_dm, period) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder_ema(dx, period)
    return adx, plus_di, minus_di


def _macd_hist(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig  = macd.ewm(span=9, adjust=False).mean()
    return macd - sig


def _stoch(high, low, close, k_period: int = 14, d_period: int = 3):
    lo = low.rolling(k_period).min()
    hi = high.rolling(k_period).max()
    k = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def _load_nifty_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """Returns DataFrame indexed by date with 30d/60d Nifty returns (using NIFTYBEES proxy)."""
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        """SELECT date, close FROM market_data_unified
            WHERE symbol=? AND timeframe='day' AND date>=? AND date<=?
         ORDER BY date""",
        conn, params=(NIFTY_PROXY, start_date, end_date + ' 23:59:59'),
    )
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.set_index('date', inplace=True)
    df['nifty_ret_30'] = df['close'].pct_change(30)
    df['nifty_ret_60'] = df['close'].pct_change(60)
    return df[['nifty_ret_30', 'nifty_ret_60']]


def load_all_bars(universe: list[str],
                  breakout_n: int = DEFAULT_BREAKOUT_N) -> dict[str, pd.DataFrame]:
    """Load OHLCV + precompute every indicator a signal_fn might need."""
    nifty_df = _load_nifty_returns(START_DATE, END_DATE)
    conn = sqlite3.connect(DB)
    out: dict[str, pd.DataFrame] = {}
    for sym in universe:
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close, volume FROM market_data_unified
                WHERE symbol=? AND timeframe='day' AND date>=? AND date<=?
             ORDER BY date""",
            conn, params=(sym, START_DATE, END_DATE + ' 23:59:59'),
        )
        if df.empty:
            continue
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.set_index('date', inplace=True)

        # Donchian-style channels (excluding today)
        df[f'high_{breakout_n}d'] = df['high'].shift(1).rolling(breakout_n).max()
        df[f'low_{breakout_n}d']  = df['low'].shift(1).rolling(breakout_n).min()
        df['high_55d'] = df['high'].shift(1).rolling(55).max()
        df['low_55d']  = df['low'].shift(1).rolling(55).min()

        # ATR (Wilder)
        prev_close = df['close'].shift(1)
        tr = pd.concat([df['high'] - df['low'],
                        (df['high'] - prev_close).abs(),
                        (df['low']  - prev_close).abs()], axis=1).max(axis=1)
        df['atr'] = _wilder_ema(tr, ATR_PERIOD)
        df['atr_pct'] = df['atr'] / df['close']

        # Volume averages
        df['vol_avg'] = df['volume'].shift(1).rolling(VOL_AVG_PERIOD).mean()

        # Moving averages
        df['sma_50']  = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['ema_20']  = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50']  = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # RSI / ADX / MACD / Stoch
        df['rsi_14'] = _rsi(df['close'], 14)
        adx, plus_di, minus_di = _adx(df['high'], df['low'], df['close'], 14)
        df['adx_14']   = adx
        df['plus_di']  = plus_di
        df['minus_di'] = minus_di
        df['macd_hist'] = _macd_hist(df['close'])
        k, d = _stoch(df['high'], df['low'], df['close'], 14, 3)
        df['stoch_k'] = k
        df['stoch_d'] = d

        # Returns
        df['ret_30'] = df['close'].pct_change(30)
        df['ret_60'] = df['close'].pct_change(60)

        # Relative strength (vs Nifty proxy) — z-scored over rolling 252 bars
        if not nifty_df.empty:
            joined = df.join(nifty_df, how='left')
            df['rs_30'] = joined['ret_30'] - joined['nifty_ret_30']
            df['rs_60'] = joined['ret_60'] - joined['nifty_ret_60']
        else:
            df['rs_30'] = np.nan
            df['rs_60'] = np.nan
        rs30_mu  = df['rs_30'].rolling(252).mean()
        rs30_sig = df['rs_30'].rolling(252).std()
        df['rs_30_z'] = (df['rs_30'] - rs30_mu) / rs30_sig.replace(0, np.nan)
        rs60_mu  = df['rs_60'].rolling(252).mean()
        rs60_sig = df['rs_60'].rolling(252).std()
        df['rs_60_z'] = (df['rs_60'] - rs60_mu) / rs60_sig.replace(0, np.nan)

        out[sym] = df
    conn.close()
    return out


# ---------------------------------------------------------------------------
# Backtest engine — direction-aware
# ---------------------------------------------------------------------------
def _check_exit_long(pos: Position, today: pd.Series, days_held: int) -> tuple[bool, float, str]:
    if today['low'] <= pos.initial_stop:
        return True, pos.initial_stop, 'INITIAL_STOP'
    if today['high'] >= pos.target_price:
        return True, pos.target_price, 'TARGET'
    if days_held >= MAX_HOLD_DAYS:
        return True, today['close'], 'MAX_HOLD'
    return False, 0.0, ''


def _check_exit_short(pos: Position, today: pd.Series, days_held: int) -> tuple[bool, float, str]:
    if today['high'] >= pos.initial_stop:
        return True, pos.initial_stop, 'INITIAL_STOP'
    if today['low'] <= pos.target_price:
        return True, pos.target_price, 'TARGET'
    if days_held >= MAX_HOLD_DAYS:
        return True, today['close'], 'MAX_HOLD'
    return False, 0.0, ''


def run_engine(bars: dict[str, pd.DataFrame],
               cfg: dict,
               signal_fn,
               direction: str,
               start_date: str = START_DATE,
               end_date: str = END_DATE,
               candidate_rank_fn=None):
    """Generic engine. signal_fn(row, cfg) -> bool. direction in {'LONG','SHORT'}."""
    cost_pct      = cfg.get('cost_pct', DEFAULT_COST_PCT)
    target_pct    = cfg.get('target_pct', 0.25)
    hard_stop_pct = cfg.get('hard_stop_pct', INITIAL_HARD_STOP_PCT)

    all_dates = sorted({d for df in bars.values() for d in df.index
                        if str(d) >= start_date and str(d) <= end_date})
    open_positions: dict[str, Position] = {}
    trades: list[Trade] = []
    equity = CAPITAL
    daily_equity = {}
    pending_entries: list[tuple[str, str]] = []

    for today_date in all_dates:
        today_str = str(today_date)

        # 1) Fill pending entries at today's open
        for sig_date, sym in pending_entries:
            df = bars[sym]
            if today_date not in df.index:
                continue
            row = df.loc[today_date]
            entry_px = row['open']
            if not (entry_px > 0):
                continue
            sig_d = pd.to_datetime(sig_date).date()
            atr_e = df.loc[sig_d, 'atr'] if sig_d in df.index else row['atr']
            if pd.isna(atr_e) or atr_e <= 0:
                continue

            if direction == 'LONG':
                stop_atr = entry_px - 2 * atr_e
                stop_floor = entry_px * (1 - hard_stop_pct)
                initial_stop = max(stop_atr, stop_floor)
                risk_per_share = entry_px - initial_stop
                target_price  = entry_px * (1 + target_pct)
            else:  # SHORT
                stop_atr = entry_px + 2 * atr_e
                stop_floor = entry_px * (1 + hard_stop_pct)
                initial_stop = min(stop_atr, stop_floor)
                risk_per_share = initial_stop - entry_px
                target_price  = entry_px * (1 - target_pct)

            if risk_per_share <= 0:
                continue
            risk_rs = equity * RISK_PER_TRADE_PCT
            qty = int(risk_rs // risk_per_share)
            cap_per_pos = equity / MAX_CONCURRENT
            if qty * entry_px > cap_per_pos:
                qty = int(cap_per_pos // entry_px)
            if qty <= 0:
                continue
            if len(open_positions) >= MAX_CONCURRENT:
                continue
            if sym in open_positions:
                continue

            pos = Position(symbol=sym, direction=direction, entry_date=today_str,
                           entry_price=entry_px, qty=qty,
                           initial_stop=initial_stop, atr_at_entry=atr_e,
                           target_price=target_price)
            open_positions[sym] = pos
        pending_entries = []

        # 2) Process exits
        to_close = []
        for sym, pos in list(open_positions.items()):
            df = bars[sym]
            if today_date not in df.index:
                continue
            row = df.loc[today_date]
            if row['close'] > pos.highest_close:
                pos.highest_close = row['close']
            if row['close'] < pos.lowest_close:
                pos.lowest_close = row['close']
            if pos.entry_date == today_str:
                continue
            entry_d = pd.to_datetime(pos.entry_date).date()
            days_held = (today_date - entry_d).days
            if pos.direction == 'LONG':
                should_exit, exit_px, reason = _check_exit_long(pos, row, days_held)
            else:
                should_exit, exit_px, reason = _check_exit_short(pos, row, days_held)
            if should_exit:
                to_close.append((sym, exit_px, reason))

        for sym, exit_px, reason in to_close:
            pos = open_positions.pop(sym)
            entry_value = pos.entry_price * pos.qty
            exit_value  = exit_px * pos.qty
            if pos.direction == 'LONG':
                gross_pnl = (exit_px - pos.entry_price) * pos.qty
            else:
                gross_pnl = (pos.entry_price - exit_px) * pos.qty
            cost = cost_pct * (entry_value + exit_value) / 2.0
            net_pnl = gross_pnl - cost
            equity += net_pnl
            entry_d = pd.to_datetime(pos.entry_date).date()
            days = (today_date - entry_d).days
            trades.append(Trade(
                variant=cfg['name'], symbol=sym, direction=pos.direction,
                entry_date=pos.entry_date, exit_date=today_str,
                entry_price=pos.entry_price, exit_price=exit_px,
                qty=pos.qty, days_held=days, exit_reason=reason,
                gross_pnl=gross_pnl, net_pnl=net_pnl,
            ))

        # 3) Scan for new entries
        if len(open_positions) < MAX_CONCURRENT:
            candidates = []
            for sym, df in bars.items():
                if sym in open_positions:
                    continue
                if today_date not in df.index:
                    continue
                row = df.loc[today_date]
                try:
                    if signal_fn(row, cfg):
                        # Default rank: volume spike (or 1.0 if missing)
                        if candidate_rank_fn is not None:
                            rank_val = candidate_rank_fn(row, cfg)
                        else:
                            vavg = row.get('vol_avg', np.nan)
                            rank_val = float(row['volume'] / vavg) if (pd.notna(vavg) and vavg > 0) else 1.0
                        candidates.append((rank_val, sym))
                except Exception:
                    continue
            candidates.sort(reverse=True)
            slots_left = MAX_CONCURRENT - len(open_positions) - len(pending_entries)
            for _, sym in candidates[:slots_left]:
                pending_entries.append((today_str, sym))

        # 4) Mark-to-market
        unrealized = 0.0
        for sym, pos in open_positions.items():
            df = bars[sym]
            if today_date not in df.index:
                continue
            close_today = df.loc[today_date, 'close']
            if pos.direction == 'LONG':
                unrealized += (close_today - pos.entry_price) * pos.qty
            else:
                unrealized += (pos.entry_price - close_today) * pos.qty
        daily_equity[today_str] = equity + unrealized

    # Close still-open positions at last bar
    for sym, pos in list(open_positions.items()):
        df = bars[sym]
        last_date = df.index[-1]
        entry_d = pd.to_datetime(pos.entry_date).date()
        if last_date < entry_d:
            continue
        last_close = df.loc[last_date, 'close']
        if pos.direction == 'LONG':
            gross_pnl = (last_close - pos.entry_price) * pos.qty
        else:
            gross_pnl = (pos.entry_price - last_close) * pos.qty
        cost = cost_pct * (pos.entry_price * pos.qty + last_close * pos.qty) / 2.0
        net_pnl = gross_pnl - cost
        equity += net_pnl
        days = (last_date - entry_d).days
        trades.append(Trade(
            variant=cfg['name'], symbol=sym, direction=pos.direction,
            entry_date=pos.entry_date, exit_date=str(last_date),
            entry_price=pos.entry_price, exit_price=last_close,
            qty=pos.qty, days_held=days, exit_reason='END_OF_BACKTEST',
            gross_pnl=gross_pnl, net_pnl=net_pnl,
        ))

    eq_series = pd.Series(daily_equity).sort_index()
    return trades, eq_series


def compute_metrics(trades: list[Trade], eq: pd.Series) -> dict:
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
            'avg_days_held','profit_factor','net_pnl','cagr_pct','sharpe',
            'max_dd_pct','calmar','final_equity']
    if not trades or eq.empty:
        return {k: 0 for k in keys}
    net_list = [t.net_pnl for t in trades]
    wins = [p for p in net_list if p > 0]; losses = [p for p in net_list if p < 0]
    total_net = sum(net_list)

    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() > 0 else 0

    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd_pct = abs(dd.min()) * 100 if not dd.empty else 0

    n_days = (pd.to_datetime(eq.index[-1]) - pd.to_datetime(eq.index[0])).days
    yrs = n_days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1) * 100 if yrs > 0 and eq.iloc[0] > 0 else 0

    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100*len(wins)/len(trades), 2),
        'avg_win':  round(sum(wins)/len(wins), 0) if wins else 0,
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


def write_trades(trades: list[Trade], path: Path) -> None:
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','direction','entry_date','exit_date',
                    'entry_price','exit_price','qty','days_held','exit_reason',
                    'gross_pnl','net_pnl'])
        for t in trades:
            w.writerow([t.variant, t.symbol, t.direction, t.entry_date, t.exit_date,
                        f'{t.entry_price:.2f}', f'{t.exit_price:.2f}',
                        t.qty, t.days_held, t.exit_reason,
                        f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])


def write_summary(rows: list[dict], path: Path) -> None:
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
            'avg_days_held','profit_factor','net_pnl','cagr_pct','sharpe',
            'max_dd_pct','calmar','final_equity']
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','direction'] + keys)
        for r in rows:
            w.writerow([r['variant'], r.get('direction','')] + [r['metrics'].get(k, '') for k in keys])


def heartbeat(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} | {msg}\n')


# Re-export simple constants for callers
__all__ = [
    'START_DATE','END_DATE','CAPITAL','MAX_CONCURRENT','RISK_PER_TRADE_PCT',
    'DEFAULT_COST_PCT','INITIAL_HARD_STOP_PCT','MAX_HOLD_DAYS',
    'load_universe','build_universe_csv','load_all_bars',
    'run_engine','compute_metrics','write_trades','write_summary',
    'heartbeat','Position','Trade',
]
