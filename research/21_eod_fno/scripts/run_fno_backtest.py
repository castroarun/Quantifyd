"""F&O-only EOD breakout — portfolio backtest (mirror of research/19 vol_3x winner).

Universe: F&O stocks (FNO_LOT_SIZES.keys() in services/data_manager.py),
filtered to those with >= 1500 daily bars since 2018-01-01.

No turnover/circuit filter — F&O eligibility itself is a liquidity floor.

Rules (EXACT mirror of research/19's validated `vol_3x`):
  Entry:
    - today's close > 252-day high (excluding today)
    - today's volume >= 3.0x trailing 50-day avg volume
    - today's close > 200-day SMA
  Stop:
    - max(entry - 2*ATR(14), entry * 0.92)   (closer-to-entry)
  Target:
    - entry * 1.25 (fixed 25%)
  Time:
    - 60-day max-hold safety stop, no trailing

Sizing:
  - Risk per trade: 1% of equity
  - Max concurrent: 10 positions
  - Notional cap per position: equity / 10
  - Cost: 0.20% round-trip (tighter spreads on F&O names vs small caps)
  - Capital: Rs 10,00,000

Period: 2018-01-01 to 2025-12-31 (~8 years).

Six variants (same family as research/19 — keeps comparability):
  baseline_252_25pct_8pct   - exact spec (vol >= 2.5x)
  vol_2x                    - vol >= 2.0x
  vol_3x                    - vol >= 3.0x   (the validated winner shape)
  target_30pct              - 30% target (vol 2.5x)
  target_20pct              - 20% target (vol 2.5x)
  cost_30bps                - 0.30% cost stress
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

logging.disable(logging.WARNING); warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
RES = Path(__file__).resolve().parents[1] / 'results'
LOGS = Path(__file__).resolve().parents[1] / 'logs'
RES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
HEARTBEAT = LOGS / 'backtest_heartbeat.txt'
UNIVERSE_FILE = RES / 'fno_universe.csv'

# Make services/ importable for FNO_LOT_SIZES
sys.path.insert(0, str(ROOT))

START_DATE = '2018-01-01'
END_DATE   = '2025-12-31'
MIN_BARS   = 1500

# Capital and sizing
CAPITAL                = 10_00_000
RISK_PER_TRADE_PCT     = 0.01
MAX_CONCURRENT         = 10
DEFAULT_COST_PCT       = 0.0020          # 0.20% round-trip (F&O = tight spreads)
INITIAL_HARD_STOP_PCT  = 0.08            # 8% day-1 backstop
MAX_HOLD_DAYS          = 60              # safety time stop

# Indicator params
ATR_PERIOD             = 14
SMA_REGIME_PERIOD      = 200
VOL_AVG_PERIOD         = 50
BREAKOUT_N             = 252


def make_variant(name, breakout_n=BREAKOUT_N, vol_mult=2.5, regime=True,
                 target_pct=0.25, hard_stop_pct=INITIAL_HARD_STOP_PCT,
                 cost_pct=DEFAULT_COST_PCT):
    return {
        'name': name,
        'breakout_n': breakout_n,
        'vol_mult': vol_mult,
        'regime_filter': regime,
        'target_pct': target_pct,
        'hard_stop_pct': hard_stop_pct,
        'cost_pct': cost_pct,
    }


VARIANTS = [
    make_variant('baseline_252_25pct_8pct'),                     # vol>=2.5x
    make_variant('vol_2x',         vol_mult=2.0),
    make_variant('vol_3x',         vol_mult=3.0),                # validated winner shape
    make_variant('target_30pct',   target_pct=0.30),
    make_variant('target_20pct',   target_pct=0.20),
    make_variant('cost_30bps',     cost_pct=0.0030),
]


@dataclass
class Position:
    symbol: str
    entry_date: str
    entry_price: float
    qty: int
    initial_stop: float
    atr_at_entry: float
    target_price: float
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


def heartbeat(msg: str) -> None:
    with HEARTBEAT.open('w') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} | {msg}\n')


def build_fno_universe() -> list[str]:
    """Return F&O symbols with enough daily bars. Caches to UNIVERSE_FILE."""
    from services.data_manager import FNO_LOT_SIZES
    candidates = sorted(FNO_LOT_SIZES.keys())

    conn = sqlite3.connect(DB)
    rows = []
    keep = []
    for sym in candidates:
        r = conn.execute("""SELECT COUNT(*), MIN(date), MAX(date)
                              FROM market_data_unified
                             WHERE symbol=? AND timeframe='day' AND date>=?""",
                         (sym, START_DATE)).fetchone()
        n, dmin, dmax = (r[0], r[1], r[2]) if r else (0, None, None)
        rows.append((sym, n, dmin, dmax))
        if n >= MIN_BARS:
            keep.append(sym)
    conn.close()

    with UNIVERSE_FILE.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['symbol', 'bars', 'date_min', 'date_max', 'kept'])
        for sym, n, dmin, dmax in rows:
            w.writerow([sym, n, dmin or '', dmax or '', 'Y' if n >= MIN_BARS else 'N'])
    return keep


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        return build_fno_universe()
    keep = []
    with UNIVERSE_FILE.open() as f:
        for r in csv.DictReader(f):
            if r.get('kept', 'N') == 'Y':
                keep.append(r['symbol'])
    if not keep:
        # File exists but malformed/empty — rebuild
        return build_fno_universe()
    return keep


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
        if df.empty:
            continue
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.set_index('date', inplace=True)

        # Rolling N-day high (excluding today)
        df['high_252d'] = df['high'].shift(1).rolling(BREAKOUT_N).max()

        # ATR (Wilder)
        prev_close = df['close'].shift(1)
        tr = pd.concat([df['high'] - df['low'],
                        (df['high'] - prev_close).abs(),
                        (df['low'] - prev_close).abs()], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()

        # 50-day vol avg (excluding today)
        df['vol_avg'] = df['volume'].shift(1).rolling(VOL_AVG_PERIOD).mean()

        # 200-SMA regime
        df['sma_200'] = df['close'].rolling(SMA_REGIME_PERIOD).mean()

        out[sym] = df
    conn.close()
    return out


def get_signal(row: pd.Series, cfg: dict) -> bool:
    high_n = row.get('high_252d')
    if pd.isna(high_n):
        return False
    if not (row['close'] > high_n):
        return False
    if cfg['vol_mult'] > 0:
        if pd.isna(row['vol_avg']) or row['vol_avg'] <= 0:
            return False
        if not (row['volume'] >= cfg['vol_mult'] * row['vol_avg']):
            return False
    if cfg['regime_filter']:
        if pd.isna(row['sma_200']):
            return False
        if not (row['close'] > row['sma_200']):
            return False
    return True


def check_exit(pos: Position, today: pd.Series, today_date,
               days_held: int, cfg: dict) -> tuple[bool, float, str]:
    if today['low'] <= pos.initial_stop:
        return True, pos.initial_stop, 'INITIAL_STOP'
    if today['high'] >= pos.target_price:
        return True, pos.target_price, 'TARGET'
    if days_held >= MAX_HOLD_DAYS:
        return True, today['close'], 'MAX_HOLD'
    return False, 0.0, ''


def run_variant(bars: dict[str, pd.DataFrame], cfg: dict,
                start_date: str = START_DATE,
                end_date: str = END_DATE) -> tuple[list[Trade], pd.Series]:
    cost_pct = cfg['cost_pct']
    target_pct = cfg['target_pct']
    hard_stop_pct = cfg['hard_stop_pct']

    all_dates = sorted({d for df in bars.values() for d in df.index
                        if str(d) >= start_date and str(d) <= end_date})
    open_positions: dict[str, Position] = {}
    trades: list[Trade] = []
    equity = CAPITAL
    daily_equity = {}
    pending_entries: list[tuple[str, str]] = []

    for i, today_date in enumerate(all_dates):
        today_str = str(today_date)

        # 1) Pending entries — fill at TODAY's open
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
            stop_atr = entry_px - 2 * atr_e
            stop_pct_floor = entry_px * (1 - hard_stop_pct)
            initial_stop = max(stop_atr, stop_pct_floor)
            risk_per_share = entry_px - initial_stop
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

            target_price = entry_px * (1 + target_pct)
            pos = Position(symbol=sym, entry_date=today_str, entry_price=entry_px,
                           qty=qty, initial_stop=initial_stop,
                           atr_at_entry=atr_e, target_price=target_price)
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
            if pos.entry_date == today_str:
                continue
            entry_d = pd.to_datetime(pos.entry_date).date()
            days_held = (today_date - entry_d).days
            should_exit, exit_px, reason = check_exit(pos, row, today_date, days_held, cfg)
            if should_exit:
                to_close.append((sym, exit_px, reason))

        for sym, exit_px, reason in to_close:
            pos = open_positions.pop(sym)
            entry_value = pos.entry_price * pos.qty
            exit_value  = exit_px * pos.qty
            gross_pnl = (exit_px - pos.entry_price) * pos.qty
            cost = cost_pct * (entry_value + exit_value) / 2.0
            net_pnl = gross_pnl - cost
            equity += net_pnl
            entry_d = pd.to_datetime(pos.entry_date).date()
            days = (today_date - entry_d).days
            trades.append(Trade(
                variant=cfg['name'], symbol=sym,
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
                if get_signal(row, cfg):
                    vavg = row['vol_avg']
                    vspike = float(row['volume'] / vavg) if (vavg and vavg > 0) else 1.0
                    candidates.append((vspike, sym))
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
            unrealized += (close_today - pos.entry_price) * pos.qty
        daily_equity[today_str] = equity + unrealized

    # Close still-open at last bar
    for sym, pos in list(open_positions.items()):
        df = bars[sym]
        last_date = df.index[-1]
        entry_d = pd.to_datetime(pos.entry_date).date()
        if last_date < entry_d:
            continue
        last_close = df.loc[last_date, 'close']
        gross_pnl = (last_close - pos.entry_price) * pos.qty
        cost = cost_pct * (pos.entry_price * pos.qty + last_close * pos.qty) / 2.0
        net_pnl = gross_pnl - cost
        equity += net_pnl
        days = (last_date - entry_d).days
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


def write_trades(trades: list[Trade], path: Path) -> None:
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','entry_date','exit_date','entry_price',
                    'exit_price','qty','days_held','exit_reason',
                    'gross_pnl','net_pnl'])
        for t in trades:
            w.writerow([t.variant, t.symbol, t.entry_date, t.exit_date,
                        f'{t.entry_price:.2f}', f'{t.exit_price:.2f}',
                        t.qty, t.days_held, t.exit_reason,
                        f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])


def main():
    t_start = time.time()
    heartbeat('start: building F&O universe')
    print('Building / loading F&O universe...', flush=True)
    uni = load_universe()
    print(f'F&O universe (>= {MIN_BARS} bars since {START_DATE}): {len(uni)} stocks', flush=True)

    print('Loading bars + computing indicators...', flush=True)
    heartbeat('loading bars')
    t0 = time.time()
    bars = load_all_bars(uni)
    print(f'  loaded {len(bars)} stocks in {time.time()-t0:.1f}s', flush=True)

    summary_keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                    'avg_days_held','profit_factor','net_pnl','cagr_pct','sharpe',
                    'max_dd_pct','calmar','final_equity']

    print()
    print(f'{"Variant":>26} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgD":>6} '
          f'{"CAGR%":>7} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7} {"FinalEq":>11}', flush=True)
    print('-' * 105, flush=True)

    summary_path = RES / 'fno_summary.csv'
    with summary_path.open('w', newline='') as f:
        csv.writer(f).writerow(['variant'] + summary_keys)

    # Aggregate trade log + equity (all variants combined)
    all_trades_path = RES / 'fno_trades.csv'
    with all_trades_path.open('w', newline='') as f:
        csv.writer(f).writerow(['variant','symbol','entry_date','exit_date','entry_price',
                                'exit_price','qty','days_held','exit_reason',
                                'gross_pnl','net_pnl'])

    for cfg in VARIANTS:
        heartbeat(f'running variant: {cfg["name"]}')
        t0 = time.time()
        trades, eq = run_variant(bars, cfg)
        m = compute_metrics(trades, eq)
        print(f'{cfg["name"]:>26} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} '
              f'{m["profit_factor"]:>6.2f} {m["avg_days_held"]:>6.1f} '
              f'{m["cagr_pct"]:>+7.2f} {m["sharpe"]:>7.2f} '
              f'{m["max_dd_pct"]:>7.2f} {m["calmar"]:>7.2f} '
              f'Rs{m["final_equity"]:>+9,.0f}  ({time.time()-t0:.1f}s)', flush=True)

        with summary_path.open('a', newline='') as f:
            csv.writer(f).writerow([cfg['name']] + [m.get(k, '') for k in summary_keys])

        # Per-variant + aggregate trade logs
        write_trades(trades, RES / f'fno_trades_{cfg["name"]}.csv')
        with all_trades_path.open('a', newline='') as f:
            w = csv.writer(f)
            for t in trades:
                w.writerow([t.variant, t.symbol, t.entry_date, t.exit_date,
                            f'{t.entry_price:.2f}', f'{t.exit_price:.2f}',
                            t.qty, t.days_held, t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

        # Per-variant equity curve
        eq.to_csv(RES / f'fno_equity_{cfg["name"]}.csv', header=['equity'])

    # Equity for the validated winner shape — single canonical file
    # (we write per-variant equity files above; also drop a copy of vol_3x as
    #  fno_equity.csv per the task spec)
    vol3 = RES / 'fno_equity_vol_3x.csv'
    if vol3.exists():
        try:
            (RES / 'fno_equity.csv').write_bytes(vol3.read_bytes())
        except Exception:
            pass

    print(f'\nTotal runtime: {time.time()-t_start:.1f}s', flush=True)
    print(f'Summary: {summary_path}', flush=True)
    heartbeat('DONE: full sweep complete')


if __name__ == '__main__':
    sys.exit(main())
