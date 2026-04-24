"""VWAP Mean-Reversion v2 — defensive universe + Stoch/RSI confirmation.

v1 result on the ORB high-beta universe was a uniform disaster (all 15 stocks
lost money, PF 0.06-0.28). Counter-trend fade on trenders doesn't work —
exactly what makes ORB win makes VWAP-MR lose on that universe.

v2 changes:
  1. Universe = 14 defensive/low-beta F&O names (FMCG + utilities + pharma),
     chosen for intraday range-bound character and zero overlap with ORB's 15
     (so no position-netting conflict in the same broker account).
  2. Adds optional confirmation indicators — run as 4 variants in one pass:
       baseline     — VWAP + ATR + 2-bar-confirm only (v1 spec on new universe)
       stoch        — + Stochastic(14,3,3) crossover in OB/OS zones
       rsi          — + RSI(14) reversal off 70/30 extremes
       stoch+rsi    — both filters ANDed
"""
from __future__ import annotations

import csv
import logging
import sqlite3
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path

logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(exist_ok=True)

UNIVERSE = [
    'HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'ITC', 'DABUR', 'COLPAL', 'MARICO',
    'POWERGRID', 'NTPC', 'ONGC', 'COALINDIA', 'SUNPHARMA', 'DRREDDY', 'CIPLA',
]

START_DATE = '2024-03-18'
END_DATE   = '2026-03-12'

# Strategy parameters (shared across variants)
ATR_PERIOD          = 14
ENTRY_WINDOW_START  = dtime(11, 30)
ENTRY_WINDOW_END    = dtime(14, 0)
EOD_EXIT            = dtime(15, 15)
DEVIATION_ENTRY     = 1.8
DEVIATION_CONFIRM   = 1.5
TARGET_ATR_MULT     = 1.0
STOP_ATR_MULT       = 1.0
TIME_STOP_BARS      = 12

# Stochastic (14, 3, 3) — OB/OS thresholds for entry gate
STOCH_K_PERIOD      = 14
STOCH_SMOOTH        = 3
STOCH_D_PERIOD      = 3
STOCH_OB            = 80.0
STOCH_OS            = 20.0

# RSI (14) — reversal thresholds
RSI_PERIOD          = 14
RSI_OB              = 70.0
RSI_OS              = 30.0

# Sizing + costs
RISK_PER_TRADE      = 2_500
MAX_NOTIONAL        = 300_000
COST_PCT            = 0.0015  # 0.15% round-trip blended

VARIANTS = {
    'baseline':      {'use_stoch': False, 'use_rsi': False},
    'stoch':         {'use_stoch': True,  'use_rsi': False},
    'rsi':           {'use_stoch': False, 'use_rsi': True},
    'stoch_and_rsi': {'use_stoch': True,  'use_rsi': True},
}


@dataclass
class Trade:
    variant: str
    symbol: str
    date: str
    direction: str
    entry_time: str
    entry: float
    stop: float
    target: float
    qty: int
    atr_at_entry: float
    deviation_at_entry: float
    exit_time: str = ''
    exit_price: float = 0.0
    exit_reason: str = ''
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    def close(self, exit_time: str, exit_price: float, reason: str):
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason
        sign = 1 if self.direction == 'LONG' else -1
        self.gross_pnl = sign * (exit_price - self.entry) * self.qty
        cost = COST_PCT * (self.entry + exit_price) * self.qty / 2.0
        self.net_pnl = self.gross_pnl - cost


def load_bars(symbol: str) -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT date, open, high, low, close, volume
             FROM market_data_unified
            WHERE symbol = ? AND timeframe = '5minute'
              AND date >= ? AND date <= ?
         ORDER BY date""",
        (symbol, START_DATE, END_DATE + ' 23:59:59'),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        ts = datetime.fromisoformat(r['date']) if isinstance(r['date'], str) else r['date']
        out.append({'ts': ts, 'o': r['open'], 'h': r['high'], 'l': r['low'],
                    'c': r['close'], 'v': r['volume']})
    return out


def run_stock_variant(symbol: str, variant: str, cfg: dict) -> tuple[list[Trade], dict]:
    """Backtest one stock for one variant. Returns (trades, daily_pnl)."""
    bars = load_bars(symbol)
    if not bars:
        return [], {}

    # Group by date
    by_date: dict[str, list[dict]] = defaultdict(list)
    for b in bars:
        by_date[b['ts'].date().isoformat()].append(b)

    trades: list[Trade] = []
    daily_pnl: dict[str, float] = defaultdict(float)

    for day, day_bars in sorted(by_date.items()):
        # Reset intraday state each day
        cum_pv = 0.0
        cum_v = 0.0
        tr_hist: deque[float] = deque(maxlen=ATR_PERIOD)
        hl_window: deque[tuple[float, float]] = deque(maxlen=STOCH_K_PERIOD)  # (high,low)
        k_smooth_window: deque[float] = deque(maxlen=STOCH_SMOOTH)
        d_window: deque[float] = deque(maxlen=STOCH_D_PERIOD)
        avg_gain = None
        avg_loss = None
        prev_close: float | None = None
        prev_deviation: float | None = None
        prev_k: float | None = None
        prev_d: float | None = None
        prev_rsi: float | None = None
        active: Trade | None = None
        bars_held = 0
        traded_today = False

        for b in day_bars:
            ts = b['ts']
            t = ts.time()
            typ = (b['h'] + b['l'] + b['c']) / 3.0
            cum_pv += typ * b['v']
            cum_v += b['v']
            vwap = cum_pv / cum_v if cum_v else b['c']

            # ATR
            if prev_close is None:
                tr = b['h'] - b['l']
            else:
                tr = max(b['h'] - b['l'], abs(b['h'] - prev_close), abs(b['l'] - prev_close))
            tr_hist.append(tr)
            atr = sum(tr_hist) / len(tr_hist) if tr_hist else 0.0

            # Stochastic (14,3,3)
            hl_window.append((b['h'], b['l']))
            stoch_k = None
            stoch_d = None
            if len(hl_window) == STOCH_K_PERIOD:
                hh = max(h for h, _ in hl_window)
                ll = min(l for _, l in hl_window)
                k_raw = 100 * (b['c'] - ll) / (hh - ll) if hh > ll else 50.0
                k_smooth_window.append(k_raw)
                if len(k_smooth_window) == STOCH_SMOOTH:
                    stoch_k = sum(k_smooth_window) / STOCH_SMOOTH
                    d_window.append(stoch_k)
                    if len(d_window) == STOCH_D_PERIOD:
                        stoch_d = sum(d_window) / STOCH_D_PERIOD

            # RSI (14)
            rsi = None
            if prev_close is not None:
                change = b['c'] - prev_close
                gain = max(change, 0.0)
                loss = max(-change, 0.0)
                if avg_gain is None:
                    avg_gain = gain
                    avg_loss = loss
                else:
                    # Wilder smoothing
                    avg_gain = (avg_gain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD
                    avg_loss = (avg_loss * (RSI_PERIOD - 1) + loss) / RSI_PERIOD
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - 100 / (1 + rs)
                elif avg_gain > 0:
                    rsi = 100.0

            # --- Exit logic
            if active is not None:
                bars_held += 1
                hit_stop = (active.direction == 'LONG'  and b['l'] <= active.stop) \
                        or (active.direction == 'SHORT' and b['h'] >= active.stop)
                hit_tgt  = (active.direction == 'LONG'  and b['h'] >= active.target) \
                        or (active.direction == 'SHORT' and b['l'] <= active.target)
                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP')
                    trades.append(active); daily_pnl[day] += active.net_pnl; active = None
                elif hit_tgt:
                    active.close(ts.isoformat(), active.target, 'TARGET')
                    trades.append(active); daily_pnl[day] += active.net_pnl; active = None
                elif bars_held >= TIME_STOP_BARS:
                    active.close(ts.isoformat(), b['c'], 'TIME_STOP')
                    trades.append(active); daily_pnl[day] += active.net_pnl; active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), b['c'], 'EOD')
                    trades.append(active); daily_pnl[day] += active.net_pnl; active = None

            # --- Entry logic
            if (active is None and not traded_today
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END
                    and atr > 0 and len(tr_hist) >= ATR_PERIOD):
                deviation = (b['c'] - vwap) / atr

                # Optional indicator gates — TRUE means "allowed / confirmed"
                stoch_short_ok = True
                stoch_long_ok  = True
                if cfg['use_stoch']:
                    if stoch_k is not None and stoch_d is not None \
                            and prev_k is not None and prev_d is not None:
                        # Bearish cross in OB zone confirms SHORT fade
                        stoch_short_ok = (prev_k >= prev_d and stoch_k < stoch_d
                                           and stoch_k > STOCH_OB)
                        # Bullish cross in OS zone confirms LONG fade
                        stoch_long_ok  = (prev_k <= prev_d and stoch_k > stoch_d
                                           and stoch_k < STOCH_OS)
                    else:
                        stoch_short_ok = stoch_long_ok = False

                rsi_short_ok = True
                rsi_long_ok  = True
                if cfg['use_rsi']:
                    if rsi is not None and prev_rsi is not None:
                        # Turned down from OB confirms SHORT
                        rsi_short_ok = (prev_rsi > RSI_OB and rsi < prev_rsi)
                        # Turned up from OS confirms LONG
                        rsi_long_ok  = (prev_rsi < RSI_OS and rsi > prev_rsi)
                    else:
                        rsi_short_ok = rsi_long_ok = False

                if prev_deviation is not None:
                    # SHORT fade
                    if (deviation > DEVIATION_ENTRY
                            and prev_deviation > DEVIATION_CONFIRM
                            and stoch_short_ok and rsi_short_ok):
                        stop = b['c'] + STOP_ATR_MULT * atr
                        target = b['c'] - TARGET_ATR_MULT * atr
                        rps = stop - b['c']
                        if rps > 0:
                            qty = int(RISK_PER_TRADE // rps)
                            if qty * b['c'] > MAX_NOTIONAL:
                                qty = int(MAX_NOTIONAL // b['c'])
                            if qty > 0:
                                active = Trade(variant, symbol, day, 'SHORT', ts.isoformat(),
                                               b['c'], stop, target, qty, atr, deviation)
                                bars_held = 0; traded_today = True
                    # LONG fade
                    elif (deviation < -DEVIATION_ENTRY
                            and prev_deviation < -DEVIATION_CONFIRM
                            and stoch_long_ok and rsi_long_ok):
                        stop = b['c'] - STOP_ATR_MULT * atr
                        target = b['c'] + TARGET_ATR_MULT * atr
                        rps = b['c'] - stop
                        if rps > 0:
                            qty = int(RISK_PER_TRADE // rps)
                            if qty * b['c'] > MAX_NOTIONAL:
                                qty = int(MAX_NOTIONAL // b['c'])
                            if qty > 0:
                                active = Trade(variant, symbol, day, 'LONG', ts.isoformat(),
                                               b['c'], stop, target, qty, atr, deviation)
                                bars_held = 0; traded_today = True

                prev_deviation = deviation

            prev_close = b['c']
            if stoch_k is not None: prev_k = stoch_k
            if stoch_d is not None: prev_d = stoch_d
            if rsi is not None: prev_rsi = rsi

        if active is not None and day_bars:
            last = day_bars[-1]
            active.close(last['ts'].isoformat(), last['c'], 'EOD_FORCED')
            trades.append(active); daily_pnl[day] += active.net_pnl

    return trades, dict(daily_pnl)


def compute_metrics(trades: list[Trade], daily_pnl: dict) -> dict:
    if not trades:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate_pct': 0,
                'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                'net_pnl': 0, 'gross_pnl': 0, 'costs': 0,
                'days_traded': 0, 'cagr_pct': 0, 'sharpe': 0,
                'max_dd': 0, 'max_dd_pct': 0, 'calmar': 0}
    net = [t.net_pnl for t in trades]
    wins = [p for p in net if p > 0]
    losses = [p for p in net if p < 0]
    total_net = sum(net)

    sorted_days = sorted(daily_pnl.keys())
    daily_series = [daily_pnl[d] for d in sorted_days]
    running = 0.0; peak = 0.0; max_dd = 0.0
    for pnl in daily_series:
        running += pnl
        if running > peak: peak = running
        dd = peak - running
        if dd > max_dd: max_dd = dd

    n_days = len(daily_series)
    if n_days > 1:
        mean = sum(daily_series) / n_days
        var = sum((x - mean) ** 2 for x in daily_series) / (n_days - 1)
        std = var ** 0.5
        sharpe = (mean / std) * (252 ** 0.5) if std > 0 else 0.0
    else:
        sharpe = 0.0

    cap = 300_000
    years = n_days / 252 if n_days else 1
    ending = cap + total_net
    cagr = ((ending / cap) ** (1 / years) - 1) * 100 if years > 0 and ending > 0 else 0.0

    return {
        'trades':        len(trades),
        'wins':          len(wins),
        'losses':        len(losses),
        'win_rate_pct':  round(100 * len(wins) / len(trades), 2),
        'avg_win':       round(sum(wins) / len(wins), 0) if wins else 0,
        'avg_loss':      round(sum(losses) / len(losses), 0) if losses else 0,
        'profit_factor': round(sum(wins) / abs(sum(losses)), 2) if losses else 0,
        'net_pnl':       round(total_net, 0),
        'gross_pnl':     round(sum(t.gross_pnl for t in trades), 0),
        'costs':         round(sum(t.gross_pnl - t.net_pnl for t in trades), 0),
        'days_traded':   n_days,
        'cagr_pct':      round(cagr, 2),
        'sharpe':        round(sharpe, 2),
        'max_dd':        round(max_dd, 0),
        'max_dd_pct':    round(100 * max_dd / cap, 2),
        'calmar':        round((cagr / (100 * max_dd / cap)), 2) if max_dd > 0 else 0,
    }


def main():
    all_trades_by_variant: dict[str, list[Trade]] = {v: [] for v in VARIANTS}
    daily_by_variant: dict[str, dict] = {v: defaultdict(float) for v in VARIANTS}

    for i, sym in enumerate(UNIVERSE, 1):
        print(f'[{i:2d}/{len(UNIVERSE)}] {sym:12s}', end='')
        for vname, vcfg in VARIANTS.items():
            trades, daily = run_stock_variant(sym, vname, vcfg)
            all_trades_by_variant[vname].extend(trades)
            for d, p in daily.items():
                daily_by_variant[vname][d] += p
            net = sum(t.net_pnl for t in trades)
            print(f'  {vname}={len(trades):>3}tr/Rs{net:>+8,.0f}', end='')
        print()

    # Trade log
    trade_path = OUT / 'trades_v2.csv'
    with trade_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','date','direction','entry_time','entry','stop','target',
                    'qty','atr_at_entry','deviation_at_entry','exit_time','exit_price',
                    'exit_reason','gross_pnl','net_pnl'])
        for v, trades in all_trades_by_variant.items():
            for t in trades:
                w.writerow([t.variant, t.symbol, t.date, t.direction, t.entry_time,
                            f'{t.entry:.2f}', f'{t.stop:.2f}', f'{t.target:.2f}', t.qty,
                            f'{t.atr_at_entry:.2f}', f'{t.deviation_at_entry:.2f}',
                            t.exit_time, f'{t.exit_price:.2f}', t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

    # Daily P&L per variant (for ORB correlation)
    daily_path = OUT / 'daily_pnl_v2.csv'
    all_days = sorted({d for v in daily_by_variant.values() for d in v.keys()})
    with daily_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date'] + list(VARIANTS.keys()))
        for d in all_days:
            w.writerow([d] + [f'{daily_by_variant[v].get(d, 0):.2f}' for v in VARIANTS])

    # Summary per variant
    summary_path = OUT / 'summary_v2.csv'
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
            'profit_factor','net_pnl','gross_pnl','costs','days_traded',
            'cagr_pct','sharpe','max_dd','max_dd_pct','calmar']
    with summary_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant'] + keys)
        for vname in VARIANTS:
            m = compute_metrics(all_trades_by_variant[vname],
                                 dict(daily_by_variant[vname]))
            w.writerow([vname] + [m.get(k, '') for k in keys])

    # Print headline
    print()
    print('=' * 100)
    print(f'VWAP-MR v2 PORTFOLIO ({len(UNIVERSE)} defensive stocks, {START_DATE} to {END_DATE})')
    print('=' * 100)
    header = f'{"Variant":15s} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Net P&L":>13} {"CAGR%":>7} {"Sharpe":>7} {"MaxDD":>10} {"Calmar":>7}'
    print(header)
    print('-' * 100)
    for vname in VARIANTS:
        m = compute_metrics(all_trades_by_variant[vname],
                             dict(daily_by_variant[vname]))
        print(f'{vname:15s} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'Rs {m["net_pnl"]:>+10,.0f} {m["cagr_pct"]:>+6.2f} {m["sharpe"]:>7.2f} '
              f'{m["max_dd"]:>+9,.0f} {m["calmar"]:>7.2f}')
    print()
    print(f'Artifacts: {OUT}')


if __name__ == '__main__':
    sys.exit(main())
