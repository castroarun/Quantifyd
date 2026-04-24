"""VWAP Mean-Reversion intraday backtest — counter-ORB system.

Thesis: when a stock is >1.8 ATR stretched from its day's cumulative VWAP in
the midday window (11:30-14:00), a fade toward VWAP has positive expectancy.
This is explicitly opposite to ORB's breakout logic — loses on trending days
(when ORB wins) and wins on choppy days, giving a negatively-correlated P&L
stream as a second intraday system.

Universe: the 15 ORB stocks, so the correlation test against the ORB backtest
is direct (same stocks, same days).

Outputs:
    results/trades.csv       — per-trade log
    results/daily_pnl.csv    — per-day P&L per stock (for ORB correlation)
    results/summary.csv      — headline metrics
"""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path

logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(exist_ok=True)

UNIVERSE = [
    'ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BPCL', 'M&M', 'BAJFINANCE',
    'TRENT', 'HAL', 'IRCTC', 'GRASIM', 'GODREJPROP', 'RELIANCE',
    'AXISBANK', 'APOLLOHOSP',
]

START_DATE = '2024-03-18'
END_DATE   = '2026-03-12'

# Strategy parameters
ATR_PERIOD          = 14
ENTRY_WINDOW_START  = dtime(11, 30)
ENTRY_WINDOW_END    = dtime(14, 0)
EOD_EXIT            = dtime(15, 15)
DEVIATION_ENTRY     = 1.8     # enter when |close-vwap|/atr > this
DEVIATION_CONFIRM   = 1.5     # prev bar also above this threshold
TARGET_ATR_MULT     = 1.0     # take profit at 1R back toward VWAP
STOP_ATR_MULT       = 1.0     # stop 1R adverse (so 1:1)
TIME_STOP_BARS      = 12      # 12 × 5min = 60 min

# Sizing + costs
RISK_PER_TRADE      = 2_500   # Rs — matches ORB anchor (0.8% of 3L)
MAX_NOTIONAL        = 300_000
COST_PCT            = 0.0015  # 0.15% round-trip blended (brokerage + STT + slippage)


@dataclass
class Trade:
    symbol: str
    date: str
    direction: str          # LONG / SHORT
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
        cost = COST_PCT * (self.entry + exit_price) * self.qty / 2.0  # one-way pct on each leg
        self.net_pnl = self.gross_pnl - cost


def load_bars(symbol: str) -> list[dict]:
    """Return list of dicts with keys: date (datetime), o,h,l,c,v."""
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
        out.append({
            'ts': ts, 'o': r['open'], 'h': r['high'], 'l': r['low'],
            'c': r['close'], 'v': r['volume'],
        })
    return out


def run_stock(symbol: str) -> tuple[list[Trade], dict[str, float]]:
    """Backtest one stock. Returns (trades, daily_pnl_by_date)."""
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
        cum_pv = 0.0           # cumulative price*volume
        cum_v = 0.0            # cumulative volume
        tr_hist: list[float] = []   # true ranges for ATR
        prev_close: float | None = None
        prev_deviation: float | None = None
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

            # True range + rolling ATR
            if prev_close is None:
                tr = b['h'] - b['l']
            else:
                tr = max(b['h'] - b['l'], abs(b['h'] - prev_close), abs(b['l'] - prev_close))
            tr_hist.append(tr)
            if len(tr_hist) > ATR_PERIOD:
                tr_hist.pop(0)
            atr = sum(tr_hist) / len(tr_hist) if tr_hist else 0.0

            # --- Exit logic first (for active trade)
            if active is not None:
                bars_held += 1
                hit_stop = (active.direction == 'LONG'  and b['l'] <= active.stop) \
                        or (active.direction == 'SHORT' and b['h'] >= active.stop)
                hit_tgt  = (active.direction == 'LONG'  and b['h'] >= active.target) \
                        or (active.direction == 'SHORT' and b['l'] <= active.target)

                # Conservative: if both hit same bar, assume stop first
                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP')
                    trades.append(active)
                    daily_pnl[day] += active.net_pnl
                    active = None
                elif hit_tgt:
                    active.close(ts.isoformat(), active.target, 'TARGET')
                    trades.append(active)
                    daily_pnl[day] += active.net_pnl
                    active = None
                elif bars_held >= TIME_STOP_BARS:
                    active.close(ts.isoformat(), b['c'], 'TIME_STOP')
                    trades.append(active)
                    daily_pnl[day] += active.net_pnl
                    active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), b['c'], 'EOD')
                    trades.append(active)
                    daily_pnl[day] += active.net_pnl
                    active = None

            # --- Entry logic
            if (active is None and not traded_today
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END
                    and atr > 0 and len(tr_hist) >= ATR_PERIOD):
                deviation = (b['c'] - vwap) / atr
                # Need 2-bar confirmation
                if prev_deviation is not None:
                    if deviation > DEVIATION_ENTRY and prev_deviation > DEVIATION_CONFIRM:
                        # Fade — SHORT
                        stop = b['c'] + STOP_ATR_MULT * atr
                        target = b['c'] - TARGET_ATR_MULT * atr
                        risk_per_share = stop - b['c']
                        if risk_per_share > 0:
                            qty = int(RISK_PER_TRADE // risk_per_share)
                            if qty * b['c'] > MAX_NOTIONAL:
                                qty = int(MAX_NOTIONAL // b['c'])
                            if qty > 0:
                                active = Trade(symbol, day, 'SHORT', ts.isoformat(),
                                               b['c'], stop, target, qty, atr, deviation)
                                bars_held = 0
                                traded_today = True
                    elif deviation < -DEVIATION_ENTRY and prev_deviation < -DEVIATION_CONFIRM:
                        # Fade — LONG
                        stop = b['c'] - STOP_ATR_MULT * atr
                        target = b['c'] + TARGET_ATR_MULT * atr
                        risk_per_share = b['c'] - stop
                        if risk_per_share > 0:
                            qty = int(RISK_PER_TRADE // risk_per_share)
                            if qty * b['c'] > MAX_NOTIONAL:
                                qty = int(MAX_NOTIONAL // b['c'])
                            if qty > 0:
                                active = Trade(symbol, day, 'LONG', ts.isoformat(),
                                               b['c'], stop, target, qty, atr, deviation)
                                bars_held = 0
                                traded_today = True
                prev_deviation = deviation

            prev_close = b['c']

        # If day ends with position still open, close at last bar
        if active is not None and day_bars:
            last = day_bars[-1]
            active.close(last['ts'].isoformat(), last['c'], 'EOD_FORCED')
            trades.append(active)
            daily_pnl[day] += active.net_pnl

    return trades, dict(daily_pnl)


def compute_metrics(trades: list[Trade], daily_pnl: dict[str, float]) -> dict:
    if not trades:
        return {'trades': 0}
    net = [t.net_pnl for t in trades]
    wins = [p for p in net if p > 0]
    losses = [p for p in net if p < 0]
    total_net = sum(net)

    # Daily P&L series (combined across all stocks)
    sorted_days = sorted(daily_pnl.keys())
    daily_series = [daily_pnl[d] for d in sorted_days]

    # Equity curve
    equity = []
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in daily_series:
        running += pnl
        equity.append(running)
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized, 252 days)
    n_days = len(daily_series)
    if n_days > 1:
        mean = sum(daily_series) / n_days
        var = sum((x - mean) ** 2 for x in daily_series) / (n_days - 1)
        std = var ** 0.5
        sharpe = (mean / std) * (252 ** 0.5) if std > 0 else 0.0
    else:
        sharpe = 0.0

    # CAGR-equivalent on Rs 3L capital
    cap = 300_000
    years = n_days / 252 if n_days else 1
    ending = cap + total_net
    cagr = ((ending / cap) ** (1 / years) - 1) * 100 if years > 0 and ending > 0 else 0.0

    return {
        'trades':         len(trades),
        'wins':           len(wins),
        'losses':         len(losses),
        'win_rate_pct':   round(100 * len(wins) / len(trades), 2),
        'avg_win':        round(sum(wins) / len(wins), 0) if wins else 0,
        'avg_loss':       round(sum(losses) / len(losses), 0) if losses else 0,
        'profit_factor':  round(sum(wins) / abs(sum(losses)), 2) if losses else 0,
        'net_pnl':        round(total_net, 0),
        'gross_pnl':      round(sum(t.gross_pnl for t in trades), 0),
        'costs':          round(sum(t.gross_pnl - t.net_pnl for t in trades), 0),
        'days_traded':    n_days,
        'cagr_pct':       round(cagr, 2),
        'sharpe':         round(sharpe, 2),
        'max_dd':         round(max_dd, 0),
        'max_dd_pct':     round(100 * max_dd / cap, 2),
        'calmar':         round((cagr / (100 * max_dd / cap)), 2) if max_dd > 0 else 0,
    }


def main():
    all_trades: list[Trade] = []
    combined_daily: dict[str, float] = defaultdict(float)
    per_stock: dict[str, dict] = {}

    for i, sym in enumerate(UNIVERSE, 1):
        print(f'[{i:2d}/{len(UNIVERSE)}] {sym} ...', end='', flush=True)
        trades, daily = run_stock(sym)
        all_trades.extend(trades)
        for d, p in daily.items():
            combined_daily[d] += p
        m = compute_metrics(trades, daily)
        per_stock[sym] = m
        net = m.get('net_pnl', 0)
        print(f' {m.get("trades",0):>4} trades  net=Rs {net:>+10,.0f}  '
              f'WR {m.get("win_rate_pct",0):.1f}%  PF {m.get("profit_factor",0):.2f}')

    # --- Trade log CSV
    trade_path = OUT / 'trades.csv'
    with trade_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['symbol','date','direction','entry_time','entry','stop','target',
                    'qty','atr_at_entry','deviation_at_entry','exit_time','exit_price',
                    'exit_reason','gross_pnl','net_pnl'])
        for t in all_trades:
            w.writerow([t.symbol, t.date, t.direction, t.entry_time,
                        f'{t.entry:.2f}', f'{t.stop:.2f}', f'{t.target:.2f}', t.qty,
                        f'{t.atr_at_entry:.2f}', f'{t.deviation_at_entry:.2f}',
                        t.exit_time, f'{t.exit_price:.2f}', t.exit_reason,
                        f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

    # --- Daily P&L CSV (for ORB correlation)
    daily_path = OUT / 'daily_pnl.csv'
    with daily_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'daily_pnl'])
        for d in sorted(combined_daily.keys()):
            w.writerow([d, f'{combined_daily[d]:.2f}'])

    # --- Summary CSV
    portfolio = compute_metrics(all_trades, dict(combined_daily))
    summary_path = OUT / 'summary.csv'
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
            'profit_factor','net_pnl','gross_pnl','costs','days_traded',
            'cagr_pct','sharpe','max_dd','max_dd_pct','calmar']
    with summary_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stock'] + keys)
        for sym, m in per_stock.items():
            w.writerow([sym] + [m.get(k, '') for k in keys])
        w.writerow(['PORTFOLIO'] + [portfolio.get(k, '') for k in keys])

    print()
    print('=' * 70)
    print(f'PORTFOLIO ({len(UNIVERSE)} stocks, {START_DATE} → {END_DATE})')
    print('=' * 70)
    print(f'  Trades:       {portfolio["trades"]}  ({portfolio["wins"]} W / {portfolio["losses"]} L)')
    print(f'  Win rate:     {portfolio["win_rate_pct"]:.1f}%')
    print(f'  Avg win:      Rs {portfolio["avg_win"]:>+10,.0f}')
    print(f'  Avg loss:     Rs {portfolio["avg_loss"]:>+10,.0f}')
    print(f'  Profit factor:{portfolio["profit_factor"]:.2f}')
    print(f'  Net P&L:      Rs {portfolio["net_pnl"]:>+12,.0f}  (gross {portfolio["gross_pnl"]:+,.0f} − costs {portfolio["costs"]:,.0f})')
    print(f'  Days traded:  {portfolio["days_traded"]}')
    print(f'  CAGR on 3L:   {portfolio["cagr_pct"]:.2f}%')
    print(f'  Sharpe:       {portfolio["sharpe"]:.2f}')
    print(f'  Max DD:       Rs {portfolio["max_dd"]:,.0f}  ({portfolio["max_dd_pct"]:.2f}%)')
    print(f'  Calmar:       {portfolio["calmar"]:.2f}')
    print()
    print(f'Artifacts written to {OUT}')


if __name__ == '__main__':
    sys.exit(main())
