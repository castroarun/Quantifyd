"""Volume Breakout — walk-forward validation.

The v2 result (PF 1.35, Sharpe 1.96, +38% CAGR on 9 curated stocks) was
in-sample: the universe was selected FROM the same period's per-stock PF
table. Hindsight bias is the open caveat. This script tests the edge on
truly out-of-sample data.

Procedure:
  1. TRAIN window 2024-03-18 → 2025-03-15 (~12 months).
     Run v1 spike_3x_rsi rules on ALL 15 candidate stocks.
     Compute per-stock PF on train period.
     Curate: keep stocks where train PF >= 0.9 → "OOS universe".
  2. TEST window 2025-03-16 → 2026-03-12 (~12 months).
     Apply v2 spec (rr_2.5_spike_3x_rsi) ONLY to the train-curated universe.
     Report OOS metrics.
  3. Cross-check: also report in-sample metrics on the curated set
     (so we can compare degradation train→test).

Pass criteria for "real edge":
  OOS PF >= 1.15
  OOS Sharpe >= 0.8
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
OUT = Path(__file__).resolve().parents[1] / 'results' / 'walk_forward'
OUT.mkdir(parents=True, exist_ok=True)

# Periods
TRAIN_START = '2024-03-18'
TRAIN_END   = '2025-03-15'
TEST_START  = '2025-03-16'
TEST_END    = '2026-03-12'

# v1 broad universe (15 stocks)
FULL_UNIVERSE = [
    'INDUSINDBK', 'PNB', 'BANKBARODA',
    'WIPRO', 'HCLTECH',
    'MARUTI', 'HEROMOTOCO', 'EICHERMOT',
    'TITAN', 'ASIANPAINT',
    'JSWSTEEL', 'HINDALCO', 'JINDALSTEL',
    'LT', 'ADANIPORTS',
]

# Selection threshold (per train period PF)
CURATION_PF_THRESHOLD = 0.9

# Trade rules — same as v2 rr_2.5_spike_3x best variant
VOL_LOOKBACK         = 20
BREAKOUT_LOOKBACK    = 10
ATR_PERIOD           = 14
RSI_PERIOD           = 14
MIN_ATR_PCT          = 0.003
ENTRY_WINDOW_START   = dtime(9, 30)
ENTRY_WINDOW_END     = dtime(14, 0)
EOD_EXIT             = dtime(15, 15)
TIME_STOP_BARS       = 18
RSI_LONG_MIN         = 55.0
RSI_SHORT_MAX        = 45.0

# v1 evaluation: spike_3x + RSI gate, R:R 1.5
V1_VMULT             = 3.0
V1_R_MULTIPLE        = 1.5

# v2 selection: spike_3x + RSI gate, R:R 2.5
V2_VMULT             = 3.0
V2_R_MULTIPLE        = 2.5

# Sizing + costs
RISK_PER_TRADE       = 2_500
MAX_NOTIONAL         = 300_000
COST_PCT             = 0.0015


@dataclass
class Trade:
    symbol: str; date: str; direction: str
    entry_time: str; entry: float; stop: float; target: float; qty: int
    atr_at_entry: float; vol_spike: float
    exit_time: str = ''; exit_price: float = 0.0; exit_reason: str = ''
    gross_pnl: float = 0.0; net_pnl: float = 0.0

    def close(self, exit_time, exit_price, reason):
        self.exit_time, self.exit_price, self.exit_reason = exit_time, exit_price, reason
        sign = 1 if self.direction == 'LONG' else -1
        self.gross_pnl = sign * (exit_price - self.entry) * self.qty
        cost = COST_PCT * (self.entry + exit_price) * self.qty / 2.0
        self.net_pnl = self.gross_pnl - cost


def load_5min_period(symbol: str, start_date: str, end_date: str) -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
            WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=?
         ORDER BY date""",
        (symbol, start_date, end_date + ' 23:59:59'),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        ts = datetime.fromisoformat(r['date']) if isinstance(r['date'], str) else r['date']
        out.append({'ts': ts, 'o': r['open'], 'h': r['high'], 'l': r['low'],
                    'c': r['close'], 'v': r['volume']})
    return out


def run_backtest(symbol: str, start_date: str, end_date: str,
                 vmult: float, r_multiple: float) -> tuple[list[Trade], dict]:
    """Run volume-breakout-with-RSI rules on a date range. Returns (trades, daily_pnl)."""
    bars = load_5min_period(symbol, start_date, end_date)
    if not bars:
        return [], {}

    # Group by date
    by_date = {}
    for b in bars:
        d = b['ts'].date().isoformat()
        by_date.setdefault(d, []).append(b)

    trades = []
    daily = {}

    for day, day_bars in sorted(by_date.items()):
        from collections import deque
        vol_window = deque(maxlen=VOL_LOOKBACK)
        hl_window  = deque(maxlen=BREAKOUT_LOOKBACK)
        tr_hist    = deque(maxlen=ATR_PERIOD)
        avg_gain = avg_loss = None
        prev_close = None
        active = None; bars_held = 0; traded_today = False
        pending_entry = None

        for b in day_bars:
            ts = b['ts']; t = ts.time()
            tr = (b['h'] - b['l']) if prev_close is None else \
                max(b['h'] - b['l'], abs(b['h'] - prev_close), abs(b['l'] - prev_close))
            tr_hist.append(tr)
            atr = sum(tr_hist) / len(tr_hist) if tr_hist else 0.0

            # RSI (Wilder)
            rsi = None
            if prev_close is not None:
                ch = b['c'] - prev_close
                gain = max(ch, 0); loss = max(-ch, 0)
                if avg_gain is None:
                    avg_gain, avg_loss = gain, loss
                else:
                    avg_gain = (avg_gain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD
                    avg_loss = (avg_loss * (RSI_PERIOD - 1) + loss) / RSI_PERIOD
                if avg_loss > 0:
                    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
                elif avg_gain > 0:
                    rsi = 100.0

            # Fill pending entry at this bar's open
            if pending_entry is not None and active is None and not traded_today:
                direction, stop, atr_e, vspike = pending_entry
                entry_px = b['o']
                if entry_px > 0:
                    rps = abs(entry_px - stop)
                    if rps > 0:
                        target = entry_px + r_multiple * rps if direction == 'LONG' else entry_px - r_multiple * rps
                        qty = int(RISK_PER_TRADE // rps)
                        if qty * entry_px > MAX_NOTIONAL: qty = int(MAX_NOTIONAL // entry_px)
                        if qty > 0:
                            active = Trade(symbol, day, direction, ts.isoformat(),
                                           entry_px, stop, target, qty, atr_e, vspike)
                            bars_held = 0; traded_today = True
                pending_entry = None

            # Exit logic
            if active is not None:
                bars_held += 1
                hit_stop = (active.direction == 'LONG'  and b['l'] <= active.stop) \
                        or (active.direction == 'SHORT' and b['h'] >= active.stop)
                hit_tgt  = (active.direction == 'LONG'  and b['h'] >= active.target) \
                        or (active.direction == 'SHORT' and b['l'] <= active.target)
                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP')
                    trades.append(active); daily[day] = daily.get(day, 0) + active.net_pnl; active = None
                elif hit_tgt:
                    active.close(ts.isoformat(), active.target, 'TARGET')
                    trades.append(active); daily[day] = daily.get(day, 0) + active.net_pnl; active = None
                elif bars_held >= TIME_STOP_BARS:
                    active.close(ts.isoformat(), b['c'], 'TIME_STOP')
                    trades.append(active); daily[day] = daily.get(day, 0) + active.net_pnl; active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), b['c'], 'EOD')
                    trades.append(active); daily[day] = daily.get(day, 0) + active.net_pnl; active = None

            # Detect entry signal
            if (active is None and pending_entry is None and not traded_today
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END
                    and len(vol_window) == VOL_LOOKBACK
                    and len(hl_window) == BREAKOUT_LOOKBACK
                    and atr > 0 and b['c'] > 0):
                avg_vol = sum(vol_window) / VOL_LOOKBACK
                if avg_vol > 0 and atr / b['c'] >= MIN_ATR_PCT:
                    vspike = b['v'] / avg_vol
                    if vspike >= vmult:
                        prior_high = max(h for h, _ in hl_window)
                        prior_low  = min(l for _, l in hl_window)
                        bull = b['c'] > b['o'] and b['c'] >= prior_high
                        bear = b['c'] < b['o'] and b['c'] <= prior_low
                        rsi_long_ok = rsi is not None and rsi >= RSI_LONG_MIN
                        rsi_short_ok = rsi is not None and rsi <= RSI_SHORT_MAX
                        if bull and rsi_long_ok:
                            pending_entry = ('LONG', b['l'], atr, vspike)
                        elif bear and rsi_short_ok:
                            pending_entry = ('SHORT', b['h'], atr, vspike)

            # Roll windows after signal detection (next bar uses prior window)
            vol_window.append(b['v'])
            hl_window.append((b['h'], b['l']))
            prev_close = b['c']

        if active is not None and day_bars:
            last = day_bars[-1]
            active.close(last['ts'].isoformat(), last['c'], 'EOD_FORCED')
            trades.append(active); daily[day] = daily.get(day, 0) + active.net_pnl

    return trades, daily


def metrics(trades: list[Trade], daily: dict, period_days: int = None) -> dict:
    if not trades:
        return {k: 0 for k in ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                               'profit_factor','net_pnl','sharpe','max_dd','max_dd_pct','cagr_pct','calmar']}
    net = [t.net_pnl for t in trades]
    wins = [p for p in net if p > 0]; losses = [p for p in net if p < 0]
    total = sum(net)
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
    yrs = (period_days / 252) if period_days else (n / 252 if n else 1)
    end = cap + total
    cagr = ((end/cap)**(1/yrs) - 1) * 100 if yrs > 0 and end > 0 else 0.0
    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100*len(wins)/len(trades), 2),
        'avg_win': round(sum(wins)/len(wins), 0) if wins else 0,
        'avg_loss': round(sum(losses)/len(losses), 0) if losses else 0,
        'profit_factor': round(sum(wins)/abs(sum(losses)), 2) if losses else 0,
        'net_pnl': round(total, 0),
        'sharpe': round(sharpe, 2),
        'max_dd': round(mdd, 0),
        'max_dd_pct': round(100*mdd/cap, 2),
        'cagr_pct': round(cagr, 2),
        'calmar': round(cagr/(100*mdd/cap), 2) if mdd > 0 else 0,
    }


def main():
    t_start = time.time()

    # =========================================================
    # PHASE 1: TRAIN — run v1 rules on FULL universe
    # =========================================================
    print('=' * 95)
    print(f'PHASE 1 — TRAIN: {TRAIN_START} -> {TRAIN_END} (universe selection)')
    print('=' * 95)
    print(f'{"Stock":>13} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Net P&L":>13} {"Selected?":>10}')
    print('-' * 70)
    train_per_stock_trades = {}
    train_per_stock_metrics = {}
    curated_universe = []
    for sym in FULL_UNIVERSE:
        trades, daily = run_backtest(sym, TRAIN_START, TRAIN_END, V1_VMULT, V1_R_MULTIPLE)
        train_per_stock_trades[sym] = (trades, daily)
        m = metrics(trades, daily)
        train_per_stock_metrics[sym] = m
        selected = m['profit_factor'] >= CURATION_PF_THRESHOLD
        if selected: curated_universe.append(sym)
        flag = 'YES' if selected else 'no'
        print(f'{sym:>13} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'Rs {m["net_pnl"]:>+10,.0f} {flag:>10}')

    print()
    print(f'TRAIN selection — universe size: {len(curated_universe)}/{len(FULL_UNIVERSE)} (PF >= {CURATION_PF_THRESHOLD})')
    print(f'Selected: {", ".join(curated_universe) if curated_universe else "(none)"}')

    # Compare to the in-sample-curated universe from v2 (for reference)
    v2_in_sample_curated = ['ASIANPAINT', 'LT', 'JSWSTEEL', 'HEROMOTOCO', 'EICHERMOT',
                            'MARUTI', 'TITAN', 'HINDALCO', 'JINDALSTEL']
    print(f'(v2 hindsight-curated set was: {", ".join(v2_in_sample_curated)})')
    overlap = set(curated_universe) & set(v2_in_sample_curated)
    only_train = set(curated_universe) - set(v2_in_sample_curated)
    only_v2 = set(v2_in_sample_curated) - set(curated_universe)
    print(f'Overlap with v2 hindsight set: {len(overlap)} stocks')
    if only_train: print(f'  Only in train-curated: {sorted(only_train)}')
    if only_v2:    print(f'  Only in v2 hindsight (excluded by train): {sorted(only_v2)}')

    if not curated_universe:
        print('\n[ABORT] Train period yielded zero stocks above PF threshold. Edge does not exist.')
        return

    # =========================================================
    # PHASE 2A: IN-SAMPLE — apply v2 rules (R:R 2.5) on curated set, train period
    # =========================================================
    print()
    print('=' * 95)
    print(f'PHASE 2A — IN-SAMPLE: v2 rules (R:R {V2_R_MULTIPLE}) on train-curated universe, train period')
    print('=' * 95)
    is_trades, is_daily = [], {}
    is_per_stock = {}
    for sym in curated_universe:
        tr, dly = run_backtest(sym, TRAIN_START, TRAIN_END, V2_VMULT, V2_R_MULTIPLE)
        is_trades.extend(tr)
        for d, p in dly.items(): is_daily[d] = is_daily.get(d, 0) + p
        is_per_stock[sym] = metrics(tr, dly)

    is_metrics = metrics(is_trades, is_daily, period_days=252)  # ~12 months
    print(f'IS — Trades: {is_metrics["trades"]} | WR: {is_metrics["win_rate_pct"]:.1f}% | '
          f'PF: {is_metrics["profit_factor"]:.2f} | Net: Rs {is_metrics["net_pnl"]:+,.0f} | '
          f'Sharpe: {is_metrics["sharpe"]:.2f} | MaxDD: {is_metrics["max_dd_pct"]:.2f}% | '
          f'CAGR: {is_metrics["cagr_pct"]:+.2f}% | Calmar: {is_metrics["calmar"]:.2f}')

    # =========================================================
    # PHASE 2B: OUT-OF-SAMPLE — v2 rules on curated set, test period
    # =========================================================
    print()
    print('=' * 95)
    print(f'PHASE 2B — OUT-OF-SAMPLE: v2 rules on TRAIN-curated universe, test period {TEST_START} -> {TEST_END}')
    print('=' * 95)
    oos_trades, oos_daily = [], {}
    oos_per_stock = {}
    for sym in curated_universe:
        tr, dly = run_backtest(sym, TEST_START, TEST_END, V2_VMULT, V2_R_MULTIPLE)
        oos_trades.extend(tr)
        for d, p in dly.items(): oos_daily[d] = oos_daily.get(d, 0) + p
        oos_per_stock[sym] = metrics(tr, dly)

    oos_metrics = metrics(oos_trades, oos_daily, period_days=252)
    print(f'OOS — Trades: {oos_metrics["trades"]} | WR: {oos_metrics["win_rate_pct"]:.1f}% | '
          f'PF: {oos_metrics["profit_factor"]:.2f} | Net: Rs {oos_metrics["net_pnl"]:+,.0f} | '
          f'Sharpe: {oos_metrics["sharpe"]:.2f} | MaxDD: {oos_metrics["max_dd_pct"]:.2f}% | '
          f'CAGR: {oos_metrics["cagr_pct"]:+.2f}% | Calmar: {oos_metrics["calmar"]:.2f}')

    # Per-stock IS vs OOS
    print()
    print('-' * 95)
    print(f'PER-STOCK COMPARISON (IS train-period vs OOS test-period, R:R {V2_R_MULTIPLE})')
    print('-' * 95)
    print(f'{"Stock":>13} {"IS-Trades":>10} {"IS-PF":>7} {"IS-Net":>11} | {"OOS-Trades":>11} {"OOS-PF":>8} {"OOS-Net":>11}')
    print('-' * 95)
    for sym in curated_universe:
        iis = is_per_stock[sym]
        oos = oos_per_stock[sym]
        print(f'{sym:>13} {iis["trades"]:>10} {iis["profit_factor"]:>7.2f} '
              f'Rs{iis["net_pnl"]:>+9,.0f} | {oos["trades"]:>11} {oos["profit_factor"]:>8.2f} '
              f'Rs{oos["net_pnl"]:>+9,.0f}')

    # =========================================================
    # PASS/FAIL
    # =========================================================
    print()
    print('=' * 95)
    print('VERDICT')
    print('=' * 95)
    pass_pf     = oos_metrics['profit_factor'] >= 1.15
    pass_sharpe = oos_metrics['sharpe'] >= 0.8
    print(f'OOS PF     {oos_metrics["profit_factor"]:.2f} (need >= 1.15)  -> {"PASS" if pass_pf else "FAIL"}')
    print(f'OOS Sharpe {oos_metrics["sharpe"]:.2f} (need >= 0.80)  -> {"PASS" if pass_sharpe else "FAIL"}')
    print()
    if pass_pf and pass_sharpe:
        print('OVERALL: PASS  -> proceed to step 2 (universe finalization, build live executor)')
    else:
        print('OVERALL: FAIL  -> the 38% CAGR was hindsight overfit; do not advance to live')
    print()
    degradation_pf = (is_metrics['profit_factor'] - oos_metrics['profit_factor'])
    degradation_sharpe = (is_metrics['sharpe'] - oos_metrics['sharpe'])
    print(f'IS->OOS degradation: PF -{degradation_pf:.2f}, Sharpe -{degradation_sharpe:.2f}')

    # Save artifacts
    summary = {
        'train_period': f'{TRAIN_START} -> {TRAIN_END}',
        'test_period':  f'{TEST_START} -> {TEST_END}',
        'curated_universe': curated_universe,
        'in_sample_metrics': is_metrics,
        'oos_metrics': oos_metrics,
        'pass': pass_pf and pass_sharpe,
    }
    with (OUT / 'verdict.txt').open('w') as f:
        for k, v in summary.items():
            f.write(f'{k}: {v}\n')

    with (OUT / 'per_stock.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stock','train_v1_trades','train_v1_PF','is_v2_trades','is_v2_PF','is_v2_net','oos_v2_trades','oos_v2_PF','oos_v2_net'])
        for sym in FULL_UNIVERSE:
            tm = train_per_stock_metrics[sym]
            iis = is_per_stock.get(sym, {'trades':0,'profit_factor':0,'net_pnl':0})
            oos = oos_per_stock.get(sym, {'trades':0,'profit_factor':0,'net_pnl':0})
            w.writerow([sym, tm['trades'], tm['profit_factor'],
                        iis.get('trades',0), iis.get('profit_factor',0), iis.get('net_pnl',0),
                        oos.get('trades',0), oos.get('profit_factor',0), oos.get('net_pnl',0)])

    print(f'\nRuntime: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
