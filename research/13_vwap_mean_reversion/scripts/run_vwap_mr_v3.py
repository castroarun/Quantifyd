"""VWAP-MR v3 — target = VWAP, tighter entry, stoch+rsi confirm.

v2 lost with PF ~0.13 across all 4 filter variants because the 1:1 R:R
(1 ATR stop, 1 ATR target) needs 50%+ WR to be profitable — we got 30%.

v3 fixes R:R by targeting VWAP itself (the mean the system claims to revert
toward) and tightens entry to 2.5σ (only extreme setups). R:R becomes roughly
1 : 2.5, so at 30% WR we're slightly positive. Also tests direction-only
variants (long-only, short-only) since defensive stocks may have asymmetric
mean-reversion.
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

# Shared params
ATR_PERIOD          = 14
ENTRY_WINDOW_START  = dtime(11, 30)
ENTRY_WINDOW_END    = dtime(14, 0)
EOD_EXIT            = dtime(15, 15)
TIME_STOP_BARS      = 18                   # 90 min — give trades more room to reach VWAP
STOP_ATR_MULT       = 1.0

# Stochastic
STOCH_K_PERIOD, STOCH_SMOOTH, STOCH_D_PERIOD = 14, 3, 3
STOCH_OB, STOCH_OS = 80.0, 20.0

# RSI
RSI_PERIOD = 14
RSI_OB, RSI_OS = 70.0, 30.0

# Sizing + costs
RISK_PER_TRADE   = 2_500
MAX_NOTIONAL     = 300_000
COST_PCT         = 0.0015

# v3 variants — all use target=VWAP + tighter entry + stoch+rsi filters.
# We test: both directions vs long-only vs short-only, and dev threshold 2.0/2.5/3.0.
VARIANTS = {
    'both_2.0s':  {'long': True,  'short': True,  'dev': 2.0},
    'both_2.5s':  {'long': True,  'short': True,  'dev': 2.5},
    'both_3.0s':  {'long': True,  'short': True,  'dev': 3.0},
    'long_2.5s':  {'long': True,  'short': False, 'dev': 2.5},
    'short_2.5s': {'long': False, 'short': True,  'dev': 2.5},
}


@dataclass
class Trade:
    variant: str; symbol: str; date: str; direction: str
    entry_time: str; entry: float; stop: float; target_vwap: float; qty: int
    atr_at_entry: float; deviation_at_entry: float
    exit_time: str = ''; exit_price: float = 0.0; exit_reason: str = ''
    gross_pnl: float = 0.0; net_pnl: float = 0.0

    def close(self, exit_time, exit_price, reason):
        self.exit_time, self.exit_price, self.exit_reason = exit_time, exit_price, reason
        sign = 1 if self.direction == 'LONG' else -1
        self.gross_pnl = sign * (exit_price - self.entry) * self.qty
        cost = COST_PCT * (self.entry + exit_price) * self.qty / 2.0
        self.net_pnl = self.gross_pnl - cost


def load_bars(symbol):
    conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
            WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=?
         ORDER BY date""",
        (symbol, START_DATE, END_DATE + ' 23:59:59'),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        ts = datetime.fromisoformat(r['date']) if isinstance(r['date'], str) else r['date']
        out.append({'ts': ts, 'o': r['open'], 'h': r['high'], 'l': r['low'], 'c': r['close'], 'v': r['volume']})
    return out


def run_stock_variant(symbol, variant, cfg):
    bars = load_bars(symbol)
    if not bars:
        return [], {}
    by_date = defaultdict(list)
    for b in bars:
        by_date[b['ts'].date().isoformat()].append(b)
    trades = []
    daily = defaultdict(float)

    for day, day_bars in sorted(by_date.items()):
        cum_pv = cum_v = 0.0
        tr_hist = deque(maxlen=ATR_PERIOD)
        hl_window = deque(maxlen=STOCH_K_PERIOD)
        k_smooth_window = deque(maxlen=STOCH_SMOOTH)
        d_window = deque(maxlen=STOCH_D_PERIOD)
        avg_gain = avg_loss = None
        prev_close = prev_dev = prev_k = prev_d = prev_rsi = None
        active = None; bars_held = 0; traded_today = False

        for b in day_bars:
            ts = b['ts']; t = ts.time()
            typ = (b['h'] + b['l'] + b['c']) / 3.0
            cum_pv += typ * b['v']; cum_v += b['v']
            vwap = cum_pv / cum_v if cum_v else b['c']

            tr = (b['h'] - b['l']) if prev_close is None else \
                max(b['h'] - b['l'], abs(b['h'] - prev_close), abs(b['l'] - prev_close))
            tr_hist.append(tr)
            atr = sum(tr_hist) / len(tr_hist) if tr_hist else 0.0

            # Stoch
            hl_window.append((b['h'], b['l']))
            stoch_k = stoch_d = None
            if len(hl_window) == STOCH_K_PERIOD:
                hh = max(h for h, _ in hl_window); ll = min(l for _, l in hl_window)
                k_raw = 100 * (b['c'] - ll) / (hh - ll) if hh > ll else 50.0
                k_smooth_window.append(k_raw)
                if len(k_smooth_window) == STOCH_SMOOTH:
                    stoch_k = sum(k_smooth_window) / STOCH_SMOOTH
                    d_window.append(stoch_k)
                    if len(d_window) == STOCH_D_PERIOD:
                        stoch_d = sum(d_window) / STOCH_D_PERIOD

            # RSI
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

            # Exit
            if active is not None:
                bars_held += 1
                hit_stop = (active.direction == 'LONG'  and b['l'] <= active.stop) \
                        or (active.direction == 'SHORT' and b['h'] >= active.stop)
                # VWAP target: long exits when high >= current vwap; short when low <= current vwap
                hit_vwap = (active.direction == 'LONG'  and b['h'] >= vwap) \
                        or (active.direction == 'SHORT' and b['l'] <= vwap)
                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP')
                    trades.append(active); daily[day] += active.net_pnl; active = None
                elif hit_vwap:
                    # fill near vwap (use current vwap as exit)
                    active.close(ts.isoformat(), vwap, 'VWAP_TARGET')
                    trades.append(active); daily[day] += active.net_pnl; active = None
                elif bars_held >= TIME_STOP_BARS:
                    active.close(ts.isoformat(), b['c'], 'TIME_STOP')
                    trades.append(active); daily[day] += active.net_pnl; active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), b['c'], 'EOD')
                    trades.append(active); daily[day] += active.net_pnl; active = None

            # Entry
            if (active is None and not traded_today
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END
                    and atr > 0 and len(tr_hist) >= ATR_PERIOD):
                dev = (b['c'] - vwap) / atr
                # Stoch gate
                stoch_short_ok = stoch_long_ok = False
                if stoch_k is not None and prev_k is not None and stoch_d is not None and prev_d is not None:
                    stoch_short_ok = (prev_k >= prev_d and stoch_k < stoch_d and stoch_k > STOCH_OB)
                    stoch_long_ok  = (prev_k <= prev_d and stoch_k > stoch_d and stoch_k < STOCH_OS)
                # RSI gate
                rsi_short_ok = rsi_long_ok = False
                if rsi is not None and prev_rsi is not None:
                    rsi_short_ok = (prev_rsi > RSI_OB and rsi < prev_rsi)
                    rsi_long_ok  = (prev_rsi < RSI_OS and rsi > prev_rsi)

                if prev_dev is not None:
                    # SHORT
                    if (cfg['short'] and dev > cfg['dev'] and prev_dev > cfg['dev'] * 0.85
                            and stoch_short_ok and rsi_short_ok):
                        stop = b['c'] + STOP_ATR_MULT * atr
                        rps = stop - b['c']
                        if rps > 0:
                            qty = int(RISK_PER_TRADE // rps)
                            if qty * b['c'] > MAX_NOTIONAL: qty = int(MAX_NOTIONAL // b['c'])
                            if qty > 0:
                                active = Trade(variant, symbol, day, 'SHORT', ts.isoformat(),
                                               b['c'], stop, vwap, qty, atr, dev)
                                bars_held = 0; traded_today = True
                    # LONG
                    elif (cfg['long'] and dev < -cfg['dev'] and prev_dev < -cfg['dev'] * 0.85
                            and stoch_long_ok and rsi_long_ok):
                        stop = b['c'] - STOP_ATR_MULT * atr
                        rps = b['c'] - stop
                        if rps > 0:
                            qty = int(RISK_PER_TRADE // rps)
                            if qty * b['c'] > MAX_NOTIONAL: qty = int(MAX_NOTIONAL // b['c'])
                            if qty > 0:
                                active = Trade(variant, symbol, day, 'LONG', ts.isoformat(),
                                               b['c'], stop, vwap, qty, atr, dev)
                                bars_held = 0; traded_today = True
                prev_dev = dev

            prev_close = b['c']
            if stoch_k is not None: prev_k = stoch_k
            if stoch_d is not None: prev_d = stoch_d
            if rsi is not None: prev_rsi = rsi

        if active is not None and day_bars:
            last = day_bars[-1]
            active.close(last['ts'].isoformat(), last['c'], 'EOD_FORCED')
            trades.append(active); daily[day] += active.net_pnl

    return trades, dict(daily)


def compute_metrics(trades, daily):
    if not trades:
        return {k: 0 for k in ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                               'profit_factor','net_pnl','gross_pnl','costs','days_traded',
                               'cagr_pct','sharpe','max_dd','max_dd_pct','calmar']}
    net = [t.net_pnl for t in trades]
    wins = [p for p in net if p > 0]; losses = [p for p in net if p < 0]
    total_net = sum(net)
    sorted_days = sorted(daily.keys())
    series = [daily[d] for d in sorted_days]
    running = peak = max_dd = 0.0
    for pnl in series:
        running += pnl
        if running > peak: peak = running
        max_dd = max(max_dd, peak - running)
    n = len(series)
    sharpe = 0.0
    if n > 1:
        mean = sum(series) / n
        std = (sum((x - mean) ** 2 for x in series) / (n - 1)) ** 0.5
        if std > 0: sharpe = (mean / std) * (252 ** 0.5)
    cap = 300_000
    years = n / 252 if n else 1
    ending = cap + total_net
    cagr = ((ending / cap) ** (1 / years) - 1) * 100 if years > 0 and ending > 0 else 0.0
    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100 * len(wins) / len(trades), 2),
        'avg_win': round(sum(wins) / len(wins), 0) if wins else 0,
        'avg_loss': round(sum(losses) / len(losses), 0) if losses else 0,
        'profit_factor': round(sum(wins) / abs(sum(losses)), 2) if losses else 0,
        'net_pnl': round(total_net, 0),
        'gross_pnl': round(sum(t.gross_pnl for t in trades), 0),
        'costs': round(sum(t.gross_pnl - t.net_pnl for t in trades), 0),
        'days_traded': n, 'cagr_pct': round(cagr, 2), 'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 0), 'max_dd_pct': round(100 * max_dd / cap, 2),
        'calmar': round((cagr / (100 * max_dd / cap)), 2) if max_dd > 0 else 0,
    }


def main():
    all_trades = {v: [] for v in VARIANTS}
    daily_tot = {v: defaultdict(float) for v in VARIANTS}
    for i, sym in enumerate(UNIVERSE, 1):
        print(f'[{i:2d}/{len(UNIVERSE)}] {sym:12s}', end='')
        for vname, vcfg in VARIANTS.items():
            trades, daily = run_stock_variant(sym, vname, vcfg)
            all_trades[vname].extend(trades)
            for d, p in daily.items(): daily_tot[vname][d] += p
            net = sum(t.net_pnl for t in trades)
            print(f'  {vname}={len(trades):>3}/Rs{net:>+8,.0f}', end='')
        print()

    summary = OUT / 'summary_v3.csv'
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss','profit_factor',
            'net_pnl','gross_pnl','costs','days_traded','cagr_pct','sharpe',
            'max_dd','max_dd_pct','calmar']
    with summary.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant'] + keys)
        for vname in VARIANTS:
            m = compute_metrics(all_trades[vname], dict(daily_tot[vname]))
            w.writerow([vname] + [m.get(k, '') for k in keys])

    print()
    print('=' * 100)
    print(f'VWAP-MR v3 PORTFOLIO (target=VWAP, stoch+rsi confirm, {len(UNIVERSE)} stocks)')
    print('=' * 100)
    print(f'{"Variant":14s} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Net P&L":>14} {"CAGR%":>7} {"Sharpe":>7} {"MaxDD":>11} {"Calmar":>7}')
    print('-' * 100)
    for vname in VARIANTS:
        m = compute_metrics(all_trades[vname], dict(daily_tot[vname]))
        print(f'{vname:14s} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'Rs {m["net_pnl"]:>+11,.0f} {m["cagr_pct"]:>+6.2f} {m["sharpe"]:>7.2f} '
              f'{m["max_dd"]:>+10,.0f} {m["calmar"]:>7.2f}')


if __name__ == '__main__':
    sys.exit(main())
