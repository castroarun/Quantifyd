"""Volume Breakout — momentum system on 5-min bars.

Thesis: a 5-min bar that prints >N× its 20-bar average volume AND breaks
the prior 10-bar high (long) / low (short) signals institutional order flow.
Enter on next bar, target 1.5R, stop at the breakout bar's opposite edge.

Not truly uncorrelated with ORB (both are momentum) but:
  - Different trigger (volume spike vs opening-range edge)
  - Different entry times (distributed 09:30-14:00 vs opening 09:30-14:00)
  - Different universe (15 mid/large-cap F&O names, no ORB overlap)
  → operationally independent, partial P&L diversification.

Universe selection: Nifty 500 F&O names with 5-min data since 2024-03, mixed
across banks / IT / auto / metals / consumer / infra, zero overlap with ORB.
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
    # Banks (no overlap with ORB's AXISBANK)
    'INDUSINDBK', 'PNB', 'BANKBARODA',
    # IT
    'WIPRO', 'HCLTECH',
    # Auto (no overlap with ORB's M&M)
    'MARUTI', 'HEROMOTOCO', 'EICHERMOT',
    # Consumer
    'TITAN', 'ASIANPAINT',
    # Metals (no overlap with ORB's TATASTEEL/VEDL)
    'JSWSTEEL', 'HINDALCO', 'JINDALSTEL',
    # Infra / caps
    'LT', 'ADANIPORTS',
]

START_DATE = '2024-03-18'
END_DATE   = '2026-03-12'

# Signal params
VOL_LOOKBACK         = 20          # 20-bar rolling avg volume baseline
BREAKOUT_LOOKBACK    = 10          # 10-bar price high/low for breakout
ATR_PERIOD           = 14
MIN_ATR_PCT          = 0.003       # skip if ATR/price < 0.3% (too quiet)

# Entry window + exit
ENTRY_WINDOW_START   = dtime(9, 30)
ENTRY_WINDOW_END     = dtime(14, 0)
EOD_EXIT             = dtime(15, 15)
TIME_STOP_BARS       = 12

# R:R
R_MULTIPLE           = 1.5         # target = 1.5× risk

# RSI filter (used by _rsi variants)
RSI_PERIOD           = 14
RSI_LONG_MIN         = 55.0
RSI_SHORT_MAX        = 45.0

# Sizing + costs
RISK_PER_TRADE       = 2_500
MAX_NOTIONAL         = 300_000
COST_PCT             = 0.0015

VARIANTS = {
    'spike_2x':         {'vmult': 2.0, 'use_rsi': False},
    'spike_3x':         {'vmult': 3.0, 'use_rsi': False},
    'spike_2x_rsi':     {'vmult': 2.0, 'use_rsi': True},
    'spike_3x_rsi':     {'vmult': 3.0, 'use_rsi': True},
}


@dataclass
class Trade:
    variant: str; symbol: str; date: str; direction: str
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
        # Intraday indicators
        vol_window = deque(maxlen=VOL_LOOKBACK)
        hl_window = deque(maxlen=BREAKOUT_LOOKBACK)   # tuples (high, low)
        tr_hist = deque(maxlen=ATR_PERIOD)
        avg_gain = avg_loss = None
        prev_close = prev_rsi = None
        active = None; bars_held = 0; traded_today = False
        pending_entry = None   # enter on NEXT bar's open

        for b in day_bars:
            ts = b['ts']; t = ts.time()

            # ATR
            tr = (b['h'] - b['l']) if prev_close is None else \
                max(b['h'] - b['l'], abs(b['h'] - prev_close), abs(b['l'] - prev_close))
            tr_hist.append(tr)
            atr = sum(tr_hist) / len(tr_hist) if tr_hist else 0.0

            # RSI
            rsi = None
            if prev_close is not None:
                ch = b['c'] - prev_close
                g = max(ch, 0); l = max(-ch, 0)
                if avg_gain is None:
                    avg_gain, avg_loss = g, l
                else:
                    avg_gain = (avg_gain * (RSI_PERIOD - 1) + g) / RSI_PERIOD
                    avg_loss = (avg_loss * (RSI_PERIOD - 1) + l) / RSI_PERIOD
                if avg_loss > 0:
                    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
                elif avg_gain > 0:
                    rsi = 100.0

            # Execute pending entry at this bar's open
            if pending_entry is not None and active is None and not traded_today:
                direction, stop, target, atr_e, vspike = pending_entry
                entry_px = b['o']
                # Recompute stop/target proportional to entry_px vs signal close
                rps = abs(entry_px - stop)
                if rps > 0:
                    qty = int(RISK_PER_TRADE // rps)
                    if qty * entry_px > MAX_NOTIONAL:
                        qty = int(MAX_NOTIONAL // entry_px)
                    if qty > 0:
                        active = Trade(variant, symbol, day, direction, ts.isoformat(),
                                       entry_px, stop, target, qty, atr_e, vspike)
                        bars_held = 0; traded_today = True
                pending_entry = None

            # Exit logic for active trade
            if active is not None:
                bars_held += 1
                hit_stop = (active.direction == 'LONG'  and b['l'] <= active.stop) \
                        or (active.direction == 'SHORT' and b['h'] >= active.stop)
                hit_tgt  = (active.direction == 'LONG'  and b['h'] >= active.target) \
                        or (active.direction == 'SHORT' and b['l'] <= active.target)
                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP')
                    trades.append(active); daily[day] += active.net_pnl; active = None
                elif hit_tgt:
                    active.close(ts.isoformat(), active.target, 'TARGET')
                    trades.append(active); daily[day] += active.net_pnl; active = None
                elif bars_held >= TIME_STOP_BARS:
                    active.close(ts.isoformat(), b['c'], 'TIME_STOP')
                    trades.append(active); daily[day] += active.net_pnl; active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), b['c'], 'EOD')
                    trades.append(active); daily[day] += active.net_pnl; active = None

            # Entry signal detection (only if no active trade, not traded yet, in window)
            if (active is None and pending_entry is None and not traded_today
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END
                    and len(vol_window) == VOL_LOOKBACK
                    and len(hl_window) == BREAKOUT_LOOKBACK
                    and atr > 0 and b['c'] > 0):
                avg_vol = sum(vol_window) / VOL_LOOKBACK
                if avg_vol > 0 and atr / b['c'] >= MIN_ATR_PCT:
                    vspike = b['v'] / avg_vol
                    if vspike >= cfg['vmult']:
                        prior_high = max(h for h, _ in hl_window)
                        prior_low  = min(l for _, l in hl_window)
                        bull = b['c'] > b['o'] and b['c'] >= prior_high
                        bear = b['c'] < b['o'] and b['c'] <= prior_low

                        rsi_long_ok = True if not cfg['use_rsi'] else (rsi is not None and rsi >= RSI_LONG_MIN)
                        rsi_short_ok = True if not cfg['use_rsi'] else (rsi is not None and rsi <= RSI_SHORT_MAX)

                        if bull and rsi_long_ok:
                            stop = b['l']       # breakout bar's low
                            risk_per_share = b['c'] - stop
                            if risk_per_share > 0:
                                target = b['c'] + R_MULTIPLE * risk_per_share
                                pending_entry = ('LONG', stop, target, atr, vspike)
                        elif bear and rsi_short_ok:
                            stop = b['h']       # breakout bar's high
                            risk_per_share = stop - b['c']
                            if risk_per_share > 0:
                                target = b['c'] - R_MULTIPLE * risk_per_share
                                pending_entry = ('SHORT', stop, target, atr, vspike)

            # Update rolling windows (after signal detection, so signal uses prior window)
            vol_window.append(b['v'])
            hl_window.append((b['h'], b['l']))
            prev_close = b['c']
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

    # Trade log
    with (OUT / 'trades.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','date','direction','entry_time','entry','stop','target',
                    'qty','atr_at_entry','vol_spike','exit_time','exit_price','exit_reason',
                    'gross_pnl','net_pnl'])
        for v, tr in all_trades.items():
            for t in tr:
                w.writerow([t.variant, t.symbol, t.date, t.direction, t.entry_time,
                            f'{t.entry:.2f}', f'{t.stop:.2f}', f'{t.target:.2f}', t.qty,
                            f'{t.atr_at_entry:.2f}', f'{t.vol_spike:.2f}',
                            t.exit_time, f'{t.exit_price:.2f}', t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

    # Daily P&L per variant (for correlation analysis later)
    all_days = sorted({d for v in daily_tot.values() for d in v.keys()})
    with (OUT / 'daily_pnl.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['date'] + list(VARIANTS.keys()))
        for d in all_days:
            w.writerow([d] + [f'{daily_tot[v].get(d, 0):.2f}' for v in VARIANTS])

    # Summary
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss','profit_factor',
            'net_pnl','gross_pnl','costs','days_traded','cagr_pct','sharpe',
            'max_dd','max_dd_pct','calmar']
    with (OUT / 'summary.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant'] + keys)
        for vname in VARIANTS:
            m = compute_metrics(all_trades[vname], dict(daily_tot[vname]))
            w.writerow([vname] + [m.get(k, '') for k in keys])

    print()
    print('=' * 100)
    print(f'VOLUME BREAKOUT PORTFOLIO ({len(UNIVERSE)} stocks, {START_DATE} to {END_DATE})')
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
