"""Small/micro-cap intraday breakout backtest engine.

Implements the rules from `data/future_plans.json` plan id
`smallcap-intraday-orb` exactly:

  Bars        : 5-min, 2024-03-18 to 2026-03-12
  OR window   : 09:15 - 09:45 (30-min OR)
  Entry       : 5-min CLOSE > OR_high (LONG) or < OR_low (SHORT)
  Filters     : VWAP align, RSI not extreme, gap > 2% skip
                (CPR width filter dropped per spec)
  Last entry  : 13:30
  EOD exit    : 15:18
  Stop        : OR opposite edge
  Target      : 1.5R or 2R (variant A/B)
  Trail       : at 14:00, if in profit move SL to entry + 0.5R
  Cost        : 0.5% round-trip
  Sizing      : Rs 2,400 risk/trade,
                qty = floor(2400 / |entry - SL|),
                cap qty * entry <= Rs 50,000 (no leverage)
  Concurrency : max 6 concurrent positions across the universe
  Capital     : Rs 3,00,000

Usage:
  python run_smallcap_backtest.py             # full sweep (universe.csv)
  python run_smallcap_backtest.py --limit 5   # dev pass on first 5 stocks
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from pathlib import Path

logging.disable(logging.WARNING)

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = Path(__file__).resolve().parents[1] / 'results'
LOGS = Path(__file__).resolve().parents[1] / 'logs'
OUT.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

UNIVERSE_CSV = OUT / 'universe.csv'
HEARTBEAT = LOGS / 'heartbeat.txt'

START_DATE = '2024-03-18'
END_DATE   = '2026-03-12'

# Session times (IST, naive)
SESSION_OPEN = dtime(9, 15)
OR_END       = dtime(9, 45)        # 30-min OR
LAST_ENTRY   = dtime(13, 30)
TRAIL_TIME   = dtime(14, 0)
EOD_EXIT     = dtime(15, 18)

# Filters
RSI_PERIOD          = 14
RSI_LONG_BLOCK      = 75.0
RSI_SHORT_BLOCK     = 25.0
GAP_BLOCK_PCT       = 0.02         # skip the day if open vs prev_close gap > 2%

# Sizing + costs
CAPITAL             = 3_00_000
RISK_PER_TRADE      = 2_400        # 0.8% of 3L
MAX_NOTIONAL_PER_POS = 50_000       # hard 50K cap, no leverage
MAX_CONCURRENT      = 6
COST_PCT_ROUND_TRIP = 0.005        # 0.5% round-trip, applied as half on entry + half on exit

# R:R variants
VARIANTS = {
    'target_1.5R': {'r_target': 1.5},
    'target_2.0R': {'r_target': 2.0},
}

TRAIL_LOCK_R = 0.5    # at 14:00 lock entry + 0.5R if in profit


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    variant: str
    symbol: str
    date: str
    direction: str
    or_high: float
    or_low: float
    entry_time: str
    entry: float
    stop: float
    target: float
    qty: int
    rsi_at_entry: float
    vwap_at_entry: float
    gap_pct: float
    exit_time: str = ''
    exit_price: float = 0.0
    exit_reason: str = ''
    trailed: bool = False
    gross_pnl: float = 0.0
    cost: float = 0.0
    net_pnl: float = 0.0

    def close(self, exit_time: str, exit_price: float, reason: str) -> None:
        self.exit_time = exit_time
        self.exit_price = float(exit_price)
        self.exit_reason = reason
        sign = 1 if self.direction == 'LONG' else -1
        self.gross_pnl = sign * (self.exit_price - self.entry) * self.qty
        # 0.5% round-trip = 0.25% on each side, applied to each leg's notional
        self.cost = COST_PCT_ROUND_TRIP * 0.5 * (self.entry + self.exit_price) * self.qty
        self.net_pnl = self.gross_pnl - self.cost


# ---------------------------------------------------------------------------
# Data load + indicators
# ---------------------------------------------------------------------------

def load_universe(limit: int | None = None) -> list[str]:
    if not UNIVERSE_CSV.exists():
        raise FileNotFoundError(
            f'{UNIVERSE_CSV} not found. Run build_universe.py first.'
        )
    syms: list[str] = []
    with UNIVERSE_CSV.open() as f:
        r = csv.DictReader(f)
        for row in r:
            syms.append(row['symbol'].strip())
    if limit is not None:
        syms = syms[:limit]
    return syms


def load_5min(conn: sqlite3.Connection, symbol: str) -> list[dict]:
    """Return list of 5-min bars chronological. Each bar:
       {'ts': datetime, 'o','h','l','c','v': float}."""
    rows = conn.execute(
        """SELECT date, open, high, low, close, volume
           FROM market_data_unified
           WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=?
           ORDER BY date""",
        (symbol, START_DATE, END_DATE + ' 23:59:59'),
    ).fetchall()
    out = []
    for r in rows:
        ts = datetime.fromisoformat(r[0]) if isinstance(r[0], str) else r[0]
        out.append({'ts': ts, 'o': float(r[1]), 'h': float(r[2]),
                    'l': float(r[3]), 'c': float(r[4]),
                    'v': float(r[5] or 0)})
    return out


def load_daily_close_map(conn: sqlite3.Connection, symbol: str) -> dict[str, float]:
    """Return {YYYY-MM-DD: close} for daily bars in window — used for gap calc."""
    rows = conn.execute(
        """SELECT date, close FROM market_data_unified
           WHERE symbol=? AND timeframe='day' AND date>=? AND date<=?
           ORDER BY date""",
        (symbol, START_DATE, END_DATE + ' 23:59:59'),
    ).fetchall()
    out = {}
    for d, c in rows:
        ds = d if isinstance(d, str) else d.isoformat()
        out[ds[:10]] = float(c)
    return out


def heartbeat(msg: str) -> None:
    try:
        with HEARTBEAT.open('a') as f:
            f.write(f'{datetime.now().isoformat(timespec="seconds")} | {msg}\n')
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Per-stock per-day simulation
# ---------------------------------------------------------------------------

def simulate_stock_day(
    symbol: str, day: str, day_bars: list[dict],
    prev_close: float | None, variant: str, r_target: float,
) -> Trade | None:
    """Run the intraday simulation for one symbol on one day under one variant.

    Returns the Trade (closed) if one was taken, else None."""
    if len(day_bars) < 8:           # need at least OR + a few post-bars
        return None

    # ---- Build OR (09:15 to 09:45) ----
    or_bars = [b for b in day_bars if b['ts'].time() < OR_END]
    if len(or_bars) < 6:            # 6 x 5-min == 30 min
        return None
    or_high = max(b['h'] for b in or_bars)
    or_low = min(b['l'] for b in or_bars)
    or_range = or_high - or_low
    if or_range <= 0:
        return None

    # ---- Daily-level filters (gap) ----
    day_open = day_bars[0]['o']
    gap_pct = 0.0
    if prev_close and prev_close > 0:
        gap_pct = (day_open - prev_close) / prev_close
    if abs(gap_pct) > GAP_BLOCK_PCT:
        return None

    # ---- VWAP + RSI cumulative state from session start ----
    cum_pv = 0.0
    cum_v = 0.0
    rsi_avg_gain: float | None = None
    rsi_avg_loss: float | None = None
    last_close: float | None = None

    # ---- Walk bars ----
    active: Trade | None = None
    last_or_close = or_bars[-1]['c']     # "prev_close" for first post-OR bar

    for b in day_bars:
        t = b['ts'].time()

        # Update VWAP
        tp = (b['h'] + b['l'] + b['c']) / 3.0
        cum_pv += tp * b['v']
        cum_v += b['v']
        vwap = cum_pv / cum_v if cum_v > 0 else b['c']

        # Update RSI (Wilder)
        rsi: float | None = None
        if last_close is not None:
            ch = b['c'] - last_close
            g = max(ch, 0.0)
            ll = max(-ch, 0.0)
            if rsi_avg_gain is None:
                rsi_avg_gain, rsi_avg_loss = g, ll
            else:
                rsi_avg_gain = (rsi_avg_gain * (RSI_PERIOD - 1) + g) / RSI_PERIOD
                rsi_avg_loss = (rsi_avg_loss * (RSI_PERIOD - 1) + ll) / RSI_PERIOD
            if rsi_avg_loss > 0:
                rsi = 100 - 100 / (1 + rsi_avg_gain / rsi_avg_loss)
            elif rsi_avg_gain > 0:
                rsi = 100.0

        last_close = b['c']

        # Skip OR bars (still building range, no trades pre-09:45)
        if t < OR_END:
            continue

        # ---- Manage active position ----
        if active is not None:
            # Trail-time lock (apply once at the first bar at/after 14:00)
            if (not active.trailed) and t >= TRAIL_TIME:
                if active.direction == 'LONG':
                    risk = active.entry - active.stop
                    new_stop = active.entry + TRAIL_LOCK_R * risk
                    # Only move SL up, and only if currently in profit at last close
                    if b['c'] >= new_stop and new_stop > active.stop:
                        active.stop = new_stop
                        active.trailed = True
                else:                                       # SHORT
                    risk = active.stop - active.entry
                    new_stop = active.entry - TRAIL_LOCK_R * risk
                    if b['c'] <= new_stop and new_stop < active.stop:
                        active.stop = new_stop
                        active.trailed = True

            # Exit checks (intrabar — high/low can hit either)
            hit_stop = (active.direction == 'LONG' and b['l'] <= active.stop) \
                    or (active.direction == 'SHORT' and b['h'] >= active.stop)
            hit_tgt  = (active.direction == 'LONG' and b['h'] >= active.target) \
                    or (active.direction == 'SHORT' and b['l'] <= active.target)

            # If both could hit on the same bar, assume the worse case (stop) —
            # standard backtest convention for ambiguous bars.
            if hit_stop:
                active.close(b['ts'].isoformat(), active.stop,
                             'TRAIL' if active.trailed else 'STOP')
                return active
            if hit_tgt:
                active.close(b['ts'].isoformat(), active.target, 'TARGET')
                return active

            # EOD square-off
            if t >= EOD_EXIT:
                active.close(b['ts'].isoformat(), b['c'], 'EOD')
                return active

            continue   # don't look for new entry while in a trade

        # ---- No position: look for breakout ----
        if t >= LAST_ENTRY:
            continue
        if t >= EOD_EXIT:
            continue

        long_breakout = b['c'] > or_high and last_or_close <= or_high
        short_breakout = b['c'] < or_low and last_or_close >= or_low
        # Note: last_or_close updates below as we walk

        if not (long_breakout or short_breakout):
            last_or_close = b['c']
            continue
        if long_breakout and short_breakout:
            last_or_close = b['c']
            continue

        direction = 'LONG' if long_breakout else 'SHORT'

        # Filter: VWAP alignment
        if direction == 'LONG' and b['c'] < vwap:
            last_or_close = b['c']
            continue
        if direction == 'SHORT' and b['c'] > vwap:
            last_or_close = b['c']
            continue

        # Filter: RSI not extreme
        rsi_at_entry = rsi if rsi is not None else 50.0
        if direction == 'LONG' and rsi_at_entry > RSI_LONG_BLOCK:
            last_or_close = b['c']
            continue
        if direction == 'SHORT' and rsi_at_entry < RSI_SHORT_BLOCK:
            last_or_close = b['c']
            continue

        # ---- Compute SL, target, qty ----
        entry = b['c']
        if direction == 'LONG':
            stop = or_low
            risk_per_share = entry - stop
            target = entry + r_target * risk_per_share
        else:
            stop = or_high
            risk_per_share = stop - entry
            target = entry - r_target * risk_per_share

        if risk_per_share <= 0:
            last_or_close = b['c']
            continue

        qty_risk = int(RISK_PER_TRADE // risk_per_share)
        qty_notional = int(MAX_NOTIONAL_PER_POS // entry)
        qty = max(0, min(qty_risk, qty_notional))
        if qty <= 0:
            last_or_close = b['c']
            continue

        active = Trade(
            variant=variant, symbol=symbol, date=day, direction=direction,
            or_high=or_high, or_low=or_low,
            entry_time=b['ts'].isoformat(), entry=entry,
            stop=stop, target=target, qty=qty,
            rsi_at_entry=round(rsi_at_entry, 2), vwap_at_entry=round(vwap, 2),
            gap_pct=round(gap_pct, 4),
        )
        last_or_close = b['c']

    # End-of-day: if still open, force close at last bar
    if active is not None and day_bars:
        last = day_bars[-1]
        active.close(last['ts'].isoformat(), last['c'], 'EOD_FORCED')
        return active

    return None


# ---------------------------------------------------------------------------
# Portfolio-level walk (with concurrency cap)
# ---------------------------------------------------------------------------

def run_variant(
    variant: str, r_target: float, universe: list[str], conn: sqlite3.Connection,
) -> tuple[list[Trade], dict[str, float]]:
    """Run all stocks in universe under one variant.

    Returns (trades, daily_pnl_total).

    Concurrency model: per the spec, max 6 concurrent positions across the
    portfolio. We compute each stock-day candidate independently first, then
    apply a portfolio-level cap by entry-time order: on each day, sort
    candidate trades by entry_time and accept only the first MAX_CONCURRENT
    that don't conflict with simultaneously-open peers.

    A "conflict" check counts how many existing open peers (entered earlier,
    not yet exited) overlap with the candidate's entry time. Because each
    stock can only take one trade per day, this approximates the live
    behavior where the executor would refuse the 7th simultaneous fill.
    """
    # Collect all candidate trades (one per stock-day max) without concurrency
    candidates: list[Trade] = []
    daily_close_cache: dict[str, dict[str, float]] = {}

    for i, sym in enumerate(universe, 1):
        heartbeat(f'{variant} | {i}/{len(universe)} {sym}')
        bars = load_5min(conn, sym)
        if not bars:
            continue
        daily_close_cache[sym] = load_daily_close_map(conn, sym)

        # Group by day
        by_day: dict[str, list[dict]] = defaultdict(list)
        for b in bars:
            by_day[b['ts'].date().isoformat()].append(b)

        prev_close: float | None = None
        sorted_days = sorted(by_day.keys())
        for day in sorted_days:
            day_bars = by_day[day]
            tr = simulate_stock_day(
                sym, day, day_bars,
                daily_close_cache[sym].get(_prev_day_str(daily_close_cache[sym], day)),
                variant, r_target,
            )
            if tr is not None:
                candidates.append(tr)
            # Carry-forward last close as crude prev_close for next day
            prev_close = day_bars[-1]['c']

    # Apply portfolio-level concurrency cap per day
    accepted: list[Trade] = []
    by_day_cands: dict[str, list[Trade]] = defaultdict(list)
    for t in candidates:
        by_day_cands[t.date].append(t)
    for day, trs in by_day_cands.items():
        trs.sort(key=lambda t: t.entry_time)
        open_peers: list[Trade] = []
        for t in trs:
            # Drop peers that exited before this candidate's entry
            open_peers = [p for p in open_peers if p.exit_time > t.entry_time]
            if len(open_peers) >= MAX_CONCURRENT:
                continue
            accepted.append(t)
            open_peers.append(t)

    # Daily P&L (net)
    daily_pnl: dict[str, float] = defaultdict(float)
    for t in accepted:
        daily_pnl[t.date] += t.net_pnl
    return accepted, dict(daily_pnl)


def _prev_day_str(close_map: dict[str, float], day: str) -> str:
    """Return the last day key in close_map strictly before `day`. '' if none."""
    keys = sorted(k for k in close_map if k < day)
    return keys[-1] if keys else ''


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[Trade], daily_pnl: dict[str, float]) -> dict:
    if not trades:
        keys = ['trades', 'wins', 'losses', 'win_rate_pct', 'avg_win', 'avg_loss',
                'profit_factor', 'net_pnl', 'gross_pnl', 'costs', 'days_traded',
                'cagr_pct', 'sharpe', 'max_dd', 'max_dd_pct', 'calmar']
        return {k: 0 for k in keys}
    nets = [t.net_pnl for t in trades]
    wins = [p for p in nets if p > 0]
    losses = [p for p in nets if p < 0]
    total_net = sum(nets)
    sorted_days = sorted(daily_pnl.keys())
    series = [daily_pnl[d] for d in sorted_days]
    running = peak = max_dd = 0.0
    for p in series:
        running += p
        if running > peak:
            peak = running
        max_dd = max(max_dd, peak - running)
    n = len(series)
    sharpe = 0.0
    if n > 1:
        mean = sum(series) / n
        std = (sum((x - mean) ** 2 for x in series) / (n - 1)) ** 0.5
        if std > 0:
            sharpe = (mean / std) * (252 ** 0.5)
    years = n / 252 if n else 1
    ending = CAPITAL + total_net
    cagr = ((ending / CAPITAL) ** (1 / years) - 1) * 100 if years > 0 and ending > 0 else 0.0
    return {
        'trades': len(trades),
        'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100 * len(wins) / len(trades), 2),
        'avg_win': round(sum(wins) / len(wins), 0) if wins else 0,
        'avg_loss': round(sum(losses) / len(losses), 0) if losses else 0,
        'profit_factor': round(sum(wins) / abs(sum(losses)), 2) if losses else 0,
        'net_pnl': round(total_net, 0),
        'gross_pnl': round(sum(t.gross_pnl for t in trades), 0),
        'costs': round(sum(t.cost for t in trades), 0),
        'days_traded': n, 'cagr_pct': round(cagr, 2), 'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 0),
        'max_dd_pct': round(100 * max_dd / CAPITAL, 2),
        'calmar': round(cagr / (100 * max_dd / CAPITAL), 2) if max_dd > 0 else 0,
    }


def per_stock_metrics(trades: list[Trade]) -> dict[str, dict]:
    by_sym: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_sym[t.symbol].append(t)
    out = {}
    for sym, trs in by_sym.items():
        # Build a per-stock daily series so MaxDD/Sharpe are consistent
        daily: dict[str, float] = defaultdict(float)
        for t in trs:
            daily[t.date] += t.net_pnl
        out[sym] = compute_metrics(trs, dict(daily))
    return out


# ---------------------------------------------------------------------------
# CSV writers (incremental)
# ---------------------------------------------------------------------------

TRADE_FIELDS = [
    'variant', 'symbol', 'date', 'direction', 'or_high', 'or_low',
    'entry_time', 'entry', 'stop', 'target', 'qty',
    'rsi_at_entry', 'vwap_at_entry', 'gap_pct',
    'exit_time', 'exit_price', 'exit_reason', 'trailed',
    'gross_pnl', 'cost', 'net_pnl',
]
SUMMARY_FIELDS = [
    'variant', 'scope', 'symbol', 'trades', 'wins', 'losses', 'win_rate_pct',
    'avg_win', 'avg_loss', 'profit_factor', 'net_pnl', 'gross_pnl', 'costs',
    'days_traded', 'cagr_pct', 'sharpe', 'max_dd', 'max_dd_pct', 'calmar',
]


def write_trades(trades: list[Trade], path: Path) -> None:
    new_file = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        if new_file:
            w.writeheader()
        for t in trades:
            w.writerow({
                'variant': t.variant, 'symbol': t.symbol, 'date': t.date,
                'direction': t.direction, 'or_high': f'{t.or_high:.2f}',
                'or_low': f'{t.or_low:.2f}', 'entry_time': t.entry_time,
                'entry': f'{t.entry:.2f}', 'stop': f'{t.stop:.2f}',
                'target': f'{t.target:.2f}', 'qty': t.qty,
                'rsi_at_entry': t.rsi_at_entry, 'vwap_at_entry': t.vwap_at_entry,
                'gap_pct': t.gap_pct, 'exit_time': t.exit_time,
                'exit_price': f'{t.exit_price:.2f}', 'exit_reason': t.exit_reason,
                'trailed': int(t.trailed),
                'gross_pnl': f'{t.gross_pnl:.2f}', 'cost': f'{t.cost:.2f}',
                'net_pnl': f'{t.net_pnl:.2f}',
            })


def write_summary_rows(rows: list[dict], path: Path) -> None:
    new_file = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in SUMMARY_FIELDS})


def write_equity(daily_pnl: dict[str, float], path: Path) -> None:
    sorted_days = sorted(daily_pnl.keys())
    eq = CAPITAL
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'daily_pnl', 'equity'])
        for d in sorted_days:
            eq += daily_pnl[d]
            w.writerow([d, f'{daily_pnl[d]:.2f}', f'{eq:.2f}'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None,
                    help='Only run first N stocks from universe (dev pass).')
    ap.add_argument('--variant', default=None,
                    help='Run only this variant (e.g. target_1.5R). Default = all.')
    args = ap.parse_args()

    universe = load_universe(args.limit)
    if not universe:
        print('Universe is empty. Run build_universe.py first.')
        return 2

    print(f'Small/microcap intraday backtest')
    print(f'  Period       : {START_DATE} to {END_DATE}')
    print(f'  Universe     : {len(universe)} stocks ({"limited" if args.limit else "full"})')
    print(f'  Variants     : {[args.variant] if args.variant else list(VARIANTS)}')
    print(f'  Capital      : Rs {CAPITAL:,}')
    print(f'  Risk/trade   : Rs {RISK_PER_TRADE:,}')
    print(f'  Max notional : Rs {MAX_NOTIONAL_PER_POS:,} per position, {MAX_CONCURRENT} concurrent')
    print()

    # Wipe heartbeat at start of run (fresh log)
    HEARTBEAT.write_text('')

    # Decide which output files to wipe up front so reruns are clean
    trades_path = OUT / 'trades.csv'
    summary_path = OUT / 'summary.csv'
    if args.variant is None and args.limit is None and trades_path.exists():
        trades_path.unlink()
    if args.variant is None and args.limit is None and summary_path.exists():
        summary_path.unlink()

    conn = sqlite3.connect(str(DB))

    variants_to_run = [args.variant] if args.variant else list(VARIANTS.keys())
    for vname in variants_to_run:
        if vname not in VARIANTS:
            print(f'Unknown variant: {vname}; valid = {list(VARIANTS)}')
            continue
        cfg = VARIANTS[vname]
        t0 = time.time()
        print(f'Running variant {vname} (r_target={cfg["r_target"]})...')
        trades, daily_pnl = run_variant(vname, cfg['r_target'], universe, conn)
        elapsed = time.time() - t0
        print(f'  variant {vname}: {len(trades)} trades in {elapsed:.1f}s')

        # Write trade log incrementally
        write_trades(trades, trades_path)

        # Per-stock + portfolio summary
        summary_rows: list[dict] = []
        port = compute_metrics(trades, daily_pnl)
        port_row = {'variant': vname, 'scope': 'PORTFOLIO', 'symbol': '', **port}
        summary_rows.append(port_row)
        for sym, m in per_stock_metrics(trades).items():
            summary_rows.append({'variant': vname, 'scope': 'STOCK',
                                 'symbol': sym, **m})
        write_summary_rows(summary_rows, summary_path)

        # Equity curve
        eq_path = OUT / f'equity_{vname}.csv'
        write_equity(daily_pnl, eq_path)

        # Print portfolio metrics
        print(f'  PORTFOLIO {vname}: trades={port["trades"]} '
              f'WR={port["win_rate_pct"]}% PF={port["profit_factor"]} '
              f'NetPnL=Rs{port["net_pnl"]:+,.0f} CAGR={port["cagr_pct"]}% '
              f'Sharpe={port["sharpe"]} MaxDD%={port["max_dd_pct"]}')
        sys.stdout.flush()

    conn.close()
    print()
    print('Done.')
    print(f'  Trades : {trades_path}')
    print(f'  Summary: {summary_path}')
    print(f'  Equity : {OUT}/equity_<variant>.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
