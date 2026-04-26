"""
EOD Breakout Scanner — Live Service
=====================================
Daily scanner for the validated EOD breakout family (research/17, 19, 21).
Runs three sub-systems in parallel: Nifty 500, Small-cap, F&O. All use the
same rules — only the universe differs.

Daily lifecycle:
  16:00 IST  scan_eod()
             - Read today's daily bars (post-close)
             - For each system: detect breakout signals
             - Persist signals as PENDING in eod_signals

  09:20 IST  record_morning_fills()
             - Read today's open prices
             - Convert PENDING signals -> OPEN positions, capped by max_concurrent
             - Track signal_id -> position_id link

  16:00 IST  check_exits()
             - For each OPEN position: check today's high/low against target/stop
             - Close position + record trade if hit
             - Update equity curve

Exits triggered:
  TARGET     - high >= target (entry * 1.25)
  STOP       - low <= initial_stop
  MAX_HOLD   - bars_held >= 60 days, exit at close
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from services.eod_breakout_db import (
    get_eod_breakout_db, SYSTEM_NIFTY500, SYSTEM_SMALLCAP, SYSTEM_FNO, ALL_SYSTEMS,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
MARKET_DATA_DB = str(ROOT / 'backtest_data' / 'market_data.db')

# ---- Universe sources ----
NIFTY500_LIST = ROOT / 'data' / 'nifty500_list.csv'
SMALLCAP_UNIVERSE_CSV = ROOT / 'research' / '19_smallcap_daily' / 'results' / 'daily_universe.csv'
FNO_UNIVERSE_CSV = ROOT / 'research' / '21_eod_fno' / 'results' / 'fno_universe.csv'  # may not exist yet

# ---- Strategy parameters (locked from validated research/17 / 19 / 21) ----
BREAKOUT_LOOKBACK = 252            # 252-day high (52-week)
VOL_LOOKBACK      = 50             # 50-day volume avg
SMA_PERIOD        = 200            # regime filter
ATR_PERIOD        = 14
MAX_HOLD_DAYS     = 60

# Per-system config — only the UNIVERSE_LOADER and tuned params differ
SYSTEM_CONFIG = {
    SYSTEM_NIFTY500: {
        'capital': 10_00_000,
        'risk_per_trade_pct': 0.01,
        'max_concurrent': 10,
        'cost_pct': 0.0020,
        'vol_threshold_mult': 2.0,
        'target_pct': 0.25,
        'initial_hard_stop_pct': 0.08,
        'description': 'Nifty 500 EOD breakout (research/17 winner)',
    },
    SYSTEM_SMALLCAP: {
        'capital': 10_00_000,
        'risk_per_trade_pct': 0.01,
        'max_concurrent': 10,
        'cost_pct': 0.0030,            # wider spreads on small caps
        'vol_threshold_mult': 3.0,     # stricter volume confirm
        'target_pct': 0.25,
        'initial_hard_stop_pct': 0.08,
        'description': 'Small/Micro-cap (research/19 vol_3x winner)',
    },
    SYSTEM_FNO: {
        'capital': 10_00_000,
        'risk_per_trade_pct': 0.01,
        'max_concurrent': 10,
        'cost_pct': 0.0020,            # tight F&O spreads
        'vol_threshold_mult': 3.0,
        'target_pct': 0.25,
        'initial_hard_stop_pct': 0.08,
        'description': 'F&O-only (research/21) — options-overlay candidate',
    },
}


# ---------- Universe loaders ----------

def _load_nifty500_universe() -> list[str]:
    syms = []
    with NIFTY500_LIST.open() as f:
        for r in csv.DictReader(f):
            syms.append(r['Symbol'])
    return syms


def _load_smallcap_universe() -> list[str]:
    if not SMALLCAP_UNIVERSE_CSV.exists():
        return []
    syms = []
    with SMALLCAP_UNIVERSE_CSV.open() as f:
        # Universe csv may have header or just symbols — handle both
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            sym = row[0].strip()
            if sym and sym.lower() != 'symbol':
                syms.append(sym)
    return syms


def _load_fno_universe() -> list[str]:
    # First try the research/21 file (if agent has produced it)
    if FNO_UNIVERSE_CSV.exists():
        syms = []
        with FNO_UNIVERSE_CSV.open() as f:
            for row in csv.reader(f):
                if not row: continue
                sym = row[0].strip()
                if sym and sym.lower() != 'symbol':
                    syms.append(sym)
        if syms:
            return syms
    # Fallback: pull from FNO_LOT_SIZES dict
    try:
        from services.data_manager import FNO_LOT_SIZES
        return list(FNO_LOT_SIZES.keys())
    except Exception:
        return []


UNIVERSE_LOADERS = {
    SYSTEM_NIFTY500: _load_nifty500_universe,
    SYSTEM_SMALLCAP: _load_smallcap_universe,
    SYSTEM_FNO:      _load_fno_universe,
}


# ---------- Data + indicators ----------

def _load_daily_bars(symbol: str, n_back_days: int = 400) -> Optional[pd.DataFrame]:
    """Load last N daily bars for a symbol. Returns None if insufficient data."""
    cutoff = (date.today() - timedelta(days=n_back_days * 2)).isoformat()
    conn = sqlite3.connect(MARKET_DATA_DB)
    try:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' AND date>=? ORDER BY date",
            conn, params=(symbol, cutoff)
        )
    finally:
        conn.close()
    if df.empty or len(df) < BREAKOUT_LOOKBACK + 5:
        return None
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.set_index('date', inplace=True)
    return df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['high_252d'] = df['high'].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df['vol_avg']   = df['volume'].shift(1).rolling(VOL_LOOKBACK).mean()
    df['sma_200']   = df['close'].rolling(SMA_PERIOD).mean()
    pc = df['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    return df


# ---------- Daily lifecycle ----------

def scan_eod(scan_date: Optional[date] = None) -> dict:
    """Run after market close. For each system: detect breakouts, persist as
    PENDING signals. Idempotent (uses INSERT OR UPDATE on signal_date+symbol).
    Returns counts per system."""
    scan_date = scan_date or date.today()
    db = get_eod_breakout_db()
    summary = {sys_id: {'signals_found': 0, 'errors': 0} for sys_id in ALL_SYSTEMS}

    for sys_id in ALL_SYSTEMS:
        cfg = SYSTEM_CONFIG[sys_id]
        symbols = UNIVERSE_LOADERS[sys_id]()
        if not symbols:
            logger.warning(f"[EOD-SCAN] {sys_id}: empty universe")
            continue
        vol_mult = cfg['vol_threshold_mult']

        for sym in symbols:
            try:
                df = _load_daily_bars(sym)
                if df is None or scan_date not in df.index:
                    continue
                df = _add_indicators(df)
                row = df.loc[scan_date]

                if pd.isna(row['high_252d']) or pd.isna(row['vol_avg']) or pd.isna(row['sma_200']):
                    continue

                signal = (
                    row['close'] > row['high_252d']
                    and row['vol_avg'] > 0
                    and row['volume'] >= vol_mult * row['vol_avg']
                    and row['close'] > row['sma_200']
                )

                if signal:
                    rank_score = float(row['volume'] / row['vol_avg'])
                    db.add_signal(
                        sys_id, scan_date.isoformat(), sym,
                        signal_close=float(row['close']),
                        breakout_high=float(row['high_252d']),
                        vol_ratio=rank_score,
                        atr=float(row['atr']) if pd.notna(row['atr']) else None,
                        sma_200=float(row['sma_200']),
                        rank_score=rank_score,
                        status='PENDING',
                    )
                    summary[sys_id]['signals_found'] += 1
            except Exception as e:
                logger.error(f"[EOD-SCAN] {sys_id} {sym} error: {e}")
                summary[sys_id]['errors'] += 1

        # Update daily state
        db.upsert_daily_state(
            sys_id, scan_date.isoformat(),
            signals_generated=summary[sys_id]['signals_found'],
            open_positions=db.count_open_positions(sys_id),
        )
        logger.info(f"[EOD-SCAN] {sys_id} on {scan_date}: {summary[sys_id]['signals_found']} signals "
                    f"({summary[sys_id]['errors']} errors)")
    return summary


def record_morning_fills(fill_date: Optional[date] = None) -> dict:
    """Run after market open (~09:20 IST). For each PENDING signal, fill at
    today's open if max_concurrent allows. Mark as FILLED or SKIPPED_FULL or
    SKIPPED_NO_DATA."""
    fill_date = fill_date or date.today()
    db = get_eod_breakout_db()
    summary = {sys_id: {'fills': 0, 'skipped_full': 0, 'skipped_no_data': 0} for sys_id in ALL_SYSTEMS}

    for sys_id in ALL_SYSTEMS:
        cfg = SYSTEM_CONFIG[sys_id]
        equity = _current_equity(db, sys_id, cfg['capital'])
        risk_inr = equity * cfg['risk_per_trade_pct']
        notional_cap = equity / cfg['max_concurrent']

        pending = db.get_pending_signals(sys_id)
        # Sort by rank_score descending — strongest volume conviction first
        pending.sort(key=lambda r: -(r.get('rank_score') or 0))

        for sig in pending:
            open_count = db.count_open_positions(sys_id)
            if open_count >= cfg['max_concurrent']:
                db.update_signal_status(sig['id'], 'SKIPPED_FULL', notes=f'open={open_count}')
                summary[sys_id]['skipped_full'] += 1
                continue

            # Read today's open
            df = _load_daily_bars(sig['symbol'], n_back_days=10)
            if df is None or fill_date not in df.index:
                # Mark as SKIPPED_NO_DATA — try again tomorrow if signal still relevant
                continue
            row = df.loc[fill_date]
            entry_price = float(row['open'])
            if not (entry_price > 0):
                continue

            atr = sig.get('atr') or 0
            stop_atr_based = entry_price - 2 * atr if atr > 0 else 0
            stop_pct_based = entry_price * (1 - cfg['initial_hard_stop_pct'])
            initial_stop = max(stop_atr_based, stop_pct_based)
            risk_per_share = entry_price - initial_stop
            if risk_per_share <= 0:
                db.update_signal_status(sig['id'], 'SKIPPED_BAD_STOP')
                continue

            qty = int(risk_inr / risk_per_share)
            if qty * entry_price > notional_cap:
                qty = int(notional_cap / entry_price)
            if qty <= 0:
                db.update_signal_status(sig['id'], 'SKIPPED_TOO_SMALL')
                continue

            target = entry_price * (1 + cfg['target_pct'])
            position_id = db.add_position(
                system_id=sys_id,
                symbol=sig['symbol'],
                signal_date=sig['signal_date'],
                entry_date=fill_date.isoformat(),
                entry_price=entry_price,
                qty=qty,
                initial_stop=initial_stop,
                target=target,
                atr_at_entry=atr,
                risk_inr=risk_inr,
                notional_inr=qty * entry_price,
                status='OPEN',
                signal_id=sig['id'],
            )
            db.update_signal_status(sig['id'], 'FILLED', position_id=position_id)
            summary[sys_id]['fills'] += 1
            logger.info(f"[EOD-FILL] {sys_id} {sig['symbol']} qty={qty} @ {entry_price:.2f} "
                        f"stop={initial_stop:.2f} target={target:.2f}")
    return summary


def _current_equity(db, sys_id, starting_capital) -> float:
    """Compute current equity = starting_capital + sum(closed trade net_pnl)."""
    trades = db.get_recent_trades(sys_id, limit=10000)
    realized = sum((t.get('net_pnl') or 0) for t in trades)
    return starting_capital + realized


def check_exits(check_date: Optional[date] = None) -> dict:
    """Run after market close. For each OPEN position: check target/stop/max-hold."""
    check_date = check_date or date.today()
    db = get_eod_breakout_db()
    summary = {sys_id: {'exits': 0, 'pnl': 0.0} for sys_id in ALL_SYSTEMS}

    for sys_id in ALL_SYSTEMS:
        cfg = SYSTEM_CONFIG[sys_id]
        positions = db.get_open_positions(sys_id)

        for pos in positions:
            try:
                df = _load_daily_bars(pos['symbol'], n_back_days=15)
                if df is None or check_date not in df.index:
                    continue
                row = df.loc[check_date]
                entry_dt = datetime.fromisoformat(pos['entry_date']).date() \
                    if isinstance(pos['entry_date'], str) else pos['entry_date']
                days_held = (check_date - entry_dt).days

                exit_price = None
                exit_reason = None

                # Stop fires first (conservative)
                if row['low'] <= pos['initial_stop']:
                    exit_price = pos['initial_stop']
                    exit_reason = 'STOP'
                elif row['high'] >= pos['target']:
                    exit_price = pos['target']
                    exit_reason = 'TARGET'
                elif days_held >= MAX_HOLD_DAYS:
                    exit_price = float(row['close'])
                    exit_reason = 'MAX_HOLD'

                if exit_reason:
                    qty = pos['qty']
                    entry = pos['entry_price']
                    gross_pnl = (exit_price - entry) * qty
                    cost = cfg['cost_pct'] * (entry + exit_price) * qty / 2
                    net_pnl = gross_pnl - cost
                    db.close_position(
                        pos['id'], exit_price, check_date.isoformat(),
                        exit_reason, gross_pnl, net_pnl, cost, days_held,
                    )
                    summary[sys_id]['exits'] += 1
                    summary[sys_id]['pnl'] += net_pnl
                    logger.info(f"[EOD-EXIT] {sys_id} {pos['symbol']} {exit_reason} "
                                f"@ {exit_price:.2f}  netPnL={net_pnl:.0f}  days={days_held}")
            except Exception as e:
                logger.error(f"[EOD-EXIT] {sys_id} {pos.get('symbol')} error: {e}")

        # Update equity curve
        eq = _current_equity(db, sys_id, cfg['capital'])
        # Add MTM of open positions for live equity
        open_pos_now = db.get_open_positions(sys_id)
        unrealized = 0.0
        for p in open_pos_now:
            df = _load_daily_bars(p['symbol'], n_back_days=5)
            if df is not None and check_date in df.index:
                cur = df.loc[check_date]['close']
                unrealized += (cur - p['entry_price']) * p['qty']
        db.upsert_equity_point(sys_id, check_date.isoformat(), round(eq + unrealized, 2))

    return summary


def run_full_daily_cycle(d: Optional[date] = None) -> dict:
    """Convenience: scan -> fill -> exit (use this if you missed a day)."""
    d = d or date.today()
    s = scan_eod(d)
    f = record_morning_fills(d)
    e = check_exits(d)
    return {'scan': s, 'fills': f, 'exits': e}
