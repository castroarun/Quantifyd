"""Nifty 500 Intraday Momentum — live signal scanner.

Reuses the backtest signal logic verbatim:
  - research/31/scripts/signals_ccrb.py     (CCRB)
  - research/30/scripts/signals_volbreakout (vol-BO)

Lifecycle (called by app.py APScheduler jobs):

  1. precompute_setup(date)   — runs ~09:10 IST. For every stock in
     STOCK_CONFIGS, fetches last ~250 daily bars + a few weeks of intraday,
     computes today's CCRB qualifying gates and vol-BO baselines, and writes
     n500m_daily_state rows.

  2. scan_for_signals(now)    — runs every minute, 09:15-14:00 IST. For each
     qualifying daily-state row that hasn't fired yet, fetches today's
     intraday bars up to `now`, runs the corresponding signal generator on
     CLOSED bars only, and returns a list of fresh signals to hand to the
     executor.

  3. compute_sl_target(signal, exit_policy, atr_pts) — derives SL + target
     from the chosen exit policy. Trailing policies (T_CHANDELIER_*,
     T_STEP_TRAIL) return SL=initial_chandelier_stop with a flag indicating
     a trailing monitor is required. v1 implements: T_NO, T_HARD_SL,
     T_ATR_SL_*, T_R_TARGET_*, T_CHANDELIER_* (initial stop only),
     T_STEP_TRAIL (initial stop only — trailing logic is a v2 TODO).
"""
from __future__ import annotations

import logging
import re
import sys
from datetime import date as _date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from services.n500m_configs import StockConfig, load_all_configs
from services.n500m_db import get_db

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# Path setup so we can import the research signal modules directly.
_R30 = ROOT / "research" / "30_volume_breakout" / "scripts"
_R31 = ROOT / "research" / "31_cpr_compression_breakout" / "scripts"
for p in (_R30, _R31):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Now import the backtest modules
from signals_volbreakout import (  # type: ignore  # noqa: E402
    vol_breakout_signals, build_first_bars, VBSignal,
)
from signals_ccrb import (  # type: ignore  # noqa: E402
    ccrb_signals, daily_setup_table, qualifying_dates,
    build_bar_pos_vol_avg, CCRBSignal,
)

# ENTRY_OPEN/CUTOFF — match the backtest constants
ENTRY_OPEN_TIME = dtime(9, 20)
ENTRY_CUTOFF_TIME = dtime(14, 0)
EOD_SQUARE_OFF_TIME = dtime(15, 15)

# Liquidity (matches research/34 runners)
LIQUIDITY_MIN_PRICE = 50.0
LIQUIDITY_MIN_TURNOVER = 5_00_00_000  # Rs 5 cr / day median


# ---------------------------------------------------------------------------
# Data fetch — reads from market_data.db (refreshed daily by the existing
# data_manager). For TRUE live trading the next-day candles must be
# downloaded by the data refresh job before scan_for_signals runs.
# ---------------------------------------------------------------------------

def _db_path() -> str:
    return str(ROOT / "backtest_data" / "market_data.db")


def fetch_5min(symbol: str, start: _date, end: _date) -> pd.DataFrame:
    """Fetch 5-min bars for `symbol` between [start, end] inclusive.
    Returns DataFrame indexed by datetime (IST), columns OHLCV.
    """
    import sqlite3
    con = sqlite3.connect(_db_path())
    try:
        rows = con.execute(
            "SELECT date, open, high, low, close, volume "
            "FROM market_data_unified "
            "WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=? "
            "ORDER BY date",
            (symbol, start.isoformat(), end.isoformat() + " 23:59:59")
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def fetch_daily(symbol: str, start: _date, end: _date) -> pd.DataFrame:
    import sqlite3
    con = sqlite3.connect(_db_path())
    try:
        rows = con.execute(
            "SELECT date, open, high, low, close, volume "
            "FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? "
            "ORDER BY date",
            (symbol, start.isoformat(), end.isoformat())
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ---------------------------------------------------------------------------
# Liquidity gate (mirrors research/34 runners)
# ---------------------------------------------------------------------------

def liquidity_ok(daily: pd.DataFrame, signal_date: _date, lookback: int = 20) -> bool:
    if daily.empty:
        return False
    sd = pd.Timestamp(signal_date)
    prior = daily.loc[daily.index < sd].tail(lookback)
    if len(prior) < lookback:
        return False
    if float(prior["close"].median()) < LIQUIDITY_MIN_PRICE:
        return False
    if float((prior["close"] * prior["volume"]).median()) < LIQUIDITY_MIN_TURNOVER:
        return False
    return True


# ---------------------------------------------------------------------------
# ATR(14, daily) — for SL/target derivation
# ---------------------------------------------------------------------------

def daily_atr(daily: pd.DataFrame, n: int = 14) -> pd.Series:
    if daily.empty or len(daily) < n + 1:
        return pd.Series(dtype=float)
    h, l, c = daily["high"], daily["low"], daily["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def atr_for_date(atr: pd.Series, dt: _date) -> Optional[float]:
    if atr.empty:
        return None
    sd = pd.Timestamp(dt)
    prior = atr.loc[atr.index < sd]
    if prior.empty:
        return None
    v = float(prior.iloc[-1])
    return v if not np.isnan(v) else None


# ---------------------------------------------------------------------------
# Phase 1 — precompute_setup: daily-bar gates per (symbol, signal config)
# ---------------------------------------------------------------------------

def precompute_setup(today: Optional[_date] = None) -> dict:
    """Run early (09:10 IST) to compute today's setup table for every stock
    in STOCK_CONFIGS. Persists daily-state rows; returns {symbol: state}."""
    today = today or _date.today()
    db = get_db()
    cfgs = load_all_configs()
    out: dict = {}
    by_symbol: dict[str, list[StockConfig]] = {}
    for c in cfgs:
        by_symbol.setdefault(c.symbol, []).append(c)

    for sym, sym_cfgs in by_symbol.items():
        try:
            daily_start = today - timedelta(days=400)
            daily = fetch_daily(sym, daily_start, today)
            if daily.empty or len(daily) < 30:
                logger.warning(f"[N500M] {sym}: insufficient daily history")
                continue

            # Liquidity gate (mirrors backtest)
            if not liquidity_ok(daily, today):
                _persist_skip(db, sym, sym_cfgs, today, "skip:liquidity")
                continue

            # CCRB setup table — one row per qualifying day
            setup_tbl = daily_setup_table(daily)
            today_ts = pd.Timestamp(today).normalize()
            today_setup = (setup_tbl.loc[today_ts]
                           if (not setup_tbl.empty and today_ts in setup_tbl.index)
                           else None)

            for cfg in sym_cfgs:
                qualifies = 0
                reason = ""
                if cfg.signal == "ccrb":
                    if today_setup is None:
                        reason = "skip:no_setup_row"
                    else:
                        # Re-check the ctx gate exactly the way backtest does
                        q = qualifying_dates(
                            setup_tbl,
                            today_narrow=cfg.params.get("today_narrow", 0.005),
                            yesterday_ctx=cfg.params.get("yesterday_ctx", "W"),
                            yesterday_wide_thresh=cfg.params.get("y_wide", 0.005),
                            yesterday_narrow_range_thresh=cfg.params.get("y_narrow", 0.009),
                        )
                        qualifies = 1 if today_ts in q else 0
                        reason = "qualifies" if qualifies else "skip:setup_gate"
                else:  # volbo — every day is a candidate; the volume + close-past-PDH gate runs intraday
                    qualifies = 1
                    reason = "candidate (volume + PDH/PDL gate runs intraday)"

                state_row = dict(
                    symbol=sym,
                    signal_type=cfg.signal,
                    trade_date=today.isoformat(),
                    timeframe=cfg.timeframe,
                    direction=cfg.direction,
                    prev_day_high=float(daily["high"].iloc[-1]) if len(daily) else None,
                    prev_day_low=float(daily["low"].iloc[-1]) if len(daily) else None,
                    prev_day_close=float(daily["close"].iloc[-1]) if len(daily) else None,
                    today_open=(float(today_setup["today_open"])
                                if today_setup is not None else None),
                    today_cpr_width_pct=(float(today_setup["today_cpr_width_pct"])
                                         if today_setup is not None else None),
                    prev_day_cpr_width_pct=(float(today_setup["prev_cpr_width_pct"])
                                            if today_setup is not None else None),
                    prev_day_range_pct=(float(today_setup["prev_range_pct"])
                                        if today_setup is not None else None),
                    setup_qualifies=qualifies,
                    setup_reason=reason,
                )
                db.upsert_daily_state(**state_row)
                out[(sym, cfg.signal, cfg.timeframe, cfg.direction)] = state_row

        except Exception as e:
            logger.exception(f"[N500M] precompute_setup({sym}) failed: {e}")

    n_qual = sum(1 for v in out.values() if v.get("setup_qualifies"))
    logger.info(f"[N500M] precompute_setup: {len(out)} configs, {n_qual} qualify today")
    return out


def _persist_skip(db, sym, cfgs: list[StockConfig], today: _date, reason: str):
    for cfg in cfgs:
        db.upsert_daily_state(
            symbol=sym,
            signal_type=cfg.signal,
            trade_date=today.isoformat(),
            timeframe=cfg.timeframe,
            direction=cfg.direction,
            setup_qualifies=0,
            setup_reason=reason,
        )


# ---------------------------------------------------------------------------
# Phase 2 — scan_for_signals: walk closed intraday bars, fire signals
# ---------------------------------------------------------------------------

def scan_for_signals(now: Optional[datetime] = None) -> list[dict]:
    """Returns list of fresh signal dicts ready for the executor.

    Each dict carries:
      symbol, signal_type, timeframe, direction, entry_price, signal_time,
      sl_price, target_price, atr_pts, exit_policy, variant_raw,
      expected_sharpe, candle_open/high/low/close/volume, vm_ratio
    """
    now = now or datetime.now()
    today = now.date()
    if now.time() < ENTRY_OPEN_TIME or now.time() > ENTRY_CUTOFF_TIME:
        return []

    db = get_db()
    cfgs = load_all_configs()

    today_signal_keys = {
        (s["symbol"], s["signal_type"], s["timeframe"], s["direction"])
        for s in db.get_today_signals(today)
        if s.get("action_taken", "").startswith("ENTERED")
    }

    fresh: list[dict] = []

    by_symbol: dict[str, list[StockConfig]] = {}
    for c in cfgs:
        by_symbol.setdefault(c.symbol, []).append(c)

    for sym, sym_cfgs in by_symbol.items():
        try:
            # Fetch enough history: today's intraday + 25 prior trading days for
            # 20-day same-bar volume baseline + 400-day daily for ATR/setup.
            intraday_start = today - timedelta(days=45)  # ~30 trading days
            daily_start = today - timedelta(days=400)
            df5 = fetch_5min(sym, intraday_start, today)
            daily = fetch_daily(sym, daily_start, today)
            if df5.empty or daily.empty:
                continue

            # CRITICAL — only feed CLOSED bars. Drop any bar whose timestamp
            # is in the future (incomplete current candle).
            cutoff = pd.Timestamp(now)
            df5 = df5.loc[df5.index < cutoff]
            if df5.empty:
                continue

            atr_series = daily_atr(daily, 14)
            atr_pts = atr_for_date(atr_series, today)

            for cfg in sym_cfgs:
                key = (sym, cfg.signal, cfg.timeframe, cfg.direction)
                if key in today_signal_keys:
                    continue  # already fired today

                state = _get_state(db, sym, cfg, today)
                if not state or not state.get("setup_qualifies"):
                    continue

                signal = _try_emit(sym, cfg, df5, daily, today)
                if signal is None:
                    continue

                # SL + target from exit_policy
                sl, tgt, trail_flag = compute_sl_target(
                    direction=cfg.direction,
                    entry_price=signal["entry_price"],
                    candle_high=signal["candle_high"],
                    candle_low=signal["candle_low"],
                    atr_pts=atr_pts,
                    exit_policy=cfg.exit_policy,
                )
                signal.update(dict(
                    sl_price=sl, target_price=tgt, atr_pts=atr_pts,
                    exit_policy=cfg.exit_policy, variant_raw=cfg.variant_raw,
                    expected_sharpe=cfg.expected_sharpe,
                    requires_trailing=trail_flag,
                ))
                fresh.append(signal)

        except Exception as e:
            logger.exception(f"[N500M] scan_for_signals({sym}) failed: {e}")

    if fresh:
        logger.info(f"[N500M] scan@{now.strftime('%H:%M')}: {len(fresh)} fresh signal(s)")
    return fresh


def _get_state(db, sym: str, cfg: StockConfig, today: _date) -> Optional[dict]:
    """Fetch today's daily_state row matching this exact config."""
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM n500m_daily_state "
            "WHERE symbol=? AND signal_type=? AND trade_date=? "
            "AND timeframe=? AND direction=?",
            (sym, cfg.signal, today.isoformat(), cfg.timeframe, cfg.direction)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _try_emit(sym: str, cfg: StockConfig, df5: pd.DataFrame, daily: pd.DataFrame,
              today: _date) -> Optional[dict]:
    """Try to fire a signal using the backtest generator. Returns None if no signal."""
    if cfg.signal == "ccrb":
        gen = ccrb_signals(
            df5, daily,
            symbol=sym, cohort=cfg.cohort, timeframe=cfg.timeframe,
            today_narrow=cfg.params.get("today_narrow", 0.005),
            yesterday_ctx=cfg.params.get("yesterday_ctx", "W"),
            yesterday_wide_thresh=cfg.params.get("y_wide", 0.005),
            yesterday_narrow_range_thresh=cfg.params.get("y_narrow", 0.009),
            vol_mode=cfg.params.get("vol_mode", "off"),
            direction=cfg.direction,
        )
        for s in gen:
            if s.date.date() != today:
                continue
            return _ccrb_to_signal_dict(sym, cfg, s)
    else:  # volbo
        fb = build_first_bars(df5, cfg.timeframe)
        if fb.empty:
            return None
        # vol-BO needs the gap_pct param parsed back
        gap_pct = cfg.params.get("gap_pct")
        gap_mode = cfg.params.get("gap_mode")
        gap_arg = None if gap_mode == "off" else gap_pct
        use_rsi = (cfg.params.get("rsi_mode") != "off"
                   and "rsi_short_thresh" in cfg.params)
        rsi_low = float(cfg.params.get("rsi_short_thresh", 40))
        rsi_high = float(cfg.params.get("rsi_long_thresh", 60))
        gen = vol_breakout_signals(
            df5, fb, daily,
            vol_mult=cfg.params.get("vm", 1.5),
            gap_pct=gap_arg,
            use_rsi=use_rsi,
            rsi_low=rsi_low, rsi_high=rsi_high,
            direction=cfg.direction,
            symbol=sym, timeframe=cfg.timeframe,
        )
        for s in gen:
            if s.date.date() != today:
                continue
            return _volbo_to_signal_dict(sym, cfg, s)
    return None


def _ccrb_to_signal_dict(sym: str, cfg: StockConfig, s: CCRBSignal) -> dict:
    return dict(
        symbol=sym,
        signal_type=cfg.signal,
        trade_date=s.date.date().isoformat(),
        signal_time=s.signal_time.isoformat(),
        timeframe=cfg.timeframe,
        direction=cfg.direction,
        entry_price=float(s.entry_price),
        candle_open=float(s.bar_open),
        candle_high=float(s.bar_high),
        candle_low=float(s.bar_low),
        candle_close=float(s.bar_close),
        candle_volume=int(s.bar_volume),
        vm_ratio=float(s.vol_ratio) if not np.isnan(s.vol_ratio) else None,
    )


def _volbo_to_signal_dict(sym: str, cfg: StockConfig, s: VBSignal) -> dict:
    return dict(
        symbol=sym,
        signal_type=cfg.signal,
        trade_date=s.date.date().isoformat(),
        signal_time=s.signal_time.isoformat(),
        timeframe=cfg.timeframe,
        direction=cfg.direction,
        entry_price=float(s.signal_price),
        candle_open=float(s.first_bar_open),
        candle_high=float(s.first_bar_high),
        candle_low=float(s.first_bar_low),
        candle_close=float(s.first_bar_close),
        candle_volume=int(s.first_bar_volume),
        vm_ratio=float(s.vol_ratio),
    )


# ---------------------------------------------------------------------------
# Phase 3 — exit-policy SL/target derivation
# ---------------------------------------------------------------------------

HARD_SL_PCT = 0.05  # T_HARD_SL = -5% from entry


def compute_sl_target(*, direction: str, entry_price: float,
                      candle_high: float, candle_low: float,
                      atr_pts: Optional[float], exit_policy: str
                      ) -> tuple[Optional[float], Optional[float], bool]:
    """Returns (sl_price, target_price, requires_trailing).

    For T_NO: (None, None, False) — exit at EOD only.
    For T_HARD_SL: SL at -5%, no target.
    For T_ATR_SL_x.x: SL = entry ± x.x * ATR.
    For T_R_TARGET_x.xR: SL = entry ± 1.0 * ATR; target = entry ± x.x * (entry - SL).
    For T_CHANDELIER_x.x: SL = candle high/low ± x.x * ATR (initial). Trailing required.
    For T_STEP_TRAIL: SL = entry ± 1.0 * ATR (initial). Trailing required.

    NOTE — trailing logic for T_CHANDELIER_* and T_STEP_TRAIL not yet
    implemented in v1; positions exit on initial stop or 15:15 EOD.
    The flag returned tells the executor to re-evaluate the SL each minute.
    """
    sign = 1 if direction == "long" else -1
    sl = None
    tgt = None
    trail = False

    if exit_policy == "T_NO":
        return None, None, False

    if exit_policy == "T_HARD_SL":
        sl = entry_price * (1 - sign * HARD_SL_PCT)
        return round(sl, 2), None, False

    m = re.match(r"T_ATR_SL_([0-9.]+)$", exit_policy)
    if m and atr_pts:
        mult = float(m.group(1))
        sl = entry_price - sign * mult * atr_pts
        return round(sl, 2), None, False

    m = re.match(r"T_R_TARGET_([0-9.]+)R$", exit_policy)
    if m and atr_pts:
        rr = float(m.group(1))
        sl = entry_price - sign * 1.0 * atr_pts
        risk_pts = abs(entry_price - sl)
        tgt = entry_price + sign * rr * risk_pts
        return round(sl, 2), round(tgt, 2), False

    m = re.match(r"T_CHANDELIER_([0-9.]+)$", exit_policy)
    if m and atr_pts:
        mult = float(m.group(1))
        anchor = candle_high if direction == "long" else candle_low
        sl = anchor - sign * mult * atr_pts
        return round(sl, 2), None, True

    if exit_policy == "T_STEP_TRAIL" and atr_pts:
        sl = entry_price - sign * 1.0 * atr_pts
        return round(sl, 2), None, True

    # Unknown policy — fail-safe to no SL/target (will exit at EOD)
    logger.warning(f"[N500M] unknown exit_policy={exit_policy} — defaulting to T_NO behaviour")
    return None, None, False
