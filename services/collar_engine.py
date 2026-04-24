"""
Collar Paper-Trading Engine
============================

Paper-mode engine that takes KC6 entry/exit signals on the Nifty 500 ∩ F&O
universe and simulates a 3-leg options collar for each:

    1. Long 1 lot Futures
    2. Long 1 lot ~5% OTM Put
    3. Short 1 lot ~5% OTM Call

No real option chain data is used. Leg prices are computed with a simplified
Black-Scholes model with flat 25% IV (configurable) and a 6.5% risk-free rate.
Futures = spot (paper mode; ignores cost-of-carry basis).

Reuses services/kc6_scanner.py for indicator + signal logic. Does NOT touch
the existing KC6 code paths.
"""

from __future__ import annotations

import calendar
import logging
import math
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import COLLAR_DEFAULTS, KC6_DEFAULTS, MARKET_DATA_DB
from services.collar_db import get_collar_db

logger = logging.getLogger(__name__)


# =============================================================================
# Black-Scholes (simplified) for paper option pricing
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, t_years: float,
             r: float, iv: float, is_call: bool) -> float:
    """
    Black-Scholes price for European option.
    Paper-mode only: flat IV, no dividend.
    Returns >= 0.01 (floor to avoid zero/neg on deep OTM).
    """
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        # Intrinsic only
        if is_call:
            return max(spot - strike, 0.01)
        return max(strike - spot, 0.01)

    sigma_t = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / sigma_t
    d2 = d1 - sigma_t

    if is_call:
        price = spot * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
    else:
        price = strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

    return max(round(price, 2), 0.01)


# =============================================================================
# Strike rounding (Indian F&O ladder)
# =============================================================================

def infer_strike_interval(spot: float) -> int:
    """
    Infer strike interval based on spot price.
    Rules (project spec):
      S < 1000    -> 10
      S < 5000    -> 50
      S < 20000   -> 100
      else        -> 500
    """
    if spot < 1000:
        return 10
    if spot < 5000:
        return 50
    if spot < 20000:
        return 100
    return 500


def round_put_strike(spot: float, otm_pct: float) -> float:
    """Put target = spot*(1 - otm_pct/100), rounded DOWN to interval."""
    target = spot * (1 - otm_pct / 100.0)
    interval = infer_strike_interval(spot)
    return math.floor(target / interval) * interval


def round_call_strike(spot: float, otm_pct: float) -> float:
    """Call target = spot*(1 + otm_pct/100), rounded UP to interval."""
    target = spot * (1 + otm_pct / 100.0)
    interval = infer_strike_interval(spot)
    return math.ceil(target / interval) * interval


# =============================================================================
# Monthly expiry math
# =============================================================================

def last_thursday_of_month(year: int, month: int) -> date:
    """Return the last Thursday of a given month (monthly F&O expiry)."""
    # calendar.monthcalendar gives weeks as lists; Thursday index = 3
    weeks = calendar.monthcalendar(year, month)
    # Walk from last week upward
    for week in reversed(weeks):
        if week[calendar.THURSDAY] != 0:
            return date(year, month, week[calendar.THURSDAY])
    raise ValueError(f"No Thursday found in {year}-{month}")


def pick_expiry(today: date, min_days: int = 7) -> date:
    """
    Pick the current-month monthly expiry unless <min_days away, else next month.
    """
    curr_expiry = last_thursday_of_month(today.year, today.month)
    days_to_curr = (curr_expiry - today).days
    if days_to_curr >= min_days:
        return curr_expiry
    # Roll to next month
    ny = today.year + (1 if today.month == 12 else 0)
    nm = 1 if today.month == 12 else today.month + 1
    return last_thursday_of_month(ny, nm)


# =============================================================================
# Signal Wrapper
# =============================================================================

@dataclass
class CollarSignal:
    symbol: str
    signal_type: str       # 'ENTRY' or 'EXIT'
    spot: float
    date: str
    # Entry-specific
    kc6_mid: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    # Exit-specific
    position_id: Optional[int] = None
    exit_reason: Optional[str] = None


# =============================================================================
# Collar Engine
# =============================================================================

class CollarEngine:
    """Paper-mode engine for the collar overlay on KC6 signals."""

    def __init__(self, config: dict = None):
        self.config = config or COLLAR_DEFAULTS.copy()
        self.db = get_collar_db()

    # -------------------------------------------------------------------------
    # Universe: Nifty 500 ∩ F&O
    # -------------------------------------------------------------------------

    def _build_universe(self) -> List[str]:
        """Return symbols that are both Nifty 500 AND have an F&O lot size."""
        try:
            from services.nifty500_universe import get_nifty500
            from services.data_manager import FNO_LOT_SIZES
            n500 = set(get_nifty500().symbols)
            fno = set(FNO_LOT_SIZES.keys())
            uni = sorted(n500 & fno)
            logger.info(f"Collar universe: {len(uni)} symbols (Nifty500 ∩ F&O)")
            return uni
        except Exception as e:
            logger.error(f"Failed to build collar universe: {e}")
            return []

    def _get_lot_size(self, symbol: str) -> int:
        from services.data_manager import FNO_LOT_SIZES
        return FNO_LOT_SIZES.get(symbol, 1)

    # -------------------------------------------------------------------------
    # Data loading (delegates to kc6_scanner for indicator compute)
    # -------------------------------------------------------------------------

    def _load_symbol_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load daily OHLCV + compute KC6 indicators for the given symbols.
        Paper mode: always read from local market_data.db.
        """
        from services.kc6_scanner import load_daily_data_from_db
        return load_daily_data_from_db(symbols, str(MARKET_DATA_DB), min_bars=250)

    # -------------------------------------------------------------------------
    # Option pricing helpers
    # -------------------------------------------------------------------------

    def _price_leg(self, spot: float, strike: float, expiry: date,
                   today: date, is_call: bool) -> float:
        """Price one option leg with current config's IV + rate."""
        dte_days = max((expiry - today).days, 0)
        t_years = dte_days / 365.0
        iv = self.config.get('iv_assumed', 0.25)
        r = self.config.get('risk_free_rate', 0.065)
        return bs_price(spot, strike, t_years, r, iv, is_call)

    # -------------------------------------------------------------------------
    # Entry flow
    # -------------------------------------------------------------------------

    def _open_collar(self, symbol: str, spot: float, entry_date: date,
                     kc6_mid: float = None, sl_price: float = None,
                     tp_price: float = None) -> Optional[Dict]:
        """Create a new collar position (paper mode)."""
        # Duplicate guard
        if self.db.get_position_by_symbol(symbol):
            logger.info(f"[COLLAR] Skip {symbol}: already have open collar")
            return None

        # Max positions guard
        max_pos = self.config.get('max_positions', 5)
        if self.db.get_open_count() >= max_pos:
            logger.info(f"[COLLAR] Skip {symbol}: max_positions reached")
            return None

        lot_size = self._get_lot_size(symbol)
        if lot_size <= 0:
            logger.warning(f"[COLLAR] Skip {symbol}: no F&O lot size")
            return None

        # Strikes
        put_strike = round_put_strike(spot, self.config.get('put_otm_pct', 5.0))
        call_strike = round_call_strike(spot, self.config.get('call_otm_pct', 5.0))

        # Expiry
        expiry = pick_expiry(entry_date, self.config.get('min_expiry_days', 7))

        # Leg prices (paper BS)
        # Futures ~= spot (ignore basis in paper mode)
        future_price = round(spot, 2)
        put_price = self._price_leg(spot, put_strike, expiry, entry_date, is_call=False)
        call_price = self._price_leg(spot, call_strike, expiry, entry_date, is_call=True)

        pos_id = self.db.add_collar(
            symbol=symbol,
            entry_date=entry_date.isoformat(),
            spot_at_entry=spot,
            put_strike=put_strike,
            call_strike=call_strike,
            expiry_date=expiry.isoformat(),
            future_entry_price=future_price,
            put_entry_price=put_price,
            call_entry_price=call_price,
            lot_size=lot_size,
            sl_price=sl_price,
            tp_price=tp_price,
            kc6_mid=kc6_mid,
        )
        logger.info(
            f"[PAPER COLLAR] OPEN {symbol} spot={spot:.2f} "
            f"FUT@{future_price} +PUT{put_strike}@{put_price} "
            f"-CALL{call_strike}@{call_price} exp={expiry} qty={lot_size}"
        )
        return {
            'position_id': pos_id,
            'symbol': symbol,
            'spot': spot,
            'put_strike': put_strike,
            'call_strike': call_strike,
            'expiry': expiry.isoformat(),
            'future_price': future_price,
            'put_price': put_price,
            'call_price': call_price,
            'lot_size': lot_size,
        }

    # -------------------------------------------------------------------------
    # Exit flow
    # -------------------------------------------------------------------------

    def _check_exit(self, pos: Dict, today_row: pd.Series,
                    today: date) -> Optional[str]:
        """
        Decide whether a collar should be exited today. Returns exit_reason
        string or None.
        Priority: EXPIRY_NEAR > STOP_LOSS > TAKE_PROFIT > MAX_HOLD > SIGNAL_KC6_MID.
        """
        cfg = self.config
        spot = float(today_row['close'])
        high = float(today_row['high'])
        low = float(today_row['low'])

        entry_date_s = pos['entry_date']
        entry_dt = (datetime.strptime(entry_date_s, '%Y-%m-%d').date()
                    if isinstance(entry_date_s, str) else entry_date_s)
        hold_days = (today - entry_dt).days

        # Expiry protection
        expiry_s = pos['expiry_date']
        expiry_dt = (datetime.strptime(expiry_s, '%Y-%m-%d').date()
                     if isinstance(expiry_s, str) else expiry_s)
        dte = (expiry_dt - today).days
        if dte <= cfg.get('expiry_exit_days', 3):
            return 'EXPIRY_NEAR'

        # SL / TP on the UNDERLYING spot (mirror KC6)
        spot_entry = pos['spot_at_entry']
        sl_level = spot_entry * (1 - cfg.get('sl_pct', 5.0) / 100.0)
        tp_level = spot_entry * (1 + cfg.get('tp_pct', 15.0) / 100.0)

        if low <= sl_level:
            return 'STOP_LOSS'
        if high >= tp_level:
            return 'TAKE_PROFIT'
        if hold_days >= cfg.get('max_hold_days', 15):
            return 'MAX_HOLD'

        # KC6 mid signal on the underlying
        kc6_mid = today_row.get('kc6_mid')
        if kc6_mid is not None and pd.notna(kc6_mid) and high > float(kc6_mid):
            return 'SIGNAL_KC6_MID'

        return None

    def _close_collar(self, pos: Dict, today: date, spot_at_exit: float,
                      exit_reason: str) -> Dict:
        """Price the 3 legs today and close in DB."""
        expiry_s = pos['expiry_date']
        expiry_dt = (datetime.strptime(expiry_s, '%Y-%m-%d').date()
                     if isinstance(expiry_s, str) else expiry_s)

        future_exit = round(spot_at_exit, 2)
        put_exit = self._price_leg(spot_at_exit, pos['put_strike'], expiry_dt,
                                   today, is_call=False)
        call_exit = self._price_leg(spot_at_exit, pos['call_strike'], expiry_dt,
                                    today, is_call=True)

        result = self.db.close_collar(
            position_id=pos['id'],
            exit_date=today.isoformat(),
            future_exit_price=future_exit,
            put_exit_price=put_exit,
            call_exit_price=call_exit,
            exit_reason=exit_reason,
            spot_at_exit=spot_at_exit,
        )
        logger.info(
            f"[PAPER COLLAR] CLOSE {pos['symbol']} reason={exit_reason} "
            f"spot={spot_at_exit:.2f} net={result['net_pnl']:+.0f} "
            f"(fut={result['future_pnl']:+.0f} put={result['put_pnl']:+.0f} "
            f"call={result['call_pnl']:+.0f})"
        )
        return result

    # -------------------------------------------------------------------------
    # Scan: entries
    # -------------------------------------------------------------------------

    def scan_and_enter(self, symbol_data: Dict[str, pd.DataFrame] = None,
                       universe_atr_ratio: float = None) -> List[Dict]:
        """Use KC6 scanner to get entry signals, then open paper collars."""
        from services.kc6_scanner import (
            scan_entries, compute_universe_atr_ratio,
        )

        universe = self._build_universe()
        if not universe:
            logger.warning("[COLLAR] Empty universe — aborting entry scan")
            return []

        if symbol_data is None:
            symbol_data = self._load_symbol_data(universe)
        if not symbol_data:
            logger.warning("[COLLAR] No data loaded — aborting entry scan")
            return []

        if universe_atr_ratio is None:
            universe_atr_ratio = compute_universe_atr_ratio(symbol_data, KC6_DEFAULTS)

        # Skip symbols with open collars
        open_positions = self.db.get_open_positions()
        existing = [p['symbol'] for p in open_positions]

        kc6_cfg = dict(KC6_DEFAULTS)
        kc6_cfg['atr_ratio_threshold'] = self.config.get('atr_ratio_threshold', 1.3)
        signals = scan_entries(symbol_data, universe_atr_ratio, existing, kc6_cfg)

        today = datetime.now().date()
        opened = []
        for sig in signals:
            remaining = self.config.get('max_positions', 5) - self.db.get_open_count()
            if remaining <= 0:
                logger.info("[COLLAR] max_positions reached — stop opening")
                break
            res = self._open_collar(
                symbol=sig['symbol'],
                spot=sig['entry_price'],
                entry_date=today,
                kc6_mid=sig.get('kc6_mid'),
                sl_price=sig.get('sl_price'),
                tp_price=sig.get('tp_price'),
            )
            if res:
                opened.append(res)
        return opened

    # -------------------------------------------------------------------------
    # Scan: exits
    # -------------------------------------------------------------------------

    def scan_and_exit(self, symbol_data: Dict[str, pd.DataFrame] = None) -> List[Dict]:
        """Check exit conditions on every open collar and close as needed."""
        open_positions = self.db.get_open_positions()
        if not open_positions:
            return []

        symbols = [p['symbol'] for p in open_positions]
        if symbol_data is None:
            symbol_data = self._load_symbol_data(symbols)

        today = datetime.now().date()
        results = []

        for pos in open_positions:
            sym = pos['symbol']
            df = symbol_data.get(sym)
            if df is None or len(df) < 2:
                logger.warning(f"[COLLAR] No data for {sym}, skip exit check")
                continue
            row = df.iloc[-1]

            reason = self._check_exit(pos, row, today)
            if reason is None:
                continue
            try:
                r = self._close_collar(pos, today, float(row['close']), reason)
                r['position_id'] = pos['id']
                r['symbol'] = sym
                results.append(r)
            except Exception as e:
                logger.error(f"[COLLAR] Close failed for {sym}: {e}")

        return results

    # -------------------------------------------------------------------------
    # Full scan pipeline
    # -------------------------------------------------------------------------

    def run_full_scan(self) -> Dict:
        """
        One-shot orchestration:
          1. Build universe (Nifty500 ∩ F&O)
          2. Load daily data + compute indicators
          3. Compute universe ATR ratio (crash filter info)
          4. Exit scan on open positions (priority over entries)
          5. Entry scan (blocked by crash filter inside scan_entries)
          6. Save daily state snapshot
        """
        from services.kc6_scanner import (
            compute_universe_atr_ratio, is_crash_filter_active,
        )

        universe = self._build_universe()
        symbol_data = self._load_symbol_data(universe) if universe else {}

        if not symbol_data:
            logger.error("[COLLAR] No symbol data — scan aborted")
            return {
                'error': 'No data loaded',
                'entries_taken': [], 'exits_taken': [],
                'universe_atr_ratio': None,
            }

        universe_atr_ratio = compute_universe_atr_ratio(symbol_data, KC6_DEFAULTS)
        crash_cfg = dict(KC6_DEFAULTS)
        crash_cfg['atr_ratio_threshold'] = self.config.get('atr_ratio_threshold', 1.3)
        crash_active = is_crash_filter_active(universe_atr_ratio, crash_cfg)

        # Exits first (free up slots)
        exits_taken = self.scan_and_exit(symbol_data)
        # Entries (scan_and_enter respects crash filter internally)
        entries_taken = self.scan_and_enter(symbol_data, universe_atr_ratio)

        today = datetime.now()
        self.db.save_daily_state(
            trade_date=today.strftime('%Y-%m-%d'),
            universe_atr_ratio=round(universe_atr_ratio, 3),
            crash_filter_active=bool(crash_active),
            positions_count=self.db.get_open_count(),
            entry_signals=len(entries_taken),  # paper: entries_taken == signals executed
            exit_signals=len(exits_taken),
            entries_taken=len(entries_taken),
            exits_taken=len(exits_taken),
            daily_pnl=self.db.get_today_pnl(),
        )

        return {
            'universe_atr_ratio': round(universe_atr_ratio, 3),
            'crash_filter_active': bool(crash_active),
            'symbols_loaded': len(symbol_data),
            'open_positions': self.db.get_open_count(),
            'entries_taken': entries_taken,
            'exits_taken': exits_taken,
            'scan_time': today.isoformat(),
        }

    # -------------------------------------------------------------------------
    # Per-position MTM (used by dashboard)
    # -------------------------------------------------------------------------

    def mark_to_market(self, position: Dict,
                       spot_now: float = None) -> Dict:
        """
        Compute current mark-to-market P&L for each leg + aggregate.
        Uses latest close from market_data.db if spot_now not supplied.
        """
        today = datetime.now().date()
        sym = position['symbol']
        if spot_now is None:
            try:
                data = self._load_symbol_data([sym])
                df = data.get(sym)
                spot_now = float(df.iloc[-1]['close']) if df is not None and len(df) else None
            except Exception:
                spot_now = None

        if spot_now is None:
            # Fall back to entry spot (net 0 MTM)
            spot_now = position['spot_at_entry']

        expiry_dt = datetime.strptime(position['expiry_date'], '%Y-%m-%d').date()
        qty = position['qty']

        fut_now = round(spot_now, 2)
        put_now = self._price_leg(spot_now, position['put_strike'], expiry_dt,
                                  today, is_call=False)
        call_now = self._price_leg(spot_now, position['call_strike'], expiry_dt,
                                   today, is_call=True)

        fut_mtm = round((fut_now - position['future_entry_price']) * qty, 2)
        put_mtm = round((put_now - position['put_entry_price']) * qty, 2)
        call_mtm = round((position['call_entry_price'] - call_now) * qty, 2)
        net_mtm = round(fut_mtm + put_mtm + call_mtm, 2)

        return {
            'spot_now': round(spot_now, 2),
            'future_now': fut_now,
            'put_now': put_now,
            'call_now': call_now,
            'future_mtm': fut_mtm,
            'put_mtm': put_mtm,
            'call_mtm': call_mtm,
            'net_mtm': net_mtm,
        }

    # -------------------------------------------------------------------------
    # Live stub (deliberately blocked)
    # -------------------------------------------------------------------------

    def place_live(self, *args, **kwargs):
        """Live trading is not supported for the collar strategy."""
        raise NotImplementedError(
            "CollarEngine is paper-only. No live order routes exist."
        )
