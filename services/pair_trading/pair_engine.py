"""Pair-Trading (Config D) — Live + Paper engine.

PAPER MODE BY DEFAULT. No real Kite orders unless:
   PAIR_TRADING_DEFAULTS['paper_trading_mode']  == False  AND
   PAIR_TRADING_DEFAULTS['live_trading_enabled'] == True   AND
   PAIR_TRADING_DEFAULTS['enabled']              == True

Architecture (mirrors orb_live_engine.py):
  - `wrap_kite_op` decorator: paper mode no-ops; live mode calls Kite.
  - `_kite_place_order` / `_kite_cancel_order`: thin shims around Kite.
  - `place_pair_order_live_or_paper`: places BOTH legs atomically (defensive
    partial-fill rollback if leg-2 errors).
  - `square_off_pair`: places opposite-direction MARKET orders for both legs.
  - `kill_switch`: squares off all open pairs + flips enabled=False for the
    week (sets weekly circuit-breaker). Re-enable manually after review.
  - `compute_daily_pnl`: realised + unrealised across all open + closed-today.
  - `daily_scan`: the once-per-day EOD entry-point that the cron job calls.

Position sizing (per pair-trade):
  - Total risk Rs.6K (Rs.3K per leg)
  - Implied per-leg SL distance from spread_sd * (stop_z - entry_z) / 2
  - qty_a = floor(Rs.3K / (priceA * implied_sl_pct))
  - qty_b = floor(beta * qty_a) at equal-rupee weighting baseline
  - Both legs rounded UP to nearest lot multiple (so we always trade ≥1 lot)
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import date, datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from services.pair_trading.cohort import Cohort, PairConfig, load_cohort
from services.pair_trading.db import get_pair_trading_db
from services.pair_trading.signal import (
    PairRules,
    SignalResult,
    evaluate_today,
    implied_sl_distance_pct,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# wrap_kite_op decorator
# ===========================================================================

def wrap_kite_op(fn: Callable) -> Callable:
    """Wrap a method that calls Kite. In paper mode, the method's body
    is replaced by a logger.info(...) and a synthetic return value
    indicating no-op. In live mode, the method runs as-written.

    The wrapped method MUST be a bound method on PairEngine (so we can
    inspect self._is_paper()). The synthetic return value is determined
    per-method by inspecting fn.__name__:
       _kite_place_order  -> "PAPER-<uuid>"
       _kite_cancel_order -> None
       _kite_modify_order -> the original order_id (so callers see success)
       _kite_order_history-> [{'status': 'COMPLETE', '_paper': True}]
    Any other wrapped method returns None in paper mode.
    """
    @wraps(fn)
    def _wrapped(self, *args, **kwargs):
        if not self._is_paper():
            return fn(self, *args, **kwargs)
        # Paper-mode no-op
        name = fn.__name__
        if name == '_kite_place_order':
            fake_id = f"PAPER-{uuid.uuid4().hex[:12]}"
            logger.info(
                f"[PairTrading PAPER] {name} no-op: "
                f"{kwargs.get('transaction_type')} {kwargs.get('quantity')} "
                f"{kwargs.get('tradingsymbol')} {kwargs.get('order_type','MARKET')} "
                f"product={kwargs.get('product','NRML')} -> {fake_id}"
            )
            return fake_id
        if name == '_kite_cancel_order':
            logger.info(f"[PairTrading PAPER] {name} no-op: {kwargs.get('order_id')}")
            return None
        if name == '_kite_modify_order':
            order_id = kwargs.get('order_id')
            logger.info(f"[PairTrading PAPER] {name} no-op: {order_id} -> {kwargs}")
            return order_id
        if name == '_kite_order_history':
            order_id = kwargs.get('order_id') or (args[0] if args else None)
            return [{
                'order_id': order_id, 'status': 'COMPLETE',
                'average_price': None, 'filled_quantity': 0,
                '_paper': True,
            }]
        logger.info(f"[PairTrading PAPER] {name} no-op")
        return None
    return _wrapped


# ===========================================================================
# Engine
# ===========================================================================

class PairEngine:
    """Live + paper pair-trading engine for the 6-pair cohort."""

    def __init__(self, config: Dict, cohort: Optional[Cohort] = None):
        self.cfg = dict(config)
        self.cohort = cohort if cohort is not None else load_cohort(self.cfg)
        self.db = get_pair_trading_db()
        self._lock = threading.Lock()
        self._engine_started_at = datetime.now()

    # ------------------------------------------------------------------
    # Mode flags
    # ------------------------------------------------------------------

    def _is_enabled(self) -> bool:
        try:
            from config import PAIR_TRADING_DEFAULTS
            return bool(PAIR_TRADING_DEFAULTS.get('enabled', False))
        except Exception:
            return False

    def _is_paper(self) -> bool:
        try:
            from config import PAIR_TRADING_DEFAULTS
            return bool(PAIR_TRADING_DEFAULTS.get('paper_trading_mode', True))
        except Exception:
            return True  # fail-closed (paper)

    def _is_live(self) -> bool:
        try:
            from config import PAIR_TRADING_DEFAULTS
            return (
                bool(PAIR_TRADING_DEFAULTS.get('enabled', False))
                and not bool(PAIR_TRADING_DEFAULTS.get('paper_trading_mode', True))
                and bool(PAIR_TRADING_DEFAULTS.get('live_trading_enabled', False))
            )
        except Exception:
            return False

    def current_mode(self) -> str:
        """Return 'off' | 'paper' | 'live'."""
        if not self._is_enabled():
            return 'off'
        if self._is_live():
            return 'live'
        return 'paper'

    # ------------------------------------------------------------------
    # Kite shims
    # ------------------------------------------------------------------

    def _get_kite(self):
        from services.kite_service import get_kite
        return get_kite()

    @wrap_kite_op
    def _kite_place_order(self, **kwargs):
        kite = self._get_kite()
        return kite.place_order(**kwargs)

    @wrap_kite_op
    def _kite_cancel_order(self, *, variety: str, order_id: str, **kwargs):
        kite = self._get_kite()
        return kite.cancel_order(variety=variety, order_id=order_id, **kwargs)

    @wrap_kite_op
    def _kite_modify_order(self, *, variety: str, order_id: str, **kwargs):
        kite = self._get_kite()
        return kite.modify_order(variety=variety, order_id=order_id, **kwargs)

    @wrap_kite_op
    def _kite_order_history(self, order_id: str):
        kite = self._get_kite()
        return kite.order_history(order_id) or []

    # ------------------------------------------------------------------
    # Futures contract resolution
    # ------------------------------------------------------------------

    def _resolve_futures_symbol(self, underlying: str) -> Optional[Dict]:
        """Return {tradingsymbol, instrument_token, lot_size, expiry} for
        the current-month futures contract of `underlying`.

        Paper mode: synthesise a placeholder symbol so signals + DB rows
        still flow even when Kite is unauthenticated.
        """
        if self._is_paper():
            now = datetime.now()
            # Best-effort placeholder: <UNDERLYING><YY><MMM>FUT
            mmm = now.strftime('%b').upper()
            yy = now.strftime('%y')
            placeholder_ts = f"{underlying}{yy}{mmm}FUT"
            try:
                from services.data_manager import FNO_LOT_SIZES
                lot = int(FNO_LOT_SIZES.get(underlying, 0))
            except Exception:
                lot = 0
            return {
                'tradingsymbol': placeholder_ts,
                'instrument_token': None,
                'lot_size': lot,
                'expiry': None,
            }
        # Live mode: query Kite NFO instruments for the front-month FUT
        try:
            kite = self._get_kite()
            instruments = kite.instruments('NFO')
        except Exception as e:
            logger.error(f"[PairTrading] kite.instruments NFO failed: {e}")
            return None
        candidates = []
        for inst in instruments:
            if (inst.get('name') == underlying
                and inst.get('instrument_type') == 'FUT'
                and inst.get('segment') == 'NFO-FUT'):
                candidates.append(inst)
        if not candidates:
            logger.error(f"[PairTrading] no NFO FUT instruments for {underlying}")
            return None
        # Sort by expiry ascending; pick the front month
        candidates.sort(key=lambda x: x.get('expiry') or date.max)
        front = candidates[0]
        return {
            'tradingsymbol': front['tradingsymbol'],
            'instrument_token': front['instrument_token'],
            'lot_size': int(front.get('lot_size') or 0),
            'expiry': front.get('expiry'),
        }

    # ------------------------------------------------------------------
    # Daily price fetch
    # ------------------------------------------------------------------

    def fetch_daily_history(self, symbol: str, lookback_days: int) -> pd.Series:
        """Fetch a daily-close pd.Series for `symbol` covering at least
        `lookback_days` of trading days. Reads from market_data.db (the
        canonical local snapshot). If absent, falls back to Kite historical_data.
        """
        # Try local DB first (no Kite calls — works even when token expired)
        try:
            import sqlite3
            from pathlib import Path
            base = Path(__file__).parent.parent.parent / 'backtest_data' / 'market_data.db'
            if base.exists():
                end_dt = date.today()
                start_dt = end_dt - timedelta(days=int(lookback_days * 2.0) + 30)
                with sqlite3.connect(str(base)) as con:
                    df = pd.read_sql(
                        """SELECT date, close FROM market_data_unified
                            WHERE timeframe='day' AND symbol=? AND date>=? AND date<=?
                            ORDER BY date""",
                        con,
                        params=(symbol, start_dt.isoformat(), end_dt.isoformat()),
                    )
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    return df['close'].astype(float)
        except Exception as e:
            logger.warning(f"[PairTrading] local DB fetch failed for {symbol}: {e}")

        # Fallback to Kite historical_data (live mode only)
        if self._is_paper():
            logger.warning(
                f"[PairTrading PAPER] {symbol} not in local DB; cannot back-fetch in paper mode"
            )
            return pd.Series(dtype=float)
        try:
            kite = self._get_kite()
            instruments = kite.instruments('NSE')
            tok = None
            for inst in instruments:
                if inst.get('tradingsymbol') == symbol and inst.get('instrument_type') == 'EQ':
                    tok = inst.get('instrument_token')
                    break
            if tok is None:
                logger.error(f"[PairTrading] no NSE token for {symbol}")
                return pd.Series(dtype=float)
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=int(lookback_days * 2.0) + 30)
            data = kite.historical_data(tok, start_dt, end_dt, 'day')
            if not data:
                return pd.Series(dtype=float)
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df['close'].astype(float)
        except Exception as e:
            logger.error(f"[PairTrading] kite.historical_data failed for {symbol}: {e}")
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def size_pair(self, pair: PairConfig, signal: SignalResult) -> Tuple[int, int]:
        """Return (qty_a_shares, qty_b_shares) for both legs. Quantities
        are share counts (NOT lots) but rounded to whole multiples of the
        respective lot sizes so the F&O order is valid.

        Approach:
          1. Implied per-leg SL pct (from entry_z + stop_z + spread_sd).
          2. Per-leg risk Rs.3K -> qty_a = floor(3000 / (priceA * sl_pct/100)).
          3. qty_b = floor(beta * priceA / priceB * qty_a) (equal-rupee weight).
          4. Round qty_a UP to nearest lot multiple of lot_size_a.
             Round qty_b UP to nearest lot multiple of lot_size_b.
          5. If either lot_size is 0 (universe drift), return (0, 0) defensively.
        """
        if pair.lot_size_a == 0 or pair.lot_size_b == 0:
            logger.warning(f"[PairTrading] {pair.name}: missing lot size — refuse to size")
            return 0, 0

        risk_per_leg = float(self.cfg.get('risk_per_pair_rs', 6000)) / 2.0
        sl_pct = implied_sl_distance_pct(pair.entry_z, pair.stop_z,
                                          signal.spread_sd, pair.beta)
        if sl_pct <= 0 or signal.priceA <= 0 or signal.priceB <= 0:
            return 0, 0
        sl_distance_a = signal.priceA * (sl_pct / 100.0)
        sl_distance_b = signal.priceB * (sl_pct / 100.0)
        if sl_distance_a <= 0 or sl_distance_b <= 0:
            return 0, 0

        # Raw share counts (per-leg risk = qty * sl_distance)
        raw_qty_a = risk_per_leg / sl_distance_a
        # Equal-rupee weighting: notional_a == notional_b ->
        #   qty_b = qty_a * priceA / priceB (beta only enters the SL distance,
        #   not the notional weighting in the simple log-ratio convention used
        #   by research/39's simulate_pair).
        # We then bias by hedge ratio: qty_b adjusted by beta to track the
        # spread relationship more tightly.
        raw_qty_b = raw_qty_a * (signal.priceA / signal.priceB) * max(pair.beta, 0.1)

        # Round UP to nearest lot multiple (always trade >=1 lot)
        lots_a = max(1, int(np.ceil(raw_qty_a / pair.lot_size_a)))
        lots_b = max(1, int(np.ceil(raw_qty_b / pair.lot_size_b)))
        qty_a = lots_a * pair.lot_size_a
        qty_b = lots_b * pair.lot_size_b
        return qty_a, qty_b

    # ------------------------------------------------------------------
    # Order placement (atomic 2-leg)
    # ------------------------------------------------------------------

    def place_pair_order_live_or_paper(
        self,
        pair: PairConfig,
        direction: int,                # +1 long_spread, -1 short_spread
        signal: SignalResult,
        qty_a: int, qty_b: int,
    ) -> Optional[int]:
        """Place BOTH legs atomically. Returns position_id on success, None on failure.

        Direction +1 (long_spread): BUY A, SELL B
        Direction -1 (short_spread): SELL A, BUY B

        Defensive partial-fill handling: if leg-2 fails (e.g., margin reject
        in live mode), we attempt to cancel + reverse leg-1 to flat out the
        single open leg. In paper mode, both legs always succeed.
        """
        if not self._is_enabled():
            logger.info(f"[PairTrading] {pair.name}: engine disabled, skipping order")
            return None
        if direction not in (1, -1):
            logger.error(f"[PairTrading] invalid direction {direction}")
            return None

        # Block live orders unless live_trading_enabled is also True
        if not self._is_paper() and not self._is_live():
            logger.error(
                f"[PairTrading] {pair.name}: paper-mode flag flipped off but "
                f"live_trading_enabled is False — REFUSING to place order"
            )
            return None

        # Resolve futures contracts for both legs
        fut_a = self._resolve_futures_symbol(pair.symA)
        fut_b = self._resolve_futures_symbol(pair.symB)
        if fut_a is None or fut_b is None:
            logger.error(f"[PairTrading] {pair.name}: futures resolution failed")
            return None

        txn_a = 'BUY' if direction == 1 else 'SELL'
        txn_b = 'SELL' if direction == 1 else 'BUY'
        exch = self.cfg.get('exchange', 'NFO')
        product = self.cfg.get('product', 'NRML')
        order_type = self.cfg.get('order_type', 'MARKET')

        paper_flag = 1 if self._is_paper() else 0

        # Insert position row first (so we have an ID to thread through orders)
        position_id = self.db.add_position(
            pair_name=pair.name,
            symA=pair.symA, symB=pair.symB,
            direction=direction,
            entry_date=date.today().isoformat(),
            entry_z=signal.z,
            target_z=0.0,
            stop_z=pair.stop_z,
            hold_cap_days=pair.hold_days,
            lookback=pair.lookback,
            alpha=pair.alpha, beta=pair.beta,
            legA_tradingsymbol=fut_a['tradingsymbol'],
            legA_qty=qty_a,
            legA_entry_price=signal.priceA,
            legA_lot_size=pair.lot_size_a,
            legB_tradingsymbol=fut_b['tradingsymbol'],
            legB_qty=qty_b,
            legB_entry_price=signal.priceB,
            legB_lot_size=pair.lot_size_b,
            paper_mode=paper_flag,
            status='OPEN',
            notes=f"entry_action={'ENTRY_LONG' if direction == 1 else 'ENTRY_SHORT'}",
        )

        # Place leg A
        try:
            order_id_a = self._kite_place_order(
                variety='regular',
                exchange=exch,
                tradingsymbol=fut_a['tradingsymbol'],
                transaction_type=txn_a,
                quantity=qty_a,
                product=product,
                order_type=order_type,
            )
        except Exception as e:
            logger.exception(f"[PairTrading] {pair.name} leg-A place_order failed: {e}")
            self.db.update_position(position_id, status='FAILED',
                                    notes=f"leg-A failed: {e}")
            return None

        self.db.add_order(
            position_id=position_id, pair_name=pair.name,
            leg='A', tradingsymbol=fut_a['tradingsymbol'],
            transaction_type=txn_a, qty=qty_a,
            order_type=order_type, kite_order_id=str(order_id_a),
            status='PLACED', exchange=exch, product=product,
            paper_mode=paper_flag, leg_role='ENTRY',
        )
        self.db.update_position(position_id, legA_kite_order_id=str(order_id_a))

        # Place leg B
        try:
            order_id_b = self._kite_place_order(
                variety='regular',
                exchange=exch,
                tradingsymbol=fut_b['tradingsymbol'],
                transaction_type=txn_b,
                quantity=qty_b,
                product=product,
                order_type=order_type,
            )
        except Exception as e:
            logger.exception(
                f"[PairTrading] {pair.name} leg-B place_order failed: {e} — "
                f"reversing leg-A to flat"
            )
            # Try to reverse leg-A so we don't end up naked on a single leg
            try:
                rev_a = 'SELL' if txn_a == 'BUY' else 'BUY'
                rev_id = self._kite_place_order(
                    variety='regular',
                    exchange=exch,
                    tradingsymbol=fut_a['tradingsymbol'],
                    transaction_type=rev_a,
                    quantity=qty_a,
                    product=product,
                    order_type='MARKET',
                )
                self.db.add_order(
                    position_id=position_id, pair_name=pair.name,
                    leg='A', tradingsymbol=fut_a['tradingsymbol'],
                    transaction_type=rev_a, qty=qty_a,
                    order_type='MARKET', kite_order_id=str(rev_id),
                    status='REVERSAL', exchange=exch, product=product,
                    paper_mode=paper_flag, leg_role='EXIT',
                    notes='leg-B failure rollback',
                )
            except Exception as rev_err:
                logger.exception(
                    f"[PairTrading] CRITICAL: leg-A reversal also failed: {rev_err} — "
                    f"position {position_id} is naked single-leg, manual intervention required"
                )
            self.db.update_position(position_id, status='FAILED',
                                    notes=f"leg-B failed: {e}; leg-A reversal attempted")
            return None

        self.db.add_order(
            position_id=position_id, pair_name=pair.name,
            leg='B', tradingsymbol=fut_b['tradingsymbol'],
            transaction_type=txn_b, qty=qty_b,
            order_type=order_type, kite_order_id=str(order_id_b),
            status='PLACED', exchange=exch, product=product,
            paper_mode=paper_flag, leg_role='ENTRY',
        )
        self.db.update_position(position_id, legB_kite_order_id=str(order_id_b))

        logger.info(
            f"[PairTrading {'PAPER' if self._is_paper() else 'LIVE'}] "
            f"{pair.name} OPENED dir={direction:+d} z={signal.z:.2f} "
            f"A={txn_a} {qty_a}@{signal.priceA:.2f}, B={txn_b} {qty_b}@{signal.priceB:.2f} "
            f"-> position {position_id}"
        )
        return position_id

    # ------------------------------------------------------------------
    # Square-off
    # ------------------------------------------------------------------

    def square_off_pair(self, position_id: int, reason: str = 'EXIT_MR',
                         current_price_a: Optional[float] = None,
                         current_price_b: Optional[float] = None,
                         current_z: Optional[float] = None) -> bool:
        """Place opposite-direction MARKET orders for both legs of an open
        pair-position. Computes net P&L using current prices (or last-known
        if not supplied). Closes the DB position + writes a trade row."""
        pos = self.db.get_position_by_id(position_id)
        if not pos or pos.get('status') != 'OPEN':
            logger.warning(f"[PairTrading] square_off: position {position_id} not OPEN")
            return False

        direction = int(pos['direction'])
        symA, symB = pos['symA'], pos['symB']
        qty_a, qty_b = int(pos['legA_qty']), int(pos['legB_qty'])
        ts_a = pos['legA_tradingsymbol']
        ts_b = pos['legB_tradingsymbol']
        entry_price_a = float(pos['legA_entry_price'])
        entry_price_b = float(pos['legB_entry_price'])

        # Reverse direction
        rev_txn_a = 'SELL' if direction == 1 else 'BUY'
        rev_txn_b = 'BUY' if direction == 1 else 'SELL'

        exch = self.cfg.get('exchange', 'NFO')
        product = self.cfg.get('product', 'NRML')
        paper_flag = 1 if self._is_paper() else 0

        # Place reversal orders (best effort — log + write trade even on partial)
        try:
            order_id_a = self._kite_place_order(
                variety='regular', exchange=exch,
                tradingsymbol=ts_a, transaction_type=rev_txn_a,
                quantity=qty_a, product=product, order_type='MARKET',
            )
        except Exception as e:
            logger.exception(f"[PairTrading] {pos['pair_name']} square-off leg-A failed: {e}")
            order_id_a = None

        try:
            order_id_b = self._kite_place_order(
                variety='regular', exchange=exch,
                tradingsymbol=ts_b, transaction_type=rev_txn_b,
                quantity=qty_b, product=product, order_type='MARKET',
            )
        except Exception as e:
            logger.exception(f"[PairTrading] {pos['pair_name']} square-off leg-B failed: {e}")
            order_id_b = None

        # Audit both reversal orders
        if order_id_a is not None:
            self.db.add_order(
                position_id=position_id, pair_name=pos['pair_name'],
                leg='A', tradingsymbol=ts_a,
                transaction_type=rev_txn_a, qty=qty_a,
                order_type='MARKET', kite_order_id=str(order_id_a),
                status='PLACED', exchange=exch, product=product,
                paper_mode=paper_flag, leg_role='EXIT',
                notes=f'square-off reason={reason}',
            )
        if order_id_b is not None:
            self.db.add_order(
                position_id=position_id, pair_name=pos['pair_name'],
                leg='B', tradingsymbol=ts_b,
                transaction_type=rev_txn_b, qty=qty_b,
                order_type='MARKET', kite_order_id=str(order_id_b),
                status='PLACED', exchange=exch, product=product,
                paper_mode=paper_flag, leg_role='EXIT',
                notes=f'square-off reason={reason}',
            )

        # Compute P&L (paper mode uses caller-supplied prices; live mode would
        # ideally pull from order_history average_price, but for this paper
        # adapter we accept either)
        ex_a = float(current_price_a) if current_price_a is not None else entry_price_a
        ex_b = float(current_price_b) if current_price_b is not None else entry_price_b

        if direction == 1:
            # Long A, short B -> P&L = (ex_a - entry_a) * qty_a + (entry_b - ex_b) * qty_b
            pnl_a = (ex_a - entry_price_a) * qty_a
            pnl_b = (entry_price_b - ex_b) * qty_b
        else:
            # Short A, long B -> P&L = (entry_a - ex_a) * qty_a + (ex_b - entry_b) * qty_b
            pnl_a = (entry_price_a - ex_a) * qty_a
            pnl_b = (ex_b - entry_price_b) * qty_b
        gross_pnl = pnl_a + pnl_b

        cost_per_side_pct = float(self.cfg.get('cost_per_side_pct', 0.03)) / 100.0
        notional_a = ex_a * qty_a + entry_price_a * qty_a
        notional_b = ex_b * qty_b + entry_price_b * qty_b
        cost_inr = (notional_a + notional_b) * cost_per_side_pct
        net_pnl = gross_pnl - cost_inr

        # Days held
        try:
            entry_d = pd.Timestamp(pos['entry_date']).date()
        except Exception:
            entry_d = date.today()
        days_held = max((date.today() - entry_d).days, 0)

        self.db.close_position(
            position_id=position_id,
            exit_date=date.today(),
            exit_z=float(current_z) if current_z is not None else 0.0,
            legA_exit_price=ex_a, legB_exit_price=ex_b,
            exit_reason=reason,
            gross_pnl_inr=round(gross_pnl, 2),
            cost_inr=round(cost_inr, 2),
            net_pnl_inr=round(net_pnl, 2),
            days_held=days_held,
            legA_kite_exit_order_id=str(order_id_a) if order_id_a else None,
            legB_kite_exit_order_id=str(order_id_b) if order_id_b else None,
        )
        self.db.add_trade(
            position_id=position_id,
            pair_name=pos['pair_name'],
            direction=direction,
            entry_date=pos['entry_date'],
            exit_date=date.today().isoformat(),
            days_held=days_held,
            entry_z=float(pos['entry_z']),
            exit_z=float(current_z) if current_z is not None else 0.0,
            gross_pnl_inr=round(gross_pnl, 2),
            cost_inr=round(cost_inr, 2),
            net_pnl_inr=round(net_pnl, 2),
            exit_reason=reason,
            paper_mode=paper_flag,
        )

        logger.info(
            f"[PairTrading {'PAPER' if self._is_paper() else 'LIVE'}] "
            f"{pos['pair_name']} CLOSED reason={reason} "
            f"gross={gross_pnl:+.0f} cost={cost_inr:.0f} net={net_pnl:+.0f} "
            f"days={days_held}"
        )
        return True

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def kill_switch(self) -> Dict[str, Any]:
        """Square off ALL open pairs at current prices, set weekly
        circuit-breaker, flip enabled=False until manual re-enable."""
        opens = self.db.get_open_positions()
        squared = 0
        for pos in opens:
            try:
                ok = self.square_off_pair(
                    position_id=pos['id'],
                    reason='KILL_SWITCH',
                )
                if ok:
                    squared += 1
            except Exception as e:
                logger.exception(f"[PairTrading] kill-switch square_off failed for {pos['id']}: {e}")
        # Flip enabled off + set weekly CB on today's daily state
        try:
            from config import PAIR_TRADING_DEFAULTS
            PAIR_TRADING_DEFAULTS['enabled'] = False
        except Exception:
            pass
        try:
            self.db.update_daily_state(
                date.today(), circuit_breaker_active=1,
                notes='kill-switch fired',
            )
        except Exception:
            pass
        logger.warning(f"[PairTrading] KILL SWITCH: squared {squared} pairs, engine disabled")
        return {'squared': squared, 'enabled': False, 'circuit_breaker_active': True}

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------

    def compute_daily_pnl(self) -> Dict[str, float]:
        """Return {realized, unrealized} for today. Realized = closed trades
        today; unrealized = open positions marked-to-market at last close."""
        today = date.today().isoformat()
        realized = 0.0
        for tr in self.db.list_trades(date_from=today, date_to=today, limit=100):
            realized += float(tr.get('net_pnl_inr') or 0)

        unrealized = 0.0
        opens = self.db.get_open_positions()
        for pos in opens:
            # MTM uses last-known close; if local DB has fresher prices, use them
            try:
                px_a = self.fetch_daily_history(pos['symA'], lookback_days=5)
                px_b = self.fetch_daily_history(pos['symB'], lookback_days=5)
                if px_a.empty or px_b.empty:
                    continue
                last_a = float(px_a.iloc[-1])
                last_b = float(px_b.iloc[-1])
            except Exception:
                continue
            qty_a = int(pos['legA_qty'] or 0)
            qty_b = int(pos['legB_qty'] or 0)
            entry_a = float(pos['legA_entry_price'] or 0)
            entry_b = float(pos['legB_entry_price'] or 0)
            d = int(pos['direction'])
            if d == 1:
                pnl = (last_a - entry_a) * qty_a + (entry_b - last_b) * qty_b
            else:
                pnl = (entry_a - last_a) * qty_a + (last_b - entry_b) * qty_b
            unrealized += pnl
        return {
            'realized': round(realized, 2),
            'unrealized': round(unrealized, 2),
        }

    # ------------------------------------------------------------------
    # The daily scan (the cron entry-point)
    # ------------------------------------------------------------------

    def daily_scan(self) -> Dict[str, Any]:
        """Run the once-per-day EOD evaluation across all 6 pairs.
        Cron schedules this at 16:00 IST after F&O close.
        Returns a summary dict."""
        if not self._is_enabled():
            logger.info("[PairTrading] daily_scan: engine disabled, skipping")
            return {'mode': 'off', 'evaluated': 0, 'entered': 0, 'exited': 0}

        mode = self.current_mode()
        scan_time = datetime.now()
        today = date.today()
        lookback_days = int(self.cfg.get('history_lookback_days', 90))

        # Ensure today's daily state row
        self.db.get_or_create_daily_state(today)
        self.db.update_daily_state(today, mode=mode, scan_completed=0,
                                    capital=float(self.cfg.get('capital', 1_000_000)))

        # Pre-fetch daily histories for all symbols
        all_syms = self.cohort.all_symbols()
        prices: Dict[str, pd.Series] = {}
        for sym in all_syms:
            ser = self.fetch_daily_history(sym, lookback_days)
            if ser is None or ser.empty:
                logger.warning(f"[PairTrading] {sym}: no daily history available")
                continue
            prices[sym] = ser
        logger.info(
            f"[PairTrading] daily_scan: mode={mode}, fetched daily history for "
            f"{len(prices)}/{len(all_syms)} symbols"
        )

        # Evaluate each pair
        opens = self.db.get_open_positions()
        opens_by_pair = {p['pair_name']: p for p in opens}

        # Concurrency cap: count currently open
        max_concurrent = int(self.cfg.get('max_concurrent', 5))
        n_open = len(opens)

        entered = 0
        exited = 0
        evaluated = 0

        for pair in self.cohort.pairs:
            evaluated += 1
            sa = prices.get(pair.symA)
            sb = prices.get(pair.symB)
            if sa is None or sb is None or sa.empty or sb.empty:
                self.db.log_signal(
                    scan_time=scan_time.isoformat(), trade_date=today.isoformat(),
                    pair_name=pair.name, symA=pair.symA, symB=pair.symB,
                    priceA=None, priceB=None,
                    spread=None, spread_mu=None, spread_sd=None,
                    z=None, action='BLOCKED',
                    block_reason='no_history',
                )
                continue

            existing = opens_by_pair.get(pair.name)
            open_pos_dict = None
            if existing:
                try:
                    entry_d = pd.Timestamp(existing['entry_date']).date()
                except Exception:
                    entry_d = today
                days_held = (today - entry_d).days
                open_pos_dict = {
                    'direction': int(existing['direction']),
                    'days_held': days_held,
                }

            rules = PairRules(
                entry_z=pair.entry_z, stop_z=pair.stop_z,
                hold_days=pair.hold_days, lookback=pair.lookback,
            )
            sig = evaluate_today(pair.name, sa, sb, pair.alpha, pair.beta,
                                  rules, open_position=open_pos_dict)
            if sig is None:
                continue

            # Always log the signal evaluation
            self.db.log_signal(
                scan_time=scan_time.isoformat(),
                trade_date=today.isoformat(),
                pair_name=pair.name, symA=pair.symA, symB=pair.symB,
                priceA=sig.priceA if np.isfinite(sig.priceA) else None,
                priceB=sig.priceB if np.isfinite(sig.priceB) else None,
                spread=sig.spread if np.isfinite(sig.spread) else None,
                spread_mu=sig.spread_mu if np.isfinite(sig.spread_mu) else None,
                spread_sd=sig.spread_sd if np.isfinite(sig.spread_sd) else None,
                z=sig.z if np.isfinite(sig.z) else None,
                action=sig.action,
                block_reason=sig.block_reason,
            )

            # Act on signal
            if sig.action in ('ENTRY_LONG', 'ENTRY_SHORT'):
                if existing is not None:
                    # Already open — entries while open are no-ops
                    continue
                if n_open >= max_concurrent:
                    self.db.log_signal(
                        scan_time=scan_time.isoformat(),
                        trade_date=today.isoformat(),
                        pair_name=pair.name, symA=pair.symA, symB=pair.symB,
                        priceA=sig.priceA, priceB=sig.priceB,
                        spread=sig.spread, spread_mu=sig.spread_mu, spread_sd=sig.spread_sd,
                        z=sig.z, action='BLOCKED',
                        block_reason=f'concurrency_cap_{max_concurrent}',
                    )
                    continue
                direction = 1 if sig.action == 'ENTRY_LONG' else -1
                qty_a, qty_b = self.size_pair(pair, sig)
                if qty_a == 0 or qty_b == 0:
                    self.db.log_signal(
                        scan_time=scan_time.isoformat(),
                        trade_date=today.isoformat(),
                        pair_name=pair.name, symA=pair.symA, symB=pair.symB,
                        priceA=sig.priceA, priceB=sig.priceB,
                        spread=sig.spread, spread_mu=sig.spread_mu, spread_sd=sig.spread_sd,
                        z=sig.z, action='BLOCKED',
                        block_reason='zero_qty_after_sizing',
                    )
                    continue
                pid = self.place_pair_order_live_or_paper(
                    pair=pair, direction=direction, signal=sig,
                    qty_a=qty_a, qty_b=qty_b,
                )
                if pid:
                    entered += 1
                    n_open += 1
            elif sig.action in ('EXIT_MR', 'EXIT_STOP', 'EXIT_TIME'):
                if existing is not None:
                    ok = self.square_off_pair(
                        position_id=existing['id'],
                        reason=sig.action,
                        current_price_a=sig.priceA,
                        current_price_b=sig.priceB,
                        current_z=sig.z,
                    )
                    if ok:
                        exited += 1
                        n_open -= 1
            # HOLD or NO_ACTION -> nothing else

        # Update daily state with summary + equity curve point
        pnl = self.compute_daily_pnl()
        capital = float(self.cfg.get('capital', 1_000_000))
        # NAV approximation: capital + cumulative net_pnl across all closed trades
        try:
            all_trades = self.db.list_trades(limit=10000)
            cum_pnl = sum(float(t.get('net_pnl_inr') or 0) for t in all_trades)
        except Exception:
            cum_pnl = 0.0
        nav = capital + cum_pnl + pnl['unrealized']

        self.db.update_daily_state(
            today,
            realized_pnl=pnl['realized'],
            unrealized_pnl=pnl['unrealized'],
            open_pairs=n_open,
            scan_completed=1,
        )
        self.db.upsert_equity_curve_point(
            trade_date=today, nav=round(nav, 2),
            realized_pnl=pnl['realized'], unrealized_pnl=pnl['unrealized'],
            open_pairs=n_open,
        )

        summary = {
            'mode': mode,
            'evaluated': evaluated,
            'entered': entered,
            'exited': exited,
            'open_pairs': n_open,
            'realized_pnl': pnl['realized'],
            'unrealized_pnl': pnl['unrealized'],
            'nav': round(nav, 2),
        }
        logger.info(f"[PairTrading] daily_scan summary: {summary}")
        return summary


# ===========================================================================
# Singleton accessor (so the Flask routes + cron job share state)
# ===========================================================================

_engine_instance: Optional[PairEngine] = None
_engine_lock = threading.Lock()


def get_pair_engine() -> PairEngine:
    """Return a process-wide PairEngine instance bound to PAIR_TRADING_DEFAULTS."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                from config import PAIR_TRADING_DEFAULTS
                _engine_instance = PairEngine(dict(PAIR_TRADING_DEFAULTS))
    return _engine_instance
