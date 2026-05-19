"""Intraday 75% WR — Engine Base
==================================

Shared base class for the 3 live configs (A, B, C). Implements:

- 3-state Off/Paper/Live mode reads (mirrors ORB pattern)
- wrap_kite_op() decorator: paper-mode no-ops with DB logging
- place_order_live_or_paper() for entries + exits
- square_off_all() and kill_switch()
- compute_daily_pnl() across config's positions
- concurrency_check() — checks across ALL configs (A+B+C) combined

Each config (A/B/C) subclasses this and provides config-specific scan() and
signal-firing logic. Cron jobs in app.py call those scan() methods.

PAPER MODE LOCK semantics:
- Live trading requires BOTH paper_trading_mode=False AND
  live_trading_enabled=True. Belt-and-suspenders.
- In paper mode, every Kite call returns a synthetic PAPER-{uuid} response
  and the order/position lifecycle is logged to the DB so the React UI shows
  exactly what would have happened.
"""

from __future__ import annotations

import logging
import math
import threading
import uuid as _uuid
from datetime import datetime, date, timedelta
from typing import Any, Optional

from services.intraday_75wr.db import get_i75_db

logger = logging.getLogger(__name__)


def _is_paper_mode(cfg: dict) -> bool:
    """True if config is in paper mode OR mode read fails (fail-closed)."""
    try:
        return bool(cfg.get('paper_trading_mode', True))
    except Exception:
        return True


def _is_live_authorized(cfg: dict) -> bool:
    """True only if BOTH flags say live: paper_trading_mode=False AND
    live_trading_enabled=True. Either flag flipped to safe blocks live."""
    try:
        if cfg.get('paper_trading_mode', True):
            return False
        if not cfg.get('live_trading_enabled', False):
            return False
        if not cfg.get('enabled', True):
            return False
        return True
    except Exception:
        return False


class IntradayEngineBase:
    """Base for the 3 intraday-75WR configs (A/B/C).

    Subclasses provide scan() per sub-signal. This base only owns the
    cross-cutting concerns: order placement, paper/live gating, sizing,
    P&L compute, kill-switch and concurrency.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.config_id = cfg['config_id']         # 'A' / 'B' / 'C'
        self.db = get_i75_db()
        self._lock = threading.Lock()
        # Per-config in-memory open-position cache, keyed by (system_id, instrument)
        # so A1, A2, A3 can each independently hold a position on the same stock
        # if signals collide. config_id alone isn't unique enough for the
        # 3-system A/B configs.
        self._positions: dict[tuple[str, str], dict] = {}
        self._kill_switched_today = False
        self._kill_switch_date: Optional[str] = None
        # Per-day daily-loss flag
        self._daily_loss_halt = False
        self._engine_started_at = datetime.now()

    # =========================================================================
    # Mode helpers
    # =========================================================================

    def is_enabled(self) -> bool:
        return bool(self.cfg.get('enabled', False))

    def is_paper(self) -> bool:
        return _is_paper_mode(self.cfg)

    def is_live(self) -> bool:
        return _is_live_authorized(self.cfg)

    # =========================================================================
    # Kite wrappers (paper-mode safe)
    # =========================================================================

    def _get_kite(self):
        """Return Kite client; deferred import to avoid circular."""
        from services.kite_service import get_kite
        return get_kite()

    def wrap_kite_op(self, op_name: str, op_callable, *args, **kwargs):
        """Run a Kite op only when live is authorised. In paper mode the
        callable is NOT invoked — returns a synthetic paper marker.

        Use this for any Kite call that has side effects (place_order,
        cancel_order, modify_order). Read-only ops (LTP, instruments) can
        run regardless because they don't move money."""
        if not self.is_live():
            paper_id = f'PAPER-{_uuid.uuid4().hex[:12]}'
            logger.info(
                f'[I75-{self.config_id} PAPER] would call {op_name} '
                f'args={args} kwargs={kwargs} -> {paper_id}'
            )
            return {'paper': True, 'order_id': paper_id, 'op': op_name}
        try:
            return op_callable(*args, **kwargs)
        except Exception as e:
            logger.error(
                f'[I75-{self.config_id}] {op_name} FAILED: {e}', exc_info=True,
            )
            raise

    def get_live_ltp(self, instruments: list[str]) -> dict[str, float]:
        """Read-only LTP fetch. Always allowed (paper or live)."""
        try:
            kite = self._get_kite()
            keys = [f'NSE:{sym}' for sym in instruments]
            quotes = kite.ltp(keys) or {}
            out: dict[str, float] = {}
            for sym in instruments:
                k = f'NSE:{sym}'
                d = quotes.get(k)
                if d:
                    out[sym] = float(d['last_price'])
            return out
        except Exception as e:
            logger.warning(
                f'[I75-{self.config_id}] LTP fetch failed: {e}', exc_info=False,
            )
            return {}

    # =========================================================================
    # Sizing
    # =========================================================================

    def compute_qty(self, entry_price: float, sl_price: float) -> dict:
        """Position sizing per spec:
            qty = floor(risk_per_trade_rs / |entry - SL|)
            cap notional at (capital * mis_leverage / max_concurrent)
            qty = min(qty, floor(max_notional / entry_price))
        Returns dict with qty + diagnostics."""
        if entry_price is None or entry_price <= 0:
            return {'qty': 0, 'reason': 'invalid_entry_price'}
        R = abs(float(entry_price) - float(sl_price))
        if R <= 0:
            return {'qty': 0, 'reason': 'zero_R'}

        risk_rs = float(self.cfg.get('risk_per_trade_rs', 3000))
        raw_qty = int(math.floor(risk_rs / R))

        capital = float(self.cfg.get('capital', 300_000))
        lev = float(self.cfg.get('mis_leverage', 5))
        max_conc = max(1, int(self.cfg.get('max_concurrent', 5)))
        max_notional_alloc = (capital * lev) / max_conc
        max_notional_cfg = float(self.cfg.get('max_notional_per_trade', capital))
        max_notional = min(max_notional_alloc, max_notional_cfg)
        cap_qty = int(math.floor(max_notional / entry_price))
        final_qty = max(min(raw_qty, cap_qty), 0)

        return {
            'qty': final_qty,
            'R_per_share': round(R, 4),
            'risk_rs_target': risk_rs,
            'risk_rs_actual': round(final_qty * R, 2),
            'notional': round(final_qty * entry_price, 2),
            'max_notional': round(max_notional, 2),
            'raw_qty_uncapped': raw_qty,
            'capped': raw_qty > cap_qty,
        }

    # =========================================================================
    # Concurrency — ACROSS A+B+C combined
    # =========================================================================

    def total_open_across_configs(self) -> int:
        """COUNT of open positions across A+B+C. Hard cap is configurable
        but defaults to 5 (per FINAL_LIVE_SETUP.md)."""
        try:
            return self.db.count_open_positions_all_systems()
        except Exception as e:
            logger.error(
                f'[I75-{self.config_id}] concurrency count failed: {e}',
                exc_info=True,
            )
            return 0

    def concurrency_check(self) -> dict:
        """Return {'allowed': bool, 'open_count': int, 'cap': int}.
        Allowed=False blocks new entries this tick."""
        try:
            from config import INTRADAY_75WR_COMBINED_MAX_CONCURRENT as CAP
        except Exception:
            CAP = 5
        open_n = self.total_open_across_configs()
        return {
            'allowed': open_n < CAP,
            'open_count': open_n,
            'cap': CAP,
        }

    # =========================================================================
    # Daily P&L + circuit breakers
    # =========================================================================

    def compute_daily_pnl(self) -> dict:
        """Compute realized + unrealized P&L for THIS config today."""
        cid = self.config_id
        # NOTE: in this DB, system_id stores a sub-signal id like 'A1', 'A2',
        # 'A3', 'C'. We aggregate across all sub-systems belonging to this
        # config by prefix-match.
        prefix = cid

        realized = 0.0
        wins = losses = 0
        try:
            today = date.today().isoformat()
            with self.db.db_lock:
                conn = self.db._get_conn()
                try:
                    rows = conn.execute(
                        'SELECT pnl_inr FROM i75_positions '
                        'WHERE status="CLOSED" AND trade_date=? '
                        'AND system_id LIKE ?',
                        (today, f'{prefix}%'),
                    ).fetchall()
                    for r in rows:
                        v = float(r['pnl_inr'] or 0)
                        realized += v
                        if v > 0:
                            wins += 1
                        elif v < 0:
                            losses += 1
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(
                f'[I75-{cid}] daily realized failed: {e}', exc_info=False,
            )

        # Unrealized
        unrealized = 0.0
        try:
            open_pos = []
            with self.db.db_lock:
                conn = self.db._get_conn()
                try:
                    rows = conn.execute(
                        'SELECT * FROM i75_positions '
                        'WHERE status="OPEN" AND system_id LIKE ?',
                        (f'{prefix}%',),
                    ).fetchall()
                    open_pos = [dict(r) for r in rows]
                finally:
                    conn.close()
            if open_pos:
                ltps = self.get_live_ltp([p['instrument'] for p in open_pos])
                for p in open_pos:
                    ltp = ltps.get(p['instrument'])
                    if ltp is None:
                        continue
                    pts = (
                        (ltp - p['entry_price'])
                        if p['direction'] == 'LONG'
                        else (p['entry_price'] - ltp)
                    )
                    unrealized += pts * float(p['qty'])
        except Exception as e:
            logger.warning(
                f'[I75-{cid}] daily unrealized failed: {e}', exc_info=False,
            )

        return {
            'realized': round(realized, 2),
            'unrealized': round(unrealized, 2),
            'total': round(realized + unrealized, 2),
            'wins': wins,
            'losses': losses,
        }

    def daily_loss_gate(self) -> dict:
        """Return whether new entries should be blocked given daily loss cap."""
        cap_rs = float(self.cfg.get(
            'daily_loss_limit_rs',
            self.cfg.get('capital', 300_000) * self.cfg.get('daily_loss_limit_pct', 0.03),
        ))
        enforce = bool(self.cfg.get('enforce_daily_loss_cap', True))
        pnl = self.compute_daily_pnl()
        realized_loss = max(0.0, -pnl['realized'])
        block = enforce and realized_loss >= cap_rs
        if block:
            self._daily_loss_halt = True
        return {
            'cap_rs': cap_rs,
            'enforce': enforce,
            'realized_loss': realized_loss,
            'realized': pnl['realized'],
            'unrealized': pnl['unrealized'],
            'block_new_entries': block,
        }

    # =========================================================================
    # Order placement (paper or live)
    # =========================================================================

    def place_order_live_or_paper(
        self, *, system_id: str, instrument: str, direction: str,
        qty: int, entry_price: float, sl_price: float, target_price: float,
        signal_meta: Optional[dict] = None, position_id_prefill: Optional[int] = None,
    ) -> Optional[dict]:
        """Place an entry order. Paper mode = synthetic fill at entry_price.
        Live mode = real Kite MIS LIMIT order with 0.2% buffer.

        Returns: dict with position_id + order_id, or None on failure.
        """
        if qty <= 0:
            logger.warning(
                f'[I75-{self.config_id}/{system_id}] {instrument}: qty={qty}, skip'
            )
            return None

        txn_type = 'BUY' if direction == 'LONG' else 'SELL'
        is_paper = not self.is_live()
        now = datetime.now()
        today = now.date().isoformat()

        # Paper-mode synthetic order id; live-mode real Kite call
        if is_paper:
            order_id = f'PAPER-{_uuid.uuid4().hex[:12]}'
            logger.info(
                f'[I75-{self.config_id}/{system_id} PAPER] {txn_type} {qty} {instrument} '
                f'@ ~{entry_price:.2f} (SL {sl_price:.2f} TGT {target_price:.2f}) '
                f'-> {order_id}'
            )
        else:
            try:
                kite = self._get_kite()
                buf = 1.002 if txn_type == 'BUY' else 0.998
                price = round(entry_price * buf, 1)
                resp = kite.place_order(
                    variety='regular', exchange='NSE',
                    tradingsymbol=instrument,
                    transaction_type=txn_type,
                    quantity=qty, product='MIS',
                    order_type='LIMIT',
                    price=price,
                )
                order_id = str(resp)
                logger.info(
                    f'[I75-{self.config_id}/{system_id} LIVE] {txn_type} {qty} {instrument} '
                    f'@ {price} -> {order_id}'
                )
            except Exception as e:
                logger.error(
                    f'[I75-{self.config_id}/{system_id}] LIVE entry FAILED for '
                    f'{instrument}: {e}', exc_info=True,
                )
                return None

        # Persist position + order rows
        try:
            import json as _json
            pos_id = self.db.add_position(
                system_id=system_id,
                instrument=instrument,
                trade_date=today,
                direction=direction,
                qty=qty,
                entry_price=round(float(entry_price), 2),
                entry_time=now.isoformat(),
                sl_price=round(float(sl_price), 2),
                target_price=round(float(target_price), 2),
                kite_entry_order_id=order_id,
                status='OPEN',
                paper_mode=1 if is_paper else 0,
                signal_meta=_json.dumps(signal_meta or {}),
            )
            self.db.log_order(
                system_id=system_id, position_id=pos_id,
                instrument=instrument, tradingsymbol=instrument,
                transaction_type=txn_type, qty=qty,
                order_type='LIMIT', price=round(float(entry_price), 2),
                kite_order_id=order_id, status='PLACED',
                exchange='NSE', product='MIS',
                paper_mode=1 if is_paper else 0,
                notes=f'ENTRY {direction}',
            )
        except Exception as e:
            logger.error(
                f'[I75-{self.config_id}/{system_id}] DB persist failed for '
                f'{instrument}: {e}', exc_info=True,
            )
            return None

        # Cache in memory for dedup
        with self._lock:
            self._positions[(system_id, instrument)] = {
                'id': pos_id,
                'system_id': system_id,
                'instrument': instrument,
                'trade_date': today,
                'direction': direction,
                'qty': qty,
                'entry_price': round(float(entry_price), 2),
                'entry_time': now.isoformat(),
                'sl_price': round(float(sl_price), 2),
                'target_price': round(float(target_price), 2),
                'kite_entry_order_id': order_id,
                'status': 'OPEN',
                'paper_mode': 1 if is_paper else 0,
            }

        return {
            'position_id': pos_id,
            'order_id': order_id,
            'paper': is_paper,
        }

    def close_position(
        self, position: dict, exit_price: float, exit_reason: str,
    ) -> Optional[dict]:
        """Close a position: place exit order (paper or live), update DB."""
        sym = position['instrument']
        direction = position['direction']
        qty = int(position['qty'])
        entry_price = float(position['entry_price'])
        is_paper = not self.is_live()
        now = datetime.now()

        txn_type = 'SELL' if direction == 'LONG' else 'BUY'

        if is_paper:
            order_id = f'PAPER-{_uuid.uuid4().hex[:12]}'
            logger.info(
                f'[I75-{self.config_id} PAPER EXIT {exit_reason}] {txn_type} {qty} '
                f'{sym} @ ~{exit_price:.2f} -> {order_id}'
            )
        else:
            try:
                kite = self._get_kite()
                buf = 1.002 if txn_type == 'BUY' else 0.998
                price = round(exit_price * buf, 1)
                resp = kite.place_order(
                    variety='regular', exchange='NSE',
                    tradingsymbol=sym,
                    transaction_type=txn_type,
                    quantity=qty, product='MIS',
                    order_type='LIMIT', price=price,
                )
                order_id = str(resp)
                logger.info(
                    f'[I75-{self.config_id} LIVE EXIT {exit_reason}] {txn_type} {qty} '
                    f'{sym} @ {price} -> {order_id}'
                )
            except Exception as e:
                logger.error(
                    f'[I75-{self.config_id}] LIVE exit FAILED for {sym}: {e}',
                    exc_info=True,
                )
                return None

        pts = (
            (exit_price - entry_price) if direction == 'LONG'
            else (entry_price - exit_price)
        )
        pnl_inr = round(pts * qty, 2)

        try:
            self.db.close_position(
                position_id=position['id'],
                exit_price=round(float(exit_price), 2),
                exit_time=now.isoformat(),
                exit_reason=exit_reason,
                pnl_pts=round(pts, 2),
                pnl_inr=pnl_inr,
                kite_exit_order_id=order_id,
            )
            self.db.log_order(
                system_id=position.get('system_id', self.config_id),
                position_id=position['id'],
                instrument=sym, tradingsymbol=sym,
                transaction_type=txn_type, qty=qty,
                order_type='LIMIT',
                price=round(float(exit_price), 2),
                kite_order_id=order_id, status='PLACED',
                exchange='NSE', product='MIS',
                paper_mode=1 if is_paper else 0,
                notes=f'EXIT {exit_reason}',
            )
        except Exception as e:
            logger.error(
                f'[I75-{self.config_id}] close DB persist failed: {e}',
                exc_info=True,
            )

        with self._lock:
            key = (position.get('system_id', self.config_id), sym)
            self._positions.pop(key, None)

        return {
            'position_id': position['id'],
            'order_id': order_id,
            'pnl_inr': pnl_inr,
            'exit_reason': exit_reason,
            'paper': is_paper,
        }

    # =========================================================================
    # Position monitoring — TP / SL / EOD
    # =========================================================================

    def monitor_positions(self) -> list[dict]:
        """Check open positions for TP/SL/EOD and close as needed.
        Called from the cron every few minutes during the session."""
        if not self.is_enabled():
            return []
        try:
            results: list[dict] = []
            # Re-load from DB so we see positions across all sub-signals
            today = date.today().isoformat()
            cid = self.config_id
            with self.db.db_lock:
                conn = self.db._get_conn()
                try:
                    rows = conn.execute(
                        'SELECT * FROM i75_positions WHERE status="OPEN" '
                        'AND trade_date=? AND system_id LIKE ?',
                        (today, f'{cid}%'),
                    ).fetchall()
                    open_positions = [dict(r) for r in rows]
                finally:
                    conn.close()
            if not open_positions:
                return []

            ltps = self.get_live_ltp(
                list({p['instrument'] for p in open_positions}),
            )

            now_t = datetime.now().time()
            eod_str = self.cfg.get('eod_squareoff_time', '15:25')
            eh, em = [int(x) for x in eod_str.split(':')]
            from datetime import time as dtime
            eod_t = dtime(eh, em)

            for pos in open_positions:
                sym = pos['instrument']
                ltp = ltps.get(sym)
                if ltp is None:
                    if now_t >= eod_t:
                        # Use entry price as last-resort close
                        ltp = float(pos['entry_price'])
                    else:
                        continue

                exit_reason = None
                direction = pos['direction']
                sl_price = float(pos['sl_price'])
                target_price = float(pos['target_price'])

                # TP / SL
                if direction == 'LONG':
                    if ltp <= sl_price:
                        exit_reason = 'SL_HIT'
                    elif ltp >= target_price:
                        exit_reason = 'TARGET_HIT'
                else:
                    if ltp >= sl_price:
                        exit_reason = 'SL_HIT'
                    elif ltp <= target_price:
                        exit_reason = 'TARGET_HIT'

                # EOD square-off
                if exit_reason is None and now_t >= eod_t:
                    exit_reason = 'EOD_SQUAREOFF'

                if exit_reason:
                    out = self.close_position(pos, ltp, exit_reason)
                    if out:
                        results.append(out)
            return results
        except Exception as e:
            logger.error(
                f'[I75-{self.config_id}] monitor_positions error: {e}',
                exc_info=True,
            )
            return []

    # =========================================================================
    # EOD square-off + kill switch
    # =========================================================================

    def square_off_all(self, reason: str = 'EOD_SQUAREOFF') -> list[dict]:
        """Force-close every open position for this config."""
        today = date.today().isoformat()
        cid = self.config_id
        with self.db.db_lock:
            conn = self.db._get_conn()
            try:
                rows = conn.execute(
                    'SELECT * FROM i75_positions WHERE status="OPEN" '
                    'AND trade_date=? AND system_id LIKE ?',
                    (today, f'{cid}%'),
                ).fetchall()
                open_positions = [dict(r) for r in rows]
            finally:
                conn.close()
        if not open_positions:
            logger.info(
                f'[I75-{self.config_id}] square_off_all: no open positions',
            )
            return []

        ltps = self.get_live_ltp(
            list({p['instrument'] for p in open_positions}),
        )
        out: list[dict] = []
        for pos in open_positions:
            ltp = ltps.get(pos['instrument'], float(pos['entry_price']))
            r = self.close_position(pos, ltp, reason)
            if r:
                out.append(r)
        logger.info(
            f'[I75-{self.config_id}] square_off_all done: {len(out)} closed',
        )
        return out

    def kill_switch(self) -> dict:
        """Immediate halt: square off everything and disable for the day."""
        results = self.square_off_all('KILL_SWITCH')
        self._kill_switched_today = True
        self._kill_switch_date = date.today().isoformat()
        logger.critical(
            f'[I75-{self.config_id}] KILL SWITCH activated — '
            f'closed {len(results)} positions, halted for the day.'
        )
        return {
            'config_id': self.config_id,
            'closed_count': len(results),
            'closed': results,
            'paper_mode': self.is_paper(),
        }

    def is_killed_today(self) -> bool:
        today = date.today().isoformat()
        if self._kill_switched_today and self._kill_switch_date == today:
            return True
        return False

    # =========================================================================
    # Cohort load
    # =========================================================================

    @staticmethod
    def load_cohort(path: str) -> list[str]:
        """Read a one-symbol-per-line cohort file. Empty lines and blank
        symbols are skipped."""
        out: list[str] = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if s:
                        out.append(s)
        except Exception as e:
            logger.error(
                f'[I75] cohort load failed {path}: {e}', exc_info=False,
            )
        return out

    # =========================================================================
    # Pre-flight: signal eligibility checks shared across configs
    # =========================================================================

    def can_enter_now(self, system_id: str, instrument: str) -> tuple[bool, str]:
        """Return (allowed, reason). Centralises gate checks before placing
        a new entry. Caller does signal eval first; this is the *final* gate."""
        if not self.is_enabled():
            return False, 'config_disabled'
        if self.is_killed_today():
            return False, 'kill_switch_today'

        # Daily loss circuit-breaker
        loss_gate = self.daily_loss_gate()
        if loss_gate.get('block_new_entries'):
            return False, 'daily_loss_breaker'

        # Combined concurrency cap
        conc = self.concurrency_check()
        if not conc['allowed']:
            return False, f'concurrency_cap_{conc["cap"]}_reached'

        # One trade per stock per day per sub-system
        if self.db.has_position_today(system_id, instrument):
            return False, 'already_traded_today'

        return True, 'ok'
