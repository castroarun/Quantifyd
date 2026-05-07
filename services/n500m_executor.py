"""Nifty 500 Intraday Momentum — order executor.

Mirrors the ORB paper-mode wrapper pattern (services/orb_live_engine.py:87-142):

  * `_is_paper()` reads the DB-stored mode (OFF / PAPER / LIVE), fail-closed.
  * `_kite_place_order/_kite_cancel_order/_kite_modify_order/_kite_order_history`
    are thin wrappers — in PAPER mode they return synthetic `PAPER-*` order
    IDs and log the would-be order; in LIVE mode they call Kite.
  * Position sizing: `risk_per_trade_inr` × `max_concurrent_trades` cap.

Public API used by app.py / scheduler:
  - submit_signal(signal_dict)         -> position_id or None
  - monitor_open_positions()           -> closes positions whose SL/TP/EOD hit
  - kill_switch()                      -> flatten everything, set mode OFF
"""
from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, date, time as dtime
from typing import Optional

from services.n500m_db import get_db
from services.n500m_scanner import EOD_SQUARE_OFF_TIME

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — overridable via config.py:N500M_DEFAULTS if desired later
# ---------------------------------------------------------------------------

RISK_PER_TRADE_INR = 3000.0
MAX_CONCURRENT_TRADES = 5
DEFAULT_PRODUCT = "MIS"
DEFAULT_EXCHANGE = "NSE"


def _get_kite():
    """Lazy-import the Kite singleton — fail-closed if unavailable in paper mode."""
    try:
        from services.kite_session import get_kite
        return get_kite()
    except Exception as e:
        logger.warning(f"[N500M] Kite unavailable: {e}")
        return None


class N500mExecutor:
    """Stateless wrapper — no in-memory position book; the DB is source of truth."""

    def __init__(self):
        self.db = get_db()
        self._lock = threading.Lock()

    # ----- mode helpers --------------------------------------------------

    def _mode(self) -> str:
        return self.db.get_mode()

    def _is_paper(self) -> bool:
        m = self._mode()
        # Fail-closed: anything other than LIVE → paper-or-off semantics
        return m != "LIVE"

    def _is_off(self) -> bool:
        return self._mode() == "OFF"

    def _kill_switched(self) -> bool:
        return self.db.is_kill_switch()

    # ----- Kite wrappers -------------------------------------------------

    def _kite_place_order(self, **kwargs) -> str:
        if self._is_paper():
            fake_id = f"PAPER-{uuid.uuid4().hex[:12]}"
            logger.info(
                f"[N500M PAPER] would place_order: "
                f"{kwargs.get('transaction_type')} {kwargs.get('quantity')} "
                f"{kwargs.get('tradingsymbol')} "
                f"{kwargs.get('order_type','MARKET')} "
                f"price={kwargs.get('price')} → {fake_id}"
            )
            return fake_id
        kite = _get_kite()
        if kite is None:
            raise RuntimeError("[N500M LIVE] Kite session unavailable; refusing to place order.")
        # variety defaults to 'regular' if not provided
        kwargs.setdefault("variety", "regular")
        order_id = kite.place_order(**kwargs)
        logger.info(f"[N500M LIVE] place_order -> {order_id} ({kwargs.get('tradingsymbol')})")
        return order_id

    def _kite_cancel_order(self, *, variety: str = "regular", order_id: str, **kw):
        if self._is_paper() or (order_id and str(order_id).startswith("PAPER-")):
            logger.info(f"[N500M PAPER] would cancel_order: {order_id}")
            return None
        kite = _get_kite()
        if kite is None:
            return None
        return kite.cancel_order(variety=variety, order_id=order_id, **kw)

    def _kite_order_history(self, order_id: str):
        if self._is_paper() or (order_id and str(order_id).startswith("PAPER-")):
            return [{"order_id": order_id, "status": "COMPLETE",
                     "average_price": None, "filled_quantity": 0,
                     "_paper": True}]
        kite = _get_kite()
        if kite is None:
            return []
        return kite.order_history(order_id) or []

    # ----- LTP fetch -----------------------------------------------------

    def get_ltp(self, symbols: list[str]) -> dict[str, float]:
        """Returns {symbol: ltp}. In paper mode fall back to last-known close
        from market_data.db so position monitoring still works without a
        live Kite session."""
        if self._is_paper():
            return self._ltp_from_db(symbols)
        kite = _get_kite()
        if kite is None:
            return self._ltp_from_db(symbols)
        try:
            instr = [f"{DEFAULT_EXCHANGE}:{s}" for s in symbols]
            quotes = kite.ltp(instr) or {}
            return {s: float(quotes.get(f"{DEFAULT_EXCHANGE}:{s}", {}).get("last_price") or 0)
                    for s in symbols}
        except Exception as e:
            logger.warning(f"[N500M] LTP fetch failed: {e}")
            return self._ltp_from_db(symbols)

    def _ltp_from_db(self, symbols: list[str]) -> dict[str, float]:
        """Last close from market_data_unified — for paper-mode P&L marking."""
        import sqlite3
        from pathlib import Path
        path = str(Path(__file__).resolve().parent.parent / "backtest_data" / "market_data.db")
        out = {}
        con = sqlite3.connect(path)
        try:
            for s in symbols:
                row = con.execute(
                    "SELECT close FROM market_data_unified "
                    "WHERE symbol=? AND timeframe='5minute' "
                    "ORDER BY date DESC LIMIT 1",
                    (s,)
                ).fetchone()
                if row:
                    out[s] = float(row[0])
        finally:
            con.close()
        return out

    # ----- position sizing ----------------------------------------------

    def position_size(self, *, entry_price: float, sl_price: Optional[float]) -> int:
        """Risk-based sizing: qty = risk_inr / risk_per_share. If no SL,
        fall back to entry_price * 0.02 (2%) as the assumed risk_per_share."""
        if entry_price <= 0:
            return 0
        if sl_price and sl_price > 0:
            risk_per_share = abs(entry_price - sl_price)
        else:
            risk_per_share = entry_price * 0.02  # T_NO assumes 2% risk
        if risk_per_share <= 0:
            return 0
        qty = int(RISK_PER_TRADE_INR / risk_per_share)
        return max(qty, 0)

    # ----- entry --------------------------------------------------------

    def submit_signal(self, signal: dict) -> Optional[int]:
        """Acts on a fresh signal from the scanner. Returns position_id or None."""
        if self._is_off():
            self._log_signal(signal, "SKIPPED:mode_off")
            return None
        if self._kill_switched():
            self._log_signal(signal, "SKIPPED:kill_switch")
            return None

        with self._lock:
            open_pos = self.db.get_open_positions()
            if len(open_pos) >= MAX_CONCURRENT_TRADES:
                self._log_signal(signal, "SKIPPED:max_concurrent")
                logger.info(f"[N500M] {signal['symbol']} skipped — max concurrent ({len(open_pos)})")
                return None

            # Duplicate check: same (symbol, signal_type, tf, dir) already open today
            sym = signal["symbol"]
            for p in open_pos:
                if (p["symbol"] == sym and p["signal_type"] == signal["signal_type"]
                        and p["timeframe"] == signal["timeframe"]
                        and p["direction"] == signal["direction"]):
                    self._log_signal(signal, "SKIPPED:duplicate")
                    return None

            qty = self.position_size(
                entry_price=signal["entry_price"],
                sl_price=signal.get("sl_price"),
            )
            if qty <= 0:
                self._log_signal(signal, "SKIPPED:zero_qty")
                return None

            mode = self._mode()
            tx = "BUY" if signal["direction"] == "long" else "SELL"

            entry_order_id = self._kite_place_order(
                tradingsymbol=sym,
                exchange=DEFAULT_EXCHANGE,
                transaction_type=tx,
                quantity=qty,
                order_type="MARKET",
                product=DEFAULT_PRODUCT,
            )

            position_id = self.db.insert_position(
                symbol=sym,
                signal_type=signal["signal_type"],
                trade_date=signal["trade_date"],
                timeframe=signal["timeframe"],
                direction=signal["direction"],
                qty=qty,
                entry_price=signal["entry_price"],
                entry_time=signal["signal_time"],
                sl_price=signal.get("sl_price"),
                target_price=signal.get("target_price"),
                atr_pts=signal.get("atr_pts"),
                exit_policy=signal["exit_policy"],
                variant_raw=signal.get("variant_raw"),
                expected_sharpe=signal.get("expected_sharpe"),
                kite_entry_order_id=entry_order_id,
                status="OPEN",
                mode=mode,
            )
            self.db.insert_order(
                position_id=position_id, symbol=sym, transaction_type=tx,
                qty=qty, order_type="MARKET",
                kite_order_id=entry_order_id, status="PLACED",
                mode=mode,
            )
            self._log_signal(signal, "ENTERED", entry_order_id=entry_order_id, position_id=position_id)
            logger.info(
                f"[N500M {mode}] ENTER {sym} {signal['direction']} qty={qty} "
                f"@{signal['entry_price']} SL={signal.get('sl_price')} "
                f"TGT={signal.get('target_price')} exit={signal['exit_policy']}"
            )
            return position_id

    def _log_signal(self, signal: dict, action: str, **extras):
        try:
            self.db.insert_signal(
                symbol=signal["symbol"],
                signal_type=signal["signal_type"],
                trade_date=signal["trade_date"],
                signal_time=signal["signal_time"],
                timeframe=signal["timeframe"],
                direction=signal["direction"],
                entry_price=signal["entry_price"],
                sl_price=signal.get("sl_price"),
                target_price=signal.get("target_price"),
                atr_pts=signal.get("atr_pts"),
                candle_open=signal.get("candle_open"),
                candle_high=signal.get("candle_high"),
                candle_low=signal.get("candle_low"),
                candle_close=signal.get("candle_close"),
                candle_volume=signal.get("candle_volume"),
                vm_ratio=signal.get("vm_ratio"),
                action_taken=action,
                notes=str(extras) if extras else None,
            )
        except Exception as e:
            logger.warning(f"[N500M] _log_signal({action}) failed: {e}")

    # ----- monitor + exit -----------------------------------------------

    def monitor_open_positions(self, now: Optional[datetime] = None) -> int:
        """Run every minute. Closes positions whose SL/TP/EOD condition hits.
        Returns count closed this tick."""
        now = now or datetime.now()
        open_pos = self.db.get_open_positions()
        if not open_pos:
            return 0

        symbols = list({p["symbol"] for p in open_pos})
        ltps = self.get_ltp(symbols)
        eod_force = now.time() >= EOD_SQUARE_OFF_TIME
        closed = 0

        for p in open_pos:
            ltp = ltps.get(p["symbol"])
            if ltp is None or ltp <= 0:
                continue

            reason = None
            sign = 1 if p["direction"] == "long" else -1

            if eod_force:
                reason = "EOD"
            else:
                # SL hit
                if p["sl_price"]:
                    sl_hit = ((ltp <= p["sl_price"]) if p["direction"] == "long"
                              else (ltp >= p["sl_price"]))
                    if sl_hit:
                        reason = "SL"
                # Target hit
                if reason is None and p["target_price"]:
                    tg_hit = ((ltp >= p["target_price"]) if p["direction"] == "long"
                              else (ltp <= p["target_price"]))
                    if tg_hit:
                        reason = "TARGET"

            if reason is None:
                continue

            self._close_position(p, exit_price=ltp, exit_time=now, reason=reason)
            closed += 1

        return closed

    def _close_position(self, p: dict, *, exit_price: float, exit_time: datetime, reason: str):
        sign = 1 if p["direction"] == "long" else -1
        tx_close = "SELL" if p["direction"] == "long" else "BUY"
        mode = p.get("mode", self._mode())

        exit_order_id = self._kite_place_order(
            tradingsymbol=p["symbol"],
            exchange=DEFAULT_EXCHANGE,
            transaction_type=tx_close,
            quantity=int(p["qty"]),
            order_type="MARKET",
            product=DEFAULT_PRODUCT,
        )

        pnl_pts = sign * (exit_price - p["entry_price"])
        pnl_inr = pnl_pts * p["qty"]

        self.db.update_position(
            p["id"],
            exit_price=round(exit_price, 2),
            exit_time=exit_time.isoformat(),
            exit_reason=reason,
            pnl_pts=round(pnl_pts, 2),
            pnl_inr=round(pnl_inr, 2),
            kite_exit_order_id=exit_order_id,
            status="CLOSED",
        )
        self.db.insert_order(
            position_id=p["id"], symbol=p["symbol"], transaction_type=tx_close,
            qty=int(p["qty"]), order_type="MARKET",
            kite_order_id=exit_order_id, status="PLACED",
            mode=mode, notes=f"close:{reason}",
        )
        logger.info(
            f"[N500M {mode}] EXIT {p['symbol']} {p['direction']} qty={p['qty']} "
            f"@{exit_price} reason={reason} P&L={pnl_inr:+.0f}"
        )

    # ----- kill switch --------------------------------------------------

    def kill_switch(self) -> int:
        """Engage kill switch: flatten all open positions at LTP, lock mode OFF.
        Returns count flattened."""
        self.db.set_kill_switch(True)
        open_pos = self.db.get_open_positions()
        if not open_pos:
            self.db.set_mode("OFF")
            return 0
        symbols = list({p["symbol"] for p in open_pos})
        ltps = self.get_ltp(symbols)
        n = 0
        for p in open_pos:
            ltp = ltps.get(p["symbol"]) or float(p["entry_price"])
            self._close_position(p, exit_price=ltp, exit_time=datetime.now(),
                                 reason="KILL_SWITCH")
            n += 1
        self.db.set_mode("OFF")
        logger.warning(f"[N500M] KILL SWITCH engaged — flattened {n} positions")
        return n


_executor_singleton: Optional[N500mExecutor] = None
_executor_lock = threading.Lock()


def get_executor() -> N500mExecutor:
    global _executor_singleton
    if _executor_singleton is None:
        with _executor_lock:
            if _executor_singleton is None:
                _executor_singleton = N500mExecutor()
    return _executor_singleton
