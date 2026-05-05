"""MST Order Executor
==================

Handles option-leg placement and squareoff for the MST strategy.
Three modes:
  - off    → no orders (engine.enabled = False; this class is bypassed entirely)
  - paper  → orders are SIMULATED — log to DB with paper_mode=1, no Kite calls
  - live   → real orders via kite.place_order with strict guardrails

Spec: docs/Design/MST-INDEX-STRATEGY-DESIGN.md §4.4 (order placement)
"""
from __future__ import annotations
import logging
import time
from datetime import date, datetime
from typing import Optional

from services import mst_db

logger = logging.getLogger(__name__)


class MSTExecutor:
    """Place / close option legs for the MST strategy.

    Strike → tradingsymbol resolution: NIFTY weekly format is
    `NIFTY<YY><M><DD><STRIKE><CE|PE>` where M is single char (1-9, O, N, D).
    Use kite.instruments('NFO') to look up exact tradingsymbol + token at
    runtime — strikes change per expiry and the format has historically
    shifted (monthly vs weekly).
    """

    LIMIT_TIMEOUT_S = 30          # if LIMIT not filled in 30s, fall back to MARKET
    LOTS = 1
    LOT_SIZE = 75                  # NIFTY current lot size (read from FNO_LOT_SIZES at startup)
    SPREAD_WIDTH = 200
    NIFTY_NAME = "NIFTY"

    def __init__(self, kite=None, paper_mode: bool = True):
        self.kite = kite
        self.paper_mode = paper_mode
        self._instruments_cache: list[dict] = []
        self._cache_loaded_at: Optional[datetime] = None

    # ---- Instrument lookup ----

    def _ensure_instruments(self):
        """Load NFO instrument list from Kite (cached for the day)."""
        if self.paper_mode and self.kite is None:
            return
        if (self._cache_loaded_at is not None and
                (datetime.now() - self._cache_loaded_at).total_seconds() < 6 * 3600):
            return
        if self.kite is None:
            return
        try:
            all_instruments = self.kite.instruments("NFO")
            self._instruments_cache = [
                i for i in all_instruments if i.get("name") == self.NIFTY_NAME
            ]
            self._cache_loaded_at = datetime.now()
            logger.info(f"[MST] Loaded {len(self._instruments_cache)} NIFTY NFO instruments")
        except Exception as e:
            logger.error(f"[MST] Failed to load NFO instruments: {e}")

    def _resolve_leg(self, strike: int, option_type: str, expiry: date) -> Optional[dict]:
        """Resolve (strike, CE/PE, expiry) → instrument dict from Kite cache."""
        self._ensure_instruments()
        for i in self._instruments_cache:
            if (i["strike"] == strike and i["instrument_type"] == option_type
                    and i["expiry"] == expiry):
                return i
        return None

    # ---- Public API: spread placement ----

    def place_debit_spread(self, *, direction: int, atm: int, width: int,
                           expiry: date, t_minus_1: date, pyramid_level: int,
                           bar_dt: str) -> bool:
        """Place a bull call spread (long MST) or bear put spread (short MST)."""
        if direction == 1:
            long_strike = atm
            short_strike = atm + width
            opt_type = "CE"
        else:
            long_strike = atm
            short_strike = atm - width
            opt_type = "PE"

        week_label = expiry.isoformat()
        legs = [
            ("BUY", long_strike, "bull_long" if direction == 1 else "put_long"),
            ("SELL", short_strike, "bull_short" if direction == 1 else "put_short"),
        ]
        return self._place_legs(legs, opt_type, expiry, t_minus_1, week_label,
                                direction, pyramid_level, bar_dt, structure="debit")

    def place_credit_spread(self, *, direction: int, short_strike: int,
                            long_strike: int, expiry: date, t_minus_1: date,
                            pyramid_level: int, bar_dt: str) -> bool:
        """Place a bear call spread (long MST hedge) or bull put spread (short MST hedge)."""
        opt_type = "CE" if direction == 1 else "PE"
        week_label = expiry.isoformat()
        legs = [
            ("SELL", short_strike, "bear_short" if direction == 1 else "putw_short"),
            ("BUY", long_strike, "bear_long" if direction == 1 else "putw_long"),
        ]
        return self._place_legs(legs, opt_type, expiry, t_minus_1, week_label,
                                direction, pyramid_level, bar_dt, structure="credit")

    def place_reset_condor(self, *, direction: int, atm: int, width: int,
                           expiry: date, t_minus_1: date, bar_dt: str) -> bool:
        """Reset structure (Reading D — 100/100/100 spot-centered) when current
        week's credit was too low. All 4 legs at once on next week's expiry."""
        if direction == 1:
            opt_type = "CE"
            k1 = atm - width // 2     # e.g. atm-50 (slightly ITM long)
            k2 = atm                   # ATM short
            k3 = atm + width // 2      # e.g. atm+50 (OTM short)
            k4 = atm + 3 * width // 2  # e.g. atm+150 (OTM long)
        else:
            opt_type = "PE"
            k1 = atm + width // 2
            k2 = atm
            k3 = atm - width // 2
            k4 = atm - 3 * width // 2

        week_label = expiry.isoformat()
        legs = [
            ("BUY",  k1, "reset_long_a" ),
            ("SELL", k2, "reset_short_a"),
            ("SELL", k3, "reset_short_b"),
            ("BUY",  k4, "reset_long_b" ),
        ]
        return self._place_legs(legs, opt_type, expiry, t_minus_1, week_label,
                                direction, pyramid_level=1, bar_dt=bar_dt,
                                structure="reset")

    # ---- Internal: leg placement ----

    def _place_legs(self, legs, opt_type, expiry, t_minus_1, week_label,
                    direction, pyramid_level, bar_dt, structure):
        """Place a list of legs atomically. Returns True if all filled, False
        if any leg failed (in which case any filled legs are CLOSED to abort)."""
        placed_leg_ids = []
        for side, strike, role in legs:
            leg_id = self._place_one_leg(
                side=side, strike=strike, option_type=opt_type,
                expiry=expiry, t_minus_1=t_minus_1, week_label=week_label,
                leg_role=role, direction=direction,
                pyramid_level=pyramid_level, bar_dt=bar_dt,
            )
            if leg_id is None:
                # Abort: close any already-placed legs
                logger.error(f"[MST] {structure} placement failed at {role} "
                             f"strike={strike} side={side}; closing prior legs")
                for prior_id in placed_leg_ids:
                    pos = self._get_position(prior_id)
                    if pos:
                        self._close_one_leg(pos, reason=f"abort_on_{role}_failure")
                return False
            placed_leg_ids.append(leg_id)
        return True

    def _place_one_leg(self, *, side: str, strike: int, option_type: str,
                       expiry: date, t_minus_1: date, week_label: str,
                       leg_role: str, direction: int, pyramid_level: int,
                       bar_dt: str) -> Optional[int]:
        """Place a single option leg. Returns the new mst_positions row id, or None."""
        instrument = self._resolve_leg(strike, option_type, expiry)
        if instrument is None and not self.paper_mode:
            logger.error(f"[MST] Cannot resolve {strike} {option_type} {expiry} in instruments")
            return None

        tradingsymbol = (instrument["tradingsymbol"] if instrument
                         else f"NIFTY{expiry.strftime('%y%b').upper()}{strike}{option_type}")
        instrument_token = instrument["instrument_token"] if instrument else None
        qty = self.LOTS * self.LOT_SIZE

        # Insert the position record BEFORE order placement, status=PENDING
        leg_id = mst_db.insert_position({
            "week_label": week_label, "direction": direction,
            "pyramid_level": pyramid_level, "leg_role": leg_role,
            "side": side, "instrument_token": instrument_token,
            "tradingsymbol": tradingsymbol, "strike": strike,
            "option_type": option_type, "qty": qty,
            "status": "PENDING", "paper_mode": self.paper_mode,
            "expiry_date": expiry.isoformat(),
            "t_minus_1_date": t_minus_1.isoformat(),
        })

        if self.paper_mode:
            # Paper: simulate fill at last close (or strike-implied price for now)
            mst_db.update_position(leg_id, status="OPEN", entry_price=0.0,
                                   entry_time=datetime.now().isoformat(),
                                   order_id=f"PAPER-{leg_id}")
            mst_db.log_order(leg_id, f"PAPER-{leg_id}", side, qty, None, "LIMIT",
                             "FILLED", paper_mode=True, bar_dt=bar_dt)
            logger.info(f"[MST PAPER] {side} {qty} {tradingsymbol} @ MID "
                        f"(leg #{leg_id}, role={leg_role}, L{pyramid_level})")
            return leg_id

        # Live mode — real order via Kite
        try:
            ltp = self._get_ltp(instrument_token)
            order_id = self.kite.place_order(
                variety="regular",
                exchange="NFO",
                tradingsymbol=tradingsymbol,
                transaction_type=side,
                quantity=qty,
                product="NRML",
                order_type="LIMIT",
                price=round(ltp, 2) if ltp else None,
                validity="DAY",
                tag=f"MST{pyramid_level}",
            )
            mst_db.update_position(leg_id, status="OPEN", entry_price=ltp,
                                   entry_time=datetime.now().isoformat(),
                                   order_id=order_id)
            mst_db.log_order(leg_id, order_id, side, qty, ltp, "LIMIT", "PLACED",
                             paper_mode=False, bar_dt=bar_dt)
            logger.info(f"[MST LIVE] {side} {qty} {tradingsymbol} @ {ltp} "
                        f"(order_id={order_id}, leg #{leg_id})")
            return leg_id
        except Exception as e:
            mst_db.update_position(leg_id, status="REJECTED",
                                   exit_reason=f"order_error: {e}")
            mst_db.log_order(leg_id, None, side, qty, None, "LIMIT", "REJECTED",
                             paper_mode=False, bar_dt=bar_dt, error_msg=str(e))
            logger.error(f"[MST LIVE] Order failed for {tradingsymbol}: {e}")
            return None

    def _close_one_leg(self, pos: dict, reason: str) -> bool:
        if pos["status"] != "OPEN":
            return False
        leg_id = pos["id"]
        side = "SELL" if pos["side"] == "BUY" else "BUY"
        qty = pos["qty"]

        if pos.get("paper_mode"):
            mst_db.update_position(leg_id, status="CLOSED", exit_price=0.0,
                                   exit_time=datetime.now().isoformat(),
                                   exit_reason=reason)
            mst_db.log_order(leg_id, f"PAPER-CLOSE-{leg_id}", side, qty, None,
                             "MARKET", "FILLED", paper_mode=True)
            logger.info(f"[MST PAPER] CLOSE {pos['tradingsymbol']} (reason={reason})")
            return True

        try:
            order_id = self.kite.place_order(
                variety="regular", exchange="NFO",
                tradingsymbol=pos["tradingsymbol"],
                transaction_type=side, quantity=qty,
                product="NRML", order_type="MARKET",
                tag=f"MSTCLOSE",
            )
            mst_db.update_position(leg_id, status="CLOSED",
                                   exit_time=datetime.now().isoformat(),
                                   exit_reason=reason)
            mst_db.log_order(leg_id, order_id, side, qty, None, "MARKET", "PLACED",
                             paper_mode=False)
            logger.info(f"[MST LIVE] CLOSE {pos['tradingsymbol']} (order_id={order_id}, reason={reason})")
            return True
        except Exception as e:
            mst_db.log_order(leg_id, None, side, qty, None, "MARKET", "REJECTED",
                             paper_mode=False, error_msg=str(e))
            logger.error(f"[MST LIVE] Close failed for {pos['tradingsymbol']}: {e}")
            return False

    def close_position(self, pos: dict, reason: str) -> bool:
        """Public API: close a single open position."""
        return self._close_one_leg(pos, reason)

    def _get_position(self, leg_id: int) -> Optional[dict]:
        for p in mst_db.get_open_positions():
            if p["id"] == leg_id:
                return p
        return None

    def _get_ltp(self, instrument_token: int) -> Optional[float]:
        if self.kite is None:
            return None
        try:
            quote = self.kite.ltp([instrument_token])
            return list(quote.values())[0]["last_price"]
        except Exception as e:
            logger.warning(f"[MST] LTP fetch failed for {instrument_token}: {e}")
            return None

    # ---- Credit estimation (used by engine to decide build-condor vs roll) ----

    def estimate_credit(self, *, direction: int, short_strike: int,
                        long_strike: int, expiry: date) -> Optional[float]:
        """Return total credit per lot (= 75 contracts) for the bear call /
        bull put spread. Used by the engine to gate condor build vs reset path.

        Paper mode: returns None (engine treats as credit-OK by default).
        Live mode: queries LTP for both strikes, computes net premium × LOT_SIZE.
        """
        if self.paper_mode or self.kite is None:
            return None
        opt_type = "CE" if direction == 1 else "PE"
        short_inst = self._resolve_leg(short_strike, opt_type, expiry)
        long_inst = self._resolve_leg(long_strike, opt_type, expiry)
        if not short_inst or not long_inst:
            logger.warning(f"[MST] Credit estimate failed: instrument lookup miss")
            return None
        try:
            tokens = [short_inst["instrument_token"], long_inst["instrument_token"]]
            quotes = self.kite.ltp(tokens)
            short_ltp = quotes[str(tokens[0])]["last_price"]
            long_ltp = quotes[str(tokens[1])]["last_price"]
            net_premium = short_ltp - long_ltp
            return net_premium * self.LOT_SIZE
        except Exception as e:
            logger.warning(f"[MST] Credit estimate exception: {e}")
            return None
