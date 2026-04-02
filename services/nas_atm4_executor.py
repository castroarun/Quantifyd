"""
NAS ATM4 — Nifty ATM Strangle Executor (Roll-to-Match on SL)
==============================================================
Subclass of NasAtmExecutor with ONE key difference:
  - On first SL hit: close stopped leg, roll it to a new strike whose premium
    matches the surviving leg's current market price (price_x). Max 1 roll.
  - On second SL hit (after roll): close stopped leg, tighten surviving SL
    to price_x (no further rolls).

Everything else (entry logic, strike selection, guardrails, eod_squareoff)
is inherited unchanged from NasAtmExecutor.
"""

import logging
import re
from datetime import datetime

import numpy as np

from services.nas_atm_executor import NasAtmExecutor
from services.nas_atm4_db import get_nas_atm4_db
from services.nas_atm_scanner import NasAtmScanner, LOT_SIZE, get_current_week_expiry
from config import NAS_ATM4_DEFAULTS

logger = logging.getLogger(__name__)


class NasAtm4Executor(NasAtmExecutor):
    """
    NAS ATM4: on first SL hit, roll the stopped leg to match the surviving
    leg's premium. On second SL hit, tighten the surviving leg's SL to
    price_x with no further rolls.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or dict(NAS_ATM4_DEFAULTS)
        self.db = get_nas_atm4_db()
        self.scanner = NasAtmScanner(self.cfg)

    # --- Roll Helpers -----------------------------------------------------

    def _find_roll_strike(self, spot, inst_type, target_prem, expiry_date):
        """Find OTM strike with premium closest to target."""
        strike_step = self.cfg.get('strike_interval', 50)
        min_otm = self.cfg.get('min_otm_distance', 100)

        if inst_type == 'CE':
            start_strike = int(round((spot + min_otm) / strike_step)) * strike_step
            direction = 1
        else:
            start_strike = int(round((spot - min_otm) / strike_step)) * strike_step
            direction = -1

        best_strike = None
        best_prem = None
        best_diff = float('inf')

        for i in range(20):  # Check 20 strikes outward
            strike = start_strike + (i * strike_step * direction)
            tsym = self._build_roll_tradingsymbol(inst_type, strike, expiry_date)
            prem = self.scanner.get_live_option_premium(tsym)
            if prem is None:
                continue
            diff = abs(prem - target_prem)
            if diff < best_diff:
                best_diff = diff
                best_strike = strike
                best_prem = prem
            # Early exit if very close
            if diff <= 2.0:
                break
            # If premium is getting too far from target, stop searching
            if prem < target_prem * 0.5:
                break

        return best_strike, best_prem

    def _build_roll_tradingsymbol(self, instrument_type, strike, expiry_date):
        """Build NFO tradingsymbol for the rolled strike."""
        return self.scanner.build_tradingsymbol(instrument_type, strike, expiry_date)

    @staticmethod
    def _parse_price_x(notes):
        """Extract price_x value from position notes field."""
        if not notes:
            return None
        match = re.search(r'price_x=([\d.]+)', notes)
        if match:
            return float(match.group(1))
        return None

    # --- SuperTrend for naked leg exit -----------------------------------

    @staticmethod
    def _compute_supertrend(candles, period=7, multiplier=2):
        """
        Compute SuperTrend on list of OHLC dicts.

        Returns (st_value, direction) for the latest bar.
        direction: 1 = uptrend (premium rising, bad for short), -1 = downtrend (premium falling, good for short)
        Returns (None, None) if not enough data.
        """
        n = len(candles)
        if n < period + 1:
            return None, None

        high = np.array([c['high'] for c in candles])
        low = np.array([c['low'] for c in candles])
        close = np.array([c['close'] for c in candles])

        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        # Wilder's ATR
        atr = np.zeros(n)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        hl2 = (high + low) / 2
        up = hl2 + multiplier * atr
        dn = hl2 - multiplier * atr

        final_up = np.copy(up)
        final_dn = np.copy(dn)
        direction = np.ones(n, dtype=int)

        for i in range(1, n):
            if up[i] < final_up[i - 1] or close[i - 1] > final_up[i - 1]:
                final_up[i] = up[i]
            else:
                final_up[i] = final_up[i - 1]

            if dn[i] > final_dn[i - 1] or close[i - 1] < final_dn[i - 1]:
                final_dn[i] = dn[i]
            else:
                final_dn[i] = final_dn[i - 1]

            if direction[i - 1] == 1:
                direction[i] = -1 if close[i] < final_dn[i] else 1
            else:
                direction[i] = 1 if close[i] > final_up[i] else -1

        st_val = final_dn[-1] if direction[-1] == 1 else final_up[-1]
        return round(float(st_val), 2), int(direction[-1])

    # --- SL Check (called on every tick) --------------------------------

    def check_and_handle_sl(self, positions=None, live_ltps=None):
        """
        Check stop losses on every tick. Roll-to-Match logic:

        CASE 1 (first SL, no prior rolls):
          - Close stopped leg
          - Get surviving leg's live premium = price_x
          - Roll stopped leg to a new strike with premium ~ price_x
          - Set new leg SL = new_premium * 1.3, surviving SL = price_x * 1.3
          - Store price_x in notes on both legs

        CASE 2 (second SL, after a roll):
          - Close stopped leg
          - Tighten surviving leg SL to price_x (flat, no multiplier)
          - No further rolls

        Returns:
            list of actions taken (dicts with details)
        """
        if positions is None:
            positions = self.db.get_active_positions()

        if not positions:
            return []

        actions = []

        # Fetch live LTPs if not provided
        if live_ltps is None:
            live_ltps = {}
            for pos in positions:
                tsym = pos.get('tradingsymbol', '')
                if tsym:
                    ltp = self.scanner.get_live_option_premium(tsym)
                    if ltp is not None:
                        live_ltps[tsym] = ltp

        # Track which strangles we've already handled this tick
        handled_strangles = set()

        # Check each position for SL breach
        for pos in positions:
            if pos['status'] != 'ACTIVE':
                continue

            strangle_id = pos.get('strangle_id')
            if strangle_id in handled_strangles:
                continue

            tsym = pos.get('tradingsymbol', '')
            live_prem = live_ltps.get(tsym)
            if live_prem is None:
                continue

            sl_price = pos.get('sl_price')
            if sl_price is None:
                continue

            # SL hit: live premium >= sl_price (for short positions, premium going up = loss)
            if live_prem < sl_price:
                continue

            handled_strangles.add(strangle_id)

            # Determine if any leg in this strangle has been rolled (adjustment_count >= 1)
            strangle_legs = [p for p in positions
                             if p.get('strangle_id') == strangle_id
                             and p['status'] == 'ACTIVE']
            has_rolled = any(
                (p.get('adjustment_count') or 0) >= 1 for p in strangle_legs
            )

            if not has_rolled:
                # --- CASE 1: First SL hit, no prior rolls ---
                action = self._handle_first_sl(pos, strangle_id, strangle_legs,
                                               live_prem, live_ltps)
            else:
                # --- CASE 2: Second SL hit, after a roll ---
                action = self._handle_second_sl(pos, strangle_id, strangle_legs,
                                                live_prem, live_ltps)

            if action:
                actions.append(action)

        return actions

    def _handle_first_sl(self, pos, strangle_id, strangle_legs, live_prem, live_ltps):
        """CASE 1: First SL hit — roll stopped leg to match surviving leg's premium."""
        logger.info(f"[NAS-ATM4] FIRST SL HIT: {pos['tradingsymbol']} "
                     f"live={live_prem:.2f} >= SL={pos['sl_price']:.2f}")

        # 1. Close the stopped leg
        self._close_leg(pos, live_prem, 'SL_HIT')

        # Update daily P&L
        pnl_this_leg = (pos['entry_price'] - live_prem) * pos['qty']
        state = self.db.get_state()
        current_daily_pnl = (state.get('daily_pnl', 0) or 0) + pnl_this_leg
        self.db.update_state(daily_pnl=round(current_daily_pnl, 2))

        action = {
            'type': 'ATM4_FIRST_SL',
            'position_id': pos['id'],
            'tradingsymbol': pos['tradingsymbol'],
            'entry_price': pos['entry_price'],
            'exit_price': live_prem,
            'sl_price': pos['sl_price'],
            'pnl': round(pnl_this_leg, 2),
        }

        # 2. Get the other (surviving) leg's live premium = price_x
        other_leg_type = 'PE' if pos['instrument_type'] == 'CE' else 'CE'
        surviving_legs = [p for p in strangle_legs
                          if p['instrument_type'] == other_leg_type
                          and p['status'] == 'ACTIVE']

        if not surviving_legs:
            logger.warning(f"[NAS-ATM4] No surviving leg found for strangle #{strangle_id}")
            # Record trade if all legs closed
            fresh = self.db.get_positions_by_strangle(strangle_id)
            still_active = [p for p in fresh if p['status'] == 'ACTIVE']
            if not still_active:
                self._record_trade(strangle_id, fresh, 'SL_HIT')
            return action

        surviving = surviving_legs[0]
        surviving_tsym = surviving.get('tradingsymbol', '')
        price_x = live_ltps.get(surviving_tsym)
        if price_x is None:
            price_x = self.scanner.get_live_option_premium(surviving_tsym)

        if price_x is None or price_x <= 0:
            logger.warning(f"[NAS-ATM4] Cannot get surviving leg premium for {surviving_tsym}, "
                           f"closing surviving leg too")
            exit_price = price_x if price_x and price_x > 0 else 0
            self._close_leg(surviving, exit_price, 'BOUNDARY_EXIT')
            fresh = self.db.get_positions_by_strangle(strangle_id)
            self._record_trade(strangle_id, fresh, 'BOUNDARY_EXIT')
            action['boundary_exit'] = True
            return action

        logger.info(f"[NAS-ATM4] Surviving {surviving_tsym} CMP = {price_x:.2f} (price_x)")

        # 3. Find a new strike for the stopped leg type with premium ~ price_x
        spot = self.scanner.get_live_spot()
        if spot is None:
            logger.warning(f"[NAS-ATM4] Cannot get spot price for roll, closing surviving leg")
            self._close_leg(surviving, price_x, 'BOUNDARY_EXIT')
            fresh = self.db.get_positions_by_strangle(strangle_id)
            self._record_trade(strangle_id, fresh, 'BOUNDARY_EXIT')
            action['boundary_exit'] = True
            return action

        expiry_date = pos.get('expiry_date')
        new_strike, new_prem = self._find_roll_strike(
            spot, pos['instrument_type'], price_x, expiry_date)

        if new_strike is None or new_prem is None:
            logger.warning(f"[NAS-ATM4] No suitable roll strike found for "
                           f"{pos['instrument_type']} target={price_x:.2f}, "
                           f"closing surviving leg")
            self._close_leg(surviving, price_x, 'BOUNDARY_EXIT')
            fresh = self.db.get_positions_by_strangle(strangle_id)
            self._record_trade(strangle_id, fresh, 'BOUNDARY_EXIT')
            action['boundary_exit'] = True
            return action

        # 4. Open new position: SELL at new strike
        new_tsym = self._build_roll_tradingsymbol(
            pos['instrument_type'], new_strike, expiry_date)

        # 5. New rolled leg SL = new_entry_premium * 1.3
        new_sl = round(new_prem * 1.3, 2)

        # 6. Surviving leg SL = price_x * 1.3
        surviving_new_sl = round(price_x * 1.3, 2)

        price_x_note = f"price_x={price_x:.2f}"

        # Place the rolled leg
        cfg = self.cfg
        now = datetime.now().isoformat()
        new_pos_id = self.db.add_position(
            strangle_id=strangle_id,
            leg=pos['leg'],
            tradingsymbol=new_tsym,
            exchange='NFO',
            transaction_type='SELL',
            instrument_type=pos['instrument_type'],
            qty=pos['qty'],
            strike=new_strike,
            expiry_date=str(expiry_date),
            entry_price=new_prem,
            entry_time=now,
            sl_price=new_sl,
            signal_type='ATM4_ROLL',
            status='ACTIVE' if cfg.get('paper_trading_mode', True) else 'PENDING',
            adjustment_count=1,
            notes=price_x_note,
        )

        # Log the order for the rolled leg
        order_status = 'PAPER' if cfg.get('paper_trading_mode', True) else 'PLACED'
        self.db.log_order(
            tradingsymbol=new_tsym,
            transaction_type='SELL',
            qty=pos['qty'],
            price=new_prem,
            order_type='MARKET',
            status=order_status,
            position_id=new_pos_id,
            signal_type='ATM4_ROLL',
        )

        # Live mode: place on Kite
        if not cfg.get('paper_trading_mode', True):
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                order_id = kite.place_order(
                    variety='regular',
                    exchange='NFO',
                    tradingsymbol=new_tsym,
                    transaction_type='SELL',
                    quantity=pos['qty'],
                    product='MIS',
                    order_type='MARKET',
                )
                self.db.update_position(new_pos_id, kite_order_id=str(order_id))
                logger.info(f"[NAS-ATM4] Kite roll order placed: {order_id} for {new_tsym}")
            except Exception as e:
                logger.error(f"[NAS-ATM4] Kite roll order failed: {e}")
                self.db.update_position(new_pos_id, status='FAILED', notes=str(e))

        # 7. Store price_x in new position notes (already done above)
        # 8. Store price_x in surviving leg's notes + update its SL
        self.db.update_position(
            surviving['id'],
            sl_price=surviving_new_sl,
            notes=price_x_note,
        )

        logger.info(f"[NAS-ATM4] ROLLED {pos['instrument_type']}: "
                     f"{pos['tradingsymbol']} -> {new_tsym} @ {new_prem:.2f} "
                     f"SL={new_sl:.2f} | price_x={price_x:.2f}")
        logger.info(f"[NAS-ATM4] Surviving {surviving_tsym} SL updated: "
                     f"{surviving.get('sl_price', 0):.2f} -> {surviving_new_sl:.2f}")

        # 10. Log signal
        self.db.log_signal(
            signal_type='ATM4_ROLL',
            spot_price=spot,
            call_strike=new_strike if pos['instrument_type'] == 'CE' else surviving['strike'],
            put_strike=new_strike if pos['instrument_type'] == 'PE' else surviving['strike'],
            call_premium=new_prem if pos['instrument_type'] == 'CE' else price_x,
            put_premium=new_prem if pos['instrument_type'] == 'PE' else price_x,
            action_taken=(f"ROLL {pos['instrument_type']} {pos['strike']}->{new_strike} "
                          f"@{new_prem:.1f} SL={new_sl:.1f} | "
                          f"price_x={price_x:.2f} | "
                          f"surviving SL->{surviving_new_sl:.1f}"),
        )

        action['roll'] = {
            'new_position_id': new_pos_id,
            'new_tradingsymbol': new_tsym,
            'new_strike': new_strike,
            'new_premium': new_prem,
            'new_sl': new_sl,
            'price_x': price_x,
            'surviving_id': surviving['id'],
            'surviving_new_sl': surviving_new_sl,
        }

        return action

    def _handle_second_sl(self, pos, strangle_id, strangle_legs, live_prem, live_ltps):
        """CASE 2: Second SL hit (after roll) — close stopped leg, tighten surviving SL."""
        logger.info(f"[NAS-ATM4] SECOND SL HIT: {pos['tradingsymbol']} "
                     f"live={live_prem:.2f} >= SL={pos['sl_price']:.2f}")

        # 1. Close the stopped leg
        self._close_leg(pos, live_prem, 'SL_HIT')

        # Update daily P&L
        pnl_this_leg = (pos['entry_price'] - live_prem) * pos['qty']
        state = self.db.get_state()
        current_daily_pnl = (state.get('daily_pnl', 0) or 0) + pnl_this_leg
        self.db.update_state(daily_pnl=round(current_daily_pnl, 2))

        action = {
            'type': 'ATM4_SECOND_SL',
            'position_id': pos['id'],
            'tradingsymbol': pos['tradingsymbol'],
            'entry_price': pos['entry_price'],
            'exit_price': live_prem,
            'sl_price': pos['sl_price'],
            'pnl': round(pnl_this_leg, 2),
        }

        # 2. Retrieve price_x from the stopped leg's notes
        price_x = self._parse_price_x(pos.get('notes'))

        # 3. Find surviving leg — disable flat SL, enable SuperTrend monitoring
        other_leg_type = 'PE' if pos['instrument_type'] == 'CE' else 'CE'
        surviving_legs = [p for p in strangle_legs
                          if p['instrument_type'] == other_leg_type
                          and p['status'] == 'ACTIVE']

        if surviving_legs:
            surviving = surviving_legs[0]
            old_sl = surviving.get('sl_price', 0)
            px_note = f"st_monitoring=true,price_x={price_x:.2f}" if price_x else "st_monitoring=true"
            # Set SL to 999999 so normal tick SL check won't fire — ST handles exit
            self.db.update_position(surviving['id'], sl_price=999999.0, notes=px_note)
            logger.info(f"[NAS-ATM4] Surviving {surviving['tradingsymbol']} SL disabled "
                         f"({old_sl:.2f} -> 999999), ST(7,2) monitoring enabled")
            action['type'] = 'ATM4_SECOND_SL'
            action['surviving_position'] = dict(surviving)
            action['surviving_tightened'] = {
                'position_id': surviving['id'],
                'tradingsymbol': surviving['tradingsymbol'],
                'old_sl': old_sl,
                'new_sl': 999999.0,
                'st_monitoring': True,
            }

        # 4. Check if all legs of this strangle are now closed
        fresh = self.db.get_positions_by_strangle(strangle_id)
        still_active = [p for p in fresh if p['status'] == 'ACTIVE']
        if not still_active:
            self._record_trade(strangle_id, fresh, 'SL_HIT')

        # Log signal
        spot = self.scanner.get_live_spot()
        self.db.log_signal(
            signal_type='ATM4_SECOND_SL',
            spot_price=spot,
            call_strike=pos['strike'] if pos['instrument_type'] == 'CE' else 0,
            put_strike=pos['strike'] if pos['instrument_type'] == 'PE' else 0,
            call_premium=live_prem if pos['instrument_type'] == 'CE' else 0,
            put_premium=live_prem if pos['instrument_type'] == 'PE' else 0,
            action_taken=(f"SECOND SL {pos['instrument_type']} {pos['tradingsymbol']} "
                          f"@{live_prem:.1f} | "
                          f"surviving SL->{price_x:.2f}" if price_x else
                          f"SECOND SL {pos['instrument_type']} {pos['tradingsymbol']} "
                          f"@{live_prem:.1f} | no price_x"),
        )

        return action

    # --- State for Dashboard -------------------------------------------

    def get_full_state(self):
        """Get complete state for dashboard API, with ATM4-specific config."""
        result = super().get_full_state()
        result['config']['max_rolls'] = self.cfg.get('max_rolls', 1)
        result['config']['roll_to_match'] = True
        result['config']['trail_to_cost_on_sl'] = False
        result['config']['re_enter_on_sl'] = False
        return result
