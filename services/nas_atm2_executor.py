"""
NAS ATM2 — Nifty ATM Strangle Executor (Exit-Both-on-SL variant)
=================================================================
Subclass of NasAtmExecutor with ONE key difference:
  - When ANY single leg hits 30% SL, BOTH legs are closed immediately.
  - No trailing the surviving leg to cost.
  - No re-entry after SL.

Everything else (entry logic, strike selection, guardrails, eod_squareoff)
is inherited unchanged from NasAtmExecutor.
"""

import logging
from datetime import datetime

from services.nas_atm_executor import NasAtmExecutor
from services.nas_atm2_db import get_nas_atm2_db
from services.nas_atm_scanner import NasAtmScanner
from config import NAS_ATM2_DEFAULTS

logger = logging.getLogger(__name__)


class NasAtm2Executor(NasAtmExecutor):
    """
    NAS ATM2: exit entire strangle when any single leg hits SL.
    No trailing, no re-entry.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or dict(NAS_ATM2_DEFAULTS)
        self.db = get_nas_atm2_db()
        self.scanner = NasAtmScanner(self.cfg)

    # --- SL Check (called on every tick) --------------------------------

    def check_and_handle_sl(self, positions=None, live_ltps=None):
        """
        Check stop losses on every tick. When ANY leg's live premium >= sl_price:
          1. Close ALL active legs in that strangle (both CE and PE)
          2. Record the completed trade
          3. Do NOT trail the surviving leg
          4. Do NOT re-enter a new strangle

        Args:
            positions: list of active position dicts (fetched if None)
            live_ltps: dict mapping tradingsymbol -> live LTP (fetched if None)

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

        # Track which strangles we've already exited this call
        exited_strangles = set()

        # Check each position for SL breach
        for pos in positions:
            if pos['status'] != 'ACTIVE':
                continue

            strangle_id = pos.get('strangle_id')
            if strangle_id in exited_strangles:
                continue

            tsym = pos.get('tradingsymbol', '')
            live_prem = live_ltps.get(tsym)
            if live_prem is None:
                continue

            sl_price = pos.get('sl_price')
            if sl_price is None:
                continue

            # SL hit: live premium >= sl_price (for short positions, premium going up = loss)
            if live_prem >= sl_price:
                logger.info(f"[NAS-ATM2] SL HIT: {tsym} live={live_prem:.2f} >= SL={sl_price:.2f}")
                logger.info(f"[NAS-ATM2] Closing BOTH legs of strangle #{strangle_id}")

                exited_strangles.add(strangle_id)

                # Find all active legs in this strangle
                strangle_legs = [p for p in positions
                                 if p.get('strangle_id') == strangle_id
                                 and p['status'] == 'ACTIVE']

                action = {
                    'type': 'SL_EXIT_BOTH',
                    'trigger_position_id': pos['id'],
                    'trigger_tradingsymbol': tsym,
                    'trigger_sl_price': sl_price,
                    'trigger_live_prem': live_prem,
                    'closed_legs': [],
                }

                # Close every leg in the strangle
                for leg_pos in strangle_legs:
                    leg_tsym = leg_pos.get('tradingsymbol', '')
                    leg_live = live_ltps.get(leg_tsym)

                    # If we don't have a live price for the other leg, fetch it
                    if leg_live is None:
                        leg_live = self.scanner.get_live_option_premium(leg_tsym)

                    # Fallback to 0 if still unavailable
                    if leg_live is None:
                        leg_live = 0
                        logger.warning(f"[NAS-ATM2] No live price for {leg_tsym}, using 0")

                    # Determine exit reason
                    is_trigger = (leg_pos['id'] == pos['id'])
                    exit_reason = 'SL_HIT' if is_trigger else 'SL_EXIT_BOTH'

                    self._close_leg(leg_pos, leg_live, exit_reason)

                    # Update daily P&L
                    pnl_this_leg = (leg_pos['entry_price'] - leg_live) * leg_pos['qty']
                    state = self.db.get_state()
                    current_daily_pnl = (state.get('daily_pnl', 0) or 0) + pnl_this_leg
                    self.db.update_state(daily_pnl=round(current_daily_pnl, 2))

                    action['closed_legs'].append({
                        'position_id': leg_pos['id'],
                        'tradingsymbol': leg_tsym,
                        'leg': leg_pos.get('instrument_type', ''),
                        'entry_price': leg_pos['entry_price'],
                        'exit_price': round(leg_live, 2),
                        'exit_reason': exit_reason,
                        'pnl': round(pnl_this_leg, 2),
                    })

                    logger.info(f"[NAS-ATM2] Closed {leg_tsym} @ {leg_live:.2f} "
                                f"(entry={leg_pos['entry_price']:.2f}, "
                                f"P&L={pnl_this_leg:.0f}) [{exit_reason}]")

                # Record the completed trade
                fresh_positions = self.db.get_positions_by_strangle(strangle_id)
                self._record_trade(strangle_id, fresh_positions, 'SL_EXIT_BOTH')

                action['total_pnl'] = round(
                    sum(leg['pnl'] for leg in action['closed_legs']), 2)

                actions.append(action)

        return actions

    # --- State for Dashboard -------------------------------------------

    def get_full_state(self):
        """Get complete state for dashboard API, with ATM2-specific config."""
        result = super().get_full_state()
        # Override config to reflect ATM2 behavior
        result['config']['exit_both_on_sl'] = True
        result['config']['re_enter_on_sl'] = False
        result['config']['trail_to_cost_on_sl'] = False
        return result
