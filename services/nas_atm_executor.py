"""
NAS ATM — Nifty ATM Strangle — Executor
==========================================
Handles trade execution for intraday Nifty ATM options strangles.

Key differences from NAS OTM:
  - Strike selection: always ATM (round spot to nearest strike interval)
  - SL per leg: entry_premium x (1 + leg_sl_pct)  [default 30%]
  - On SL hit: close stopped leg, trail surviving leg SL to entry (breakeven),
    then enter a NEW ATM strangle at current spot (cascading re-entry)
  - No premium-matching adjustments — only SL-based exits and re-entries
"""

import logging
from datetime import datetime, date, timedelta

from services.nas_atm_db import get_nas_atm_db
from services.nas_atm_scanner import NasAtmScanner, LOT_SIZE, get_current_week_expiry
from config import NAS_ATM_DEFAULTS

logger = logging.getLogger(__name__)


class NasAtmExecutor:
    """
    State machine for NAS ATM intraday strangle execution.
    Designed to be called on every tick for SL checks, and periodically for entries.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or dict(NAS_ATM_DEFAULTS)
        self.db = get_nas_atm_db()
        self.scanner = NasAtmScanner(self.cfg)

    # --- Guardrails ----------------------------------------------------

    def _check_guardrails(self, is_entry=True):
        """Pre-order safety checks. Returns (passed, reason)."""
        cfg = self.cfg

        if not cfg.get('enabled', True):
            return False, 'System disabled'

        # Daily order limit
        today_orders = self.db.get_today_order_count()
        if today_orders >= cfg.get('max_daily_orders', 40):
            return False, f'Daily order limit ({cfg["max_daily_orders"]}) reached'

        if is_entry:
            # Max concurrent strangles
            active = self.db.get_active_positions()
            strangle_ids = set(p.get('strangle_id') for p in active if p.get('strangle_id'))
            max_strangles = cfg.get('max_strangles', 5)
            if len(strangle_ids) >= max_strangles:
                return False, f'Max strangles ({max_strangles}) reached'

            # Daily P&L circuit breaker
            state = self.db.get_state()
            daily_pnl = state.get('daily_pnl', 0) or 0
            max_loss = cfg.get('max_daily_loss', 25000)
            if daily_pnl < -max_loss:
                return False, f'Daily loss Rs {abs(daily_pnl):.0f} exceeds limit Rs {max_loss}'

            # Entry time window
            now = datetime.now()
            current_time = now.strftime('%H:%M')
            entry_start = cfg.get('entry_start_time', '09:30')
            entry_end = cfg.get('entry_end_time', '14:50')
            if not (entry_start <= current_time <= entry_end):
                return False, f'Outside entry window ({entry_start}-{entry_end})'

        return True, 'OK'

    # --- Tradingsymbol Builder -----------------------------------------

    def _build_tradingsymbol(self, instrument_type, strike, expiry):
        """Build NFO tradingsymbol — delegates to scanner."""
        return self.scanner.build_tradingsymbol(instrument_type, strike, expiry)

    # --- Order Placement -----------------------------------------------

    def _place_order(self, tradingsymbol, transaction_type, qty, price,
                     leg, instrument_type, strike, expiry_date,
                     signal_type, strangle_id, sl_price=None):
        """
        Place an order. Paper mode: fill immediately. Live: place on Kite.
        Returns position_id or None.
        """
        cfg = self.cfg
        now = datetime.now().isoformat()

        position_id = self.db.add_position(
            strangle_id=strangle_id,
            leg=leg,
            tradingsymbol=tradingsymbol,
            exchange='NFO',
            transaction_type=transaction_type,
            instrument_type=instrument_type,
            qty=qty,
            strike=strike,
            expiry_date=str(expiry_date),
            entry_price=price,
            entry_time=now,
            sl_price=sl_price,
            signal_type=signal_type,
            status='ACTIVE' if cfg.get('paper_trading_mode', True) else 'PENDING',
        )

        order_status = 'PAPER' if cfg.get('paper_trading_mode', True) else 'PLACED'
        self.db.log_order(
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            qty=qty,
            price=price,
            order_type='MARKET',
            status=order_status,
            position_id=position_id,
            signal_type=signal_type,
        )

        # Live mode: place on Kite
        if not cfg.get('paper_trading_mode', True):
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                order_id = kite.place_order(
                    variety='regular',
                    exchange='NFO',
                    tradingsymbol=tradingsymbol,
                    transaction_type=transaction_type,
                    quantity=qty,
                    product='MIS',
                    order_type='MARKET',
                )
                self.db.update_position(position_id, kite_order_id=str(order_id))
                logger.info(f"[NAS-ATM] Kite order placed: {order_id} for {tradingsymbol}")
            except Exception as e:
                logger.error(f"[NAS-ATM] Kite order failed: {e}")
                self.db.update_position(position_id, status='FAILED', notes=str(e))
                return None

        logger.info(f"[NAS-ATM] [{order_status}] {transaction_type} {qty} {tradingsymbol} "
                     f"@ {price:.2f} SL={sl_price:.2f} ({signal_type})")
        return position_id

    def _close_leg(self, pos, exit_price, exit_reason):
        """Close a single position leg."""
        cfg = self.cfg
        self.db.close_position(pos['id'], round(exit_price, 2), exit_reason)

        # Live mode: buy back
        if not cfg.get('paper_trading_mode', True):
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                kite.place_order(
                    variety='regular',
                    exchange='NFO',
                    tradingsymbol=pos['tradingsymbol'],
                    transaction_type='BUY',
                    quantity=pos['qty'],
                    product='MIS',
                    order_type='MARKET',
                )
            except Exception as e:
                logger.error(f"[NAS-ATM] Exit order failed for {pos['tradingsymbol']}: {e}")

        self.db.log_order(
            tradingsymbol=pos['tradingsymbol'],
            transaction_type='BUY',
            qty=pos['qty'],
            price=exit_price,
            order_type='MARKET',
            status='PAPER' if cfg.get('paper_trading_mode', True) else 'PLACED',
            position_id=pos['id'],
            signal_type=exit_reason,
        )

        logger.info(f"[NAS-ATM] [EXIT] {pos['tradingsymbol']} -- {exit_reason} @ {exit_price:.2f}")

    # --- Entry Execution -----------------------------------------------

    def execute_strangle_entry(self, spot=None, scan_result=None):
        """
        Execute an ATM strangle entry (sell CE + PE at same ATM strike).

        Gets ATM strike from current spot, fetches live premiums,
        computes per-leg SL, and places both legs.

        Returns (strangle_id, message) or (None, reason).
        """
        ok, reason = self._check_guardrails(is_entry=True)
        if not ok:
            logger.info(f"[NAS-ATM] Entry guardrail: {reason}")
            return None, reason

        cfg = self.cfg
        lots = cfg.get('lots_per_leg', 5)
        qty = lots * LOT_SIZE
        leg_sl_pct = cfg.get('leg_sl_pct', 0.30)

        # Get current spot if not provided
        if spot is None:
            spot = self.scanner.get_live_spot()
            if spot is None:
                return None, 'Cannot fetch live spot price'

        # ATM strike
        atm_strike = self.scanner.get_atm_strike(spot)

        # Expiry
        today = date.today()
        expiry = get_current_week_expiry(today)

        # Build tradingsymbols
        ce_symbol = self._build_tradingsymbol('CE', atm_strike, expiry)
        pe_symbol = self._build_tradingsymbol('PE', atm_strike, expiry)

        # Get live premiums
        ce_premium = self.scanner.get_live_option_premium(ce_symbol)
        pe_premium = self.scanner.get_live_option_premium(pe_symbol)

        if ce_premium is None or pe_premium is None:
            return None, f'Cannot fetch live premiums for {ce_symbol}/{pe_symbol}'

        if ce_premium <= 0 or pe_premium <= 0:
            return None, f'Invalid premiums CE={ce_premium} PE={pe_premium}'

        # Compute SL per leg: entry_premium x (1 + leg_sl_pct)
        ce_sl = round(ce_premium * (1 + leg_sl_pct), 2)
        pe_sl = round(pe_premium * (1 + leg_sl_pct), 2)

        strangle_id = self.db.get_next_strangle_id()

        # Sell CALL
        pos_ce = self._place_order(
            tradingsymbol=ce_symbol,
            transaction_type='SELL',
            qty=qty,
            price=ce_premium,
            leg='CE',
            instrument_type='CE',
            strike=atm_strike,
            expiry_date=expiry,
            signal_type='ATM_SELL_CE',
            strangle_id=strangle_id,
            sl_price=ce_sl,
        )

        # Sell PUT
        pos_pe = self._place_order(
            tradingsymbol=pe_symbol,
            transaction_type='SELL',
            qty=qty,
            price=pe_premium,
            leg='PE',
            instrument_type='PE',
            strike=atm_strike,
            expiry_date=expiry,
            signal_type='ATM_SELL_PE',
            strangle_id=strangle_id,
            sl_price=pe_sl,
        )

        # Log signal
        self.db.log_signal(
            signal_type='ATM_ENTRY',
            spot_price=spot,
            call_strike=atm_strike,
            put_strike=atm_strike,
            call_premium=ce_premium,
            put_premium=pe_premium,
            action_taken=(f"SELL {atm_strike}CE @{ce_premium:.1f} SL={ce_sl:.1f} + "
                          f"{atm_strike}PE @{pe_premium:.1f} SL={pe_sl:.1f} x{lots}L"),
        )

        # Update daily PnL tracking in state
        self.db.update_state(
            spot_price=spot,
            last_signal_time=datetime.now().isoformat(),
        )

        logger.info(f"[NAS-ATM] Strangle #{strangle_id}: SELL {atm_strike}CE @{ce_premium:.1f} + "
                     f"{atm_strike}PE @{pe_premium:.1f} | SL CE={ce_sl:.1f} PE={pe_sl:.1f} | {lots}L")

        return strangle_id, 'OK'

    # --- SL Check (called on every tick) --------------------------------

    def check_and_handle_sl(self, positions=None, live_ltps=None):
        """
        Check stop losses on every tick. For each active position:
        - If live premium >= sl_price -> leg is stopped out
        - On SL hit:
          1. Close the stopped leg at live premium
          2. Trail the other leg's SL to its entry price (breakeven)
          3. If within entry window, enter a NEW ATM strangle

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

        cfg = self.cfg
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

        # Check each position for SL breach
        for pos in positions:
            if pos['status'] != 'ACTIVE':
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
                logger.info(f"[NAS-ATM] SL HIT: {tsym} live={live_prem:.2f} >= SL={sl_price:.2f}")

                # 1. Close the stopped leg
                self._close_leg(pos, live_prem, 'SL_HIT')

                # Update daily P&L
                pnl_this_leg = (pos['entry_price'] - live_prem) * pos['qty']
                state = self.db.get_state()
                current_daily_pnl = (state.get('daily_pnl', 0) or 0) + pnl_this_leg
                self.db.update_state(daily_pnl=round(current_daily_pnl, 2))

                action = {
                    'type': 'SL_HIT',
                    'position_id': pos['id'],
                    'tradingsymbol': tsym,
                    'entry_price': pos['entry_price'],
                    'exit_price': live_prem,
                    'sl_price': sl_price,
                    'pnl': round(pnl_this_leg, 2),
                }

                # 2. Trail the other leg's SL to its entry price (breakeven)
                other_leg_type = 'PE' if pos['instrument_type'] == 'CE' else 'CE'
                other_legs = [p for p in positions
                              if p.get('strangle_id') == pos.get('strangle_id')
                              and p['instrument_type'] == other_leg_type
                              and p['status'] == 'ACTIVE']

                for other in other_legs:
                    old_sl = other.get('sl_price', 0)
                    new_sl = other['entry_price']  # Trail to cost/breakeven
                    self.db.update_position(other['id'], sl_price=new_sl)
                    logger.info(f"[NAS-ATM] Trailed {other['tradingsymbol']} SL: "
                                f"{old_sl:.2f} -> {new_sl:.2f} (breakeven)")
                    action['trailed_leg'] = {
                        'position_id': other['id'],
                        'tradingsymbol': other['tradingsymbol'],
                        'old_sl': old_sl,
                        'new_sl': new_sl,
                    }

                # 3. Check if all legs of this strangle are now closed -> record trade
                all_strangle_positions = [p for p in positions
                                          if p.get('strangle_id') == pos.get('strangle_id')]
                # Refresh the stopped leg's status in our local list
                still_active = [p for p in all_strangle_positions
                                if p['id'] != pos['id'] and p['status'] == 'ACTIVE']
                if not still_active:
                    # Both legs closed — record the trade
                    self._record_trade(pos.get('strangle_id'), all_strangle_positions, 'SL_HIT')

                # 4. Enter a NEW ATM strangle if within entry window and allowed
                if cfg.get('re_enter_on_sl', True):
                    spot = self.scanner.get_live_spot()
                    if spot:
                        new_sid, msg = self.execute_strangle_entry(spot=spot)
                        if new_sid:
                            action['re_entry'] = {'strangle_id': new_sid, 'spot': spot}
                            logger.info(f"[NAS-ATM] Re-entered new strangle #{new_sid} after SL")
                        else:
                            action['re_entry_blocked'] = msg
                            logger.info(f"[NAS-ATM] Re-entry blocked: {msg}")

                actions.append(action)

        return actions

    # --- Exit All -------------------------------------------------------

    def exit_all_positions(self, exit_reason):
        """Close all active positions using live quotes."""
        active = self.db.get_active_positions()
        if not active:
            return []

        results = []
        strangle_groups = {}

        for pos in active:
            sid = pos.get('strangle_id', 0)
            if sid not in strangle_groups:
                strangle_groups[sid] = []
            strangle_groups[sid].append(pos)

        for pos in active:
            # Get exit price from live quote
            exit_price = None
            if pos.get('tradingsymbol'):
                try:
                    exit_price = self.scanner.get_live_option_premium(pos['tradingsymbol'])
                except Exception:
                    pass

            if exit_price is None:
                exit_price = 0

            self._close_leg(pos, exit_price, exit_reason)

            # Update daily P&L
            pnl_this_leg = (pos['entry_price'] - exit_price) * pos['qty']
            state = self.db.get_state()
            current_daily_pnl = (state.get('daily_pnl', 0) or 0) + pnl_this_leg
            self.db.update_state(daily_pnl=round(current_daily_pnl, 2))

            results.append({
                'position_id': pos['id'],
                'tradingsymbol': pos['tradingsymbol'],
                'exit_reason': exit_reason,
                'exit_price': round(exit_price, 2),
            })

        # Record trades per strangle
        for sid, positions_group in strangle_groups.items():
            self._record_trade(sid, positions_group, exit_reason)

        return results

    def _record_trade(self, strangle_id, positions, exit_reason, spot_at_exit=None):
        """Record a completed strangle trade when ALL legs are closed."""
        ce_pos = [p for p in positions if p['instrument_type'] == 'CE']
        pe_pos = [p for p in positions if p['instrument_type'] == 'PE']

        call_entry = ce_pos[0]['entry_price'] if ce_pos else 0
        put_entry = pe_pos[0]['entry_price'] if pe_pos else 0

        # Get exit prices: from the position if already closed, otherwise 0
        call_exit = 0
        if ce_pos:
            call_exit = ce_pos[0].get('exit_price') or 0
            # If still active (partially closed strangle), use 0
            if ce_pos[0]['status'] == 'ACTIVE':
                call_exit = 0

        put_exit = 0
        if pe_pos:
            put_exit = pe_pos[0].get('exit_price') or 0
            if pe_pos[0]['status'] == 'ACTIVE':
                put_exit = 0

        call_strike = ce_pos[0]['strike'] if ce_pos else 0
        put_strike = pe_pos[0]['strike'] if pe_pos else 0

        total_collected = call_entry + put_entry
        total_paid = call_exit + put_exit
        lots = (ce_pos[0]['qty'] // LOT_SIZE) if ce_pos else 0
        gross_pnl = (total_collected - total_paid) * LOT_SIZE * lots
        net_pnl = gross_pnl - (80 * 2)  # brokerage: Rs 40/order x 4 legs

        entry_time = min(p['entry_time'] for p in positions) if positions else None

        if spot_at_exit is None:
            spot_at_exit = self.scanner.get_live_spot()

        self.db.add_trade(
            strangle_id=strangle_id,
            trade_date=date.today().isoformat(),
            entry_time=entry_time,
            exit_time=datetime.now().isoformat(),
            spot_at_entry=None,
            spot_at_exit=spot_at_exit,
            call_strike=call_strike,
            put_strike=put_strike,
            call_entry_premium=call_entry,
            put_entry_premium=put_entry,
            call_exit_premium=call_exit,
            put_exit_premium=put_exit,
            total_premium_collected=round(total_collected, 2),
            total_premium_paid=round(total_paid, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            lots=lots,
            adjustments=0,
            exit_reason=exit_reason,
            expiry_date=positions[0].get('expiry_date') if positions else None,
        )

    # --- EOD Squareoff --------------------------------------------------

    def eod_squareoff(self):
        """Mandatory end-of-day squareoff + daily summary."""
        exits = self.exit_all_positions('eod_squareoff')

        # Save daily state
        stats = self.db.get_stats()
        today_trades = [t for t in self.db.get_recent_trades(limit=100)
                        if t.get('trade_date') == date.today().isoformat()]

        daily_pnl = sum(t.get('net_pnl', 0) or 0 for t in today_trades)
        wins = sum(1 for t in today_trades if (t.get('net_pnl', 0) or 0) > 0)
        losses = sum(1 for t in today_trades if (t.get('net_pnl', 0) or 0) <= 0)

        self.db.save_daily_state(
            trade_date=date.today().isoformat(),
            trades_taken=len(today_trades),
            adjustments=0,
            daily_pnl=round(daily_pnl, 2),
            cumulative_pnl=round(stats.get('total_pnl', 0), 2),
            win_count=wins,
            loss_count=losses,
        )

        # Reset daily counters
        self.db.update_state(daily_pnl=0, total_adjustments=0)

        logger.info(f"[NAS-ATM] EOD: {len(exits)} positions closed, daily P&L: Rs {daily_pnl:.0f}")
        return exits

    # --- Emergency ------------------------------------------------------

    def emergency_exit_all(self):
        """Kill switch — close all active positions immediately."""
        exits = self.exit_all_positions('EMERGENCY')
        logger.warning(f"[NAS-ATM] EMERGENCY EXIT: closed {len(exits)} positions")
        return len(exits)

    # --- Time-based Exit Check ------------------------------------------

    def check_time_exit(self):
        """Check if it's time for EOD squareoff."""
        cfg = self.cfg
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        eod_time = cfg.get('eod_squareoff_time', '15:15')

        if current_time >= eod_time:
            active = self.db.get_active_positions()
            if active:
                logger.info(f"[NAS-ATM] EOD squareoff triggered at {current_time}")
                return self.eod_squareoff()
        return []

    # --- State for Dashboard -------------------------------------------

    def get_full_state(self):
        """Get complete state for dashboard API."""
        state = self.db.get_state()
        stats = self.db.get_stats()
        active = self.db.get_active_positions()
        signals = self.db.get_recent_signals(limit=20)
        trades = self.db.get_recent_trades(limit=30)

        # Group active by leg
        ce_positions = [p for p in active if p.get('leg') == 'CE']
        pe_positions = [p for p in active if p.get('leg') == 'PE']

        # Today's closed positions
        closed_today = self.db.get_today_closed_positions()

        return {
            'state': state,
            'stats': stats,
            'config': {
                'paper_trading_mode': self.cfg.get('paper_trading_mode', True),
                'enabled': self.cfg.get('enabled', True),
                'lots_per_leg': self.cfg.get('lots_per_leg', 5),
                'leg_sl_pct': self.cfg.get('leg_sl_pct', 0.30),
                'max_strangles': self.cfg.get('max_strangles', 5),
                'eod_squareoff_time': self.cfg.get('eod_squareoff_time', '15:15'),
                'entry_start_time': self.cfg.get('entry_start_time', '09:30'),
                'entry_end_time': self.cfg.get('entry_end_time', '14:50'),
                'min_squeeze_bars': self.cfg.get('min_squeeze_bars', 1),
            },
            'positions': {
                'ce': ce_positions,
                'pe': pe_positions,
                'total_active': len(active),
                'closed_today': closed_today,
            },
            'recent_signals': signals,
            'recent_trades': trades,
        }
