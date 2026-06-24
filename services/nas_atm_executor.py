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

    def _check_guardrails(self, is_entry=True, bypass_cooldown=False):
        """Pre-order safety checks. Returns (passed, reason)."""
        from datetime import datetime as _dt
        from services.nas_kill_switch import is_killed as _nas_killed
        cfg = self.cfg

        # Persistent panic kill — survives Flask restarts via sentinel file.
        if _nas_killed():
            return False, 'NAS PANIC kill active'

        if not cfg.get('enabled', True):
            return False, 'System disabled'

        # Day-of-week filter: skip entries on configured weekdays
        # (e.g. ATM2/V4 systems skip Wed/Thu). Basic ATM has no skip set.
        if is_entry:
            skip_days = cfg.get('skip_weekdays') or ()
            if _dt.now().weekday() in skip_days:
                return False, f'Skipped today (weekday filter: {skip_days})'

            # Day & Gap matrix gate (services.nas_day_matrix): per-system DTE/gap
            # decides ENTRY; mode sets live/paper via cfg['_force_mode']. Keyed by
            # cfg['matrix_key']. Fail-open to legacy if key absent or gate errors.
            _mkey = cfg.get('matrix_key')
            if _mkey:
                try:
                    from services.nas_day_matrix import gate as _mgate
                    _mg = _mgate(_mkey)
                except Exception:
                    _mg = None
                if _mg is not None:
                    if not _mg.get('allow'):
                        return False, 'day-matrix: ' + str(_mg.get('reason'))
                    cfg['_force_mode'] = _mg.get('mode') or 'paper'

        # DTE entry gate (research/51): only OPEN new positions within
        # max_dte_at_entry days of the weekly expiry. The chain replay showed the
        # edge is at 1 DTE while 4+ DTE bleeds. Expiry-day-agnostic (live expiry);
        # fail-open on calc error so a glitch never silently halts trading.
        if is_entry and cfg.get('max_dte_at_entry') is not None:
            try:
                from services.nas_scanner import get_current_week_expiry
                from datetime import date as _date
                _exp = get_current_week_expiry()
                if hasattr(_exp, 'date') and not isinstance(_exp, _date):
                    _exp = _exp.date()
                _dte = (_exp - _date.today()).days
                if _dte > cfg['max_dte_at_entry']:
                    return False, f"DTE {_dte} > max {cfg['max_dte_at_entry']} (0/1-DTE only)"
            except Exception as _e:
                logger.warning(f"[NAS-ATM] DTE gate calc failed (allowing entry): {_e}")

        # Daily order limit
        today_orders = self.db.get_today_order_count()
        if today_orders >= cfg.get('max_daily_orders', 40):
            return False, f'Daily order limit ({cfg["max_daily_orders"]}) reached'

        if is_entry:
            # Only 1 active strangle at a time
            active = self.db.get_active_positions()
            strangle_ids = set(p.get('strangle_id') for p in active if p.get('strangle_id'))
            max_strangles = cfg.get('max_strangles', 1)
            if len(strangle_ids) >= max_strangles:
                return False, f'Active strangle already exists ({len(strangle_ids)} active)'

            # Max re-entry cycles per day — count ALL completed strangle cycles today,
            # not just SL_HIT. The 2026-06-08 churn exited via ST_EXIT, which the old
            # SL_HIT-only counter never saw -> the cap never tripped -> per-candle re-entry
            # loop (research/60). Count distinct closed strangle_ids.
            today_trades = self.db.get_today_closed_positions()
            closed_cycles = len(set(t.get('strangle_id') for t in today_trades if t.get('strangle_id')))
            max_reentries = cfg.get('max_reentries', 5)
            if closed_cycles >= max_reentries:
                return False, f'Max re-entries ({max_reentries}) reached today ({closed_cycles} cycles)'

            # Re-entry cooldown — block a new strangle within reentry_cooldown_min of the
            # last exit. Breaks the pinned-market per-candle close->reopen loop. 0 = off.
            cooldown_min = cfg.get('reentry_cooldown_min', 0)
            if cooldown_min and today_trades and not bypass_cooldown:
                from datetime import datetime as _cd
                def _pt(ts):
                    s = str(ts) if ts else ''
                    for _fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
                        try:
                            return _cd.strptime(s[:26], _fmt)
                        except Exception:
                            pass
                    return None
                _exits = [d for d in (_pt(t.get('exit_time')) for t in today_trades) if d]
                last_exit = max(_exits) if _exits else None
                if last_exit:
                    _mins = (_cd.now() - last_exit).total_seconds() / 60.0
                    if 0 <= _mins < cooldown_min:
                        return False, f'Re-entry cooldown ({_mins:.1f} < {cooldown_min} min since last exit)'

            # Daily P&L circuit breaker -- compute from TODAY's closed positions,
            # NOT the persisted state['daily_pnl'] (which goes stale if a prior EOD
            # reset was skipped, e.g. during a freeze -- a stale -37638 wrongly blocked
            # 916-ATM2 on 2026-06-16). today_trades is already in scope; date-aware.
            daily_pnl = sum(((p.get('entry_price') or 0) - (p.get('exit_price') or 0)) * (p.get('qty') or 0)
                            for p in today_trades if p.get('exit_price') is not None)
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
                     signal_type, strangle_id, sl_price=None, entry_spot=None):
        """
        Place an order. Paper mode: fill immediately. Live: place on Kite.
        Returns position_id or None.
        """
        cfg = self.cfg
        now = datetime.now().isoformat()
        # Day-aware live/paper: REAL Kite orders only on live_weekdays (Mon/Tue/Fri);
        # every other day runs as PAPER so signals + P&L still record. 2026-06-03.
        # day-matrix gate (services.nas_day_matrix) overrides live/paper via cfg['_force_mode'].
        _fm = cfg.get('_force_mode')
        if _fm in ('live', 'paper'):
            _is_paper = (_fm == 'paper')
        else:
            _live_day = datetime.now().weekday() in cfg.get('live_weekdays', (0, 1, 4))
            _is_paper = cfg.get('paper_trading_mode', True) or (not _live_day)
        mode = 'paper' if _is_paper else 'live'

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
            entry_spot=entry_spot,
            sl_price=sl_price,
            signal_type=signal_type,
            status='ACTIVE' if _is_paper else 'PENDING',
            mode=mode,
        )

        order_status = 'PAPER' if _is_paper else 'PLACED'
        self.db.log_order(
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            qty=qty,
            price=price,
            order_type='MARKET',
            status=order_status,
            position_id=position_id,
            signal_type=signal_type,
            mode=mode,
        )

        # Live mode: place on Kite
        if not _is_paper:
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                from services.nas_kill_switch import is_frozen as _nas_frozen
                if _nas_frozen():
                    raise RuntimeError('NAS manual-freeze active - order blocked')
                order_id = kite.place_order(
                    variety='regular',
                    exchange='NFO',
                    tradingsymbol=tradingsymbol,
                    transaction_type=transaction_type,
                    quantity=qty,
                    product='MIS',
                    order_type='MARKET',
                )
                # MARKET orders fill effectively instantly. Flip to ACTIVE
                # so the SL monitor, adjustment logic, dashboard, and ticker
                # subscribe loop all see the position right away. Any rare
                # rejection is handled by the except branch below.
                self.db.update_position(
                    position_id,
                    status='ACTIVE',
                    kite_order_id=str(order_id),
                )
                logger.info(f"[NAS-ATM] Kite order placed (ACTIVE): {order_id} for {tradingsymbol}")
            except Exception as e:
                logger.error(f"[NAS-ATM] Kite order failed: {e}")
                self.db.update_position(position_id, status='FAILED', notes=str(e))
                return None

        logger.info(f"[NAS-ATM] [{order_status}] {transaction_type} {qty} {tradingsymbol} "
                     f"@ {price:.2f} SL={sl_price:.2f} ({signal_type})")
        return position_id

    def _confirm_exit_fill(self, order_id, tries=6, delay=0.4):
        """Poll Kite order_history until the exit order reaches a terminal state.

        Returns (status, avg_price) where status is one of 'COMPLETE',
        'REJECTED', 'CANCELLED', or 'PENDING' (still working / unknown after
        the poll window). MARKET orders normally fill on the first poll.
        """
        import time as _time
        from services.kite_service import get_kite
        kite = get_kite()
        last_status = 'PENDING'
        for _ in range(tries):
            _time.sleep(delay)
            try:
                hist = kite.order_history(order_id)
            except Exception as e:
                logger.warning(f"[NAS-ATM] order_history({order_id}) failed: {e}")
                continue
            if not hist:
                continue
            last = hist[-1]
            last_status = str(last.get('status') or '').upper()
            if last_status == 'COMPLETE':
                return 'COMPLETE', (last.get('average_price') or 0)
            if last_status in ('REJECTED', 'CANCELLED'):
                logger.error(f"[NAS-ATM] exit order {order_id} {last_status}: "
                             f"{last.get('status_message')}")
                return last_status, None
        return (last_status if last_status in ('REJECTED', 'CANCELLED') else 'PENDING'), None

    def _close_leg(self, pos, exit_price, exit_reason):
        """Close a single position leg — broker-truthful.

        The DB is marked CLOSED only when the buyback is actually confirmed at
        the exchange (or in paper mode). On the live path:

          * CONFIRMED (COMPLETE)          -> status CLOSED at the real fill price
          * placed but not yet confirmed  -> status CLOSING (+ kite_order_id) so
                                             the SL/ST monitor never re-fires a
                                             second buyback and the leg is not
                                             falsely closed; the reconciler
                                             finalises it against the orderbook.
          * place_order failed / REJECTED -> leg LEFT ACTIVE so the monitor keeps
                                             watching and retries; never orphaned.

        Returns True if the leg is closed or a confirmed exit is in flight;
        False if the exit failed and the leg is still live (the caller must
        treat the position as still open).
        """
        cfg = self.cfg
        # Same-day MIS: a position opened as paper is closed as paper. Day-aware so a
        # paper day never fires a real Kite close for a position with no real open.
        _fm = cfg.get('_force_mode')
        if _fm in ('live', 'paper'):
            paper = (_fm == 'paper')
        else:
            paper = cfg.get('paper_trading_mode', True) or (
                datetime.now().weekday() not in cfg.get('live_weekdays', (0, 1, 4)))

        # Paper mode: fill immediately (unchanged behaviour).
        if paper:
            self.db.close_position(pos['id'], round(exit_price, 2), exit_reason)
            self.db.log_order(
                tradingsymbol=pos['tradingsymbol'], transaction_type='BUY',
                qty=pos['qty'], price=exit_price, order_type='MARKET',
                status='PAPER', position_id=pos['id'],
                signal_type=exit_reason, mode='paper',
            )
            logger.info(f"[NAS-ATM] [EXIT] {pos['tradingsymbol']} -- {exit_reason} @ {exit_price:.2f}")
            return True

        # Live mode: place the buyback FIRST, confirm, THEN update the DB.
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            from services.nas_kill_switch import is_frozen as _nas_frozen
            if _nas_frozen():
                raise RuntimeError('NAS manual-freeze active - order blocked')
            order_id = kite.place_order(
                variety='regular', exchange='NFO',
                tradingsymbol=pos['tradingsymbol'], transaction_type='BUY',
                quantity=pos['qty'], product='MIS', order_type='MARKET',
            )
        except Exception as e:
            # Nothing reached the exchange -> keep the leg ACTIVE and retry on
            # the next SL/ST tick. NEVER mark it closed (that orphans a live short).
            logger.critical(f"[NAS-ATM] EXIT ORDER FAILED for {pos['tradingsymbol']} "
                            f"({exit_reason}) -- leg LEFT ACTIVE for retry: {e}")
            self.db.update_position(pos['id'], notes=f"EXIT_FAILED {exit_reason}: {e}")
            self.db.log_order(
                tradingsymbol=pos['tradingsymbol'], transaction_type='BUY',
                qty=pos['qty'], price=exit_price, order_type='MARKET',
                status='FAILED', position_id=pos['id'],
                signal_type=exit_reason, mode='live',
            )
            return False

        status, avg = self._confirm_exit_fill(order_id)

        if status == 'COMPLETE':
            fill = round(avg or exit_price, 2)
            self.db.close_position(pos['id'], fill, exit_reason)
            self.db.update_position(pos['id'], kite_order_id=str(order_id))
            self.db.log_order(
                tradingsymbol=pos['tradingsymbol'], transaction_type='BUY',
                qty=pos['qty'], price=fill, order_type='MARKET',
                status='COMPLETE', position_id=pos['id'],
                signal_type=exit_reason, mode='live',
            )
            logger.info(f"[NAS-ATM] [EXIT] {pos['tradingsymbol']} -- {exit_reason} "
                        f"@ {fill:.2f} (order {order_id})")
            return True

        if status in ('REJECTED', 'CANCELLED'):
            # The buyback did NOT fill -> the leg is still live. Keep it ACTIVE
            # so the monitor retries; never orphan it.
            logger.critical(f"[NAS-ATM] EXIT {status} for {pos['tradingsymbol']} "
                            f"order {order_id} -- leg LEFT ACTIVE for retry")
            self.db.update_position(pos['id'], notes=f"EXIT_{status} {exit_reason} order={order_id}")
            self.db.log_order(
                tradingsymbol=pos['tradingsymbol'], transaction_type='BUY',
                qty=pos['qty'], price=exit_price, order_type='MARKET',
                status=status, position_id=pos['id'],
                signal_type=exit_reason, mode='live',
            )
            return False

        # Placed but unconfirmed within the poll window. Park in CLOSING with the
        # order id so (a) the monitor won't re-fire another buyback (no double
        # cover) and (b) the reconciler finalises it once the orderbook settles.
        self.db.update_position(
            pos['id'], status='CLOSING', kite_order_id=str(order_id),
            exit_reason=exit_reason, notes=f"EXIT_PENDING {exit_reason} order={order_id}",
        )
        self.db.log_order(
            tradingsymbol=pos['tradingsymbol'], transaction_type='BUY',
            qty=pos['qty'], price=exit_price, order_type='MARKET',
            status='PENDING', position_id=pos['id'],
            signal_type=exit_reason, mode='live',
        )
        logger.warning(f"[NAS-ATM] [EXIT-PENDING] {pos['tradingsymbol']} -- {exit_reason} "
                       f"order {order_id} placed, fill unconfirmed -> CLOSING "
                       f"(reconciler will finalise)")
        return True

    # --- Entry Execution -----------------------------------------------

    def execute_strangle_entry(self, spot=None, scan_result=None, bypass_cooldown=False):
        """
        Execute an ATM strangle entry (sell CE + PE at same ATM strike).

        Gets ATM strike from current spot, fetches live premiums,
        computes per-leg SL, and places both legs.

        Returns (strangle_id, message) or (None, reason).
        """
        ok, reason = self._check_guardrails(is_entry=True, bypass_cooldown=bypass_cooldown)
        if not ok:
            logger.info(f"[NAS-ATM] Entry guardrail: {reason}")
            return None, reason

        cfg = self.cfg
        # Per-mode sizing (user 2026-06-24): the PAPER book runs paper_lots_per_leg (10 lots) for a
        # meaningful daily P&L curve; LIVE real money stays at lots_per_leg (1). _force_mode is set by
        # the day-matrix gate in _check_guardrails above; matches _place_order's live/paper decision.
        _fm = cfg.get('_force_mode')
        _is_paper_size = (_fm == 'paper') if _fm else cfg.get('paper_trading_mode', True)
        lots = cfg.get('paper_lots_per_leg', cfg.get('lots_per_leg', 5)) if _is_paper_size else cfg.get('lots_per_leg', 5)
        qty = lots * LOT_SIZE
        leg_sl_pct = cfg.get('leg_sl_pct', 0.30)

        # Get current spot if not provided
        if spot is None:
            spot = self.scanner.get_live_spot()
            if not spot:
                # fallback: direct REST quote of the NIFTY index, so the 09:16 one-shot
                # doesn't miss on a transient ticker-spot gap (2026-06-16 ATM/ATM4 missed).
                try:
                    from services.kite_service import get_kite
                    spot = get_kite().ltp(['NSE:NIFTY 50'])['NSE:NIFTY 50']['last_price']
                    logger.info(f"[NAS-ATM] spot REST fallback used: {spot}")
                except Exception as _e:
                    logger.warning(f"[NAS-ATM] spot REST fallback failed: {_e}")
                    spot = None
            if not spot:
                return None, 'Cannot fetch live spot price'

        # Expiry
        today = date.today()
        expiry = get_current_week_expiry(today)

        # ATM strike — snap to the FORWARD, not spot. Index options are priced
        # off the forward (C - P = F - K), so the equal-premium / delta-neutral
        # strike is the forward, not spot. Spot-based rounding picks a call-rich
        # strike when futures trade over spot (cost of carry) -> imbalanced
        # straddle (2026-06-01: 23550 CE 134 / PE 77). Derive the live synthetic
        # forward from the spot-nearest strike's own premiums (forward =
        # K + (CE - PE)) and re-snap. Falls back to the spot strike on any quote
        # failure, so this can never be worse than spot-based selection.
        strike_step = cfg.get('strike_interval', 50)
        atm_strike = self.scanner.get_atm_strike(spot)

        # Build tradingsymbols
        ce_symbol = self._build_tradingsymbol('CE', atm_strike, expiry)
        pe_symbol = self._build_tradingsymbol('PE', atm_strike, expiry)

        # Get live premiums
        ce_premium = self.scanner.get_live_option_premium(ce_symbol)
        pe_premium = self.scanner.get_live_option_premium(pe_symbol)

        if ce_premium is None or pe_premium is None:
            return None, f'Cannot fetch live premiums for {ce_symbol}/{pe_symbol}'

        # Re-snap ATM to the live synthetic forward (delta-neutral strike)
        try:
            forward = atm_strike + (ce_premium - pe_premium)
            fwd_strike = int(round(forward / strike_step) * strike_step)
            if fwd_strike != atm_strike:
                f_ce_sym = self._build_tradingsymbol('CE', fwd_strike, expiry)
                f_pe_sym = self._build_tradingsymbol('PE', fwd_strike, expiry)
                f_ce = self.scanner.get_live_option_premium(f_ce_sym)
                f_pe = self.scanner.get_live_option_premium(f_pe_sym)
                if f_ce and f_pe and f_ce > 0 and f_pe > 0:
                    logger.info(
                        f"[NAS-ATM] Forward snap: spot={spot:.1f} spot-ATM={atm_strike} "
                        f"(CE={ce_premium:.1f}/PE={pe_premium:.1f}, gap={ce_premium-pe_premium:.1f}) "
                        f"-> forward={forward:.1f} fwd-ATM={fwd_strike} "
                        f"(CE={f_ce:.1f}/PE={f_pe:.1f}, gap={f_ce-f_pe:.1f})")
                    atm_strike = fwd_strike
                    ce_symbol, pe_symbol = f_ce_sym, f_pe_sym
                    ce_premium, pe_premium = f_ce, f_pe
                else:
                    logger.warning(
                        f"[NAS-ATM] Forward snap skipped — no quote for {fwd_strike}; "
                        f"keeping spot-ATM {atm_strike}")
        except Exception as e:
            logger.warning(f"[NAS-ATM] Forward snap error ({e}); keeping spot-ATM {atm_strike}")

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
            entry_spot=spot,
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
            entry_spot=spot,
        )

        # Partial-fill rollback: if exactly one leg succeeded, close the
        # survivor to prevent a naked short. Common cause: Kite rejected the
        # second leg for insufficient margin while the first was already
        # placed. Without this, we'd carry uncovered upside/downside risk.
        if (pos_ce is None) != (pos_pe is None):
            survivor_id = pos_ce if pos_ce else pos_pe
            survivor_leg_name = 'CE' if pos_ce else 'PE'
            failed_leg_name = 'PE' if pos_ce else 'CE'
            try:
                active = self.db.get_active_positions()
                survivor_pos = next((p for p in active if p['id'] == survivor_id), None)
                if survivor_pos:
                    try:
                        live_prem = self.scanner.get_live_option_premium(survivor_pos['tradingsymbol'])
                        exit_price = live_prem if live_prem else survivor_pos.get('entry_price', 0)
                    except Exception:
                        exit_price = survivor_pos.get('entry_price', 0)
                    self._close_leg(survivor_pos, exit_price, 'PARTIAL_FILL_ROLLBACK')
                    logger.warning(
                        f"[NAS-ATM] Strangle #{strangle_id}: {failed_leg_name} leg failed -> "
                        f"closed orphan {survivor_leg_name} leg ({survivor_pos['tradingsymbol']}) "
                        f"to avoid naked short"
                    )
            except Exception as e:
                logger.error(
                    f"[NAS-ATM] Partial-fill rollback for strangle #{strangle_id} failed: {e}",
                    exc_info=True,
                )
            return None, f'{failed_leg_name} leg failed - {survivor_leg_name} rolled back'

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
                # FIX 2026-06-12: don't silently skip the SL check when the ticker's
                # live_ltps is missing this leg (subscription gap) -- fetch it via REST.
                live_prem = self.scanner.get_live_option_premium(tsym)
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

                # 2. Hold surviving leg with ST(7,2) monitoring — disable normal SL
                other_leg_type = 'PE' if pos['instrument_type'] == 'CE' else 'CE'
                other_legs = [p for p in positions
                              if p.get('strangle_id') == pos.get('strangle_id')
                              and p['instrument_type'] == other_leg_type
                              and p['status'] == 'ACTIVE']

                for other in other_legs:
                    old_sl = other.get('sl_price', 0)
                    # Disable normal SL — ST(7,2) on option premium will handle exit
                    self.db.update_position(other['id'],
                                            sl_price=999999.0,
                                            notes='st_monitoring=true')
                    logger.info(f"[NAS-ATM] Surviving {other['tradingsymbol']} "
                                f"SL disabled (was {old_sl:.2f}) — ST(7,2) monitoring active")
                    action['type'] = 'ATM_SL_NAKED'
                    action['surviving_position'] = other
                    action['trailed_leg'] = {
                        'position_id': other['id'],
                        'tradingsymbol': other['tradingsymbol'],
                        'old_sl': old_sl,
                        'new_sl': 999999.0,
                        'st_monitoring': True,
                    }

                # 3. Check if all legs of this strangle are now closed -> record trade
                # Re-read from DB to get fresh statuses (in-memory list is stale)
                fresh_strangle = self.db.get_positions_by_strangle(pos.get('strangle_id'))
                still_active = [p for p in fresh_strangle if p['status'] == 'ACTIVE']
                if not still_active:
                    # Both legs closed — record the trade
                    self._record_trade(pos.get('strangle_id'), fresh_strangle, 'SL_HIT')

                # No re-entry — surviving leg runs with ST trailing

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
        """Record a completed strangle trade when ALL legs are closed.

        Re-reads positions from DB to get fresh exit_price/status values
        (the in-memory list may be stale after _close_leg updates).
        """
        # Check for duplicate: skip if this strangle already has a trade recorded
        existing = self.db.get_recent_trades(limit=100)
        if any(t.get('strangle_id') == strangle_id for t in existing):
            logger.debug(f"[NAS-ATM] Trade already recorded for strangle #{strangle_id}, skipping")
            return

        # Only finalise once EVERY leg has actually closed. A leg still ACTIVE,
        # CLOSING, or PENDING means an exit is unfilled / still confirming at the
        # exchange — recording now would book a partial (zero-exit) trade. Defer;
        # the reconciler re-invokes this once the leg settles to CLOSED/FAILED.
        all_legs = self.db.get_positions_by_strangle(strangle_id)
        if any(p.get('status') in ('ACTIVE', 'CLOSING', 'PENDING') for p in all_legs):
            logger.debug(f"[NAS-ATM] strangle #{strangle_id} still has open legs "
                         f"-- deferring trade record")
            return

        # Re-read from DB to get fresh exit prices and statuses
        fresh_positions = self.db.get_positions_by_strangle(strangle_id)
        if not fresh_positions:
            fresh_positions = positions  # fallback

        ce_pos = [p for p in fresh_positions if p['instrument_type'] == 'CE']
        pe_pos = [p for p in fresh_positions if p['instrument_type'] == 'PE']

        call_entry = ce_pos[0]['entry_price'] if ce_pos else 0
        put_entry = pe_pos[0]['entry_price'] if pe_pos else 0
        call_exit = ce_pos[0].get('exit_price') or 0 if ce_pos else 0
        put_exit = pe_pos[0].get('exit_price') or 0 if pe_pos else 0
        call_strike = ce_pos[0]['strike'] if ce_pos else 0
        put_strike = pe_pos[0]['strike'] if pe_pos else 0

        total_collected = call_entry + put_entry
        total_paid = call_exit + put_exit
        lots = (ce_pos[0]['qty'] // LOT_SIZE) if ce_pos else (
            (pe_pos[0]['qty'] // LOT_SIZE) if pe_pos else 0)
        gross_pnl = (total_collected - total_paid) * LOT_SIZE * lots
        net_pnl = gross_pnl - (80 * 2)  # brokerage: Rs 40/order x 4 legs

        entry_time = min(p['entry_time'] for p in fresh_positions) if fresh_positions else None

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
            expiry_date=fresh_positions[0].get('expiry_date') if fresh_positions else None,
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
                'max_strangles': self.cfg.get('max_strangles', 1),
                'max_reentries': self.cfg.get('max_reentries', 5),
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
