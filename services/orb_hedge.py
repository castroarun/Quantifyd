"""
ORB Tail Hedge — NIFTY OTM option bought on heavy directional imbalance
========================================================================

When the ORB book has many concurrent same-direction positions, a sharp
intraday reversal hits all legs together and can blow through per-leg
SLs with slippage. A small NIFTY OTM option (opposite direction of the
book's skew) is a cheap tail hedge that caps the "everything reverses"
scenario.

Trigger (runs every 30s inside monitor_positions, between
`hedge_eval_start` and `hedge_eval_end`):

  skew = |short_notional - long_notional| / total_notional  >= 0.70
  active ORB positions                                      >= 10

On first trigger in the eval window:
  net SHORT book → BUY NIFTY weekly OTM CE (1.5% above spot)
  net LONG  book → BUY NIFTY weekly OTM PE (1.5% below spot)

One-shot per day. Held till `hedge_exit_time` (15:15 IST). Not re-armed
if ORB positions exit early or skew changes.

Paper mode (`hedge_paper_mode=True` in ORB_DEFAULTS, default for v1):
  compute the trigger, log the would-have-been order, persist a row in
  orb_hedges, but do NOT place a real Kite order. Gives a week of
  shadow data before going live.
"""

import logging
from datetime import datetime, date, time as dtime
from typing import Optional

from services.orb_db import get_orb_db

logger = logging.getLogger(__name__)


def _parse_hhmm(s: str, now: datetime) -> datetime:
    """Return a datetime at today's HH:MM local time."""
    h, m = [int(x) for x in s.split(':')]
    return now.replace(hour=h, minute=m, second=0, microsecond=0)


class OrbTailHedge:
    """Stateful helper. One instance per ORBLiveEngine (attached as
    `engine._hedge`). Shares the engine's _hedge_fired_today /
    _hedge_record flags."""

    def __init__(self, engine):
        self.engine = engine
        self.cfg = engine.cfg
        self.db = engine.db
        # Ensure the hedge table exists
        try:
            self._ensure_table()
        except Exception as e:
            logger.warning(f"[HEDGE] table ensure failed: {e}")

    # ─── Schema ────────────────────────────────────────────────

    def _ensure_table(self):
        """Create orb_hedges table if missing. Idempotent."""
        conn = self.db._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS orb_hedges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE NOT NULL,
                    fire_time TIMESTAMP NOT NULL,
                    direction TEXT NOT NULL,           -- CE or PE
                    trigger_reason TEXT,
                    net_side TEXT,                     -- SHORT or LONG (the book's net side)
                    skew REAL,
                    positions_count INTEGER,
                    spot REAL,
                    strike REAL,
                    expiry DATE,
                    tradingsymbol TEXT,
                    lots INTEGER,
                    qty INTEGER,
                    entry_premium REAL,
                    exit_premium REAL,
                    exit_time TIMESTAMP,
                    pnl_inr REAL,
                    status TEXT DEFAULT 'OPEN',        -- OPEN / EXITED / PAPER
                    paper_mode INTEGER DEFAULT 0,
                    kite_entry_order_id TEXT,
                    kite_exit_order_id TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_orb_hedges_date
                    ON orb_hedges(trade_date);
                CREATE INDEX IF NOT EXISTS idx_orb_hedges_status
                    ON orb_hedges(status);
            """)
            conn.commit()
        finally:
            conn.close()

    # ─── Core evaluation ───────────────────────────────────────

    def check_and_place(self, open_positions, ltps, now: Optional[datetime] = None):
        """Called every monitor tick. Decides whether to fire the hedge.

        Parameters
        ----------
        open_positions : list[dict]
            Snapshot of currently-open ORB positions (from `engine._positions`).
        ltps : dict[str, float]
            Live LTPs keyed by instrument symbol.
        now : datetime, optional
            Wall-clock. Defaults to datetime.now().
        """
        if not self.cfg.get('hedge_enabled', True):
            return
        if self.engine._hedge_fired_today:
            return
        if not open_positions:
            return

        now = now or datetime.now()
        start_dt = _parse_hhmm(self.cfg.get('hedge_eval_start', '10:00'), now)
        end_dt = _parse_hhmm(self.cfg.get('hedge_eval_end', '14:00'), now)
        if now < start_dt or now > end_dt:
            return

        min_positions = int(self.cfg.get('hedge_min_positions', 10))
        if len(open_positions) < min_positions:
            return

        # Compute skew on notional (entry × qty)
        short_notional = 0.0
        long_notional = 0.0
        for p in open_positions:
            notional = float(p['entry_price']) * float(p['qty'])
            if p['direction'] == 'SHORT':
                short_notional += notional
            else:
                long_notional += notional
        total = short_notional + long_notional
        if total <= 0:
            return

        skew = abs(short_notional - long_notional) / total
        threshold = float(self.cfg.get('hedge_skew_threshold', 0.70))
        if skew < threshold:
            return

        # All gates pass — place the hedge
        net_side = 'SHORT' if short_notional > long_notional else 'LONG'
        try:
            self._fire_hedge(
                net_side=net_side, skew=skew,
                positions_count=len(open_positions), now=now,
            )
        except Exception as e:
            logger.error(f"[HEDGE] fire failed: {e}", exc_info=True)

    def _fire_hedge(self, net_side, skew, positions_count, now):
        """Place the hedge order (or log in paper mode)."""
        # 1) Resolve NIFTY spot
        from services.nas_scanner import NasScanner, get_current_week_expiry
        scanner = getattr(self.engine, '_hedge_scanner', None)
        if scanner is None:
            scanner = NasScanner(self.cfg)
            self.engine._hedge_scanner = scanner
        spot = scanner.get_live_spot()
        if not spot:
            logger.warning("[HEDGE] could not resolve NIFTY spot; skipping")
            return

        # 2) Strike selection
        otm_pct = float(self.cfg.get('hedge_otm_pct', 0.015))
        if net_side == 'SHORT':
            instrument_type = 'CE'
            raw_strike = spot * (1.0 + otm_pct)
        else:
            instrument_type = 'PE'
            raw_strike = spot * (1.0 - otm_pct)
        strike_step = self.cfg.get('strike_interval', 50)
        strike = int(round(raw_strike / strike_step) * strike_step)

        # 3) Expiry — nearest with ≥ 2 DTE to avoid instant-theta on expiry day
        today = now.date()
        expiry = get_current_week_expiry(today)
        if expiry and (expiry - today).days < 2:
            # Skip current week, grab the next
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                nifty_expiries = sorted(set(
                    i['expiry'] for i in kite.instruments('NFO')
                    if i['name'] == 'NIFTY' and i['instrument_type'] in ('CE', 'PE')
                    and i['expiry'] > expiry
                ))
                if nifty_expiries:
                    expiry = nifty_expiries[0]
            except Exception as e:
                logger.warning(f"[HEDGE] next-expiry lookup failed: {e}")

        # 4) Build tradingsymbol
        try:
            tradingsymbol = scanner._build_tradingsymbol(instrument_type, strike, expiry)
        except Exception as e:
            logger.error(f"[HEDGE] tradingsymbol build failed: {e}")
            return

        # 5) Get entry premium (best-effort)
        entry_premium = None
        try:
            entry_premium = scanner.get_live_option_premium(tradingsymbol)
        except Exception:
            pass

        # 6) Size
        lots = int(self.cfg.get('hedge_lots', 1))
        lot_size = int(self.cfg.get('lot_size', 75))
        qty = lots * lot_size

        paper_mode = bool(self.cfg.get('hedge_paper_mode', True))
        trigger_reason = (
            f"skew={skew:.2f}>={self.cfg.get('hedge_skew_threshold', 0.70):.2f} "
            f"count={positions_count}>={self.cfg.get('hedge_min_positions', 10)}"
        )

        # 7) Place order (or log in paper mode)
        kite_order_id = None
        status = 'PAPER' if paper_mode else 'OPEN'
        if not paper_mode:
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                # LIMIT @ ~1% above last known premium for fast fill; fall back to MARKET if unknown
                if entry_premium and entry_premium > 0:
                    limit_px = round(entry_premium * 1.01, 1)
                    order_id = kite.place_order(
                        variety='regular', exchange='NFO',
                        tradingsymbol=tradingsymbol,
                        transaction_type='BUY', quantity=qty,
                        product='MIS', order_type='LIMIT', price=limit_px,
                    )
                else:
                    order_id = kite.place_order(
                        variety='regular', exchange='NFO',
                        tradingsymbol=tradingsymbol,
                        transaction_type='BUY', quantity=qty,
                        product='MIS', order_type='MARKET',
                    )
                kite_order_id = str(order_id)
                logger.critical(
                    f"[HEDGE] LIVE HEDGE PLACED: BUY {qty} {tradingsymbol} "
                    f"(~Rs {entry_premium or '??'}/contract) · {trigger_reason} · "
                    f"spot={spot:.1f} order_id={kite_order_id}"
                )
            except Exception as e:
                logger.error(f"[HEDGE] LIVE hedge order failed: {e}", exc_info=True)
                return

        # 8) Persist record
        try:
            conn = self.db._get_conn()
            cur = conn.execute(
                """INSERT INTO orb_hedges
                   (trade_date, fire_time, direction, trigger_reason, net_side,
                    skew, positions_count, spot, strike, expiry, tradingsymbol,
                    lots, qty, entry_premium, status, paper_mode,
                    kite_entry_order_id, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (today.isoformat(), now.isoformat(), instrument_type,
                 trigger_reason, net_side, round(skew, 4), positions_count,
                 round(spot, 2), strike,
                 expiry.isoformat() if expiry else None, tradingsymbol,
                 lots, qty, round(entry_premium, 2) if entry_premium else None,
                 status, 1 if paper_mode else 0, kite_order_id,
                 f"v1 hedge · paper={paper_mode}")
            )
            conn.commit()
            hedge_id = cur.lastrowid
            conn.close()
        except Exception as e:
            logger.error(f"[HEDGE] DB insert failed: {e}")
            hedge_id = None

        # 9) Mark fired + cache record
        self.engine._hedge_fired_today = True
        self.engine._hedge_record = {
            'id': hedge_id,
            'tradingsymbol': tradingsymbol,
            'qty': qty, 'lots': lots,
            'strike': strike, 'expiry': expiry.isoformat() if expiry else None,
            'direction': instrument_type,
            'entry_premium': entry_premium,
            'paper_mode': paper_mode,
            'kite_entry_order_id': kite_order_id,
        }

        # 10) Notify
        try:
            from services.notifications import get_notification_service
            ns = get_notification_service(self.cfg)
            tag = 'PAPER' if paper_mode else 'LIVE'
            ns.send_alert(
                'risk_alert',
                f'ORB tail hedge fired ({tag})',
                f"{tag}: BUY 1 lot {tradingsymbol} · spot={spot:.1f} "
                f"strike={strike} ({instrument_type}) · {trigger_reason} · "
                f"entry_premium~Rs {entry_premium or '??'}. Will square off at "
                f"{self.cfg.get('hedge_exit_time', '15:15')}.",
                data=self.engine._hedge_record, priority='high',
            )
        except Exception:
            pass

    # ─── Exit at end of day ────────────────────────────────────

    def check_and_exit(self, now: Optional[datetime] = None):
        """Called every monitor tick. Exit the hedge if past exit_time."""
        if not self.engine._hedge_record:
            return
        # If already flagged exited, do nothing
        if self.engine._hedge_record.get('exited'):
            return
        now = now or datetime.now()
        exit_dt = _parse_hhmm(self.cfg.get('hedge_exit_time', '15:15'), now)
        if now < exit_dt:
            return

        record = self.engine._hedge_record
        paper_mode = record.get('paper_mode', True)
        tradingsymbol = record['tradingsymbol']
        qty = record['qty']

        from services.nas_scanner import NasScanner
        scanner = getattr(self.engine, '_hedge_scanner', None) or NasScanner(self.cfg)
        exit_premium = None
        try:
            exit_premium = scanner.get_live_option_premium(tradingsymbol)
        except Exception:
            pass

        kite_exit_order_id = None
        if not paper_mode:
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                order_id = kite.place_order(
                    variety='regular', exchange='NFO',
                    tradingsymbol=tradingsymbol,
                    transaction_type='SELL', quantity=qty,
                    product='MIS', order_type='MARKET',
                )
                kite_exit_order_id = str(order_id)
                logger.info(
                    f"[HEDGE] LIVE EXIT placed: SELL {qty} {tradingsymbol} "
                    f"(premium ~Rs {exit_premium}) order_id={kite_exit_order_id}"
                )
            except Exception as e:
                logger.error(f"[HEDGE] LIVE hedge exit failed: {e}")

        # Compute hedge P&L (long option)
        pnl_inr = None
        entry = record.get('entry_premium')
        if entry is not None and exit_premium is not None:
            pnl_inr = round((exit_premium - entry) * qty, 2)

        # Persist exit
        try:
            if record.get('id'):
                conn = self.db._get_conn()
                conn.execute(
                    """UPDATE orb_hedges
                       SET exit_premium=?, exit_time=?, pnl_inr=?, status=?,
                           kite_exit_order_id=?
                       WHERE id=?""",
                    (round(exit_premium, 2) if exit_premium else None,
                     now.isoformat(), pnl_inr,
                     'PAPER_EXITED' if paper_mode else 'EXITED',
                     kite_exit_order_id, record['id'])
                )
                conn.commit()
                conn.close()
        except Exception as e:
            logger.error(f"[HEDGE] exit DB update failed: {e}")

        record['exited'] = True
        record['exit_premium'] = exit_premium
        record['pnl_inr'] = pnl_inr

        try:
            from services.notifications import get_notification_service
            ns = get_notification_service(self.cfg)
            tag = 'PAPER' if paper_mode else 'LIVE'
            ns.send_alert(
                'risk_alert',
                f'ORB tail hedge exited ({tag})',
                f"{tag}: SELL {qty} {tradingsymbol} · entry={entry} exit={exit_premium} "
                f"· pnl=Rs {pnl_inr}",
                data=record, priority='normal',
            )
        except Exception:
            pass
