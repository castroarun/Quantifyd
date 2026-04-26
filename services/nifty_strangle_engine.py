"""
Nifty ORB Strangle — Paper-Trading Engine (multi-variant)
==========================================================

Per-variant entry / exit / MTM lifecycle for the Nifty ORB strangle paper system.
Uses Kite live option chain for strike resolution and option pricing when
available, falls back gracefully when the chain is missing.

Public methods (per variant_id):
    run_entry_scan(variant_id) -> dict   — try to enter today
    run_mtm_tick(variant_id)   -> dict   — refresh open-position MTM
    run_exit_check(variant_id) -> dict   — check SL on the underlying
    run_eod_squareoff(variant_id) -> dict  — force-close at 15:25
    run_master_tick()          -> dict   — top-level dispatcher (all variants)
    place_live(*a, **k)        -> NotImplementedError

Strike selection runs at entry time:
    1. Pull NFO instruments → filter to NIFTY weekly Tuesday expiry (incl DTE=0)
    2. For each candidate strike in scan range, pull quote (mid + IV if any)
    3. Compute delta from BS using IV (back-implied if not in chain)
    4. Pick strike whose delta best matches each variant's target

Costs per closed trade per lot:
    gross_pnl_per_unit = (pe_credit - pe_exit) + (ce_credit - ce_exit)
    gross_pnl_per_lot  = gross * lot_size
    costs_per_lot      = (4 fills * slippage * lot) + brokerage_round_trip
                       + total_credit * stt_pct * lot
    net_pnl_per_lot    = gross_pnl - costs
"""

from __future__ import annotations

import calendar
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    STRANGLE_DEFAULTS,
    STRANGLE_VARIANTS,
    STRANGLE_VARIANTS_BY_ID,
)
from services.nifty_strangle_db import get_strangle_db
from services.nifty_strangle_scanner import (
    EntrySignal,
    check_entry_signal,
    compute_cpr,
    compute_sl_price,
    detect_or_window,
    is_sl_breached,
    load_nifty_5min_kite,
    load_nifty_5min_local,
    load_prev_day_hlc_kite,
    load_prev_day_hlc_local,
)

# Reuse Collar's already-validated BS (math.erf, no scipy)
from services.collar_engine import bs_price, _norm_cdf

logger = logging.getLogger(__name__)


NIFTY_INSTRUMENT_TOKEN = 256265   # NIFTY 50 index spot


# =============================================================================
# Greeks (for delta-based strike selection)
# =============================================================================

def bs_delta(spot: float, strike: float, t_years: float, r: float,
             iv: float, is_call: bool) -> float:
    """BS delta. Calls: 0..1. Puts: -1..0. Floors at small DTE."""
    if t_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        # At expiry, delta = 1 ITM call / -1 ITM put / 0 OTM
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    sigma_t = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / sigma_t
    if is_call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def imply_iv_from_mid(spot: float, strike: float, t_years: float,
                      r: float, mid: float, is_call: bool,
                      iv_floor: float = 0.05, iv_cap: float = 1.0) -> float:
    """Bisection-implied IV from option mid. Returns iv_default on failure."""
    if mid is None or mid <= 0 or t_years <= 0:
        return float('nan')
    lo, hi = max(iv_floor, 0.01), max(iv_cap, 1.5)
    for _ in range(40):
        mid_iv = 0.5 * (lo + hi)
        px = bs_price(spot, strike, t_years, r, mid_iv, is_call)
        if abs(px - mid) < 0.05:
            return mid_iv
        if px > mid:
            hi = mid_iv
        else:
            lo = mid_iv
    return 0.5 * (lo + hi)


# =============================================================================
# Expiry handling — nearest Tuesday weekly (or whatever Kite shows nearest)
# =============================================================================

def _next_tuesday_or_today(today: date) -> date:
    """Return today if today is Tuesday, else the next Tuesday."""
    # Mon=0, Tue=1, ..., Sun=6
    delta = (1 - today.weekday()) % 7
    return today + timedelta(days=delta)


def get_nearest_nifty_expiry(today: date = None) -> Optional[date]:
    """
    Pull Kite instruments and return the nearest NIFTY weekly expiry on or
    after today. We DO include DTE=0 (Tuesday same-day expiry) per spec.
    Falls back to next-Tuesday math if Kite is unavailable.
    """
    if today is None:
        today = date.today()
    try:
        from services.kite_service import get_kite
        kite = get_kite()
        instruments = kite.instruments('NFO')
        nifty_exp = sorted(set(
            i['expiry'] for i in instruments
            if i.get('name') == 'NIFTY' and
            i.get('instrument_type') in ('CE', 'PE') and
            i.get('expiry') is not None and
            i['expiry'] >= today
        ))
        if nifty_exp:
            return nifty_exp[0]
    except Exception as e:
        logger.debug(f"[Strangle] Kite expiry lookup failed: {e}")
    return _next_tuesday_or_today(today)


# =============================================================================
# Engine
# =============================================================================

class StrangleEngine:
    """Multi-variant Nifty ORB strangle paper engine."""

    def __init__(self, defaults: Dict = None,
                 variants: List[Dict] = None,
                 mock_provider=None):
        """
        mock_provider (optional): a duck-typed object with the methods
            get_spot_ltp() -> float
            get_intraday_5min() -> pd.DataFrame  (with rsi5m column)
            get_chain(expiry, strikes_min, strikes_max) ->
                {strike: {'CE': {'mid': x, 'iv': y, 'tradingsymbol': z},
                          'PE': {...}}}
        Used by the test harness to bypass Kite.
        """
        self.cfg = dict(defaults or STRANGLE_DEFAULTS)
        self.variants = list(variants or STRANGLE_VARIANTS)
        self.db = get_strangle_db()
        self.mock = mock_provider

        # Cache the Kite instruments list for one engine instance
        self._instruments_cache = None

        # Cache today's CPR (computed from previous-day HLC). dict[date_iso -> cpr_dict]
        self._cpr_cache: Dict[str, Optional[Dict[str, float]]] = {}

        # Persist variant configs once at boot
        for v in self.variants:
            try:
                self.db.upsert_variant(
                    variant_id=v['id'], name=v['name'],
                    config_json=json.dumps(v),
                    enabled=bool(v.get('enabled', True)),
                )
            except Exception as e:
                logger.warning(f"[Strangle] upsert_variant({v['id']}) failed: {e}")

    # -------------------------------------------------------------------------
    # Live-data helpers (Kite-aware, with mock + fallback paths)
    # -------------------------------------------------------------------------

    def _spot_ltp(self) -> Optional[float]:
        if self.mock is not None:
            try:
                return float(self.mock.get_spot_ltp())
            except Exception:
                return None
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            q = kite.ltp(['NSE:NIFTY 50'])
            return float(q['NSE:NIFTY 50']['last_price'])
        except Exception as e:
            logger.debug(f"[Strangle] spot LTP fetch failed: {e}")
            return None

    def _get_today_cpr(self, today: date) -> Optional[Dict[str, float]]:
        """Compute (and cache) today's CPR from previous-trading-day's daily HLC.
        Order of sources: mock provider, Kite daily, local market_data.db.
        Returns None if no source can supply prev-day HLC.
        """
        cache_key = today.isoformat()
        if cache_key in self._cpr_cache:
            return self._cpr_cache[cache_key]

        prev = None
        if self.mock is not None and hasattr(self.mock, 'get_prev_day_hlc'):
            try:
                prev = self.mock.get_prev_day_hlc()
            except Exception:
                prev = None
        if prev is None:
            prev = load_prev_day_hlc_kite(today)
        if prev is None:
            prev = load_prev_day_hlc_local('NIFTY50', today)

        cpr = None
        if prev is not None:
            try:
                h, l, c = prev[0], prev[1], prev[2]
                cpr = compute_cpr(float(h), float(l), float(c))
                logger.info(f"[Strangle] CPR for {cache_key}: BC={cpr['BC']:.1f} "
                            f"TC={cpr['TC']:.1f} (zone {cpr['CPR_low']:.1f}-{cpr['CPR_high']:.1f})")
            except Exception as e:
                logger.warning(f"[Strangle] compute_cpr failed: {e}")
                cpr = None
        self._cpr_cache[cache_key] = cpr
        return cpr

    def _intraday_5min(self) -> pd.DataFrame:
        if self.mock is not None:
            try:
                return self.mock.get_intraday_5min()
            except Exception:
                return pd.DataFrame()
        # Prefer live Kite intraday; fallback to local DB
        df = load_nifty_5min_kite(lookback_days=3)
        if df is None or df.empty:
            today = date.today()
            df = load_nifty_5min_local(start_date=today - timedelta(days=5),
                                       end_date=today)
        return df

    def _load_nifty_instruments(self):
        if self._instruments_cache is not None:
            return self._instruments_cache
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            insts = kite.instruments('NFO')
            self._instruments_cache = [
                i for i in insts
                if i.get('name') == 'NIFTY' and
                i.get('instrument_type') in ('CE', 'PE')
            ]
            return self._instruments_cache
        except Exception as e:
            logger.debug(f"[Strangle] Kite instruments fetch failed: {e}")
            self._instruments_cache = []
            return []

    def _resolve_tradingsymbol(self, instrument_type: str, strike: int,
                               expiry: date) -> Optional[str]:
        """Look up tradingsymbol from cache; else build a best-effort string."""
        for i in self._load_nifty_instruments():
            if (i.get('expiry') == expiry and
                    int(i.get('strike') or 0) == int(strike) and
                    i.get('instrument_type') == instrument_type):
                return i.get('tradingsymbol')
        # Best-effort fallback (Kite weekly format)
        yy = expiry.year % 100
        last_day = calendar.monthrange(expiry.year, expiry.month)[1]
        is_monthly = expiry.day > last_day - 7
        if is_monthly:
            mm = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',
                  7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}
            return f"NIFTY{yy}{mm[expiry.month]}{int(strike)}{instrument_type}"
        mc = {1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',
              7:'7',8:'8',9:'9',10:'O',11:'N',12:'D'}
        return f"NIFTY{yy}{mc[expiry.month]}{expiry.day}{int(strike)}{instrument_type}"

    def _quote_chain(self, spot: float, expiry: date) -> Dict[int, Dict]:
        """
        Pull a slice of the option chain (ATM ± strike_scan_pct of spot).
        Returns {strike: {'CE': {...}, 'PE': {...}}} where each side has:
            mid, bid, ask, ltp, iv (may be None), tradingsymbol
        """
        if self.mock is not None:
            try:
                lo = int(spot * (1 - self.cfg['strike_scan_pct']))
                hi = int(spot * (1 + self.cfg['strike_scan_pct']))
                return self.mock.get_chain(expiry, lo, hi)
            except Exception:
                return {}

        try:
            from services.kite_service import get_kite
            kite = get_kite()
        except Exception as e:
            logger.debug(f"[Strangle] Kite unavailable for chain: {e}")
            return {}

        interval = int(self.cfg.get('strike_interval', 50))
        scan_pct = float(self.cfg.get('strike_scan_pct', 0.10))
        atm = int(round(spot / interval) * interval)
        lo = atm - int((atm * scan_pct) // interval) * interval
        hi = atm + int((atm * scan_pct) // interval) * interval
        strikes = list(range(lo, hi + interval, interval))

        # Build symbols
        keys_to_strike = {}
        for s in strikes:
            for it in ('CE', 'PE'):
                tsym = self._resolve_tradingsymbol(it, s, expiry)
                if tsym:
                    keys_to_strike[f"NFO:{tsym}"] = (s, it, tsym)

        chain: Dict[int, Dict] = {s: {} for s in strikes}
        keys = list(keys_to_strike.keys())
        for batch_start in range(0, len(keys), 40):
            batch = keys[batch_start:batch_start + 40]
            try:
                quotes = kite.quote(batch)
            except Exception as e:
                logger.warning(f"[Strangle] quote batch failed: {e}")
                continue
            for k in batch:
                qd = quotes.get(k, {}) or {}
                strike, it, tsym = keys_to_strike[k]
                depth = qd.get('depth') or {}
                buy0 = (depth.get('buy') or [{}])[0]
                sell0 = (depth.get('sell') or [{}])[0]
                bid = float(buy0.get('price') or 0.0)
                ask = float(sell0.get('price') or 0.0)
                ltp = float(qd.get('last_price') or 0.0)
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                else:
                    mid = ltp if ltp > 0 else None
                # IV — Kite sometimes ships it under "implied_volatility" or in "ohlc"
                iv = None
                try:
                    ivv = qd.get('implied_volatility')
                    if ivv is not None and float(ivv) > 0:
                        iv = float(ivv) / 100.0  # Kite returns %
                except Exception:
                    pass
                chain[strike][it] = {
                    'mid': mid, 'bid': bid, 'ask': ask, 'ltp': ltp,
                    'iv': iv, 'tradingsymbol': tsym,
                }
        return chain

    # -------------------------------------------------------------------------
    # Strike selection (delta-based)
    # -------------------------------------------------------------------------

    def _pick_delta_strike(self, chain: Dict[int, Dict], spot: float,
                          expiry: date, leg_type: str,
                          delta_target: float) -> Optional[Dict]:
        """
        Pick strike whose delta best matches target. Returns:
            {'strike': int, 'mid': float, 'iv': float, 'delta': float,
             'tradingsymbol': str}
        leg_type in ('PE', 'CE'). delta_target in put/call sign convention.
        """
        if not chain:
            return None
        today = date.today()
        dte_days = max((expiry - today).days, 0)
        # If DTE=0, give it a tiny non-zero year fraction so BS is well-behaved
        # at intraday horizon (~5h).
        t_years = max(dte_days, 1) / 365.0
        if dte_days == 0:
            t_years = (5.0 / 24.0) / 365.0
        r = float(self.cfg.get('risk_free_rate', 0.065))
        iv_floor = float(self.cfg.get('iv_floor', 0.12))
        iv_cap = float(self.cfg.get('iv_cap', 0.25))
        iv_default = float(self.cfg.get('iv_default', 0.18))
        is_call = (leg_type == 'CE')

        best = None
        best_diff = float('inf')
        for strike, sides in sorted(chain.items()):
            side = sides.get(leg_type) if isinstance(sides, dict) else None
            if not side:
                continue
            mid = side.get('mid')
            iv = side.get('iv')
            if iv is None or iv <= 0:
                # Back-imply
                if mid and mid > 0:
                    iv = imply_iv_from_mid(spot, strike, t_years, r, mid,
                                           is_call, iv_floor, iv_cap)
                if iv is None or not (iv == iv) or iv <= 0:
                    iv = iv_default
            iv = max(iv_floor, min(iv_cap * 2, iv))  # sane clamp
            d = bs_delta(spot, strike, t_years, r, iv, is_call)
            diff = abs(d - delta_target)
            if diff < best_diff and mid and mid > 0:
                best_diff = diff
                best = {
                    'strike': int(strike),
                    'mid': float(mid),
                    'iv': float(iv),
                    'delta': float(d),
                    'tradingsymbol': side.get('tradingsymbol'),
                }
        return best

    # -------------------------------------------------------------------------
    # Cost calc
    # -------------------------------------------------------------------------

    def _round_trip_costs(self, lot_size: int, total_credit: float) -> float:
        slip = float(self.cfg.get('slippage_per_leg_per_side', 1.0))
        brk = float(self.cfg.get('brokerage_round_trip', 80.0))
        stt_pct = float(self.cfg.get('stt_pct_on_credit', 0.0005))
        # 4 fills * slippage * lot_size  (entry+exit, two legs)
        slip_total = 4.0 * slip * lot_size
        stt = max(total_credit, 0.0) * stt_pct * lot_size
        return round(slip_total + brk + stt, 2)

    # -------------------------------------------------------------------------
    # Entry
    # -------------------------------------------------------------------------

    def run_entry_scan(self, variant_id: str) -> Dict:
        """
        For one variant:
          1. Skip if already-open position for this variant today.
          2. Compute today's OR for the variant's window.
          3. Apply day filters (Q4, calm-only).
          4. Detect first valid break with K=0 strict RSI.
          5. Pull option chain; pick PE + CE strikes by delta target.
          6. Insert paper position + legs in DB.
        Returns a dict describing what happened.
        """
        v = STRANGLE_VARIANTS_BY_ID.get(variant_id) or \
            next((x for x in self.variants if x['id'] == variant_id), None)
        if not v:
            return {'variant_id': variant_id, 'status': 'unknown_variant'}
        if not v.get('enabled', True):
            return {'variant_id': variant_id, 'status': 'disabled'}

        today = date.today()
        today_str = today.isoformat()

        # Already entered today?
        existing = self.db.get_open_positions(variant_id)
        if existing:
            return {'variant_id': variant_id, 'status': 'already_open',
                    'positions': len(existing)}

        # Has a closed entry already today? (skip duplicate entries)
        trades_today = [t for t in self.db.get_trades(variant_id, limit=20)
                        if (t.get('entry_date') or '').startswith(today_str)]
        if trades_today:
            return {'variant_id': variant_id, 'status': 'already_traded_today',
                    'trades_today': len(trades_today)}

        df = self._intraday_5min()
        if df is None or df.empty:
            self.db.upsert_daily_state(
                variant_id=variant_id, trade_date=today_str,
                notes='no 5-min data', last_event_ts=datetime.now().isoformat())
            return {'variant_id': variant_id, 'status': 'no_data'}

        # Time guard: only enter after no_entry_after
        try:
            ne = self.cfg.get('no_entry_after', '14:00')
            h, m = map(int, ne.split(':'))
            no_entry_after = dtime(h, m)
        except Exception:
            no_entry_after = dtime(14, 0)
        if datetime.now().time() >= no_entry_after:
            return {'variant_id': variant_id, 'status': 'past_entry_window'}

        # OR detection (used for state snapshot regardless of break)
        or_res = detect_or_window(df, int(v['or_min']), today)
        if or_res is None:
            return {'variant_id': variant_id, 'status': 'or_window_incomplete'}
        or_high, or_low, or_end_ts = or_res
        or_width_pct = (or_high - or_low) / or_low * 100.0 if or_low else 0.0

        # Day filters
        passes_q4 = not (v.get('apply_q4_filter', True) and
                         or_width_pct > float(v.get('q4_threshold_pct', 0.67)))
        passes_calm = not (v.get('apply_calm_filter', False) and
                           or_width_pct >= float(v.get('calm_threshold_pct', 0.40)))
        day_ok = passes_q4 and passes_calm

        self.db.upsert_daily_state(
            variant_id=variant_id, trade_date=today_str,
            or_high=or_high, or_low=or_low, or_width_pct=or_width_pct,
            day_filter_passed=day_ok,
            last_event_ts=datetime.now().isoformat())
        if not day_ok:
            return {'variant_id': variant_id, 'status': 'day_filter_blocked',
                    'or_width_pct': or_width_pct}

        # CPR (only fetched when at least one variant needs it; cached per-day on the engine)
        cpr = self._get_today_cpr(today) if v.get('apply_cpr_against_filter', False) else None
        if v.get('apply_cpr_against_filter', False) and cpr is None:
            logger.warning(f"[Strangle:{variant_id}] CPR-against variant but no prev-day HLC; "
                           f"failsafe skip.")
            return {'variant_id': variant_id, 'status': 'no_cpr_data'}

        # Entry signal
        sig = check_entry_signal(df, v, today, no_entry_after, cpr=cpr)
        if sig is None:
            return {'variant_id': variant_id, 'status': 'no_signal',
                    'or_width_pct': or_width_pct,
                    'or_high': or_high, 'or_low': or_low}

        self.db.upsert_daily_state(
            variant_id=variant_id, trade_date=today_str,
            signal_seen=True, rsi_confirmed=True,
            last_event_ts=datetime.now().isoformat())

        # Pull chain
        spot = sig.entry_price   # use the breakout candle close as our entry "spot"
        expiry = get_nearest_nifty_expiry(today)
        if expiry is None:
            return {'variant_id': variant_id, 'status': 'no_expiry'}

        chain = self._quote_chain(spot, expiry)
        if not chain:
            logger.warning(f"[Strangle:{variant_id}] No option chain available — "
                           f"skipping entry (paper resilience).")
            return {'variant_id': variant_id, 'status': 'no_chain',
                    'expiry': expiry.isoformat()}

        # Delta targets per direction
        if sig.direction == 'LONG':
            pe_target = float(v['pe_delta_target_long'])
            ce_target = float(v['ce_delta_target_long'])
        else:
            pe_target = float(v['pe_delta_target_short'])
            ce_target = float(v['ce_delta_target_short'])

        pe = self._pick_delta_strike(chain, spot, expiry, 'PE', pe_target)
        ce = self._pick_delta_strike(chain, spot, expiry, 'CE', ce_target)
        if not pe or not ce:
            return {'variant_id': variant_id, 'status': 'strike_not_found',
                    'pe': pe, 'ce': ce}

        sl_price = compute_sl_price(sig.direction, sig.or_high, sig.or_low,
                                    float(self.cfg.get('sl_or_width_multiplier', 1.0)))

        lot_size = int(v.get('lot_size') or self.cfg.get('lot_size', 65))
        position_id = self.db.open_position(
            variant_id=variant_id,
            entry_date=today_str,
            entry_ts=sig.entry_ts.isoformat(),
            direction=sig.direction,
            spot_at_entry=spot,
            or_high=sig.or_high, or_low=sig.or_low,
            or_width_pct=sig.or_width_pct,
            sl_price=sl_price,
            expiry_date=expiry.isoformat(),
            pe_strike=pe['strike'], ce_strike=ce['strike'],
            pe_entry_price=pe['mid'], ce_entry_price=ce['mid'],
            pe_delta=pe['delta'], ce_delta=ce['delta'],
            lot_size=lot_size,
            pe_tradingsymbol=pe.get('tradingsymbol'),
            ce_tradingsymbol=ce.get('tradingsymbol'),
            pe_iv=pe.get('iv'), ce_iv=ce.get('iv'),
        )

        self.db.upsert_daily_state(
            variant_id=variant_id, trade_date=today_str,
            entry_taken=True,
            last_event_ts=datetime.now().isoformat())

        logger.info(
            f"[Strangle:{variant_id}] PAPER OPEN dir={sig.direction} "
            f"spot={spot:.2f} OR=[{sig.or_low:.2f},{sig.or_high:.2f}] "
            f"PE{pe['strike']}@{pe['mid']:.2f} (Δ={pe['delta']:.3f}) "
            f"CE{ce['strike']}@{ce['mid']:.2f} (Δ={ce['delta']:.3f}) "
            f"SL={sl_price:.2f} expiry={expiry}"
        )
        return {
            'variant_id': variant_id, 'status': 'opened',
            'position_id': position_id,
            'direction': sig.direction,
            'entry_ts': sig.entry_ts.isoformat(),
            'spot': spot,
            'pe': pe, 'ce': ce,
            'sl_price': sl_price,
            'expiry': expiry.isoformat(),
        }

    # -------------------------------------------------------------------------
    # MTM tick
    # -------------------------------------------------------------------------

    def run_mtm_tick(self, variant_id: str) -> Dict:
        """Refresh option mids on each open position; persist to legs.mtm_now.
        Doesn't close anything — that's `run_exit_check`."""
        positions = self.db.get_open_positions(variant_id)
        if not positions:
            return {'variant_id': variant_id, 'updated': 0}

        spot = self._spot_ltp()
        updated = 0
        for pos in positions:
            try:
                expiry = date.fromisoformat(pos['expiry_date'])
                chain = self._quote_chain(spot or pos['spot_at_entry'], expiry)
                pe_q = chain.get(int(pos['pe_strike']), {}).get('PE') if chain else None
                ce_q = chain.get(int(pos['ce_strike']), {}).get('CE') if chain else None
                if pe_q and pe_q.get('mid'):
                    self.db.update_leg_mtm(pos['id'], 'PE', float(pe_q['mid']))
                    updated += 1
                if ce_q and ce_q.get('mid'):
                    self.db.update_leg_mtm(pos['id'], 'CE', float(ce_q['mid']))
                    updated += 1
            except Exception as e:
                logger.warning(f"[Strangle:{variant_id}] MTM tick err pos={pos['id']}: {e}")
        return {'variant_id': variant_id, 'updated': updated, 'spot': spot}

    # -------------------------------------------------------------------------
    # Exit check (SL on underlying)
    # -------------------------------------------------------------------------

    def run_exit_check(self, variant_id: str) -> Dict:
        positions = self.db.get_open_positions(variant_id)
        if not positions:
            return {'variant_id': variant_id, 'closed': 0}
        spot = self._spot_ltp()
        if spot is None:
            return {'variant_id': variant_id, 'closed': 0, 'reason': 'no_spot'}
        closed = 0
        results = []
        for pos in positions:
            try:
                if is_sl_breached(pos['direction'], pos['sl_price'], spot):
                    res = self._close_position(pos, exit_reason='SL',
                                               spot_now=spot)
                    if res:
                        closed += 1
                        results.append(res)
            except Exception as e:
                logger.error(f"[Strangle:{variant_id}] exit check err pos={pos['id']}: {e}")
        return {'variant_id': variant_id, 'closed': closed,
                'results': results, 'spot': spot}

    def run_eod_squareoff(self, variant_id: str) -> Dict:
        positions = self.db.get_open_positions(variant_id)
        if not positions:
            return {'variant_id': variant_id, 'closed': 0}
        spot = self._spot_ltp()
        closed = 0
        results = []
        for pos in positions:
            try:
                res = self._close_position(pos, exit_reason='EOD',
                                           spot_now=spot or pos['spot_at_entry'])
                if res:
                    closed += 1
                    results.append(res)
            except Exception as e:
                logger.error(f"[Strangle:{variant_id}] EOD close err pos={pos['id']}: {e}")
        return {'variant_id': variant_id, 'closed': closed, 'results': results}

    def _close_position(self, pos: Dict, exit_reason: str,
                        spot_now: float) -> Optional[Dict]:
        """Pull current mids → DB.close_position with cost calc."""
        expiry = date.fromisoformat(pos['expiry_date'])
        chain = self._quote_chain(spot_now, expiry)

        pe_q = chain.get(int(pos['pe_strike']), {}).get('PE') if chain else None
        ce_q = chain.get(int(pos['ce_strike']), {}).get('CE') if chain else None

        # Fallback to BS-implied price using stored entry IV if mid missing
        today = date.today()
        dte_days = max((expiry - today).days, 0)
        t_years = max(dte_days, 1) / 365.0
        if dte_days == 0:
            # Tighten t_years toward zero as the day progresses (rough)
            t_years = (1.0 / 24.0) / 365.0
        r = float(self.cfg.get('risk_free_rate', 0.065))

        if pe_q and pe_q.get('mid') and pe_q['mid'] > 0:
            pe_exit = float(pe_q['mid'])
        else:
            iv = self._lookup_leg_iv(pos['id'], 'PE') or float(self.cfg.get('iv_default', 0.18))
            pe_exit = bs_price(spot_now, pos['pe_strike'], t_years, r, iv, is_call=False)

        if ce_q and ce_q.get('mid') and ce_q['mid'] > 0:
            ce_exit = float(ce_q['mid'])
        else:
            iv = self._lookup_leg_iv(pos['id'], 'CE') or float(self.cfg.get('iv_default', 0.18))
            ce_exit = bs_price(spot_now, pos['ce_strike'], t_years, r, iv, is_call=True)

        total_credit = float(pos['pe_entry_price']) + float(pos['ce_entry_price'])
        costs = self._round_trip_costs(int(pos['lot_size']), total_credit)

        result = self.db.close_position(
            position_id=pos['id'],
            exit_date=date.today().isoformat(),
            exit_ts=datetime.now().isoformat(),
            pe_exit_price=pe_exit, ce_exit_price=ce_exit,
            spot_at_exit=spot_now, exit_reason=exit_reason,
            costs=costs,
        )
        self.db.upsert_daily_state(
            variant_id=pos['variant_id'], trade_date=date.today().isoformat(),
            exit_taken=True, exit_reason=exit_reason,
            last_event_ts=datetime.now().isoformat())

        logger.info(
            f"[Strangle:{pos['variant_id']}] PAPER CLOSE pos={pos['id']} "
            f"reason={exit_reason} PE_exit={pe_exit:.2f} CE_exit={ce_exit:.2f} "
            f"net={result['net_pnl']:+.2f} (costs Rs{costs:.2f})"
        )
        return result

    def _lookup_leg_iv(self, position_id: int, leg_type: str) -> Optional[float]:
        pos = self.db.get_position(position_id)
        if not pos:
            return None
        for l in pos.get('legs', []):
            if l['leg_type'] == leg_type:
                return l.get('iv_at_entry')
        return None

    # -------------------------------------------------------------------------
    # Master tick (one-job dispatcher)
    # -------------------------------------------------------------------------

    def run_master_tick(self) -> Dict:
        """
        Cron-friendly master dispatcher; called every 60s by app.py.
          - For each variant: if its OR window has closed AND clock is before
            the strategy's no_entry_after AND no entry today yet → entry scan.
          - For every open position across all variants → MTM tick + exit check.
          - At exactly 15:25 (or later) → EOD square-off everything.
        Returns a summary dict for logging.
        """
        now = datetime.now()
        today_dt = now.date()
        # Skip weekends
        if today_dt.weekday() >= 5:
            return {'status': 'weekend'}

        try:
            ne = self.cfg.get('no_entry_after', '14:00')
            h, m = map(int, ne.split(':'))
            no_entry_after = dtime(h, m)
            eod = self.cfg.get('eod_squareoff_time', '15:25')
            eh, em = map(int, eod.split(':'))
            eod_time = dtime(eh, em)
        except Exception:
            no_entry_after, eod_time = dtime(14, 0), dtime(15, 25)

        summary = {'now': now.isoformat(), 'variants': {}}

        # 1. EOD square-off first if past EOD time
        if now.time() >= eod_time:
            for v in self.variants:
                r = self.run_eod_squareoff(v['id'])
                summary['variants'][v['id']] = {'eod': r}
            return summary

        # 2. MTM + exit check on open positions
        for v in self.variants:
            slot = summary['variants'].setdefault(v['id'], {})
            try:
                slot['mtm'] = self.run_mtm_tick(v['id'])
            except Exception as e:
                logger.error(f"[Strangle:{v['id']}] MTM tick err: {e}")
                slot['mtm'] = {'error': str(e)}
            try:
                slot['exit'] = self.run_exit_check(v['id'])
            except Exception as e:
                logger.error(f"[Strangle:{v['id']}] exit check err: {e}")
                slot['exit'] = {'error': str(e)}

        # 3. Entry scan if within window
        if now.time() < no_entry_after:
            for v in self.variants:
                try:
                    slot = summary['variants'].setdefault(v['id'], {})
                    # Only scan if the OR window has closed for this variant
                    or_close_min = 9 * 60 + 15 + int(v['or_min'])
                    or_close_t = dtime(or_close_min // 60, or_close_min % 60)
                    if now.time() >= or_close_t:
                        slot['entry'] = self.run_entry_scan(v['id'])
                except Exception as e:
                    logger.error(f"[Strangle:{v['id']}] entry scan err: {e}")
                    slot = summary['variants'].setdefault(v['id'], {})
                    slot['entry'] = {'error': str(e)}

        return summary

    # -------------------------------------------------------------------------
    # MTM snapshot for dashboard
    # -------------------------------------------------------------------------

    def snapshot_position_mtm(self, position: Dict) -> Dict:
        """Compute live MTM in INR per leg + total. For dashboard rendering."""
        try:
            qty = int(position.get('qty') or position.get('lot_size'))
            pe_now = None
            ce_now = None
            for l in position.get('legs', []):
                if l['leg_type'] == 'PE':
                    pe_now = l.get('mtm_now')
                elif l['leg_type'] == 'CE':
                    ce_now = l.get('mtm_now')
            pe_now = pe_now if pe_now is not None else position['pe_entry_price']
            ce_now = ce_now if ce_now is not None else position['ce_entry_price']
            pe_mtm = round((position['pe_entry_price'] - pe_now) * qty, 2)
            ce_mtm = round((position['ce_entry_price'] - ce_now) * qty, 2)
            total = round(pe_mtm + ce_mtm, 2)
            return {
                'pe_now': float(pe_now), 'ce_now': float(ce_now),
                'pe_mtm': pe_mtm, 'ce_mtm': ce_mtm, 'net_mtm': total,
            }
        except Exception:
            return {'pe_now': None, 'ce_now': None,
                    'pe_mtm': None, 'ce_mtm': None, 'net_mtm': None}

    # -------------------------------------------------------------------------
    # Live trading — deliberately disabled
    # -------------------------------------------------------------------------

    def place_live(self, *args, **kwargs):
        raise NotImplementedError(
            "StrangleEngine is paper-only. No live order routes exist."
        )


# Singleton accessor (one engine for the whole app) ------------------------
_engine_instance = None
_engine_lock = None


def get_strangle_engine() -> StrangleEngine:
    global _engine_instance, _engine_lock
    if _engine_lock is None:
        import threading
        _engine_lock = threading.Lock()
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = StrangleEngine()
    return _engine_instance
