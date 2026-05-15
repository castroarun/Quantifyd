"""
F&O Live Scanner Service
=========================

For each of the ~81 F&O stocks, computes a live setup snapshot:

  - spot / LTP + day-change %
  - today's volume-surge ratio (today vol vs trailing-20-day avg daily vol;
    during market hours uses live cumulative volume via kite.quote)
  - current-week weekly-CPR width % (reuses the CPR formula from
    cpr_covered_call_service) + a narrow flag relative to a caller-supplied
    threshold
  - daily-trend state (close vs SMA50 and SMA200 -> up / down / flat)
  - prev-day-range and prev-week-range break state ("escaped" only when
    price is beyond the prior WEEK range)
  - clean directional first-candle flag (today's first daily candle if
    available)
  - composite 0-100 setup score = weighted blend of:
        volume-surge ............ 35
        CPR-narrowness vs thr ... 20
        trend + range-escape
          direction agreement ... 25
        clean candle ............ 20

The heavy daily-derived parts (CPR, SMA50/200, prev-week range, trailing
20-day avg volume) are cached per trading day. Only LTP / cumulative volume
refresh live.

Module exposes a cached singleton via get_scanner_service().
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from services.data_manager import FNO_LOT_SIZES, get_data_manager

logger = logging.getLogger(__name__)

# Score weights (sum = 100)
W_VOLUME = 35.0
W_CPR = 20.0
W_TREND_RANGE = 25.0
W_CLEAN = 20.0


def _is_market_open(now: Optional[datetime] = None) -> bool:
    """NSE cash/F&O session 09:15-15:30 IST Mon-Fri. Server runs in IST."""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


def _weekly_cpr_width_pct(daily_df: pd.DataFrame, today: datetime) -> Optional[float]:
    """
    Current week's CPR derived from the PREVIOUS week's H/L/C.

    Reuses the formula from cpr_covered_call_service._calculate_weekly_cpr:
        pivot = (H + L + C) / 3
        bc    = (H + L) / 2
        tc    = 2*pivot - bc
        width = abs(tc - bc)
        width_pct = width / pivot * 100
    """
    week_start = today - timedelta(days=today.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    prev_week_start = week_start - timedelta(days=7)
    prev_week_end = week_start - timedelta(days=1)

    idx = daily_df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    mask = (idx >= prev_week_start) & (idx <= prev_week_end)
    prev = daily_df[mask]
    if prev.empty or len(prev) < 3:
        return None

    ph = float(prev['high'].max())
    pl = float(prev['low'].min())
    pc = float(prev['close'].iloc[-1])
    pivot = (ph + pl + pc) / 3.0
    if pivot == 0:
        return None
    bc = (ph + pl) / 2.0
    tc = 2.0 * pivot - bc
    width = abs(tc - bc)
    return width / pivot * 100.0


def _intraday_open_facts(dm, symbol: str, ref_date: datetime,
                         tf_minutes: int = 30, n_baseline: int = 20) -> Optional[dict]:
    """Spec-faithful opening-candle facts from 5-min data:

      - clean: 'long'|'short'|'none' for the OPENING `tf_minutes` candle of
        the most recent session (body >= 60% of range, in that direction)
      - vol_surge_open: that opening candle's volume / mean of the prior
        `n_baseline` sessions' OPENING-candle volumes  (this is the spec
        volume surge — opening-candle vs same-slot baseline, NOT full day)
      - last_5m_close / asof: freshest intraday price + its timestamp
        (so a stale name like BAJAJ-AUTO is visibly stale, not faked)

    Returns None when the symbol has no intraday 5-min data.
    """
    try:
        df5 = dm.load_data(symbol, '5minute',
                           ref_date - timedelta(days=60), ref_date)
    except Exception:
        return None
    if df5 is None or df5.empty:
        return None
    df5 = df5.sort_index()
    idx = df5.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
        df5.index = idx

    # opening tf_minutes volume per session (and OHLC for the last session)
    day_norm = idx.normalize()
    sessions = list(pd.unique(day_norm))
    if not sessions:
        return None
    open_vol_by_session = []
    last_ohlc = None
    for d in sessions:
        sb = df5[day_norm == d]
        if sb.empty:
            continue
        win_end = sb.index[0] + pd.Timedelta(minutes=tf_minutes)
        first = sb[sb.index < win_end]
        if first.empty:
            continue
        ov = float(first['volume'].sum())
        open_vol_by_session.append(ov)
        last_ohlc = (
            float(first['open'].iloc[0]), float(first['high'].max()),
            float(first['low'].min()), float(first['close'].iloc[-1]),
        )
    if last_ohlc is None or len(open_vol_by_session) < 2:
        return None

    cur_open_vol = open_vol_by_session[-1]
    prior = open_vol_by_session[-(n_baseline + 1):-1]
    base = sum(prior) / len(prior) if prior else None
    vol_surge_open = (round(cur_open_vol / base, 2)
                      if base and base > 0 else None)

    o, h, lo, c = last_ohlc
    rng = h - lo
    if rng <= 0:
        clean = 'none'
    else:
        bf = abs(c - o) / rng
        clean = ('long' if (c > o and bf >= 0.6)
                 else 'short' if (c < o and bf >= 0.6) else 'none')

    return {
        'clean': clean,
        'vol_surge_open': vol_surge_open,
        'last_5m_close': float(df5['close'].iloc[-1]),
        'asof': idx[-1].to_pydatetime(),
    }


def _prev_week_range(daily_df: pd.DataFrame, today: datetime):
    week_start = today - timedelta(days=today.weekday())
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    prev_week_start = week_start - timedelta(days=7)
    prev_week_end = week_start - timedelta(days=1)
    idx = daily_df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
    mask = (idx >= prev_week_start) & (idx <= prev_week_end)
    prev = daily_df[mask]
    if prev.empty:
        return None, None
    return float(prev['high'].max()), float(prev['low'].min())


class ScannerService:
    """Computes live F&O setup snapshots. Daily-derived parts cached per day."""

    def __init__(self):
        self.db_path = Path(__file__).parent.parent / 'backtest_data' / 'market_data.db'
        self._dm = get_data_manager()
        self._lock = threading.Lock()
        # daily cache keyed by trade-date string -> { symbol -> daily_facts }
        self._daily_cache_date: Optional[str] = None
        self._daily_cache: Dict[str, dict] = {}
        self.symbols: List[str] = sorted(FNO_LOT_SIZES.keys())

    # ---------- daily-derived (cached per day) ----------

    def _daily_facts(self, symbol: str, today: datetime) -> Optional[dict]:
        """SMA50/200, CPR width, prev-day & prev-week range, trailing-20 vol,
        clean first-candle. Cached per trading day."""
        try:
            to_date = today
            from_date = today - timedelta(days=420)
            df = self._dm.load_data(symbol, 'day', from_date, to_date)
        except Exception:
            return None
        if df is None or len(df) < 60:
            return None

        df = df.sort_index()
        close = df['close']
        sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None
        last_close = float(close.iloc[-1])

        # trailing-20-day average DAILY volume (exclude the most recent bar so
        # "today" can be compared against the prior baseline)
        vol = df['volume']
        avg20_vol = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
        last_bar_vol = float(vol.iloc[-1])

        # previous trading day's high/low
        if len(df) >= 2:
            prev_day_high = float(df['high'].iloc[-2])
            prev_day_low = float(df['low'].iloc[-2])
        else:
            prev_day_high = prev_day_low = None

        pw_high, pw_low = _prev_week_range(df, today)
        cpr_w = _weekly_cpr_width_pct(df, today)

        # today's first (and so far only daily) candle, if the DB has a bar
        # dated today; treat it as the "first candle" proxy on a daily TF
        last_date = df.index[-1]
        is_today_bar = (
            isinstance(last_date, (pd.Timestamp, datetime))
            and last_date.date() == today.date()
        )
        # Spec-faithful intraday: opening 30-min candle clean flag +
        # opening-candle volume surge (NOT full-day) + freshest 5-min price.
        intra = _intraday_open_facts(self._dm, symbol, today, 30, 20)
        clean_candle = intra['clean'] if intra else None
        vol_surge_open = intra['vol_surge_open'] if intra else None
        last_5m_close = intra['last_5m_close'] if intra else None
        intra_asof = intra['asof'] if intra else None
        if clean_candle is None and is_today_bar:  # daily-bar fallback only
            o = float(df['open'].iloc[-1])
            h = float(df['high'].iloc[-1])
            lo = float(df['low'].iloc[-1])
            c = float(df['close'].iloc[-1])
            rng = h - lo
            if rng > 0:
                body_frac = abs(c - o) / rng
                clean_candle = ('long' if (c > o and body_frac >= 0.6)
                                else 'short' if (c < o and body_frac >= 0.6)
                                else 'none')

        # prior daily close (for a real off-hours day-change %)
        prev_daily_close = float(close.iloc[-2]) if len(close) >= 2 else None
        last_daily_date = df.index[-1]
        if isinstance(last_daily_date, (pd.Timestamp, datetime)):
            last_daily_date = last_daily_date.to_pydatetime() if hasattr(
                last_daily_date, 'to_pydatetime') else last_daily_date

        return {
            'sma50': sma50,
            'sma200': sma200,
            'last_daily_close': last_close,
            'prev_daily_close': prev_daily_close,
            'last_daily_date': last_daily_date,
            'avg20_vol': avg20_vol,
            'last_bar_vol': last_bar_vol,
            'prev_day_high': prev_day_high,
            'prev_day_low': prev_day_low,
            'prev_week_high': pw_high,
            'prev_week_low': pw_low,
            'cpr_width_pct': cpr_w,
            'clean_candle': clean_candle,
            'vol_surge_open': vol_surge_open,
            'last_5m_close': last_5m_close,
            'intra_asof': intra_asof,
            'has_today_bar': is_today_bar,
        }

    def _ensure_daily_cache(self, today: datetime):
        key = today.strftime('%Y-%m-%d')
        with self._lock:
            if self._daily_cache_date == key and self._daily_cache:
                return
            cache: Dict[str, dict] = {}
            for sym in self.symbols:
                f = self._daily_facts(sym, today)
                if f:
                    cache[sym] = f
            self._daily_cache = cache
            self._daily_cache_date = key
            logger.info(
                "[Scanner] Daily cache rebuilt for %s — %d/%d symbols",
                key, len(cache), len(self.symbols),
            )

    # ---------- live LTP / volume ----------

    def _live_quotes(self) -> Dict[str, dict]:
        """Batch quote() for LTP + cumulative volume. Empty dict on failure."""
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            keys = ['NSE:' + s for s in self.symbols]
            out: Dict[str, dict] = {}
            # kite.quote handles up to ~500 instruments per call; 81 is fine
            q = kite.quote(keys)
            for s in self.symbols:
                d = q.get('NSE:' + s)
                if not d:
                    continue
                out[s] = {
                    'ltp': d.get('last_price'),
                    'volume': d.get('volume'),
                    'ohlc': d.get('ohlc') or {},
                }
            return out
        except Exception as e:
            logger.debug("[Scanner] live quote failed: %s", e)
            return {}

    # ---------- scoring ----------

    @staticmethod
    def _score(vol_surge: Optional[float], cpr_w: Optional[float],
               cpr_threshold: float, trend: str, range_state: str,
               clean: Optional[str], direction: Optional[str]) -> float:
        score = 0.0

        # volume-surge component (35): ramps 1.0x -> 0, 3.0x+ -> full
        if vol_surge is not None and vol_surge > 0:
            v = max(0.0, min(1.0, (vol_surge - 1.0) / 2.0))
            score += W_VOLUME * v

        # CPR-narrowness vs threshold (20): at/below threshold -> full,
        # linearly fades to 0 by 2x threshold
        if cpr_w is not None and cpr_threshold > 0:
            if cpr_w <= cpr_threshold:
                score += W_CPR
            else:
                frac = max(0.0, 1.0 - (cpr_w - cpr_threshold) / cpr_threshold)
                score += W_CPR * frac

        # trend + range-escape direction agreement (25)
        tr = 0.0
        if direction == 'long' and trend == 'up':
            tr += 0.6
        elif direction == 'short' and trend == 'down':
            tr += 0.6
        elif trend == 'flat':
            tr += 0.2
        if direction == 'long' and range_state in ('week_high', 'day_high'):
            tr += 0.4 if range_state == 'week_high' else 0.2
        elif direction == 'short' and range_state in ('week_low', 'day_low'):
            tr += 0.4 if range_state == 'week_low' else 0.2
        score += W_TREND_RANGE * min(1.0, tr)

        # clean candle (20): full if candle direction matches trade direction
        if clean and clean in ('long', 'short'):
            score += W_CLEAN if clean == direction else W_CLEAN * 0.3

        return round(score, 1)

    # ---------- public ----------

    def compute_all(self, cpr_threshold_pct: float = 0.5) -> dict:
        now = datetime.now()
        market_open = _is_market_open(now)
        self._ensure_daily_cache(now)
        quotes = self._live_quotes() if market_open else {}

        rows = []
        for sym in self.symbols:
            facts = self._daily_cache.get(sym)
            if not facts:
                continue

            q = quotes.get(sym, {})
            ltp = q.get('ltp')
            prev_close = (q.get('ohlc') or {}).get('close')
            # as-of: live during market, else the freshest bar we actually
            # have (5-min if newer than the last daily bar) — so a stale
            # name (e.g. data only to 7-May) shows its true age, not a fake.
            last_5m_close = facts.get('last_5m_close')
            intra_asof = facts.get('intra_asof')
            last_daily_date = facts.get('last_daily_date')
            if ltp is not None:
                as_of = now
            elif (last_5m_close is not None and intra_asof is not None
                  and (last_daily_date is None or intra_asof >= last_daily_date)):
                ltp = last_5m_close          # 5-min fresher than daily
                prev_close = facts['last_daily_close']
                as_of = intra_asof
            else:
                ltp = facts['last_daily_close']
                prev_close = facts.get('prev_daily_close') or ltp
                as_of = last_daily_date
            if prev_close is None:
                prev_close = (q.get('ohlc') or {}).get('close') \
                    or facts.get('prev_daily_close') or ltp

            day_change_pct = None
            if prev_close:
                day_change_pct = round((ltp - prev_close) / prev_close * 100.0, 2)

            # volume surge — SPEC-FAITHFUL: opening 30-min candle volume vs
            # trailing-20 opening-candle baseline (NOT full-day volume).
            # Falls back to the daily ratio only when no intraday exists.
            vol_surge = facts.get('vol_surge_open')
            vol_basis = 'open30m'
            if vol_surge is None:
                avg20 = facts['avg20_vol']
                tv = q.get('volume') if q.get('volume') is not None \
                    else facts['last_bar_vol']
                vol_surge = round(tv / avg20, 2) if avg20 and avg20 > 0 else None
                vol_basis = 'fullday_fallback'

            # daily trend
            sma50 = facts['sma50']
            sma200 = facts['sma200']
            if sma50 and sma200:
                if ltp > sma50 and sma50 >= sma200:
                    trend = 'up'
                elif ltp < sma50 and sma50 <= sma200:
                    trend = 'down'
                else:
                    trend = 'flat'
            elif sma50:
                trend = 'up' if ltp > sma50 else 'down'
            else:
                trend = 'flat'

            # range-break state — "escaped" only beyond prior WEEK range
            pwh = facts['prev_week_high']
            pwl = facts['prev_week_low']
            pdh = facts['prev_day_high']
            pdl = facts['prev_day_low']
            range_state = 'inside'
            escaped = False
            if pwh is not None and ltp > pwh:
                range_state, escaped = 'week_high', True
            elif pwl is not None and ltp < pwl:
                range_state, escaped = 'week_low', True
            elif pdh is not None and ltp > pdh:
                range_state = 'day_high'
            elif pdl is not None and ltp < pdl:
                range_state = 'day_low'

            direction = 'long' if trend == 'up' else ('short' if trend == 'down' else None)

            cpr_w = facts['cpr_width_pct']
            cpr_narrow = (cpr_w is not None and cpr_w <= cpr_threshold_pct)

            score = self._score(
                vol_surge, cpr_w, cpr_threshold_pct, trend,
                range_state, facts['clean_candle'], direction,
            )

            rows.append({
                'symbol': sym,
                'ltp': round(ltp, 2) if ltp is not None else None,
                'day_change_pct': day_change_pct,
                'volume_surge': vol_surge,
                'cpr_width_pct': round(cpr_w, 3) if cpr_w is not None else None,
                'cpr_narrow': cpr_narrow,
                'trend': trend,
                'range_state': range_state,
                'range_escaped': escaped,
                'clean_candle': facts['clean_candle'],
                'direction': direction,
                'score': score,
                'vol_basis': vol_basis,
                'as_of': as_of.isoformat() if hasattr(as_of, 'isoformat') else None,
            })

        rows.sort(key=lambda r: r['score'], reverse=True)
        # freshest data point across the universe (staleness indicator)
        asofs = [r['as_of'] for r in rows if r.get('as_of')]
        data_max = max(asofs) if asofs else None
        return {
            'generated_at': now.isoformat(),
            'market_open': market_open,
            'cpr_threshold': cpr_threshold_pct,
            'count': len(rows),
            'data_max_asof': data_max,
            'vol_surge_basis': 'opening 30-min candle vs trailing-20 '
                               'opening-candle volume (spec); per-row '
                               'vol_basis=fullday_fallback when no intraday',
            'note': 'Off-hours LTP = freshest bar in market_data.db; many '
                    'F&O names lag (DB not refreshed to today). Check per-row '
                    'as_of for true age. Live CMP only during market hours.',
            'rows': rows,
        }


_instance: Optional[ScannerService] = None
_instance_lock = threading.Lock()


def get_scanner_service() -> ScannerService:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = ScannerService()
    return _instance
