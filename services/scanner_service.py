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


def _intraday_first_candle_clean(dm, symbol: str, ref_date: datetime,
                                 tf_minutes: int = 30) -> Optional[str]:
    """Clean-directional flag for the OPENING `tf_minutes` candle of the most
    recent trading session in the 5-min data (the real spec trigger, not a
    daily-bar proxy). Returns 'long' | 'short' | 'none' | None (no intraday)."""
    try:
        to_d = ref_date
        from_d = ref_date - timedelta(days=10)
        df5 = dm.load_data(symbol, '5minute', from_d, to_d)
    except Exception:
        return None
    if df5 is None or df5.empty:
        return None
    df5 = df5.sort_index()
    idx = df5.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx)
        df5.index = idx
    # most recent session present in the data
    last_day = idx[-1].normalize()
    day_bars = df5[idx.normalize() == last_day]
    if day_bars.empty:
        return None
    # opening tf_minutes window of that session
    sess_open = day_bars.index[0]
    win_end = sess_open + pd.Timedelta(minutes=tf_minutes)
    first = day_bars[day_bars.index < win_end]
    if first.empty:
        return None
    o = float(first['open'].iloc[0])
    h = float(first['high'].max())
    lo = float(first['low'].min())
    c = float(first['close'].iloc[-1])
    rng = h - lo
    if rng <= 0:
        return 'none'
    body_frac = abs(c - o) / rng
    if c > o and body_frac >= 0.6:
        return 'long'
    if c < o and body_frac >= 0.6:
        return 'short'
    return 'none'


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
        # TRUE opening 30-min candle from 5-min data (real spec trigger);
        # fall back to the daily-bar proxy only if no intraday available.
        clean_candle = _intraday_first_candle_clean(self._dm, symbol, today, 30)
        if clean_candle is None and is_today_bar:
            o = float(df['open'].iloc[-1])
            h = float(df['high'].iloc[-1])
            lo = float(df['low'].iloc[-1])
            c = float(df['close'].iloc[-1])
            rng = h - lo
            if rng > 0:
                body_frac = abs(c - o) / rng
                if c > o and body_frac >= 0.6:
                    clean_candle = 'long'
                elif c < o and body_frac >= 0.6:
                    clean_candle = 'short'
                else:
                    clean_candle = 'none'

        # prior daily close (for a real off-hours day-change %)
        prev_daily_close = float(close.iloc[-2]) if len(close) >= 2 else None

        return {
            'sma50': sma50,
            'sma200': sma200,
            'last_daily_close': last_close,
            'prev_daily_close': prev_daily_close,
            'avg20_vol': avg20_vol,
            'last_bar_vol': last_bar_vol,
            'prev_day_high': prev_day_high,
            'prev_day_low': prev_day_low,
            'prev_week_high': pw_high,
            'prev_week_low': pw_low,
            'cpr_width_pct': cpr_w,
            'clean_candle': clean_candle,
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
            if ltp is None:
                ltp = facts['last_daily_close']
            if prev_close is None:
                # off-hours: show last completed session's move
                # (last daily close vs the prior daily close), not 0
                prev_close = facts.get('prev_daily_close') or facts['last_daily_close']

            day_change_pct = None
            if prev_close:
                day_change_pct = round((ltp - prev_close) / prev_close * 100.0, 2)

            # volume surge: live cumulative vol if available, else last daily bar
            today_vol = q.get('volume')
            if today_vol is None:
                today_vol = facts['last_bar_vol']
            avg20 = facts['avg20_vol']
            vol_surge = round(today_vol / avg20, 2) if avg20 and avg20 > 0 else None

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
            })

        rows.sort(key=lambda r: r['score'], reverse=True)
        return {
            'generated_at': now.isoformat(),
            'market_open': market_open,
            'cpr_threshold': cpr_threshold_pct,
            'count': len(rows),
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
