"""
NWV — Nifty Weekly View — Data Source Helpers
================================================
Thin adapter between Kite / yfinance and the NWV engine. Keeps
services/nwv_engine.py pure — it receives numbers, returns a view.

Exports:
  fetch_weekly_hlc(week_start)     -> prev week H/L/C + prev Friday close
  fetch_monthly_hlc(month_start)   -> prev month H/L/C (for monthly CPR)
  fetch_first_30min_candle()       -> OHLCV for today 09:15–09:45
  fetch_monday_open()              -> first tick / 09:15 open
  fetch_vix_and_percentile(lookback=60) -> (current_vix, 60-day percentile)
  compute_adx_daily(lookback=30)   -> ADX(14) from Kite daily bars
  fetch_daily_pivots()             -> S2/S1/PP/R1/R2 from yesterday's D bar

All functions are defensive — return None on failure and log. The
scheduled job in app.py aggregates these into the engine call.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger(__name__)

NIFTY_SYMBOL = 'NIFTY 50'         # Kite index symbol (NSE exchange)
INDIAVIX_SYMBOL = 'INDIA VIX'     # Kite index symbol (NSE exchange)


# ─── Kite helpers ──────────────────────────────────────────

def _kite():
    from services.kite_service import get_kite
    return get_kite()


def _nifty_index_token() -> Optional[int]:
    """Resolve NIFTY 50 index instrument token from Kite."""
    try:
        kite = _kite()
        for i in kite.instruments('NSE'):
            if i.get('tradingsymbol') == NIFTY_SYMBOL and i.get('segment') == 'INDICES':
                return i['instrument_token']
        # Fallback: segment may vary
        for i in kite.instruments('NSE'):
            if i.get('name') == 'NIFTY 50' and i.get('instrument_type') == 'EQ':
                return i['instrument_token']
    except Exception as e:
        logger.warning(f"[NWV-DATA] nifty token resolve failed: {e}")
    return None


def _vix_index_token() -> Optional[int]:
    try:
        kite = _kite()
        for i in kite.instruments('NSE'):
            if i.get('tradingsymbol') == INDIAVIX_SYMBOL:
                return i['instrument_token']
    except Exception as e:
        logger.warning(f"[NWV-DATA] vix token resolve failed: {e}")
    return None


def _fetch_daily_candles(token: int, lookback_days: int = 30) -> List[Dict]:
    try:
        kite = _kite()
        to = datetime.now()
        frm = to - timedelta(days=lookback_days + 10)  # +10 for market holidays
        candles = kite.historical_data(token, frm, to, 'day')
        return candles or []
    except Exception as e:
        logger.warning(f"[NWV-DATA] daily candles fetch failed: {e}")
        return []


# ─── Weekly / Monthly HLC ──────────────────────────────────

def fetch_weekly_hlc(week_start: date) -> Optional[Dict[str, float]]:
    """Previous ISO-week's H/L/C. `week_start` is this trade-week's Monday.
    Returns {'high','low','close','prev_fri_close'} or None."""
    token = _nifty_index_token()
    if not token:
        return None
    candles = _fetch_daily_candles(token, lookback_days=20)
    if not candles:
        return None

    # ISO week range of the PREVIOUS week
    prev_week_start = week_start - timedelta(days=7)
    prev_week_end = week_start - timedelta(days=1)

    week_candles = [c for c in candles if _to_date(c['date']) and
                    prev_week_start <= _to_date(c['date']) <= prev_week_end]
    if not week_candles:
        return None

    h = max(c['high'] for c in week_candles)
    l = min(c['low'] for c in week_candles)
    c = week_candles[-1]['close']
    # The "prev Friday close" is the last candle on or before prev_week_end
    # (could be Thursday if Friday was a holiday).
    prev_fri_close = week_candles[-1]['close']

    return {'high': h, 'low': l, 'close': c, 'prev_fri_close': prev_fri_close}


def fetch_monthly_hlc(ref_date: date) -> Optional[Dict[str, float]]:
    """Previous calendar month's H/L/C. Returns {'high','low','close'} or None."""
    token = _nifty_index_token()
    if not token:
        return None
    candles = _fetch_daily_candles(token, lookback_days=45)
    if not candles:
        return None

    first_this_month = ref_date.replace(day=1)
    last_prev_month = first_this_month - timedelta(days=1)
    first_prev_month = last_prev_month.replace(day=1)

    in_range = [c for c in candles if _to_date(c['date']) and
                first_prev_month <= _to_date(c['date']) <= last_prev_month]
    if not in_range:
        return None

    return {
        'high':  max(c['high'] for c in in_range),
        'low':   min(c['low']  for c in in_range),
        'close': in_range[-1]['close'],
    }


# ─── Today's first 30-min candle ───────────────────────────

def fetch_first_30min_candle(d: Optional[date] = None) -> Optional[Dict]:
    """Fetch today's 09:15–09:45 NIFTY spot candle as an aggregated 5-min
    pair. Returns {'open','high','low','close','volume'} or None.
    NIFTY index has no volume — `volume` is 0 from Kite.
    """
    token = _nifty_index_token()
    if not token:
        return None
    d = d or date.today()
    session_start = datetime.combine(d, datetime.min.time()).replace(hour=9, minute=15)
    session_end   = session_start + timedelta(minutes=30)
    try:
        kite = _kite()
        bars = kite.historical_data(token, session_start, session_end, '5minute')
    except Exception as e:
        logger.warning(f"[NWV-DATA] 5-min fetch failed: {e}")
        return None
    if not bars:
        return None

    # Combine 2 × 5-min into one 30-min candle, or accept raw 30-minute fetch
    o = bars[0]['open']
    c = bars[-1]['close']
    h = max(b['high'] for b in bars)
    l = min(b['low'] for b in bars)
    v = sum(b.get('volume', 0) for b in bars)
    return {'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}


def fetch_monday_open() -> Optional[float]:
    """Today's 9:15 open. Returns None if market not open yet."""
    c = fetch_first_30min_candle()
    return c['open'] if c else None


# ─── India VIX + percentile ────────────────────────────────

def fetch_vix_and_percentile(lookback: int = 60) -> Tuple[Optional[float], Optional[float]]:
    """Current INDIA VIX value and its percentile over the last `lookback`
    trading days. Uses Kite first; yfinance fallback."""
    # Try Kite
    token = _vix_index_token()
    if token:
        try:
            to = datetime.now()
            frm = to - timedelta(days=lookback + 10)
            bars = _kite().historical_data(token, frm, to, 'day')
            if bars:
                closes = [b['close'] for b in bars if b.get('close')]
                if closes:
                    current = closes[-1]
                    history = closes[-lookback:] if len(closes) > lookback else closes
                    rank = sum(1 for x in history if x <= current) / len(history) * 100.0
                    return float(current), round(rank, 2)
        except Exception as e:
            logger.warning(f"[NWV-DATA] VIX via Kite failed: {e}")

    # yfinance fallback
    try:
        import yfinance as yf
        df = yf.Ticker('^INDIAVIX').history(period=f'{lookback + 10}d')
        if df is None or df.empty:
            return None, None
        closes = df['Close'].tolist()
        current = closes[-1]
        history = closes[-lookback:] if len(closes) > lookback else closes
        rank = sum(1 for x in history if x <= current) / len(history) * 100.0
        return float(current), round(rank, 2)
    except Exception as e:
        logger.warning(f"[NWV-DATA] VIX via yfinance failed: {e}")
    return None, None


# ─── Daily pivots (yesterday's H/L/C) ──────────────────────

def fetch_daily_pivots() -> Optional[Dict[str, float]]:
    token = _nifty_index_token()
    if not token:
        return None
    candles = _fetch_daily_candles(token, lookback_days=10)
    if len(candles) < 2:
        return None

    today = date.today()
    prev = None
    for c in reversed(candles):
        cd = _to_date(c['date'])
        if cd and cd < today:
            prev = c
            break
    if not prev:
        return None

    h, l, clo = prev['high'], prev['low'], prev['close']
    pp = (h + l + clo) / 3.0
    return {
        'pp': pp,
        'r1': 2 * pp - l,
        's1': 2 * pp - h,
        'r2': pp + (h - l),
        's2': pp - (h - l),
    }


# ─── ADX (Wilder, 14) on daily ─────────────────────────────

def compute_adx_daily(lookback: int = 30) -> Optional[float]:
    """Standard Wilder's ADX(14) from daily bars. Returns latest ADX."""
    token = _nifty_index_token()
    if not token:
        return None
    candles = _fetch_daily_candles(token, lookback_days=lookback + 5)
    if len(candles) < 16:
        return None

    period = 14
    highs  = [c['high']  for c in candles]
    lows   = [c['low']   for c in candles]
    closes = [c['close'] for c in candles]

    plus_dm  = [0.0]
    minus_dm = [0.0]
    tr       = [highs[0] - lows[0]]
    for i in range(1, len(candles)):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        ))

    # Wilder smoothing
    def wilder_smooth(series, period):
        if len(series) < period:
            return None
        out = [sum(series[:period])]
        for v in series[period:]:
            out.append(out[-1] - (out[-1] / period) + v)
        return out

    tr_s = wilder_smooth(tr, period)
    pdm_s = wilder_smooth(plus_dm, period)
    mdm_s = wilder_smooth(minus_dm, period)
    if not tr_s or not pdm_s or not mdm_s:
        return None

    plus_di  = [100 * p / t if t else 0 for p, t in zip(pdm_s, tr_s)]
    minus_di = [100 * m / t if t else 0 for m, t in zip(mdm_s, tr_s)]
    dx = [100 * abs(p - m) / (p + m) if (p + m) else 0
          for p, m in zip(plus_di, minus_di)]
    if len(dx) < period:
        return None

    adx = [sum(dx[:period]) / period]
    for v in dx[period:]:
        adx.append((adx[-1] * (period - 1) + v) / period)
    return round(adx[-1], 2)


# ─── tiny internals ────────────────────────────────────────

def _to_date(d):
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        return datetime.strptime(str(d)[:10], '%Y-%m-%d').date()
    except Exception:
        return None
