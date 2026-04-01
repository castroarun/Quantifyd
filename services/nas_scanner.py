"""
NAS — Nifty ATR Strangle — Scanner
=====================================
Computes ATR squeeze signals on NIFTY 5-minute bars.
Detects entry, adjustment, and exit conditions for intraday strangles.

Core logic:
  - ATR(14) on 5-min candles
  - When ATR < SMA(ATR, 50) for 3+ bars → squeeze → sell strangle
  - Strike placement uses DAILY ATR × 1.5 for OTM distance
  - Adjustments: leg premium doubles (losing) or halves (winning)
"""

import logging
import sqlite3
import calendar
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Constants
LOT_SIZE = 75               # Nifty lot size
STRIKE_STEP = 50             # Nifty options strike gap
RISK_FREE_RATE = 0.065       # India risk-free rate
TRADING_DAYS_PER_YEAR = 252
BARS_PER_DAY_5MIN = 75       # 375 min / 5 min = 75 bars


# ─── Black-Scholes ──────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def estimate_iv(atr, close_price, bars_per_day=1):
    """Estimate IV from ATR with intraday scaling."""
    if atr <= 0 or close_price <= 0:
        return 0.15
    if bars_per_day > 1:
        daily_atr_pct = (atr / close_price) * np.sqrt(bars_per_day)
    else:
        daily_atr_pct = atr / close_price
    daily_std = daily_atr_pct / 1.2
    realized_vol = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    iv = realized_vol * 1.20
    return max(iv, 0.08)


def round_strike(price, direction='nearest'):
    """Round price to nearest NIFTY strike interval."""
    if direction == 'up':
        return int(np.ceil(price / STRIKE_STEP) * STRIKE_STEP)
    elif direction == 'down':
        return int(np.floor(price / STRIKE_STEP) * STRIKE_STEP)
    return int(round(price / STRIKE_STEP) * STRIKE_STEP)


# ─── Expiry Logic ───────────────────────────────────────────

def get_next_thursday(ref_date=None):
    """Get next Thursday (Nifty weekly expiry)."""
    if ref_date is None:
        ref_date = date.today()
    days_ahead = (3 - ref_date.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # Skip today if it's Thursday
    return ref_date + timedelta(days=days_ahead)


def get_current_week_expiry(ref_date=None):
    """
    Get the nearest NIFTY options expiry from Kite instruments API.
    Falls back to Thursday-based calculation if API unavailable.
    """
    if ref_date is None:
        ref_date = date.today()

    try:
        from services.kite_service import get_kite
        kite = get_kite()
        instruments = kite.instruments('NFO')
        nifty_expiries = sorted(set(
            i['expiry'] for i in instruments
            if i['name'] == 'NIFTY' and i['instrument_type'] in ('CE', 'PE')
            and i['expiry'] >= ref_date
        ))
        if nifty_expiries:
            return nifty_expiries[0]
    except Exception as e:
        logger.debug(f"Expiry lookup from Kite failed: {e}")

    # Fallback: next Thursday
    days_ahead = (3 - ref_date.weekday()) % 7
    if days_ahead == 0:
        return ref_date
    return ref_date + timedelta(days=days_ahead)


def is_expiry_day(ref_date=None):
    """Check if today is an expiry day (checks Kite instruments, falls back to Thursday)."""
    if ref_date is None:
        ref_date = date.today()

    try:
        expiry = get_current_week_expiry(ref_date)
        return expiry == ref_date
    except Exception:
        return ref_date.weekday() == 3


# ─── Scanner ────────────────────────────────────────────────

class NasScanner:
    """
    Scans NIFTY 5-min bars for ATR squeeze signals.

    Config keys (from NAS_DEFAULTS):
      atr_period, atr_ma_period, min_squeeze_bars,
      strike_distance_atr, daily_atr_period,
      premium_double_trigger, premium_half_trigger,
      entry_start_time, entry_end_time, skip_expiry_day, max_vix
    """

    def __init__(self, config: dict):
        self.cfg = config

    def load_5min_bars(self, lookback_days=5):
        """Load recent NIFTY 5-min bars directly from Kite API."""
        df = self._fetch_5min_from_kite(lookback_days)

        if df.empty:
            logger.warning("No NIFTY 5-min data from Kite API")
            return df

        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Loaded {len(df)} NIFTY 5-min bars from Kite (latest: {df['date'].iloc[-1]})")
        return df

    def _fetch_5min_from_kite(self, lookback_days=5):
        """Fetch NIFTY 5-min historical candles from Kite API."""
        try:
            from services.kite_service import get_kite
            import time

            kite = get_kite()
            # NIFTY 50 index instrument token = 256265
            instrument_token = 256265
            to_date = datetime.now()
            from_date = to_date - timedelta(days=lookback_days)

            all_rows = []
            # Kite 5-min limit: 7 days per chunk
            chunk_start = from_date
            while chunk_start < to_date:
                chunk_end = min(chunk_start + timedelta(days=7), to_date)
                try:
                    candles = kite.historical_data(
                        instrument_token, chunk_start, chunk_end,
                        interval='5minute', oi=False)
                    for c in candles:
                        all_rows.append({
                            'date': c['date'],
                            'open': c['open'],
                            'high': c['high'],
                            'low': c['low'],
                            'close': c['close'],
                            'volume': c.get('volume', 0),
                        })
                    time.sleep(0.35)  # rate limit
                except Exception as e:
                    logger.warning(f"Kite 5-min fetch error ({chunk_start}): {e}")
                chunk_start = chunk_end

            if all_rows:
                return pd.DataFrame(all_rows)
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Kite API unavailable for 5-min data: {e}")
            return pd.DataFrame()

    def load_daily_bars(self, lookback_days=30):
        """Load daily bars for daily ATR calculation (strike placement)."""
        from config import MARKET_DATA_DB

        conn = sqlite3.connect(str(MARKET_DATA_DB))
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        df = pd.read_sql_query(
            """SELECT date, open, high, low, close, volume
               FROM market_data_unified
               WHERE symbol IN ('NIFTY50', 'NIFTY') AND timeframe='day'
               AND date >= ? ORDER BY date""",
            conn, params=(cutoff,))
        conn.close()

        if df.empty:
            logger.info("No NIFTY daily data in DB — fetching from Kite API")
            df = self._fetch_daily_from_kite(lookback_days)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _fetch_daily_from_kite(self, lookback_days=30):
        """Fetch NIFTY daily candles from Kite API."""
        try:
            from services.kite_service import get_kite

            kite = get_kite()
            instrument_token = 256265  # NIFTY 50 index
            to_date = datetime.now()
            from_date = to_date - timedelta(days=lookback_days)

            candles = kite.historical_data(
                instrument_token, from_date, to_date,
                interval='day', oi=False)

            if candles:
                return pd.DataFrame([{
                    'date': c['date'],
                    'open': c['open'],
                    'high': c['high'],
                    'low': c['low'],
                    'close': c['close'],
                    'volume': c.get('volume', 0),
                } for c in candles])
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Kite API unavailable for daily data: {e}")
            return pd.DataFrame()

    def compute_indicators(self, df):
        """Compute ATR squeeze indicators on 5-min OHLC DataFrame."""
        cfg = self.cfg
        atr_period = cfg.get('atr_period', 14)
        atr_ma_period = cfg.get('atr_ma_period', 50)

        high, low, close = df['high'], df['low'], df['close']

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        # ATR (Wilder's smoothing / RMA — matches Kite & TradingView)
        df['atr'] = tr.ewm(alpha=1.0/atr_period, min_periods=atr_period, adjust=False).mean()

        # ATR Moving Average (SMA — smoother baseline for squeeze detection)
        df['atr_ma'] = df['atr'].rolling(atr_ma_period).mean()

        # Squeeze detection: ATR < ATR_MA
        df['is_squeezing'] = (df['atr'] < df['atr_ma']).astype(int)

        # Consecutive squeeze count
        squeeze_groups = (df['is_squeezing'] != df['is_squeezing'].shift(1)).cumsum()
        df['squeeze_count'] = df.groupby(squeeze_groups)['is_squeezing'].cumsum()

        # BB width for confirmation (optional)
        bb_period = 20
        sma_bb = close.rolling(bb_period).mean()
        std_bb = close.rolling(bb_period).std()
        bb_width = (2 * 2.0 * std_bb) / sma_bb * 100
        bb_width_ma = bb_width.rolling(bb_period).mean()
        df['bb_squeeze'] = (bb_width < bb_width_ma).astype(int)

        # IV estimate from 5-min ATR
        df['est_iv'] = df.apply(
            lambda r: estimate_iv(r['atr'], r['close'], BARS_PER_DAY_5MIN)
            if r['atr'] > 0 and not np.isnan(r['atr']) else 0.15, axis=1)

        return df

    def compute_daily_atr(self, daily_df=None):
        """Get daily ATR for strike placement."""
        if daily_df is None:
            daily_df = self.load_daily_bars()

        if daily_df.empty:
            return None

        period = self.cfg.get('daily_atr_period', 14)
        high, low, close = daily_df['high'], daily_df['low'], daily_df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        daily_df['atr'] = tr.rolling(period).mean()

        latest = daily_df.iloc[-1]
        return round(float(latest['atr']), 2) if not np.isnan(latest['atr']) else None

    def compute_strikes(self, spot, daily_atr):
        """Compute strangle strikes based on daily ATR (legacy fallback)."""
        cfg = self.cfg
        distance = cfg.get('strike_distance_atr', 1.5)

        call_strike = round_strike(spot + distance * daily_atr, 'up')
        put_strike = round_strike(spot - distance * daily_atr, 'down')

        return call_strike, put_strike

    def find_strike_by_premium(self, spot, instrument_type, target_premium, expiry,
                                min_premium=5.0, max_premium=24.0, min_otm_distance=100):
        """
        Find the OTM strike whose live premium is closest to target_premium.

        Scans strikes starting from min_otm_distance OTM outward, fetching
        live Kite quotes, and picks the closest match within [min_premium, max_premium].

        Args:
            spot: Current NIFTY spot price
            instrument_type: 'CE' or 'PE'
            target_premium: Target premium per leg (e.g., 20.0)
            expiry: Expiry date (date or str)
            min_premium: Don't select strikes with premium <= this (default 5)
            max_premium: Don't select strikes with premium > this (default 24)
            min_otm_distance: Minimum OTM distance in points (default 100)

        Returns:
            (strike, premium) or (None, None) if no suitable strike found
        """
        from services.kite_service import get_kite
        kite = get_kite()

        # Determine scan range: start from min_otm_distance, scan outward
        atm = round_strike(spot, 'nearest')

        if instrument_type == 'CE':
            start_strike = round_strike(spot + min_otm_distance, 'up')
            # Scan outward (higher strikes = cheaper premium)
            strikes = list(range(start_strike, atm + 2000, STRIKE_STEP))
        else:
            start_strike = round_strike(spot - min_otm_distance, 'down')
            # Scan outward (lower strikes = cheaper premium)
            strikes = list(range(start_strike, atm - 2000, -STRIKE_STEP))

        if not strikes:
            return None, None

        # Build tradingsymbols and batch-fetch quotes
        symbols = []
        for s in strikes:
            tsym = self._build_tradingsymbol(instrument_type, s, expiry)
            symbols.append((s, f'NFO:{tsym}'))

        # Fetch in batches of 40 (Kite quote API limit)
        best_strike = None
        best_premium = None
        best_diff = float('inf')

        for batch_start in range(0, len(symbols), 40):
            batch = symbols[batch_start:batch_start + 40]
            keys = [sym for _, sym in batch]

            try:
                quotes = kite.quote(keys)
            except Exception as e:
                logger.warning(f"[NAS] Quote batch failed: {e}")
                continue

            for strike, key in batch:
                data = quotes.get(key, {})
                ltp = data.get('last_price', 0)

                if ltp <= min_premium:
                    continue
                if ltp > max_premium:
                    continue

                diff = abs(ltp - target_premium)
                if diff < best_diff:
                    best_diff = diff
                    best_strike = strike
                    best_premium = ltp

            # Early exit if we found a good match (within Rs 2 of target)
            if best_diff <= 2.0:
                break

            import time
            time.sleep(0.35)

        if best_strike:
            logger.info(f"[NAS] Premium-based strike: {best_strike}{instrument_type} "
                        f"@{best_premium:.2f} (target={target_premium:.2f})")
        else:
            logger.warning(f"[NAS] No {instrument_type} strike found with premium "
                           f"in [{min_premium}, {max_premium}] near target {target_premium}")

        return best_strike, best_premium

    def estimate_premiums(self, spot, call_strike, put_strike, iv, dte_days, expiry=None):
        """
        Get option premiums — live Kite quotes first, BS fallback.
        Live quotes capture IV skew and real market pricing.
        """
        # Try live Kite quotes first
        live_call = None
        live_put = None
        if expiry:
            try:
                call_sym = self._build_tradingsymbol('CE', call_strike, expiry)
                put_sym = self._build_tradingsymbol('PE', put_strike, expiry)
                live_call = self.get_live_option_premium(call_sym)
                live_put = self.get_live_option_premium(put_sym)
                if live_call and live_put and live_call > 0 and live_put > 0:
                    logger.info(f"[NAS] Live premiums: CE {call_sym}={live_call:.2f}, PE {put_sym}={live_put:.2f}")
                    return round(live_call, 2), round(live_put, 2)
            except Exception as e:
                logger.debug(f"Live premium fetch failed, using BS: {e}")

        # Fallback to Black-Scholes
        T = max(dte_days, 0.5) / 365.0
        call_prem = bs_call(spot, call_strike, T, RISK_FREE_RATE, iv)
        put_prem = bs_put(spot, put_strike, T, RISK_FREE_RATE, iv)
        logger.debug(f"[NAS] BS premiums: CE {call_strike}={call_prem:.2f}, PE {put_strike}={put_prem:.2f}")
        return round(call_prem, 2), round(put_prem, 2)

    def _build_tradingsymbol(self, instrument_type, strike, expiry):
        """
        Build NFO tradingsymbol for NIFTY options.

        Kite uses two formats:
          - Monthly expiry: NIFTY{YY}{MON}{STRIKE}{TYPE}  e.g. NIFTY26MAR22500CE
          - Weekly expiry:  NIFTY{YY}{M}{DD}{STRIKE}{TYPE} e.g. NIFTY2632422500CE
            where M = 1-9 for Jan-Sep, O/N/D for Oct/Nov/Dec

        We use the instruments cache to resolve the correct tradingsymbol.
        Falls back to constructing it if cache miss.
        """
        if isinstance(expiry, str):
            expiry = date.fromisoformat(expiry[:10])

        # Try instruments cache first (most reliable)
        cached = self._get_cached_tradingsymbol(instrument_type, int(strike), expiry)
        if cached:
            return cached

        # Fallback: construct symbol
        yy = expiry.year % 100
        # Check if it's a month-end expiry (last week of month) → monthly format
        import calendar
        last_day = calendar.monthrange(expiry.year, expiry.month)[1]
        is_monthly = expiry.day > last_day - 7  # Last 7 days = monthly series

        if is_monthly:
            month_map = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
                         7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
            return f"NIFTY{yy}{month_map[expiry.month]}{int(strike)}{instrument_type}"
        else:
            # Weekly: M = month digit (1-9) or O/N/D
            month_code = {1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',
                          7:'7',8:'8',9:'9',10:'O',11:'N',12:'D'}
            return f"NIFTY{yy}{month_code[expiry.month]}{expiry.day}{int(strike)}{instrument_type}"

    def _get_cached_tradingsymbol(self, instrument_type, strike, expiry):
        """Look up tradingsymbol from Kite instruments cache."""
        if not hasattr(self, '_instruments_cache') or self._instruments_cache is None:
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                instruments = kite.instruments('NFO')
                self._instruments_cache = [
                    i for i in instruments
                    if i['name'] == 'NIFTY' and i['instrument_type'] in ('CE', 'PE')
                ]
                logger.info(f"[NAS] Cached {len(self._instruments_cache)} NIFTY option instruments")
            except Exception as e:
                logger.warning(f"[NAS] Instruments cache load failed: {e}")
                self._instruments_cache = []
                return None

        for i in self._instruments_cache:
            if (i['expiry'] == expiry and
                i['strike'] == strike and
                i['instrument_type'] == instrument_type):
                return i['tradingsymbol']

        return None

    def _get_india_vix(self):
        """Get current India VIX from Kite API."""
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            quote = kite.quote(['NSE:INDIA VIX'])
            return quote.get('NSE:INDIA VIX', {}).get('last_price')
        except Exception as e:
            logger.debug(f"India VIX fetch failed: {e}")
            return None

    def get_live_spot(self):
        """Get current NIFTY spot price from Kite API."""
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            quote = kite.quote(['NSE:NIFTY 50'])
            return quote.get('NSE:NIFTY 50', {}).get('last_price')
        except Exception as e:
            logger.warning(f"Live spot fetch failed: {e}")
            return None

    def get_live_option_premium(self, tradingsymbol):
        """Get live option premium from Kite for a specific contract."""
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            key = f'NFO:{tradingsymbol}'
            quote = kite.quote([key])
            return quote.get(key, {}).get('last_price')
        except Exception as e:
            logger.warning(f"Option quote fetch failed for {tradingsymbol}: {e}")
            return None

    def scan(self, df_5min=None, daily_df=None):
        """
        Run full scan. Returns dict with current state and any signals.

        Returns:
            {
                'atr': float,
                'atr_ma': float,
                'is_squeezing': bool,
                'squeeze_count': int,
                'spot': float,
                'iv': float,
                'daily_atr': float,
                'call_strike': int,
                'put_strike': int,
                'call_premium': float,
                'put_premium': float,
                'signals': [...],
                'filters': {...},
            }
        """
        if df_5min is None:
            df_5min = self.load_5min_bars()

        if df_5min.empty or len(df_5min) < 60:
            return {'error': 'Not enough 5-min data', 'signals': []}

        df_5min = self.compute_indicators(df_5min)
        daily_atr = self.compute_daily_atr(daily_df)

        latest = df_5min.iloc[-1]
        cfg = self.cfg

        spot = round(float(latest['close']), 2)
        atr = round(float(latest['atr']), 2) if not np.isnan(latest['atr']) else 0
        atr_ma = round(float(latest['atr_ma']), 2) if not np.isnan(latest['atr_ma']) else 0
        iv = round(float(latest['est_iv']), 4)
        is_squeezing = bool(latest['is_squeezing'])
        squeeze_count = int(latest['squeeze_count']) if is_squeezing else 0

        # Expiry
        today = date.today()
        expiry = get_current_week_expiry(today)
        dte = max((expiry - today).days, 1)

        # Premium-based strike selection (find strikes with ~Rs 20 premium)
        target_prem = cfg.get('target_entry_premium', 20.0)
        min_prem_guard = cfg.get('min_leg_premium', 5.0)
        max_prem_guard = cfg.get('max_leg_premium', 24.0)
        min_otm = cfg.get('min_otm_distance', 100)

        call_strike, call_prem = self.find_strike_by_premium(
            spot, 'CE', target_prem, expiry, min_prem_guard, max_prem_guard, min_otm)
        put_strike, put_prem = self.find_strike_by_premium(
            spot, 'PE', target_prem, expiry, min_prem_guard, max_prem_guard, min_otm)

        # Fallback to ATR-based if live quotes fail
        if call_strike is None or put_strike is None:
            logger.warning("[NAS] Premium-based strike selection failed, falling back to ATR")
            if daily_atr:
                call_strike_fb, put_strike_fb = self.compute_strikes(spot, daily_atr)
            else:
                fallback_daily_atr = atr * np.sqrt(BARS_PER_DAY_5MIN)
                call_strike_fb, put_strike_fb = self.compute_strikes(spot, fallback_daily_atr)
            if call_strike is None:
                call_strike = call_strike_fb
                call_prem_fb, _ = self.estimate_premiums(spot, call_strike, put_strike_fb, iv, dte, expiry)
                call_prem = call_prem_fb
            if put_strike is None:
                put_strike = put_strike_fb
                _, put_prem_fb = self.estimate_premiums(spot, call_strike, put_strike, iv, dte, expiry)
                put_prem = put_prem_fb

        call_prem = call_prem or 0
        put_prem = put_prem or 0

        result = {
            'atr': atr,
            'atr_ma': atr_ma,
            'is_squeezing': is_squeezing,
            'squeeze_count': squeeze_count,
            'spot': spot,
            'iv': iv,
            'daily_atr': daily_atr,
            'call_strike': call_strike,
            'put_strike': put_strike,
            'call_premium': call_prem,
            'put_premium': put_prem,
            'total_premium': round(call_prem + put_prem, 2),
            'expiry': str(expiry),
            'dte': dte,
            'date': str(latest['date']),
            'signals': [],
            'filters': {},
        }

        # ─── Filter checks ────────────────────────────────────

        now = datetime.now()
        current_time = now.strftime('%H:%M')

        # Time window
        entry_start = cfg.get('entry_start_time', '09:30')
        entry_end = cfg.get('entry_end_time', '14:30')
        in_entry_window = entry_start <= current_time <= entry_end
        result['filters']['in_entry_window'] = in_entry_window

        # Expiry day
        is_expiry = is_expiry_day(today)
        skip_expiry = cfg.get('skip_expiry_day', True)
        result['filters']['is_expiry_day'] = is_expiry
        result['filters']['skip_expiry'] = skip_expiry and is_expiry

        # Min premium
        min_prem = cfg.get('min_combined_premium', 30)
        result['filters']['premium_ok'] = (call_prem + put_prem) >= min_prem

        # VIX check (None = disabled)
        max_vix = cfg.get('max_vix')
        vix = self._get_india_vix() if max_vix else None
        result['vix'] = vix
        result['filters']['vix_ok'] = max_vix is None or vix is None or vix <= max_vix

        # ─── Entry signal ─────────────────────────────────────

        min_squeeze = cfg.get('min_squeeze_bars', 3)
        entry_blocked = False
        block_reason = None

        if not in_entry_window:
            entry_blocked = True
            block_reason = f'Outside entry window ({entry_start}-{entry_end})'
        elif skip_expiry and is_expiry:
            entry_blocked = True
            block_reason = 'Expiry day — skipping'
        elif not result['filters']['premium_ok']:
            entry_blocked = True
            block_reason = f'Combined premium {call_prem + put_prem:.1f} < min {min_prem}'
        elif max_vix is not None and vix is not None and vix > max_vix:
            entry_blocked = True
            block_reason = f'VIX {vix:.1f} > max {max_vix}'

        if not entry_blocked and is_squeezing and squeeze_count >= min_squeeze:
            result['signals'].append({
                'type': 'SQUEEZE_ENTRY',
                'action': 'SELL STRANGLE',
                'call_strike': call_strike,
                'put_strike': put_strike,
                'call_premium': call_prem,
                'put_premium': put_prem,
                'total_premium': round(call_prem + put_prem, 2),
                'expiry': str(expiry),
                'dte': dte,
            })
        elif entry_blocked:
            result['filters']['block_reason'] = block_reason

        return result

    def check_adjustments(self, positions, current_spot, current_iv=None):
        """
        Check if any active position legs need adjustment.

        Premium-matching adjustment logic:
          - Losing leg (premium doubles): roll to match the OTHER leg's current premium
          - Winning leg (premium halves): roll to match the OTHER leg's current premium
          - Always equalise premiums between legs ("squeeze out / squeeze in")

        Guards:
          - No adjustment if target premium <= min_leg_premium (Rs 5)
          - No adjustment if new strike < min_otm_distance (100 pts) from spot
          - New leg premium capped at max_leg_premium (Rs 24)

        Returns list of dicts with 'target_premium' for the new leg.
        """
        cfg = self.cfg
        double_trigger = cfg.get('premium_double_trigger', 2.0)
        half_trigger = cfg.get('premium_half_trigger', 0.5)
        min_prem_guard = cfg.get('min_leg_premium', 5.0)
        adjustments = []

        today = date.today()

        # Build a map of current premiums per leg
        leg_premiums = {}  # 'CE' → current_prem, 'PE' → current_prem
        for pos in positions:
            inst_type = pos['instrument_type']
            current_prem = pos.get('_live_premium')
            if current_prem is None:
                try:
                    # Fetch live premium from Kite
                    tsym = pos.get('tradingsymbol', '')
                    if tsym:
                        current_prem = self.get_live_option_premium(tsym)
                except Exception:
                    pass
            if current_prem is None:
                try:
                    strike = pos['strike']
                    exp_str = str(pos.get('expiry_date', ''))[:10]
                    exp_date = date.fromisoformat(exp_str) if exp_str else today + timedelta(days=3)
                    rem_dte = max((exp_date - today).days, 0.5)
                    T = rem_dte / 365.0
                    iv = current_iv or estimate_iv(10, current_spot, 1)
                    if inst_type == 'CE':
                        current_prem = bs_call(current_spot, strike, T, RISK_FREE_RATE, iv)
                    else:
                        current_prem = bs_put(current_spot, strike, T, RISK_FREE_RATE, iv)
                except Exception as e:
                    logger.warning(f"Adjustment check error for pos {pos['id']}: {e}")
                    continue

            pos['_current_premium'] = current_prem
            leg_premiums[inst_type] = current_prem

        for pos in positions:
            entry_prem = pos['entry_price']
            inst_type = pos['instrument_type']
            adj_count = pos.get('adjustment_count', 0)
            max_adj = cfg.get('max_adjustments_per_leg', 2)
            current_prem = pos.get('_current_premium', 0)

            if adj_count >= max_adj:
                continue
            if current_prem <= 0:
                continue

            ratio = current_prem / entry_prem if entry_prem > 0 else 0

            # Get the OTHER leg's current premium (target for new strike)
            other_leg = 'PE' if inst_type == 'CE' else 'CE'
            other_prem = leg_premiums.get(other_leg, 0)

            if ratio >= double_trigger:
                # Losing leg — squeeze out: roll to match other leg's current premium
                target = other_prem
                if target <= min_prem_guard:
                    logger.info(f"[NAS] Skip adj: other leg premium {target:.1f} <= min {min_prem_guard}")
                    continue
                adjustments.append({
                    'position_id': pos['id'],
                    'action': 'ADJUST_LOSING',
                    'leg': pos['leg'],
                    'instrument_type': inst_type,
                    'strike': pos['strike'],
                    'current_premium': round(current_prem, 2),
                    'entry_premium': round(entry_prem, 2),
                    'ratio': round(ratio, 2),
                    'target_premium': round(target, 2),
                    'other_leg_premium': round(other_prem, 2),
                })
            elif ratio <= half_trigger:
                # Winning leg — squeeze in: roll to match other leg's current premium
                target = other_prem
                if target <= min_prem_guard:
                    logger.info(f"[NAS] Skip adj: other leg premium {target:.1f} <= min {min_prem_guard}")
                    continue
                adjustments.append({
                    'position_id': pos['id'],
                    'action': 'ADJUST_WINNING',
                    'leg': pos['leg'],
                    'instrument_type': inst_type,
                    'strike': pos['strike'],
                    'current_premium': round(current_prem, 2),
                    'entry_premium': round(entry_prem, 2),
                    'ratio': round(ratio, 2),
                    'target_premium': round(target, 2),
                    'other_leg_premium': round(other_prem, 2),
                })

        return adjustments

    def check_exits(self, positions, current_spot, strangle_entry_premium, current_iv=None):
        """
        Check if the full strangle needs to exit.

        Returns list of (exit_reason, details) or empty list.
        """
        cfg = self.cfg
        exits = []
        now = datetime.now()
        current_time = now.strftime('%H:%M')

        # Time-based exits
        eod_time = cfg.get('eod_squareoff_time', '15:15')
        time_exit = cfg.get('time_exit', '14:45')

        if current_time >= eod_time:
            exits.append(('eod_squareoff', f'EOD squareoff at {eod_time}'))
            return exits

        if current_time >= time_exit:
            exits.append(('time_exit', f'Time exit at {time_exit}'))
            return exits

        if not positions:
            return exits

        # Per-leg premium adjustments (ROLL_OUT/ROLL_IN) handle risk management
        # No combined SL — each leg is managed individually via cross-leg comparison

        return exits
