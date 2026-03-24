"""
Options Data Manager — Index Option Chain Downloader
======================================================
Downloads and stores periodic option chain snapshots for:
  - NIFTY (NSE/NFO)
  - BANKNIFTY (NSE/NFO)
  - SENSEX (BSE/BFO)

Purpose: Build a historical options database for backtesting various
options strategies (strangles, straddles, iron condors, etc.)

Data captured every 5 min during market hours:
  - Strike, Expiry, CE/PE
  - LTP, Bid, Ask, OI, Volume, IV
  - Underlying spot at snapshot time

Storage: Separate SQLite DB (options_data.db) to avoid bloating market_data.db.

Kite API usage:
  - kite.instruments("NFO") / kite.instruments("BFO") — once daily, cached
  - kite.quote([list of tradingsymbols]) — up to 500 per call, every 5 min
  - Rate limit: 3 req/sec → 0.35s between calls
"""

import os
import time
import sqlite3
import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ─── Config ─────────────────────────────────────────────────

_volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
DATA_DIR = Path(_volume_path) / "data" if _volume_path else Path(__file__).parent.parent / "backtest_data"
OPTIONS_DB_PATH = str(DATA_DIR / "options_data.db")

# Indices to track
INDEX_CONFIG = {
    'NIFTY': {
        'exchange': 'NFO',          # Options exchange
        'spot_exchange': 'NSE',     # Spot price exchange
        'spot_symbol': 'NIFTY 50',  # Kite spot symbol
        'name_filter': 'NIFTY',     # instruments() name field
        'strike_step': 50,
        'strikes_each_side': 20,    # Capture 20 strikes above + 20 below ATM
        'lot_size': 75,
    },
    'BANKNIFTY': {
        'exchange': 'NFO',
        'spot_exchange': 'NSE',
        'spot_symbol': 'NIFTY BANK',
        'name_filter': 'BANKNIFTY',
        'strike_step': 100,
        'strikes_each_side': 20,
        'lot_size': 15,
    },
    'SENSEX': {
        'exchange': 'BFO',
        'spot_exchange': 'BSE',
        'spot_symbol': 'SENSEX',
        'name_filter': 'SENSEX',
        'strike_step': 100,
        'strikes_each_side': 20,
        'lot_size': 10,
    },
}


# ─── Database ────────────────────────────────────────────────

class OptionsDataDB:
    """SQLite persistence for options chain snapshots."""

    def __init__(self, db_path=None):
        self.db_path = db_path or OPTIONS_DB_PATH
        self.db_lock = threading.Lock()
        self._init_database()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_database(self):
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    -- Option chain snapshots (the core table)
                    CREATE TABLE IF NOT EXISTS option_chain (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_time TIMESTAMP NOT NULL,
                        symbol VARCHAR(15) NOT NULL,
                        expiry_date DATE NOT NULL,
                        strike REAL NOT NULL,
                        instrument_type VARCHAR(5) NOT NULL,
                        tradingsymbol VARCHAR(60),
                        ltp REAL,
                        bid REAL,
                        ask REAL,
                        oi INTEGER,
                        volume INTEGER,
                        iv REAL,
                        delta REAL,
                        gamma REAL,
                        theta REAL,
                        vega REAL,
                        underlying_spot REAL,
                        lot_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    -- Indexes for fast backtesting queries
                    CREATE INDEX IF NOT EXISTS idx_oc_symbol_time
                        ON option_chain(symbol, snapshot_time);
                    CREATE INDEX IF NOT EXISTS idx_oc_symbol_expiry_strike
                        ON option_chain(symbol, expiry_date, strike, instrument_type);
                    CREATE INDEX IF NOT EXISTS idx_oc_snapshot_time
                        ON option_chain(snapshot_time);

                    -- Underlying spot snapshots (lighter table for just spot prices)
                    CREATE TABLE IF NOT EXISTS underlying_spot (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_time TIMESTAMP NOT NULL,
                        symbol VARCHAR(15) NOT NULL,
                        spot_price REAL NOT NULL,
                        day_open REAL,
                        day_high REAL,
                        day_low REAL,
                        volume INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE INDEX IF NOT EXISTS idx_us_symbol_time
                        ON underlying_spot(symbol, snapshot_time);

                    -- Download log (track what's been captured)
                    CREATE TABLE IF NOT EXISTS download_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_time TIMESTAMP NOT NULL,
                        symbol VARCHAR(15) NOT NULL,
                        instruments_captured INTEGER DEFAULT 0,
                        expiries_captured INTEGER DEFAULT 0,
                        duration_ms INTEGER,
                        status VARCHAR(20) DEFAULT 'OK',
                        error TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
                logger.info(f"Options database initialized at {self.db_path}")
            finally:
                conn.close()

    def insert_snapshot_batch(self, rows: List[dict]):
        """Insert a batch of option chain rows."""
        if not rows:
            return 0

        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.executemany("""
                    INSERT INTO option_chain
                    (snapshot_time, symbol, expiry_date, strike, instrument_type,
                     tradingsymbol, ltp, bid, ask, oi, volume, iv,
                     delta, gamma, theta, vega, underlying_spot, lot_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [(
                    r['snapshot_time'], r['symbol'], r['expiry_date'],
                    r['strike'], r['instrument_type'], r.get('tradingsymbol'),
                    r.get('ltp'), r.get('bid'), r.get('ask'),
                    r.get('oi'), r.get('volume'), r.get('iv'),
                    r.get('delta'), r.get('gamma'), r.get('theta'), r.get('vega'),
                    r.get('underlying_spot'), r.get('lot_size'),
                ) for r in rows])
                conn.commit()
                return len(rows)
            finally:
                conn.close()

    def insert_spot(self, snapshot_time, symbol, spot_price,
                    day_open=None, day_high=None, day_low=None, volume=None):
        """Insert underlying spot snapshot."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO underlying_spot
                    (snapshot_time, symbol, spot_price, day_open, day_high, day_low, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (snapshot_time, symbol, spot_price, day_open, day_high, day_low, volume))
                conn.commit()
            finally:
                conn.close()

    def log_download(self, snapshot_time, symbol, instruments_captured,
                     expiries_captured, duration_ms, status='OK', error=None):
        with self.db_lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO download_log
                    (snapshot_time, symbol, instruments_captured, expiries_captured,
                     duration_ms, status, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (snapshot_time, symbol, instruments_captured,
                      expiries_captured, duration_ms, status, error))
                conn.commit()
            finally:
                conn.close()

    def get_stats(self):
        """Get database statistics."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                stats = {}
                # Total rows
                row = conn.execute("SELECT COUNT(*) as cnt FROM option_chain").fetchone()
                stats['total_rows'] = row['cnt']

                # Per symbol
                rows = conn.execute("""
                    SELECT symbol, COUNT(*) as cnt,
                           MIN(snapshot_time) as first_snap,
                           MAX(snapshot_time) as last_snap,
                           COUNT(DISTINCT DATE(snapshot_time)) as days
                    FROM option_chain GROUP BY symbol
                """).fetchall()
                stats['by_symbol'] = {r['symbol']: dict(r) for r in rows}

                # Today's snapshots
                today = date.today().isoformat()
                row = conn.execute("""
                    SELECT COUNT(DISTINCT snapshot_time) as snaps, COUNT(*) as rows
                    FROM option_chain WHERE snapshot_time >= ?
                """, (today,)).fetchone()
                stats['today_snapshots'] = row['snaps']
                stats['today_rows'] = row['rows']

                # DB file size
                try:
                    stats['db_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 1)
                except OSError:
                    stats['db_size_mb'] = 0

                return stats
            finally:
                conn.close()

    def get_option_chain(self, symbol, snapshot_time=None, expiry_date=None):
        """
        Get option chain for backtesting.

        Args:
            symbol: NIFTY, BANKNIFTY, SENSEX
            snapshot_time: specific time (latest if None)
            expiry_date: filter by expiry (nearest if None)

        Returns: list of dicts with full chain data
        """
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = "SELECT * FROM option_chain WHERE symbol=?"
                params = [symbol]

                if snapshot_time:
                    sql += " AND snapshot_time=?"
                    params.append(snapshot_time)
                else:
                    # Get latest snapshot
                    latest = conn.execute(
                        "SELECT MAX(snapshot_time) as t FROM option_chain WHERE symbol=?",
                        (symbol,)).fetchone()
                    if latest and latest['t']:
                        sql += " AND snapshot_time=?"
                        params.append(latest['t'])

                if expiry_date:
                    sql += " AND expiry_date=?"
                    params.append(expiry_date)

                sql += " ORDER BY strike, instrument_type"
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_available_snapshots(self, symbol, trade_date=None, limit=100):
        """Get list of available snapshot times for a symbol."""
        with self.db_lock:
            conn = self._get_conn()
            try:
                sql = """SELECT DISTINCT snapshot_time, COUNT(*) as instruments
                         FROM option_chain WHERE symbol=?"""
                params = [symbol]
                if trade_date:
                    sql += " AND DATE(snapshot_time)=?"
                    params.append(trade_date)
                sql += " GROUP BY snapshot_time ORDER BY snapshot_time DESC LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


# ─── Singleton ───────────────────────────────────────────────

_db_instance = None
_db_lock = threading.Lock()


def get_options_db():
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = OptionsDataDB()
    return _db_instance


# ─── Option Chain Downloader ────────────────────────────────

class OptionsDataManager:
    """
    Downloads option chain snapshots for index options via Kite API.

    Usage:
        manager = OptionsDataManager()
        manager.capture_all()  # Captures NIFTY + BANKNIFTY + SENSEX
    """

    def __init__(self, kite=None):
        self.kite = kite
        self.db = get_options_db()
        self._instruments_cache = {}  # exchange -> list
        self._cache_date = None

    def _get_kite(self):
        """Lazy-load Kite instance."""
        if self.kite:
            return self.kite
        from services.kite_service import get_kite
        self.kite = get_kite()
        return self.kite

    def _load_instruments(self, exchange: str) -> List[dict]:
        """Load instruments from Kite API (cached daily)."""
        today = date.today()
        if self._cache_date == today and exchange in self._instruments_cache:
            return self._instruments_cache[exchange]

        kite = self._get_kite()
        try:
            all_inst = kite.instruments(exchange)
            self._instruments_cache[exchange] = all_inst
            self._cache_date = today
            logger.info(f"Loaded {len(all_inst)} instruments from {exchange}")
            return all_inst
        except Exception as e:
            logger.error(f"Failed to load {exchange} instruments: {e}")
            return self._instruments_cache.get(exchange, [])

    def _get_spot_price(self, cfg: dict) -> Optional[float]:
        """Get current spot price for an index."""
        kite = self._get_kite()
        try:
            key = f"{cfg['spot_exchange']}:{cfg['spot_symbol']}"
            quote = kite.quote([key])
            data = quote.get(key, {})
            return data.get('last_price')
        except Exception as e:
            logger.error(f"Spot price fetch failed for {cfg['spot_symbol']}: {e}")
            return None

    def _filter_option_instruments(self, instruments: List[dict], symbol: str,
                                    spot: float, cfg: dict) -> List[dict]:
        """
        Filter instruments to only relevant option strikes near ATM.

        Returns list of instrument dicts for CE + PE within N strikes of spot.
        """
        name_filter = cfg['name_filter']
        strike_step = cfg['strike_step']
        n_strikes = cfg['strikes_each_side']

        # Only options (CE/PE), not futures
        options = [inst for inst in instruments
                   if inst.get('name') == name_filter
                   and inst.get('instrument_type') in ('CE', 'PE')
                   and inst.get('expiry') is not None]

        if not options:
            logger.warning(f"No options found for {name_filter}")
            return []

        # ATM strike
        atm = round(spot / strike_step) * strike_step

        # Strike range
        low_strike = atm - (n_strikes * strike_step)
        high_strike = atm + (n_strikes * strike_step)

        # Filter to strike range and only near-term expiries (next 3)
        expiries = sorted(set(inst['expiry'] for inst in options))
        near_expiries = expiries[:4]  # Current + next 3 expiries

        filtered = [inst for inst in options
                    if low_strike <= inst['strike'] <= high_strike
                    and inst['expiry'] in near_expiries]

        logger.info(f"{symbol}: ATM={atm}, range={low_strike}-{high_strike}, "
                     f"expiries={len(near_expiries)}, instruments={len(filtered)}")
        return filtered

    def capture_symbol(self, symbol: str) -> dict:
        """
        Capture one full option chain snapshot for a symbol.

        Returns: {instruments_captured, expiries, duration_ms, error}
        """
        cfg = INDEX_CONFIG.get(symbol)
        if not cfg:
            return {'error': f'Unknown symbol: {symbol}'}

        start_time = time.time()
        snapshot_time = datetime.now().isoformat(timespec='seconds')

        try:
            # 1. Get spot price
            spot = self._get_spot_price(cfg)
            if not spot:
                raise ValueError(f"Could not get spot price for {symbol}")

            time.sleep(0.35)  # Rate limit

            # 2. Load instruments
            instruments = self._load_instruments(cfg['exchange'])
            if not instruments:
                raise ValueError(f"No instruments from {cfg['exchange']}")

            # 3. Filter to relevant strikes
            option_instruments = self._filter_option_instruments(
                instruments, symbol, spot, cfg)
            if not option_instruments:
                raise ValueError(f"No option instruments matched for {symbol}")

            # 4. Build tradingsymbol list and fetch quotes in batches of 500
            all_rows = []
            expiries_seen = set()

            # Process in batches
            batch_size = 450  # Leave margin under 500 limit
            tsym_to_inst = {}
            for inst in option_instruments:
                key = f"{cfg['exchange']}:{inst['tradingsymbol']}"
                tsym_to_inst[key] = inst

            keys = list(tsym_to_inst.keys())

            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]

                kite = self._get_kite()
                quotes = kite.quote(batch_keys)
                time.sleep(0.35)  # Rate limit

                for key, qdata in quotes.items():
                    inst = tsym_to_inst.get(key)
                    if not inst:
                        continue

                    expiry = inst['expiry']
                    if isinstance(expiry, date):
                        expiry_str = expiry.isoformat()
                    else:
                        expiry_str = str(expiry)[:10]
                    expiries_seen.add(expiry_str)

                    # Extract greeks if available (Kite provides for some)
                    greeks = qdata.get('greeks', {}) or {}

                    depth = qdata.get('depth', {})
                    buy_depth = depth.get('buy', [{}])
                    sell_depth = depth.get('sell', [{}])
                    best_bid = buy_depth[0].get('price', 0) if buy_depth else 0
                    best_ask = sell_depth[0].get('price', 0) if sell_depth else 0

                    row = {
                        'snapshot_time': snapshot_time,
                        'symbol': symbol,
                        'expiry_date': expiry_str,
                        'strike': inst['strike'],
                        'instrument_type': inst['instrument_type'],
                        'tradingsymbol': inst['tradingsymbol'],
                        'ltp': qdata.get('last_price'),
                        'bid': best_bid,
                        'ask': best_ask,
                        'oi': qdata.get('oi'),
                        'volume': qdata.get('volume'),
                        'iv': greeks.get('iv'),
                        'delta': greeks.get('delta'),
                        'gamma': greeks.get('gamma'),
                        'theta': greeks.get('theta'),
                        'vega': greeks.get('vega'),
                        'underlying_spot': spot,
                        'lot_size': cfg['lot_size'],
                    }
                    all_rows.append(row)

            # 5. Save to DB
            inserted = self.db.insert_snapshot_batch(all_rows)

            # 6. Save spot price
            self.db.insert_spot(snapshot_time, symbol, spot)

            duration_ms = int((time.time() - start_time) * 1000)

            # 7. Log download
            self.db.log_download(
                snapshot_time=snapshot_time,
                symbol=symbol,
                instruments_captured=inserted,
                expiries_captured=len(expiries_seen),
                duration_ms=duration_ms,
            )

            logger.info(f"[OPTIONS] {symbol}: captured {inserted} instruments "
                         f"across {len(expiries_seen)} expiries in {duration_ms}ms")

            return {
                'symbol': symbol,
                'instruments_captured': inserted,
                'expiries': sorted(expiries_seen),
                'spot': spot,
                'duration_ms': duration_ms,
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[OPTIONS] {symbol} capture failed: {e}")
            self.db.log_download(
                snapshot_time=snapshot_time,
                symbol=symbol,
                instruments_captured=0,
                expiries_captured=0,
                duration_ms=duration_ms,
                status='ERROR',
                error=str(e),
            )
            return {'symbol': symbol, 'error': str(e)}

    def capture_all(self) -> List[dict]:
        """
        Capture option chain snapshots for all configured indices.
        Called every 5 min during market hours.

        Returns list of results per symbol.
        """
        results = []
        for symbol in INDEX_CONFIG:
            try:
                result = self.capture_symbol(symbol)
                results.append(result)
                time.sleep(0.5)  # Brief pause between indices
            except Exception as e:
                logger.error(f"[OPTIONS] capture_all error for {symbol}: {e}")
                results.append({'symbol': symbol, 'error': str(e)})

        total = sum(r.get('instruments_captured', 0) for r in results)
        logger.info(f"[OPTIONS] capture_all complete: {total} total instruments across "
                     f"{len(results)} indices")
        return results

    def get_live_spot(self, symbol: str) -> Optional[float]:
        """Get current spot price — useful for NAS scanner."""
        cfg = INDEX_CONFIG.get(symbol)
        if not cfg:
            return None
        return self._get_spot_price(cfg)

    def get_live_option_quote(self, tradingsymbol: str, exchange: str = 'NFO') -> Optional[dict]:
        """Get live quote for a specific option contract."""
        kite = self._get_kite()
        try:
            key = f"{exchange}:{tradingsymbol}"
            quote = kite.quote([key])
            return quote.get(key)
        except Exception as e:
            logger.error(f"Option quote failed for {tradingsymbol}: {e}")
            return None
