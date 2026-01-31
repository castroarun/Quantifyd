"""
Nifty 500 Universe Manager
===========================

Manages the Nifty 500 stock universe for the Momentum + Quality strategy.
Loads constituents from a static CSV (refreshed quarterly from NSE).

Features:
- Load Nifty 500 symbols with sector/industry classification
- Detect financial vs non-financial stocks for quality gate routing
- Cache instrument tokens from Kite API (avoids repeated instrument lookups)
- Provide universe subsets (financials, non-financials, by sector)
"""

import csv
import sqlite3
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
DB_DIR = Path(__file__).parent.parent / 'backtest_data'
NIFTY500_CSV = DATA_DIR / 'nifty500_list.csv'
MARKET_DATA_DB = DB_DIR / 'market_data.db'

# Sectors that use financial quality criteria (ROA/ROE instead of D/E)
FINANCIAL_SECTORS = {
    'Financial Services',
    'Banks',
    'Banking',
}

FINANCIAL_INDUSTRIES = {
    'bank', 'banks', 'banking',
    'nbfc', 'finance', 'financial',
    'housing finance', 'insurance',
    'asset management',
}


@dataclass
class Stock:
    """Single stock in the Nifty 500 universe."""
    symbol: str
    company: str
    industry: str
    sector: str
    is_financial: bool


@dataclass
class Nifty500Universe:
    """Complete Nifty 500 universe with classification."""
    stocks: List[Stock] = field(default_factory=list)
    _by_symbol: Dict[str, Stock] = field(default_factory=dict, repr=False)
    loaded_at: Optional[datetime] = None

    def __post_init__(self):
        self._by_symbol = {s.symbol: s for s in self.stocks}

    @property
    def symbols(self) -> List[str]:
        return [s.symbol for s in self.stocks]

    @property
    def financial_symbols(self) -> List[str]:
        return [s.symbol for s in self.stocks if s.is_financial]

    @property
    def non_financial_symbols(self) -> List[str]:
        return [s.symbol for s in self.stocks if not s.is_financial]

    def get(self, symbol: str) -> Optional[Stock]:
        return self._by_symbol.get(symbol)

    def is_financial(self, symbol: str) -> bool:
        stock = self._by_symbol.get(symbol)
        return stock.is_financial if stock else False

    def by_sector(self, sector: str) -> List[Stock]:
        return [s for s in self.stocks if s.sector == sector]

    def sectors(self) -> Dict[str, int]:
        counts = {}
        for s in self.stocks:
            counts[s.sector] = counts.get(s.sector, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def __len__(self):
        return len(self.stocks)


def _is_financial_stock(sector: str, industry: str) -> bool:
    """Determine if a stock should use financial quality criteria."""
    if sector in FINANCIAL_SECTORS:
        return True
    industry_lower = industry.lower()
    return any(kw in industry_lower for kw in FINANCIAL_INDUSTRIES)


def load_nifty500(csv_path: Path = None) -> Nifty500Universe:
    """
    Load Nifty 500 universe from CSV file.

    CSV format: Symbol,Company,Industry,Sector

    Args:
        csv_path: Path to CSV file. Defaults to data/nifty500_list.csv

    Returns:
        Nifty500Universe with all stocks loaded
    """
    csv_path = csv_path or NIFTY500_CSV

    if not csv_path.exists():
        logger.error(f"Nifty 500 CSV not found: {csv_path}")
        raise FileNotFoundError(
            f"Nifty 500 list not found at {csv_path}. "
            "Download from NSE and save as data/nifty500_list.csv"
        )

    stocks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get('Symbol', '').strip()
            if not symbol:
                continue
            company = row.get('Company', '').strip()
            industry = row.get('Industry', '').strip()
            sector = row.get('Sector', '').strip()
            is_fin = _is_financial_stock(sector, industry)

            stocks.append(Stock(
                symbol=symbol,
                company=company,
                industry=industry,
                sector=sector,
                is_financial=is_fin,
            ))

    universe = Nifty500Universe(stocks=stocks, loaded_at=datetime.now())
    universe._by_symbol = {s.symbol: s for s in stocks}

    logger.info(
        f"Loaded Nifty 500: {len(stocks)} stocks, "
        f"{len(universe.financial_symbols)} financial, "
        f"{len(universe.non_financial_symbols)} non-financial"
    )
    return universe


# =========================================================================
# Instrument Token Cache
# =========================================================================

def _init_instrument_cache(db_path: Path = None):
    """Create instrument cache table if not exists."""
    db_path = db_path or MARKET_DATA_DB
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS instrument_cache (
            tradingsymbol TEXT PRIMARY KEY,
            instrument_token INTEGER NOT NULL,
            instrument_type TEXT,
            exchange TEXT,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def cache_instruments(kite, exchange: str = "NSE", db_path: Path = None):
    """
    Bulk-cache all instrument tokens from Kite API.

    This should be called once (or daily) to avoid per-symbol API calls
    in the data_manager's _fetch_from_kite method.

    Args:
        kite: KiteConnect instance
        exchange: Exchange to fetch instruments for
        db_path: Path to database
    """
    db_path = db_path or MARKET_DATA_DB
    _init_instrument_cache(db_path)

    logger.info(f"Fetching instrument list from Kite API ({exchange})...")
    instruments = kite.instruments(exchange)
    logger.info(f"Got {len(instruments)} instruments from {exchange}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    inserted = 0
    for inst in instruments:
        if inst.get('instrument_type') == 'EQ':
            cursor.execute("""
                INSERT OR REPLACE INTO instrument_cache
                (tradingsymbol, instrument_token, instrument_type, exchange, cached_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                inst['tradingsymbol'],
                inst['instrument_token'],
                inst['instrument_type'],
                inst['exchange'],
                datetime.now().isoformat(),
            ))
            inserted += 1

    conn.commit()
    conn.close()
    logger.info(f"Cached {inserted} equity instrument tokens")


def get_instrument_token(symbol: str, db_path: Path = None) -> Optional[int]:
    """
    Look up instrument token from cache.

    Args:
        symbol: NSE trading symbol
        db_path: Path to database

    Returns:
        Instrument token or None if not cached
    """
    db_path = db_path or MARKET_DATA_DB
    _init_instrument_cache(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT instrument_token FROM instrument_cache WHERE tradingsymbol = ?",
        (symbol,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def get_instrument_tokens_bulk(symbols: List[str], db_path: Path = None) -> Dict[str, int]:
    """
    Look up instrument tokens for multiple symbols at once.

    Args:
        symbols: List of NSE trading symbols
        db_path: Path to database

    Returns:
        Dict mapping symbol -> instrument_token (only found symbols)
    """
    db_path = db_path or MARKET_DATA_DB
    _init_instrument_cache(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    placeholders = ','.join('?' * len(symbols))
    cursor.execute(
        f"SELECT tradingsymbol, instrument_token FROM instrument_cache "
        f"WHERE tradingsymbol IN ({placeholders})",
        symbols
    )
    result = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    found = len(result)
    missing = len(symbols) - found
    if missing > 0:
        logger.warning(
            f"Instrument cache: {found}/{len(symbols)} found, "
            f"{missing} missing. Run cache_instruments() to update."
        )
    return result


def get_cached_count(db_path: Path = None) -> int:
    """Return number of cached instrument tokens."""
    db_path = db_path or MARKET_DATA_DB
    _init_instrument_cache(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM instrument_cache")
    count = cursor.fetchone()[0]
    conn.close()
    return count


# =========================================================================
# Data Availability Check
# =========================================================================

def check_data_coverage(
    universe: Nifty500Universe,
    db_path: Path = None
) -> Dict:
    """
    Check how many Nifty 500 stocks have historical price data.

    Returns:
        Dict with coverage stats and lists of missing symbols
    """
    db_path = db_path or MARKET_DATA_DB

    if not db_path.exists():
        return {
            'total_universe': len(universe),
            'have_daily_data': 0,
            'missing_daily': universe.symbols,
            'coverage_pct': 0.0,
        }

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT symbol FROM market_data_unified
        WHERE timeframe = 'day'
    """)
    have_data = {row[0] for row in cursor.fetchall()}
    conn.close()

    universe_set = set(universe.symbols)
    covered = universe_set & have_data
    missing = universe_set - have_data

    return {
        'total_universe': len(universe),
        'have_daily_data': len(covered),
        'missing_daily': sorted(missing),
        'coverage_pct': round(len(covered) / len(universe) * 100, 1) if universe else 0,
        'extra_symbols': sorted(have_data - universe_set),
    }


# =========================================================================
# Singleton accessor
# =========================================================================

_universe: Optional[Nifty500Universe] = None


def get_nifty500() -> Nifty500Universe:
    """Get the Nifty 500 universe (singleton, loaded once)."""
    global _universe
    if _universe is None:
        _universe = load_nifty500()
    return _universe


def reload_nifty500() -> Nifty500Universe:
    """Force reload the Nifty 500 universe from CSV."""
    global _universe
    _universe = load_nifty500()
    return _universe
