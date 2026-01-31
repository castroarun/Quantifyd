"""
Fundamental Data Service for Momentum + Quality Strategy
=========================================================

Fetches and computes fundamental quality metrics from Yahoo Finance.
Handles both financial (banks/NBFCs) and non-financial stocks.

Metrics computed:
- Revenue CAGR (3-year) + YoY positive growth check
- Debt-to-Equity ratio (non-financial only)
- Operating Profit Margin (3-year) + no-decline check
- ROA / ROE (financial stocks)
- Composite quality score with weighted ranking

Data source: Yahoo Finance (yfinance) for MVP.
Phase 2: Screener.in for NPA data, deeper history.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / 'data' / 'fundamentals_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_EXPIRY_HOURS = 24  # Refresh once per day


# =========================================================================
# Data Classes
# =========================================================================

@dataclass
class FundamentalCriteria:
    """Configurable thresholds for quality filtering."""

    # Revenue Growth (all stocks)
    min_revenue_growth_3y_cagr: float = 0.15
    require_revenue_positive_each_year: bool = True

    # Non-Financial: Debt
    max_debt_to_equity: float = 0.20

    # Non-Financial: Operating Margins
    min_opm_3y: float = 0.15
    require_opm_no_decline: bool = True

    # Financial (Banks/NBFCs)
    min_roa: float = 0.01  # 1.0%
    min_roe: float = 0.12  # 12%


@dataclass
class RevenueMetrics:
    """Revenue growth analysis for a stock."""
    cagr_3y: Optional[float] = None
    yoy_growth: List[float] = field(default_factory=list)
    revenue_values: List[float] = field(default_factory=list)
    is_positive_each_year: bool = False
    passes_cagr: bool = False
    passes_trend: bool = False
    data_years: int = 0


@dataclass
class DebtMetrics:
    """Debt analysis for non-financial stocks."""
    debt_to_equity: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    passes: bool = False


@dataclass
class MarginMetrics:
    """Operating margin analysis."""
    opm_3y: List[float] = field(default_factory=list)
    all_above_threshold: bool = False
    no_decline: bool = False
    passes: bool = False


@dataclass
class FinancialMetrics:
    """Bank/NBFC specific metrics."""
    roa: Optional[float] = None
    roe: Optional[float] = None
    passes_roa: bool = False
    passes_roe: bool = False
    passes: bool = False


@dataclass
class QualityScore:
    """Composite quality assessment for a stock."""
    symbol: str
    is_financial: bool
    revenue: RevenueMetrics = field(default_factory=RevenueMetrics)
    debt: DebtMetrics = field(default_factory=DebtMetrics)
    margins: MarginMetrics = field(default_factory=MarginMetrics)
    financial: FinancialMetrics = field(default_factory=FinancialMetrics)
    composite_score: float = 0.0
    passes_all: bool = False
    fail_reasons: List[str] = field(default_factory=list)
    fetched_at: Optional[str] = None


# =========================================================================
# Yahoo Finance Symbol Mapping
# =========================================================================

def get_yahoo_symbol(nse_symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance symbol."""
    special = {
        'M&M': 'M&M.NS',
        'M&MFIN': 'M&MFIN.NS',
        'L&TFH': 'L&TFH.NS',
        'NIFTY50': '^NSEI',
        'NIFTY500': '^CRSLDX',
    }
    if nse_symbol in special:
        return special[nse_symbol]
    return f"{nse_symbol}.NS"


# =========================================================================
# Core Metric Calculators
# =========================================================================

def calculate_revenue_growth(
    financials: pd.DataFrame,
    criteria: FundamentalCriteria = None,
) -> RevenueMetrics:
    """
    Calculate 3-year revenue CAGR and YoY growth trend.

    Args:
        financials: Yahoo Finance annual financials DataFrame
        criteria: Quality thresholds

    Returns:
        RevenueMetrics with CAGR, YoY growth, and pass/fail
    """
    criteria = criteria or FundamentalCriteria()
    metrics = RevenueMetrics()

    if financials is None or financials.empty:
        return metrics

    # Find revenue row
    revenue_key = None
    for key in ['Total Revenue', 'TotalRevenue', 'Revenue']:
        if key in financials.index:
            revenue_key = key
            break

    if not revenue_key:
        return metrics

    revenue_series = financials.loc[revenue_key].dropna()
    if len(revenue_series) < 2:
        return metrics

    # Yahoo returns most recent first, reverse for chronological order
    revenues = list(reversed(revenue_series.tolist()))
    metrics.revenue_values = [round(v / 1e7, 2) for v in revenues]  # In crores
    metrics.data_years = len(revenues)

    # Need at least 4 data points for 3-year CAGR
    if len(revenues) >= 4:
        try:
            if revenues[-4] > 0 and revenues[-1] > 0:
                metrics.cagr_3y = (revenues[-1] / revenues[-4]) ** (1 / 3) - 1
                metrics.passes_cagr = metrics.cagr_3y >= criteria.min_revenue_growth_3y_cagr
        except (ZeroDivisionError, ValueError):
            pass

    # YoY growth for last 3 years
    for i in range(max(1, len(revenues) - 3), len(revenues)):
        if revenues[i - 1] > 0:
            yoy = (revenues[i] - revenues[i - 1]) / revenues[i - 1]
            metrics.yoy_growth.append(round(yoy, 4))

    if metrics.yoy_growth:
        metrics.is_positive_each_year = all(g > 0 for g in metrics.yoy_growth)
        metrics.passes_trend = metrics.is_positive_each_year or not criteria.require_revenue_positive_each_year

    return metrics


def calculate_debt_to_equity(
    balance_sheet: pd.DataFrame,
    criteria: FundamentalCriteria = None,
) -> DebtMetrics:
    """
    Calculate Debt-to-Equity ratio from latest balance sheet.

    Args:
        balance_sheet: Yahoo Finance balance sheet DataFrame
        criteria: Quality thresholds

    Returns:
        DebtMetrics with D/E ratio and pass/fail
    """
    criteria = criteria or FundamentalCriteria()
    metrics = DebtMetrics()

    if balance_sheet is None or balance_sheet.empty:
        return metrics

    # Find debt field
    debt_key = None
    for key in ['Total Debt', 'TotalDebt', 'Long Term Debt']:
        if key in balance_sheet.index:
            debt_key = key
            break

    # Find equity field
    equity_key = None
    for key in ['Stockholders Equity', 'Total Stockholder Equity',
                'Common Stock Equity', 'Total Equity Gross Minority Interest']:
        if key in balance_sheet.index:
            equity_key = key
            break

    if debt_key:
        debt_series = balance_sheet.loc[debt_key].dropna()
        if len(debt_series) > 0:
            metrics.total_debt = float(debt_series.iloc[0])

    if equity_key:
        equity_series = balance_sheet.loc[equity_key].dropna()
        if len(equity_series) > 0:
            metrics.total_equity = float(equity_series.iloc[0])

    if metrics.total_debt is not None and metrics.total_equity and metrics.total_equity > 0:
        metrics.debt_to_equity = metrics.total_debt / metrics.total_equity
        metrics.passes = metrics.debt_to_equity <= criteria.max_debt_to_equity
    elif metrics.total_debt is None or metrics.total_debt == 0:
        # No debt = passes
        metrics.debt_to_equity = 0.0
        metrics.passes = True

    return metrics


def calculate_opm(
    financials: pd.DataFrame,
    criteria: FundamentalCriteria = None,
) -> MarginMetrics:
    """
    Calculate Operating Profit Margin for last 3 years.

    Args:
        financials: Yahoo Finance annual financials DataFrame
        criteria: Quality thresholds

    Returns:
        MarginMetrics with OPM trend and pass/fail
    """
    criteria = criteria or FundamentalCriteria()
    metrics = MarginMetrics()

    if financials is None or financials.empty:
        return metrics

    # Find operating income
    oi_key = None
    for key in ['Operating Income', 'OperatingIncome', 'EBIT']:
        if key in financials.index:
            oi_key = key
            break

    rev_key = None
    for key in ['Total Revenue', 'TotalRevenue', 'Revenue']:
        if key in financials.index:
            rev_key = key
            break

    if not oi_key or not rev_key:
        return metrics

    oi_series = financials.loc[oi_key].dropna()
    rev_series = financials.loc[rev_key].dropna()

    # Align by columns (dates)
    common_cols = oi_series.index.intersection(rev_series.index)
    if len(common_cols) < 3:
        return metrics

    # Take last 3 years, reversed for chronological
    oi_vals = list(reversed(oi_series[common_cols].head(3).tolist()))
    rev_vals = list(reversed(rev_series[common_cols].head(3).tolist()))

    for oi, rev in zip(oi_vals, rev_vals):
        if rev > 0:
            metrics.opm_3y.append(round(oi / rev, 4))
        else:
            metrics.opm_3y.append(0.0)

    if len(metrics.opm_3y) >= 3:
        metrics.all_above_threshold = all(
            opm >= criteria.min_opm_3y for opm in metrics.opm_3y
        )
        metrics.no_decline = all(
            metrics.opm_3y[i] >= metrics.opm_3y[i - 1]
            for i in range(1, len(metrics.opm_3y))
        )
        metrics.passes = metrics.all_above_threshold and (
            metrics.no_decline or not criteria.require_opm_no_decline
        )

    return metrics


def calculate_financial_metrics(
    info: Dict,
    criteria: FundamentalCriteria = None,
) -> FinancialMetrics:
    """
    Calculate ROA and ROE for financial stocks (banks/NBFCs).

    Args:
        info: Yahoo Finance ticker.info dict
        criteria: Quality thresholds

    Returns:
        FinancialMetrics with ROA/ROE and pass/fail
    """
    criteria = criteria or FundamentalCriteria()
    metrics = FinancialMetrics()

    roa = info.get('returnOnAssets', 0) or 0
    roe = info.get('returnOnEquity', 0) or 0

    metrics.roa = round(roa, 4) if roa else None
    metrics.roe = round(roe, 4) if roe else None

    if metrics.roa is not None:
        metrics.passes_roa = metrics.roa >= criteria.min_roa
    if metrics.roe is not None:
        metrics.passes_roe = metrics.roe >= criteria.min_roe

    metrics.passes = metrics.passes_roa and metrics.passes_roe
    return metrics


# =========================================================================
# Composite Quality Assessment
# =========================================================================

def assess_quality(
    symbol: str,
    is_financial: bool,
    criteria: FundamentalCriteria = None,
    use_cache: bool = True,
) -> QualityScore:
    """
    Full quality assessment for a single stock.

    Fetches data from Yahoo Finance and computes all quality metrics.

    Args:
        symbol: NSE trading symbol
        is_financial: Whether this is a bank/NBFC
        criteria: Quality thresholds
        use_cache: Whether to use cached data

    Returns:
        QualityScore with all metrics and composite score
    """
    if yf is None:
        raise ImportError("yfinance is required: pip install yfinance")

    criteria = criteria or FundamentalCriteria()
    score = QualityScore(symbol=symbol, is_financial=is_financial)

    # Check cache
    if use_cache:
        cached = _load_cache(symbol)
        if cached:
            return cached

    yahoo_sym = get_yahoo_symbol(symbol)

    try:
        ticker = yf.Ticker(yahoo_sym)
        info = ticker.info or {}
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet

        # Revenue growth (all stocks)
        score.revenue = calculate_revenue_growth(financials, criteria)

        if is_financial:
            # Banks/NBFCs: ROA + ROE instead of D/E
            score.financial = calculate_financial_metrics(info, criteria)
            score.margins = calculate_opm(financials, criteria)

            if not score.revenue.passes_cagr:
                score.fail_reasons.append(
                    f"Revenue CAGR {score.revenue.cagr_3y:.1%} < {criteria.min_revenue_growth_3y_cagr:.0%}"
                    if score.revenue.cagr_3y is not None else "Revenue data insufficient"
                )
            if not score.revenue.passes_trend:
                score.fail_reasons.append("Revenue declined YoY in at least one year")
            if not score.financial.passes_roa:
                score.fail_reasons.append(
                    f"ROA {score.financial.roa:.2%} < {criteria.min_roa:.1%}"
                    if score.financial.roa is not None else "ROA data missing"
                )
            if not score.financial.passes_roe:
                score.fail_reasons.append(
                    f"ROE {score.financial.roe:.2%} < {criteria.min_roe:.0%}"
                    if score.financial.roe is not None else "ROE data missing"
                )

            score.passes_all = (
                score.revenue.passes_cagr
                and score.revenue.passes_trend
                and score.financial.passes
            )

            # Composite score for ranking (financial)
            rev_score = min(score.revenue.cagr_3y / 0.30, 1.0) if score.revenue.cagr_3y else 0
            roa_score = min(score.financial.roa / 0.02, 1.0) if score.financial.roa else 0
            roe_score = min(score.financial.roe / 0.20, 1.0) if score.financial.roe else 0
            score.composite_score = round(
                rev_score * 0.35 + roa_score * 0.35 + roe_score * 0.30, 4
            )

        else:
            # Non-financial: D/E + OPM
            score.debt = calculate_debt_to_equity(balance_sheet, criteria)
            score.margins = calculate_opm(financials, criteria)

            if not score.revenue.passes_cagr:
                score.fail_reasons.append(
                    f"Revenue CAGR {score.revenue.cagr_3y:.1%} < {criteria.min_revenue_growth_3y_cagr:.0%}"
                    if score.revenue.cagr_3y is not None else "Revenue data insufficient"
                )
            if not score.revenue.passes_trend:
                score.fail_reasons.append("Revenue declined YoY in at least one year")
            if not score.debt.passes:
                score.fail_reasons.append(
                    f"D/E {score.debt.debt_to_equity:.2f} > {criteria.max_debt_to_equity:.2f}"
                    if score.debt.debt_to_equity is not None else "D/E data missing"
                )
            if not score.margins.passes:
                if not score.margins.all_above_threshold:
                    score.fail_reasons.append(
                        f"OPM below {criteria.min_opm_3y:.0%} in at least one year"
                    )
                if not score.margins.no_decline:
                    score.fail_reasons.append("OPM declined YoY in at least one year")

            score.passes_all = (
                score.revenue.passes_cagr
                and score.revenue.passes_trend
                and score.debt.passes
                and score.margins.passes
            )

            # Composite score for ranking (non-financial)
            # Weights: Revenue 30%, D/E 25%, OPM 25%, OPM growth 20%
            rev_score = min(score.revenue.cagr_3y / 0.30, 1.0) if score.revenue.cagr_3y else 0
            de_score = max(0, 1.0 - (score.debt.debt_to_equity or 1.0) / 0.50) if score.debt.debt_to_equity is not None else 0
            opm_score = min(np.mean(score.margins.opm_3y) / 0.30, 1.0) if score.margins.opm_3y else 0
            opm_growth = 1.0 if score.margins.no_decline else 0.5
            score.composite_score = round(
                rev_score * 0.30 + de_score * 0.25 + opm_score * 0.25 + opm_growth * 0.20, 4
            )

        score.fetched_at = datetime.now().isoformat()

        # Save to cache
        _save_cache(symbol, score)

        # Rate limit: avoid hammering Yahoo Finance
        time.sleep(0.5)

    except Exception as e:
        logger.error(f"Error assessing {symbol}: {e}")
        score.fail_reasons.append(f"Data fetch error: {str(e)}")

    return score


def assess_quality_batch(
    symbols: List[str],
    is_financial_fn=None,
    criteria: FundamentalCriteria = None,
    use_cache: bool = True,
    progress_callback=None,
) -> List[QualityScore]:
    """
    Batch quality assessment for multiple stocks.

    Args:
        symbols: List of NSE symbols
        is_financial_fn: Function(symbol) -> bool for financial detection
        criteria: Quality thresholds
        use_cache: Whether to use cached data
        progress_callback: Optional callback(idx, total, symbol, status)

    Returns:
        List of QualityScore, sorted by composite_score descending
    """
    results = []
    total = len(symbols)

    for idx, symbol in enumerate(symbols, 1):
        is_fin = is_financial_fn(symbol) if is_financial_fn else False

        if progress_callback:
            progress_callback(idx, total, symbol, 'fetching')

        score = assess_quality(symbol, is_fin, criteria, use_cache)
        results.append(score)

        if progress_callback:
            status = 'pass' if score.passes_all else 'fail'
            progress_callback(idx, total, symbol, status)

        logger.info(
            f"[{idx}/{total}] {symbol}: "
            f"{'PASS' if score.passes_all else 'FAIL'} "
            f"(score={score.composite_score:.3f})"
        )

    # Sort by composite score descending
    results.sort(key=lambda s: s.composite_score, reverse=True)
    return results


def rank_quality_stocks(
    scores: List[QualityScore],
    top_n: int = 30,
) -> List[QualityScore]:
    """
    Filter passing stocks and return top N by composite score.

    Args:
        scores: List of QualityScore from assess_quality_batch
        top_n: Number of top stocks to return

    Returns:
        Top N passing stocks by composite score
    """
    passing = [s for s in scores if s.passes_all]
    passing.sort(key=lambda s: s.composite_score, reverse=True)
    return passing[:top_n]


# =========================================================================
# Cache Management
# =========================================================================

def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.json"


def _load_cache(symbol: str) -> Optional[QualityScore]:
    """Load cached quality score if fresh enough."""
    path = _cache_path(symbol)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        fetched_at = data.get('fetched_at')
        if fetched_at:
            cached_time = datetime.fromisoformat(fetched_at)
            if datetime.now() - cached_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                return _dict_to_score(data)
    except Exception as e:
        logger.debug(f"Cache read error for {symbol}: {e}")
    return None


def _save_cache(symbol: str, score: QualityScore):
    """Save quality score to cache."""
    try:
        data = _score_to_dict(score)
        _cache_path(symbol).write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.debug(f"Cache write error for {symbol}: {e}")


def _score_to_dict(score: QualityScore) -> Dict:
    """Convert QualityScore to serializable dict."""
    return {
        'symbol': score.symbol,
        'is_financial': score.is_financial,
        'composite_score': score.composite_score,
        'passes_all': score.passes_all,
        'fail_reasons': score.fail_reasons,
        'fetched_at': score.fetched_at,
        'revenue': {
            'cagr_3y': score.revenue.cagr_3y,
            'yoy_growth': score.revenue.yoy_growth,
            'revenue_values': score.revenue.revenue_values,
            'is_positive_each_year': score.revenue.is_positive_each_year,
            'passes_cagr': score.revenue.passes_cagr,
            'passes_trend': score.revenue.passes_trend,
            'data_years': score.revenue.data_years,
        },
        'debt': {
            'debt_to_equity': score.debt.debt_to_equity,
            'total_debt': score.debt.total_debt,
            'total_equity': score.debt.total_equity,
            'passes': score.debt.passes,
        },
        'margins': {
            'opm_3y': score.margins.opm_3y,
            'all_above_threshold': score.margins.all_above_threshold,
            'no_decline': score.margins.no_decline,
            'passes': score.margins.passes,
        },
        'financial': {
            'roa': score.financial.roa,
            'roe': score.financial.roe,
            'passes_roa': score.financial.passes_roa,
            'passes_roe': score.financial.passes_roe,
            'passes': score.financial.passes,
        },
    }


def _dict_to_score(data: Dict) -> QualityScore:
    """Reconstruct QualityScore from cached dict."""
    rev = data.get('revenue', {})
    debt = data.get('debt', {})
    margins = data.get('margins', {})
    fin = data.get('financial', {})

    return QualityScore(
        symbol=data['symbol'],
        is_financial=data['is_financial'],
        composite_score=data.get('composite_score', 0),
        passes_all=data.get('passes_all', False),
        fail_reasons=data.get('fail_reasons', []),
        fetched_at=data.get('fetched_at'),
        revenue=RevenueMetrics(
            cagr_3y=rev.get('cagr_3y'),
            yoy_growth=rev.get('yoy_growth', []),
            revenue_values=rev.get('revenue_values', []),
            is_positive_each_year=rev.get('is_positive_each_year', False),
            passes_cagr=rev.get('passes_cagr', False),
            passes_trend=rev.get('passes_trend', False),
            data_years=rev.get('data_years', 0),
        ),
        debt=DebtMetrics(
            debt_to_equity=debt.get('debt_to_equity'),
            total_debt=debt.get('total_debt'),
            total_equity=debt.get('total_equity'),
            passes=debt.get('passes', False),
        ),
        margins=MarginMetrics(
            opm_3y=margins.get('opm_3y', []),
            all_above_threshold=margins.get('all_above_threshold', False),
            no_decline=margins.get('no_decline', False),
            passes=margins.get('passes', False),
        ),
        financial=FinancialMetrics(
            roa=fin.get('roa'),
            roe=fin.get('roe'),
            passes_roa=fin.get('passes_roa', False),
            passes_roe=fin.get('passes_roe', False),
            passes=fin.get('passes', False),
        ),
    )


def clear_cache(symbol: str = None):
    """Clear fundamentals cache. If symbol given, clear only that stock."""
    if symbol:
        path = _cache_path(symbol)
        if path.exists():
            path.unlink()
            logger.info(f"Cleared cache for {symbol}")
    else:
        count = 0
        for f in CACHE_DIR.glob('*.json'):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached fundamentals")
