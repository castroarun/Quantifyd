"""
Holdings Service
================

Provides holdings data processing and fundamental data from Yahoo Finance.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

import yfinance as yf
import pandas as pd

from .kite_service import get_kite

# Setup logging
logger = logging.getLogger(__name__)

# Cache directory for fundamental data
CACHE_DIR = Path(__file__).parent.parent / "backtest_data" / "fundamentals_cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_HOURS = 24  # Refresh fundamental data daily

# NSE symbol to Yahoo Finance symbol mapping
NSE_TO_YAHOO: Dict[str, str] = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "AXISBANK": "AXISBANK.NS",
    "LT": "LT.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "TITAN": "TITAN.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "POWERGRID": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "M&M": "M&M.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "COALINDIA": "COALINDIA.NS",
    "ONGC": "ONGC.NS",
    "BPCL": "BPCL.NS",
    "IOC": "IOC.NS",
    "GRASIM": "GRASIM.NS",
    "CIPLA": "CIPLA.NS",
    "DRREDDY": "DRREDDY.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "SBILIFE": "SBILIFE.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "UPL": "UPL.NS",
    "SHREECEM": "SHREECEM.NS",
}

# Sector mapping for stocks
STOCK_SECTORS: Dict[str, str] = {
    "RELIANCE": "Energy",
    "TCS": "IT",
    "HDFCBANK": "Banking",
    "INFY": "IT",
    "ICICIBANK": "Banking",
    "KOTAKBANK": "Banking",
    "SBIN": "Banking",
    "BHARTIARTL": "Telecom",
    "ITC": "FMCG",
    "AXISBANK": "Banking",
    "LT": "Capital Goods",
    "HINDUNILVR": "FMCG",
    "ASIANPAINT": "Consumer Durables",
    "MARUTI": "Auto",
    "TITAN": "Consumer Durables",
    "SUNPHARMA": "Pharma",
    "BAJFINANCE": "Financial Services",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    "ULTRACEMCO": "Cement",
    "POWERGRID": "Power",
    "NTPC": "Power",
    "NESTLEIND": "FMCG",
    "M&M": "Auto",
    "TATAMOTORS": "Auto",
    "TATASTEEL": "Metals",
    "JSWSTEEL": "Metals",
    "ADANIENT": "Diversified",
    "ADANIPORTS": "Infrastructure",
    "COALINDIA": "Mining",
    "ONGC": "Energy",
    "BPCL": "Energy",
    "IOC": "Energy",
    "GRASIM": "Cement",
    "CIPLA": "Pharma",
    "DRREDDY": "Pharma",
    "DIVISLAB": "Pharma",
    "APOLLOHOSP": "Healthcare",
    "BRITANNIA": "FMCG",
    "TATACONSUM": "FMCG",
    "EICHERMOT": "Auto",
    "HEROMOTOCO": "Auto",
    "BAJAJ-AUTO": "Auto",
    "BAJAJFINSV": "Financial Services",
    "INDUSINDBK": "Banking",
    "SBILIFE": "Insurance",
    "HDFCLIFE": "Insurance",
    "UPL": "Chemicals",
    "SHREECEM": "Cement",
}

# Industry P/E benchmarks (approximate)
INDUSTRY_PE: Dict[str, float] = {
    "Banking": 15.0,
    "IT": 28.0,
    "FMCG": 45.0,
    "Energy": 12.0,
    "Telecom": 40.0,
    "Auto": 22.0,
    "Pharma": 30.0,
    "Financial Services": 20.0,
    "Metals": 10.0,
    "Cement": 25.0,
    "Power": 15.0,
    "Infrastructure": 18.0,
    "Healthcare": 35.0,
    "Insurance": 65.0,
    "Consumer Durables": 55.0,
    "Capital Goods": 28.0,
    "Chemicals": 25.0,
    "Diversified": 20.0,
    "Mining": 8.0,
}

# Stock logo abbreviations
STOCK_LOGOS: Dict[str, str] = {
    "RELIANCE": "RIL",
    "TCS": "TCS",
    "HDFCBANK": "HDFC",
    "INFY": "INFY",
    "ICICIBANK": "ICICI",
    "KOTAKBANK": "KMB",
    "SBIN": "SBI",
    "BHARTIARTL": "ARTL",
    "ITC": "ITC",
    "AXISBANK": "AXIS",
    "LT": "L&T",
    "HINDUNILVR": "HUL",
    "ASIANPAINT": "APNT",
    "MARUTI": "MSIL",
    "TITAN": "TITN",
    "SUNPHARMA": "SUN",
    "BAJFINANCE": "BAF",
    "WIPRO": "WPRO",
    "HCLTECH": "HCLT",
    "TECHM": "TM",
    "HAL": "HAL",
    "ONGC": "ONGC",
    "BPCL": "BPCL",
    "COALINDIA": "COAL",
    "NTPC": "NTPC",
    "POWERGRID": "PGRD",
}

# Stock full company names
STOCK_NAMES: Dict[str, str] = {
    "RELIANCE": "Reliance Industries Ltd",
    "TCS": "Tata Consultancy Services Ltd",
    "HDFCBANK": "HDFC Bank Ltd",
    "INFY": "Infosys Ltd",
    "ICICIBANK": "ICICI Bank Ltd",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel Ltd",
    "ITC": "ITC Ltd",
    "AXISBANK": "Axis Bank Ltd",
    "LT": "Larsen & Toubro Ltd",
    "HINDUNILVR": "Hindustan Unilever Ltd",
    "ASIANPAINT": "Asian Paints Ltd",
    "MARUTI": "Maruti Suzuki India Ltd",
    "TITAN": "Titan Company Ltd",
    "SUNPHARMA": "Sun Pharmaceutical Ind Ltd",
    "BAJFINANCE": "Bajaj Finance Ltd",
    "WIPRO": "Wipro Ltd",
    "HCLTECH": "HCL Technologies Ltd",
    "TECHM": "Tech Mahindra Ltd",
    "ULTRACEMCO": "UltraTech Cement Ltd",
    "POWERGRID": "Power Grid Corp of India Ltd",
    "NTPC": "NTPC Ltd",
    "NESTLEIND": "Nestle India Ltd",
    "M&M": "Mahindra & Mahindra Ltd",
    "TATAMOTORS": "Tata Motors Ltd",
    "TATASTEEL": "Tata Steel Ltd",
    "JSWSTEEL": "JSW Steel Ltd",
    "ADANIENT": "Adani Enterprises Ltd",
    "ADANIPORTS": "Adani Ports and SEZ Ltd",
    "COALINDIA": "Coal India Ltd",
    "ONGC": "Oil & Natural Gas Corp Ltd",
    "BPCL": "Bharat Petroleum Corp Ltd",
    "IOC": "Indian Oil Corp Ltd",
    "GRASIM": "Grasim Industries Ltd",
    "CIPLA": "Cipla Ltd",
    "DRREDDY": "Dr Reddy's Laboratories Ltd",
    "DIVISLAB": "Divi's Laboratories Ltd",
    "APOLLOHOSP": "Apollo Hospitals Enterprise Ltd",
    "BRITANNIA": "Britannia Industries Ltd",
    "TATACONSUM": "Tata Consumer Products Ltd",
    "EICHERMOT": "Eicher Motors Ltd",
    "HEROMOTOCO": "Hero MotoCorp Ltd",
    "BAJAJ-AUTO": "Bajaj Auto Ltd",
    "BAJAJFINSV": "Bajaj Finserv Ltd",
    "INDUSINDBK": "IndusInd Bank Ltd",
    "SBILIFE": "SBI Life Insurance Co Ltd",
    "HDFCLIFE": "HDFC Life Insurance Co Ltd",
    "UPL": "UPL Ltd",
    "SHREECEM": "Shree Cement Ltd",
    "HAL": "Hindustan Aeronautics Ltd",
    "BEL": "Bharat Electronics Ltd",
    "BHEL": "Bharat Heavy Electricals Ltd",
    "GAIL": "GAIL (India) Ltd",
    "SAIL": "Steel Authority of India Ltd",
    "VEDL": "Vedanta Ltd",
    "HINDALCO": "Hindalco Industries Ltd",
    "JINDALSTEL": "Jindal Steel & Power Ltd",
    "GODREJCP": "Godrej Consumer Products Ltd",
    "DABUR": "Dabur India Ltd",
    "COLPAL": "Colgate-Palmolive (India) Ltd",
    "MARICO": "Marico Ltd",
    "PIDILITIND": "Pidilite Industries Ltd",
    "BERGEPAINT": "Berger Paints India Ltd",
    "HAVELLS": "Havells India Ltd",
    "VOLTAS": "Voltas Ltd",
    "SIEMENS": "Siemens Ltd",
    "ABB": "ABB India Ltd",
    "CUMMINSIND": "Cummins India Ltd",
}

# Stock descriptions
STOCK_DESCRIPTIONS: Dict[str, str] = {
    "RELIANCE": "India's largest private sector company. Diversified in petrochemicals, refining, telecom (Jio), and retail.",
    "TCS": "India's largest IT services company. Global leader in consulting with 150+ locations worldwide.",
    "HDFCBANK": "India's largest private sector bank by assets. Known for strong asset quality and digital leadership.",
    "INFY": "India's 2nd largest IT company. Pioneer in digital transformation and AI/ML services.",
    "ICICIBANK": "India's 2nd largest private bank with strong retail and digital banking presence.",
    "KOTAKBANK": "Major private sector bank known for conservative approach and strong management.",
    "SBIN": "India's largest public sector bank with dominant market share in deposits and loans.",
    "BHARTIARTL": "India's 2nd largest telecom operator with strong 5G rollout and Africa presence.",
    "ITC": "Diversified conglomerate in FMCG, hotels, paperboards. Known for high dividend yield.",
    "AXISBANK": "3rd largest private sector bank in India with growing digital presence.",
}


def get_yahoo_symbol(nse_symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance symbol"""
    return NSE_TO_YAHOO.get(nse_symbol, f"{nse_symbol}.NS")


def get_holdings() -> List[Dict[str, Any]]:
    """
    Get holdings from Zerodha Kite API.
    Returns list of holdings with P/L calculations.
    """
    try:
        kite = get_kite()
        holdings = kite.holdings()

        # Process each holding
        processed = []
        total_invested = 0

        for h in holdings:
            invested = h.get("quantity", 0) * h.get("average_price", 0)
            current = h.get("quantity", 0) * h.get("last_price", 0)
            pnl = current - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0

            total_invested += invested

            symbol = h.get("tradingsymbol", "")
            processed.append({
                "symbol": symbol,
                "name": STOCK_NAMES.get(symbol, symbol),  # Use proper company name
                "quantity": h.get("quantity", 0),
                "average_price": h.get("average_price", 0),
                "last_price": h.get("last_price", 0),
                "invested": invested,
                "current": current,
                "pnl": pnl,
                "pnl_pct": round(pnl_pct, 2),
                "is_profit": pnl >= 0,
                "sector": STOCK_SECTORS.get(symbol, "Other"),
                "logo": STOCK_LOGOS.get(symbol, symbol[:3].upper()),
                "description": STOCK_DESCRIPTIONS.get(symbol, ""),
            })

        # Calculate portfolio percentage for each holding
        for h in processed:
            h["portfolio_pct"] = round(h["invested"] / total_invested * 100, 1) if total_invested > 0 else 0

        return processed

    except Exception as e:
        logger.error(f"Error fetching holdings: {e}")
        return []


def get_fundamentals(symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get fundamental data for a stock from Yahoo Finance.
    Uses caching to avoid repeated API calls.

    Args:
        symbol: NSE stock symbol (e.g., "RELIANCE")
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        Dictionary with fundamental data
    """
    cache_file = CACHE_DIR / f"{symbol}.json"

    # Check cache first
    if not force_refresh and cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text())
            cache_time = datetime.fromisoformat(cache_data.get("cached_at", "2000-01-01"))
            if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
                logger.info(f"Using cached fundamentals for {symbol}")
                return cache_data.get("data", {})
        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}")

    # Fetch from Yahoo Finance
    try:
        yahoo_symbol = get_yahoo_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)

        # Get stock info
        info = ticker.info or {}

        # Get financial statements
        financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials

        # Extract revenue and net income (last 5 years)
        revenue_5y = []
        profit_5y = []
        opm_5y = []

        if financials is not None and not financials.empty:
            # Revenue
            if "Total Revenue" in financials.index:
                revenue_data = financials.loc["Total Revenue"].dropna().head(5)
                revenue_5y = [round(v / 1e9, 2) for v in reversed(revenue_data.tolist())]  # In billions

            # Net Income
            if "Net Income" in financials.index:
                profit_data = financials.loc["Net Income"].dropna().head(5)
                profit_5y = [round(v / 1e9, 2) for v in reversed(profit_data.tolist())]  # In billions

            # Operating Income (for OPM calculation)
            if "Operating Income" in financials.index and "Total Revenue" in financials.index:
                op_income = financials.loc["Operating Income"].dropna().head(5)
                revenue = financials.loc["Total Revenue"].dropna().head(5)
                opm_5y = [round(op / rev * 100, 1) if rev > 0 else 0
                          for op, rev in zip(reversed(op_income.tolist()), reversed(revenue.tolist()))]

        # Get dividend info
        div_yield = info.get("dividendYield", 0) or 0
        div_yield_pct = round(div_yield * 100, 2)

        ex_div_date = info.get("exDividendDate")
        if ex_div_date:
            try:
                ex_div_date = datetime.fromtimestamp(ex_div_date).strftime("%b %Y")
            except Exception:
                ex_div_date = None

        # Get key ratios
        pe_ratio = info.get("trailingPE") or info.get("forwardPE") or 0
        de_ratio = info.get("debtToEquity", 0) or 0
        de_ratio = round(de_ratio / 100, 2) if de_ratio else 0  # Convert percentage to ratio

        roe = info.get("returnOnEquity", 0) or 0
        roe = round(roe * 100, 1) if roe else 0  # Convert to percentage

        # ROCE approximation (use ROA * (1 + D/E))
        roa = info.get("returnOnAssets", 0) or 0
        roce = round(roa * (1 + de_ratio) * 100, 1) if roa else 0

        # Get sector and industry P/E
        sector = STOCK_SECTORS.get(symbol, "Other")
        industry_pe = INDUSTRY_PE.get(sector, 20.0)

        # Build result
        result = {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "description": STOCK_DESCRIPTIONS.get(symbol, info.get("longBusinessSummary", "")[:200] if info.get("longBusinessSummary") else ""),
            "sector": sector,
            "logo": STOCK_LOGOS.get(symbol, symbol[:3].upper()),

            # Financials (5-year trends)
            "revenue_5y": revenue_5y if revenue_5y else [0, 0, 0, 0, 0],
            "profit_5y": profit_5y if profit_5y else [0, 0, 0, 0, 0],
            "opm_5y": opm_5y if opm_5y else [0, 0, 0, 0, 0],

            # Key Ratios
            "pe_ratio": round(pe_ratio, 1) if pe_ratio else 0,
            "industry_pe": industry_pe,
            "de_ratio": de_ratio,
            "roe": roe,
            "roce": roce,

            # Dividend
            "div_yield": div_yield_pct,
            "next_div_date": ex_div_date or "TBA",

            # Other info
            "market_cap": info.get("marketCap", 0),
            "book_value": info.get("bookValue", 0),
        }

        # Cache the result
        try:
            cache_data = {
                "cached_at": datetime.now().isoformat(),
                "data": result
            }
            cache_file.write_text(json.dumps(cache_data, indent=2))
            logger.info(f"Cached fundamentals for {symbol}")
        except Exception as e:
            logger.warning(f"Cache write error for {symbol}: {e}")

        return result

    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        # Return default values
        return {
            "symbol": symbol,
            "name": symbol,
            "description": STOCK_DESCRIPTIONS.get(symbol, ""),
            "sector": STOCK_SECTORS.get(symbol, "Other"),
            "logo": STOCK_LOGOS.get(symbol, symbol[:3].upper()),
            "revenue_5y": [0, 0, 0, 0, 0],
            "profit_5y": [0, 0, 0, 0, 0],
            "opm_5y": [0, 0, 0, 0, 0],
            "pe_ratio": 0,
            "industry_pe": INDUSTRY_PE.get(STOCK_SECTORS.get(symbol, "Other"), 20.0),
            "de_ratio": 0,
            "roe": 0,
            "roce": 0,
            "div_yield": 0,
            "next_div_date": "TBA",
            "market_cap": 0,
            "book_value": 0,
        }


def get_historical_prices(symbol: str, period: str = "1y") -> List[float]:
    """
    Get historical closing prices for sparkline/chart.

    Args:
        symbol: NSE stock symbol
        period: Time period (e.g., "1y", "6mo", "3mo")

    Returns:
        List of closing prices
    """
    try:
        yahoo_symbol = get_yahoo_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            return []

        return hist["Close"].tolist()

    except Exception as e:
        logger.error(f"Error fetching historical prices for {symbol}: {e}")
        return []


def get_portfolio_summary(holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate portfolio summary from holdings.

    Args:
        holdings: List of processed holdings

    Returns:
        Portfolio summary with totals
    """
    total_invested = sum(h.get("invested", 0) for h in holdings)
    total_current = sum(h.get("current", 0) for h in holdings)
    total_pnl = total_current - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    # Sector breakdown
    sectors = {}
    for h in holdings:
        sector = h.get("sector", "Other")
        if sector not in sectors:
            sectors[sector] = {"invested": 0, "current": 0, "pnl": 0, "count": 0}
        sectors[sector]["invested"] += h.get("invested", 0)
        sectors[sector]["current"] += h.get("current", 0)
        sectors[sector]["pnl"] += h.get("pnl", 0)
        sectors[sector]["count"] += 1

    # Calculate sector returns
    for sector, data in sectors.items():
        data["pnl_pct"] = round(data["pnl"] / data["invested"] * 100, 1) if data["invested"] > 0 else 0

    return {
        "total_invested": total_invested,
        "total_current": total_current,
        "total_pnl": total_pnl,
        "total_pnl_pct": round(total_pnl_pct, 1),
        "is_profit": total_pnl >= 0,
        "stock_count": len(holdings),
        "sectors": sectors,
    }


def format_currency(amount: float, prefix: str = "₹") -> str:
    """Format amount in Indian currency style (L for Lakh, Cr for Crore)"""
    if abs(amount) >= 1e7:
        return f"{prefix}{amount/1e7:.2f}Cr"
    elif abs(amount) >= 1e5:
        return f"{prefix}{amount/1e5:.2f}L"
    elif abs(amount) >= 1e3:
        return f"{prefix}{amount/1e3:.1f}K"
    else:
        return f"{prefix}{amount:.0f}"
