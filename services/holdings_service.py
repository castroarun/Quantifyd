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
    "HAL": "HAL.NS",
    "BEL": "BEL.NS",
    "BHEL": "BHEL.NS",
    "GAIL": "GAIL.NS",
    "SAIL": "SAIL.NS",
    "VEDL": "VEDL.NS",
    "HINDALCO": "HINDALCO.NS",
    "JINDALSTEL": "JINDALSTEL.NS",
    "GODREJCP": "GODREJCP.NS",
    "DABUR": "DABUR.NS",
    "COLPAL": "COLPAL.NS",
    "MARICO": "MARICO.NS",
    "PIDILITIND": "PIDILITIND.NS",
    "BERGEPAINT": "BERGEPAINT.NS",
    "HAVELLS": "HAVELLS.NS",
    "VOLTAS": "VOLTAS.NS",
    "SIEMENS": "SIEMENS.NS",
    "ABB": "ABB.NS",
    "CUMMINSIND": "CUMMINSIND.NS",
    "TRENT": "TRENT.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    # Index
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NYKAA": "NYKAA.NS",
    "DMART": "DMART.NS",
    "POLYCAB": "POLYCAB.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "LTIM": "LTIM.NS",
    "MPHASIS": "MPHASIS.NS",
    "COFORGE": "COFORGE.NS",
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
    "HAL": "Defence",
    "BEL": "Defence",
    "BHEL": "Capital Goods",
    "GAIL": "Energy",
    "SAIL": "Metals",
    "VEDL": "Metals",
    "HINDALCO": "Metals",
    "JINDALSTEL": "Metals",
    "GODREJCP": "FMCG",
    "DABUR": "FMCG",
    "COLPAL": "FMCG",
    "MARICO": "FMCG",
    "PIDILITIND": "Chemicals",
    "BERGEPAINT": "Consumer Durables",
    "HAVELLS": "Consumer Durables",
    "VOLTAS": "Consumer Durables",
    "SIEMENS": "Capital Goods",
    "ABB": "Capital Goods",
    "CUMMINSIND": "Capital Goods",
    "TRENT": "Retail",
    "ZOMATO": "Internet",
    "PAYTM": "Fintech",
    "NYKAA": "Internet",
    "DMART": "Retail",
    "POLYCAB": "Capital Goods",
    "PERSISTENT": "IT",
    "LTIM": "IT",
    "MPHASIS": "IT",
    "COFORGE": "IT",
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
    "Defence": 35.0,
    "Retail": 60.0,
    "Internet": 100.0,
    "Fintech": 50.0,
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
    "HAL": "India's largest aerospace and defence company. Manufactures fighter jets, helicopters, and UAVs.",
    "BEL": "Leading defence electronics company. Key player in radar, sonar, and communication systems.",
    "LT": "India's largest engineering company. Diversified in infrastructure, technology, and defence.",
    "TITAN": "Leading lifestyle company. Dominant in watches, jewellery (Tanishq), and eyewear.",
    "BAJFINANCE": "India's largest non-banking lender. Leader in consumer finance and digital lending.",
    "SUNPHARMA": "India's largest pharma company globally. Strong specialty portfolio.",
    "MARUTI": "India's largest passenger vehicle manufacturer with 40%+ market share.",
    "TATAMOTORS": "Global automaker. Owns Jaguar Land Rover. Leader in EVs in India.",
    "TATASTEEL": "India's largest private steel producer. Strong European presence via Tata Steel Europe.",
    "NTPC": "India's largest power generator with 70+ GW capacity. Transitioning to renewables.",
    "COALINDIA": "World's largest coal producer. Supplies 80%+ of India's coal requirement.",
}


def get_yahoo_symbol(nse_symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance symbol"""
    return NSE_TO_YAHOO.get(nse_symbol, f"{nse_symbol}.NS")


def _get_market_rank(market_cap: int) -> Optional[int]:
    """
    Get approximate market rank in Nifty 500 based on market cap tiers.
    This is an approximation - for exact ranking, all 500 stocks would need to be compared.

    Market Cap Tiers (approx. as of 2024):
    - Top 10: > 10L Cr (₹10,00,000 Cr)
    - Top 20: > 5L Cr
    - Top 50: > 1.5L Cr
    - Top 100: > 50,000 Cr
    - Top 200: > 20,000 Cr
    - Top 500: > 5,000 Cr
    """
    if not market_cap:
        return None

    # Convert to Crores (1 Cr = 10 million = 1e7)
    mcap_cr = market_cap / 1e7

    if mcap_cr >= 1000000:  # > 10L Cr
        return 5  # Top 10 approx
    elif mcap_cr >= 500000:  # > 5L Cr
        return 15  # Top 20 approx
    elif mcap_cr >= 150000:  # > 1.5L Cr
        return 35  # Top 50 approx
    elif mcap_cr >= 50000:  # > 50K Cr
        return 75  # Top 100 approx
    elif mcap_cr >= 20000:  # > 20K Cr
        return 150  # Top 200 approx
    elif mcap_cr >= 5000:  # > 5K Cr
        return 350  # Top 500 approx
    else:
        return None  # Below Nifty 500


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

        # Calculate portfolio percentages for each holding
        total_current = sum(h["current"] for h in processed)
        for h in processed:
            h["portfolio_pct"] = round(h["invested"] / total_invested * 100, 1) if total_invested > 0 else 0
            h["current_pct"] = round(h["current"] / total_current * 100, 1) if total_current > 0 else 0

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

        # Get dividend info - handle Yahoo Finance inconsistency
        # Some stocks return dividendYield as decimal (0.0092 = 0.92%)
        # Others seem to return it already multiplied (0.92 = 0.92%)
        div_yield = info.get("dividendYield", 0) or 0

        # Sanity check: if div_yield > 0.5 (50%), use alternative calculation
        if div_yield > 0.5:
            # Try trailingAnnualDividendYield or calculate from rate/price
            trailing_yield = info.get("trailingAnnualDividendYield", 0) or 0
            if trailing_yield and trailing_yield < 0.5:
                div_yield_pct = round(trailing_yield * 100, 2)
            else:
                # Calculate from dividend rate and current price
                div_rate = info.get("dividendRate", 0) or 0
                current_price = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0
                if div_rate and current_price:
                    div_yield_pct = round((div_rate / current_price) * 100, 2)
                else:
                    div_yield_pct = 0
        else:
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

        # Additional ratios
        current_ratio = info.get("currentRatio", 0) or 0
        quick_ratio = info.get("quickRatio", 0) or 0
        pb_ratio = info.get("priceToBook", 0) or 0
        ps_ratio = info.get("priceToSalesTrailing12Months", 0) or 0
        ev_ebitda = info.get("enterpriseToEbitda", 0) or 0
        enterprise_value = info.get("enterpriseValue", 0) or 0
        payout_ratio = info.get("payoutRatio", 0) or 0
        if payout_ratio:
            payout_ratio = round(payout_ratio * 100, 1)

        # Additional stats
        high_52w = info.get("fiftyTwoWeekHigh", 0) or 0
        low_52w = info.get("fiftyTwoWeekLow", 0) or 0
        avg_volume = info.get("averageVolume", 0) or 0
        beta = info.get("beta", 0) or 0
        eps = info.get("trailingEps", 0) or info.get("forwardEps", 0) or 0
        employees = info.get("fullTimeEmployees", 0) or 0
        website = info.get("website", "") or ""
        industry = info.get("industry", "") or ""

        # Dividend history (5 years) - try to get from dividends
        dividend_5y = [0, 0, 0, 0, 0]
        try:
            dividends = ticker.dividends
            if dividends is not None and len(dividends) > 0:
                # Group by year and sum
                div_by_year = dividends.groupby(dividends.index.year).sum()
                years = sorted(div_by_year.index)[-5:]
                dividend_5y = [round(div_by_year.get(y, 0), 2) for y in years]
                # Pad to 5 years if needed
                while len(dividend_5y) < 5:
                    dividend_5y.insert(0, 0)
        except Exception:
            pass

        # Helper to find field by multiple possible names
        def find_field(df, names):
            """Find first matching field name from list of alternatives"""
            if df is None:
                return None
            for name in names:
                if name in df.index:
                    return name
            return None

        # Get balance sheet and cash flow
        balance_sheet = None
        cashflow = None
        try:
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            logger.debug(f"{symbol}: Balance sheet fields: {list(balance_sheet.index) if balance_sheet is not None else 'None'}")
            logger.debug(f"{symbol}: Cash flow fields: {list(cashflow.index) if cashflow is not None else 'None'}")
        except Exception as e:
            logger.warning(f"{symbol}: Error fetching balance sheet/cashflow: {e}")

        # 5-year ROE (calculated from Net Income / Stockholders Equity)
        roe_5y = []
        try:
            if financials is not None and balance_sheet is not None:
                ni_key = find_field(financials, ["Net Income", "Net Income Common Stockholders", "NetIncome"])
                eq_key = find_field(balance_sheet, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest", "StockholdersEquity"])
                if ni_key and eq_key:
                    ni_series = financials.loc[ni_key].dropna().head(5)
                    eq_series = balance_sheet.loc[eq_key].dropna().head(5)
                    for ni, eq in zip(ni_series, eq_series):
                        if eq and eq > 0:
                            roe_5y.append(round(ni / eq * 100, 1))
                    roe_5y = list(reversed(roe_5y))
                else:
                    logger.debug(f"{symbol}: ROE calc missing - NI key: {ni_key}, Equity key: {eq_key}")
        except Exception as e:
            logger.warning(f"{symbol}: ROE calculation error: {e}")

        # 5-year ROCE (Return on Capital Employed = EBIT / Capital Employed)
        roce_5y = []
        try:
            if balance_sheet is not None and financials is not None:
                ebit_key = find_field(financials, ["EBIT", "Operating Income", "Ebit", "OperatingIncome"])
                assets_key = find_field(balance_sheet, ["Total Assets", "TotalAssets"])
                liab_key = find_field(balance_sheet, ["Current Liabilities", "Total Current Liabilities", "CurrentLiabilities"])
                if ebit_key and assets_key and liab_key:
                    ebit_series = financials.loc[ebit_key].dropna().head(5)
                    assets_series = balance_sheet.loc[assets_key].dropna().head(5)
                    liab_series = balance_sheet.loc[liab_key].dropna().head(5)
                    for ebit, assets, liab in zip(ebit_series, assets_series, liab_series):
                        cap_employed = assets - liab
                        if cap_employed and cap_employed > 0:
                            roce_5y.append(round(ebit / cap_employed * 100, 1))
                    roce_5y = list(reversed(roce_5y))
                else:
                    logger.debug(f"{symbol}: ROCE calc missing - EBIT: {ebit_key}, Assets: {assets_key}, Liab: {liab_key}")
        except Exception as e:
            logger.warning(f"{symbol}: ROCE calculation error: {e}")

        # 5-year Total Debt from balance sheet
        debt_5y = []
        try:
            if balance_sheet is not None:
                debt_key = find_field(balance_sheet, ["Total Debt", "TotalDebt", "Long Term Debt", "Total Non Current Liabilities Net Minority Interest"])
                if debt_key:
                    debt_series = balance_sheet.loc[debt_key].dropna().head(5)
                    debt_5y = [round(v / 1e9, 2) for v in reversed(debt_series.tolist())]
                else:
                    logger.debug(f"{symbol}: Debt calc missing - no debt field found in balance sheet")
        except Exception as e:
            logger.warning(f"{symbol}: Debt calculation error: {e}")

        # 5-year FCF and Capex from cash flow
        fcf_5y = []
        capex_5y = []
        try:
            if cashflow is not None:
                fcf_key = find_field(cashflow, ["Free Cash Flow", "FreeCashFlow"])
                capex_key = find_field(cashflow, ["Capital Expenditure", "CapitalExpenditure", "Capital Expenditures"])
                if fcf_key:
                    fcf_series = cashflow.loc[fcf_key].dropna().head(5)
                    fcf_5y = [round(v / 1e9, 2) for v in reversed(fcf_series.tolist())]
                else:
                    logger.debug(f"{symbol}: FCF calc missing - no FCF field found")
                if capex_key:
                    capex_series = cashflow.loc[capex_key].dropna().head(5)
                    capex_5y = [round(abs(v) / 1e9, 2) for v in reversed(capex_series.tolist())]
        except Exception as e:
            logger.warning(f"{symbol}: FCF/Capex calculation error: {e}")

        # Pad arrays - only pad if there's at least 1 real value, otherwise leave empty
        def pad_array(arr, size=5):
            if not arr or all(v == 0 for v in arr):
                return arr  # Return as-is if empty or all zeros
            # Only pad front with zeros if we have some real data
            while len(arr) < size:
                arr.insert(0, 0)
            return arr[:size]

        # Helper to calculate YoY and 5Y growth
        def calc_growth(arr):
            if len(arr) < 2:
                return 0, 0
            yoy = round((arr[-1] - arr[-2]) / abs(arr[-2]) * 100, 1) if arr[-2] != 0 else 0
            five_y = round((arr[-1] - arr[0]) / abs(arr[0]) * 100, 1) if arr[0] != 0 else 0
            return yoy, five_y

        revenue_5y = pad_array(revenue_5y if revenue_5y else [])
        profit_5y = pad_array(profit_5y if profit_5y else [])
        opm_5y = pad_array(opm_5y if opm_5y else [])
        dividend_5y = pad_array(dividend_5y if dividend_5y else [])
        roe_5y = pad_array(roe_5y if roe_5y else [])
        roce_5y = pad_array(roce_5y if roce_5y else [])
        debt_5y = pad_array(debt_5y if debt_5y else [])
        fcf_5y = pad_array(fcf_5y if fcf_5y else [])
        capex_5y = pad_array(capex_5y if capex_5y else [])

        # Calculate growth metrics for chart labels
        revenue_yoy, revenue_5y_growth = calc_growth(revenue_5y)
        profit_yoy, profit_5y_growth = calc_growth(profit_5y)
        fcf_yoy, fcf_5y_growth = calc_growth(fcf_5y)

        # Build result
        result = {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "description": STOCK_DESCRIPTIONS.get(symbol, info.get("longBusinessSummary", "")[:500] if info.get("longBusinessSummary") else ""),
            "sector": sector,
            "logo": STOCK_LOGOS.get(symbol, symbol[:3].upper()),

            # Financials (5-year trends)
            "revenue_5y": revenue_5y,
            "profit_5y": profit_5y,
            "opm_5y": opm_5y,
            "dividend_5y": dividend_5y,
            "roe_5y": roe_5y,
            "roce_5y": roce_5y,
            "debt_5y": debt_5y,
            "fcf_5y": fcf_5y,
            "capex_5y": capex_5y,

            # Growth metrics for chart labels
            "revenue_yoy": revenue_yoy,
            "revenue_5y_growth": revenue_5y_growth,
            "profit_yoy": profit_yoy,
            "profit_5y_growth": profit_5y_growth,
            "fcf_yoy": fcf_yoy,
            "fcf_5y_growth": fcf_5y_growth,

            # Key Ratios
            "pe_ratio": round(pe_ratio, 1) if pe_ratio else 0,
            "industry_pe": industry_pe,
            "de_ratio": de_ratio,
            "roe": roe,
            "roce": roce,
            "current_ratio": round(current_ratio, 2) if current_ratio else 0,
            "quick_ratio": round(quick_ratio, 2) if quick_ratio else 0,
            "pb_ratio": round(pb_ratio, 2) if pb_ratio else 0,
            "ps_ratio": round(ps_ratio, 2) if ps_ratio else 0,
            "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else 0,

            # Dividend
            "div_yield": div_yield_pct,
            "payout_ratio": payout_ratio,
            "next_div_date": ex_div_date or "TBA",

            # Key stats
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": enterprise_value,
            "book_value": info.get("bookValue", 0),
            "high_52w": high_52w,
            "low_52w": low_52w,
            "avg_volume": avg_volume,
            "beta": round(beta, 2) if beta else 0,
            "eps": round(eps, 2) if eps else 0,

            # Company profile
            "industry": industry,
            "employees": employees,
            "website": website,

            # Additional insights
            "analyst_rating": info.get("recommendationKey", ""),
            "analyst_count": info.get("numberOfAnalystOpinions", 0),
            "target_mean_price": info.get("targetMeanPrice", 0),
            "target_high_price": info.get("targetHighPrice", 0),
            "target_low_price": info.get("targetLowPrice", 0),
            "earnings_growth": round((info.get("earningsGrowth", 0) or 0) * 100, 1),
            "revenue_growth": round((info.get("revenueGrowth", 0) or 0) * 100, 1),
            "profit_margins": round((info.get("profitMargins", 0) or 0) * 100, 1),
            "gross_margins": round((info.get("grossMargins", 0) or 0) * 100, 1),
            "operating_margins": round((info.get("operatingMargins", 0) or 0) * 100, 1),
            "free_cashflow": info.get("freeCashflow", 0) or 0,
            "operating_cashflow": info.get("operatingCashflow", 0) or 0,

            # Capex (calculated: Operating CF - Free CF)
            "capex": (info.get("operatingCashflow", 0) or 0) - (info.get("freeCashflow", 0) or 0),

            # Earnings & Calendar
            "next_earnings_date": "",  # Will be set below
            "last_earnings_date": "",

            # Balance Sheet highlights
            "total_debt": info.get("totalDebt", 0) or 0,
            "total_cash": info.get("totalCash", 0) or 0,
            "debt_to_cash": round((info.get("totalDebt", 0) or 0) / (info.get("totalCash", 1) or 1), 2),

            # Price position
            "price_vs_52w_high": round(((info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0) / high_52w * 100), 1) if high_52w else 0,
            "price_vs_52w_low": round(((info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0) / low_52w * 100), 1) if low_52w else 0,

            # All Time High (will be calculated below)
            "all_time_high": 0,
            "pct_from_ath": None,

            # Market rank (approximate based on market cap tiers)
            "market_rank": _get_market_rank(info.get("marketCap", 0)),

            # Forward metrics
            "forward_pe": round(info.get("forwardPE", 0) or 0, 1),
            "forward_eps": round(info.get("forwardEps", 0) or 0, 2),
            "peg_ratio": round(info.get("pegRatio", 0) or 0, 2),

            # Business summary (full)
            "business_summary": info.get("longBusinessSummary", "") or "",
        }

        # Calculate All-Time High (5-year max as proxy to avoid slow full history fetch)
        try:
            hist_5y = ticker.history(period="5y")
            if hist_5y is not None and not hist_5y.empty and "High" in hist_5y.columns:
                ath = hist_5y["High"].max()
                current_price = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0
                if ath and current_price:
                    result["all_time_high"] = round(ath, 2)
                    result["pct_from_ath"] = round((1 - current_price / ath) * 100, 1)
        except Exception as e:
            logger.warning(f"Could not calculate ATH for {symbol}: {e}")

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
            "name": STOCK_NAMES.get(symbol, symbol),
            "description": STOCK_DESCRIPTIONS.get(symbol, ""),
            "sector": STOCK_SECTORS.get(symbol, "Other"),
            "logo": STOCK_LOGOS.get(symbol, symbol[:3].upper()),
            "revenue_5y": [0, 0, 0, 0, 0],
            "profit_5y": [0, 0, 0, 0, 0],
            "opm_5y": [0, 0, 0, 0, 0],
            "dividend_5y": [0, 0, 0, 0, 0],
            "pe_ratio": 0,
            "industry_pe": INDUSTRY_PE.get(STOCK_SECTORS.get(symbol, "Other"), 20.0),
            "de_ratio": 0,
            "roe": 0,
            "roce": 0,
            "current_ratio": 0,
            "quick_ratio": 0,
            "pb_ratio": 0,
            "ps_ratio": 0,
            "ev_ebitda": 0,
            "div_yield": 0,
            "payout_ratio": 0,
            "next_div_date": "TBA",
            "market_cap": 0,
            "enterprise_value": 0,
            "book_value": 0,
            "high_52w": 0,
            "low_52w": 0,
            "avg_volume": 0,
            "beta": 0,
            "eps": 0,
            "industry": "",
            "employees": 0,
            "website": "",
            "analyst_rating": "",
            "analyst_count": 0,
            "target_mean_price": 0,
            "target_high_price": 0,
            "target_low_price": 0,
            "earnings_growth": 0,
            "revenue_growth": 0,
            "profit_margins": 0,
            "gross_margins": 0,
            "operating_margins": 0,
            "free_cashflow": 0,
            "operating_cashflow": 0,
            "capex": 0,
            "next_earnings_date": "",
            "last_earnings_date": "",
            "total_debt": 0,
            "total_cash": 0,
            "debt_to_cash": 0,
            "price_vs_52w_high": 0,
            "price_vs_52w_low": 0,
            "all_time_high": 0,
            "pct_from_ath": None,
            "market_rank": None,
            "forward_pe": 0,
            "forward_eps": 0,
            "peg_ratio": 0,
            "business_summary": "",
        }


def get_historical_prices(symbol: str, period: str = "1y", interval: str = None) -> List[float]:
    """
    Get historical closing prices for sparkline/chart.

    Args:
        symbol: NSE stock symbol
        period: Time period (e.g., "1y", "6mo", "3mo", "1d" for intraday)
        interval: Data interval (e.g., "5m", "15m" for intraday, None for daily)

    Returns:
        List of closing prices
    """
    try:
        yahoo_symbol = get_yahoo_symbol(symbol)
        ticker = yf.Ticker(yahoo_symbol)

        # For intraday data, use specified interval
        if interval:
            hist = ticker.history(period=period, interval=interval)
        else:
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


def get_trading_data(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get trading data for multiple symbols including:
    - Today's % change
    - Weekly CPR (Central Pivot Range) status
    - 20/50 EMA bullish/bearish status
    - Current LTP

    Args:
        symbols: List of NSE stock symbols

    Returns:
        Dictionary mapping symbol to trading data
    """
    result = {}

    for symbol in symbols:
        try:
            yahoo_symbol = get_yahoo_symbol(symbol)
            ticker = yf.Ticker(yahoo_symbol)

            # Get recent history for calculations (2 months for EMA)
            hist = ticker.history(period="3mo")

            if hist.empty or len(hist) < 2:
                result[symbol] = _default_trading_data(symbol)
                continue

            # Today's data
            current_price = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2]
            today_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0
            day_change = current_price - prev_close if prev_close else 0

            # Day high/low from today's candle
            day_high = hist["High"].iloc[-1] if "High" in hist else current_price
            day_low = hist["Low"].iloc[-1] if "Low" in hist else current_price
            day_open = hist["Open"].iloc[-1] if "Open" in hist else prev_close

            # Calculate Weekly CPR (Central Pivot Range)
            # Use last week's high, low, close for weekly pivot
            cpr_status = "N/A"
            try:
                # Get last 5-7 trading days for weekly data
                weekly_data = hist.tail(7)
                if len(weekly_data) >= 5:
                    weekly_high = weekly_data["High"].max()
                    weekly_low = weekly_data["Low"].min()
                    weekly_close = weekly_data["Close"].iloc[-1]

                    # Calculate CPR
                    pivot = (weekly_high + weekly_low + weekly_close) / 3
                    bc = (weekly_high + weekly_low) / 2  # Bottom Central
                    tc = pivot + (pivot - bc)  # Top Central

                    # CPR range
                    cpr_top = max(bc, tc)
                    cpr_bottom = min(bc, tc)

                    if current_price > cpr_top:
                        cpr_status = "Above CPR"
                    elif current_price < cpr_bottom:
                        cpr_status = "Below CPR"
                    else:
                        cpr_status = "In CPR"
            except Exception as e:
                logger.warning(f"CPR calculation failed for {symbol}: {e}")

            # Calculate 20/50 EMA status
            ema_status = "N/A"
            try:
                if len(hist) >= 50:
                    ema_20 = hist["Close"].ewm(span=20, adjust=False).mean()
                    ema_50 = hist["Close"].ewm(span=50, adjust=False).mean()

                    current_ema_20 = ema_20.iloc[-1]
                    current_ema_50 = ema_50.iloc[-1]

                    # Check if price is above both EMAs and EMA20 > EMA50
                    if current_price > current_ema_20 and current_price > current_ema_50:
                        if current_ema_20 > current_ema_50:
                            ema_status = "Bullish"
                        else:
                            ema_status = "Neutral"
                    elif current_price < current_ema_20 and current_price < current_ema_50:
                        if current_ema_20 < current_ema_50:
                            ema_status = "Bearish"
                        else:
                            ema_status = "Neutral"
                    else:
                        ema_status = "Neutral"
                elif len(hist) >= 20:
                    # Only have EMA20
                    ema_20 = hist["Close"].ewm(span=20, adjust=False).mean()
                    current_ema_20 = ema_20.iloc[-1]
                    if current_price > current_ema_20:
                        ema_status = "Bullish"
                    else:
                        ema_status = "Bearish"
            except Exception as e:
                logger.warning(f"EMA calculation failed for {symbol}: {e}")

            result[symbol] = {
                "ltp": round(current_price, 2),
                "today_pct": round(today_pct, 2),
                "day_change": round(day_change, 2),
                "day_open": round(day_open, 2),
                "day_high": round(day_high, 2),
                "day_low": round(day_low, 2),
                "prev_close": round(prev_close, 2),
                "cpr_status": cpr_status,
                "ema_status": ema_status,
            }

        except Exception as e:
            logger.error(f"Error getting trading data for {symbol}: {e}")
            result[symbol] = _default_trading_data(symbol)

    return result


def _default_trading_data(symbol: str) -> Dict[str, Any]:
    """Return default trading data when calculation fails"""
    return {
        "ltp": 0,
        "today_pct": 0,
        "cpr_status": "N/A",
        "ema_status": "N/A",
    }
