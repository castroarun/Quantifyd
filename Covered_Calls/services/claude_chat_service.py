"""
Claude Chat Service for Portfolio Research
Integrates Claude API with portfolio tools for stock research chat functionality.
"""

import os
import json
from typing import Any, Generator
from anthropic import Anthropic

# Import portfolio functions
from .holdings_service import (
    get_holdings,
    get_fundamentals,
    get_historical_prices,
    get_portfolio_summary,
    get_trading_data,
    STOCK_SECTORS,
    STOCK_DESCRIPTIONS,
    STOCK_NAMES,
    INDUSTRY_PE,
)


# Initialize Anthropic client
def get_claude_client() -> Anthropic:
    """Get or create Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return Anthropic(api_key=api_key)


# Tool definitions for Claude API
PORTFOLIO_TOOLS = [
    {
        "name": "get_holdings",
        "description": "Get current portfolio holdings with P/L, invested amount, current value, and portfolio percentage. Use this to understand what stocks are in the portfolio.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_by": {
                    "type": "string",
                    "description": "Sort holdings by: pnl_pct (P/L %), pnl (P/L amount), invested, current, alpha (alphabetical)",
                    "enum": ["pnl_pct", "pnl", "invested", "current", "alpha"],
                    "default": "pnl_pct"
                },
                "sector": {
                    "type": "string",
                    "description": "Filter by sector (optional). Available sectors: Technology, Financial, Energy, Pharma, Auto, Consumer, Industrial, Telecom"
                }
            }
        },
    },
    {
        "name": "get_stock_fundamentals",
        "description": "Get fundamental data for a stock including P/E ratio, ROE, ROCE, D/E ratio, revenue trends, net profit trends, operating margin, and dividend info. Essential for stock analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol (e.g., RELIANCE, INFY, TCS, HDFCBANK)"
                }
            },
            "required": ["symbol"]
        },
    },
    {
        "name": "get_stock_price_history",
        "description": "Get historical price data for a stock to analyze price trends and performance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol"
                },
                "period": {
                    "type": "string",
                    "description": "Time period for historical data",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                    "default": "1y"
                }
            },
            "required": ["symbol"]
        },
    },
    {
        "name": "get_trading_signals",
        "description": "Get technical trading signals including CPR (Central Pivot Range) position, EMA (20/50) status, and today's price change percentage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock symbols to get trading signals for"
                }
            },
            "required": ["symbols"]
        },
    },
    {
        "name": "compare_stocks",
        "description": "Compare two stocks side by side on key metrics like P/E, ROE, ROCE, revenue growth, profit margins, and debt levels. Useful for deciding between stocks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol1": {
                    "type": "string",
                    "description": "First stock symbol to compare"
                },
                "symbol2": {
                    "type": "string",
                    "description": "Second stock symbol to compare"
                }
            },
            "required": ["symbol1", "symbol2"]
        },
    },
    {
        "name": "get_portfolio_summary",
        "description": "Get overall portfolio summary including total invested amount, current value, total P/L, P/L percentage, and sector-wise breakdown.",
        "input_schema": {
            "type": "object",
            "properties": {}
        },
    },
    {
        "name": "get_sector_analysis",
        "description": "Analyze a specific sector in the portfolio. Shows all holdings in that sector with their performance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "Sector name: Technology, Financial, Energy, Pharma, Auto, Consumer, Industrial, Telecom, Other"
                }
            },
            "required": ["sector"]
        },
    },
    {
        "name": "get_top_performers",
        "description": "Get top or bottom performing stocks in the portfolio by P/L percentage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of stocks to return",
                    "default": 5
                },
                "bottom": {
                    "type": "boolean",
                    "description": "If true, return worst performers instead of best",
                    "default": False
                }
            }
        },
    },
]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a portfolio tool and return the result as JSON string."""

    if name == "get_holdings":
        holdings = get_holdings()
        sort_by = arguments.get("sort_by", "pnl_pct")
        sector_filter = arguments.get("sector")

        # Filter by sector if specified
        if sector_filter:
            holdings = [h for h in holdings if STOCK_SECTORS.get(h["symbol"], "Other") == sector_filter]

        # Sort holdings
        if sort_by == "alpha":
            holdings.sort(key=lambda x: (x.get("name") or x["symbol"]).lower())
        else:
            holdings.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

        return json.dumps(holdings, indent=2, default=str)

    elif name == "get_stock_fundamentals":
        symbol = arguments["symbol"].upper()
        fundamentals = get_fundamentals(symbol)

        # Add sector and description
        fundamentals["sector"] = STOCK_SECTORS.get(symbol, "Other")
        fundamentals["description"] = STOCK_DESCRIPTIONS.get(symbol, "")
        fundamentals["company_name"] = STOCK_NAMES.get(symbol, symbol)
        fundamentals["industry_pe"] = INDUSTRY_PE.get(STOCK_SECTORS.get(symbol), None)

        return json.dumps(fundamentals, indent=2, default=str)

    elif name == "get_stock_price_history":
        symbol = arguments["symbol"].upper()
        period = arguments.get("period", "1y")
        prices = get_historical_prices(symbol, period=period)
        return json.dumps(prices, indent=2, default=str)

    elif name == "get_trading_signals":
        symbols = [s.upper() for s in arguments["symbols"]]
        trading_data = get_trading_data(symbols)
        return json.dumps(trading_data, indent=2, default=str)

    elif name == "compare_stocks":
        symbol1 = arguments["symbol1"].upper()
        symbol2 = arguments["symbol2"].upper()

        fund1 = get_fundamentals(symbol1)
        fund2 = get_fundamentals(symbol2)

        comparison = {
            "stock1": {
                "symbol": symbol1,
                "name": STOCK_NAMES.get(symbol1, symbol1),
                "sector": STOCK_SECTORS.get(symbol1, "Other"),
                **fund1
            },
            "stock2": {
                "symbol": symbol2,
                "name": STOCK_NAMES.get(symbol2, symbol2),
                "sector": STOCK_SECTORS.get(symbol2, "Other"),
                **fund2
            }
        }

        return json.dumps(comparison, indent=2, default=str)

    elif name == "get_portfolio_summary":
        summary = get_portfolio_summary()
        return json.dumps(summary, indent=2, default=str)

    elif name == "get_sector_analysis":
        sector = arguments["sector"]
        holdings = get_holdings()

        sector_holdings = [h for h in holdings if STOCK_SECTORS.get(h["symbol"], "Other") == sector]

        if not sector_holdings:
            return json.dumps({"error": f"No holdings in sector: {sector}"})

        total_invested = sum(h.get("invested", 0) for h in sector_holdings)
        total_current = sum(h.get("current", 0) for h in sector_holdings)
        total_pnl = sum(h.get("pnl", 0) for h in sector_holdings)

        analysis = {
            "sector": sector,
            "stock_count": len(sector_holdings),
            "total_invested": total_invested,
            "total_current": total_current,
            "total_pnl": total_pnl,
            "pnl_pct": (total_pnl / total_invested * 100) if total_invested > 0 else 0,
            "holdings": sector_holdings
        }

        return json.dumps(analysis, indent=2, default=str)

    elif name == "get_top_performers":
        count = arguments.get("count", 5)
        bottom = arguments.get("bottom", False)

        holdings = get_holdings()
        holdings.sort(key=lambda x: x.get("pnl_pct", 0), reverse=not bottom)

        top = holdings[:count]
        result = {
            "type": "bottom_performers" if bottom else "top_performers",
            "count": count,
            "stocks": top
        }

        return json.dumps(result, indent=2, default=str)

    raise ValueError(f"Unknown tool: {name}")


# Conversation history storage (in-memory, per session)
conversation_histories: dict[str, list] = {}


def get_system_prompt() -> str:
    """Get the system prompt for portfolio research assistant."""
    return """You are a helpful portfolio research assistant integrated into a stock holdings dashboard. You have access to real-time portfolio data, fundamentals, and trading signals.

Your capabilities:
1. **Portfolio Analysis**: View holdings, P/L, sector allocation, top/bottom performers
2. **Stock Fundamentals**: P/E ratios, ROE, ROCE, debt levels, revenue/profit trends, dividends
3. **Trading Signals**: CPR levels, EMA positions, intraday price changes
4. **Stock Comparison**: Compare any two stocks on key metrics

Guidelines:
- Be concise but informative. This is a trading dashboard, users want quick insights.
- When presenting financial data, format numbers nicely (use lakhs/crores for Indian context)
- Highlight key insights and actionable information
- If asked about stocks not in the portfolio, you can still fetch their fundamentals
- For trading recommendations, always note that these are informational only, not financial advice

Available sectors: Technology, Financial, Energy, Pharma, Auto, Consumer, Industrial, Telecom, Other

Remember: Users are viewing their actual portfolio data. Be helpful and accurate."""


def chat(session_id: str, user_message: str) -> Generator[str, None, None]:
    """
    Process a chat message and yield response chunks (streaming).

    Args:
        session_id: Unique session identifier for conversation history
        user_message: The user's message

    Yields:
        Response text chunks
    """
    client = get_claude_client()

    # Get or create conversation history
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    history = conversation_histories[session_id]

    # Add user message to history
    history.append({"role": "user", "content": user_message})

    # Keep only last 20 messages to manage context
    if len(history) > 20:
        history = history[-20:]
        conversation_histories[session_id] = history

    # Initial API call
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=get_system_prompt(),
        tools=PORTFOLIO_TOOLS,
        messages=history,
    )

    # Process response, handling tool calls
    while response.stop_reason == "tool_use":
        # Extract tool use blocks
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        # Build tool results
        tool_results = []
        for tool_use in tool_uses:
            try:
                result = execute_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })
            except Exception as e:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps({"error": str(e)}),
                    "is_error": True
                })

        # Add assistant response and tool results to history
        history.append({"role": "assistant", "content": response.content})
        history.append({"role": "user", "content": tool_results})

        # Continue conversation with tool results
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=get_system_prompt(),
            tools=PORTFOLIO_TOOLS,
            messages=history,
        )

    # Extract final text response
    final_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            final_text += block.text

    # Add assistant response to history
    history.append({"role": "assistant", "content": final_text})

    yield final_text


def chat_sync(session_id: str, user_message: str) -> str:
    """
    Synchronous version of chat (non-streaming).

    Args:
        session_id: Unique session identifier
        user_message: The user's message

    Returns:
        Complete response text
    """
    result = ""
    for chunk in chat(session_id, user_message):
        result += chunk
    return result


def clear_history(session_id: str) -> None:
    """Clear conversation history for a session."""
    if session_id in conversation_histories:
        del conversation_histories[session_id]


def get_suggested_questions() -> list[str]:
    """Get suggested questions for the chat interface."""
    return [
        "Show me my portfolio summary",
        "What are my top 5 performers?",
        "Analyze my Technology sector holdings",
        "Compare RELIANCE and TCS",
        "Which stocks have high debt levels?",
        "Show me stocks with good dividend yield",
        "What's the P/E ratio of INFY?",
        "Which stocks are trading above their 50 EMA?",
    ]
