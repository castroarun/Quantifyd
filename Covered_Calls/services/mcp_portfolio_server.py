"""
MCP Server for Portfolio Data
Exposes holdings, fundamentals, and trading data as MCP tools for Claude integration.
"""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import AnyUrl

# Import holdings service functions
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

# Create MCP server instance
server = Server("portfolio-mcp")


# ============================================================================
# RESOURCES - Static data that can be read
# ============================================================================

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available portfolio resources."""
    return [
        Resource(
            uri=AnyUrl("portfolio://holdings"),
            name="Current Holdings",
            description="All current stock holdings with P/L, quantities, and values",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("portfolio://summary"),
            name="Portfolio Summary",
            description="Portfolio summary including total value, P/L, and sector allocation",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("portfolio://sectors"),
            name="Sector Allocation",
            description="Stock sector classifications and groupings",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Read a portfolio resource."""
    uri_str = str(uri)

    if uri_str == "portfolio://holdings":
        holdings = get_holdings()
        return json.dumps(holdings, indent=2, default=str)

    elif uri_str == "portfolio://summary":
        summary = get_portfolio_summary()
        return json.dumps(summary, indent=2, default=str)

    elif uri_str == "portfolio://sectors":
        return json.dumps(STOCK_SECTORS, indent=2)

    raise ValueError(f"Unknown resource: {uri}")


# ============================================================================
# TOOLS - Actions that can be performed
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available portfolio tools."""
    return [
        Tool(
            name="get_holdings",
            description="Get current portfolio holdings with P/L, invested amount, current value, and portfolio percentage",
            inputSchema={
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "description": "Sort holdings by: pnl_pct, pnl, invested, current, alpha",
                        "enum": ["pnl_pct", "pnl", "invested", "current", "alpha"],
                        "default": "pnl_pct"
                    },
                    "sector": {
                        "type": "string",
                        "description": "Filter by sector (optional)"
                    }
                }
            },
        ),
        Tool(
            name="get_stock_fundamentals",
            description="Get fundamental data for a stock including P/E, ROE, ROCE, D/E ratio, revenue, net profit trends, and dividend info",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., RELIANCE, INFY, TCS)"
                    }
                },
                "required": ["symbol"]
            },
        ),
        Tool(
            name="get_stock_price_history",
            description="Get historical price data for a stock",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period: 1mo, 3mo, 6mo, 1y, 2y, 5y",
                        "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                        "default": "1y"
                    }
                },
                "required": ["symbol"]
            },
        ),
        Tool(
            name="get_trading_signals",
            description="Get trading signals including CPR (Central Pivot Range), EMA status, and today's price change",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock symbols"
                    }
                },
                "required": ["symbols"]
            },
        ),
        Tool(
            name="compare_stocks",
            description="Compare two stocks on key metrics like P/E, ROE, ROCE, revenue growth, and profit margins",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol1": {
                        "type": "string",
                        "description": "First stock symbol"
                    },
                    "symbol2": {
                        "type": "string",
                        "description": "Second stock symbol"
                    }
                },
                "required": ["symbol1", "symbol2"]
            },
        ),
        Tool(
            name="get_portfolio_summary",
            description="Get overall portfolio summary including total invested, current value, P/L, and sector breakdown",
            inputSchema={
                "type": "object",
                "properties": {}
            },
        ),
        Tool(
            name="get_sector_analysis",
            description="Analyze a specific sector in the portfolio",
            inputSchema={
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Sector name (e.g., Technology, Financial, Energy)"
                    }
                },
                "required": ["sector"]
            },
        ),
        Tool(
            name="get_top_performers",
            description="Get top performing stocks in portfolio by P/L percentage",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of top performers to return",
                        "default": 5
                    },
                    "bottom": {
                        "type": "boolean",
                        "description": "If true, return bottom performers instead",
                        "default": False
                    }
                }
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a portfolio tool."""

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

        return [TextContent(type="text", text=json.dumps(holdings, indent=2, default=str))]

    elif name == "get_stock_fundamentals":
        symbol = arguments["symbol"]
        fundamentals = get_fundamentals(symbol)

        # Add sector and description
        fundamentals["sector"] = STOCK_SECTORS.get(symbol, "Other")
        fundamentals["description"] = STOCK_DESCRIPTIONS.get(symbol, "")
        fundamentals["company_name"] = STOCK_NAMES.get(symbol, symbol)
        fundamentals["industry_pe"] = INDUSTRY_PE.get(STOCK_SECTORS.get(symbol), None)

        return [TextContent(type="text", text=json.dumps(fundamentals, indent=2, default=str))]

    elif name == "get_stock_price_history":
        symbol = arguments["symbol"]
        period = arguments.get("period", "1y")
        prices = get_historical_prices(symbol, period=period)
        return [TextContent(type="text", text=json.dumps(prices, indent=2, default=str))]

    elif name == "get_trading_signals":
        symbols = arguments["symbols"]
        trading_data = get_trading_data(symbols)
        return [TextContent(type="text", text=json.dumps(trading_data, indent=2, default=str))]

    elif name == "compare_stocks":
        symbol1 = arguments["symbol1"]
        symbol2 = arguments["symbol2"]

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

        return [TextContent(type="text", text=json.dumps(comparison, indent=2, default=str))]

    elif name == "get_portfolio_summary":
        summary = get_portfolio_summary()
        return [TextContent(type="text", text=json.dumps(summary, indent=2, default=str))]

    elif name == "get_sector_analysis":
        sector = arguments["sector"]
        holdings = get_holdings()

        sector_holdings = [h for h in holdings if STOCK_SECTORS.get(h["symbol"], "Other") == sector]

        if not sector_holdings:
            return [TextContent(type="text", text=json.dumps({"error": f"No holdings in sector: {sector}"}))]

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

        return [TextContent(type="text", text=json.dumps(analysis, indent=2, default=str))]

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

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    raise ValueError(f"Unknown tool: {name}")


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run_server():
    """Entry point for running the MCP server."""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()
