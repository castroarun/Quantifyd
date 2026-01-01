# Using Claude Code for Portfolio Research

This project includes an MCP server that exposes your portfolio data to Claude Code. Since you have Claude Max, this method incurs **no additional charges**.

## Setup (One-Time)

1. **Ensure Flask app is running** (for Zerodha API access):
   ```bash
   python app.py
   ```

2. **Open Claude Code in this project**:
   ```bash
   cd c:\Users\Castro\Documents\Projects\Covered_Calls
   claude
   ```

3. The MCP server is already registered in `.claude/settings.local.json`

## Available Portfolio Tools

When you ask Claude Code about your portfolio, it can use these tools:

| Tool | Description | Example Query |
|------|-------------|---------------|
| `get_holdings` | View all holdings with P/L | "Show my holdings sorted by P/L%" |
| `get_stock_fundamentals` | P/E, ROE, ROCE, revenue trends | "What's the P/E of RELIANCE?" |
| `get_stock_price_history` | Historical prices (1mo-5y) | "Show INFY price trend" |
| `get_trading_signals` | CPR, EMA, today's % | "Which stocks are bullish?" |
| `compare_stocks` | Side-by-side comparison | "Compare TCS vs INFY" |
| `get_portfolio_summary` | Total value, P/L, sectors | "Portfolio summary" |
| `get_sector_analysis` | Analyze specific sector | "Analyze my Technology holdings" |
| `get_top_performers` | Best/worst stocks | "Top 5 performers" |

## Example Questions to Ask Claude Code

### Portfolio Overview
- "Show me my portfolio summary"
- "What's my total P/L?"
- "Which sector has the most allocation?"

### Stock Analysis
- "Analyze RELIANCE fundamentals"
- "What's the ROE and ROCE of HDFCBANK?"
- "Is INFY undervalued based on P/E?"
- "Compare TCS and WIPRO on key metrics"

### Trading Insights
- "Which of my stocks are above their weekly CPR?"
- "Show me stocks with bullish EMA setup"
- "Which stocks fell the most today?"

### Sector Analysis
- "How is my Financial sector performing?"
- "Which stocks in Technology have the highest ROE?"

### Screening
- "Which of my stocks have dividend yield > 2%?"
- "Show stocks with D/E ratio < 0.5"
- "Which stocks are near their 52-week high?"

## How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Claude Code │ ──► │ MCP Server  │ ──► │ Flask App   │
│ (Your CLI)  │     │ (portfolio) │     │ (Zerodha)   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │    Claude Max     │   Holdings Data   │   Live Prices
       │    (No extra $)   │   Fundamentals    │   from Zerodha
       └───────────────────┴───────────────────┴──────────────
```

## Troubleshooting

### MCP Server Not Found
If Claude Code doesn't recognize portfolio commands:
1. Restart Claude Code
2. Check `.claude/settings.local.json` has `mcpServers.portfolio` config

### Data Not Loading
If portfolio data isn't available:
1. Ensure Flask app is running: `python app.py`
2. Ensure you're logged into Zerodha (visit http://localhost:5000/login)

### Cache Issues
Fundamental data is cached for 24 hours. To force refresh:
- Delete cache files in `cache/fundamentals/`
- Or use `/api/fundamentals/SYMBOL?refresh=true` in browser
