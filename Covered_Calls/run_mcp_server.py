#!/usr/bin/env python
"""
MCP Server Entry Point for Portfolio Data
Run this to expose portfolio data to Claude Code.

Usage:
    python run_mcp_server.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Run the MCP server
from services.mcp_portfolio_server import run_server

if __name__ == "__main__":
    run_server()
