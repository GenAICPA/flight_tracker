#!/usr/bin/env python3
"""
Command-line entry point for Flight Tracker MCP Server
"""

from . import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
