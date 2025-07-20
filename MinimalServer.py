#!/usr/bin/env python3
"""
Minimal MCP Server Test
"""
import asyncio
import sys
from mcp.server import Server, InitializationOptions, NotificationOptions
from mcp.types import Tool, TextContent, CallToolRequest
from mcp.server.stdio import stdio_server

# Create server
server = Server("test-server")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List test tools"""
    try:
        print("list_tools called", file=sys.stderr)
        return [
            Tool(
                name="test_tool",
                description="A simple test tool",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Test message"
                        }
                    }
                }
            )
        ]
    except Exception as e:
        print(f"Error in list_tools: {e}", file=sys.stderr)
        return []

@server.call_tool()
async def handle_call_tool(tool_call: CallToolRequest) -> list[TextContent]:
    """Handle test tool calls"""
    try:
        name = tool_call.params.name
        arguments = tool_call.params.arguments
        print(f"call_tool: {name}, args: {arguments}", file=sys.stderr)
        return [TextContent(
            type="text",
            text=f"Test tool {name} called with: {arguments}"
        )]
    except Exception as e:
        print(f"Error in call_tool: {e}", file=sys.stderr)
        return [TextContent(
            type="text",
            text=f"Error: {e}"
        )]

async def main():
    """Main entry point"""
    print("Starting minimal test server...", file=sys.stderr)
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            print("Server initialized, running...", file=sys.stderr)
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(
                    notification_options=NotificationOptions(tools_changed=True)
                ),
            )
    except Exception as e:
        print(f"Server run error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped", file=sys.stderr)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
