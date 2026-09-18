import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(SERVER_DIR, "mcp_server.py")


class MCPClient:

    def __init__(self):
        self.server = StdioServerParameters(
            command=sys.executable,
            args=[SERVER_PATH],
            cwd=SERVER_DIR,
        )

    async def call_tool(self, name, arguments):

        async with stdio_client(self.server) as (read, write):

            async with ClientSession(read, write) as session:

                await session.initialize()

                return await session.call_tool(
                    name,
                    arguments
                )

    async def list_tools(self):

        async with stdio_client(self.server) as (read, write):

            async with ClientSession(read, write) as session:

                await session.initialize()

                return await session.list_tools()