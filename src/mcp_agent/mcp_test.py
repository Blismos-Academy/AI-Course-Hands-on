import asyncio
from mcp_client import MCPClient


async def main():

    client = MCPClient()

    tools = await client.list_tools()

    print("MCP Connected")
    print("Tools:")

    for tool in tools.tools:
        print("-", tool.name)

    result = await client.call_tool(
        "create_calendar_event",
        {
            "title": "MCP Calendar Test",
            "start_time": "2026-09-15T16:30:00+05:30",
            "duration_minutes": 30
        }
    )

    print("\nResult:")
    print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())