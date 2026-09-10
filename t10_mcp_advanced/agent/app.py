import asyncio
import json
import os

from commons.constants import GPT_5_6_LUNA, OPENAI_API_KEY
from commons.models.message import Message
from commons.models.role import Role
from t10_mcp_advanced.agent.agent import CustomAgentMCP
from t10_mcp_advanced.agent.clients.custom_mcp_client import CustomMCPClient
from t10_mcp_advanced.agent.clients.mcp_client import MCPClient


async def _collect_tools(
    client: MCPClient | CustomMCPClient,
    tools: list[dict],
    tool_name_client_map: dict[str, MCPClient | CustomMCPClient],
):
    for tool in await client.get_tools():
        tools.append(tool)
        tool_name_client_map[tool.get("function", {}).get("name")] = client
        print(f"{json.dumps(tool, indent=2)}")


async def main():
    tools: list[dict] = []
    tool_name_client_map: dict[str, MCPClient | CustomMCPClient] = {}

    ums_mcp_client = await MCPClient.create("http://localhost:8006/mcp")
    await _collect_tools(ums_mcp_client, tools, tool_name_client_map)

    # fetch_mcp_client = await CustomMCPClient.create("http://localhost:8006/mcp")
    # await _collect_tools(fetch_mcp_client, tools, tool_name_client_map)

    dial_client = CustomAgentMCP(
        api_key=OPENAI_API_KEY,
        model=GPT_5_6_LUNA,
        tools=tools,
        tool_name_client_map=tool_name_client_map,
    )

    messages: list[Message] = [
        Message(
            role=Role.SYSTEM,
            content="You are an advanced AI agent. Your goal is to assist users with their questions.",
        )
    ]

    print("MCP-based Agent is ready! Type your question or 'exti' to exit.")
    while True:
        user_input = input("\n> ").strip()
        if user_input in ("exit", "quit"):
            break

        messages.append(Message(role=Role.USER, content=user_input))
        ai_message: Message = await dial_client.get_completion(messages)
        messages.append(ai_message)


if __name__ == "__main__":
    asyncio.run(main())
