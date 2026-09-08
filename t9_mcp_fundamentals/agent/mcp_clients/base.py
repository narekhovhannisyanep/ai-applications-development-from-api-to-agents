from abc import ABC, abstractmethod
from typing import Any

from mcp import ClientSession
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    GetPromptResult,
    ListToolsResult,
    Prompt,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl


class MCPClient(ABC):
    def __init__(self) -> None:
        self.session: ClientSession | None = None

    @abstractmethod
    async def __aenter__(self): ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools from MCP server"""
        if not self.session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        tools_result: ListToolsResult = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a specific tool on the MCP server"""
        if not self.session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        tool_result: CallToolResult = await self.session.call_tool(tool_name, tool_args)

        if not tool_result.content:
            return "No content returned from tool"

        content = tool_result.content[0]
        print(f"    ⚙️: {content}\n")

        if isinstance(content, TextContent):
            return content.text

        return content

    async def get_resources(self) -> list[Resource]:
        """Get available resources from MCP server"""
        if not self.session:
            raise RuntimeError("MCP client not connected.")

        try:
            resources_result = await self.session.list_resources()
            return resources_result.resources
        except Exception as e:
            print(e)
            return []

    async def get_resource(self, uri: AnyUrl) -> str:
        """Get specific resource content"""
        if not self.session:
            raise RuntimeError("MCP client not connected.")

        resource_result: ReadResourceResult = await self.session.read_resource(uri=uri)

        if not resource_result.contents:
            return ""

        content = resource_result.contents[0]

        if isinstance(content, TextResourceContents):
            return content.text
        elif isinstance(content, BlobResourceContents):
            return content.blob
        else:
            raise TypeError(f"Unsupported resource type for {uri}")

    async def get_prompts(self) -> list[Prompt]:
        """Get available prompts from MCP server"""
        if not self.session:
            raise RuntimeError("MCP client not connected.")

        try:
            prompt_result = await self.session.list_prompts()
            return prompt_result.prompts
        except Exception as e:
            print(e)
            return []

    async def get_prompt(self, name: str) -> str:
        """Get specific prompt content"""
        if not self.session:
            raise RuntimeError("MCP client not connected.")

        prompt_result: GetPromptResult = await self.session.get_prompt(name)
        combined_content = ""

        for msg in prompt_result.messages:
            if hasattr(msg, "content") and isinstance(msg.content, TextContent):
                combined_content += msg.content.text + "\n"

            if hasattr(msg, "content") and isinstance(msg.content, str):
                combined_content += msg.content + "\n"

        return combined_content.strip()
