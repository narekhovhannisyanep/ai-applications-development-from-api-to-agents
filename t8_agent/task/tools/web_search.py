from typing import Any

import requests

from commons.constants import GPT_5_6_LUNA, OPENAI_RESPONSES_ENDPOINT
from t8_agent.task.tools.base import BaseTool


class WebSearchTool(BaseTool):
    def __init__(self, open_ai_api_key: str):
        self.__api_key = f"Bearer {open_ai_api_key}"
        self.__endpoint = OPENAI_RESPONSES_ENDPOINT

    @property
    def name(self) -> str:
        return "web_search_tool"

    @property
    def description(self) -> str:
        return "Tool for WEB searching."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query to search for on the web.",
                }
            },
            "required": ["request"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            response = requests.post(
                url=self.__endpoint,
                headers={
                    "Authorization": self.__api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": GPT_5_6_LUNA,
                    "tools": [
                        {
                            "type": "web_search",
                            "search_context_size": "low",
                            "external_web_access": False,
                        }
                    ],
                    "input": str(arguments["request"]),
                },
            )
            response.raise_for_status()

            data = response.json()

            for item in data.get("output", []):
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") == "output_text":
                            return block["text"]

            return "No result returned from web search."
        except Exception as e:
            return f"Error {response.status_code} {response.text}"
