from typing import Any

from t10_mcp_advanced.mcp_server.tools.users.base import BaseUserServiceTool


class SearchUsersTool(BaseUserServiceTool):
    @property
    def name(self) -> str:
        return "search_users"

    @property
    def description(self) -> str:
        return "Provides information about users mathing by name, surname, email or gender."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "surname": {"type": "string", "description": "User surname"},
                "email": {"type": "string", "description": "User email"},
                "gender": {"type": "string", "description": "User gender"},
            },
            "required": [],
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        return self._user_client.search_users(**arguments)
