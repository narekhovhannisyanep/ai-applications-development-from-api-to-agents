from typing import Any

from t10_mcp_advanced.mcp_server.tools.users.base import BaseUserServiceTool


class DeleteUserTool(BaseUserServiceTool):
    @property
    def name(self) -> str:
        return "delete_user"

    @property
    def description(self) -> str:
        return "Delete a user from the system"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "number", "description": "id of user to delete"}
            },
            "required": ["id"],
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        user_id: int = int(arguments["id"])
        return self._user_client.delete_user(user_id)
