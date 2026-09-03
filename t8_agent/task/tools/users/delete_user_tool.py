from typing import Any

from t8_agent.task.tools.users.base import BaseUserServiceTool


class DeleteUserTool(BaseUserServiceTool):
    @property
    def name(self) -> str:
        return "delete_users"

    @property
    def description(self) -> str:
        return "Delete user by ID."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "number", "description": "ID of user to delete."}
            },
            "required": ["id"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            id = arguments["id"]
            return self._user_client.delete_user(int(id))
        except Exception as e:
            return f"Error while deleting {id} user: {e}"
