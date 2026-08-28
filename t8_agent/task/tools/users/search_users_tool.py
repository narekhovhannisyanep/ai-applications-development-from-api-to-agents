import inspect
from typing import Any

from t8_agent.task.tools.users.base import BaseUserServiceTool


class SearchUsersTool(BaseUserServiceTool):
    @property
    def name(self) -> str:
        return "search_users"

    @property
    def description(self) -> str:
        # TODO: Provide description of this tool
        return inspect.cleandoc("""
            Search user by various criteria.
            Name, surname and email support partial matching(case-sensitive).
            Gender must be an exact match.
        """)

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "surnae": {"type": "string", "description": "User surname"},
                "email": {"type": "string", "description": "User email"},
                "gender": {"enum": ["male", "female", "other"]},
            },
            "required": [],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            return self._user_client.search_users(**arguments)
        except Exception as e:
            return f"Error while searching users: {e}"
