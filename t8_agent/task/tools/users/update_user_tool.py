from typing import Any

from pydantic import ValidationError

from commons.user_service.user_info import UserUpdate
from t8_agent.task.tools.users.base import BaseUserServiceTool


class UpdateUserTool(BaseUserServiceTool):
    @property
    def name(self) -> str:
        return "update_user"

    @property
    def description(self) -> str:
        return "Update user by ID"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {
                    "type": "number",
                    "description": "User ID that should be updated.",
                },
                "new_info": UserUpdate.model_json_schema(),
            },
            "required": ["id"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            id = arguments["id"]
            new_info = UserUpdate.model_validate(arguments["new_info"])
            return self._user_client.update_user(id, new_info)
        except ValidationError as e:
            return f"Validation error: {e}"
        except Exception as e:
            return f"Error while update the {id} user: {e}"
