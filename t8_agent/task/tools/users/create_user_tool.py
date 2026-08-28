from typing import Any

from pydantic import ValidationError

from commons.user_service.user_info import UserCreate
from t8_agent.task.tools.users.base import BaseUserServiceTool


class CreateUserTool(BaseUserServiceTool):
    @property
    def name(self) -> str:
        return "add_user"

    @property
    def description(self) -> str:
        return "Create a new user"

    @property
    def input_schema(self) -> dict[str, Any]:
        return UserCreate.model_json_schema()

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            new_user = UserCreate.model_validate(arguments)
            return self._user_client.add_user(new_user)
        except ValidationError as e:
            return f"Validation error: {e}"
        except Exception as e:
            return f"Error while creating a new user: {e}"
