from commons.constants import (
    ANTHROPIC_API_KEY,
    CLAUDE_HAIKU_4_5,
    GPT_5_6_LUNA,
    OPENAI_API_KEY,
)
from commons.models.conversation import Conversation
from commons.models.message import Message
from commons.models.role import Role
from commons.user_service.client import UserServiceClient
from t8_agent.task.agents.anthropic import AnthropicBasedAgent
from t8_agent.task.agents.openai import OpenAIBasedAgent
from t8_agent.task.prompts import SYSTEM_PROMPT
from t8_agent.task.tools.base import BaseTool
from t8_agent.task.tools.users.create_user_tool import CreateUserTool
from t8_agent.task.tools.users.delete_user_tool import DeleteUserTool
from t8_agent.task.tools.users.get_user_by_id_tool import GetUserByIdTool
from t8_agent.task.tools.users.search_users_tool import SearchUsersTool
from t8_agent.task.tools.users.update_user_tool import UpdateUserTool
from t8_agent.task.tools.web_search import WebSearchTool


def main():
    user_client = UserServiceClient()
    tools: list[BaseTool] = [
        WebSearchTool(open_ai_api_key=OPENAI_API_KEY),
        GetUserByIdTool(user_client=user_client),
        SearchUsersTool(user_client=user_client),
        CreateUserTool(user_client=user_client),
        UpdateUserTool(user_client=user_client),
        DeleteUserTool(user_client=user_client),
    ]

    agent = OpenAIBasedAgent(
        model=GPT_5_6_LUNA,
        api_key=OPENAI_API_KEY,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    # agent = AnthropicBasedAgent(
    #     model=CLAUDE_HAIKU_4_5,
    #     api_key=ANTHROPIC_API_KEY,
    #     tools=tools,
    #     system_prompt=SYSTEM_PROMPT,
    # )

    conversation = Conversation()

    print("Type your question or 'exit' to quit.")
    print("Sample:")
    print("Add James Bond as a new user")

    while True:
        query = input("🤔 ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        conversation.add_message(Message(role=Role.USER, content=query))

        agent_message = agent.get_response(
            conversation.get_messages(), print_request=True
        )
        conversation.add_message(agent_message)
        print(f"🤖 {agent_message.content}")
        print("=" * 100)
        print()


main()
