import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from commons.constants import GPT_5_4_NANO, OPENAI_API_KEY
from t6_grounding.user_service_client import UserServiceClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BATCH_SYSTEM_PROMPT = """
You are a user search assistant.

INSTRUCTIONS:
- Analyze the search criteria from the user questions.
- Examine each user in the provided list and determine if they match.
- For matching user, extract and return their complete information.
- Be inclusive - if a user partially matches or could potentially match, include them.


OUTPUT FORMAT:
- If you find matching users, return full details exactly as provided, maintaining original format.
- If uncertain about a match: Include the user with a note about why they might match.
- Return exactly "NO_MATCHES_FOUND" if no users match.
"""

FINAL_SYSTEM_PROMPT = """
You are a helpful assistant that provices comprehensive answers based on user search results.
"""

USER_PROMPT = """
STRUCTURE:

<User Data>
{context}

<search Query>
{query}
"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens: list[int] = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "batch_count": len(self.batch_tokens),
            "batch_tokens": self.batch_tokens,
        }


llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    user_strings = []

    for user in context:
        lines = ["USER:"]
        for key, value in user.items():
            lines.append(f"  {key}: {value}")
        user_strings.append("\n".join(lines))

    return "\n\n".join(user_strings)


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Sending request to LLM...")

    input_messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        llm_message = await llm_client.chat.completions.create(
            model=GPT_5_4_NANO, temperature=0.0, messages=input_messages
        )
        usage = getattr(llm_message, "usage", None)
        total_tokens = usage.total_tokens if usage else 0
        token_tracker.add_tokens(total_tokens)
        content = llm_message.choices[0].message.content or ""
        print(f"\nToken Count: {total_tokens}\nContent: {content}")
        return content
    except (OpenAIError, TimeoutError, OSError) as e:
        print(f"API Error encountered: {e}")
        return "NO_MATCHES_FOUND"


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("𖨆: ").strip()

    if not user_question.strip():
        print("Message cannot be empty!")
        return

    user_service_client = UserServiceClient()
    BATCH_SIZE = 100
    all_users = user_service_client.get_all_users()
    user_batches = [
        all_users[i : i + BATCH_SIZE] for i in range(0, len(all_users), BATCH_SIZE)
    ][0:5]

    parallel_tasks = [
        generate_response(
            BATCH_SYSTEM_PROMPT,
            USER_PROMPT.format(context=join_context(batch), query=user_question),
        )
        for batch in user_batches
    ]

    batch_result = await asyncio.gather(*parallel_tasks)

    print("\n--- Compiling results ---")
    relevant_results = [
        res
        for res in batch_result
        if "NO_MATCHES_FOUND" not in res.upper() and res.strip()
    ]

    print("\n=== SEARCH RESULTS ===")
    if not relevant_results:
        print("No users found. Update your search query to get better results.")
        return

    final_response = await generate_response(
        FINAL_SYSTEM_PROMPT,
        USER_PROMPT.format(context="\n\n".join(relevant_results), query=user_question),
    )
    print(final_response)

    print("\n=== Performance ===")
    print(token_tracker.get_summary())


if __name__ == "__main__":
    asyncio.run(main())
