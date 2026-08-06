import asyncio

from commons.constants import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_ENDPOINT,
    CLAUDE_HAIKU_4_5,
    DEFAULT_SYSTEM_PROMPT,
)
from t1_llm_api.anthropic.client import AnthropicAIClient
from t1_llm_api.anthropic.custom_client import CustomAnthropicAIClient
from t1_llm_api.base_app import start

anthropic_client = AnthropicAIClient(
    endpoint=ANTHROPIC_ENDPOINT,
    model_name=CLAUDE_HAIKU_4_5,
    api_key=ANTHROPIC_API_KEY,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)
anthropic_custom_client = CustomAnthropicAIClient(
    endpoint=ANTHROPIC_ENDPOINT,
    model_name=CLAUDE_HAIKU_4_5,
    api_key=ANTHROPIC_API_KEY,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)

asyncio.run(start(True, anthropic_custom_client))
