import asyncio

from commons.constants import (
    DEFAULT_SYSTEM_PROMPT,
    OPENAI_API_KEY,
    OPENAI_CHAT_COMPLETIONS_ENDPOINT,
)
from t1_llm_api.base_app import start
from t1_llm_api.openai.chat.completions.client import OpenAIClient
from t1_llm_api.openai.chat.completions.custom_client import CustomOpenAIClient

GPT_5_4_NANO = "gpt-5.4-nano"

openai_client = OpenAIClient(
    endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    model_name=GPT_5_4_NANO,
    api_key=OPENAI_API_KEY,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)
openai_custom_client = CustomOpenAIClient(
    endpoint=OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    model_name=GPT_5_4_NANO,
    api_key=OPENAI_API_KEY,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)

asyncio.run(start(True, openai_client))
