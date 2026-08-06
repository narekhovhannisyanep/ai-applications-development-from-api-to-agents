import asyncio

from commons.constants import (
    DEFAULT_SYSTEM_PROMPT,
    GPT_5_4_NANO,
    OPENAI_API_KEY,
    OPENAI_RESPONSES_ENDPOINT,
)
from t1_llm_api.base_app import start
from t1_llm_api.openai.responses.client import OpenAIResponsesClient
from t1_llm_api.openai.responses.custom_client import CustomOpenAIResponsesClient

openai_client = OpenAIResponsesClient(
    endpoint=OPENAI_RESPONSES_ENDPOINT,
    model_name=GPT_5_4_NANO,
    api_key=OPENAI_API_KEY,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)
openai_custom_client = CustomOpenAIResponsesClient(
    endpoint=OPENAI_RESPONSES_ENDPOINT,
    model_name=GPT_5_4_NANO,
    api_key=OPENAI_API_KEY,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)

asyncio.run(start(True, openai_custom_client))
