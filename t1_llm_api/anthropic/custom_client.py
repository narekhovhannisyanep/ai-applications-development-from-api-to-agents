import json

import aiohttp
import requests

from anthropic.types import (
    MessageParam,
)
from anthropic.types.message_create_params import (
    MessageCreateParamsNonStreaming,
    MessageCreateParamsStreaming,
)
from commons.constants import ANTHROPIC_VERSION
from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomAnthropicAIClient(AIClient):
    """
    Custom HTTP client for Anthropic's Claude API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Claude's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no content blocks.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Requires 'x-api-key' header and 'anthropic-version' header.
            Claude's API returns content as an array of content blocks.
            The response is printed to stdout before being returned.
        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

        input_messages: list[MessageParam] = [
            MessageParam(
                role="user" if msg.role == "user" else "assistant", content=msg.content
            )
            for msg in messages
        ]

        request_options: MessageCreateParamsNonStreaming = {
            "model": self._model_name,
            "system": self._system_prompt,
            "messages": input_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "cache_control": {"type": "ephemeral"},
        }

        aiResponse = requests.post(
            url=self._endpoint, headers=headers, json=request_options
        )

        if not aiResponse.ok:
            raise ValueError(f"HTTP {aiResponse.status_code} {aiResponse.text}")

        data = aiResponse.json()
        content = data.get("content", [])[0].get("text", "AI message is missing!!!")
        print(content)
        return Message(Role.ASSISTANT, content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Anthropic's SSE format, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Listens for 'content_block_delta' events with 'text_delta' type.
            Stops processing when 'message_stop' event is received.
            Each delta is printed to stdout as it arrives.
        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

        input_messages: list[MessageParam] = [
            MessageParam(
                role="user" if msg.role == "user" else "assistant", content=msg.content
            )
            for msg in messages
        ]

        request_options: MessageCreateParamsStreaming = {
            "model": self._model_name,
            "system": self._system_prompt,
            "messages": input_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "cache_control": {"type": "ephemeral"},
            "stream": True,
        }

        deltaContents: list[str] = []

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                url=self._endpoint, headers=headers, json=request_options
            ) as stream,
        ):
            if not stream.ok:
                error_text = await stream.text()
                raise RuntimeError(
                    f"Anthropic API Error ({stream.status}): {error_text}"
                )

            async for line_bytes in stream.content:
                line = line_bytes.decode("utf-8").strip()

                if not line or not line.startswith("data: "):
                    continue

                try:
                    data = json.loads(line[6:].strip())
                except json.JSONDecodeError:
                    continue

                if (
                    data.get("type") == "content_block_delta"
                    and data.get("delta", {}).get("type") == "text_delta"
                ):
                    deltaContents.append(data.get("delta", {}).get("text", ""))
                    print(data.get("delta").get("text"), end="", flush=True)

        print()
        return Message(Role.ASSISTANT, "".join(deltaContents))
