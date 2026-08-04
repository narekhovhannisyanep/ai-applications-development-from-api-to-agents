import json

import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Google Gemini API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Gemini's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    _last_interaction_id = None

    def _extract_content(self, data: dict) -> str:
        """
        Extract content from the response data.

        Args:
            data (dict): The response data.

        Returns:
            str: the text content extracted from the response data.
        """
        contents = []

        for step in data.get("steps", []):
            if "content" in step:
                for content in step.get("content", []):
                    if content.get("type") == "text":
                        contents.append(content.get("text"))

        return "".join(contents)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no candidates.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The URL is constructed by appending ':generateContent' to the model endpoint.
            Uses 'x-goog-api-key' header for authentication.
            Response candidates contain content parts that are concatenated.
        """
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}

        request_data = {
            "model": self._model_name,
            "system_instruction": self._system_prompt,
            "input": messages[-1].content,
            "store": True,
            "generation_config": {"max_output_tokens": kwargs.get("max_tokens", 1024)},
        }

        if self._last_interaction_id:
            request_data["previous_interaction_id"] = self._last_interaction_id

        try:
            response = requests.post(
                url="https://generativelanguage.googleapis.com/v1beta/interactions",
                headers=headers,
                json=request_data,
                timeout=kwargs.get("timeout", 10),
            )
        except requests.exceptions.RequestException as req_err:
            print(f"A Reques error occurred: {req_err}")

        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code} {response.text}")

        response_data = response.json()
        self._last_interaction_id = response_data.get("id")

        content = self._extract_content(response_data)
        print(content)

        return Message(role="model", content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Gemini's SSE format, with text chunks
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The URL is constructed with ':streamGenerateContent?alt=sse' endpoint.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each SSE chunk contains candidates with content parts.
            Each text chunk is printed to stdout as it arrives.
        """
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}
        request_data = {
            "model": self._model_name,
            "system_instruction": self._system_prompt,
            "input": messages[-1].content,
            "store": True,
            "stream": True,
            "generation_config": {"max_output_tokens": kwargs.get("max_tokens", 1024)},
        }
        if self._last_interaction_id:
            request_data["previous_interaction_id"] = self._last_interaction_id

        print("self._last_interaction_id: ", self._last_interaction_id)

        contents = []

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    url="https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse",
                    headers=headers,
                    json=request_data,
                ) as response,
            ):
                if response.status != 200:
                    error_text = await response.text()
                    print(f"{response.status} {error_text}")
                    return

                async for line in response.content:
                    line_str = line.decode("utf-8")
                    if not line_str.startswith("data: "):
                        continue

                    data = json.loads(line_str[6:])
                    event_type = data.get("event_type")

                    if event_type == "error":
                        print(data.get("error", {}).get("message"))
                        continue

                    if event_type == "interaction.created":
                        self._last_interaction_id = data.get("interaction", {}).get(
                            "id"
                        )
                        continue

                    if event_type == "step.delta":
                        delta = data.get("delta", {})
                        if delta_text := delta.get("text"):
                            print(delta_text, end="")
                            contents.append(delta_text)

        except ValueError:
            # Skip json.loads errors. (e.g. data: [Done])
            pass
        except requests.exceptions.RequestException as req_err:
            raise ValueError(
                f"A Request error occured: {req_err}"
            ) from requests.exceptions.RequestException

        print()
        return Message(role="model", content="".join(contents))
