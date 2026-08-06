import json

import requests

from commons.constants import GEMINI_API_KEY, GEMINI_ENDPOINT
from commons.models.message import Message
from commons.models.role import Role
from t2_llms_output_tuning._clients._base_client import AIClient


class GeminiAIClient(AIClient):
    def __init__(self, model_name: str):
        super().__init__(
            endpoint=GEMINI_ENDPOINT,
            model_name=model_name,
            api_key=GEMINI_API_KEY,
            api_key_header_name="x-goog-api-key",
        )
        self.current_interactions_id = None

    def response(
        self,
        messages: list[Message],
        print_request: bool,
        print_only_content: bool,
        **kwargs,
    ) -> Message:
        url = self._endpoint

        headers = {"Content-Type": "application/json", "x-goog-api-key": self._api_key}

        generation_config = kwargs.get("generation_config", {})
        generation_config.setdefault("max_output_tokens", 1024)
        request_data = {
            "input": messages[-1].content,
            "generation_config": generation_config,
            "model": self._model_name,
            **kwargs,
        }
        if self.current_interactions_id:
            request_data["previous_interaction_id"] = self.current_interactions_id

        safety_settings = kwargs.get("safetySettings")
        if safety_settings:
            request_data["safetySettings"] = safety_settings

        if print_request:
            self._print_request(request_data, headers)

        response = requests.post(url=url, headers=headers, json=request_data)

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        data = response.json()

        if data.get("id"):
            self.current_interactions_id = data.get("id")

        # print(json.dumps(data, indent=2, sort_keys=True))

        steps = data.get("steps")
        if not steps:
            raise ValueError("No steps present in the response!!!")

        for step in steps:
            if step.get("type") == "model_output":
                content = step.get("content")
                break
        else:
            raise ValueError("No content present in the steps!!!")

        for contentPart in content:
            if contentPart.get("type") == "text":
                content_text = contentPart.get("text")
                break
        else:
            raise ValueError("No content text is present in content!!!")

        # parts = candidates[0].get("content", {}).get("parts", [])
        # content = "".join(part.get("text", "") for part in parts)
        print("" + "=" * 50 + " RESPONSE " + "=" * 50)
        if print_only_content:
            print(content_text)
        else:
            print(json.dumps(data, indent=2, sort_keys=True))
        print("=" * 109)
        return Message(Role.ASSISTANT, content)

    # @staticmethod
    # def _to_gemini_contents(messages: list[Message]) -> list[dict]:
    #     contents = []
    #     for msg in messages:
    #         contents.append({"role": msg.role.value, "parts": [{"text": msg.content}]})
    #     return contents
