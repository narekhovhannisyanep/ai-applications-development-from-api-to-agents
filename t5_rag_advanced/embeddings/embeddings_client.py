import json

import requests
from openai.types import CreateEmbeddingResponse, EmbeddingCreateParams

from commons.constants import OPENAI_API_KEY, OPENAI_EMBEDDINGS_ENDPOINT


class EmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, endpoint: str, model_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = endpoint
        self._api_key = "Bearer " + api_key
        self._model_name = model_name

    def get_embeddings(
        self,
        inputs: str | list[str],
        dimensions: int | None,
        print_response: bool = False,
    ) -> dict[int, list[float]]:
        """
        Generate dict of indexed embeddings:
            inputs[0](text) -> [0][embedding]
            inputs[1](text) -> [1][embedding]
            ...

        Args:
            inputs: input text, can be singular string or list of strings
            dimensions: number of dimensions
            print_response: to print response in chat or not
        """
        validated_inputs = inputs if isinstance(inputs, list) else [inputs]

        if not validated_inputs or any(not input.strip() for input in validated_inputs):
            raise ValueError(
                "Inputs list cannot be empty, and cannot contain empty strings."
            )

        headers = {"Content-Type": "application/json", "Authorization": self._api_key}

        request_body: EmbeddingCreateParams = {
            "model": self._model_name,
            "input": validated_inputs,
        }
        if dimensions is not None:
            request_body["dimensions"] = dimensions

        try:
            response = requests.post(
                url=self._endpoint,
                headers=headers,
                json=request_body,
                timeout=(3.05, 27),
            )
            response.raise_for_status()

        except requests.exceptions.HTTPError as http_err:
            raise requests.exceptions.HTTPError(
                f"OpenAI API Error {response.status_code} {response.text}"
            ) from http_err
        except requests.exceptions.Timeout as timeout_err:
            raise requests.exceptions.Timeout(
                "The request to OpenAI timed out."
            ) from timeout_err

        validated_response = CreateEmbeddingResponse.model_validate(response.json())

        embeddings_dict = {
            item.index: item.embedding for item in validated_response.data
        }

        if print_response:
            print(json.dumps(embeddings_dict, indent=2))

        return embeddings_dict
