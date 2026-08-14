import base64
import pathlib

import requests

from commons.constants import (
    OPENAI_API_KEY,
    OPENAI_CHAT_COMPLETIONS_ENDPOINT,
)

current_dir = pathlib.Path(__file__).resolve().parent
question_file_name = current_dir / "question.mp3"
headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

with open(question_file_name, "rb") as question_file:
    encoded_question = base64.b64encode(question_file.read()).decode("utf-8")
    json_payload = {
        "model": "gpt-audio-mini",
        "modalities": ["text", "audio"],
        "audio": {"voice": "nova", "format": "mp3"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in the recording?"},
                    {
                        "type": "input_audio",
                        "input_audio": {"format": "mp3", "data": encoded_question},
                    },
                ],
            }
        ],
    }

    response = requests.post(
        url=OPENAI_CHAT_COMPLETIONS_ENDPOINT, headers=headers, json=json_payload
    )

if response.status_code != 200:
    raise requests.exceptions.HTTPError(f"HTTP {response.status_code} {response.text}")

data = response.json()
audio_content = (
    data.get("choices", [])[0].get("message", {}).get("audio", {}).get("data")
)
print(data)
print()
with open("audio_output.mp3", "wb") as audio_output_file:
    audio_output_file.write(base64.b64decode(audio_content))
