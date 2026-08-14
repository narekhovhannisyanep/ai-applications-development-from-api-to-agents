import json
from datetime import datetime

import requests

from commons.constants import GPT_5_4_NANO, OPENAI_API_KEY, OPENAI_HOST


class Voice:
    alloy: str = "alloy"
    ash: str = "ash"
    ballad: str = "ballad"
    coral: str = "coral"
    echo: str = "echo"
    fable: str = "fable"
    nova: str = "nova"
    onyx: str = "onyx"
    sage: str = "sage"
    shimmer: str = "shimmer"


url = f"{OPENAI_HOST}/v1/audio/speech"
headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}
json_payload = {
    "model": "gpt-4o-mini-tts",
    "input": "Why can't we say that black is white?",
    "voice": "nova",
    "response_format": "mp3",
}

response = requests.post(url=url, headers=headers, json=json_payload)
if response.status_code != 200:
    raise requests.exceptions.HTTPError(f"HTTP {response.status_code} {response.text}")

print(response)

with open("audio_output.mp3", "wb") as audio_output:
    audio_output.write(response.content)
