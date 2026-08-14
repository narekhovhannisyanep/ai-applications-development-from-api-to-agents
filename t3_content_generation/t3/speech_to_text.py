import pathlib

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST

url = f"{OPENAI_HOST}/v1/audio/transcriptions"
current_dir = pathlib.Path(__file__).resolve().parent
sample_audio_file_name = current_dir / "audio_sample.mp3"

headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

data_payload = {
    "model": "gpt-4o-mini-transcribe",
    "languages": "en",
    "response_format": "text",
    "prompt": "Pay attention to 'Dial Commnunity' name.",
}

with open(sample_audio_file_name, "rb") as sample_audio_file:
    files_payload = {"file": (sample_audio_file)}
    response = requests.post(
        url=url, headers=headers, data=data_payload, files=files_payload, timeout=10
    )
    # response.raise_for_status()

if response.status_code != 200:
    raise requests.exceptions.HTTPError(f"HTTP {response.status_code} {response.text}")

# payload = response.json()
print(response.text)
