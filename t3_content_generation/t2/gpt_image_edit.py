import base64
import pathlib

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST

url = f"{OPENAI_HOST}/v1/images/edits"
current_dir = pathlib.Path(__file__).resolve().parent
input_image_path = current_dir / "logo.png"

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

with open(input_image_path, "rb") as logo:
    data_payload = {
        "model": "gpt-image-1-mini",
        "prompt": "Add magical sparkles and glowing aura around the logo.",
        # "image": [logo],
        "size": "auto",
        "quality": "low",
        "output_format": "png",
        "n": 1,
    }

    files_payload = [("image", ("logo.png", logo, "image/png"))]

    edited_logo_response = requests.post(
        url=url, headers=headers, data=data_payload, files=files_payload
    )

edited_logo_data = edited_logo_response.json()
print("edited_logo_data", edited_logo_data)
edited_image = base64.b64decode(edited_logo_data.get("data", [])[0].get("b64_json"))

with open(current_dir / "edited_logo.png", "wb") as f:
    f.write(edited_image)
