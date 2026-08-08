import base64
import pathlib
from datetime import datetime

import requests

from commons.constants import OPENAI_API_KEY, OPENAI_HOST

# https://developers.openai.com/api/reference/resources/images/methods/edit
# ---
# Request (multipart/form-data, NOT json):
# curl -X POST "https://api.openai.com/v1/images/edits" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -F "model=gpt-image-1" \
#     -F "image=@logo.png" \
#     -F "prompt=Add magical sparkles and glowing aura around the logo"
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": "Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO..."
#     }
#   ]
# }

# TODO:
# You need to edit an existing image with `gpt-image-2` model:
#   - Take a local image (e.g. 'logo.png') and a prompt describing the edit
#   - Send it to the OpenAI images edit API
#   - Decode the returned base64 image and save it locally
# ---
# Hints:
#   - Use /v1/images/edits endpoint
#   - The request must be 'multipart/form-data' (NOT json) — pass the image as a file and the prompt/model as form fields
#   - The edited image will be returned in base64 format

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
