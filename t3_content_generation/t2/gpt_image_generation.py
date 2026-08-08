import base64
import pathlib

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3

# https://developers.openai.com/api/reference/resources/images/methods/generate
# ---
# Request:
# curl -X POST "https://api.openai.com/v1/images/generations" \
#     -H "Authorization: Bearer $OPENAI_API_KEY" \
#     -H "Content-type: application/json" \
#     -d '{
#         "model": "gpt-image-2",
#         "prompt": "smiling catdog."
#     }'
# Response:
# {
#   "created": 1699900000,
#   "data": [
#     {
#       "b64_json": Qt0n6ArYAEABGOhEoYgVAJFdt8jM79uW2DO...,
#     }
#   ]
# }


# TODO:
# You need to create some images with `gpt-image-2` model:
#   - Generate an image with 'Smiling catdog'
#   - Decode and save it locally
# ---
# Hints:
#   - Use OpenAIClientT3 to connect to OpenAI API
#   - Use /v1/images/generations endpoint
#   - The image will be returned in base64 format

image_response = OpenAIClientT3(f"{OPENAI_HOST}/v1/images/generations").call(
    print_request=True,
    print_response=True,
    model="gpt-image-1-mini",
    prompt="smiling catdog",
    n=1,
    size="1024x1024",
    quality="low",
)

image_bytes = base64.b64decode(image_response.get("data", [])[0].get("b64_json"))

catdog_image_file = pathlib.Path(__file__).resolve().parent / "catdog.png"

with open(catdog_image_file, "wb") as f:
    f.write(image_bytes)
