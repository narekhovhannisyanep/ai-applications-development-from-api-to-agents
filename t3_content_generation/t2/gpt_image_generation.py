import base64
import pathlib

from commons.constants import OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3

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
