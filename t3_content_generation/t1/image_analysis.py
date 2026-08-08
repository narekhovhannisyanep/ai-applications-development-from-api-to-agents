import base64
from pathlib import Path

from commons.constants import GPT_5_4_NANO, OPENAI_HOST
from t3_content_generation._openai_client import OpenAIClientT3


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


parent_dir = Path(__file__).resolve().parent
dial_logo_path = parent_dir / "logo.png"
base64_dial_logo = encode_image(dial_logo_path)

print(f"base64_dial_logo : {base64_dial_logo}")

input_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Generate poem based on images."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg"
                },
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_dial_logo}"},
            },
        ],
    }
]

OpenAIClientT3(f"{OPENAI_HOST}/v1/chat/completions").call(
    print_request=True,
    print_response=True,
    # model="gpt-realtime-2.1-mini",
    model=GPT_5_4_NANO,
    messages=input_messages,
    max_completion_tokens=1024,
)
