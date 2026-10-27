import os
import sys
import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["MODEL_API_KEY"],
    base_url="https://api.meta.ai/v1",
    timeout=120.0,
    max_retries=1,
)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    raise FileNotFoundError(image_path)

mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"

image_base64 = base64.b64encode(
    image_path.read_bytes()
).decode("ascii")

image_url = f"data:{mime};base64,{image_base64}"

response = client.responses.create(
    model="sam-3.1",
    input=[
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "person",
                },
                {
                    "type": "input_image",
                    "image_url": image_url,
                },
            ],
        }
    ],
    metadata={"mask_encoding": "one_bit"},
)

print("Response received")
print("Response status:", response.status)

output = response.output_text or ""

print("Output length:", len(output))
print("Output preview:", output[:300])

if not output:
    print("No matching objects found.")
else:
    Path("sam_response.txt").write_text(output)
    print("Saved segmentation output.")
