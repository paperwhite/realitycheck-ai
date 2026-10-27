import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
api_key=os.environ["MODEL_API_KEY"],
base_url="https://api.meta.ai/v1",)

response=client.responses.create(
model="muse-spark-1.3",
input=(
"You are helping build RealityCheckAI, a computer vision robustness testing system."
"Suggest three deterministic image transformations that could test object detection robustness."
),)

print(response.output_text)
