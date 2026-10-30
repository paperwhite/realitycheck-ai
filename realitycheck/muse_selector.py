import json
import os

from openai import OpenAI

from realitycheck.transforms import ALLOWED_TRANSFORMS


def choose_next_experiment(experiment_history):
    """
    Ask Muse Spark to select a controlled robustness test.
    """

    client = OpenAI(
        api_key=os.environ["MODEL_API_KEY"],
        base_url="https://api.meta.ai/v1",
    )

    prompt = f"""
You are an experiment planner for an object-detection
robustness testing system.

Your task is to select ONE new experiment.

Allowed transformations and parameter values:

{json.dumps(ALLOWED_TRANSFORMS, indent=2)}

Previous experiments:

{json.dumps(experiment_history, indent=2)}

Requirements:
1. Choose only from the allowed transformations.
2. Choose only an allowed parameter value.
3. Do not repeat an existing experiment.
4. Prefer an experiment that tests a different
   failure mode or helps investigate an observed failure.
5. Explain the hypothesis in one short sentence.

Return ONLY a JSON object with this structure:

{{
  "transform": "brightness",
  "value": 0.4,
  "hypothesis": "Lower illumination may cause missed detections."
}}
"""

    response = client.responses.create(
        model="muse-spark-1.3",
        input=prompt,
    )

    raw = response.output_text.strip()

    # Handle optional Markdown JSON fences.
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()

    proposal = json.loads(raw)

    if not isinstance(proposal, dict):
        raise ValueError("Muse response must be a JSON object")

    transform = proposal["transform"]
    value = proposal["value"]

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or transform not in ALLOWED_TRANSFORMS
        or value not in ALLOWED_TRANSFORMS[transform]
    ):
        raise ValueError(
            f"Muse selected an invalid experiment: {proposal}"
        )

    previous = {
        (item["transform"], item["value"])
        for item in experiment_history
        if "transform" in item and "value" in item
    }

    if (transform, value) in previous:
        raise ValueError(
            "Muse selected an experiment already performed"
        )

    return {
        "transform": transform,
        "value": value,
        "hypothesis": str(proposal.get("hypothesis", ""))[:300],
    }
