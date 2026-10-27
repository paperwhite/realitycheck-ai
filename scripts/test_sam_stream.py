import asyncio
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from meta_sam_parser import (
    parse_responses_stream,
    image_segmentation_format,
    decode_mask_to_raster,
)

load_dotenv()

async def main():
    image_path = Path(sys.argv[1])
    image_bytes = image_path.read_bytes()

    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")

    client = AsyncOpenAI(
        api_key=os.environ["MODEL_API_KEY"],
        base_url="https://api.meta.ai/v1",
        timeout=120.0,
    )

    events = await client.responses.create(
        model="sam-3.1",
        stream=True,
        input=[{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "person"},
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{encoded}",
                },
            ],
        }],
        metadata={"mask_encoding": "one_bit"},
    )

    parsed = parse_responses_stream(
        events,
        image_segmentation_format(),
    )

    result = await parsed.final_result()

    print("Outcome:", result.outcome)
    print("Records:", len(result.records))
    print("Diagnostics:", len(result.diagnostics))

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    (output / "sam_raw.txt").write_text(result.raw_output)

    summary = []

    for record in result.records:
        print(
            "Record:",
            type(record).__name__,
            "Object:",
            getattr(record, "object_id", None),
        )

        if hasattr(record, "mask"):
            mask = record.mask
            raster = decode_mask_to_raster(mask)

            print(
                "Mask dimensions:",
                mask.width,
                mask.height,
            )
            print("Decoded bytes:", len(raster))

        summary.append({
            "type": type(record).__name__,
            "object_id": getattr(record, "object_id", None),
        })

    (output / "sam_records.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("Saved results to outputs/")

asyncio.run(main())
