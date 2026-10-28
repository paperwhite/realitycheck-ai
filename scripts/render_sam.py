import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
)


def main(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    raw = Path("outputs/sam_raw.txt").read_text()

    parser = image_segmentation_format().create_parser()
    parser.push(raw, emit=False)
    finished = parser.finish(CompletedOutcome())

    # ParserFinish exposes the final result; verify its field if needed.
    result = finished.result

    boxes = {}
    masks = {}

    for record in result.records:
        if isinstance(record, SegmentationBoxRecord):
            boxes[record.object_id] = record
        elif isinstance(record, SegmentationMaskRecord):
            masks[record.object_id] = record

    print("Boxes:", len(boxes))
    print("Masks:", len(masks))
    print("Image size:", image.size)

    overlay = image.copy()
    colors = [
        (255, 60, 60),
        (60, 220, 100),
        (60, 130, 255),
        (255, 180, 40),
        (200, 80, 240),
    ]

    combined_mask = Image.new("L", image.size, 0)
    objects = []

    for index, (object_id, record) in enumerate(masks.items()):
        mask = record.mask
        raster = decode_mask_to_raster(mask)

        assert len(raster) == mask.width * mask.height

        local_mask = Image.frombytes(
            "L",
            (mask.width, mask.height),
            bytes(raster),
        )

        # Convert 0/1 mask to 0/255.
        local_mask = local_mask.point(lambda p: 255 if p else 0)

        bounds = record.bounds
        left = round(bounds.left)
        top = round(bounds.top)
        right = round(bounds.right)
        bottom = round(bounds.bottom)

        if right <= left or bottom <= top:
            print("Skipping invalid bounds:", object_id)
            continue

        if local_mask.size != (right - left, bottom - top):
            local_mask = local_mask.resize(
                (right - left, bottom - top),
                Image.Resampling.NEAREST,
            )

        full_mask = Image.new("L", image.size, 0)
        full_mask.paste(local_mask, (left, top))
        combined_mask = Image.frombytes(
            "L",
            image.size,
            bytes(
                max(a, b)
                for a, b in zip(
                    combined_mask.tobytes(),
                    full_mask.tobytes(),
                )
            ),
        )

        color = colors[index % len(colors)]
        tint = Image.new("RGB", image.size, color)
        alpha = full_mask.point(lambda p: round(p * 0.4))

        overlay.paste(
            Image.composite(tint, overlay, alpha),
        )

        box = boxes.get(object_id)
        if box:
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(
                [box.left, box.top, box.right, box.bottom],
                outline=color,
                width=2,
            )
            draw.text(
                (box.left, max(0, box.top - 12)),
                f"ID {object_id}",
                fill=color,
            )

        objects.append({
            "object_id": object_id,
            "bounds": [left, top, right, bottom],
            "mask_width": mask.width,
            "mask_height": mask.height,
        })

    output = Path("outputs")
    output.mkdir(exist_ok=True)

    overlay.save(output / "sam_overlay.png")
    combined_mask.save(output / "sam_mask.png")

    (output / "sam_objects.json").write_text(
        json.dumps(objects, indent=2)
    )

    print("Rendered objects:", len(objects))
    print("Saved outputs/sam_overlay.png")
    print("Saved outputs/sam_mask.png")
    print("Saved outputs/sam_objects.json")


if __name__ == "__main__":
    main(sys.argv[1])
