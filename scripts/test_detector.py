from pathlib import Path
import sys

from PIL import Image, ImageEnhance
from ultralytics import YOLO

image_path = Path(sys.argv[1])

output_dir = Path("outputs/detector")
output_dir.mkdir(parents=True, exist_ok=True)

# Downloaded automatically on first run.
model = YOLO("yolo11n.pt")

original = Image.open(image_path).convert("RGB")

# Deterministic brightness reduction.
darkened = ImageEnhance.Brightness(original).enhance(0.6)

darkened_path = output_dir / "darkened.jpg"
darkened.save(darkened_path, quality=95)

for label, image in [
    ("baseline", original),
    ("brightness_0_6", darkened),
]:
    results = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    print(f"\n{label}")
    print("Detections:", len(result.boxes))

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        print(
            f"  {model.names[class_id]}: "
            f"{confidence:.3f}"
        )

    annotated = Image.fromarray(result.plot()[..., ::-1])
    annotated.save(output_dir / f"{label}.png")

print("\nSaved outputs to:", output_dir)
