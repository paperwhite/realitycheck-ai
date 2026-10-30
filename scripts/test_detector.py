from pathlib import Path
import sys

from PIL import Image, ImageEnhance
from ultralytics import YOLO
import json
from realitycheck.evaluate import (
    extract_detections,
    compare_detections,
)

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

predictions = {}

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
    predictions[label] = extract_detections(result)

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

report = compare_detections(
    predictions["baseline"],
    predictions["brightness_0_6"],
    iou_threshold=0.5,
)

report_path = output_dir / "robustness_report.json"

report_path.write_text(json.dumps(report, indent=2))

print("\n=== ROBUSTNESS REPORT ===")
print("Baseline detections:", report["baseline_count"])
print("Perturbed detections:", report["perturbed_count"])
print("Matched:", report["matched_count"])
print("Lost:", report["lost_count"])
print("New:", report["new_count"])

persistence = report["detection_persistence"]

if persistence is not None:
    print(f"Detection persistence: {persistence:.1%}")
else:
    print("Detection persistence: N/A (no baseline detections)")

delta = report["mean_confidence_delta"]

if delta is not None:
    print(f"Mean confidence change: {delta:+.3f}")
else:
    print("Mean confidence change: N/A (no matches)")

print("Report saved:", report_path)
