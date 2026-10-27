import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path("data/coco")
OUT = Path("data/samples")
OUT.mkdir(parents=True, exist_ok=True)

with open(ROOT / "annotations/instances_val2017.json") as f:
    coco = json.load(f)

categories = {
    c["id"]: c["name"]
    for c in coco["categories"]
}

images = {
    img["id"]: img
    for img in coco["images"]
}

annotations = defaultdict(list)

for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)

candidates = []

for image_id, anns in annotations.items():
    names = {
        categories[a["category_id"]]
        for a in anns
    }

    if "person" in names and (
        "car" in names or "bicycle" in names
    ):
        candidates.append(image_id)

random.seed(42)
selected = random.sample(
    sorted(candidates),
    min(20, len(candidates))
)

manifest = []

for image_id in selected:
    img = images[image_id]

    src = ROOT / "val2017" / img["file_name"]
    dst = OUT / img["file_name"]

    shutil.copy2(src, dst)

    manifest.append({
        "image_id": image_id,
        "filename": img["file_name"],
        "license_id": img["license"],
        "coco_url": img.get("coco_url"),
        "flickr_url": img.get("flickr_url"),
        "annotations": annotations[image_id]
    })

with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Selected {len(selected)} images")
print(f"Saved to {OUT}")
