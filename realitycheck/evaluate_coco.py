import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance
from ultralytics import YOLO

from realitycheck.coco_eval import evaluate_image
from realitycheck.evaluate import extract_detections


def load_coco_ground_truth(annotation_path, image_path, model_names):
    data = json.loads(Path(annotation_path).read_text())

    image_path = Path(image_path)

    matching_images = [
        image for image in data["images"]
        if image["file_name"] == image_path.name
    ]

    if len(matching_images) != 1:
        raise ValueError(
            f"Expected exactly one COCO image matching "
            f"{image_path.name}; found {len(matching_images)}"
        )

    image_info = matching_images[0]
    image_id = image_info["id"]

    category_names = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    yolo_ids = {
        name.lower(): class_id
        for class_id, name in model_names.items()
    }

    ground_truth = []

    for annotation in data["annotations"]:
        if annotation["image_id"] != image_id:
            continue

        # COCO crowd regions require special evaluation rules.
        # Exclude them from this initial simplified evaluator.
        if annotation.get("iscrowd", 0):
            continue

        category_name = category_names[
            annotation["category_id"]
        ].lower()

        if category_name not in yolo_ids:
            continue

        x, y, w, h = annotation["bbox"]

        ground_truth.append({
            "class_id": yolo_ids[category_name],
            "bbox": [x, y, x + w, y + h],
            "category": category_name,
        })

    return ground_truth, image_info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--annotations", required=True)

    args = parser.parse_args()

    model = YOLO("yolo11n.pt")

    image = Image.open(args.image).convert("RGB")

    ground_truth, image_info = load_coco_ground_truth(
        args.annotations,
        args.image,
        model.names,
    )

    if image.size != (
        image_info["width"],
        image_info["height"],
    ):
        raise ValueError(
            "Image dimensions do not match COCO annotations"
        )

    print("Image:", args.image)
    print("Ground-truth objects:", len(ground_truth))

    experiments = {
        "baseline": image,
        "brightness_0_6": ImageEnhance.Brightness(
            image
        ).enhance(0.6),
    }

    report = {
        "image": Path(args.image).name,
        "ground_truth_count": len(ground_truth),
        "evaluation": {},
    }

    for name, experiment_image in experiments.items():
        result = model.predict(
            source=experiment_image,
            conf=0.25,
            imgsz=640,
            device="cpu",
            verbose=False,
        )[0]

        predictions = extract_detections(result)

        metrics = evaluate_image(
            predictions,
            ground_truth,
            iou_threshold=0.5,
        )

        report["evaluation"][name] = metrics

        print(f"\n{name}")
        print("TP:", metrics["tp"])
        print("FP:", metrics["fp"])
        print("FN:", metrics["fn"])
        print("Precision:", metrics["precision"])
        print("Recall:", metrics["recall"])

    baseline = report["evaluation"]["baseline"]
    perturbed = report["evaluation"]["brightness_0_6"]

    if (
        baseline["recall"] is not None
        and perturbed["recall"] is not None
    ):
        delta = perturbed["recall"] - baseline["recall"]

        print(f"\nRecall change: {delta:+.3f}")

    output = Path("outputs/evaluation")
    output.mkdir(parents=True, exist_ok=True)

    (output / "coco_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nSaved:", output / "coco_report.json")


if __name__ == "__main__":
    main()
