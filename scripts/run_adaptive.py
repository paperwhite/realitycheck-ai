import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

from realitycheck.coco_eval import evaluate_image
from realitycheck.evaluate import (
    extract_detections,
    compare_detections,
)
from realitycheck.transforms import apply_transform
from realitycheck.muse_selector import choose_next_experiment

# Adjust this import to match where your loader currently lives.
from realitycheck.evaluate_coco import load_coco_ground_truth


load_dotenv()


def detect(model, image):
    result = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        device="cpu",
        verbose=False,
    )[0]

    return extract_detections(result)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--iterations", type=int, default=3)

    args = parser.parse_args()

    if not 1 <= args.iterations <= 10:
        raise ValueError("Iterations must be between 1 and 10")

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")

    model = YOLO("yolo11n.pt")

    ground_truth, image_info = load_coco_ground_truth(
        args.annotations,
        args.image,
        model.names,
    )

    if image.size != (
        image_info["width"],
        image_info["height"],
    ):
        raise ValueError("Image dimensions do not match annotations")

    baseline_predictions = detect(model, image)

    baseline_metrics = evaluate_image(
        baseline_predictions,
        ground_truth,
    )

    print("\n=== BASELINE ===")
    print("TP:", baseline_metrics["tp"])
    print("FP:", baseline_metrics["fp"])
    print("FN:", baseline_metrics["fn"])
    print("Precision:", baseline_metrics["precision"])
    print("Recall:", baseline_metrics["recall"])

    output_dir = Path("outputs/adaptive")
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []

    # Seed the planner with the experiment already completed.
    history.append({
        "transform": "brightness",
        "value": 0.6,
        "baseline_detections": 20,
        "perturbed_detections": 18,
        "matched": 18,
        "lost": 2,
        "detection_persistence": 0.90,
        "mean_confidence_delta": -0.018,
        "precision": 0.50,
        "recall": 9 / 37,
    })

    for iteration in range(args.iterations):

        print(f"\n=== ITERATION {iteration + 1} ===")

        proposal = choose_next_experiment(history)

        print("Muse proposal:")
        print(json.dumps(proposal, indent=2))

        transformed = apply_transform(
            image,
            proposal["transform"],
            proposal["value"],
        )

        filename = (
            f"{iteration + 1}_"
            f"{proposal['transform']}_"
            f"{proposal['value']}"
        )

        transformed.save(output_dir / f"{filename}.png")

        predictions = detect(model, transformed)

        metrics = evaluate_image(
            predictions,
            ground_truth,
        )

        stability = compare_detections(
            baseline_predictions,
            predictions,
            iou_threshold=0.5,
        )

        baseline_recall = baseline_metrics["recall"]
        perturbed_recall = metrics["recall"]

        recall_delta = (
            perturbed_recall - baseline_recall
            if baseline_recall is not None
            and perturbed_recall is not None
            else None
        )

        experiment = {
            **proposal,
            "baseline_detections": len(baseline_predictions),
            "perturbed_detections": len(predictions),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "recall_delta": recall_delta,
            "matched": stability["matched_count"],
            "lost": stability["lost_count"],
            "new": stability["new_count"],
            "detection_persistence": stability[
                "detection_persistence"
            ],
            "mean_confidence_delta": stability[
                "mean_confidence_delta"
            ],
        }

        history.append(experiment)

        print("\nEvaluation:")
        print("TP:", metrics["tp"])
        print("FP:", metrics["fp"])
        print("FN:", metrics["fn"])
        print("Precision:", metrics["precision"])
        print("Recall:", metrics["recall"])
        print("Recall delta:", recall_delta)
        print(
            "Detection persistence:",
            stability["detection_persistence"],
        )

        # Persist after every iteration.
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2)
        )

    print("\n=== COMPLETE ===")
    print("Experiments:", len(history))
    print("Results:", output_dir / "history.json")


if __name__ == "__main__":
    main()
