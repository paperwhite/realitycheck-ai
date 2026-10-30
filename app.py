import json
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

from realitycheck.evaluate import (
    compare_detections,
    extract_detections,
)
from realitycheck.muse_selector import choose_next_experiment
from realitycheck.transforms import apply_transform

load_dotenv()

model = YOLO("yolo11n.pt")


def detect(image):
    result = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        device="cpu",
        verbose=False,
    )[0]

    predictions = extract_detections(result)

    # YOLO returns BGR; convert to RGB.
    visualization = Image.fromarray(
        result.plot()[..., ::-1]
    )

    return predictions, visualization


def run_realitycheck(image, transform, value):
    if image is None:
        raise gr.Error("Please upload an image.")

    image = image.convert("RGB")

    baseline, baseline_visual = detect(image)

    # Manual selection or Muse-driven selection.
    if transform == "Muse Spark (Adaptive)":
        proposal = choose_next_experiment([])

        transform_name = proposal["transform"]
        transform_value = proposal["value"]
        hypothesis = proposal["hypothesis"]

    else:
        transform_name = transform
        transform_value = float(value)
        hypothesis = "User-selected robustness test."

    modified = apply_transform(
        image,
        transform_name,
        transform_value,
    )

    perturbed, perturbed_visual = detect(modified)

    metrics = compare_detections(
        baseline,
        perturbed,
        iou_threshold=0.5,
    )

    persistence = metrics["detection_persistence"]
    confidence_delta = metrics["mean_confidence_delta"]

    report = {
        "experiment": {
            "transform": transform_name,
            "value": transform_value,
            "hypothesis": hypothesis,
        },
        "baseline_detections": metrics["baseline_count"],
        "perturbed_detections": metrics["perturbed_count"],
        "matched_detections": metrics["matched_count"],
        "lost_detections": metrics["lost_count"],
        "new_detections": metrics["new_count"],
        "detection_persistence": persistence,
        "mean_confidence_delta": confidence_delta,
        "evaluation_type": "prediction_stability",
        "ground_truth_available": False,
    }

    return (
        baseline_visual,
        perturbed_visual,
        json.dumps(report, indent=2),
    )


def run_adaptive_tests(image, rounds=3):
    if image is None:
        raise gr.Error("Please upload an image.")

    image = image.convert("RGB")

    baseline, baseline_visual = detect(image)

    history = []
    visualizations = []

    for round_number in range(rounds):
        proposal = None

        for attempt in range(3):
            try:
                proposal = choose_next_experiment(history)
                break
            except (ValueError, KeyError) as exc:
                print(
                f"Muse proposal rejected "
                f"(attempt {attempt + 1}/3): {exc}"
                )

        if proposal is None:
            raise gr.Error(
            "Muse could not select a valid new experiment. "
            "Please try again."
            )

        transformed = apply_transform(
            image,
            proposal["transform"],
            proposal["value"],
        )

        predictions, visualization = detect(transformed)

        metrics = compare_detections(
            baseline,
            predictions,
            iou_threshold=0.5,
        )

        experiment = {
            "round": round_number + 1,
            "transform": proposal["transform"],
            "value": proposal["value"],
            "hypothesis": proposal["hypothesis"],
            "baseline_detections": metrics["baseline_count"],
            "perturbed_detections": metrics["perturbed_count"],
            "matched": metrics["matched_count"],
            "lost": metrics["lost_count"],
            "new": metrics["new_count"],
            "detection_persistence": metrics[
                "detection_persistence"
            ],
            "mean_confidence_delta": metrics[
                "mean_confidence_delta"
            ],
        }

        history.append(experiment)

        visualizations.append(
            (
                visualization,
                (
                    f"Round {round_number + 1}: "
                    f"{proposal['transform']} "
                    f"({proposal['value']})"
                ),
            )
        )

    return (
        baseline_visual,
        visualizations,
        history,
    )

with gr.Blocks(title="RealityCheck AI") as demo:

    gr.Markdown(
        """
        # 🔍 RealityCheck AI

        ### Find out what makes an AI vision model fail.

        Upload an image and test how object detection
        changes under controlled visual perturbations.

        **Powered by Meta Muse Spark and YOLO11**

        *SAM-guided testing and ground-truth evaluation
        are available in the backend and will be
        integrated into the interface next.*
        """
    )

    with gr.Row():
        image_input = gr.Image(
            label="Upload an Image",
            type="pil",
        )

        with gr.Column():
            transform_input = gr.Dropdown(
                choices=[
                    "Muse Spark (Adaptive)",
                    "brightness",
                    "gaussian_blur",
                    "contrast",
                ],
                value="Muse Spark (Adaptive)",
                label="Experiment Selection",
            )

            value_input = gr.Dropdown(
                choices=[
                    "0.4", "0.6", "0.8",
                    "1.0", "2.0", "3.0",
                    "0.5", "0.7", "1.3",
                ],
                value="0.6",
                label="Parameter (Manual Mode)",
            )

            run_button = gr.Button(
                "Run RealityCheck",
                variant="primary",
            )

    with gr.Row():
        baseline_output = gr.Image(
            label="Original — YOLO Detections"
        )

        perturbed_output = gr.Image(
            label="Perturbed — YOLO Detections"
        )

    report_output = gr.Code(
        label="Robustness Report",
        language="json",
    )

    run_button.click(
        fn=run_realitycheck,
        inputs=[
            image_input,
            transform_input,
            value_input,
        ],
        outputs=[
            baseline_output,
            perturbed_output,
            report_output,
        ],
    )
    
    gr.Markdown("## 🧠 Adaptive Robustness Testing")

    rounds_input = gr.Slider(
    minimum=1,
    maximum=3,
    value=3,
    step=1,
    label="Number of Adaptive Rounds",
    )

    adaptive_button = gr.Button(
    "Run Adaptive Testing",
    variant="primary",
    )

    adaptive_baseline = gr.Image(
    label="Baseline Detection",
    )

    adaptive_gallery = gr.Gallery(
    label="Adaptive Experiments",
    columns=3,
    height=350,
    ) 

    adaptive_report = gr.JSON(
    label="Experiment History",
    )

    adaptive_button.click(
    fn=run_adaptive_tests,
    inputs=[image_input, rounds_input],
    outputs=[
	adaptive_baseline,
	adaptive_gallery,
	adaptive_report,
    ],
    )


if __name__ == "__main__":
    demo.launch()
